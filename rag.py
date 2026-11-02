"""Retrieval-Augmented Generation (RAG) over PDFs: chunk -> embed -> index -> retrieve -> generate.

Hugging Face models end to end:
  * embeddings: sentence-transformers/all-MiniLM-L6-v2
  * generator : google/flan-t5-small (or any seq2seq model, e.g. the LoRA-merged one trained in app.py)
  * index     : FAISS inner-product index over L2-normalised vectors (inner product == cosine similarity)

Usage:
    index = RagIndex()
    index.add_pdf("Stuff/Federer.pdf")
    result = answer_question("How many Wimbledon titles?", index, Generator(), k=3)
    print(result.answer, [hit.score for hit in result.hits])
"""
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # faiss + torch both ship an OpenMP runtime on Windows

import faiss  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pypdf import PdfReader  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: E402

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-small"

PROMPT = (
    "Answer the question using only the context. "
    "If the context does not contain the answer, reply \"I don't know\".\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)


# --------------------------------------------------------------------------- ingestion
def extract_text(pdf_path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, max_words: int = 90, overlap: int = 20) -> list[str]:
    """Sentence-aware sliding window: pack whole sentences up to `max_words`, then start the next
    chunk with roughly `overlap` words from the end of the previous one so facts on a boundary survive."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    sentences = [s.replace("\n", " ").strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text)]
    sentences = [s for s in sentences if s]

    chunks: list[str] = []
    current: list[str] = []
    words = 0
    for sentence in sentences:
        n = len(sentence.split())
        if current and words + n > max_words:
            chunks.append(" ".join(current))
            tail: list[str] = []
            tail_words = 0
            for previous in reversed(current):
                tail.insert(0, previous)
                tail_words += len(previous.split())
                if tail_words >= overlap:
                    break
            current, words = tail, tail_words
        current.append(sentence)
        words += n
    if current:
        chunks.append(" ".join(current))
    return chunks


# --------------------------------------------------------------------------- retrieval
@dataclass
class Hit:
    chunk_id: int
    score: float
    text: str
    source: str


class RagIndex:
    """Vector store: embeds chunks with a Hugging Face sentence-transformer and searches them with FAISS."""

    def __init__(self, embed_model: str = EMBED_MODEL, device: str | None = None):
        self.embedder = SentenceTransformer(embed_model, device=device)
        dim_fn = getattr(self.embedder, "get_embedding_dimension", None) or self.embedder.get_sentence_embedding_dimension
        self.index = faiss.IndexFlatIP(dim_fn())  # inner product on normalised vectors == cosine similarity
        self.chunks: list[str] = []
        self.sources: list[str] = []

    def __len__(self) -> int:
        return len(self.chunks)

    def has_source(self, source: str) -> bool:
        return source in self.sources

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.embedder.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype="float32")

    def add_chunks(self, chunks: list[str], source: str = "") -> int:
        if not chunks:
            return 0
        self.index.add(self._embed(chunks))
        self.chunks += chunks
        self.sources += [source] * len(chunks)
        return len(chunks)

    def add_pdf(self, pdf_path, **chunk_kwargs) -> int:
        """Index a PDF (re-adding the same file is a no-op). Several PDFs can share one index."""
        source = Path(pdf_path).name
        if self.has_source(source):
            return 0
        return self.add_chunks(chunk_text(extract_text(pdf_path), **chunk_kwargs), source=source)

    def search(self, query: str, k: int = 3) -> list[Hit]:
        if not self.chunks:
            return []
        scores, ids = self.index.search(self._embed([query]), min(k, len(self.chunks)))
        return [Hit(int(i), float(s), self.chunks[i], self.sources[i]) for s, i in zip(scores[0], ids[0]) if i >= 0]


# --------------------------------------------------------------------------- generation
class Generator:
    """Greedy-decoding wrapper around a seq2seq LM, so answers are reproducible."""

    def __init__(self, model=None, tokenizer=None, model_name: str = GEN_MODEL):
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.model = model or AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 32) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()


@dataclass
class RagAnswer:
    answer: str
    hits: list[Hit]
    prompt: str


def build_prompt(question: str, hits: list[Hit], tokenizer=None, max_tokens: int = 480) -> str:
    """Context blocks (best first) + the question last. If the prompt would not fit the model's window,
    drop the lowest-ranked chunks (never the question). Chunks are not numbered on purpose: a small
    model such as FLAN-T5-small tends to copy a marker like "[1]" as its answer."""
    hits = list(hits)
    while True:
        context = "\n\n".join(hit.text for hit in hits)
        prompt = PROMPT.format(context=context, question=question)
        too_long = tokenizer is not None and len(tokenizer(prompt).input_ids) > max_tokens
        if not too_long or len(hits) <= 1:
            return prompt
        hits.pop()


def answer_question(question: str, index: RagIndex, generator: Generator, k: int = 3, max_new_tokens: int = 32) -> RagAnswer:
    hits = index.search(question, k=k)
    prompt = build_prompt(question, hits, tokenizer=generator.tokenizer)
    return RagAnswer(generator.generate(prompt, max_new_tokens=max_new_tokens), hits, prompt)
