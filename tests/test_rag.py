"""Unit tests for the RAG module and the benchmark metrics (run from the repo root: pytest)."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import benchmark  # noqa: E402
import rag  # noqa: E402


# --- chunking ---------------------------------------------------------------------------------
def test_chunks_respect_the_word_budget_and_overlap():
    sentences = [f"Sentence number {i} says something useful about topic {i}." for i in range(40)]
    chunks = rag.chunk_text(" ".join(sentences), max_words=30, overlap=10)
    assert len(chunks) > 3
    assert all(len(c.split()) <= 30 + 10 for c in chunks)  # budget plus at most one overlap tail
    shared = set(chunks[0].split()) & set(chunks[1].split())
    assert len(shared) >= 5  # consecutive chunks share context


def test_chunking_handles_empty_and_short_text():
    assert rag.chunk_text("") == []
    assert rag.chunk_text("Just one sentence.") == ["Just one sentence."]


# --- prompt building -----------------------------------------------------------------------------
def test_prompt_drops_low_ranked_chunks_but_keeps_the_question():
    hits = [rag.Hit(i, 1.0 - i / 10, f"chunk{i} " + "word " * 100, "doc.pdf") for i in range(5)]
    tokenizer = lambda text: SimpleNamespace(input_ids=text.split())  # one "token" per word  # noqa: E731
    prompt = rag.build_prompt("Who wrote it?", hits, tokenizer=tokenizer, max_tokens=250)
    assert "Question: Who wrote it?" in prompt
    assert "chunk0" in prompt and "chunk4" not in prompt  # best chunks kept, lowest-ranked dropped
    assert "[1]" not in prompt  # no citation markers for the small model to copy


# --- metrics -----------------------------------------------------------------------------------------
def test_metrics():
    assert benchmark.normalize("The  Roger Federer Foundation!") == "roger federer foundation"
    assert benchmark.exact_match("Five times.", "five times") == 1.0
    assert benchmark.f1("5 joint-record titles", "5 titles") > 0.5
    assert benchmark.f1("", "") == 1.0 and benchmark.f1("abc", "xyz") == 0.0
    assert benchmark.rouge_l("x y z w", "x z w") > 0.8  # LCS = 3 of 4 predicted / 3 gold
    assert benchmark.contains("1968", "The Open Era began in 1968 with ...")
    assert not benchmark.contains("", "anything")


def test_split_is_deterministic_and_disjoint():
    pairs = [{"question": f"q{i}", "answer": str(i)} for i in range(20)]
    train_a, test_a = benchmark.split_pairs(pairs)
    train_b, test_b = benchmark.split_pairs(pairs)
    assert test_a == test_b and len(test_a) == 6 and len(train_a) == 14
    assert not {p["question"] for p in train_a} & {p["question"] for p in test_a}


# --- retrieval (uses the real Hugging Face embedding model; downloads ~90 MB on first run) -----------------
def test_retrieval_finds_the_relevant_chunk():
    index = rag.RagIndex()
    index.add_chunks([
        "Roger Federer won eight Wimbledon titles and twenty Grand Slams.",
        "Bernard Bolzano wrote about the paradoxes of the infinite in mathematics.",
        "Fundamental theorem of space curves relates curvature and torsion.",
    ], source="demo")
    top = index.search("How many Wimbledon titles did Federer win?", k=2)
    assert top[0].chunk_id == 0 and top[0].score > top[1].score
    assert index.search("paradoxes of infinity", k=1)[0].chunk_id == 1
    assert index.has_source("demo") and len(index) == 3
