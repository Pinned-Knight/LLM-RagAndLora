"""Benchmark: closed-book FLAN-T5 vs RAG vs LoRA fine-tuning on held-out questions.

    python benchmark.py                 # everything (trains a small LoRA on CPU, a few minutes)
    python benchmark.py --skip-lora     # base vs RAG only (about a minute)

Protocol
  * Domains: the sample PDFs in Stuff/ plus the Q&A pairs generated from them by gen_data.py.
  * Bolzano (186 pairs): shuffled with a fixed seed, 70% train / 30% held-out test. The LoRA model sees only the
    train pairs. Every system is scored on the same held-out questions.
  * Federer (16 pairs): too small to train and test on, so only the systems that need no training are scored (all 16).
  * Metrics: exact match, token F1, ROUGE-L, mean latency per question, and "answer in retrieved context@k"
    (does the gold answer string appear in the top-k chunks?) as a retrieval-quality proxy.

Writes results/benchmark.json, results/predictions.json and results/benchmark.md.
"""
import argparse
import json
import random
import re
import string
import tempfile
import time
from collections import Counter
from pathlib import Path

import rag  # imported first: sets the OpenMP workaround before torch/faiss load

DOMAINS = [
    {"name": "Philosophy of math (Bolzano)", "pdf": "Stuff/Split - test.pdf", "qa": "Stuff/Split - test_dataset.json", "lora": True},
    {"name": "Sports (Federer)", "pdf": "Stuff/Federer.pdf", "qa": "Stuff/Federer_dataset.json", "lora": False},
]


# --------------------------------------------------------------------------- metrics
def normalize(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    overlap = sum((Counter(p) & Counter(g)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def rouge_l(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    table = [[0] * (len(g) + 1) for _ in range(len(p) + 1)]
    for i, a in enumerate(p, 1):
        for j, b in enumerate(g, 1):
            table[i][j] = table[i - 1][j - 1] + 1 if a == b else max(table[i - 1][j], table[i][j - 1])
    lcs = table[-1][-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(p), lcs / len(g)
    return 2 * precision * recall / (precision + recall)


def contains(gold: str, text: str) -> bool:
    gold = normalize(gold)
    return bool(gold) and gold in normalize(text)


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


# --------------------------------------------------------------------------- LoRA training
def train_lora(train_pairs, epochs=8, rank=8, alpha=32, lr=1e-3, batch_size=8, seed=42):
    """LoRA on the attention q/v projections of FLAN-T5-small; returns the merged model + training info."""
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer,
                              Seq2SeqTrainingArguments, set_seed)

    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(rag.GEN_MODEL)
    base = AutoModelForSeq2SeqLM.from_pretrained(rag.GEN_MODEL)
    model = get_peft_model(base, LoraConfig(r=rank, lora_alpha=alpha, target_modules=["q", "v"], lora_dropout=0.1,
                                            bias="none", task_type=TaskType.SEQ_2_SEQ_LM))
    trainable, total = model.get_nb_trainable_parameters()

    def encode(example):
        encoded = tokenizer(example["question"], truncation=True, max_length=64)
        encoded["labels"] = tokenizer(text_target=example["answer"], truncation=True, max_length=32)["input_ids"]
        return encoded

    dataset = Dataset.from_list(train_pairs).map(encode, remove_columns=["question", "answer"])
    args = Seq2SeqTrainingArguments(
        output_dir=tempfile.mkdtemp(prefix="lora_"), per_device_train_batch_size=batch_size, num_train_epochs=epochs,
        learning_rate=lr, logging_steps=10, save_strategy="no", report_to=[], seed=seed, label_names=["labels"])
    # labels are padded with -100 by the collator, so padding is ignored by the loss
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=dataset,
                             data_collator=DataCollatorForSeq2Seq(tokenizer, model=base, label_pad_token_id=-100))
    started = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - started
    losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    info = {"epochs": epochs, "rank": rank, "alpha": alpha, "lr": lr, "batch_size": batch_size,
            "train_pairs": len(train_pairs), "trainable_params": trainable, "total_params": total,
            "trainable_pct": round(100 * trainable / total, 2), "train_seconds": round(seconds, 1),
            "first_logged_loss": round(losses[0], 3) if losses else None,
            "last_logged_loss": round(losses[-1], 3) if losses else None}
    return model.merge_and_unload(), tokenizer, info


# --------------------------------------------------------------------------- evaluation
def run_system(label, answer_fn, pairs):
    rows, latencies = [], []
    for pair in pairs:
        started = time.perf_counter()
        prediction = answer_fn(pair["question"])
        latencies.append((time.perf_counter() - started) * 1000)
        rows.append({"question": pair["question"], "gold": pair["answer"], "prediction": prediction})
    summary = {
        "system": label, "n": len(rows),
        "exact_match": round(mean(exact_match(r["prediction"], r["gold"]) for r in rows), 4),
        "f1": round(mean(f1(r["prediction"], r["gold"]) for r in rows), 4),
        "rouge_l": round(mean(rouge_l(r["prediction"], r["gold"]) for r in rows), 4),
        "latency_ms": round(mean(latencies), 1),
    }
    print(f"  {label:<34} EM {summary['exact_match']:.3f}  F1 {summary['f1']:.3f}  ROUGE-L {summary['rouge_l']:.3f}  {summary['latency_ms']:.0f} ms")
    return summary, rows


def split_pairs(pairs, test_fraction=0.3, seed=42):
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    n_test = max(1, round(len(pairs) * test_fraction))
    return pairs[n_test:], pairs[:n_test]


def to_markdown(results) -> str:
    lines = []
    for domain in results["domains"]:
        lines += [f"### {domain['name']} (held-out questions: n={domain['n_test']})", "",
                  "| System | Exact match | F1 | ROUGE-L | Latency / question |", "|---|---|---|---|---|"]
        for s in domain["systems"]:
            lines.append(f"| {s['system']} | {s['exact_match']:.3f} | {s['f1']:.3f} | {s['rouge_l']:.3f} | {s['latency_ms']:.0f} ms |")
        r = domain["retrieval"]
        lines += ["", f"Retrieval, answer found in the top-k chunks: k=1 **{r['answer_in_context@1']:.2f}**, "
                      f"k=3 **{r['answer_in_context@3']:.2f}**, k=5 **{r['answer_in_context@5']:.2f}** "
                      f"({domain['n_chunks']} chunks indexed).", ""]
        if domain.get("lora"):
            l = domain["lora"]
            lines += [f"LoRA: r={l['rank']}, alpha={l['alpha']}, lr={l['lr']}, {l['epochs']} epochs on {l['train_pairs']} pairs; "
                      f"{l['trainable_params']:,} of {l['total_params']:,} weights trainable ({l['trainable_pct']}%); "
                      f"trained in {l['train_seconds']:.0f} s on this machine.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-lora", action="store_true", help="do not train/evaluate the LoRA systems")
    parser.add_argument("--k", type=int, default=3, help="chunks retrieved per question for RAG")
    parser.add_argument("--epochs", type=int, default=8, help="LoRA epochs")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    base = rag.Generator()
    results = {"config": {"embedder": rag.EMBED_MODEL, "generator": rag.GEN_MODEL, "k": args.k, "chunk_words": 90,
                          "chunk_overlap_words": 20, "split_seed": 42}, "domains": []}
    all_predictions = {}

    for domain in DOMAINS:
        print(f"\n== {domain['name']} ==")
        pairs = json.loads(Path(domain["qa"]).read_text(encoding="utf-8"))
        use_lora = domain["lora"] and not args.skip_lora
        train, test = split_pairs(pairs) if domain["lora"] else ([], pairs)
        entry = {"name": domain["name"], "n_pairs": len(pairs), "n_test": len(test)}

        index = rag.RagIndex()
        index.add_pdf(domain["pdf"])
        entry["n_chunks"] = len(index)
        entry["retrieval"] = {f"answer_in_context@{k}": round(mean(
            contains(p["answer"], " ".join(h.text for h in index.search(p["question"], k=k))) for p in test), 4)
            for k in (1, 3, 5)}

        systems, predictions = [], {}
        for label, fn in [
            ("Base FLAN-T5-small, no context", lambda q: base.generate(q)),
            (f"RAG (top-{args.k}) + base FLAN-T5-small", lambda q: rag.answer_question(q, index, base, k=args.k).answer),
        ]:
            summary, rows = run_system(label, fn, test)
            systems.append(summary)
            predictions[label] = rows

        if use_lora:
            print(f"  training LoRA on {len(train)} pairs ...")
            model, tokenizer, info = train_lora(train, epochs=args.epochs)
            tuned = rag.Generator(model=model, tokenizer=tokenizer)
            entry["lora"] = info
            for label, fn in [
                ("LoRA fine-tuned, no context", lambda q: tuned.generate(q)),
                (f"RAG (top-{args.k}) + LoRA model", lambda q: rag.answer_question(q, index, tuned, k=args.k).answer),
            ]:
                summary, rows = run_system(label, fn, test)
                systems.append(summary)
                predictions[label] = rows

        entry["systems"] = systems
        results["domains"].append(entry)
        all_predictions[domain["name"]] = predictions

    (out / "benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out / "predictions.json").write_text(json.dumps(all_predictions, indent=2), encoding="utf-8")
    (out / "benchmark.md").write_text(to_markdown(results), encoding="utf-8")
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
