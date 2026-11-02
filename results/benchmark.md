### Philosophy of math (Bolzano) (held-out questions: n=56)

| System | Exact match | F1 | ROUGE-L | Latency / question |
|---|---|---|---|---|
| Base FLAN-T5-small, no context | 0.089 | 0.124 | 0.124 | 200 ms |
| RAG (top-3) + base FLAN-T5-small | 0.071 | 0.169 | 0.169 | 308 ms |
| LoRA fine-tuned, no context | 0.107 | 0.128 | 0.128 | 92 ms |
| RAG (top-3) + LoRA model | 0.143 | 0.207 | 0.207 | 240 ms |

Retrieval, answer found in the top-k chunks: k=1 **0.23**, k=3 **0.34**, k=5 **0.34** (152 chunks indexed).

LoRA: r=8, alpha=32, lr=0.001, 8 epochs on 130 pairs; 344,064 of 77,305,216 weights trainable (0.45%); trained in 66 s on this machine.

### Sports (Federer) (held-out questions: n=16)

| System | Exact match | F1 | ROUGE-L | Latency / question |
|---|---|---|---|---|
| Base FLAN-T5-small, no context | 0.000 | 0.000 | 0.000 | 133 ms |
| RAG (top-3) + base FLAN-T5-small | 0.250 | 0.648 | 0.627 | 237 ms |

Retrieval, answer found in the top-k chunks: k=1 **0.44**, k=3 **0.50**, k=5 **0.50** (11 chunks indexed).
