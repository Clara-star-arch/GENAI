# Evaluation summary

## Retrieval (page-level)

| Configuration | Hit@1 | Hit@3 | Hit@5 | MRR@10 |
|---|---|---|---|---|
| Dense only | 86% | 97% | 100% | 0.917 |
| BM25 only | 86% | 97% | 97% | 0.917 |
| Hybrid (RRF) | 89% | 97% | 100% | 0.930 |
| Hybrid + rerank | 89% | 94% | 100% | 0.929 |
| Hybrid + rerank + boost | 91% | 94% | 100% | 0.943 |

## Answer quality

| Question type | n | Key facts | Correct citation | Faithfulness | Relevance (1-5) | Correctness (1-5) | Declined correctly |
|---|---|---|---|---|---|---|---|
| Class questions | 12 | 96% | 100% | 100% | 5.00 | 4.83 | - |
| Other facts | 13 | 100% | 100% | 100% | 5.00 | 5.00 | - |
| Paraphrases | 6 | 100% | 100% | 100% | 5.00 | 5.00 | - |
| Follow-ups | 4 | 100% | 100% | 100% | 5.00 | 5.00 | - |
| Off-topic | 5 | - | - | - | - | - | 100% |
| Not on site | 2 | - | - | - | - | - | 100% |
| Overall | 42 | 98% | 100% | 100% | 5.00 | 4.94 | 100% |

- Latency: median 8.43s, p95 10.99s
- Answers rewritten by the hallucination check: 0; safe fallbacks: 0
