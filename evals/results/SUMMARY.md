# Evaluation Results Summary

# Retrieval Evaluation Results

Evaluated on 15 answerable questions.

| Metric | vector | bm25 | hybrid |
| --- | --- | --- | --- |
| Hit Rate @ 3 | 80.00% | 93.33% | 93.33% |
| Hit Rate @ 6 | 93.33% | 93.33% | 100.00% |
| MRR | 0.7044 | 0.8333 | 0.8356 |


---


## Agent Architecture Ablation Study

Comparison of single-agent, multi-agent without verifier, and full multi-agent pipeline with verifier on the core golden dataset.

| Metric | single | multi_noverify | multi |
| --- | --- | --- | --- |
| Evidence recall | 100.00% | Not run | 100.00% |
| Faithfulness | 85.42% | Not run | 85.16% |
| Correctness (overall) | 96.88% | Not run | 96.88% |
| Correctness (multi-hop only) | 90.00% | Not run | 90.00% |
| Abstention accuracy | 100.00% | Not run | 100.00% |
| False abstention rate | 0.00% | Not run | 0.00% |
| Avg latency (s) | 82.9s | Not run | 77.9s |
| Avg tokens per question | 11918 | Not run | 14853 |

---

## Web Search Routing Evaluation

Evaluation on golden web dataset testing search routing and factual accuracy.

| Metric | Result |
| --- | --- |
| Status | Not run |
