# 2026-10-19 — thinking off at vendor sampling, Gemma 4 E2B QAT Q4_0, GTX 1650 Ti

Gemma's half of a paired control run with Qwen3.5-2B. The report, run script and comparison are in
`local-llamacpp-1650ti-qwen352b-q4_0/benchmarks/runs/2026-10-19-quality-sampled-nothink-1650ti/`.

Thinking off at Gemma's own sampling (`sampling.json`), 32,000-token limit: GSM8K 0.910 on the fixed 300
(greedy 0.927, thinking on 0.930) and BFCL 0.885 (greedy 0.917, thinking on 0.887). Sampling costs Gemma 3.3
points on BFCL (−0.055 to −0.013); thinking changes neither task beyond its range.
