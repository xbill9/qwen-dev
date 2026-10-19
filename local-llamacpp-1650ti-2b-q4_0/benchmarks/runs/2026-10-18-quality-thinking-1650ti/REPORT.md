# 2026-10-18 — thinking on, Gemma 4 E2B QAT Q4_0, GTX 1650 Ti

Gemma's half of a paired run with Qwen3.5-2B. The report, run script and comparison are in
`local-llamacpp-1650ti-qwen352b-q4_0/benchmarks/runs/2026-10-18-quality-thinking-1650ti/`.

With thinking on at Gemma's own sampling (`sampling.json`), 32,000-token limit: GSM8K 0.930 on a fixed
random 300 (thinking off 0.927 on the same 300, +0.003, −0.023 to +0.030) and BFCL 0.887 (thinking off
0.917, −0.030, −0.058 to −0.005). Median reply 808 tokens on GSM8K, none cut off.
