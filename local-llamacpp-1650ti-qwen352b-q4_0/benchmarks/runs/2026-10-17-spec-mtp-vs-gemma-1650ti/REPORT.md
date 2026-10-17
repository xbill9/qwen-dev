# 2026-10-17 — speculative decoding on the GTX 1650 Ti, Qwen3.5-2B's MTP head against Gemma 4 E2B's drafter

**Question:** for one user on this card, does drafting change which 2B model generates faster?

**Answer: it erases Qwen's lead.** Without drafting Qwen Q4_0 decodes 1.05x as fast as Gemma E2B (82.2
against 77.9 tok/s, greedy). Each model's best drafter setting is 1 token, and there they tie: 91.0 against
92.1 tok/s greedy (0.99x), 86.1 against 90.7 sampled (0.95x). Gemma's drafter gains more, 1.18x greedy
against Qwen's 1.11x.

## Method

`spec_ab.py`, the 3050 drafter runs' method (`../../../../local-llamacpp-3050oem-qwen352b-q4_0/benchmarks/runs/2026-10-16-spec-mtp-3050oem`),
both models on one binary (llama.cpp `7182ef71e`, sm_75): one llama-server per arm on port 8099, `-c 8192
--parallel 1`, `-fa 1`, f16 KV, `--reasoning off`, GPU at ≤50 °C before each arm. Six prompts (code, chat,
story, explain, facts, summary), max 256 tokens, prompt cache off; greedy ×3, then one sampled pass
(temperature 0.8, seed 1234). Files: Qwen's Q4_0 (`bd99d5ae…`), drafting from the MTP layer inside it
(`--spec-type draft-mtp`, no `-md`); Gemma's exact QAT Q4_0 (`419db9a6…`) with Google's QAT assistant drafter
stored exact Q4_0 (`060f5a01…`, `-md … -ngld all`), as `local-llamacpp-1650ti-2b-q4_0` serves it. Arms: no
draft, 1 and 2 drafted tokens. `summarize.py` computes every table (`summary.txt`, `summary.json`); raw records
with reply texts in `results.jsonl`. Largest spread across greedy repeats: 2.4% (Qwen), 0.9% (Gemma).

## Results, geometric mean of six prompts

| | no draft | 1 token | 2 tokens |
| --- | ---: | ---: | ---: |
| Qwen Q4_0, greedy tok/s | 82.2 | **91.0 (1.107x)** | 84.3 (1.026x) |
| Gemma E2B, greedy tok/s | 77.9 | **92.1 (1.181x)** | 83.9 (1.076x) |
| Qwen / Gemma, greedy | 1.054 | 0.988 | 1.005 |
| Qwen Q4_0, sampled tok/s | 81.8 | **86.1 (1.052x)** | 75.2 (0.918x) |
| Gemma E2B, sampled tok/s | 77.7 | **90.7 (1.167x)** | 82.4 (1.060x) |
| Qwen / Gemma, sampled | 1.053 | 0.950 | 0.912 |

- **1 drafted token is best for both models**, greedy and sampled. At 2, Qwen loses on sampled text (0.92x)
  and Gemma keeps a small gain (1.06x).
- **Code drafts best for both** (Qwen 1.22x, Gemma 1.36x at 1 token); story and summary least (1.05–1.13x).
- **Acceptance of the first drafted token**, greedy: Qwen 0.62–0.90 per prompt, Gemma 0.46–0.85. Qwen
  accepts more yet gains less.
- **Both gain less than on the 3050**: Qwen 1.149x greedy there at 1 token on this binary, Gemma 1.38x at 1
  token on `77c4540ce` (`local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-drafter-gaps-3050oem`).
  Gemma's 1.181x here is in line with this card's earlier runs: 1.164x on `77c4540ce`, 1.180x on `f95b0d9`
  (`local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-14-drafter-accuracy-1650ti`).
- **VRAM:** Qwen 1391 → 1515 MiB with its MTP head at 1 token, Gemma 1491 → 1571 MiB with its drafter.

## Greedy replies

Every arm repeats itself exactly. **Qwen at 1 drafted token matches its no-draft reply on all 6 prompts**, and
differs on all 6 at 2 (on the 3050 it matched up to 3). **Gemma's drafter changes all 6 replies at 1 and at 2
tokens** on this binary. Whether drafting changes answer quality is measured separately
(`../2026-10-17-quality-mtp-1650ti` for Qwen; for Gemma on this card, no change on GSM8K and BFCL in
`../../../../local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-14-drafter-accuracy-1650ti`).

## Not measured

More than one request at a time, long prompts, thinking on, and Qwen's bf16 file with drafting.
