# 2026-10-16 — speculative decoding with Qwen3.5-2B's own MTP head, RTX 3050 OEM

**Question:** for a single user, does drafting from the multi-token-prediction layer inside Qwen3.5-2B raise
generation speed?

**Answer: yes, 1.15x greedy and 1.09x sampled with 1 drafted token per step**, and greedy replies stay
byte-identical to serving without it up to 3 drafted tokens. Gemma E2B's separate drafter gave 1.39x on this
card (`local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-13-spec-mtp-3050oem`, llama.cpp `77c4540ce`).

## The drafter

Nothing extra to download. The Q4_0 GGUF this rig serves carries Qwen's MTP layer (`blk.24.nextn.*`, Q4_0
like the other matrices), and llama.cpp `7182ef71e` drafts from it with `--spec-type draft-mtp` and no `-md`:
the server logs `creating MTP draft context against the target model`, and the 15 "unused tensor" warnings of
the no-draft arm disappear.

## Method

`spec_ab.py`, the Gemma drafter run's method: one llama-server per arm on port 8099, `-c 8192`, serving flags
otherwise, GPU cooled to ≤40 °C before each arm. Six prompts (code, chat, story, explain, facts, summary),
greedy, max 256 tokens, prompt cache off, 3 repeats; then one sampled pass per prompt (temperature 0.8, seed
1234). Arms: no draft, and draft-mtp at `--spec-draft-n-max` 1, 2, 3, 4, 6. Raw records with reply texts in
`results.jsonl`; tables computed by `summarize.py` into `summary.txt`. Largest within-cell spread across the 3
greedy repeats: 3.6%.

## Generation speed, greedy (median of 3; x vs no draft; fraction of drafted tokens accepted)

| prompt | no draft | 1 token | 2 tokens | 3 tokens | 4 tokens | 6 tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| code | 124.3 | 155.3 (1.25x) 0.92 | **169.9 (1.37x)** 0.86 | 163.6 (1.32x) 0.77 | 148.1 (1.19x) 0.62 | 119.3 (0.96x) 0.47 |
| chat | 124.4 | **144.6 (1.16x)** 0.77 | 143.1 (1.15x) 0.62 | 121.1 (0.97x) 0.46 | 99.2 (0.80x) 0.33 | 76.4 (0.61x) 0.24 |
| story | 124.2 | **135.5 (1.09x)** 0.65 | 126.2 (1.02x) 0.49 | 104.4 (0.84x) 0.35 | 86.0 (0.69x) 0.25 | 64.3 (0.52x) 0.17 |
| explain | 124.1 | **139.1 (1.12x)** 0.70 | 136.5 (1.10x) 0.58 | 118.0 (0.95x) 0.44 | 102.0 (0.82x) 0.34 | 76.5 (0.62x) 0.24 |
| facts | 124.1 | 142.6 (1.15x) 0.75 | **146.3 (1.18x)** 0.67 | 126.2 (1.02x) 0.51 | 115.2 (0.93x) 0.42 | 85.5 (0.69x) 0.29 |
| summary | 123.9 | **139.7 (1.13x)** 0.72 | 135.0 (1.09x) 0.55 | 117.9 (0.95x) 0.45 | 94.6 (0.76x) 0.29 | 70.0 (0.56x) 0.20 |
| **geometric mean** | 1.00 | **1.149** | 1.145 | 0.998 | 0.852 | 0.647 |

In tok/s the geometric means are 124.2 with no draft, 142.7 at 1 token and 142.2 at 2.

Sampled at temperature 0.8: geometric mean 1.086 / 1.047 / 0.920 / 0.782 / 0.596 for 1 / 2 / 3 / 4 / 6 tokens.
At 1 token it ranges from 0.95x (story) to 1.23x (code).

- **1 and 2 tokens tie on greedy; 1 wins on sampled text** (1.086 against 1.047) and is the setting to serve.
  2 tokens pays only on code and facts.
- **Acceptance falls fast with draft length.** One drafted token is accepted 65–92% of the time; by the third
  it is 35–77%, and from 3 tokens on drafting loses on everything but code.
- Gemma E2B's drafter on this card accepted less on its first tokens yet paid more (1.39x at 2 tokens). Its
  drafter is a separate 4-layer model; Qwen's is one extra layer run per drafted token. Why the ratios differ
  is not tested here.

## Greedy replies

**Identical to the no-draft arm on all 6 prompts at 1, 2 and 3 drafted tokens**, and different on all 6 at 4
and 6. Every arm repeats itself exactly. Gemma E2B's drafter changed all 18 greedy replies at 2 tokens.

## VRAM, at `-c 8192`

1435 MiB with no draft, 1559 at 1 token, 1577 at 2, 1655 at 6 (`nvidia-smi` after warm-up).

## Not measured

Answer quality with drafting on (greedy text is unchanged at 1–3 tokens, so greedy scores would be too),
several requests at once, long context, the rig's `-c 131072`, and the bf16 file.
