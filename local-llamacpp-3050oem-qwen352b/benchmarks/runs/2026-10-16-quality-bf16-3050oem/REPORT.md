# 2026-10-16 — answer quality, Qwen3.5-2B bf16 against its Q4_0 and Gemma 4 E2B / E4B, RTX 3050 OEM

**Answer: the 4-bit rounding costs Qwen 5.5 to 6.5 points per task, and Gemma E2B still leads the unquantized
Qwen on every task.** Q4_0 accounts for 25% of Qwen's gap to Gemma E2B on the suite, 35% on GSM8K and 26% on
BFCL; the rest is the model.

## Setup

Qwen's bf16 GGUF (`1526c207…`, converted from `Qwen/Qwen3.5-2B` revision `15852e8c` by llama.cpp `7182ef71e`),
in `run_quality.sh`: the Q4_0 run of `local-llamacpp-3050oem-qwen352b-q4_0/benchmarks/runs/2026-10-16-quality-vs-gemma-3050oem`
with only the file changed, same binary, same day. Server 4,284 MiB. Gemma records are the 2026-10-13 run's.
`compare.py` pairs every arm on the same records; full tables in `COMPARE.md`, numbers in `compare.json`.
No errors in any task.

## Results

| task | Qwen Q4_0 | **Qwen bf16** | Gemma E2B Q4 | Gemma E4B Q4 |
| --- | ---: | ---: | ---: | ---: |
| Suite (3,880) | 0.422 | **0.488** | 0.683 | 0.730 |
| GSM8K (1,319) | 0.743 | **0.798** | 0.898 | 0.933 |
| BFCL simple (400) | 0.710 | **0.765** | 0.925 | 0.910 |

| paired difference | suite | GSM8K | BFCL |
| --- | --- | --- | --- |
| Qwen Q4_0 − Qwen bf16 | −0.065 (−0.081 to −0.049) | −0.055 (−0.077 to −0.032) | −0.055 (−0.092 to −0.020) |
| Qwen bf16 − Gemma E2B Q4 | −0.195 (−0.213 to −0.177) | −0.100 (−0.122 to −0.078) | −0.160 (−0.207 to −0.115) |
| Qwen bf16 − Gemma E4B Q4 | −0.243 (−0.260 to −0.226) | −0.135 (−0.156 to −0.114) | −0.145 (−0.195 to −0.098) |

- **The Q4_0 loss is real on all three tasks**, every range clear of zero. On the suite it falls mostly on
  pubmedqa (−0.180), massive-de-DE (−0.163) and vitaminc-dev (−0.104).
- **The habits that sink Qwen on these prompts are there at bf16.** It still calls most comments toxic
  (civil_comments 0.227) and avoids the top consistency level (summeval-consistency 0.139); all labels
  returned 82.2%, against Gemma E2B's 96.2%.
- **bf16 calls the tool more often:** 48 BFCL replies with no tool call, against 79 at Q4_0 and 6 for Gemma
  E2B.
- **bf16 runs long less often:** 52 GSM8K replies hit 2,048 tokens, against 100 at Q4_0 and 5 for Gemma E2B.

## Run time on this card

Suite 8.7 min, GSM8K 48.5 min, BFCL 3.9 min (Q4_0: 6.5, 38.2, 3.3).

## Not measured

Qwen with thinking on, Q8_0, an imatrix Q4 build, and prompt variants.
