# 2026-10-16 — answer quality, Qwen3.5-2B Q4_0 and bf16 on the GTX 1650 Ti

**Answer: both files score on this card what they scored on the RTX 3050.** bf16 is within one point of the
3050 on every task and Q4_0 within 2.2 points, with no range clear of zero beyond the suite's +0.007. Q4_0 costs Qwen 5.8 to
7.5 points per task against bf16. Gemma 4 E2B Q4 (scored on the 3050) leads both.

## Setup

`run_quality.sh` runs both arms back to back on llama.cpp `7182ef71e` (sm_75), with the 3050 runs' flags:
`-c 32768 --parallel 8 -fa 1`, f16 KV, `-t 6 -tb 12`, `--reasoning off`, no drafter.

- Q4_0: this host's file (`bd99d5ae…`), every layer on the card, server 1806 MiB.
- bf16: this host's file (`ecbfdc76…`) with the last 6 of 24 blocks on the CPU
  (`-ot 'blk\.(18|19|20|21|22|23)\.=CPU'`), the fewest that fits; server 3622 MiB. Its logs are in
  `../../../../local-llamacpp-1650ti-qwen352b/benchmarks/runs/2026-10-16-quality-1650ti/`.
- Tasks: the Nimble 13-subset suite (3,880 records, through `vllm_shim.py`, `--tokenizer Qwen/Qwen3.5-2B`),
  GSM8K (1,319, 2,048-token limit) and BFCL simple (400), each at 8 requests at a time.

`compare.py` pairs each arm with the 3050 runs and Gemma E2B on the same records: full tables in
`COMPARE.md`, numbers in `compare.json`. No errors in any task.

## Results

| task | Qwen Q4_0, 1650 Ti | Qwen Q4_0, 3050 | Qwen bf16, 1650 Ti | Qwen bf16, 3050 | Gemma E2B Q4, 3050 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Suite (3,880) | **0.430** | 0.422 | **0.488** | 0.488 | 0.683 |
| GSM8K (1,319) | **0.734** | 0.743 | **0.807** | 0.798 | 0.898 |
| BFCL simple (400) | **0.688** | 0.710 | **0.762** | 0.765 | 0.925 |

| paired difference (95% range) | suite | GSM8K | BFCL |
| --- | --- | --- | --- |
| Q4_0: 1650 Ti − 3050 | +0.007 (+0.002 to +0.013) | −0.009 (−0.030 to +0.011) | −0.022 (−0.045 to +0.000) |
| bf16: 1650 Ti − 3050 | +0.000 (−0.003 to +0.003) | +0.009 (−0.001 to +0.019) | −0.003 (−0.007 to +0.000) |
| 1650 Ti: Q4_0 − bf16 | −0.058 (−0.073 to −0.043) | −0.073 (−0.096 to −0.049) | −0.075 (−0.113 to −0.037) |
| 1650 Ti: Q4_0 − Gemma E2B | −0.253 (−0.273 to −0.233) | −0.164 (−0.187 to −0.140) | −0.237 (−0.287 to −0.190) |
| 1650 Ti: bf16 − Gemma E2B | −0.195 (−0.213 to −0.177) | −0.091 (−0.113 to −0.069) | −0.163 (−0.210 to −0.117) |

- **bf16 matches the 3050 record for record**: 33 of 3,880 suite records differ, 44 of 1,319 GSM8K problems
  and 1 of 400 BFCL calls. The 6 blocks on the CPU change no score.
- **Q4_0 differs more between hosts**, in line with its logits (`../2026-10-16-parity-crosshost-1650ti`): 127
  suite records, 188 GSM8K problems and 21 BFCL calls differ. The suite's +0.007 is the only range clear of
  zero, and it is under one point.
- **The Q4_0 loss is clear on all three tasks**, and it is 23% of Qwen Q4_0's gap to Gemma E2B on the suite,
  44% on GSM8K and 32% on BFCL (25 / 35 / 26% on the 3050). On the suite it falls mostly on massive-de-DE
  (−0.157), pubmedqa (−0.156) and vitaminc-dev (−0.085); civil_comments (+0.070) and paws (+0.016) move the
  other way.
- **Q4_0 skips the tool more often**: 88 BFCL replies with no tool call, against 48 for bf16 (both hosts), 79
  for Q4_0 on the 3050 and 6 for Gemma E2B.
- **Q4_0 runs long more often**: 107 GSM8K replies hit 2,048 tokens, against 52 for bf16 (52 on the 3050 too)
  and 100 for Q4_0 on the 3050.
- All labels returned on the suite: Q4_0 90.2%, bf16 82.3% (3050: 90.8%, 82.2%).

## Run time on this card

| arm | suite | GSM8K | BFCL |
| --- | ---: | ---: | ---: |
| Q4_0 | 44.6 min | 64.8 min | 9.3 min |
| bf16, 6 blocks on CPU | 55.7 min | 186.5 min | 17.5 min |
| Q4_0, 1650 Ti / 3050 | 6.89x | 1.69x | 2.86x |
| bf16, 1650 Ti / 3050 | 6.41x | 3.85x | 4.43x |

The card ran below its first-light clocks: `driver.log` reads 1035–1890 MHz at the step boundaries, against
1860–1875 MHz held through first light. Clock speed changes run time, not scores. This run does not separate
the card, the clocks and the host CPU in the suite's 6–7x.

## Not measured

Qwen with thinking on, Q8_0, an imatrix Q4 build, Gemma E2B on this card at these settings.
