# 2026-10-16 — answer quality, Qwen3.5-2B bf16 on the GTX 1650 Ti

**Answer: with 6 of 24 blocks on the CPU, the bf16 file scores on this card what it scored on the RTX 3050.**
Suite 0.488, GSM8K 0.807, BFCL 0.762, against 0.488 / 0.798 / 0.765 there.

## Setup

The bf16 arm of `run_quality.sh` (copied here; it runs both arms): this host's bf16 GGUF (`ecbfdc76…`,
`model-sha256.txt`) on llama.cpp `7182ef71e` (sm_75), `-ngl 99 -ot 'blk\.(18|19|20|21|22|23)\.=CPU'
-c 32768 --parallel 8 -fa 1`, f16 KV, `-t 6 -tb 12`, `--reasoning off`. Server 3622 MiB. Same tasks and
harness as the 3050 run (`../../../../local-llamacpp-3050oem-qwen352b/benchmarks/runs/2026-10-16-quality-bf16-3050oem/`).
The comparison, with the Q4_0 arm, is in
`../../../../local-llamacpp-1650ti-qwen352b-q4_0/benchmarks/runs/2026-10-16-quality-1650ti/` (`REPORT.md`,
`COMPARE.md`, `compare.json`). No errors in any task.

## Results

| task | bf16, 1650 Ti | bf16, 3050 | 1650 Ti − 3050 (95% range) | records that differ |
| --- | ---: | ---: | --- | ---: |
| Suite (3,880) | 0.488 | 0.488 | +0.000 (−0.003 to +0.003) | 33 |
| GSM8K (1,319) | 0.807 | 0.798 | +0.009 (−0.001 to +0.019) | 44 |
| BFCL simple (400) | 0.762 | 0.765 | −0.003 (−0.007 to +0.000) | 1 |

- Q4_0 on this card scores 0.430 / 0.734 / 0.688, 5.8 to 7.5 points below this file on every task.
- 48 BFCL replies with no tool call and 52 GSM8K replies at the 2,048-token limit, the same counts as on the
  3050.

## Run time

Suite 55.7 min, GSM8K 186.5 min, BFCL 17.5 min: 6.41x, 3.85x and 4.43x the 3050's, and 1.25x, 2.88x and
1.87x the Q4_0 file's on this card. `driver.log` reads 1035–1860 MHz at the step boundaries.
