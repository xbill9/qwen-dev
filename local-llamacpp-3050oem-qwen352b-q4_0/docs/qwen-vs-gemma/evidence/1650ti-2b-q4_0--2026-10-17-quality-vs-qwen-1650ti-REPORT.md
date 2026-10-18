# 2026-10-17 — answer quality, Gemma 4 E2B against Qwen3.5-2B on one GTX 1650 Ti

**Question:** Qwen3.5-2B was scored on this card (`../../../../local-llamacpp-1650ti-qwen352b-q4_0/benchmarks/runs/2026-10-16-quality-1650ti`)
but Gemma E2B only on the RTX 3050. Does the Qwen–Gemma gap hold with both models on the same card, binary
and flags?

**Answer: yes.** Gemma E2B scores on this card what it scored on the 3050 (suite 0.681, GSM8K 0.901, BFCL
0.917), and its lead over each Qwen file moves by at most 1.5 points between the cards, with every gap's range
far from zero.

## Setup

Gemma's exact QAT Q4_0 GGUF (`419db9a6…`, the file the 3050 run scored), in `run_quality.sh`: llama.cpp
`7182ef71e` (sm_75), the binary of the Qwen runs on this card, and their flags: `-c 32768 --parallel 8 -fa 1`,
f16 KV, `-t 6 -tb 12`, `--reasoning off`, no drafter. `per_layer_token_embd` stays in system memory, as this
rig serves it. Server 1710 MiB. Same three tasks, all at 8 requests at a time: the Nimble 13-subset suite
(3,880 records), GSM8K (1,319, 2,048-token limit) and BFCL simple (400). The 3050 Gemma run differs in binary
(`77c4540ce`) and in holding the per-layer table on the card. `compare.py` pairs every arm on the same records;
full tables in `COMPARE.md`, numbers in `compare.json`. No errors in any task.

## Results

| task | Gemma E2B QAT Q4_0 | Qwen bf16 | Qwen Q4_0 |
| --- | ---: | ---: | ---: |
| Suite (3,880) | **0.681** | 0.488 | 0.430 |
| GSM8K (1,319) | **0.901** | 0.807 | 0.734 |
| BFCL simple (400) | **0.917** | 0.762 | 0.688 |

All on this card. Paired differences, 95% ranges:

| | suite | GSM8K | BFCL |
| --- | --- | --- | --- |
| Gemma: 1650 Ti − 3050 | −0.002 (−0.006 to +0.002) | +0.003 (−0.008 to +0.014) | −0.007 (−0.020 to +0.003) |
| Qwen bf16 − Gemma, 1650 Ti | −0.193 (−0.211 to −0.174) | −0.094 (−0.115 to −0.072) | −0.155 (−0.203 to −0.110) |
| Qwen Q4_0 − Gemma, 1650 Ti | −0.251 (−0.271 to −0.231) | −0.167 (−0.190 to −0.143) | −0.230 (−0.280 to −0.180) |

| gap moved, 1650 Ti − 3050 | suite | GSM8K | BFCL |
| --- | ---: | ---: | ---: |
| Qwen bf16 − Gemma | +0.002 | +0.006 | +0.005 |
| Qwen Q4_0 − Gemma | +0.010 | −0.012 | −0.015 |

- **Gemma matches its 3050 run record for record**: 52 of 3,880 suite records, 48 of 1,319 GSM8K problems and
  5 of 400 BFCL calls differ. The binary, the card and where the per-layer table sits change no score.
- **The gap measured across cards was the gap.** With both models on this card, unquantized Qwen trails
  Gemma's 4-bit build by 19.3 suite points, 9.4 on GSM8K and 15.5 on BFCL.
- **Qwen leads on two subsets, squad2 (+0.130) and boolq (+0.110)**, and trails most on civil_comments
  (−0.667), summeval-consistency (−0.597) and multinli (−0.365), bf16 against Gemma (per-subset table in
  `COMPARE.md`; ranges not corrected for 13 comparisons).
- **Answer length:** 3 Gemma GSM8K replies hit 2,048 tokens, against 52 for Qwen bf16 and 107 for Qwen Q4_0.
- **Tool calls:** 5 Gemma BFCL replies made no tool call, against 48 for Qwen bf16 and 88 for Qwen Q4_0.
- All labels returned on the suite: Gemma 96.3%, Qwen bf16 82.3%, Qwen Q4_0 90.2%.

## Run time on this card

Suite 59.0 min, GSM8K 52.1 min, BFCL 4.8 min. Against Qwen Q4_0 on this card: 1.32x as long on the suite,
0.80x on GSM8K and 0.51x on BFCL. Qwen Q4_0 writes more: 706,288 tokens of GSM8K replies against Gemma's
528,448, and 44,127 of BFCL against 12,763. `driver.log` reads 1035–1860 MHz
at the step boundaries.

## Not measured

Gemma built without QAT, and either model's drafter, are follow-up runs on this card (`2026-10-17-quality-rtn-1650ti`,
`2026-10-17-spec-mtp-vs-gemma-1650ti`). Thinking on, prompt variants.
