# 2026-10-17 — does QAT explain Gemma 4 E2B's lead over Qwen3.5-2B? GTX 1650 Ti

**Question:** Gemma E2B's Q4_0 file holds Google's quantization-aware-trained weights; Qwen publishes no QAT
checkpoint, so its Q4_0 is plain rounding. How much of Gemma's lead is QAT?

**Answer: a tenth to a fifth of it.** Gemma rounded from its ordinary bf16 release, built the way Qwen's file
was, scores 0.632 / 0.881 / 0.895 on suite / GSM8K / BFCL, 4.8 / 2.0 / 2.2 points below the QAT file. It still
leads Qwen Q4_0 by 20.2 / 14.7 / 20.7 points and Qwen bf16 by 14.4 / 7.4 / 13.3.

## The file

`run_quality.sh` step 0: `convert_hf_to_gguf.py --outtype bf16` from `google/gemma-4-E2B-it` (snapshot
`3e22461f`), then `llama-quantize … Q4_0` with no imatrix, llama.cpp `7182ef71e`, the recipe of
`local-llamacpp-1650ti-qwen352b-q4_0`'s `make quantize`. SHA-256 in `model-sha256.txt` (Q4_0 `c96dec27…`).
`compare.py` reads both GGUFs with gguf-py:

| file | size | by tensor type |
| --- | ---: | --- |
| Gemma E2B rounded Q4_0 | 3.344 GB | Q6_K 67.5%, Q4_0 31.7%, bf16 0.8% |
| Gemma E2B QAT Q4_0 (`419db9a6…`) | 2.605 GB | Q4_0 100% |
| Qwen3.5-2B Q4_0 (`bd99d5ae…`) | 1.226 GB | Q4_0 65.8%, Q6_K 34.0% |

- **The rounded file keeps its embedding tables at Q6_K** (`token_embd`, `per_layer_token_embd`) and
  `per_layer_model_proj` at bf16, the `llama-quantize` default; Qwen's tied `token_embd` is Q6_K the same way.
  The weight matrices are Q4_0 in all three. The rounded Gemma file therefore stores its tables more precisely
  than the QAT file does.
- It also carries 60 tensors the QAT export omits: `attn_k`, `attn_k_norm` and `attn_v` for the 20 layers
  that share an earlier layer's KV cache. Whether llama.cpp reads them is untested. Every metadata key outside
  `general.*` is identical between the two files; the converter's `Unknown RoPE type: proportional` warning
  (`convert.log`) leaves the RoPE keys the same as the QAT file's.

## Setup

The QAT run's method exactly (`../../../../local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-17-quality-vs-qwen-1650ti`):
same binary, `-c 32768 --parallel 8 -fa 1`, f16 KV, `-t 6 -tb 12`, `--reasoning off`, no drafter; the Nimble
13-subset suite (3,880 records), GSM8K (1,319, 2,048-token limit) and BFCL simple (400), 8 requests at a time.
Server 1838 MiB (QAT file: 1710). `compare.py` pairs every arm on the same records: full tables in
`COMPARE.md`, numbers in `compare.json`. No errors in any task.

## Results, all on this card

| task | Gemma QAT Q4_0 | **Gemma rounded Q4_0** | Qwen bf16 | Qwen Q4_0 |
| --- | ---: | ---: | ---: | ---: |
| Suite (3,880) | 0.681 | **0.632** | 0.488 | 0.430 |
| GSM8K (1,319) | 0.901 | **0.881** | 0.807 | 0.734 |
| BFCL simple (400) | 0.917 | **0.895** | 0.762 | 0.688 |

| paired difference (95% range) | suite | GSM8K | BFCL |
| --- | --- | --- | --- |
| Gemma: rounded − QAT | −0.048 (−0.061 to −0.035) | −0.020 (−0.034 to −0.005) | −0.022 (−0.050 to +0.005) |
| Qwen Q4_0 − Gemma rounded | −0.202 (−0.222 to −0.183) | −0.147 (−0.171 to −0.124) | −0.207 (−0.263 to −0.155) |
| Qwen bf16 − Gemma rounded | −0.144 (−0.163 to −0.125) | −0.074 (−0.096 to −0.052) | −0.133 (−0.182 to −0.083) |

| Qwen Q4_0's gap to Gemma QAT | suite | GSM8K | BFCL |
| --- | ---: | ---: | ---: |
| whole gap | −0.251 | −0.167 | −0.230 |
| QAT's part (rounded − QAT) | −0.048 | −0.020 | −0.022 |
| **QAT's share** | **19%** | **12%** | **10%** |

- **QAT is worth 2 to 5 points to Gemma at 4 bits**, clear of zero on the suite and GSM8K. Rounding costs
  Qwen 5.8 / 7.3 / 7.5 points against its bf16 file on this card; Gemma's rounded loss against its QAT file is
  smaller on every task. The two losses are measured against different references (Qwen's bf16, Gemma's QAT
  file), so this is a comparison of sizes, not a paired test.
- **The suite loss is concentrated**: pubmedqa −0.376 and summeval-consistency −0.271 against the QAT file,
  with boolq −0.087 and vitaminc-dev −0.083; 6 of 13 subsets move less than 3 points either way. All labels
  returned falls from 96.3% to 83.6%.
- **Rounding makes Gemma skip the tool more**: 17 BFCL replies with no tool call against 5 for the QAT file,
  48 for Qwen bf16 and 88 for Qwen Q4_0.
- **Answer length holds**: 4 GSM8K replies at the 2,048-token limit, against 3 for the QAT file and 107 for
  Qwen Q4_0.
- **Rounded Gemma still beats unrounded Qwen on all three tasks**, every range clear of zero.

## Run time

Suite 58.5 min, GSM8K 59.5 min, BFCL 5.0 min (`driver.log`; QAT file 59.0 / 52.1 / 4.8).

## Not measured

Gemma bf16 on llama.cpp (it does not fit this card), an imatrix build of either model, a rounded build with
the tables at Q4_0, Qwen QAT (none published).
