# 2026-10-16 — does llama.cpp run Qwen3.5-2B correctly?

**Answer: yes.** llama.cpp builds the same prompt as Qwen's own chat template, and its bf16 next-token
probabilities are closer to a full-precision transformers reference than transformers' own bf16 path is.
The Q4_0 build is the source of the drift: it changes the chosen answer on 22 of 65 suite prompts.

## Prompt rendering

`template_check.py` renders each prompt through llama-server's `/apply-template` (the template embedded in the
GGUF, llama.cpp `7182ef71e`) and through transformers' `apply_chat_template` on the Hugging Face checkout
(revision `15852e8c`, `enable_thinking=False`): **all 400 BFCL simple prompts, with their tools, and 100 GSM8K
prompts are byte-identical** (`template_check.txt`).

## Next-token probabilities

`logit_check.py` builds the suite harness's exact prompts (`run_suite.own_prompt`) for the first 5 records of
each of the 13 subsets and reads the top 32 next-token log-probabilities. The reference is transformers 5.12.1
on the CPU in float32 over Qwen's bf16 weights (`reference-fp32.jsonl`); each test is read the way the suite
reads it (top 32, a label outside them scored at the 32nd token's value, renormalised over the labels).

| test, against transformers fp32 | prompts | same chosen label | same top-1 token | label-probability difference, median / max | top-5 log-probability difference, median / max |
|---|---:|---:|---:|---|---|
| transformers bf16 (GPU) | 65 | 61 | 59 | 0.0128 / 0.563 | 0.124 / 2.98 |
| llama.cpp bf16 GGUF | 65 | 62 | 60 | 0.0027 / 0.562 | 0.050 / 3.35 |
| llama.cpp Q4_0 GGUF | 65 | 43 | 47 | 0.1136 / 0.488 | 1.638 / 5.13 |

- The largest differences sit on the same massive-en-US prompts for both bf16 paths: 18 labels share little
  probability there, so small changes in low-probability tokens move the renormalised label probabilities.
  Transformers' bf16 moves them as far as llama.cpp's does.
- Q4_0 moves the top-5 log-probabilities 33 times further than llama.cpp's bf16 at the median.

## Setup

transformers 5.12.1 (Debian), PyTorch 2.14.1+cu130 installed for this check into its own user base
(`PYTHONUSERBASE=/home/xbill/pyuser-ref`, with torchvision 0.29.1), because Debian's PyTorch has no
optimised BLAS. The transformers DeltaNet layers ran on its PyTorch fallback (flash-linear-attention not
installed). llama-server: `-ngl 99 -c 8192 -fa 1 --reasoning off`, one GGUF at a time on :8080.

`summary-*.json` and `logit_check-*.jsonl` hold every per-prompt number; `SUMMARY.md` is the table above.
