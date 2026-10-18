# 2026-10-16 — first light, Qwen3.5-2B Q4_0 on the RTX 3050 OEM

`local-llamacpp-3050oem-qwen352b-q4_0`: Qwen/Qwen3.5-2B (revision `15852e8c`), converted to bf16 GGUF and
quantized to the standard llama-quantize Q4_0 here, no imatrix (SHA-256 `1c27608b…`, `model-sha256.txt`).
Benched beside Gemma 4 E2B's exact Q4_0 GGUF (`419db9a6…`, the file `local-llamacpp-3050oem-2b-q4_0` serves)
on **one llama.cpp binary**, `7182ef71e` (`llama-version.txt`), same card, same hour.

## Method

`bench.sh`: `llama-bench -p 512,2048 -n 128 -r 5 -ngl 99`, five cells in fixed order, alternating models,
each started once the GPU read ≤40 °C, with temperature, SM clock and power before and after
(`results/cells.txt`). Each model runs its serving config: `-fa 1`, f16 KV, `-t 6`; Gemma also places
`per_layer_token_embd` on the card (`-ot`), as its rig serves. Means of the two serving-config cells per model
and all ratios are computed by the script in `results/summary.txt`. Raw JSON in `results/`.

The models' files differ. Gemma's is Q4_0 in every weight matrix, on Google's QAT grid. Qwen's is Q4_0 in 194
matrices (806 MB) with `token_embd`, which is also its tied output head, at Q6_K (417 MB), rounded from bf16
with no QAT. llama-bench counts 1.943B parameters for the Qwen file and 4.629B for Gemma's (E2B's per-layer
embeddings are most of that).

## Results

| cell | pp512 t/s | pp2048 t/s | tg128 t/s |
| :--- | ---: | ---: | ---: |
| Qwen3.5-2B, `-fa 1` | 5211.34 ± 160.67 | 5326.55 ± 26.55 | 132.76 ± 0.23 |
| Gemma 4 E2B, `-fa 1` | 4209.68 ± 139.74 | 4053.14 ± 12.98 | 118.20 ± 0.25 |
| Qwen3.5-2B, `-fa 0` | 5003.36 ± 163.13 | 4941.39 ± 28.02 | 131.18 ± 0.36 |
| Qwen3.5-2B, `-fa 1`, repeat | 5175.55 ± 153.18 | 5305.72 ± 27.67 | 132.93 ± 0.44 |
| Gemma 4 E2B, `-fa 1`, repeat | 4197.69 ± 132.24 | 4047.75 ± 21.64 | 118.39 ± 0.22 |

- **Qwen against Gemma:** 1.123x decode (132.84 against 118.30 tok/s), 1.235x prefill at 512 tokens and
  1.312x at 2048.
- **Qwen's prefill rises with prompt length** (5193 → 5316 t/s from 512 to 2048) where Gemma's falls
  (4204 → 4050). Three of Qwen's four layers are linear attention, whose cost does not grow with the prompt.
- **Flash attention on Qwen:** 1.042x prefill at 512, 1.078x at 2048, 1.012x decode. Kept on.
- **Repeats:** within 0.7% of the first cell for both models. The card held 1890–1912 MHz and peaked at 53 °C.
- Gemma's 118.20 tok/s on `7182ef71e` against 116.14 on `77c4540ce` (2026-10-13, PLE on the host) is the
  sibling's figure moving with the binary and the tensor placement; the comparison above uses only today's.

## Serving

`make serve` (tpu.env: `-ngl 99 -c 131072 -ctk f16 -ctv f16 -fa 1 -t 6 -tb 12 --parallel 1 --reasoning off
--metrics`, no drafter):

- **VRAM: 2995 MiB of 8192** (`serve/vram.txt`). Gemma's sibling at the same context with its PLE table on
  the card read 3610 MiB (2026-10-13).
- llama.cpp ignores Qwen's MTP layer (`blk.24.*`, logged as unused tensors in `serve/llama-server.log`).
- One chat completion, temperature 0, "In one sentence, what is a TPU?": 34 tokens at 121.2 tok/s decode,
  420.0 t/s prompt on 22 tokens (`serve/chat.json`). **The answer is wrong**: "a specialized processing unit
  designed to handle tasks like image processing, video decoding, and neural network inference, often used in
  mobile devices and edge computing." Gemma E2B answered the same prompt correctly on the sibling. One prompt
  is an anecdote, not a quality score.
- `attest.py`: `device=gpu`, `-ngl 99`, `ggml-cuda`, `libcublas`, `libcuda`, `libcudart` mapped, no mismatch
  (`serve/attest.txt`).

## Not measured

Answer quality (suite, GSM8K, BFCL), a Q8_0 or bf16 arm to separate the Q4_0 rounding loss from the model,
concurrency, long context, and Qwen's MTP head as a drafter.
