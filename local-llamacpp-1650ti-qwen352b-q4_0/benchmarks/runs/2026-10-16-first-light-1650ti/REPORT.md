# 2026-10-16 — first light, Qwen3.5-2B Q4_0 on the GTX 1650 Ti

`local-llamacpp-1650ti-qwen352b-q4_0`: Qwen/Qwen3.5-2B (revision `15852e8c`), converted to bf16 GGUF and
quantized to the standard llama-quantize Q4_0 on this host, no imatrix (SHA-256 `bd99d5ae…`,
`model-sha256.txt`). Benched beside Gemma 4 E2B's exact Q4_0 GGUF (`419db9a6…`, the file
`local-llamacpp-1650ti-2b-q4_0` serves) on **one llama.cpp binary**, `7182ef7` built for sm_75
(`llama-version.txt`), same card, same hour.

## Method

`bench.sh`, the 3050 sister's script with two changes: `llama-bench -p 512,2048 -n 128 -r 5 -ngl 99`, five
cells in fixed order, alternating models, each started once the GPU read ≤50 °C (this card idles at 45–50 °C),
with temperature, SM clock and power before and after (`results/cells.txt`). Each model runs its serving
config on this card: `-fa 1`, f16 KV, `-t 6`; Gemma leaves `per_layer_token_embd` in the mmap, as its rig
serves here. `summarize.py` computes the means of the two serving-config cells per model and every ratio
(`results/summary.txt`). Raw JSON in `results/`.

The Q4_0 file has the same size as the 3050 host's (1,236,741,088 B) and a different SHA-256 (`1c27608b…`
there); `../2026-10-16-parity-crosshost-1650ti/` measures what that changes.

## Results

| cell | pp512 t/s | pp2048 t/s | tg128 t/s |
| :--- | ---: | ---: | ---: |
| Qwen3.5-2B, `-fa 1` | 511.36 ± 1.51 | 499.05 ± 1.02 | 85.39 ± 0.02 |
| Gemma 4 E2B, `-fa 1` | 358.82 ± 0.81 | 331.10 ± 0.09 | 81.41 ± 0.05 |
| Qwen3.5-2B, `-fa 0` | 507.95 ± 1.20 | 495.39 ± 0.23 | 84.17 ± 0.03 |
| Qwen3.5-2B, `-fa 1`, repeat | 511.63 ± 1.16 | 498.59 ± 0.17 | 85.14 ± 0.07 |
| Gemma 4 E2B, `-fa 1`, repeat | 358.65 ± 0.63 | 330.89 ± 0.10 | 80.94 ± 0.12 |

- **Qwen against Gemma:** 1.050x decode (85.26 against 81.17 tok/s), 1.426x prefill at 512 tokens and
  1.507x at 2048. On the 3050 the same pair measured 1.123x / 1.235x / 1.312x.
- **Flash attention on Qwen:** 1.007x prefill at both lengths, 1.013x decode. Kept on.
- **Repeats:** within 0.6% of the first cell for both models. The card held 1860–1875 MHz and peaked at 59 °C.
- **Against the 3050 sister**, same model and commit: 0.642x decode and 0.094–0.098x prefill. Prefill on this
  card runs without tensor cores.

## Serving

`llama-server` with tpu.env's flags (`-ngl 99 -c 131072 -ctk f16 -ctv f16 -fa 1 -t 6 -tb 12 --parallel 1
--reasoning off --metrics`, `--host 127.0.0.1`, no drafter):

- **VRAM: 2951 MiB of 4096** at the full 131,072-token context (`serve/vram.txt`); only 6 of 24 layers hold
  a KV cache. The 3050 sister read 2995 MiB at the same context.
- llama.cpp logs Qwen's MTP layer (`blk.24.*`) as unused tensors, and logs `common_fit_params: failed to fit
  params ... n_gpu_layers already set by user to 99, abort`: the automatic fit step stands down because `-ngl`
  is given, and the model then loads and serves (`serve/llama-server.log`).
- One chat completion, temperature 0, "In one sentence, what is a TPU?": 37 tokens at 76.7 tok/s decode,
  163.8 t/s prompt on 22 tokens (`serve/chat.json`). The answer is wrong, as on the 3050: "a specialized
  processor designed to handle tasks like image processing, text analysis, and data manipulation".
- `attest.py`: `device=gpu`, `-ngl 99`, `ggml-cuda`, `libcublas`, `libcuda`, `libcudart` mapped, no mismatch
  (`serve/attest.txt`).

## Not measured

Answer quality on this card (suite, GSM8K, BFCL), concurrency, long context, q8_0 KV, and Qwen's MTP head as a
drafter here.
