# 2026-10-19 — first light, Qwen3.5-2B Q4_0 on the Tesla T4

`gpu-llamacpp-t4-qwen352b-q4_0`: Qwen/Qwen3.5-2B (revision `15852e8c`), converted to bf16 GGUF and quantized
to the standard llama-quantize Q4_0 on this host, no imatrix (SHA-256 `d2bc253a…`, `model-sha256.txt`).
Benched beside Gemma 4 E2B's exact QAT Q4_0 GGUF (`419db9a6…`) on **one llama.cpp binary**, `7182ef71e`
built for sm_75 with CUDA 13.4 (`llama-version.txt`), the commit of the 1650 Ti and 3050 runs.

Host: the Tesla T4 (TU104, sm_75, tensor cores, 15,360 MiB, 70 W) attached to the Compute Engine VM
`debian13`, an `n1-standard-2` (2 vCPU, 7.3 GiB) in `us-west2-b` (`gpu-static.txt`).

## Method

`bench.sh`, the 1650 Ti first-light script with `-t 2` for this VM's 2 vCPUs: `llama-bench -p 512,2048
-n 128 -r 5 -ngl 99`, five cells alternating models, each started at ≤50 °C, temperature, SM clock and power
before and after (`results/cells.txt`). Each model runs its serving config, `-fa 1` and f16 KV; Gemma holds
`per_layer_token_embd` on the card (`-ot`), as the 3050 rig serves it. `summarize.py` computes every mean and
ratio (`results/summary.txt`). `chain.sh` ran this and the two runs beside it in sequence (`chain.log`).

## Results

| cell | pp512 t/s | pp2048 t/s | tg128 t/s |
| :--- | ---: | ---: | ---: |
| Qwen3.5-2B, `-fa 1` | 4443.14 ± 230.97 | 4581.56 ± 22.63 | 130.08 ± 0.22 |
| Gemma 4 E2B, `-fa 1` | 3685.27 ± 130.27 | 3556.69 ± 9.27 | 116.78 ± 0.17 |
| Qwen3.5-2B, `-fa 0` | 4068.13 ± 650.56 | 4268.34 ± 26.03 | 125.42 ± 0.32 |
| Qwen3.5-2B, `-fa 1`, repeat | 4331.69 ± 290.12 | 4513.83 ± 21.98 | 127.72 ± 0.26 |
| Gemma 4 E2B, `-fa 1`, repeat | 3577.14 ± 335.74 | 3532.38 ± 16.42 | 116.17 ± 0.33 |

- **Qwen against Gemma:** 1.107x decode (128.90 against 116.48 tok/s), 1.208x prefill at 512 tokens and
  1.283x at 2,048.
- **Flash attention on Qwen:** 1.078x prefill at 512, 1.065x at 2,048, 1.028x decode. Kept on.
- **Repeats:** the second cell of each model is 0.975–0.995x the first. The card ran at 1590 MHz, its
  maximum, through every cell, and peaked at 58 °C.

## Three cards, one binary

| | T4 | RTX 3050 OEM | GTX 1650 Ti |
| --- | ---: | ---: | ---: |
| Qwen tg128, tok/s | 128.90 | 132.84 | 85.26 |
| Gemma E2B tg128, tok/s | 116.48 | 118.30 | 81.17 |
| Qwen / Gemma, decode | 1.107x | 1.123x | 1.050x |
| Qwen pp512, t/s | 4387 | 5193 | 512 |
| Gemma pp512, t/s | 3631 | 4204 | 359 |
| Qwen / Gemma, prefill 512 | 1.208x | 1.235x | 1.426x |
| Qwen / Gemma, prefill 2048 | 1.283x | 1.312x | 1.507x |

3050: `../../../../local-llamacpp-3050oem-qwen352b-q4_0/benchmarks/runs/2026-10-16-first-light-3050oem`;
1650 Ti: `../../../../local-llamacpp-1650ti-qwen352b-q4_0/benchmarks/runs/2026-10-16-first-light-1650ti`. Each
model file is the one built on that host (the Qwen Q4_0 files share size and revision, not hash).

- **The T4 decodes both models at 0.97–0.98x the 3050** and prefills at 0.84–0.88x. Against the 1650 Ti it
  decodes at 1.44–1.51x and prefills at 8.6–10.7x: the T4 has tensor cores and the 1650 Ti has none (the likely cause, not profiled).
- **Qwen's decode lead is 1.05–1.12x on all three cards**, its prefill lead 1.21–1.51x, smallest on the cards
  with tensor cores.

## Serving

At the rig's `tpu.env` settings (`-c 131072`, one slot, no drafter) not measured yet; at `-c 8192` Qwen
holds 1439 MiB and Gemma 2799 MiB (`../2026-10-19-spec-mtp-vs-gemma-t4`).
