# 2026-10-16 — first light, Qwen3.5-2B bf16 on the GTX 1650 Ti

`local-llamacpp-1650ti-qwen352b`: Qwen/Qwen3.5-2B (revision `15852e8c`) as the bf16 GGUF that
`local-llamacpp-1650ti-qwen352b-q4_0`'s `make quantize` converts on this host (SHA-256 `ecbfdc76…`,
`model-sha256.txt`), on llama.cpp `7182ef7` built for sm_75 (`llama-version.txt`).

**Answer: it runs, split between the card and the CPU, at about 20 tok/s decode.** The file holds 3.62 GiB
of weights and the card 4 GiB, so all of it on the card leaves no room for a compute buffer.

## Fit

`llama-bench -ngl 99` loads every weight and then fails to allocate its compute buffer, at 501.52 MiB with the
default micro-batch of 512 and still at 62.69 MiB with `-ub 64`. Some weights have to stay on the CPU.

## Method

`bench.sh`: `llama-bench -p 512,2048 -n 128 -r 3 -fa 1`, f16 KV, `-t 6`, each cell started once the GPU read
≤50 °C. Two kinds of split for the bf16 file:

- `cpuK`: `-ngl 99` with the last K of the 24 blocks on the CPU (`-ot 'blk\.(…)\.=CPU'`), so the tied output
  head stays on the card.
- `ngl20`: llama.cpp's own split, blocks 20–23 and the output head on the CPU.

And the Q4_0 file at full offload as the reference. `summarize.py` computes the table and ratios
(`results/summary.txt`); raw JSON and logs in `results/`.

## Results

| cell | pp512 t/s | pp2048 t/s | tg128 t/s | tg128 against Q4_0 |
| :--- | ---: | ---: | ---: | ---: |
| bf16, 2 blocks on CPU | out of memory | | | |
| bf16, 4 blocks on CPU | out of memory | | | |
| bf16, 6 blocks on CPU | 490.87 ± 3.22 | 477.44 ± 0.48 | **20.22 ± 0.51** | 0.238x |
| bf16, 8 blocks on CPU | 481.30 ± 1.25 | 470.72 ± 0.06 | 18.14 ± 0.05 | 0.213x |
| bf16, `-ngl 20` | 485.79 ± 1.31 | 471.65 ± 2.05 | 19.90 ± 0.32 | 0.234x |
| Q4_0, full offload | 511.43 ± 1.30 | 496.92 ± 0.23 | 85.11 ± 0.01 | 1x |

- **6 blocks on the CPU is the fewest that fits**; with 2 or 4, the 501.52 MiB compute buffer fails at
  context creation.
- **Decode is 0.21–0.24x the Q4_0 file's in every split, and prefill 0.94–0.96x.** Where the output head
  sits changes decode by 1.6% (6 blocks with the head on the card against `-ngl 20` with it on the CPU).
- The card held 1860–1875 MHz and peaked at 62 °C (`results/cells.txt`).

## Serving

`llama-server -ngl 99 -ot 'blk\.(18|19|20|21|22|23)\.=CPU' -c 8192 -ctk f16 -ctv f16 -fa 1 -t 6 -tb 12
--parallel 1 --reasoning off --metrics`, `--host 127.0.0.1`:

- **VRAM: 3207 MiB of 4096** (`serve/vram.txt`).
- One chat completion, temperature 0, "In one sentence, what is a TPU?": 33 tokens at 20.2 tok/s decode,
  84.4 t/s prompt on 22 tokens (`serve/chat.json`). The answer is right: "A TPU (Tensor Processing Unit) is a
  specialized hardware component designed to accelerate machine learning and deep neural network tasks".
  The Q4_0 file's answer to the same prompt on this card is wrong. One prompt.
- `attest.py` (from the Q4_0 rig): `device=gpu`, `-ngl 99`, `ggml-cuda`, `libcublas`, `libcuda`,
  `libcudart` mapped, no mismatch (`serve/attest.txt`).

## Not measured

Answer quality on this card, a `-ub` below 512 at 4 blocks on the CPU, contexts above 8,192, concurrency.
