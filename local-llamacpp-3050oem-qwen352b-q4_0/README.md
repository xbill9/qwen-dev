# local-llamacpp-3050oem-qwen352b-q4_0

`llama-server` from [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp),
driven directly, serving **Qwen/Qwen3.5-2B** as a standard Q4_0 GGUF off one
**NVIDIA RTX 3050 OEM** in a desktop workstation.

**Status 2026-10-16: serving.** First light against Gemma 4 E2B on the same binary
and card: **132.84 tok/s decode (1.12x Gemma), 5,193 t/s prefill at 512 tokens
(1.24x) and 5,316 at 2,048 (1.31x)**, 2995 MiB of 8192 at a 131,072-token
context. Write-up in `benchmarks/runs/2026-10-16-first-light-3050oem/REPORT.md`.

**Answer quality is well below Gemma E2B's** on the same harness: suite 0.422 against
0.683, GSM8K 0.743 against 0.898, BFCL 0.710 against 0.925
(`benchmarks/runs/2026-10-16-quality-vs-gemma-3050oem/REPORT.md`).

| | |
| --- | --- |
| Platform | `local` — no control plane; the card is in this machine |
| Runtime | `llamacpp` — one process, one GGUF named on the command line, llama.cpp `7182ef71e` |
| Hardware | `3050oem` — GA106, Ampere, compute capability 8.6, 8192 MiB; host i7-5820K, 46 GiB |
| Model | `qwen352b` — `Qwen/Qwen3.5-2B` |
| Encoding | `q4_0` — llama-quantize's standard Q4_0: matrices Q4_0, tied embedding Q6_K, no imatrix, no QAT |

Forked from [`local-llamacpp-3050oem-2b-q4_0`](../local-llamacpp-3050oem-2b-q4_0/):
same card, runtime and serving flags; the model is the difference.

## Quick start

```bash
make install                          # pip install -r requirements.txt into the system python3
export PATH=/usr/local/cuda/bin:$PATH # nvcc is not on PATH on this host
make build                            # llama.cpp with CUDA for sm_86, -j10
make download                         # Qwen's bf16 safetensors at the pinned revision
make quantize                         # bf16 GGUF, then Q4_0, sha256-checked
make serve                            # llama-server in the foreground; Ctrl-C is a full teardown
make status                           # is it up?
make query                            # one chat completion against 127.0.0.1:8080
```

`tpu.env` is the source of truth for all of it. The directory name is
documentation — never copy a slot value into a flag.

## Layout

```
tpu.env            source of truth — model, hardware, endpoint, serving flags
server.py          MCP server (`MCPServer`): GPU/model info, start/stop, status, query, attest
attest.py          reads which binary is serving port 8080 out of /proc
inspect_gguf.py    re-derives the resident split from the artifact
sweep.py           context × concurrency serving sweep
Makefile           build / download / quantize / serve / status / query / info / test / lint
tests/             offline unittest suite
benchmarks/        schema + README synced from the monorepo root; runs/ holds this rig's runs
```

## See also

`../MODELS.md`, `../HARDWARE.md`, `../NAMING.md`, `CLAUDE.md`.
