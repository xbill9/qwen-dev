# local-llamacpp-1650ti-qwen352b-q4_0

`llama-server` from [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp),
driven directly, serving **Qwen/Qwen3.5-2B** as a standard Q4_0 GGUF off one
**NVIDIA GTX 1650 Ti** in a laptop.

**Status 2026-10-16: serving.** First light against Gemma 4 E2B on the same binary
and card: **85.26 tok/s decode (1.05x Gemma), 511.5 t/s prefill at 512 tokens
(1.43x) and 498.8 at 2,048 (1.51x)**, 2951 MiB of 4096 at a 131,072-token
context. Write-up in `benchmarks/runs/2026-10-16-first-light-1650ti/REPORT.md`.

| | |
| --- | --- |
| Platform | `local` — no control plane; the card is in this machine |
| Runtime | `llamacpp` — one process, one GGUF named on the command line, llama.cpp `7182ef71e` |
| Hardware | `1650ti` — TU117, Turing, compute capability 7.5, 4096 MiB, no tensor cores |
| Model | `qwen352b` — `Qwen/Qwen3.5-2B` |
| Encoding | `q4_0` — llama-quantize's standard Q4_0: matrices Q4_0, tied embedding Q6_K, no imatrix, no QAT |

Forked from [`local-llamacpp-3050oem-qwen352b-q4_0`](../local-llamacpp-3050oem-qwen352b-q4_0/):
same model, runtime and serving flags; the card is the difference. Answer quality was
measured there.

## Quick start

```bash
make install                          # pip install -r requirements.txt into the system python3
make build                            # llama.cpp with CUDA for sm_75, -j6
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
