# gpu-llamacpp-t4-qwen352b-q4_0

`llama-server` from [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp),
driven directly, serving **Qwen/Qwen3.5-2B** as a standard Q4_0 GGUF off one
**NVIDIA Tesla T4** attached to the Compute Engine VM `debian13`.

**Status 2026-10-19: serving.** First light against Gemma 4 E2B on the same binary
and card: **128.90 tok/s decode (1.107x Gemma), prefill 1.208x Gemma at 512 tokens
and 1.283x at 2,048** (`benchmarks/runs/2026-10-19-first-light-t4/REPORT.md`).
Serving VRAM at the rig's `-c 131072` is not measured yet; at `-c 8192` it holds 1439 MiB.

| | |
| --- | --- |
| Platform | `gpu` — a Compute Engine VM (`n1-standard-2`, 2 vCPU, 7.3 GiB, `us-west2-b`) with the card attached |
| Runtime | `llamacpp` — one process, one GGUF named on the command line, llama.cpp `7182ef71e` |
| Hardware | `t4` — TU104, Turing, compute capability 7.5, tensor cores, 15,360 MiB, 70 W |
| Model | `qwen352b` — `Qwen/Qwen3.5-2B` |
| Encoding | `q4_0` — llama-quantize's standard Q4_0: matrices Q4_0, tied embedding Q6_K, no imatrix, no QAT |

Forked from [`local-llamacpp-1650ti-qwen352b-q4_0`](../local-llamacpp-1650ti-qwen352b-q4_0/):
same model, runtime, llama.cpp commit and serving flags; the card and host are the difference,
with `-t 2 -tb 2` for the VM's 2 vCPUs and `-cram 2048` because llama-server's default 8,192 MiB
host prompt cache is more than its RAM. **The vLLM server on this VM holds the same card and
port 8000**: stop it with `~/bin/vllm-t4 stop` before `make serve`.

## Measurements

- **Answer quality** (`benchmarks/runs/2026-10-19-quality-t4/`): Gemma E2B's QAT Q4_0 leads this
  file by 26.0 suite points (0.681 vs 0.422), 15.8 on GSM8K (0.892 vs 0.735) and 22.5 on BFCL
  (0.922 vs 0.698), within 1 point of the gap on the 1650 Ti. Gemma's drafter leaves all three
  level and finishes GSM8K 1.42x faster.
- **Drafting** (`benchmarks/runs/2026-10-19-spec-mtp-vs-gemma-t4/`): Qwen's own MTP head at 1
  drafted token gives 1.096x greedy (130.9 tok/s), Gemma's QAT drafter 1.224x (131.7), so with
  both drafting the two tie. Not served: `SPEC_DRAFT_MODEL` stays empty.

## Quick start

```bash
make install                          # pip install -r requirements.txt into the system python3
make build                            # llama.cpp with CUDA for sm_75, -j2
make download                         # Qwen's bf16 safetensors at the pinned revision
make quantize                         # bf16 GGUF, then Q4_0, sha256-checked
~/bin/vllm-t4 stop                    # free the card
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
