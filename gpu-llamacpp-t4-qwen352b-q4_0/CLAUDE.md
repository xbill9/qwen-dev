# CLAUDE.md — gpu-llamacpp-t4-qwen352b-q4_0

Guidance for working inside this rig. The siblings are not layers; nothing is
imported across a rig boundary. Read this file before changing anything.

## What this rig is

`llama-server` from `ggml-org/llama.cpp`, driven directly, serving
**Qwen/Qwen3.5-2B** (revision `15852e8c`) as a Q4_0 GGUF built here, off the
**Tesla T4** (TU104, sm_75, tensor cores, 15,360 MiB, 70 W) attached to the
Compute Engine VM `debian13` (`n1-standard-2`, 2 vCPU, 7.3 GiB RAM, `us-west2-b`)
that serves the `gpu-vllm-t4-*` rigs. One process, no control plane, Ctrl-C is a
complete teardown.

**Forked 2026-10-19 from `local-llamacpp-1650ti-qwen352b-q4_0`. Slots 1 and 3
differ**: slot 1 is `gpu` because a cloud VM keeps its platform value (`NAMING.md`),
as for `gpu-vllm-t4-2b`. Same model, runtime, llama.cpp commit (`7182ef71e`) and
serving flags, except `-t 2 -tb 2` for the 2 vCPUs and `CACHE_RAM=2048` (below).

## The vLLM server shares the card

**vLLM on this VM holds the same T4 (13.6 GiB) and port 8000.** Stop it with
`~/bin/vllm-t4 stop` before `make serve`, and start it again with
`~/bin/vllm-t4 start` afterwards. This rig serves on 8080, loopback only.

## Host limits

- **RAM is 7.3 GiB.** llama-server's host prompt cache (`-cram`) defaults to
  8,192 MiB; at the default, `-c 32768 --parallel 8` on Qwen reached 6.0 GB
  resident and 6 GB of swap within two minutes (each Qwen entry is ~41 MiB).
  `CACHE_RAM=2048` holds it near 3.5 GB. The prompt cache changes no score
  (`../local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-accuracy-cache-3050oem`).
- **The root disk is 32 GB and nearly full**, so models, llama.cpp and the CUDA
  13.4 toolkit live on `/opt1`, with a 16 GiB `/opt1/swapfile`.
- `make build` runs at `-j2` (`BUILD_JOBS`). Never pass a bare `-j`.

## Model — BUILT 2026-10-19 ON THIS HOST

- `make download` then `make quantize`. **This host's Q4_0 hashes `d2bc253a…`**,
  against `bd99d5ae…` on the 1650 Ti host and `1c27608b…` on the 3050 host: same
  size, revision and commit. The bf16 GGUF (`dc07c774…`) also differs from both
  other hosts'. Converting twice here gives identical bytes, so the converter is
  deterministic per host and the difference comes from the software around it.
  Which tensors differ is untested. `MODEL_SHA256` pins this host's file.
- File contents as on the sisters: 194 matrices Q4_0, tied `token_embd` Q6_K,
  F32 norms, Qwen's MTP layer `blk.24.nextn.*` unused unless drafting.
- No lazy tensors: everything is resident. Gemma's `per_layer_token_embd`
  rules do not apply here.
- `REASONING=off`; `query_model` keeps `max_tokens` 1024 for requests that turn
  thinking on.

## Measurements

`benchmarks/runs/2026-10-19-first-light-t4/` — llama-bench pp512/pp2048/tg128,
r=5, Qwen and Gemma E2B's exact QAT Q4_0 alternating on one binary (`7182ef71e`,
sm_75), means of two cells each:

| | pp512 | pp2048 | tg128 |
| --- | ---: | ---: | ---: |
| Qwen3.5-2B Q4_0 | 4387 | 4548 | **128.90** |
| Gemma 4 E2B exact Q4_0 | 3631 | 3545 | 116.48 |
| Qwen / Gemma | 1.208x | 1.283x | 1.107x |

- The T4 decodes both models at 0.97–0.98x the RTX 3050 and 1.44–1.51x the GTX
  1650 Ti; it prefills at 0.84–0.88x the 3050 and 8.6–10.7x the 1650 Ti.
- `-fa 1` against `-fa 0` on Qwen: 1.065–1.078x prefill, 1.028x decode. On.
- The card held 1590 MHz, its maximum, through every cell and peaked at 58 °C.
- **Serving VRAM at `-c 131072` is not measured.** At `-c 8192` Qwen holds 1439 MiB.

**MTP head as a drafter** (`benchmarks/runs/2026-10-19-spec-mtp-vs-gemma-t4/`):
one request at a time, 1 drafted token gives 1.096x greedy (130.9 tok/s) and
1.078x sampled, greedy text identical to no drafting; 2 and 3 tokens are slower
sampled. Gemma's QAT drafter at 1 token gives 1.224x (131.7), so with both
drafting the two tie (0.99x). Not served: `SPEC_DRAFT_MODEL` stays empty.

**Answer quality** (`benchmarks/runs/2026-10-19-quality-t4/`, `-c 32768
--parallel 8`, both models and Gemma's drafter on one binary): Gemma E2B leads
this file by 26.0 suite points (0.681 vs 0.422), 15.8 on GSM8K (0.892 vs 0.735)
and 22.5 on BFCL (0.922 vs 0.698), within 1 point of the 1650 Ti's gap. Gemma
scores on this card what it scored on the 1650 Ti; Qwen's suite is 0.8 points
lower, the 3050 host's file's score. Gemma's drafter at 1 token leaves all three
level, 86.5% of drafted tokens accepted, GSM8K 1.42x faster.

**Not measured here:** concurrency sweeps, long context, q8_0 KV, Qwen bf16,
Qwen's MTP head at 8 requests, thinking on.

## Inherited and still true here

- Raw `/v1/completions` returns an empty completion on chat models; use
  `/v1/chat/completions`.
- `sweep.py`: `decode_tps` counts inter-chunk gaps, not tokens; check the prompt
  cache was defeated via `/metrics`; llama.cpp splits `--ctx-size` across
  `--parallel` slots. Its default `--model` and prompt text still name Gemma.
- `start_model_server` spawns with `subprocess.Popen`, because asyncio's
  subprocess transport kills a live child when it is torn down.
- On the sisters llama.cpp logs `common_fit_params: failed to fit params ... abort`
  at load because `-ngl` is set, and the model serves. None of this rig's server
  logs carries that line.

## Conventions

- Tests are `unittest`, never pytest: `python3 -m unittest discover -s tests -v`.
- Every subprocess call goes through `run_command(cmd: list[str])` using
  `asyncio.create_subprocess_exec`. **Never `shell=True`.**
- MCP tools are `async def` returning markdown strings with emoji status
  prefixes (`✅`, `❌`, `📡`).
- `Optional[str]`, not `X | None`.
- Use the system `python3` and install into it. **Never create a virtualenv.**
- `tpu.env` is the source of truth and is committed. Never add `*.env` to
  `.gitignore`.

## Canonical root references

`@MODELS.md`, `@HARDWARE.md`, `@QUANTIZATION.md`, `@NAMING.md`,
`@RIG-ANALYSIS.md`. They describe Gemma 4; nothing in them has been checked
against Qwen yet.
