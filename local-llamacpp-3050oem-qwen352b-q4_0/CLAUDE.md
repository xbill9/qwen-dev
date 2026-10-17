# CLAUDE.md — local-llamacpp-3050oem-qwen352b-q4_0

Guidance for working inside this rig. The siblings are not layers; nothing is
imported across a rig boundary. Read this file before changing anything.

## What this rig is

`llama-server` from `ggml-org/llama.cpp`, driven directly, serving
**Qwen/Qwen3.5-2B** (revision `15852e8c`) as a Q4_0 GGUF built here, off the
same **RTX 3050 OEM** in the desktop `gpu` that serves
`local-llamacpp-3050oem-2b-q4_0`. One process, no control plane, Ctrl-C is a
complete teardown.

**Forked 2026-10-16 from `local-llamacpp-3050oem-2b-q4_0`. Slot 4 is the only
slot that differs**: same card, runtime and serving flags. The pair is a
model-against-model comparison, with one caveat that decides how to read it:
Gemma's file is Q4_0 on Google's **QAT** grid, and Qwen publishes no QAT
checkpoint and no GGUF of this model, so this file is plain rounding from bf16.
A quality gap between the two may be the rounding as much as the model.

**Slot 4 is `qwen352b`**: Qwen family, version 3.5, size 2B, no separators
(`NAMING.md` slot 4). A bare `2b` would claim Gemma 4 E2B.

## Model — BUILT 2026-10-16

- `make download` fetches Qwen's bf16 safetensors to `MODEL_HF_DIR`;
  `make quantize` runs `convert_hf_to_gguf.py --outtype bf16` and then
  `llama-quantize ... Q4_0`, with no imatrix, and checks `MODEL_SHA256`. The
  hash is what `7182ef71e` produced; another llama.cpp commit may differ.
- The standard Q4_0 mix is **not all Q4_0**: 194 matrices Q4_0 (806 MB),
  `token_embd` Q6_K (417 MB, 34% of the file, also the tied output head),
  F32 norms. `make info` re-derives it.
- 24 layers: three of every four Gated DeltaNet (linear attention), every
  fourth full attention. Only the 6 full-attention layers hold a KV cache.
- No lazy tensors: everything is resident. Gemma's `per_layer_token_embd`
  rules (`-ot`, never `--no-mmap`) do not apply here.
- The file carries Qwen's MTP layer (`blk.24.nextn.*`); llama.cpp logs it as
  unused unless speculative decoding is configured. Untested as a drafter.
- Qwen3.5 thinks by default. `REASONING=off` in `tpu.env`; a request can turn
  it on with `chat_template_kwargs`, and then `content` stays empty until the
  thinking closes, so `query_model` keeps `max_tokens` 1024.

## Measurements

`benchmarks/runs/2026-10-16-first-light-3050oem/` — llama-bench pp512/pp2048/tg128,
r=5, Qwen and Gemma E2B alternating on one binary (`7182ef71e`):

| | pp512 | pp2048 | tg128 |
| --- | ---: | ---: | ---: |
| Qwen3.5-2B Q4_0 | 5193.44 | 5316.13 | **132.84** |
| Gemma 4 E2B exact Q4_0 | 4203.68 | 4050.45 | 118.30 |
| Qwen / Gemma | 1.235x | 1.312x | 1.123x |

- `-fa 1` against `-fa 0`: 1.042x / 1.078x prefill, 1.012x decode. On.
- **2995 MiB of 8192** serving at `-c 131072`, no drafter.
- One sanity chat ("what is a TPU?") came back fluent and wrong.

**Answer quality, `benchmarks/runs/2026-10-16-quality-vs-gemma-3050oem/`** —
the Gemma quality run's harness and settings (`-c 32768 --parallel 8`, no
drafter), scored against its E2B/E4B records:

| | Gemma E2B | Gemma E4B | Qwen3.5-2B |
| --- | ---: | ---: | ---: |
| Suite (3,880) | 0.683 | 0.730 | **0.422** |
| GSM8K (1,319) | 0.898 | 0.933 | **0.743** |
| BFCL simple (400) | 0.925 | 0.910 | **0.710** |

Every gap to E2B has a range clear of zero. On letter-choice subsets Qwen
writes the option name instead of the letter (label mass 0.41–0.63 against
Gemma's 0.98+); on BFCL 79 of 116 misses solve the problem in prose with no
tool call; on GSM8K 100 replies hit 2,048 tokens against Gemma's 5. **The suite
harness needs `--tokenizer Qwen/Qwen3.5-2B`** for this model: its default is a
Gemma tokenizer, whose label ids mean nothing in Qwen's vocabulary.

**bf16 arm** (`../local-llamacpp-3050oem-qwen352b/benchmarks/runs/2026-10-16-quality-bf16-3050oem/`):
suite 0.488, GSM8K 0.798, BFCL 0.765. Q4_0 costs 5.5–6.5 points per task, a
quarter to a third of the gap; Gemma E2B Q4 still leads bf16 Qwen on all three.

**Qwen's own MTP head as drafter** (`benchmarks/runs/2026-10-16-spec-mtp-3050oem/`):
`--spec-type draft-mtp` with no `-md` drafts from `blk.24.nextn`. 1 drafted
token is best: 1.149x greedy (124.2 → 142.7 tok/s), 1.086x sampled; 2 ties on
greedy; 3+ loses. Greedy text byte-identical to no draft at 1–3 tokens. **Not
yet the serving default**: `make serve` and `server.py` only enable drafting
when `SPEC_DRAFT_MODEL` names a separate file.

**Not measured:** Q8_0, thinking on, concurrency, long context, q8_0 KV.

## Inherited and still true here

- Raw `/v1/completions` returns an empty completion on chat models; use
  `/v1/chat/completions`.
- `sweep.py`: `decode_tps` counts inter-chunk gaps, not tokens; check the prompt
  cache was defeated via `/metrics`; llama.cpp splits `--ctx-size` across
  `--parallel` slots. Its default `--model` and prompt text still name Gemma.
- `start_model_server` spawns with `subprocess.Popen`, because asyncio's
  subprocess transport kills a live child when it is torn down.
- Port 8080 is shared with every `local-llamacpp-*` rig. Only one serves at a
  time on a host.
- `nvcc` is not on PATH: `export PATH=/usr/local/cuda/bin:$PATH` before
  `make build`. Never pass a bare `-j`.

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
