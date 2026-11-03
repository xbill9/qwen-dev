# CLAUDE.md — local-llamacpp-1650ti-qwen352b-q4_0

Guidance for working inside this rig. The siblings are not layers; nothing is
imported across a rig boundary. Read this file before changing anything.

## What this rig is

`llama-server` from `ggml-org/llama.cpp`, driven directly, serving
**Qwen/Qwen3.5-2B** (revision `15852e8c`) as a Q4_0 GGUF built here, off the
same **GTX 1650 Ti** (TU117, sm_75, 4 GiB, no tensor cores) in the laptop
`yogagpu` that serves `local-llamacpp-1650ti-2b-q4_0`. One process, no control
plane, Ctrl-C is a complete teardown.

**Forked 2026-10-16 from `local-llamacpp-3050oem-qwen352b-q4_0`. Slot 3 is the
only slot that differs**: same model, runtime, llama.cpp commit (`7182ef71e`) and
serving flags, except `HOST=127.0.0.1` as on every rig on this laptop. Against
`local-llamacpp-1650ti-2b-q4_0` only slot 4 differs, so this card now carries a
Qwen/Gemma pair measured on one binary.

## Model — BUILT 2026-10-16 ON THIS HOST

- `make download` then `make quantize`, as on the sister. **This host's file
  hashes `bd99d5ae…`, the 3050 host's `1c27608b…`**: same size, same revision,
  same commit, and the bf16 GGUFs differ too, so the difference enters at
  `convert_hf_to_gguf.py`. Which bytes differ is untested. On 65 suite prompts
  the two files pick the same label 62 times, median label-probability gap
  0.018 against Q4_0-vs-fp32's 0.115 (`benchmarks/runs/2026-10-16-parity-crosshost-1650ti`).
  `MODEL_SHA256` pins this host's file.
- File contents as on the sister: 194 matrices Q4_0, tied `token_embd` Q6_K,
  F32 norms, Qwen's MTP layer `blk.24.nextn.*` unused unless drafting.
- No lazy tensors: everything is resident. Gemma's `per_layer_token_embd`
  rules do not apply here.
- `REASONING=off`; `query_model` keeps `max_tokens` 1024 for requests that turn
  thinking on.

## Measurements

`benchmarks/runs/2026-10-16-first-light-1650ti/` — llama-bench pp512/pp2048/tg128,
r=5, Qwen and Gemma E2B alternating on one binary (`7182ef7`, sm_75):

| | pp512 | pp2048 | tg128 |
| --- | ---: | ---: | ---: |
| Qwen3.5-2B Q4_0 | 511.50 | 498.82 | **85.26** |
| Gemma 4 E2B exact Q4_0 | 358.74 | 330.99 | 81.17 |
| Qwen / Gemma | 1.426x | 1.507x | 1.050x |

- On the 3050 the same pair is 1.235x / 1.312x / 1.123x. Against the sister
  this card runs Qwen at 0.642x decode and 0.094–0.098x prefill.
- `-fa 1` against `-fa 0`: 1.007x prefill, 1.013x decode. On.
- **2951 MiB of 4096** serving at `-c 131072`, no drafter. Gemma's sibling on
  this card serves 8,192.
- One sanity chat ("what is a TPU?") came back fluent and wrong, as on the sister.

**Answer quality on this card matches the sister's**
(`benchmarks/runs/2026-10-16-quality-1650ti/`, both files, same flags as the
3050 runs): Q4_0 suite 0.430, GSM8K 0.734, BFCL 0.688 against the 3050's
0.422 / 0.743 / 0.710, only the suite's +0.007 clear of zero; bf16 0.488 /
0.807 / 0.762 against 0.488 / 0.798 / 0.765. Q4_0 costs 5.8–7.5 points per
task; Gemma E2B (3050) leads Q4_0 by 25.3 / 16.4 / 23.7. Runs took 1.7–6.9x
as long as on the 3050, with the card below its first-light clocks.

**bf16 arm** (`../local-llamacpp-1650ti-qwen352b/`): does not fit this card whole, so 6 of 24 blocks
run on the CPU; 20.22 tok/s decode (0.24x this file), prefill 0.96x, 3207 MiB at `-c 8192`. Its logits
match the 3050 host's bf16 on all 65 parity prompts.

**Against Gemma E2B on this card, 2026-10-17**: on one binary and flags Gemma's
QAT Q4_0 leads this file by 25.1 / 16.7 / 23.0 points and the bf16 file by
19.3 / 9.4 / 15.5 (`../local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-17-quality-vs-qwen-1650ti`);
Gemma rounded without QAT still leads by 20.2 / 14.7 / 20.7
(`../local-llamacpp-1650ti-2b-q4_0rtn/benchmarks/runs/2026-10-17-quality-rtn-1650ti`).

**MTP head as a drafter** (`benchmarks/runs/2026-10-17-spec-mtp-vs-gemma-1650ti`,
`.../2026-10-17-quality-mtp-1650ti`): one request at a time, 1 drafted token
gives 1.107x greedy (91.0 tok/s) and 1.052x sampled, greedy text identical;
Gemma's drafter reaches 92.1. At `--parallel 8` it leaves GSM8K and BFCL level
and makes each task 1.08–1.09x slower, 90.1% of drafts accepted. Not served:
`SPEC_DRAFT_MODEL` stays empty.

**Not measured here:** concurrency sweeps, long context, q8_0 KV.

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
- llama.cpp logs `common_fit_params: failed to fit params ... abort` at load:
  the automatic fit stands down because `-ngl` is set. The model serves.
- `make build` runs at `-j6`: an unbounded CUDA build OOMs this 15 GiB laptop.
  Never pass a bare `-j`.

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
