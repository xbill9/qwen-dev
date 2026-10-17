# CLAUDE.md — local-llamacpp-1650ti-qwen352b

**An artifact rig: measurements only.** It holds results for `Qwen/Qwen3.5-2B` at bf16, the reference
build (so no slot-5 encoding), served by llama.cpp on the GTX 1650 Ti. No `server.py`, no MCP server,
no `tpu.env`, and none is owed (`NAMING.md`, artifact rigs). The serving rig is
`local-llamacpp-1650ti-qwen352b-q4_0`; its `make quantize` writes the bf16 GGUF this rig's runs used
(`/home/xbill/models/Qwen3.5-2B-gguf/Qwen3.5-2B-bf16.gguf`, SHA-256 `ecbfdc76…`, llama.cpp `7182ef71e`).
Sister of `local-llamacpp-3050oem-qwen352b`; slot 3 is the only difference.

**The file does not fit on the card.** 3.62 GiB of weights on a 4 GiB card leaves no room for a compute
buffer, so it serves with the last 6 of 24 blocks on the CPU (`-ngl 99 -ot 'blk\.(18|19|20|21|22|23)\.=CPU'`),
the fewest that fits at the default micro-batch.

Runs:

- `benchmarks/runs/2026-10-16-first-light-1650ti/` — 20.22 tok/s decode with 6 blocks on the CPU (0.24x the
  Q4_0 file on this card), prefill 0.96x, 3207 MiB of 4096 at `-c 8192`.
- `benchmarks/runs/2026-10-16-parity-crosshost-1650ti/` — same label as the 3050 host's bf16 on all 65
  prompts, median probability gap 0.0027; 62 of 65 match fp32, as on the 3050.

- `benchmarks/runs/2026-10-16-quality-1650ti/` — suite 0.488, GSM8K 0.807, BFCL 0.762 with 6 blocks on the
  CPU, against 0.488 / 0.798 / 0.765 on the 3050 (no range clear of zero); 5.8–7.5 points above Q4_0 on this
  card. The comparison lives in the Q4_0 rig's run of the same name.
