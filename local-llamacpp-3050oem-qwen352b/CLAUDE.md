# CLAUDE.md — local-llamacpp-3050oem-qwen352b

**An artifact rig: measurements only.** It holds results for `Qwen/Qwen3.5-2B` at bf16, the reference
build (so no slot-5 encoding), served by llama.cpp on the RTX 3050 OEM. No `server.py`, no MCP server,
no `tpu.env`, and none is owed (`NAMING.md`, artifact rigs). The serving rig is
`local-llamacpp-3050oem-qwen352b-q4_0`; its `make quantize` writes the bf16 GGUF this rig's runs used
(`/home/xbill/models/Qwen3.5-2B-gguf/Qwen3.5-2B-bf16.gguf`, SHA-256 `1526c207…`, llama.cpp `7182ef71e`).

Runs:

- `benchmarks/runs/2026-10-16-quality-bf16-3050oem/` — suite 0.488, GSM8K 0.798, BFCL 0.765; Q4_0 costs
  5.5–6.5 points per task, and Gemma E2B Q4 still leads by 10–19.5.
