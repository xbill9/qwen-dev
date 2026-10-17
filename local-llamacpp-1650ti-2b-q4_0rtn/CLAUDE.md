# CLAUDE.md — local-llamacpp-1650ti-2b-q4_0rtn

**An artifact rig: measurements only.** It holds results for Gemma 4 E2B rounded to Q4_0 from Google's bf16
release (`google/gemma-4-E2B-it`, snapshot `3e22461f`), with no QAT and no imatrix, served by llama.cpp on the
GTX 1650 Ti in `yogagpu`. No `server.py`, no MCP server, no `tpu.env`, and none is owed (`NAMING.md`, artifact
rigs). The serving rig for this card and model is `local-llamacpp-1650ti-2b-q4_0`, which serves Google's QAT
weights stored exact Q4_0.

**Slot 5 is `q4_0rtn`**, on the `w8a8rtn` precedent: the same GGUF block format as `q4_0`, but its values are
rounded from the bf16 release instead of taken from the QAT grid (`NAMING.md`, slot 5's fifth rule). It is
built the way `local-llamacpp-1650ti-qwen352b-q4_0` builds Qwen3.5-2B, so the two are rounded alike.

**The file is not all Q4_0.** `llama-quantize Q4_0` keeps `token_embd` and `per_layer_token_embd` at Q6_K and
`per_layer_model_proj` at bf16: 3.344 GB, 67.5% of it Q6_K and 31.7% Q4_0, against the QAT file's 2.605 GB, all
Q4_0. The weight matrices are Q4_0 in both. Built on this host by llama.cpp `7182ef71e`:
`/home/xbill/models/gemma-4-E2B-it-gguf/gemma-4-E2B-it-Q4_0.gguf`, SHA-256 `c96dec27…` (bf16 GGUF beside it).

Runs:

- `benchmarks/runs/2026-10-17-quality-rtn-1650ti/` — suite 0.632, GSM8K 0.881, BFCL 0.895: 4.8 / 2.0 / 2.2
  points below the QAT file, so QAT is 10–19% of Gemma's lead over Qwen3.5-2B Q4_0 on this card. Rounded Gemma
  still leads Qwen Q4_0 by 20.2 / 14.7 / 20.7 and Qwen bf16 by 14.4 / 7.4 / 13.3.
