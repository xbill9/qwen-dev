# 2026-10-16 — does this host's Q4_0 build compute like the 3050 host's?

**Answer: closely.** The two hosts built Qwen3.5-2B Q4_0 from the same revision (`15852e8c`) on the same
llama.cpp commit (`7182ef71e`) and got files of identical size with different SHA-256s: `bd99d5ae…` here,
`1c27608b…` on the 3050 host. The bf16 GGUFs differ as well (`ecbfdc76…` here, `1526c207…` there), so the
difference enters at `convert_hf_to_gguf.py`. The 3050 host was unreachable, so which bytes differ is untested.

## Method

`logit_check.py` is the 3050 parity run's harness, copied unchanged from
`local-llamacpp-3050oem-qwen352b/benchmarks/runs/2026-10-16-parity-transformers-3050oem/` with its cached
transformers fp32 reference (`reference-fp32.jsonl`). It builds the suite harness's exact prompts for the
first 5 records of each of 13 subsets and reads the top-32 next-token log-probabilities from llama-server on
this card, serving this host's Q4_0 at the rig's flags (`--test llama --tag q4_0-1650ti`). `crosshost.py`
compares each prompt against the 3050 host's Q4_0 results (`logit_check-q4_0-3050oem.jsonl`, copied from the
same run), keyed on subset and id, and writes `crosshost.json`.

## Results

| | prompts | same chosen label | label-probability difference, median / max |
|---|---:|---:|---|
| this host's Q4_0 against the 3050 host's Q4_0 | 65 | 62 | 0.018 / 0.075 |
| this host's Q4_0 against transformers fp32 | 65 | 43 | 0.115 / 0.510 |
| 3050 host's Q4_0 against transformers fp32 (that run) | 65 | 43 | 0.114 / 0.488 |

- Every prompt tokenizes to the same length on both hosts.
- The cross-host difference is about a sixth of the Q4_0 rounding's: both builds agree with the fp32 reference
  on 43 of 65 prompts.
- Two things differ between the arms, the file and the GPU (sm_75 without tensor cores against sm_86 with
  them), and this run does not separate them.
