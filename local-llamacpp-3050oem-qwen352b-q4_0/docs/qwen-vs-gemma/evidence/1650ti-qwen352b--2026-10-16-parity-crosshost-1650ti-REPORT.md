# 2026-10-16 — does this host's bf16 build compute like the 3050 host's?

**Answer: yes.** On 65 suite prompts, the bf16 GGUF on this card picks the same label as the 3050 host's bf16
on all 65, and matches the transformers fp32 reference on 62, as the 3050's does.

The two files come from the same revision (`15852e8c`) and llama.cpp commit (`7182ef71e`) and have the same
size, and their SHA-256s differ: `ecbfdc76…` here, `1526c207…` on the 3050 host. Which bytes differ is
untested.

## Method

`logit_check.py` is the 3050 parity run's harness, copied unchanged from
`local-llamacpp-3050oem-qwen352b/benchmarks/runs/2026-10-16-parity-transformers-3050oem/` with its cached
fp32 reference (`reference-fp32.jsonl`) and its transformers bf16 results (`logit_check-hf-bf16.jsonl`). It
reads the top-32 next-token log-probabilities for the suite harness's exact prompts, 5 records from each of
13 subsets, from llama-server on this card serving the bf16 file with 6 of 24 blocks on the CPU (the
first-light serving split). `crosshost.py` pairs each prompt with the 3050 host's bf16 result
(`logit_check-bf16-3050oem.jsonl`), keyed on subset and id, and writes `crosshost.json`.

## Results

| | prompts | same chosen label | label-probability difference, median / max |
|---|---:|---:|---|
| this host's bf16 against the 3050 host's bf16 | 65 | 65 | 0.0027 / 0.026 |
| this host's bf16 against transformers fp32 | 65 | 62 | 0.0010 / 0.558 |
| 3050 host's bf16 against transformers fp32 (that run) | 65 | 62 | 0.0027 / 0.562 |

- Every prompt tokenizes to the same length on both hosts.
- The two hosts agree on every label, and their largest probability gap is 0.026, against 0.075 between
  the two hosts' Q4_0 files (`../../../local-llamacpp-1650ti-qwen352b-q4_0/benchmarks/runs/2026-10-16-parity-crosshost-1650ti/`).
- The file, the card and the CPU split all differ between the arms, and this run does not separate them.
