# 2026-10-17 — does Qwen3.5-2B's MTP head change its answers? GTX 1650 Ti

**Answer: no.** With its MTP head drafting 1 token, Qwen Q4_0 scores GSM8K 0.748 and BFCL 0.698 against
0.734 and 0.688 without drafting, both differences inside their ranges. With 8 requests at a time drafting
is slower: each task took 1.08–1.09x as long.

## Setup

`run_quality.sh`: the Q4_0 arm of `../2026-10-16-quality-1650ti` (same file `bd99d5ae…`, llama.cpp `7182ef71e`,
`-c 32768 --parallel 8 -fa 1`, f16 KV, `-t 6 -tb 12`, `--reasoning off`) plus `--spec-type draft-mtp
--spec-draft-n-max 1`, the best one-request setting on this card (`../2026-10-17-spec-mtp-vs-gemma-1650ti`).
GSM8K (1,319, 2,048-token limit) and BFCL simple (400), 8 requests at a time. The suite is left out: it reads
one token's probabilities, which drafting does not produce. Server 2122 MiB (1806 without the drafter).
`compare.py` pairs the two arms record for record and reads the drafting counters from `/metrics`
(`COMPARE.md`, `compare.json`). No errors.

## Results

| task | no draft | MTP, 1 token | difference (95% range) | right only without / with | same reply text |
| --- | ---: | ---: | --- | ---: | ---: |
| GSM8K | 0.734 | 0.748 | +0.014 (−0.007 to +0.036) | 93 / 112 | 68 of 1,319 |
| BFCL simple | 0.688 | 0.698 | +0.010 (−0.007 to +0.028) | 5 / 9 | 303 of 400 |

| task | arm | wall time | reply tokens | tok/s over 8 slots | at the length limit |
| --- | --- | ---: | ---: | ---: | ---: |
| GSM8K | no draft | 64.8 min | 706,288 | 181.7 | 107 |
| GSM8K | MTP, 1 token | 70.2 min | 684,576 | 162.5 | 94 |
| BFCL | no draft | 9.3 min | 44,127 | 78.8 | 12 |
| BFCL | MTP, 1 token | 10.2 min | 43,669 | 71.6 | 11 |

- **Scores are level**; both point estimates lean toward the drafted arm, and neither range excludes zero.
- **Most reply texts differ**: 68 of 1,319 GSM8K replies and 303 of 400 BFCL replies are byte-identical. At
  one request at a time the drafted greedy text matched exactly; how often two no-draft runs at 8 slots
  differ from each other was not measured, so this run does not separate drafting from batching.
- **The drafts are accepted 90.1% of the time** (344,756 of 382,574) and still cost throughput: 0.89x the
  tokens per second at GSM8K. On this card Gemma E2B's drafter likewise paid only at 1–2 requests at once
  (`../../../../local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-15-drafter-load-1650ti`).

## Not measured

Two no-draft repeats at 8 slots (the text-identity baseline), 2 or 4 requests at a time, the bf16 file with
drafting.
