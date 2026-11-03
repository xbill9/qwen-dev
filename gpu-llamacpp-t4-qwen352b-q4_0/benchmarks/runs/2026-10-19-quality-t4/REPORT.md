# 2026-10-19 — answer quality on the Tesla T4, Qwen3.5-2B Q4_0 against Gemma 4 E2B, with and without its drafter

**Question:** on the T4, does the Qwen–Gemma quality gap measured on the GTX 1650 Ti hold, and does Gemma's
QAT drafter change any answer score?

**Answer: the gap holds and the drafter changes no score.** Gemma E2B leads Qwen Q4_0 by 26.0 suite points,
15.8 on GSM8K and 22.5 on BFCL, each range far from zero and each within 1 point of the 1650 Ti's gap. With 1
drafted token Gemma scores level on all three tasks and finishes GSM8K 1.42x faster.

## Setup

`run_quality.sh`, the 1650 Ti protocol with the card changed. One llama.cpp binary, `7182ef71e` built for sm_75,
`-c 32768 --parallel 8 -fa 1`, f16 KV, `--reasoning off`, `-t 2 -tb 2` for this 2-vCPU VM, and `-cram 2048`
because llama-server's default 8,192 MiB host prompt cache is more than the VM's 7.3 GiB of RAM. Three arms,
run in sequence on one afternoon:

| arm | file | server VRAM |
| --- | --- | ---: |
| Qwen Q4_0 | Qwen3.5-2B Q4_0 built on this host, `d2bc253a…` | 1856 MiB |
| Gemma E2B | exact QAT Q4_0, `419db9a6…`, `per_layer_token_embd` on the card | 3020 MiB |
| Gemma E2B + drafter | the same, with Google's QAT assistant drafter as exact Q4_0 (`060f5a01…`), 1 drafted token | 3088 MiB |

The drafted arm takes its draft length from Gemma's fastest greedy arm in `../2026-10-19-spec-mtp-vs-gemma-t4`.
Three tasks, all at 8 requests at a time: the Nimble 13-subset suite (3,880 records, through `vllm_shim.py`,
`--tokenizer Qwen/Qwen3.5-2B` for Qwen), GSM8K (1,319, 2,048-token limit) and BFCL simple (400). `compare.py`
pairs every arm on the same records and adds the 1650 Ti runs
(`../../../../local-llamacpp-1650ti-qwen352b-q4_0/benchmarks/runs/2026-10-16-quality-1650ti`,
`../../../../local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-17-quality-vs-qwen-1650ti`); full tables in
`COMPARE.md`, numbers in `compare.json`. No errors in any task.

## Results

| task | Gemma E2B | Gemma E2B + drafter | Qwen Q4_0 |
| --- | ---: | ---: | ---: |
| Suite (3,880) | **0.681** | **0.682** | 0.422 |
| GSM8K (1,319) | **0.892** | **0.895** | 0.735 |
| BFCL simple (400) | **0.922** | **0.922** | 0.698 |

Paired differences, 95% ranges:

| | suite | GSM8K | BFCL |
| --- | --- | --- | --- |
| Qwen − Gemma, T4 | −0.260 (−0.280 to −0.239) | −0.158 (−0.182 to −0.134) | −0.225 (−0.275 to −0.175) |
| Gemma drafter − none, T4 | +0.000 (−0.003 to +0.004) | +0.002 (−0.008 to +0.012) | +0.000 (+0.000 to +0.000) |
| Gemma: T4 − 1650 Ti | +0.001 (−0.003 to +0.005) | −0.008 (−0.020 to +0.003) | +0.005 (−0.005 to +0.015) |
| Qwen Q4_0: T4 − 1650 Ti | −0.008 (−0.014 to −0.002) | +0.001 (−0.020 to +0.021) | +0.010 (−0.013 to +0.033) |

| Qwen − Gemma gap | T4 | 1650 Ti | moved |
| --- | ---: | ---: | ---: |
| suite | −0.260 | −0.251 | −0.009 |
| GSM8K | −0.158 | −0.167 | +0.009 |
| BFCL | −0.225 | −0.230 | +0.005 |

- **The gap is the same on both cards**, within 1 point on every task.
- **Gemma scores on the T4 what it scored on the 1650 Ti**, every range across zero.
- **Qwen's suite score is 0.8 points lower here, the only cross-card change clear of zero.** The two hosts
  built Q4_0 files that hash differently (`d2bc253a…` here, `bd99d5ae…` on the 1650 Ti) from bf16 GGUFs that
  also differ; the 3050 host's file scored 0.422, the same as this one. Which bytes differ is untested.
- **Qwen leads on two subsets, boolq (+0.087) and squad2 (+0.047)**, and trails most on summeval-consistency
  (−0.674), civil_comments (−0.640) and pubmedqa (−0.396) (per-subset table in `COMPARE.md`; ranges not
  corrected for 13 comparisons).
- **Answer length:** 100 Qwen GSM8K replies hit 2,048 tokens, against 3 for Gemma. Qwen wrote 702,741 tokens
  of GSM8K replies against Gemma's 527,255.
- **Tool calls:** 82 Qwen BFCL replies made no tool call, against 6 for Gemma; that is 82 of Qwen's 121 wrong
  answers.
- All labels returned on the suite: Gemma 96.2%, Qwen 90.3%.

## The drafter

- **Scores level on every task.** On BFCL the two Gemma arms give the same right/wrong on all 400 calls and
  the same text on 398. On GSM8K 235 of 1,319 replies have the same text and 1,272 the same right/wrong:
  22 problems go wrong with the drafter and 25 go right. On the suite 3,829 of 3,880 records get the same
  right/wrong.
- **86.5% of drafted tokens accepted** (250,198 of 289,162, the server's own counters).
- **Run time, drafter against none:** GSM8K 23.0 min against 32.6 (1.42x faster), BFCL 1.4 against 1.5 min,
  suite 8.2 against 7.6 min (1.08x as long). The suite reads one label per record, which leaves the drafter little to draft (likely cause, not profiled).

## Run time on this card

| | suite | GSM8K | BFCL |
| --- | ---: | ---: | ---: |
| Qwen Q4_0 | 8.9 min | 42.3 min | 3.7 min |
| Gemma E2B | 7.6 min | 32.6 min | 1.5 min |
| Gemma E2B + drafter | 8.2 min | 23.0 min | 1.4 min |

Qwen takes 1.17x Gemma's time on the suite, 1.30x on GSM8K and 2.43x on BFCL, where it writes 43,678 reply
tokens against Gemma's 12,759. Gemma's GSM8K took 52.1 min on the 1650 Ti. `driver.log` reads 585–1590 MHz at
the step boundaries.

## Not measured

Qwen's own MTP head as a drafter at 8 requests (measured on the 1650 Ti:
`../../../../local-llamacpp-1650ti-qwen352b-q4_0/benchmarks/runs/2026-10-17-quality-mtp-1650ti`), Qwen bf16 on
this card, thinking on, prompt variants.
