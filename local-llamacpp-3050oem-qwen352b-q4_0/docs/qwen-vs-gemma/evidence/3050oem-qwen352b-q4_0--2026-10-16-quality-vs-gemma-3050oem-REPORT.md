# 2026-10-16 — answer quality, Qwen3.5-2B Q4_0 against Gemma 4 E2B and E4B, RTX 3050 OEM

**Answer: Gemma E2B is better on all three tasks, by wide margins.** Qwen3.5-2B scores 26.0 points lower
on the suite, 15.5 lower on GSM8K and 21.5 lower on BFCL, every range clear of zero. E4B leads Qwen by more.

## Setup

Qwen3.5-2B standard Q4_0 (`1c27608b…`, built in `../2026-10-16-first-light-3050oem`) on llama.cpp `7182ef71e`,
one llama-server with the flags of the Gemma quality run less its `-ot` (Gemma's per-layer table has no
counterpart): `-ngl 99 -c 32768 --parallel 8 -ctk f16 -ctv f16 -fa 1 -t 6 -tb 12 --reasoning off`,
**no drafter**. `run_quality.sh` ran the same three tasks with the same harness code and settings as
`local-llamacpp-3050oem-4b-q4_0/benchmarks/runs/2026-10-13-quality-e2b-vs-e4b-3050oem`:

- **Nimble 13-subset suite**, 3,880 records, `nimble_suite/run_suite.py` through `vllm_shim.py`, `JEV_TOPK=32`,
  8 at a time. `--tokenizer Qwen/Qwen3.5-2B`, so the answer lead and label ids are Qwen tokens.
- **GSM8K**, 1,319 problems, `gen_eval.py`, 2,048-token limit, greedy, 8 at a time.
- **BFCL simple**, 400 tool-calling records, `gen_eval.py`, greedy, 8 at a time.

The Gemma records are that run's, from 2026-10-13 on llama.cpp `77c4540ce`; they were not re-run. Gemma's
accuracy runs on this card scored level across binaries and settings (`local-llamacpp-3050oem-2b-q4_0`,
`2026-10-14-accuracy-*`). Pairing and ranges are `quant_compare.paired` and `gen_compare.compare`, run by
`compare.py`; full tables in `COMPARE.md`, numbers in `compare.json`. No errors in any task.

## Results

| task | Gemma E2B | Gemma E4B | Qwen3.5-2B | Qwen − E2B | 95% range |
| --- | ---: | ---: | ---: | ---: | --- |
| Suite, 13 classification subsets (3,880) | 0.683 | 0.730 | 0.422 | **−0.260** | −0.281 to −0.240 |
| GSM8K, math word problems (1,319) | 0.898 | 0.933 | 0.743 | **−0.155** | −0.177 to −0.132 |
| BFCL simple, tool calling (400) | 0.925 | 0.910 | 0.710 | **−0.215** | −0.265 to −0.165 |

Records Gemma E2B gets right and Qwen wrong, against the other way: suite 1,432 against 422, GSM8K 239 against
35, BFCL 102 against 16.

**Per suite subset**, Qwen is ahead only on boolq (+0.083) and squad2 (+0.054, range includes zero). The
largest gaps are summeval-consistency −0.667, civil_comments −0.637, pubmedqa −0.412, vitaminc-dev −0.382,
multinli −0.375 and massive-de-DE −0.346.

## What the low scores are made of

The prompts are right: the rebuilt Qwen prompt ends in Qwen's own chat template with thinking closed
(`<|im_start|>assistant\n<think>\n\n</think>\n\ndecision:`), and the label tokens are the ones Qwen
puts its probability on for the yes/no and score subsets (label mass 0.94–0.999).

- **Option letters.** The four letter-choice subsets ask for a letter (`A`, `B`, `C`) and Qwen often writes
  the option's name instead: on one vitaminc record its top token is ` SUPPORT` (−0.79) ahead of ` B` and
  ` C`. Its label mass there is 0.41–0.63 (vitaminc 0.413, multinli 0.496, pubmedqa 0.497, massive
  0.58–0.63) against Gemma's 0.98–0.99, and the scorer picks among the letters that remain. multinli comes
  out `neutral` on 298 of 299 records.
- **One-sided yes/no and score answers.** civil_comments: Qwen calls 248 of 300 comments toxic, against 32
  in the gold labels (Gemma 29). paws: "not a paraphrase" 242 of 250 (gold 129). summeval-consistency:
  Qwen never picks the top level, which is the gold level in 121 of 144.
- **GSM8K runs long.** 100 of Qwen's replies hit the 2,048-token limit (9 of them still right), against 5
  for Gemma E2B. With thinking off it still deliberates in the answer ("Wait, is it possible the 15 cups
  and 25 cups are the amounts *per chicken*?"). Median reply 369 tokens.
- **BFCL: it answers instead of calling the tool.** 79 of its 116 wrong answers made no tool call (66
  finished, 13 ran out of the 512 tokens), mostly by solving the problem in prose. Gemma E2B made no call
  6 times.

## Reading it

These numbers mix two things that this run does not separate. One is the model. The other is the
**rounding**: Gemma's file sits on the 4-bit grid Google trained it for, and Qwen's is plain Q4_0 from
bf16, with no QAT available. A bf16 or Q8_0 arm of Qwen on the same harness would show how much of the gap
the rounding accounts for. The format-following failures (letters, tool calls) look like model behaviour
under these prompts rather than rounding, but that is a reading, not a measurement.

## Run time on this card

Suite 6.5 min, GSM8K 38.2 min, BFCL 3.3 min (Gemma E2B: 6.4, 29.3, 1.3). The GSM8K and BFCL times are
longer because Qwen's replies are.

## Not measured

Qwen at bf16 or Q8_0; Qwen with thinking on; any prompt changes, such as accepting the option name as an
answer.
