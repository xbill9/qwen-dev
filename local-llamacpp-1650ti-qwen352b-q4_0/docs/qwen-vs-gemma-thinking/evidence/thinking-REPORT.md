# 2026-10-18 — thinking on, Qwen3.5-2B Q4_0 against Gemma 4 E2B QAT Q4_0, GTX 1650 Ti

**Answer: thinking on helps Qwen a lot and Gemma not at all, and Gemma still leads on both tasks.** With
thinking on, Qwen gains 9.7 points on GSM8K and 15.0 on BFCL. Gemma is level on GSM8K (+0.3) and loses 3.0
on BFCL. The gap shrinks from 19.7 to 10.3 points on GSM8K and from 23.0 to 5.0 on BFCL, every range
still clear of zero. Gemma with thinking off still leads Qwen with thinking on by 10.0 and 8.0. Qwen's
GSM8K replies are 9.4x Gemma's in median length with thinking on, and its GSM8K took 10.1 hours against 31
minutes.

## Setup

Same card, llama.cpp binary (`7182ef7`, sm_75) and GGUFs as the thinking-off runs it pairs with:
`../2026-10-16-quality-1650ti` (Qwen, `bd99d5ae…`) and
`local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-17-quality-vs-qwen-1650ti` (Gemma, `419db9a6…`).
`run_thinking.sh` changed four things:

- `--reasoning on`. Thinking goes to `reasoning_content` and is kept in each record as `reasoning`; only
  `content` is scored.
- **Each vendor's thinking-mode sampling**, in `sampling.json` beside each model's records. Qwen's model
  card: temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 1.5, repetition_penalty 1.0.
  Gemma's `generation_config.json`: temperature 1.0, top_p 0.95, top_k 64. The thinking-off runs were
  greedy, so each "on vs off" pair changes thinking and sampling together. Greedy thinking was tried first
  on 16 GSM8K problems with an 8,192-token limit: Qwen hit the limit on 11, and 9 of those ended in one
  sentence repeated dozens of times.
- **A 32,000-token limit** on both tasks (Qwen's card recommends 32,768 for most queries), with
  `-c 131072 --parallel 4` so each slot holds 32,768 tokens, 4 requests at a time.
- **GSM8K is a fixed random 300 of the 1,319 problems** (`gsm8k-300-ids.txt`, `random.Random(20261018)`),
  for time. BFCL is all 400. The thinking-off records are cut to the same 300 ids for pairing.

Sampled, so a re-run gives different text. Correctness is `gen_eval.py`'s own `right` field; pairs and
95% ranges are `gen_compare.compare` (10,000 resamples), run by `compare.py`. Full tables in `COMPARE.md`,
numbers in `compare.json`. No errors.

## Results

| task | Qwen off | Qwen on | Gemma off | Gemma on |
| --- | ---: | ---: | ---: | ---: |
| GSM8K, 300 | 0.730 | 0.827 | 0.927 | 0.930 |
| BFCL simple, 400 | 0.688 | 0.838 | 0.917 | 0.887 |

| pair | difference | 95% range |
| --- | ---: | --- |
| GSM8K: Qwen on − Qwen off | +0.097 | +0.043 to +0.150 |
| GSM8K: Gemma on − Gemma off | +0.003 | −0.023 to +0.030 |
| GSM8K: Qwen − Gemma, thinking off | −0.197 | −0.247 to −0.147 |
| GSM8K: Qwen − Gemma, thinking on | −0.103 | −0.143 to −0.063 |
| GSM8K: Qwen on − Gemma off | −0.100 | −0.143 to −0.060 |
| BFCL: Qwen on − Qwen off | +0.150 | +0.100 to +0.198 |
| BFCL: Gemma on − Gemma off | −0.030 | −0.058 to −0.005 |
| BFCL: Qwen − Gemma, thinking off | −0.230 | −0.280 to −0.180 |
| BFCL: Qwen − Gemma, thinking on | −0.050 | −0.090 to −0.013 |
| BFCL: Qwen on − Gemma off | −0.080 | −0.117 to −0.043 |

On the full 1,319 the thinking-off scores were Qwen 0.734 and Gemma 0.901. On these 300 they are 0.730 and
0.927, so the sample matches the full set for Qwen and is 2.6 points kind to Gemma.

## What changed with thinking on

- **Qwen's GSM8K replies run long.** Median 7,607 tokens, mean 11,360, against Gemma's 808 and 903. 31 of
  300 hit the 32,000-token limit (5 still right); 20 of those were still thinking at the limit, and 21 end
  in the same 200 characters repeated five or more times. Counting all 31 right gives 0.913, still below
  Gemma's 0.930.
- **Qwen calls the tool.** Replies with no tool call fall from 88 to 8 of 400. Its remaining misses are
  mostly wrong argument values (53).
- **Gemma loses a little on BFCL** (28 records flip: 20 right only with thinking off, 8 only with it on),
  mostly to wrong argument values (35 against 24).

## Run time on this card

| | BFCL 400 | GSM8K 300 |
| --- | ---: | ---: |
| Gemma, thinking on | 15.8 min | 31.2 min |
| Qwen, thinking on | 24.3 min | 10.1 h |

## Not measured

Thinking on with greedy decoding at full size, or thinking off with each vendor's sampling, so the share of
each change is not separated; the suite with thinking on; the other 1,019 GSM8K problems; the RTX 3050.
