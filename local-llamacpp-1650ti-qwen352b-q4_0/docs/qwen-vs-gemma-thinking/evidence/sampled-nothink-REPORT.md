# 2026-10-19 — thinking off at vendor sampling: splitting sampling from thinking, GTX 1650 Ti

**Answer: on tool calls Qwen's gain is thinking; on math most of it is sampling.** The 2026-10-18 run turned
thinking on and switched from greedy decoding to each vendor's thinking-mode sampling in one step. This run
holds the sampling and turns thinking off, so it splits the two:

| Qwen3.5-2B Q4_0 | GSM8K, 300 | 95% range | BFCL, 400 | 95% range |
| --- | ---: | --- | ---: | --- |
| Sampling (greedy off → sampled off) | +0.060 | +0.003 to +0.117 | −0.028 | −0.068 to +0.013 |
| Thinking (sampled off → sampled on) | +0.037 | −0.013 to +0.087 | **+0.177** | +0.128 to +0.228 |
| Both | +0.097 | +0.043 to +0.150 | +0.150 | +0.100 to +0.198 |

| Gemma 4 E2B QAT Q4_0 | GSM8K | 95% range | BFCL | 95% range |
| --- | ---: | --- | ---: | --- |
| Sampling | −0.017 | −0.047 to +0.010 | **−0.033** | −0.055 to −0.013 |
| Thinking | +0.020 | −0.010 to +0.050 | +0.003 | −0.025 to +0.030 |
| Both | +0.003 | −0.023 to +0.030 | −0.030 | −0.058 to −0.005 |

Gemma's lead at matched settings: thinking off at vendor sampling 12.0 points on GSM8K (7.3 to 16.7) and 22.5
on BFCL (17.7 to 27.5); thinking on 10.3 and 5.0.

## Setup

`run_sampled_nothink.sh`: the 2026-10-18 run's script (`../2026-10-18-quality-thinking-1650ti/run_thinking.sh`)
with `--reasoning off` and nothing else changed. Same card, llama.cpp `7182ef7`, GGUFs (`model-sha256.txt`),
`-c 131072 --parallel 4`, 4 requests at a time, 32,000-token limit, the same sampling (`sampling.json`), the
same 300 GSM8K ids and all 400 BFCL. Gemma's records are in
`local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-19-quality-sampled-nothink-1650ti`. Pairs are
`gen_compare.compare` (10,000 resamples), run by `compare.py`; tables in `COMPARE.md`, numbers in
`compare.json`. No errors. One sampled run per arm.

## What the sampling changes

- **Qwen's math replies stop running away.** Greedy with thinking off, 26 of 300 GSM8K replies hit the
  2,048-token limit; at vendor sampling (`presence_penalty` 1.5) 4 of 300 hit the 32,000 limit, with a median
  reply of 430 tokens against 365. That is where the +6.0 comes from; the range only just clears zero.
- **Qwen still answers tool calls in prose with thinking off**: 83 of 400 BFCL replies make no call at vendor
  sampling, against 88 greedy and 8 with thinking on.
- **Gemma loses 3.3 points on BFCL to sampling at temperature 1.0**, and thinking does not win them back.

## Run time

| | BFCL 400 | GSM8K 300 |
| --- | ---: | ---: |
| Gemma | 5.6 min | 15.1 min |
| Qwen | 32.2 min | 2.29 h |

## Not measured

Thinking on with greedy decoding beyond the 16-problem trial; other sampling settings; repeat runs, so the
spread of one sampled run is not measured.
