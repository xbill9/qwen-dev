---
title: "Qwen and Gemma With Thinking On: Thinking Pays Qwen on Tool Calls, Gemma Still Leads"
published: false
description: "Step-by-step: turning thinking on for Qwen3.5-2B and Gemma 4 E2B under llama.cpp on a 4 GB GTX 1650 Ti, choosing sampling settings that keep Qwen's thinking from looping, and separating what thinking and sampling are each worth to both models on math and tool calling, and what thinking costs."
tags: gemma, qwen, llamacpp, cuda
cover_image: https://raw.githubusercontent.com/xbill9/qwen-dev/main/local-llamacpp-1650ti-qwen352b-q4_0/docs/qwen-vs-gemma-thinking/devto-cover.8dab7794.jpg
---

This article follows [Comparing Qwen and Gemma on the Low End](https://dev.to/gde/comparing-qwen-and-gemma-on-the-low-end-qwen-is-faster-without-drafting-gemma-has-better-quality-30bo), which ran Qwen3.5-2B and Gemma 4 E2B with thinking off. Here both models think, on the same card, the same llama.cpp binary and the same 4-bit files.

https://github.com/xbill9/qwen-dev/tree/main/local-llamacpp-1650ti-qwen352b-q4_0

**Thinking is worth 17.7 points to Qwen3.5-2B on BFCL tool calling and nothing measurable to Gemma 4 E2B. On GSM8K math, Qwen's 9.7-point rise is mostly its recommended sampling settings, which stop its long answers running away. With both models thinking, Gemma still leads by 10.3 points on math and 5.0 on tool calls. Qwen's thinking runs to a median of 7,607 tokens per math answer against Gemma's 808.**

---

#### What is this project trying to Do?

The first article found Gemma 4 E2B ahead of Qwen3.5-2B on every task, with thinking off for both. Qwen3.5 is built to think, so the first question a Qwen user asks is what happens with thinking on.

This article answers it on the GTX 1650 Ti, a 4 GB laptop card, for the two generation tasks: GSM8K math word problems and BFCL simple tool calls. Both models run as 4-bit GGUF files: Qwen's standard Q4_0 and Gemma's QAT Q4_0.

---

#### At This Point You Should Have…

- An NVIDIA GPU with a CUDA driver — here a GTX 1650 Ti (4 GB, Turing)
- llama.cpp built with CUDA — commit `7182ef71e` here
- The two GGUF files from the first article: `Qwen3.5-2B-Q4_0.gguf` and Gemma's `gemma-4-E2B-it-q4_0-exact.gguf`
- Python 3 for the task runner — the system `python3`, no virtualenv
- The repository cloned, and `jev-tpu-v5e1/` holding `gen_eval.py`

---

#### Step 1 — Turn Thinking On

One server flag turns thinking on for both models:

```shell
llama-server -m Qwen3.5-2B-Q4_0.gguf --host 127.0.0.1 --port 8080 -ngl 99 \
  -c 131072 --parallel 4 -ctk f16 -ctv f16 -fa 1 -t 6 -tb 12 --reasoning on --metrics
```

llama-server puts the thinking in `message.reasoning_content` and the answer in `message.content`. Only `content` is scored; the task runner keeps the thinking in each record for reading later.

`-c 131072 --parallel 4` gives each of 4 slots 32,768 tokens, so a request has room to think. The server log confirms the split:

```text
0.01.412.127 I srv    load_model: initializing, n_slots = 4, n_ctx_slot = 32768, kv_unified = 'false'
```

---

#### Step 2 — Use Each Vendor's Sampling Settings

The thinking-off runs used greedy decoding. With thinking on, greedy decoding breaks Qwen. On 16 GSM8K problems with an 8,192-token limit:

```text
trial-greedy-8k/gsm8k-qwen.jsonl: n=16 limit=8192 right=5 hit_limit=11 ends_in_loop=9 still_thinking=11
```

Qwen got 5 of 16 right. It hit the limit on 11, every one still thinking, and 9 of those ended in one sentence repeated dozens of times:

```text
    Wait, I need to check if the prompt requires the thinking steps to be in the final output. Yes.
    Okay.
    Wait, I need to check if the prompt requires the thinking steps to be in the final output. Yes.
    Okay.
```

Qwen's model card gives settings for thinking mode, and Gemma's `generation_config.json` gives its own:

```text
Qwen3.5-2B, thinking mode for text tasks: temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0
Gemma 4 E2B: "temperature": 1.0, "top_k": 64, "top_p": 0.95
```

Each model ran at its own vendor's settings. `gen_eval.py` merges them into every request from one variable:

```shell
JEV_SAMPLING='{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":1.5,"repeat_penalty":1.0}' \
JEV_GEN_MAX_TOKENS=32000 python3 gen_eval.py run http://127.0.0.1:8080 Qwen/Qwen3.5-2B gsm8k gsm8k-qwen.jsonl \
  --ids gsm8k-300-ids.txt --concurrency 4
```

---

#### Step 3 — Give Thinking Room

Qwen's card recommends an output length of 32,768 tokens for most queries. Both models got a 32,000-token limit on both tasks.

At that length Qwen's math takes hours on this card, so GSM8K is a fixed random 300 of its 1,319 problems, drawn once with a fixed seed and used for both models. BFCL is all 400. With thinking off, the 300 score Qwen at 0.730 and Gemma at 0.927, against 0.734 and 0.901 on all 1,319: the sample matches the full set for Qwen and favours Gemma by 2.6 points.

---

#### Step 4 — Score Both Models

Each model with thinking on is paired question by question against its own thinking-off records on the same 300 problems and 400 requests, with a 95% range from 10,000 resamples:

| GTX 1650 Ti | Qwen off | Qwen on | Gemma off | Gemma on |
| :--- | ---: | ---: | ---: | ---: |
| GSM8K, 300 | 0.730 | 0.827 | 0.927 | **0.930** 🥇 |
| BFCL simple, 400 | 0.688 | 0.838 | **0.917** 🥇 | 0.887 |

| Thinking on − thinking off | Change | 95% range |
| :--- | ---: | :--- |
| Qwen, GSM8K | **+0.097** | +0.043 to +0.150 |
| Qwen, BFCL | **+0.150** | +0.100 to +0.198 |
| Gemma, GSM8K | +0.003 | −0.023 to +0.030 |
| Gemma, BFCL | −0.030 | −0.058 to −0.005 |

| Gemma's lead over Qwen | GSM8K | BFCL |
| :--- | ---: | ---: |
| Both thinking off | 19.7 points | 23.0 points |
| Both thinking on | **10.3** (6.3 to 14.3) | **5.0** (1.3 to 9.0) |
| Gemma off, Qwen on | 10.0 | 8.0 |

- **Qwen rises 9.7 and 15.0 points**, both ranges clear of zero.
- **Gemma is level on math and loses 3.0 points on tool calls.** Gemma with thinking off still beats Qwen with thinking on.
- **On BFCL, Qwen now calls the tool.** Replies with no tool call fall from 88 of 400 to 8. Its misses are now mostly wrong argument values.

Step 4 changed two things at once: thinking, and greedy decoding to each vendor's sampling. Step 5 separates them.

---

#### Step 5 — Separate Thinking From Sampling

One more run holds the sampling and turns thinking off: the same server, files, limit, problems and settings as Step 4, with `--reasoning off`. Three runs per model, compared in pairs:

| GTX 1650 Ti | Greedy, thinking off | Vendor sampling, thinking off | Vendor sampling, thinking on |
| :--- | ---: | ---: | ---: |
| Qwen, GSM8K | 0.730 | 0.790 | 0.827 |
| Qwen, BFCL | 0.688 | 0.660 | 0.838 |
| Gemma, GSM8K | 0.927 | 0.910 | **0.930** 🥇 |
| Gemma, BFCL | **0.917** 🥇 | 0.885 | 0.887 |

| Change, 95% range | Sampling alone | Thinking alone |
| :--- | :--- | :--- |
| Qwen, GSM8K | +0.060 (+0.003 to +0.117) | +0.037 (−0.013 to +0.087) |
| Qwen, BFCL | −0.028 (−0.068 to +0.013) | **+0.177 (+0.128 to +0.228)** |
| Gemma, GSM8K | −0.017 (−0.047 to +0.010) | +0.020 (−0.010 to +0.050) |
| Gemma, BFCL | −0.033 (−0.055 to −0.013) | +0.003 (−0.025 to +0.030) |

- **On tool calls, thinking is the whole of Qwen's gain**: +17.7 points. With thinking off, Qwen still answers 83 of 400 in prose; with thinking on, 8.
- **On math, most of Qwen's gain is sampling.** Greedy, 26 of 300 replies ran past the 2,048-token limit; with the card's settings, including `presence_penalty` 1.5, 4 of 300 reached 32,000. Thinking adds 3.7 points on top, a range that includes zero.
- **Gemma's 3-point BFCL loss is the sampling**, at temperature 1.0. Thinking changes neither of Gemma's scores beyond its range.

At matched settings Gemma leads by 12.0 and 22.5 points with thinking off, and by 10.3 and 5.0 with thinking on.

---

#### Step 6 — Count What Thinking Costs

From the records and the task runner's own wall times, 4 requests at a time:

```text
qwen gsm8k: n=300 right=248 wall_s=36269.8 wall_min=604.5 wall_h=10.07 s_per_right=146.2 tokens_total=3407907 tokens_per_right=13742 tokens_median=7607
qwen bfcl: n=400 right=335 wall_s=1456.4 wall_min=24.3 wall_h=0.40 s_per_right=4.3 tokens_total=109946 tokens_per_right=328 tokens_median=160
gemma gsm8k: n=300 right=279 wall_s=1872.6 wall_min=31.2 wall_h=0.52 s_per_right=6.7 tokens_total=270970 tokens_per_right=971 tokens_median=808
gemma bfcl: n=400 right=355 wall_s=949.5 wall_min=15.8 wall_h=0.26 s_per_right=2.7 tokens_total=117153 tokens_per_right=330 tokens_median=284
```

| GTX 1650 Ti, thinking on | Qwen | Gemma |
| :--- | ---: | ---: |
| GSM8K reply, median tokens | 7,607 | **808** 🥇 |
| GSM8K tokens per right answer | 13,742 | **971** 🥇 |
| GSM8K seconds per right answer | 146.2 | **6.7** 🥇 |
| GSM8K, 300 problems | 10.1 hours | **31 minutes** 🥇 |
| BFCL tokens per right answer | **328** 🥇 | 330 |

On math, Qwen spends about 14 times Gemma's tokens and 22 times its time for each right answer. On tool calls the two spend the same tokens.

31 of Qwen's 300 math replies used all 32,000 tokens; 20 of them were still thinking and 21 ended in repeated text. Scoring all 31 as right gives Qwen 0.913, still below Gemma's 0.930.

---

#### 🔎 Tip: Do Not Run Qwen Greedy

Greedy decoding is the usual setting for repeatable scores. With thinking on it loops Qwen: 9 of 16 math problems ended in repeated text. With thinking off it costs Qwen 6.0 points on math, from runaway answers. Use the card's settings, and keep `presence_penalty` in them; it is the one aimed at repetition.

---

#### 🔎 Tip: The Token Limit Covers the Thinking

`max_tokens` counts thinking tokens as well as the answer. A limit sized for thinking-off answers (512 for tool calls, 2,048 for math here) cuts a thinking model off mid-thought, with an empty `content`. Size the limit, and each slot's share of `-c`, for the thinking.

---

#### Compare and Contrast

| GTX 1650 Ti, thinking on | Qwen3.5-2B Q4_0 | Gemma 4 E2B QAT Q4_0 |
| :--- | :--- | :--- |
| GSM8K / BFCL | 0.827 / 0.838 | **0.930 / 0.887** 🥇 |
| Gain from thinking alone | **+3.7 / +17.7** 🥇 | +2.0 / +0.3 |
| Gain from sampling alone | **+6.0 / −2.8** 🥇 | −1.7 / −3.3 |
| GSM8K median reply | 7,607 tokens | **808 tokens** 🥇 |
| GSM8K, 300 problems | 10.1 hours | **31 minutes** 🥇 |
| Greedy thinking | loops | **runs** 🥇 |

---

#### So, Which One?

Gemma 4 E2B, with thinking off and greedy decoding. It scores higher than Qwen with thinking on, on both tasks, and answers math in a fraction of the time.

If you use Qwen, use its card's sampling settings, and turn thinking on for tool calls: it is worth 17.7 points there for few tokens. On math, thinking adds 3.7 points that may not be real, at 14 times Gemma's tokens per right answer. Leave Gemma's thinking off for these tasks; it gains nothing.

---

#### Summary

The goal of this article was to measure Qwen3.5-2B and Gemma 4 E2B with thinking on, where the first article ran them with thinking off. The key to the solution was running each model at its vendor's thinking-mode sampling with room to think, and a third run with the same sampling and thinking off, so that every answer pairs against the same question under each change alone. The results were:

- 🟢 **Thinking lifts Qwen on tool calls**: +17.7 points on BFCL, with replies that skip the tool down from 83 to 8.
- ⚠️ **On math, most of Qwen's rise is sampling**: +6.0 from the card's settings, +3.7 from thinking, a range that includes zero.
- ❌ **Thinking does not lift Gemma**: +2.0 on GSM8K and +0.3 on BFCL, neither clear of zero.
- 🟢 **Gemma still leads**: by 10.3 and 5.0 points with both thinking, and by 10.0 and 8.0 with only Qwen thinking.
- ⚠️ **Qwen's math thinking is expensive**: 13,742 tokens and 146 seconds per right answer, against Gemma's 971 and 6.7.
- ⚠️ **Greedy decoding loops Qwen's thinking**; its card's sampling settings fix it.

Scope: one GTX 1650 Ti (4 GB), llama.cpp `7182ef71e`, the same Q4_0 files as the first article, measured 2026-10-18 and 2026-10-19. Two tasks: GSM8K on a fixed random 300 of 1,319 problems and BFCL simple on all 400; the classification suite was not run with thinking on. The two sampled runs used each vendor's thinking-mode sampling and a 32,000-token limit, 4 requests at a time, once per model; the greedy thinking-off records are from the first article's runs, with 2,048 and 512-token limits, 8 at a time. Thinking with greedy decoding was run on 16 problems only. Sampled runs give different text on a re-run. The scoring code and the card were set up for this repository's Gemma work.

The strategy for measuring thinking on Qwen3.5-2B and Gemma 4 E2B on a local GPU was validated with an incremental step by step approach.

---

#### References

* [Comparing Qwen and Gemma on the Low End | dev.to](https://dev.to/gde/comparing-qwen-and-gemma-on-the-low-end-qwen-is-faster-without-drafting-gemma-has-better-quality-30bo)
* [local-llamacpp-1650ti-qwen352b-q4_0 | GitHub](https://github.com/xbill9/qwen-dev/tree/main/local-llamacpp-1650ti-qwen352b-q4_0)
* [Qwen/Qwen3.5-2B | Hugging Face](https://huggingface.co/Qwen/Qwen3.5-2B)
* [google/gemma-4-E2B-it | Hugging Face](https://huggingface.co/google/gemma-4-E2B-it)
* [xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf | Hugging Face](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf)
* [llama.cpp | GitHub](https://github.com/ggml-org/llama.cpp)
