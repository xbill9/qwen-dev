---
title: "Qwen and Gemma With Thinking On: Qwen Narrows the Gap, Gemma Still Leads"
published: false
description: "Step-by-step: turning thinking on for Qwen3.5-2B and Gemma 4 E2B under llama.cpp on a 4 GB GTX 1650 Ti, choosing sampling settings that keep Qwen's thinking from looping, and measuring what thinking is worth to each model on math and tool calling, and what it costs."
tags: gemma, qwen, llamacpp, cuda
cover_image: COVER
---

This article follows [Comparing Qwen and Gemma on the Low End](https://dev.to/gde/comparing-qwen-and-gemma-on-the-low-end-qwen-is-faster-without-drafting-gemma-has-better-quality-30bo), which ran Qwen3.5-2B and Gemma 4 E2B with thinking off. Here both models think, on the same card, the same llama.cpp binary and the same 4-bit files.

https://github.com/xbill9/qwen-dev/tree/main/local-llamacpp-1650ti-qwen352b-q4_0

**With thinking on, Qwen3.5-2B gains 9.7 points on GSM8K math and 15.0 on BFCL tool calling, and Gemma 4 E2B gains nothing. Gemma's lead shrinks from 19.7 to 10.3 points on math and from 23.0 to 5.0 on tool calling, and holds on both. Qwen's math answers run to a median of 7,607 tokens against Gemma's 808, and took 10.1 hours against 31 minutes.**

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

- **Thinking is worth 9.7 and 15.0 points to Qwen**, both ranges clear of zero.
- **Gemma gains nothing on math and loses 3.0 points on tool calls.** Gemma with thinking off still beats Qwen with thinking on.
- **On BFCL, Qwen now calls the tool.** Replies with no tool call fall from 88 of 400 to 8. Its misses are now mostly wrong argument values.

---

#### Step 5 — Separate Thinking From Sampling

SPLIT

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

#### 🔎 Tip: Do Not Run Qwen's Thinking Greedy

Greedy decoding is the usual setting for repeatable scores, and with thinking off it serves Qwen fine. With thinking on it loops: 9 of 16 math problems ended in repeated text. Use the card's thinking settings, and keep `presence_penalty` in them; it is the one aimed at repetition.

---

#### 🔎 Tip: The Token Limit Covers the Thinking

`max_tokens` counts thinking tokens as well as the answer. A limit sized for thinking-off answers (512 for tool calls, 2,048 for math here) cuts a thinking model off mid-thought, with an empty `content`. Size the limit, and each slot's share of `-c`, for the thinking.

---

#### Compare and Contrast

| GTX 1650 Ti, thinking on | Qwen3.5-2B Q4_0 | Gemma 4 E2B QAT Q4_0 |
| :--- | :--- | :--- |
| GSM8K / BFCL | 0.827 / 0.838 | **0.930 / 0.887** 🥇 |
| Gain from thinking | **+9.7 / +15.0** 🥇 | +0.3 / −3.0 |
| GSM8K median reply | 7,607 tokens | **808 tokens** 🥇 |
| GSM8K, 300 problems | 10.1 hours | **31 minutes** 🥇 |
| Greedy thinking | loops | **runs** 🥇 |

---

#### So, Which One?

Gemma 4 E2B, with thinking off. It scores higher than Qwen with thinking on, on both tasks, and answers math in a fraction of the time.

Turn thinking on for Qwen if you use Qwen: it is worth 10 to 15 points, and on tool calls it costs few tokens. On math it costs 14 times Gemma's tokens per right answer. Leave Gemma's thinking off for these tasks; it gains nothing.

---

#### Summary

The goal of this article was to measure Qwen3.5-2B and Gemma 4 E2B with thinking on, where the first article ran them with thinking off. The key to the solution was running each model at its vendor's thinking-mode sampling with room to think, and pairing every answer against the same question with thinking off. The results were:

- 🟢 **Thinking lifts Qwen**: +9.7 points on GSM8K and +15.0 on BFCL.
- ❌ **Thinking does not lift Gemma**: +0.3 on GSM8K, −3.0 on BFCL.
- 🟢 **Gemma still leads**: by 10.3 and 5.0 points with both thinking, and by 10.0 and 8.0 with only Qwen thinking.
- ⚠️ **Qwen's math thinking is expensive**: 13,742 tokens and 146 seconds per right answer, against Gemma's 971 and 6.7.
- ⚠️ **Greedy decoding loops Qwen's thinking**; its card's sampling settings fix it.

Scope: one GTX 1650 Ti (4 GB), llama.cpp `7182ef71e`, the same Q4_0 files as the first article, measured 2026-10-18 and 2026-10-19. Two tasks: GSM8K on a fixed random 300 of 1,319 problems and BFCL simple on all 400; the classification suite was not run with thinking on. Thinking-on runs used each vendor's sampling and a 32,000-token limit, 4 requests at a time, once per model; the thinking-off records they pair with are greedy with 2,048 and 512-token limits, 8 at a time. Sampled runs give different text on a re-run. The RTX 3050 was not used. The scoring code and the card were set up for this repository's Gemma work.

The strategy for measuring thinking on Qwen3.5-2B and Gemma 4 E2B on a local GPU was validated with an incremental step by step approach.

---

#### References

* [Comparing Qwen and Gemma on the Low End | dev.to](https://dev.to/gde/comparing-qwen-and-gemma-on-the-low-end-qwen-is-faster-without-drafting-gemma-has-better-quality-30bo)
* [local-llamacpp-1650ti-qwen352b-q4_0 | GitHub](https://github.com/xbill9/qwen-dev/tree/main/local-llamacpp-1650ti-qwen352b-q4_0)
* [Qwen/Qwen3.5-2B | Hugging Face](https://huggingface.co/Qwen/Qwen3.5-2B)
* [google/gemma-4-E2B-it | Hugging Face](https://huggingface.co/google/gemma-4-E2B-it)
* [xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf | Hugging Face](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf)
* [llama.cpp | GitHub](https://github.com/ggml-org/llama.cpp)
