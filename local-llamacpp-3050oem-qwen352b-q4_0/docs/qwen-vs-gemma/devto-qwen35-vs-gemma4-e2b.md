---
title: "Qwen3.5-2B Against Gemma 4 E2B on Two Budget GPUs: 1.12x the Speed, 10 to 26 Points Behind"
published: false
description: "Step-by-step: serving Qwen3.5-2B with llama.cpp on an 8 GB RTX 3050 and a 4 GB GTX 1650 Ti beside Gemma 4 E2B, checking that llama.cpp runs it correctly, and measuring speed, answer quality, 4-bit rounding, QAT and drafting with Qwen's own MTP head."
tags: gemma, qwen, llamacpp, cuda
cover_image: https://raw.githubusercontent.com/xbill9/qwen-dev/main/local-llamacpp-3050oem-qwen352b-q4_0/docs/qwen-vs-gemma/devto-cover.9ce8fe8e.jpg
---

This article provides a step by step deployment guide for Qwen3.5-2B to a local, desktop and laptop hosted GPU enabled system, served by llama.cpp beside Gemma 4 E2B on the same cards. A suite of Python MCP tools is built to simplify management of the llama.cpp hosted deployment.

https://github.com/xbill9/qwen-dev/tree/main/local-llamacpp-3050oem-qwen352b-q4_0

**Qwen3.5-2B decodes 1.12x as fast as Gemma 4 E2B on an RTX 3050 and prefills 1.24–1.31x as fast. Gemma answers better on every task measured: its 4-bit build leads Qwen's 4-bit build by 15.5 to 26.0 points and Qwen's unquantized bf16 by 10.0 to 19.5. With each model's drafter turned on, the speed lead disappears.**

---

#### What is this project trying to Do?

The rigs in this repository serve Gemma 4 E2B from local hardware: an RTX 3050 OEM in a desktop and a GTX 1650 Ti in a laptop. Both run Google's quantization-aware-trained (QAT) 4-bit build, and both have measured scores for it on the same three tasks.

The question here: what does a Qwen of the same size class give on the same cards? Qwen3.5-2B counts 1.943B parameters in its text GGUF. Everything below runs both models on one llama.cpp binary, the same flags and the same task code, so the model is the only difference.

---

#### At This Point You Should Have…

- An NVIDIA GPU with a CUDA driver — here an RTX 3050 OEM (8 GB, Ampere, driver 615.78.08) and a GTX 1650 Ti (4 GB, Turing)
- llama.cpp built with CUDA — commit `7182ef71e` here
- Python 3 for the converter and the MCP server — the system `python3`, no virtualenv
- The repository cloned, and `local-llamacpp-3050oem-qwen352b-q4_0/` as your working directory
- Claude Code, or any MCP client that speaks stdio

---

#### The Two Models

| | Qwen3.5-2B | Gemma 4 E2B |
| :--- | :--- | :--- |
| Layers | 24: three of every four Gated DeltaNet (linear attention), every fourth full attention | 35 |
| Parameters, as llama-bench counts them | 1.943B | 4.629B, most of it per-layer embeddings |
| Vocabulary | 248,320, tied input and output | 262,144 |
| 4-bit release | none; no QAT checkpoint and no GGUF published | Google's QAT Q4_0 |
| Drafter | an MTP layer inside the checkpoint | a separate 4-layer assistant model |

Only Qwen's 6 full-attention layers hold a KV cache. The 18 DeltaNet layers keep a fixed-size state per request, so Qwen's prefill cost grows more slowly with prompt length.

---

#### Step 1 — Build the Qwen Q4_0 File

Qwen publishes bf16 weights only, so the rig builds its own 4-bit file. `make quantize` converts the Hugging Face checkout to a bf16 GGUF and runs the standard `llama-quantize` Q4_0, with no importance matrix:

```shell
python3 ~/llama.cpp/convert_hf_to_gguf.py ~/models/Qwen3.5-2B --outtype bf16 --outfile Qwen3.5-2B-bf16.gguf
~/llama.cpp/build/bin/llama-quantize Qwen3.5-2B-bf16.gguf Qwen3.5-2B-Q4_0.gguf Q4_0
```

The result, from the rig's `tpu.env`:

```text
#   convert_hf_to_gguf.py --outtype bf16   -> 3,897,388,000 B, 335 tensors
#   llama-quantize ... Q4_0 (no imatrix)   -> 1,236,741,088 B
# llama-quantize's standard Q4_0 mix: 194 matrices Q4_0 (806.4 MB), token_embd
# Q6_K (417.2 MB, which is also the output head since it is tied), 140 F32 norms
```

This file is plain rounding from bf16. Gemma's Q4_0 file holds weights Google trained to sit on the 4-bit grid. Step 7 measures what that difference is worth.

---

#### Step 2 — Serve It

`make serve` runs llama-server with the Gemma rig's flags, so the two serve the same 131,072-token window:

```shell
llama-server -m Qwen3.5-2B-Q4_0.gguf --host 0.0.0.0 --port 8080 -ngl 99 -c 131072 \
  -ctk f16 -ctv f16 -fa 1 -t 6 -tb 12 --parallel 1 --reasoning off --metrics
```

Qwen3.5 thinks by default. `--reasoning off` puts answers in `content`, as on the Gemma rig; a request can still turn thinking on with `"chat_template_kwargs": {"enable_thinking": true}`.

At the full context it uses **2995 MiB of the 3050's 8192**, and 2951 MiB of the 1650 Ti's 4096. The server log shows the MTP layer loaded and left idle:

```text
0.01.266.683 W model has unused tensor blk.24.attn_norm.weight (size = 8192 bytes) -- ignoring
0.01.266.690 W model has unused tensor blk.24.post_attention_norm.weight (size = 8192 bytes) -- ignoring
0.01.266.699 W model has unused tensor blk.24.attn_q.weight (size = 4718592 bytes) -- ignoring
```

---

#### Step 3 — Check That llama.cpp Runs Qwen Correctly

A low score could come from the runtime. Two checks against Qwen's own Hugging Face code rule that out.

**Prompts.** Every prompt was rendered through llama-server's `/apply-template` and through transformers' `apply_chat_template`. All 400 tool-calling prompts, with their tools, and 100 math prompts came out byte-identical.

**Next-token probabilities.** The reference is transformers in float32 on the CPU over Qwen's bf16 weights, on 65 prompts from the quality suite:

| Against transformers fp32 | Same chosen answer | Answer-probability difference, median |
| :--- | ---: | ---: |
| transformers bf16, GPU | 61 of 65 | 0.0128 |
| **llama.cpp bf16 GGUF** | **62 of 65** 🥇 | **0.0027** 🥇 |
| llama.cpp Q4_0 GGUF | 43 of 65 | 0.1136 |

llama.cpp's bf16 file sits closer to the float32 reference than transformers' own bf16 path does. The drift comes from the 4-bit rounding.

---

#### Step 4 — Measure Speed on One Binary

`llama-bench` ran both models on the same build, card and hour, five cells alternating between models, each started with the GPU at 40 °C or below:

```shell
llama-bench -p 512,2048 -n 128 -r 5 -ngl 99 -fa 1 -t 6
```

The summary script's output on the RTX 3050:

```text
pp512: qwen 5193.44  gemma 4203.68  qwen/gemma 1.235x  qwen fa1/fa0 1.042x  qwen rep/first 0.993x  gemma rep/first 0.997x
pp2048: qwen 5316.13  gemma 4050.45  qwen/gemma 1.312x  qwen fa1/fa0 1.078x  qwen rep/first 0.996x  gemma rep/first 0.999x
tg128: qwen 132.84  gemma 118.30  qwen/gemma 1.123x  qwen fa1/fa0 1.012x  qwen rep/first 1.001x  gemma rep/first 1.002x
```

| | RTX 3050 | GTX 1650 Ti |
| :--- | ---: | ---: |
| Qwen decode, tok/s | 132.84 | 85.26 |
| Gemma decode, tok/s | 118.30 | 81.17 |
| Qwen / Gemma, decode | 1.12x | 1.05x |
| Qwen / Gemma, prefill at 512 / 2,048 tokens | 1.24x / 1.31x | 1.43x / 1.51x |

Qwen's prefill rises with prompt length (5193 to 5316 t/s) while Gemma's falls (4204 to 4050). That fits the DeltaNet layers, whose cost does not grow with the prompt.

---

#### Step 5 — Measure Answer Quality

Three tasks, the same task code and settings as the Gemma runs, 8 requests at a time, greedy:

- **Suite**: 13 classification sets, 3,880 questions, scored from the next-token probabilities
- **GSM8K**: 1,319 math word problems, 2,048-token limit
- **BFCL simple**: 400 tool-calling requests

Qwen's suite prompts end in Qwen's own chat template with thinking closed, and its answers are read from Qwen's own tokens. Every pair of runs is compared question by question, with a 95% range from resampling.

| RTX 3050 | Gemma E2B QAT Q4 | Qwen Q4_0 | Qwen − Gemma | 95% range |
| :--- | ---: | ---: | ---: | :--- |
| Suite | **0.683** 🥇 | 0.422 | −0.260 | −0.281 to −0.240 |
| GSM8K | **0.898** 🥇 | 0.743 | −0.155 | −0.177 to −0.132 |
| BFCL simple | **0.925** 🥇 | 0.710 | −0.215 | −0.265 to −0.165 |

Every range is far from zero. On the suite, Qwen is ahead only on boolq and squad2.

Where Qwen loses points:

- **Option letters.** Asked for `A`, `B` or `C`, it often writes the option's name instead. On multinli it answers `neutral` on 298 of 299 questions.
- **One-sided answers.** It calls 248 of 300 comments toxic, where the gold labels mark 32.
- **Long math answers.** 100 GSM8K replies hit the 2,048-token limit, against 5 for Gemma. With thinking off it still deliberates inside the answer.
- **Prose instead of tool calls.** 79 of its 116 wrong BFCL answers made no tool call, mostly by solving the problem in text. Gemma skipped the tool 6 times.

---

#### Step 6 — Score Qwen Unquantized

Qwen's 4-bit file is plain rounding, so part of the gap may be the rounding. The bf16 GGUF from Step 1 ran through the same three tasks on the same binary, the same day:

| RTX 3050 | Qwen Q4_0 | Qwen bf16 | Gemma E2B QAT Q4 |
| :--- | ---: | ---: | ---: |
| Suite | 0.422 | 0.488 | **0.683** 🥇 |
| GSM8K | 0.743 | 0.798 | **0.898** 🥇 |
| BFCL simple | 0.710 | 0.765 | **0.925** 🥇 |

- **Rounding to Q4_0 costs Qwen 5.5 to 6.5 points per task**, every range clear of zero.
- **Gemma's 4-bit build still leads unquantized Qwen by 19.5, 10.0 and 16.0 points.** The rounding accounts for a quarter to a third of the gap; the rest is the model.
- The same habits show at bf16: most comments called toxic, 48 BFCL replies with no tool call, 52 GSM8K replies at the length limit.

---

#### Step 7 — Measure What QAT Is Worth to Gemma

Gemma's 4-bit file comes from quantization-aware training; Qwen's comes from rounding. To compare like with like, Gemma E2B's ordinary bf16 release was rounded to Q4_0 with the exact recipe used for Qwen, and scored on the GTX 1650 Ti beside the other three:

| GTX 1650 Ti | Gemma QAT Q4_0 | Gemma rounded Q4_0 | Qwen bf16 | Qwen Q4_0 |
| :--- | ---: | ---: | ---: | ---: |
| Suite | **0.681** 🥇 | 0.632 🥈 | 0.488 | 0.430 |
| GSM8K | **0.901** 🥇 | 0.881 🥈 | 0.807 | 0.734 |
| BFCL simple | **0.917** 🥇 | 0.895 🥈 | 0.762 | 0.688 |

| Share of Qwen Q4_0's gap to Gemma QAT | Suite | GSM8K | BFCL |
| :--- | ---: | ---: | ---: |
| Whole gap | −0.251 | −0.167 | −0.230 |
| QAT's part | −0.048 | −0.020 | −0.022 |
| **QAT's share** | **19%** | **12%** | **10%** |

QAT is worth 2 to 5 points to Gemma at 4 bits. **Rounded Gemma still beats unquantized Qwen on all three tasks**, by 14.4, 7.4 and 13.3 points.

---

#### Step 8 — Run the Bf16 File on a 4 GB Card

Qwen's bf16 GGUF holds 3.62 GiB of weights, and the 1650 Ti has 4 GiB. With every layer on the card, the 501.52 MiB compute buffer fails to allocate. Moving the last 6 of 24 blocks to the CPU is the fewest that fits:

```shell
llama-server -m Qwen3.5-2B-bf16.gguf -ngl 99 -ot 'blk\.(18|19|20|21|22|23)\.=CPU' \
  -c 8192 -ctk f16 -ctv f16 -fa 1 -t 6 -tb 12 --parallel 1 --reasoning off --metrics
```

| GTX 1650 Ti | Decode, tok/s | Prefill at 512, t/s | GPU memory |
| :--- | ---: | ---: | ---: |
| Q4_0, all on the card | **85.11** 🥇 | **511.43** 🥇 | 2951 MiB at 131k context |
| bf16, 6 blocks on the CPU | 20.22 | 490.87 | 3207 MiB at 8k context |

Decode drops to 0.24x and prefill holds at 0.96x. The split costs no score: bf16 on this card scored 0.488, 0.807 and 0.762, level with the 3050.

---

#### Step 9 — Draft With Qwen's MTP Head

The Q4_0 file already carries Qwen's multi-token-prediction layer. llama.cpp drafts from it with `--spec-type draft-mtp` and no extra file:

```shell
llama-server -m Qwen3.5-2B-Q4_0.gguf -ngl 99 -c 8192 -fa 1 \
  --spec-type draft-mtp --spec-draft-n-max 1
```

The log confirms it and reports acceptance per request:

```text
0.02.045.088 I common_speculative_init_result: creating MTP draft context against the target model '/home/xbill/models/Qwen3.5-2B-gguf/Qwen3.5-2B-Q4_0.gguf'
0.04.830.818 I slot print_timing: id  0 | task 8 | draft acceptance = 0.91729 (  122 accepted /   133 generated), mean len =  1.92
```

Six prompts (code, chat, story, explain, facts, summary), one request at a time, geometric mean against no drafting on the RTX 3050:

| Drafted tokens | 1 | 2 | 3 | 4 | 6 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Greedy | **1.149x** 🥇 | 1.145x | 0.998x | 0.852x | 0.647x |
| Sampled, temperature 0.8 | **1.086x** 🥇 | 1.047x | 0.920x | 0.782x | 0.596x |

- **1 drafted token is the setting to serve.** Code gains most (1.25x greedy); story least (1.09x).
- **Greedy replies stay byte-identical to no drafting up to 3 drafted tokens** on the 3050.
- On the same card, Gemma's separate drafter gave 1.39x at 2 tokens.

---

#### Step 10 — Race the Two Drafters on One Card

On the GTX 1650 Ti, both models on one binary, one request at a time, each with its best drafter setting of 1 token:

| GTX 1650 Ti, greedy tok/s | No drafter | 1 drafted token |
| :--- | ---: | ---: |
| Qwen Q4_0, own MTP head | **82.2** 🥇 | 91.0 (1.107x) |
| Gemma E2B, QAT drafter | 77.9 | **92.1 (1.181x)** 🥇 |
| Qwen / Gemma | 1.054 | 0.988 |

Sampled at temperature 0.8 the result is the same: 86.1 against 90.7 tok/s. Gemma's drafter gains more, so the two tie.

Drafting leaves Qwen's answers level: GSM8K 0.748 and BFCL 0.698 with the MTP head, against 0.734 and 0.688 without. At 8 requests at once, each task took 1.08–1.09x as long with drafting.

---

#### 🔎 Tip: Pass Qwen's Tokenizer to Probability-Scored Tasks

A task that scores answers from next-token probabilities has to look up the answer labels in the model's own vocabulary. Qwen's differs from Gemma's, so give the scorer `--tokenizer Qwen/Qwen3.5-2B`; the default Gemma ids would read the wrong tokens.

---

#### 🔎 Tip: Two Hosts, Two Different Hashes

The 3050 host and the 1650 Ti host built Qwen's Q4_0 from the same revision on the same llama.cpp commit, and got files of the same size with different SHA-256s (`1c27608b…` and `bd99d5ae…`). The bf16 files differ too, so the change enters at conversion. They pick the same answer on 62 of 65 prompts and score within 2.2 points on every task. Pin a hash per host, and keep one file if you need byte-identical results across machines.

---

#### Compare and Contrast

| | Qwen3.5-2B Q4_0 | Gemma 4 E2B QAT Q4_0 |
| :--- | :--- | :--- |
| Decode, RTX 3050 | **132.84 tok/s** 🥇 | 118.30 tok/s |
| Decode with drafter, GTX 1650 Ti | 91.0 tok/s | **92.1 tok/s** 🥇 |
| Suite / GSM8K / BFCL, RTX 3050 | 0.422 / 0.743 / 0.710 | **0.683 / 0.898 / 0.925** 🥇 |
| GPU memory at 131k context, RTX 3050 | **2995 MiB** 🥇 | 3610 MiB |
| 4-bit build | rounded locally | **QAT, published by Google** 🥇 |
| Drafter | built in, no extra file | separate 59 MB file |

---

#### So, Which One?

Gemma 4 E2B. Qwen3.5-2B decodes faster without a drafter, prefills faster on long prompts and uses less memory at long context. Gemma answers better on every task measured, at 4 bits against Qwen's bf16, and with both drafters on the speed difference is gone.

Qwen's built-in MTP head is a convenience: drafting with no extra file, and greedy text unchanged at 1 drafted token. Choose Qwen when prefill speed on long prompts matters more than the answers.

---

#### Summary

The goal of this article was to serve Qwen3.5-2B with llama.cpp beside Gemma 4 E2B on two budget GPUs and compare them on speed and answer quality. The key to the solution was running both models on one llama.cpp binary with the same flags and the same task code, and scoring Qwen at bf16 and Gemma rounded the way Qwen's file is built. The results were:

- 🟢 **llama.cpp runs Qwen3.5-2B correctly**: identical prompts to Qwen's template, and its bf16 probabilities sit closer to float32 than transformers' bf16 does.
- 🟢 **Qwen is faster without drafting**: 1.12x decode and 1.24–1.31x prefill on the RTX 3050, 1.05x and 1.43–1.51x on the GTX 1650 Ti.
- ❌ **Qwen scores lower on every task**: 15.5 to 26.0 points behind Gemma at 4 bits, 10.0 to 19.5 behind at bf16.
- ⚠️ **Rounding is a quarter to a third of Qwen's gap**, and QAT is a tenth to a fifth of Gemma's lead.
- 🟢 **Qwen's MTP head drafts with no extra file**: 1.15x greedy at 1 token, greedy text unchanged.
- ⚠️ **With both drafters on, the two tie on speed**: 91.0 against 92.1 tok/s on the GTX 1650 Ti.

Scope: one RTX 3050 OEM (8 GB) and one GTX 1650 Ti (4 GB), llama.cpp `7182ef71e` for every Qwen run and for Gemma on the 1650 Ti, measured 2026-10-16 and 2026-10-17. The Gemma quality figures on the 3050 come from 2026-10-13 on llama.cpp `77c4540ce`; Gemma's runs on the 1650 Ti on `7182ef71e` scored level with them. Speed cells ran 3 to 5 repeats; each quality task ran once per file. Qwen ran with thinking off throughout, and no importance-matrix or Q8_0 build of either model was scored.

The strategy for using MCP for serving and comparing Qwen3.5-2B and Gemma 4 E2B on local GPUs was validated with an incremental step by step approach.

---

#### References

* [local-llamacpp-3050oem-qwen352b-q4_0 | GitHub](https://github.com/xbill9/qwen-dev/tree/main/local-llamacpp-3050oem-qwen352b-q4_0)
* [local-llamacpp-1650ti-qwen352b-q4_0 | GitHub](https://github.com/xbill9/qwen-dev/tree/main/local-llamacpp-1650ti-qwen352b-q4_0)
* [Qwen/Qwen3.5-2B | Hugging Face](https://huggingface.co/Qwen/Qwen3.5-2B)
* [xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf | Hugging Face](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf)
* [google/gemma-4-E2B-it | Hugging Face](https://huggingface.co/google/gemma-4-E2B-it)
* [Gemma 4's MTP Drafter, Explained | dev.to](https://dev.to/gde/gemma-4s-mtp-drafter-explained-speculative-decoding-and-a-4-bit-drafter-at-60-of-the-size-1oc0)
* [llama.cpp | GitHub](https://github.com/ggml-org/llama.cpp)
