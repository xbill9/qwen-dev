# Gap analysis — Qwen3.5-2B against Gemma 4 E2B on the RTX 3050 OEM

Every objection a reader can raise against the comparison, what the runs so far say about it, and what
would close it. Status as of 2026-10-16.

## What is measured

| | Qwen3.5-2B Q4_0 | Qwen3.5-2B bf16 | Gemma 4 E2B QAT Q4_0 | Gemma 4 E4B QAT Q4_0 |
| --- | ---: | ---: | ---: | ---: |
| Suite, 13 classification subsets (3,880) | 0.422 | 0.488 | 0.683 | 0.730 |
| GSM8K (1,319), greedy, thinking off | 0.743 | 0.798 | 0.898 | 0.933 |
| BFCL simple (400), greedy, thinking off | 0.710 | 0.765 | 0.925 | 0.910 |
| Generation, tok/s (tg128) | 132.84 | — | 118.30 | — |

## Closed

| objection | finding | evidence |
| --- | --- | --- |
| "llama.cpp's Qwen3.5 support is broken" (Gated DeltaNet is new) | bf16 next-token probabilities are closer to a float32 transformers reference than transformers' own bf16: same chosen label on 62 of 65 suite prompts against 61, median label-probability difference 0.0027 against 0.0128 | `local-llamacpp-3050oem-qwen352b/benchmarks/runs/2026-10-16-parity-transformers-3050oem` |
| "the chat template or tools were rendered wrong" | all 400 BFCL prompts with tools and 100 GSM8K prompts byte-identical to transformers' `apply_chat_template` | same run, `template_check.txt` |
| "the tool-call parser dropped Qwen's calls" | none of the 127 no-call BFCL replies (79 Q4_0, 48 bf16) contains call markup; they answer in prose | `bfcl-qwen.jsonl` in both quality runs |
| "Q4_0 crippled Qwen, Gemma got QAT" | bf16 Qwen scored on the same harness: Q4_0 costs 5.5–6.5 points per task; Gemma E2B's Q4 leads bf16 Qwen by 19.5 / 10.0 / 16.0, every range clear of zero | `local-llamacpp-3050oem-qwen352b/benchmarks/runs/2026-10-16-quality-bf16-3050oem` |
| "the suite harness used Gemma's tokenizer" | the Qwen runs pass `--tokenizer Qwen/Qwen3.5-2B`; label mass 0.94–0.999 on the yes/no and score subsets | `REPORT.md` of the Q4_0 quality run |
| "the speed numbers favour one model" | both on one llama.cpp binary, the same hour, alternating cells, repeats within 0.7% | `benchmarks/runs/2026-10-16-first-light-3050oem` |

## Open — these decide whether the result stands

| # | objection | why it matters | what closes it | cost on this card (estimate) |
| --- | --- | --- | --- | --- |
| 1 | **Thinking was off.** Qwen reports BFCL-V4, TAU2 and its reasoning scores in thinking mode; Gemma 4 also thinks | the vendors' own cards put Qwen3.5-2B **ahead** of E2B on all three benchmarks they share: MMLU-Pro 66.5 vs 60.0, GPQA Diamond 51.6 vs 43.4, Tau2 48.8 vs 24.5 | GSM8K and BFCL with thinking on, both models, a larger token budget | 4–5 h |
| 2 | **Greedy, not the vendors' sampling.** Qwen recommends temperature 1.0, presence_penalty 2.0 in non-thinking mode; greedy may cause the 100 GSM8K replies that ran to the limit | removes "you ran it wrong" | GSM8K and BFCL at each vendor's recommended sampling, both models, 2–3 seeds | 3–4 h |
| 3 | **Size class.** E2B is 5.1B parameters with per-layer embeddings (4.63B in the GGUF) against Qwen's 1.94B; its file is 2.48 GB against 1.24 | "compare Qwen3.5-4B, it's the same file size" | Qwen3.5-4B on the same three tasks (Q8_0 fits; bf16 does not) | 2 h |
| 4 | **A Gemma-built harness.** The suite's prompts and letter labels come from a proxy developed against Gemma models; Qwen often writes the option name where a letter is asked for | the suite gap (19.5 points at bf16) is the largest and the most attackable | lead with GSM8K and BFCL; add a benchmark both vendors publish (MMLU-Pro subset, or GPQA Diamond) scored the vendors' way, to test whether this stack reproduces the vendors' own numbers | 3–5 h, plus a harness |
| 5 | **Gemma's records are from 2026-10-13** on llama.cpp `77c4540ce` | a different binary; small, but it is a gap | re-run Gemma E2B on `7182ef71e` | 40 min |
| 6 | **Gemma was never run at bf16 here.** The bf16-vs-bf16 comparison uses Gemma's TPU bf16 runs (suite 0.683, GSM8K 0.910, BFCL 0.927) on a different stack | the clean claim is bf16 against bf16 on one stack | Gemma E2B bf16 GGUF on this card (its per-layer table stays in the mmap, so it should fit; untested) | 1.5 h incl. download |

## Out of scope, stated as such

- Vision, audio, video and long context: not measured for either model.
- Several requests at once: speed is single-user only.
- Benchmark contamination: unknown for both models.
- One card, one runtime. A vLLM or SGLang run is the stack Qwen recommends; none here.
