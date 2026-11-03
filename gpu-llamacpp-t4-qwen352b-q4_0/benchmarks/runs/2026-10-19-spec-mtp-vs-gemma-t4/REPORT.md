# 2026-10-19 — speculative decoding on the Tesla T4, Qwen3.5-2B's MTP head against Gemma 4 E2B's QAT drafter

**Question:** for one user on the T4, does drafting change which 2B model generates faster?

**Answer: it erases Qwen's lead, as on the GTX 1650 Ti.** Without drafting Qwen Q4_0 decodes 1.11x as fast
as Gemma E2B (119.5 against 107.6 tok/s, greedy). Each model's best drafter setting is 1 token, and there they
tie: 130.9 against 131.7 tok/s greedy (0.99x), 124.8 against 127.7 sampled (0.98x). Gemma's drafter gains
more, 1.22x greedy against Qwen's 1.10x.

## Method

`spec_ab.py`, the 1650 Ti run's method (`../../../../local-llamacpp-1650ti-qwen352b-q4_0/benchmarks/runs/2026-10-17-spec-mtp-vs-gemma-1650ti`)
with `-t 2 -tb 2` for this 2-vCPU VM and a third draft length: one llama-server per arm on port 8099,
llama.cpp `7182ef71e` (sm_75), `-c 8192 --parallel 1`, `-fa 1`, f16 KV, `--reasoning off`, GPU at ≤50 °C before
each arm. Six prompts (code, chat, story, explain, facts, summary), max 256 tokens, prompt cache off; greedy
×3, then one sampled pass (temperature 0.8, seed 1234). Decode tok/s is llama.cpp's own `timings`. Files:
Qwen's Q4_0 built here (`d2bc253a…`), drafting from the MTP layer inside it (`--spec-type draft-mtp`, no `-md`);
Gemma's exact QAT Q4_0 (`419db9a6…`) with `per_layer_token_embd` on the card and Google's QAT assistant drafter
stored exact Q4_0 (`060f5a01…`, `-md … -ngld all`). Arms: no draft, 1, 2 and 3 drafted tokens.
`summarize.py` computes every table (`summary.txt`, `summary.json`); raw records with reply texts in
`results.jsonl`. Largest spread across greedy repeats: 3.0% (Qwen), 2.0% (Gemma).

## Results, geometric mean of six prompts

| | no draft | 1 token | 2 tokens | 3 tokens |
| --- | ---: | ---: | ---: | ---: |
| Qwen Q4_0, greedy tok/s | 119.5 | **130.9 (1.096x)** | 122.8 (1.028x) | 107.4 (0.899x) |
| Gemma E2B, greedy tok/s | 107.6 | **131.7 (1.224x)** | 129.3 (1.202x) | 121.8 (1.132x) |
| Qwen / Gemma, greedy | 1.110 | 0.994 | 0.950 | 0.882 |
| Qwen Q4_0, sampled tok/s | 115.8 | **124.8 (1.078x)** | 110.8 (0.957x) | 95.2 (0.822x) |
| Gemma E2B, sampled tok/s | 106.5 | **127.7 (1.200x)** | 123.9 (1.164x) | 115.1 (1.081x) |
| Qwen / Gemma, sampled | 1.087 | 0.977 | 0.894 | 0.827 |

- **1 drafted token is best for both models**, greedy and sampled. Qwen loses from 2 tokens on sampled text
  and from 3 greedy; Gemma keeps a gain at every length.
- **Code drafts best for both**: Qwen 1.22–1.26x at 1–2 tokens, Gemma 1.44x at 1 and 1.73x at 3. Story and
  summary least (Gemma 1.08–1.17x at 1 token).
- **Acceptance of drafted tokens at 1 token**, greedy: Qwen 0.62–0.91 per prompt, Gemma 0.40–0.90. Qwen
  accepts more and gains less.
- **VRAM:** Qwen 1439 → 1563 MiB with its MTP head at 1 token, Gemma 2799 → 2879 MiB with its drafter.

## Three cards

| best drafted arm (1 token), greedy | T4 | RTX 3050 OEM | GTX 1650 Ti |
| --- | ---: | ---: | ---: |
| Qwen MTP gain | 1.096x | 1.149x | 1.107x |
| Gemma QAT drafter gain | 1.224x | 1.38x | 1.181x |
| Qwen / Gemma, no draft | 1.110 | — | 1.054 |
| Qwen / Gemma, both at 1 token | 0.994 | — | 0.988 |

3050: Qwen `../../../../local-llamacpp-3050oem-qwen352b-q4_0/benchmarks/runs/2026-10-16-spec-mtp-3050oem`
(`7182ef71e`); Gemma `../../../../local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-drafter-gaps-3050oem`
(`77c4540ce`, a different binary, so no Qwen/Gemma ratio is formed there). 1650 Ti: the run above.

- **On both cards measured with both models on one binary, drafting at 1 token leaves them level** (0.99x).
- Gemma's drafter gains most on the 3050 (1.38x) and about the same on the T4 and the 1650 Ti (1.18–1.22x).

## Greedy replies

Every arm repeats itself exactly. **Qwen at 1 drafted token matches its no-draft reply on all 6 prompts**, and
on none at 2 or 3, as on the 1650 Ti. **Gemma's drafter changes 6 of 6 replies at 1 token** and 5 of 6 at 2
and 3. Whether that changes answer quality is measured in `../2026-10-19-quality-t4`.

## Not measured

More than one request at a time, long prompts, thinking on. The same prompts under vLLM on this card, with
Gemma's QAT W4A16 build and bf16 QAT drafter, are in
`~/gemma4-dev/gpu-vllm-t4-2b-w4a16/benchmarks/runs/2026-10-19-vllm-vs-llamacpp-t4`.
