# Gap analysis — "Comparing Qwen and Gemma on the Low End" (dev.to draft 4873703), 2026-10-18

What a reader who prefers Qwen will object to, checked against the run data where possible. Numbers
from `evidence/best_case.py` (output in `evidence/best_case.txt`) unless a run is named.

## Objections checked: Gemma still leads

| Objection | Check | Result |
| --- | --- | --- |
| GSM8K cut Qwen's long answers off at 2,048 tokens | score every cut-off reply right | Q4_0 0.743 → 0.812, bf16 0.798 → 0.836; Gemma 0.898 |
| llama.cpp mis-parsed Qwen's tool calls | search no-call BFCL replies for call-like text | 0 of 79 (Q4_0), 0 of 48 (bf16); the replies solve the problem in prose |
| — | score every no-call BFCL reply right | Q4_0 0.710 → 0.908, bf16 0.765 → 0.885; Gemma 0.925 |
| The suite scores format, not knowledge | drop civil_comments, summeval-consistency, multinli (1650 Ti) | bf16 gap −0.193 → −0.112; Gemma still leads |
| Gemma is the bigger model (4.6B against 1.9B parameters) | file size | Qwen bf16 3.90 GB loses every task to Gemma's 2.61 GB QAT file |
| Q4_0 with no imatrix is Qwen's worst quant | bf16 is the ceiling any 4-bit build can reach | Gemma leads bf16 on every task; Q4_K_M / imatrix untested |

The suite objection is partly right: three subsets where Qwen answers one way almost every time, or
writes the option name for the letter, shrink the gap from 0.193 to 0.112 when removed. The suite is the weakest of the
three results.

## Gaps that could change a conclusion

| Gap | Status |
| --- | --- |
| **Thinking off for both models.** Qwen3.5 thinks by default and deliberates in its answers with thinking off. | Not run. Needs a longer token limit and Gemma with thinking on as well. The largest open question, on GSM8K most of all. |
| Greedy decoding throughout; Qwen's recommended sampling settings | Not checked |
| Three tasks only: no coding, Chinese, instruction following, long-context tasks; German and English intent classification only for multilingual | Not run; the title generalises from three tasks |
| Prefill measured to 2,048 tokens; Qwen at one request at a time only | Not run; Qwen's linear-attention layers favour long prompts |
| Scoring code from the Gemma work, cards chosen for Gemma, article filed under the Gemma GDE organization | Disclose |

## Corrections owed in the article

1. Step 3 says two checks "rule that out", but they cover prompt rendering and next-token probabilities,
   not output parsing. Add the tool-call text check above.
2. The Compare and Contrast memory row sets Qwen's 2995 MiB against Gemma's 3610 MiB, which came from a
   different llama.cpp build with Gemma's per-layer table on the card. On the 1650 Ti, with that table in
   system memory, Gemma's quality server used 1710 MiB against Qwen's 1806. Drop "Qwen uses less memory".
3. "Choose Qwen when prefill speed on long prompts matters" goes past the 2,048-token prefill data.
4. State "on these three tasks" and "thinking off for both" next to "better quality".
5. Answer the size objection with the file sizes.

## Next runs, cheapest first

- Article edits above (no runs).
- Thinking on, both models, GSM8K and BFCL with an 8k-token limit, RTX 3050. Duration not estimated;
  bf16 GSM8K took 48.5 min at 2,048.
- Qwen Q4_K_M with an importance matrix; the suite rescored to accept option names.
