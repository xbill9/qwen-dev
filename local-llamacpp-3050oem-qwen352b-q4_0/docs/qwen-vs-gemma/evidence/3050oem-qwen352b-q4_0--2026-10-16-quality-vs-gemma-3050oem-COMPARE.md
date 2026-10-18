# Answer quality: Qwen3.5-2B Q4_0 against Gemma 4 E2B / E4B exact Q4_0, RTX 3050 OEM

## Nimble 13-subset suite

| run | records | accuracy | all labels returned |
|---|---:|---:|---:|
| Qwen3.5-2B Q4_0 | 3880 | 0.4224 | 90.8% |
| Gemma E2B Q4 | 3880 | 0.6827 | 96.2% |
| Gemma E4B Q4 | 3880 | 0.7304 | 82.0% |

Paired on the same records (difference = B - A; 95% range from 2,000 bootstrap draws):

| A | B | n | A | B | difference | 95% range | A right, B wrong | A wrong, B right |
|---|---|---:|---:|---:|---:|---|---:|---:|
| Gemma E2B Q4 | Qwen3.5-2B Q4_0 | 3880 | 0.683 | 0.422 | **-0.260** | -0.281 to -0.240 | 1432 | 422 |
| Gemma E4B Q4 | Qwen3.5-2B Q4_0 | 3880 | 0.730 | 0.422 | **-0.308** | -0.328 to -0.289 | 1530 | 335 |

Per subset, Qwen - Gemma E2B (secondary; ranges not corrected for 13 comparisons):

| subset | n | E2B | Qwen | Qwen - E2B | 95% range |
|---|---:|---:|---:|---:|---|
| aegis2 | 250 | 0.752 | 0.688 | -0.064 | -0.136 to +0.004 |
| boolq | 300 | 0.623 | 0.707 | +0.083 | +0.003 to +0.163 |
| civil_comments | 300 | 0.903 | 0.267 | -0.637 | -0.697 to -0.573 |
| helpsteer2 | 249 | 0.369 | 0.257 | -0.112 | -0.181 to -0.048 |
| massive-de-DE | 350 | 0.746 | 0.400 | -0.346 | -0.406 to -0.283 |
| massive-en-US | 350 | 0.806 | 0.589 | -0.217 | -0.269 to -0.163 |
| multinli | 299 | 0.682 | 0.308 | -0.375 | -0.458 to -0.294 |
| paws | 250 | 0.748 | 0.548 | -0.200 | -0.260 to -0.136 |
| pubmedqa | 250 | 0.608 | 0.196 | -0.412 | -0.492 to -0.332 |
| squad2 | 299 | 0.686 | 0.739 | +0.054 | -0.017 to +0.127 |
| summeval-consistency | 144 | 0.757 | 0.090 | -0.667 | -0.757 to -0.569 |
| summeval-relevance | 240 | 0.292 | 0.171 | -0.121 | -0.188 to -0.058 |
| vitaminc-dev | 599 | 0.736 | 0.354 | -0.382 | -0.431 to -0.332 |

## GSM8K (2,048-token limit) and BFCL simple

| task | A | B | n | A | B | difference | 95% range | A only | B only |
|---|---|---|---:|---:|---:|---:|---|---:|---:|
| gsm8k | Gemma E2B Q4 | Qwen3.5-2B Q4_0 | 1319 | 0.898 | 0.743 | **-0.155** | -0.177 to -0.132 | 239 | 35 |
| gsm8k | Gemma E4B Q4 | Qwen3.5-2B Q4_0 | 1319 | 0.933 | 0.743 | **-0.190** | -0.212 to -0.167 | 267 | 17 |
| bfcl_simple | Gemma E2B Q4 | Qwen3.5-2B Q4_0 | 400 | 0.925 | 0.710 | **-0.215** | -0.265 to -0.165 | 102 | 16 |
| bfcl_simple | Gemma E4B Q4 | Qwen3.5-2B Q4_0 | 400 | 0.910 | 0.710 | **-0.200** | -0.253 to -0.150 | 101 | 21 |

BFCL failure reasons (gen_eval.py's `reason`, wrong answers only):

| run | wrong | reasons |
|---|---:|---|
| Qwen3.5-2B Q4_0 | 116 | 79 × 0 tool calls; 2 × 2 tool calls; 1 × value operating_hours=11; 1 × value cell_type="human cell"; 1 × value food_type="Thai food"; 1 × value coord1=[33.4484, 112.074] |
| Gemma E2B Q4 | 30 | 6 × 0 tool calls; 1 × missing root_type; 1 × value species="human"; 1 × value cell_type="human cell"; 1 × value database="database"; 1 × value annual_return=7 |
| Gemma E4B Q4 | 36 | 14 × 0 tool calls; 2 × value interest_rate=5; 1 × value isothermal=false; 1 × value species="human"; 1 × value cell_type="human cell"; 1 × value end_location="Washington, D.C., DC" |
