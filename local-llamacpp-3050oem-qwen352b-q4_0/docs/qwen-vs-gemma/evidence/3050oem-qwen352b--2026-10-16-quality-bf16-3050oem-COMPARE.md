# Answer quality: Qwen3.5-2B bf16 against its Q4_0 and Gemma 4 E2B / E4B exact Q4_0, RTX 3050 OEM

## Nimble 13-subset suite

| run | records | accuracy | all labels returned |
|---|---:|---:|---:|
| Qwen3.5-2B bf16 | 3880 | 0.4879 | 82.2% |
| Qwen3.5-2B Q4_0 | 3880 | 0.4224 | 90.8% |
| Gemma E2B Q4 | 3880 | 0.6827 | 96.2% |
| Gemma E4B Q4 | 3880 | 0.7304 | 82.0% |

Paired on the same records (difference = B - A; 95% range from 2,000 bootstrap draws):

| A | B | n | A | B | difference | 95% range | A right, B wrong | A wrong, B right |
|---|---|---:|---:|---:|---:|---|---:|---:|
| Qwen3.5-2B bf16 | Qwen3.5-2B Q4_0 | 3880 | 0.488 | 0.422 | **-0.065** | -0.081 to -0.049 | 603 | 349 |
| Gemma E2B Q4 | Qwen3.5-2B bf16 | 3880 | 0.683 | 0.488 | **-0.195** | -0.213 to -0.177 | 1131 | 375 |
| Gemma E4B Q4 | Qwen3.5-2B bf16 | 3880 | 0.730 | 0.488 | **-0.243** | -0.260 to -0.226 | 1200 | 259 |

Per subset, Qwen Q4_0 - Qwen bf16 (secondary; ranges not corrected for 13 comparisons):

| subset | n | bf16 | Q4_0 | Q4_0 - bf16 | 95% range |
|---|---:|---:|---:|---:|---|
| aegis2 | 250 | 0.692 | 0.688 | -0.004 | -0.056 to +0.048 |
| boolq | 300 | 0.737 | 0.707 | -0.030 | -0.077 to +0.020 |
| civil_comments | 300 | 0.227 | 0.267 | +0.040 | -0.003 to +0.083 |
| helpsteer2 | 249 | 0.345 | 0.257 | -0.088 | -0.153 to -0.024 |
| massive-de-DE | 350 | 0.563 | 0.400 | -0.163 | -0.211 to -0.114 |
| massive-en-US | 350 | 0.654 | 0.589 | -0.066 | -0.109 to -0.023 |
| multinli | 299 | 0.334 | 0.308 | -0.027 | -0.047 to -0.010 |
| paws | 250 | 0.520 | 0.548 | +0.028 | +0.008 to +0.048 |
| pubmedqa | 250 | 0.376 | 0.196 | -0.180 | -0.248 to -0.108 |
| squad2 | 299 | 0.809 | 0.739 | -0.070 | -0.114 to -0.027 |
| summeval-consistency | 144 | 0.139 | 0.090 | -0.049 | -0.125 to +0.021 |
| summeval-relevance | 240 | 0.246 | 0.171 | -0.075 | -0.133 to -0.017 |
| vitaminc-dev | 599 | 0.457 | 0.354 | -0.104 | -0.164 to -0.042 |

## GSM8K (2,048-token limit) and BFCL simple

| task | A | B | n | A | B | difference | 95% range | A only | B only |
|---|---|---|---:|---:|---:|---:|---|---:|---:|
| gsm8k | Qwen3.5-2B bf16 | Qwen3.5-2B Q4_0 | 1319 | 0.798 | 0.743 | **-0.055** | -0.077 to -0.032 | 158 | 86 |
| gsm8k | Gemma E2B Q4 | Qwen3.5-2B bf16 | 1319 | 0.898 | 0.798 | **-0.100** | -0.122 to -0.078 | 186 | 54 |
| gsm8k | Gemma E4B Q4 | Qwen3.5-2B bf16 | 1319 | 0.933 | 0.798 | **-0.135** | -0.156 to -0.114 | 205 | 27 |
| bfcl_simple | Qwen3.5-2B bf16 | Qwen3.5-2B Q4_0 | 400 | 0.765 | 0.710 | **-0.055** | -0.092 to -0.020 | 41 | 19 |
| bfcl_simple | Gemma E2B Q4 | Qwen3.5-2B bf16 | 400 | 0.925 | 0.765 | **-0.160** | -0.207 to -0.115 | 79 | 15 |
| bfcl_simple | Gemma E4B Q4 | Qwen3.5-2B bf16 | 400 | 0.910 | 0.765 | **-0.145** | -0.195 to -0.098 | 80 | 22 |

BFCL failure reasons (gen_eval.py's `reason`, wrong answers only):

| run | wrong | reasons |
|---|---:|---|
| Qwen3.5-2B bf16 | 94 | 48 × 0 tool calls; 2 × value interest_rate=5; 2 × value key="C major"; 2 × value room_type="single room"; 2 × value base_currency="US Dollar"; 1 × value formatted=false |
| Qwen3.5-2B Q4_0 | 116 | 79 × 0 tool calls; 2 × 2 tool calls; 1 × value operating_hours=11; 1 × value cell_type="human cell"; 1 × value food_type="Thai food"; 1 × value coord1=[33.4484, 112.074] |
| Gemma E2B Q4 | 30 | 6 × 0 tool calls; 1 × missing root_type; 1 × value species="human"; 1 × value cell_type="human cell"; 1 × value database="database"; 1 × value annual_return=7 |
| Gemma E4B Q4 | 36 | 14 × 0 tool calls; 2 × value interest_rate=5; 1 × value isothermal=false; 1 × value species="human"; 1 × value cell_type="human cell"; 1 × value end_location="Washington, D.C., DC" |
