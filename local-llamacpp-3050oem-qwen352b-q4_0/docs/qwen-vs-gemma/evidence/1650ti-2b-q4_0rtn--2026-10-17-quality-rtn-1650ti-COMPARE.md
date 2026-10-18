# Answer quality: Gemma 4 E2B with and without QAT, against Qwen3.5-2B, GTX 1650 Ti

## What the files hold (bytes by tensor type, from gguf-py)

| file | total GB | by type |
|---|---:|---|
| Gemma E2B rounded Q4_0 | 3.344 | Q6_K 67.5%, Q4_0 31.7%, BF16 0.8%, F32 0.0% |
| Gemma E2B QAT Q4_0 | 2.605 | Q4_0 100.0%, F32 0.0% |
| Qwen Q4_0 | 1.226 | Q4_0 65.8%, Q6_K 34.0%, F32 0.2% |

## Nimble 13-subset suite

| run | records | accuracy | all labels returned |
|---|---:|---:|---:|
| Gemma E2B rounded Q4_0 | 3880 | 0.6322 | 83.6% |
| Gemma E2B QAT Q4_0 | 3880 | 0.6807 | 96.3% |
| Qwen Q4_0 | 3880 | 0.4299 | 90.2% |
| Qwen bf16 | 3880 | 0.4881 | 82.3% |

Paired on the same records (difference = B - A; 95% range from 2,000 bootstrap draws):

| A | B | n | A | B | difference | 95% range | A right, B wrong | A wrong, B right |
|---|---|---:|---:|---:|---:|---|---:|---:|
| Gemma E2B QAT Q4_0 | Gemma E2B rounded Q4_0 | 3880 | 0.681 | 0.632 | **-0.048** | -0.061 to -0.035 | 417 | 229 |
| Gemma E2B rounded Q4_0 | Qwen Q4_0 | 3880 | 0.632 | 0.430 | **-0.202** | -0.222 to -0.183 | 1223 | 438 |
| Gemma E2B rounded Q4_0 | Qwen bf16 | 3880 | 0.632 | 0.488 | **-0.144** | -0.163 to -0.125 | 1040 | 481 |
| Gemma E2B QAT Q4_0 | Qwen Q4_0 | 3880 | 0.681 | 0.430 | **-0.251** | -0.271 to -0.231 | 1408 | 435 |

Per subset, rounded - QAT (secondary; ranges not corrected for 13 comparisons):

| subset | n | QAT | rounded | rounded - QAT | 95% range |
|---|---:|---:|---:|---:|---|
| aegis2 | 250 | 0.744 | 0.772 | +0.028 | -0.012 to +0.072 |
| boolq | 300 | 0.627 | 0.540 | -0.087 | -0.123 to -0.053 |
| civil_comments | 300 | 0.900 | 0.900 | +0.000 | -0.027 to +0.023 |
| helpsteer2 | 249 | 0.365 | 0.398 | +0.032 | -0.008 to +0.072 |
| massive-de-DE | 350 | 0.734 | 0.757 | +0.023 | -0.006 to +0.051 |
| massive-en-US | 350 | 0.809 | 0.826 | +0.017 | -0.003 to +0.040 |
| multinli | 299 | 0.696 | 0.679 | -0.017 | -0.067 to +0.037 |
| paws | 250 | 0.740 | 0.716 | -0.024 | -0.072 to +0.020 |
| pubmedqa | 250 | 0.604 | 0.228 | -0.376 | -0.460 to -0.296 |
| squad2 | 299 | 0.686 | 0.729 | +0.043 | +0.000 to +0.087 |
| summeval-consistency | 144 | 0.757 | 0.486 | -0.271 | -0.361 to -0.188 |
| summeval-relevance | 240 | 0.296 | 0.254 | -0.042 | -0.100 to +0.017 |
| vitaminc-dev | 599 | 0.730 | 0.646 | -0.083 | -0.119 to -0.050 |

## GSM8K (2,048-token limit) and BFCL simple

| task | A | B | n | A | B | difference | 95% range | A only | B only |
|---|---|---|---:|---:|---:|---:|---|---:|---:|
| gsm8k | Gemma E2B QAT Q4_0 | Gemma E2B rounded Q4_0 | 1319 | 0.901 | 0.881 | **-0.020** | -0.034 to -0.005 | 61 | 35 |
| gsm8k | Gemma E2B rounded Q4_0 | Qwen Q4_0 | 1319 | 0.881 | 0.734 | **-0.147** | -0.171 to -0.124 | 239 | 45 |
| gsm8k | Gemma E2B rounded Q4_0 | Qwen bf16 | 1319 | 0.881 | 0.807 | **-0.074** | -0.096 to -0.052 | 162 | 64 |
| gsm8k | Gemma E2B QAT Q4_0 | Qwen Q4_0 | 1319 | 0.901 | 0.734 | **-0.167** | -0.190 to -0.143 | 255 | 35 |
| bfcl_simple | Gemma E2B QAT Q4_0 | Gemma E2B rounded Q4_0 | 400 | 0.917 | 0.895 | **-0.022** | -0.050 to +0.005 | 19 | 10 |
| bfcl_simple | Gemma E2B rounded Q4_0 | Qwen Q4_0 | 400 | 0.895 | 0.688 | **-0.207** | -0.263 to -0.155 | 108 | 25 |
| bfcl_simple | Gemma E2B rounded Q4_0 | Qwen bf16 | 400 | 0.895 | 0.762 | **-0.133** | -0.182 to -0.083 | 80 | 27 |
| bfcl_simple | Gemma E2B QAT Q4_0 | Qwen Q4_0 | 400 | 0.917 | 0.688 | **-0.230** | -0.280 to -0.180 | 105 | 13 |

## How much of Qwen Q4_0's gap to Gemma QAT is QAT

Gap = Qwen Q4_0 - Gemma QAT. QAT's part = rounded - QAT. Left after QAT = Qwen Q4_0 - Gemma rounded.

| task | gap | QAT's part | left after QAT | QAT's share |
|---|---:|---:|---:|---:|
| suite | -0.251 | -0.048 | -0.202 | 19% |
| gsm8k | -0.167 | -0.020 | -0.147 | 12% |
| bfcl_simple | -0.230 | -0.022 | -0.207 | 10% |

GSM8K replies that hit the 2,048-token limit (gen_eval.py's `finish` == `length`):

| run | truncated |
|---|---:|
| Gemma E2B rounded Q4_0 | 4 |
| Gemma E2B QAT Q4_0 | 3 |
| Qwen Q4_0 | 107 |
| Qwen bf16 | 52 |

BFCL failure reasons (gen_eval.py's `reason`, wrong answers only):

| run | wrong | reasons |
|---|---:|---|
| Gemma E2B rounded Q4_0 | 42 | 17 × 0 tool calls; 2 × missing detail; 1 × missing root_type; 1 × value species="human"; 1 × value cell_type="human cell"; 1 × value conditions={"school": "Bluebird High School"} |
| Gemma E2B QAT Q4_0 | 33 | 5 × 0 tool calls; 2 × value interest_rate=5; 1 × missing root_type; 1 × value species="human"; 1 × value cell_type="human cell"; 1 × value database="database" |
| Qwen Q4_0 | 125 | 88 × 0 tool calls; 2 × 2 tool calls; 2 × value room_type="single room"; 1 × value operating_hours=11; 1 × value cell_type="human cell"; 1 × value food_type="Thai food" |
| Qwen bf16 | 95 | 48 × 0 tool calls; 2 × value interest_rate=5; 2 × value key="C major"; 2 × value room_type="single room"; 2 × value base_currency="US Dollar"; 1 × value formatted=false |
