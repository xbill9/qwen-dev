# Answer quality: Gemma 4 E2B and Qwen3.5-2B on the GTX 1650 Ti, against the RTX 3050

## Nimble 13-subset suite

| run | records | accuracy | all labels returned |
|---|---:|---:|---:|
| Gemma E2B Q4, 1650 Ti | 3880 | 0.6807 | 96.3% |
| Gemma E2B Q4, 3050 | 3880 | 0.6827 | 96.2% |
| Qwen Q4_0, 1650 Ti | 3880 | 0.4299 | 90.2% |
| Qwen bf16, 1650 Ti | 3880 | 0.4881 | 82.3% |
| Qwen Q4_0, 3050 | 3880 | 0.4224 | 90.8% |
| Qwen bf16, 3050 | 3880 | 0.4879 | 82.2% |

Paired on the same records (difference = B - A; 95% range from 2,000 bootstrap draws):

| A | B | n | A | B | difference | 95% range | A right, B wrong | A wrong, B right |
|---|---|---:|---:|---:|---:|---|---:|---:|
| Gemma E2B Q4, 3050 | Gemma E2B Q4, 1650 Ti | 3880 | 0.683 | 0.681 | **-0.002** | -0.006 to +0.002 | 30 | 22 |
| Gemma E2B Q4, 1650 Ti | Qwen Q4_0, 1650 Ti | 3880 | 0.681 | 0.430 | **-0.251** | -0.271 to -0.231 | 1408 | 435 |
| Gemma E2B Q4, 1650 Ti | Qwen bf16, 1650 Ti | 3880 | 0.681 | 0.488 | **-0.193** | -0.211 to -0.174 | 1127 | 380 |
| Gemma E2B Q4, 3050 | Qwen Q4_0, 3050 | 3880 | 0.683 | 0.422 | **-0.260** | -0.281 to -0.240 | 1432 | 422 |
| Gemma E2B Q4, 3050 | Qwen bf16, 3050 | 3880 | 0.683 | 0.488 | **-0.195** | -0.213 to -0.177 | 1131 | 375 |

Per subset, Qwen bf16 - Gemma E2B on the 1650 Ti (secondary; ranges not corrected for 13 comparisons):

| subset | n | Gemma E2B | Qwen bf16 | Qwen Q4_0 | Qwen bf16 - Gemma | 95% range |
|---|---:|---:|---:|---:|---:|---|
| aegis2 | 250 | 0.744 | 0.692 | 0.696 | -0.052 | -0.108 to +0.004 |
| boolq | 300 | 0.627 | 0.737 | 0.720 | +0.110 | +0.047 to +0.173 |
| civil_comments | 300 | 0.900 | 0.233 | 0.303 | -0.667 | -0.727 to -0.603 |
| helpsteer2 | 249 | 0.365 | 0.341 | 0.261 | -0.024 | -0.076 to +0.028 |
| massive-de-DE | 350 | 0.734 | 0.566 | 0.409 | -0.169 | -0.217 to -0.117 |
| massive-en-US | 350 | 0.809 | 0.654 | 0.591 | -0.154 | -0.203 to -0.106 |
| multinli | 299 | 0.696 | 0.331 | 0.308 | -0.365 | -0.445 to -0.281 |
| paws | 250 | 0.740 | 0.520 | 0.536 | -0.220 | -0.280 to -0.156 |
| pubmedqa | 250 | 0.604 | 0.368 | 0.212 | -0.236 | -0.308 to -0.164 |
| squad2 | 299 | 0.686 | 0.816 | 0.732 | +0.130 | +0.067 to +0.194 |
| summeval-consistency | 144 | 0.757 | 0.160 | 0.090 | -0.597 | -0.694 to -0.500 |
| summeval-relevance | 240 | 0.296 | 0.246 | 0.171 | -0.050 | -0.129 to +0.029 |
| vitaminc-dev | 599 | 0.730 | 0.452 | 0.367 | -0.277 | -0.321 to -0.234 |

## GSM8K (2,048-token limit) and BFCL simple

| task | A | B | n | A | B | difference | 95% range | A only | B only |
|---|---|---|---:|---:|---:|---:|---|---:|---:|
| gsm8k | Gemma E2B Q4, 3050 | Gemma E2B Q4, 1650 Ti | 1319 | 0.898 | 0.901 | **+0.003** | -0.008 to +0.014 | 22 | 26 |
| gsm8k | Gemma E2B Q4, 1650 Ti | Qwen Q4_0, 1650 Ti | 1319 | 0.901 | 0.734 | **-0.167** | -0.190 to -0.143 | 255 | 35 |
| gsm8k | Gemma E2B Q4, 1650 Ti | Qwen bf16, 1650 Ti | 1319 | 0.901 | 0.807 | **-0.094** | -0.115 to -0.072 | 175 | 51 |
| gsm8k | Gemma E2B Q4, 3050 | Qwen Q4_0, 3050 | 1319 | 0.898 | 0.743 | **-0.155** | -0.177 to -0.132 | 239 | 35 |
| gsm8k | Gemma E2B Q4, 3050 | Qwen bf16, 3050 | 1319 | 0.898 | 0.798 | **-0.100** | -0.122 to -0.078 | 186 | 54 |
| bfcl_simple | Gemma E2B Q4, 3050 | Gemma E2B Q4, 1650 Ti | 400 | 0.925 | 0.917 | **-0.007** | -0.020 to +0.003 | 4 | 1 |
| bfcl_simple | Gemma E2B Q4, 1650 Ti | Qwen Q4_0, 1650 Ti | 400 | 0.917 | 0.688 | **-0.230** | -0.280 to -0.180 | 105 | 13 |
| bfcl_simple | Gemma E2B Q4, 1650 Ti | Qwen bf16, 1650 Ti | 400 | 0.917 | 0.762 | **-0.155** | -0.203 to -0.110 | 78 | 16 |
| bfcl_simple | Gemma E2B Q4, 3050 | Qwen Q4_0, 3050 | 400 | 0.925 | 0.710 | **-0.215** | -0.265 to -0.165 | 102 | 16 |
| bfcl_simple | Gemma E2B Q4, 3050 | Qwen bf16, 3050 | 400 | 0.925 | 0.765 | **-0.160** | -0.207 to -0.115 | 79 | 15 |

## The Qwen − Gemma gap on each card

| Qwen file | task | gap, 1650 Ti | gap, 3050 | 1650 Ti − 3050 |
|---|---|---:|---:|---:|
| Qwen Q4_0, 1650 Ti | suite | -0.251 | -0.260 | +0.010 |
| Qwen Q4_0, 1650 Ti | gsm8k | -0.167 | -0.155 | -0.012 |
| Qwen Q4_0, 1650 Ti | bfcl_simple | -0.230 | -0.215 | -0.015 |
| Qwen bf16, 1650 Ti | suite | -0.193 | -0.195 | +0.002 |
| Qwen bf16, 1650 Ti | gsm8k | -0.094 | -0.100 | +0.006 |
| Qwen bf16, 1650 Ti | bfcl_simple | -0.155 | -0.160 | +0.005 |

GSM8K replies that hit the 2,048-token limit (gen_eval.py's `finish` == `length`):

| run | truncated |
|---|---:|
| Gemma E2B Q4, 1650 Ti | 3 |
| Gemma E2B Q4, 3050 | 5 |
| Qwen Q4_0, 1650 Ti | 107 |
| Qwen bf16, 1650 Ti | 52 |
| Qwen Q4_0, 3050 | 100 |
| Qwen bf16, 3050 | 52 |

BFCL failure reasons (gen_eval.py's `reason`, wrong answers only):

| run | wrong | reasons |
|---|---:|---|
| Gemma E2B Q4, 1650 Ti | 33 | 5 × 0 tool calls; 2 × value interest_rate=5; 1 × missing root_type; 1 × value species="human"; 1 × value cell_type="human cell"; 1 × value database="database" |
| Gemma E2B Q4, 3050 | 30 | 6 × 0 tool calls; 1 × missing root_type; 1 × value species="human"; 1 × value cell_type="human cell"; 1 × value database="database"; 1 × value annual_return=7 |
| Qwen Q4_0, 1650 Ti | 125 | 88 × 0 tool calls; 2 × 2 tool calls; 2 × value room_type="single room"; 1 × value operating_hours=11; 1 × value cell_type="human cell"; 1 × value food_type="Thai food" |
| Qwen bf16, 1650 Ti | 95 | 48 × 0 tool calls; 2 × value interest_rate=5; 2 × value key="C major"; 2 × value room_type="single room"; 2 × value base_currency="US Dollar"; 1 × value formatted=false |
| Qwen Q4_0, 3050 | 116 | 79 × 0 tool calls; 2 × 2 tool calls; 1 × value operating_hours=11; 1 × value cell_type="human cell"; 1 × value food_type="Thai food"; 1 × value coord1=[33.4484, 112.074] |
| Qwen bf16, 3050 | 94 | 48 × 0 tool calls; 2 × value interest_rate=5; 2 × value key="C major"; 2 × value room_type="single room"; 2 × value base_currency="US Dollar"; 1 × value formatted=false |
