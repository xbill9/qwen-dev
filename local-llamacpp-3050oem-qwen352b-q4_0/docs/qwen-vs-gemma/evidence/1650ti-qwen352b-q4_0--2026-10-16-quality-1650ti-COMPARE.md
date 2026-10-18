# Answer quality: Qwen3.5-2B Q4_0 and bf16 on the GTX 1650 Ti, against the 3050 and Gemma 4 E2B

## Nimble 13-subset suite

| run | records | accuracy | all labels returned |
|---|---:|---:|---:|
| Qwen bf16, 1650 Ti | 3880 | 0.4881 | 82.3% |
| Qwen Q4_0, 1650 Ti | 3880 | 0.4299 | 90.2% |
| Qwen bf16, 3050 | 3880 | 0.4879 | 82.2% |
| Qwen Q4_0, 3050 | 3880 | 0.4224 | 90.8% |
| Gemma E2B Q4, 3050 | 3880 | 0.6827 | 96.2% |

Paired on the same records (difference = B - A; 95% range from 2,000 bootstrap draws):

| A | B | n | A | B | difference | 95% range | A right, B wrong | A wrong, B right |
|---|---|---:|---:|---:|---:|---|---:|---:|
| Qwen bf16, 1650 Ti | Qwen Q4_0, 1650 Ti | 3880 | 0.488 | 0.430 | **-0.058** | -0.073 to -0.043 | 588 | 362 |
| Qwen Q4_0, 3050 | Qwen Q4_0, 1650 Ti | 3880 | 0.422 | 0.430 | **+0.007** | +0.002 to +0.013 | 49 | 78 |
| Qwen bf16, 3050 | Qwen bf16, 1650 Ti | 3880 | 0.488 | 0.488 | **+0.000** | -0.003 to +0.003 | 16 | 17 |
| Gemma E2B Q4, 3050 | Qwen Q4_0, 1650 Ti | 3880 | 0.683 | 0.430 | **-0.253** | -0.273 to -0.233 | 1413 | 432 |
| Gemma E2B Q4, 3050 | Qwen bf16, 1650 Ti | 3880 | 0.683 | 0.488 | **-0.195** | -0.213 to -0.177 | 1130 | 375 |

Per subset, Qwen Q4_0 - Qwen bf16 on the 1650 Ti (secondary; ranges not corrected for 13 comparisons):

| subset | n | bf16 | Q4_0 | Q4_0 - bf16 | 95% range |
|---|---:|---:|---:|---:|---|
| aegis2 | 250 | 0.692 | 0.696 | +0.004 | -0.044 to +0.052 |
| boolq | 300 | 0.737 | 0.720 | -0.017 | -0.067 to +0.033 |
| civil_comments | 300 | 0.233 | 0.303 | +0.070 | +0.023 to +0.120 |
| helpsteer2 | 249 | 0.341 | 0.261 | -0.080 | -0.145 to -0.016 |
| massive-de-DE | 350 | 0.566 | 0.409 | -0.157 | -0.200 to -0.109 |
| massive-en-US | 350 | 0.654 | 0.591 | -0.063 | -0.106 to -0.020 |
| multinli | 299 | 0.331 | 0.308 | -0.023 | -0.043 to -0.007 |
| paws | 250 | 0.520 | 0.536 | +0.016 | +0.004 to +0.032 |
| pubmedqa | 250 | 0.368 | 0.212 | -0.156 | -0.224 to -0.084 |
| squad2 | 299 | 0.816 | 0.732 | -0.084 | -0.127 to -0.043 |
| summeval-consistency | 144 | 0.160 | 0.090 | -0.069 | -0.146 to +0.000 |
| summeval-relevance | 240 | 0.246 | 0.171 | -0.075 | -0.133 to -0.017 |
| vitaminc-dev | 599 | 0.452 | 0.367 | -0.085 | -0.147 to -0.023 |

## GSM8K (2,048-token limit) and BFCL simple

| task | A | B | n | A | B | difference | 95% range | A only | B only |
|---|---|---|---:|---:|---:|---:|---|---:|---:|
| gsm8k | Qwen bf16, 1650 Ti | Qwen Q4_0, 1650 Ti | 1319 | 0.807 | 0.734 | **-0.073** | -0.096 to -0.049 | 172 | 76 |
| gsm8k | Qwen Q4_0, 3050 | Qwen Q4_0, 1650 Ti | 1319 | 0.743 | 0.734 | **-0.009** | -0.030 to +0.011 | 100 | 88 |
| gsm8k | Qwen bf16, 3050 | Qwen bf16, 1650 Ti | 1319 | 0.798 | 0.807 | **+0.009** | -0.001 to +0.019 | 16 | 28 |
| gsm8k | Gemma E2B Q4, 3050 | Qwen Q4_0, 1650 Ti | 1319 | 0.898 | 0.734 | **-0.164** | -0.187 to -0.140 | 255 | 39 |
| gsm8k | Gemma E2B Q4, 3050 | Qwen bf16, 1650 Ti | 1319 | 0.898 | 0.807 | **-0.091** | -0.113 to -0.069 | 175 | 55 |
| bfcl_simple | Qwen bf16, 1650 Ti | Qwen Q4_0, 1650 Ti | 400 | 0.762 | 0.688 | **-0.075** | -0.113 to -0.037 | 46 | 16 |
| bfcl_simple | Qwen Q4_0, 3050 | Qwen Q4_0, 1650 Ti | 400 | 0.710 | 0.688 | **-0.022** | -0.045 to +0.000 | 15 | 6 |
| bfcl_simple | Qwen bf16, 3050 | Qwen bf16, 1650 Ti | 400 | 0.765 | 0.762 | **-0.003** | -0.007 to +0.000 | 1 | 0 |
| bfcl_simple | Gemma E2B Q4, 3050 | Qwen Q4_0, 1650 Ti | 400 | 0.925 | 0.688 | **-0.237** | -0.287 to -0.190 | 108 | 13 |
| bfcl_simple | Gemma E2B Q4, 3050 | Qwen bf16, 1650 Ti | 400 | 0.925 | 0.762 | **-0.163** | -0.210 to -0.117 | 80 | 15 |

BFCL failure reasons (gen_eval.py's `reason`, wrong answers only):

| run | wrong | reasons |
|---|---:|---|
| Qwen bf16, 1650 Ti | 95 | 48 × 0 tool calls; 2 × value interest_rate=5; 2 × value key="C major"; 2 × value room_type="single room"; 2 × value base_currency="US Dollar"; 1 × value formatted=false |
| Qwen Q4_0, 1650 Ti | 125 | 88 × 0 tool calls; 2 × 2 tool calls; 2 × value room_type="single room"; 1 × value operating_hours=11; 1 × value cell_type="human cell"; 1 × value food_type="Thai food" |
| Qwen bf16, 3050 | 94 | 48 × 0 tool calls; 2 × value interest_rate=5; 2 × value key="C major"; 2 × value room_type="single room"; 2 × value base_currency="US Dollar"; 1 × value formatted=false |
| Qwen Q4_0, 3050 | 116 | 79 × 0 tool calls; 2 × 2 tool calls; 1 × value operating_hours=11; 1 × value cell_type="human cell"; 1 × value food_type="Thai food"; 1 × value coord1=[33.4484, 112.074] |
| Gemma E2B Q4, 3050 | 30 | 6 × 0 tool calls; 1 × missing root_type; 1 × value species="human"; 1 × value cell_type="human cell"; 1 × value database="database"; 1 × value annual_return=7 |
