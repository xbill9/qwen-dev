# Answer quality: Qwen3.5-2B Q4_0 against Gemma 4 E2B, with and without its drafter, on the Tesla T4

## Nimble 13-subset suite

| run | records | accuracy | all labels returned |
|---|---:|---:|---:|
| Qwen Q4_0, T4 | 3880 | 0.4219 | 90.3% |
| Gemma E2B, T4 | 3880 | 0.6814 | 96.2% |
| Gemma E2B + drafter, T4 | 3880 | 0.6817 | 96.2% |
| Qwen Q4_0, 1650 Ti | 3880 | 0.4299 | 90.2% |
| Gemma E2B, 1650 Ti | 3880 | 0.6807 | 96.3% |

Paired on the same records (difference = B - A; 95% range from 2,000 bootstrap draws):

| A | B | n | A | B | difference | 95% range | A right, B wrong | A wrong, B right |
|---|---|---:|---:|---:|---:|---|---:|---:|
| Gemma E2B, T4 | Qwen Q4_0, T4 | 3880 | 0.681 | 0.422 | **-0.260** | -0.280 to -0.239 | 1432 | 425 |
| Gemma E2B, T4 | Gemma E2B + drafter, T4 | 3880 | 0.681 | 0.682 | **+0.000** | -0.003 to +0.004 | 25 | 26 |
| Qwen Q4_0, 1650 Ti | Qwen Q4_0, T4 | 3880 | 0.430 | 0.422 | **-0.008** | -0.014 to -0.002 | 79 | 48 |
| Gemma E2B, 1650 Ti | Gemma E2B, T4 | 3880 | 0.681 | 0.681 | **+0.001** | -0.003 to +0.005 | 25 | 28 |
| Gemma E2B, 1650 Ti | Qwen Q4_0, 1650 Ti | 3880 | 0.681 | 0.430 | **-0.251** | -0.271 to -0.231 | 1408 | 435 |

Per subset, Qwen Q4_0 - Gemma E2B on the T4 (secondary; ranges not corrected for 13 comparisons):

| subset | n | Gemma | Qwen | Qwen - Gemma | 95% range |
|---|---:|---:|---:|---:|---|
| aegis2 | 250 | 0.744 | 0.648 | -0.096 | -0.168 to -0.028 |
| boolq | 300 | 0.623 | 0.710 | +0.087 | +0.003 to +0.170 |
| civil_comments | 300 | 0.907 | 0.267 | -0.640 | -0.703 to -0.577 |
| helpsteer2 | 249 | 0.361 | 0.265 | -0.096 | -0.169 to -0.032 |
| massive-de-DE | 350 | 0.737 | 0.406 | -0.331 | -0.389 to -0.269 |
| massive-en-US | 350 | 0.803 | 0.591 | -0.211 | -0.263 to -0.157 |
| multinli | 299 | 0.689 | 0.311 | -0.378 | -0.458 to -0.294 |
| paws | 250 | 0.748 | 0.552 | -0.196 | -0.256 to -0.136 |
| pubmedqa | 250 | 0.588 | 0.192 | -0.396 | -0.476 to -0.316 |
| squad2 | 299 | 0.692 | 0.739 | +0.047 | -0.027 to +0.120 |
| summeval-consistency | 144 | 0.764 | 0.090 | -0.674 | -0.764 to -0.576 |
| summeval-relevance | 240 | 0.312 | 0.167 | -0.146 | -0.217 to -0.083 |
| vitaminc-dev | 599 | 0.731 | 0.357 | -0.374 | -0.421 to -0.324 |

## GSM8K (2,048-token limit) and BFCL simple

| task | A | B | n | A | B | difference | 95% range | A only | B only |
|---|---|---|---:|---:|---:|---:|---|---:|---:|
| gsm8k | Gemma E2B, T4 | Qwen Q4_0, T4 | 1319 | 0.892 | 0.735 | **-0.158** | -0.182 to -0.134 | 243 | 35 |
| gsm8k | Gemma E2B, T4 | Gemma E2B + drafter, T4 | 1319 | 0.892 | 0.895 | **+0.002** | -0.008 to +0.012 | 22 | 25 |
| gsm8k | Qwen Q4_0, 1650 Ti | Qwen Q4_0, T4 | 1319 | 0.734 | 0.735 | **+0.001** | -0.020 to +0.021 | 93 | 94 |
| gsm8k | Gemma E2B, 1650 Ti | Gemma E2B, T4 | 1319 | 0.901 | 0.892 | **-0.008** | -0.020 to +0.003 | 34 | 23 |
| gsm8k | Gemma E2B, 1650 Ti | Qwen Q4_0, 1650 Ti | 1319 | 0.901 | 0.734 | **-0.167** | -0.190 to -0.143 | 255 | 35 |
| bfcl_simple | Gemma E2B, T4 | Qwen Q4_0, T4 | 400 | 0.922 | 0.698 | **-0.225** | -0.275 to -0.175 | 107 | 17 |
| bfcl_simple | Gemma E2B, T4 | Gemma E2B + drafter, T4 | 400 | 0.922 | 0.922 | **+0.000** | +0.000 to +0.000 | 0 | 0 |
| bfcl_simple | Qwen Q4_0, 1650 Ti | Qwen Q4_0, T4 | 400 | 0.688 | 0.698 | **+0.010** | -0.013 to +0.033 | 8 | 12 |
| bfcl_simple | Gemma E2B, 1650 Ti | Gemma E2B, T4 | 400 | 0.917 | 0.922 | **+0.005** | -0.005 to +0.015 | 1 | 3 |
| bfcl_simple | Gemma E2B, 1650 Ti | Qwen Q4_0, 1650 Ti | 400 | 0.917 | 0.688 | **-0.230** | -0.280 to -0.180 | 105 | 13 |

## Qwen − Gemma gap, T4 against 1650 Ti

| | T4 | 1650 Ti | moved |
|---|---:|---:|---:|
| suite | -0.260 | -0.251 | -0.009 |
| gsm8k | -0.158 | -0.167 | +0.009 |
| bfcl | -0.225 | -0.230 | +0.005 |

## Replies

| run | GSM8K cut off at 2,048 | GSM8K reply tokens | BFCL no tool call | BFCL reply tokens |
|---|---:|---:|---:|---:|
| Qwen Q4_0, T4 | 100 | 702,741 | 82 | 43,678 |
| Gemma E2B, T4 | 3 | 527,255 | 6 | 12,759 |
| Gemma E2B + drafter, T4 | 3 | 527,430 | 6 | 12,752 |
| Qwen Q4_0, 1650 Ti | 107 | 706,288 | 88 | 44,127 |
| Gemma E2B, 1650 Ti | 3 | 528,448 | 5 | 12,763 |

BFCL failure reasons (gen_eval.py's `reason`, wrong answers only):

| run | wrong | reasons |
|---|---:|---|
| Qwen Q4_0, T4 | 121 | 82 × 0 tool calls; 1 × value formatted=false; 1 × value operating_hours=11; 1 × value cell_type="human cell"; 1 × value food_type="Thai food"; 1 × value conditions={"school": "Bluebird High School"} |
| Gemma E2B, T4 | 31 | 6 × 0 tool calls; 1 × missing root_type; 1 × value species="human"; 1 × value cell_type="human cell"; 1 × value database="database"; 1 × value annual_return=7 |
| Gemma E2B + drafter, T4 | 31 | 6 × 0 tool calls; 1 × missing root_type; 1 × value species="human"; 1 × value cell_type="human cell"; 1 × value database="database"; 1 × value annual_return=7 |

## Gemma with and without its drafter, record by record

| task | records | same text | same right/wrong |
|---|---:|---:|---:|
| gsm8k | 1319 | 235 | 1272 |
| bfcl | 400 | 398 | 400 |
| suite | 3880 | - | 3829 |
