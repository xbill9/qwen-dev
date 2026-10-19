# Thinking on against thinking off — compare.py output

## GSM8K, fixed random 300 of 1,319

| model | thinking | n | accuracy | median tokens | mean | max | cut off (right) | best case | errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Qwen3.5-2B Q4_0 | on | 300 | 0.827 | 7607 | 11360 | 32000 | 31 (5) | 0.913 | 0 |
| Qwen3.5-2B Q4_0 | off | 300 | 0.730 | 365 | 535 | 2048 | 26 (6) | 0.797 | 0 |
| Gemma 4 E2B QAT Q4_0 | on | 300 | 0.930 | 808 | 903 | 5070 | 0 (0) | 0.930 | 0 |
| Gemma 4 E2B QAT Q4_0 | off | 300 | 0.927 | 344 | 392 | 1586 | 0 (0) | 0.927 | 0 |

| pair (test vs reference) | n | reference | test | test − ref | 95% range | ref only | test only |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| qwen on vs off | 300 | 0.730 | 0.827 | +0.097 | +0.043 to +0.150 | 19 | 48 |
| e2b on vs off | 300 | 0.927 | 0.930 | +0.003 | -0.023 to +0.030 | 8 | 9 |
| qwen vs e2b, thinking off | 300 | 0.927 | 0.730 | -0.197 | -0.247 to -0.147 | 66 | 7 |
| qwen vs e2b, thinking on | 300 | 0.930 | 0.827 | -0.103 | -0.143 to -0.063 | 35 | 4 |
| qwen on vs e2b off | 300 | 0.927 | 0.827 | -0.100 | -0.143 to -0.060 | 37 | 7 |

## BFCL simple, 400

| model | thinking | n | accuracy | median tokens | mean | max | cut off (right) | best case | errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Qwen3.5-2B Q4_0 | on | 400 | 0.838 | 160 | 275 | 32000 | 1 (0) | 0.840 | 0 |
| Qwen3.5-2B Q4_0 | off | 400 | 0.688 | 59 | 110 | 512 | 12 (0) | 0.718 | 0 |
| Gemma 4 E2B QAT Q4_0 | on | 400 | 0.887 | 284 | 293 | 1280 | 0 (0) | 0.887 | 0 |
| Gemma 4 E2B QAT Q4_0 | off | 400 | 0.917 | 31 | 32 | 92 | 0 (0) | 0.917 | 0 |

BFCL replies with no tool call, and failure reasons:

- Qwen3.5-2B Q4_0, thinking on: 8 with no call; {'0': 8, 'value': 53, 'missing': 3, 'name': 1}
- Qwen3.5-2B Q4_0, thinking off: 88 with no call; {'0': 88, 'value': 32, 'missing': 2, 'name': 1, '2': 2}
- Gemma 4 E2B QAT Q4_0, thinking on: 7 with no call; {'missing': 2, 'value': 35, '0': 7, 'name': 1}
- Gemma 4 E2B QAT Q4_0, thinking off: 5 with no call; {'missing': 3, 'value': 24, '0': 5, 'name': 1}

| pair (test vs reference) | n | reference | test | test − ref | 95% range | ref only | test only |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| qwen on vs off | 400 | 0.688 | 0.838 | +0.150 | +0.100 to +0.198 | 24 | 84 |
| e2b on vs off | 400 | 0.917 | 0.887 | -0.030 | -0.058 to -0.005 | 20 | 8 |
| qwen vs e2b, thinking off | 400 | 0.917 | 0.688 | -0.230 | -0.280 to -0.180 | 105 | 13 |
| qwen vs e2b, thinking on | 400 | 0.887 | 0.838 | -0.050 | -0.090 to -0.013 | 42 | 22 |
| qwen on vs e2b off | 400 | 0.917 | 0.838 | -0.080 | -0.117 to -0.043 | 47 | 15 |
