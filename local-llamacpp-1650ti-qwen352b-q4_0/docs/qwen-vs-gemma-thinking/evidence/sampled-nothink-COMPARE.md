# Sampling against thinking — compare.py output

## GSM8K, fixed random 300

| model | arm | n | accuracy | median tokens | hit limit | no tool call |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-2B Q4_0 | greedy-off | 300 | 0.730 | 365 | 26 |  |
| Qwen3.5-2B Q4_0 | sampled-off | 300 | 0.790 | 430 | 4 |  |
| Qwen3.5-2B Q4_0 | sampled-on | 300 | 0.827 | 7607 | 31 |  |
| Gemma 4 E2B QAT Q4_0 | greedy-off | 300 | 0.927 | 344 | 0 |  |
| Gemma 4 E2B QAT Q4_0 | sampled-off | 300 | 0.910 | 356 | 0 |  |
| Gemma 4 E2B QAT Q4_0 | sampled-on | 300 | 0.930 | 808 | 0 |  |

| pair (test vs reference) | n | reference | test | test − ref | 95% range | ref only | test only |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| qwen: sampling, thinking off | 300 | 0.730 | 0.790 | +0.060 | +0.003 to +0.117 | 28 | 46 |
| qwen: thinking, at vendor sampling | 300 | 0.790 | 0.827 | +0.037 | -0.013 to +0.087 | 24 | 35 |
| qwen: both | 300 | 0.730 | 0.827 | +0.097 | +0.043 to +0.150 | 19 | 48 |
| e2b: sampling, thinking off | 300 | 0.927 | 0.910 | -0.017 | -0.047 to +0.010 | 12 | 7 |
| e2b: thinking, at vendor sampling | 300 | 0.910 | 0.930 | +0.020 | -0.010 to +0.050 | 9 | 15 |
| e2b: both | 300 | 0.927 | 0.930 | +0.003 | -0.023 to +0.030 | 8 | 9 |
| qwen vs e2b, greedy-off | 300 | 0.927 | 0.730 | -0.197 | -0.247 to -0.147 | 66 | 7 |
| qwen vs e2b, sampled-off | 300 | 0.910 | 0.790 | -0.120 | -0.167 to -0.073 | 45 | 9 |
| qwen vs e2b, sampled-on | 300 | 0.930 | 0.827 | -0.103 | -0.143 to -0.063 | 35 | 4 |

## BFCL simple, 400

| model | arm | n | accuracy | median tokens | hit limit | no tool call |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-2B Q4_0 | greedy-off | 400 | 0.688 | 59 | 12 | 88 |
| Qwen3.5-2B Q4_0 | sampled-off | 400 | 0.660 | 59 | 1 | 83 |
| Qwen3.5-2B Q4_0 | sampled-on | 400 | 0.838 | 160 | 1 | 8 |
| Gemma 4 E2B QAT Q4_0 | greedy-off | 400 | 0.917 | 31 | 0 | 5 |
| Gemma 4 E2B QAT Q4_0 | sampled-off | 400 | 0.885 | 32 | 0 | 6 |
| Gemma 4 E2B QAT Q4_0 | sampled-on | 400 | 0.887 | 284 | 0 | 7 |

| pair (test vs reference) | n | reference | test | test − ref | 95% range | ref only | test only |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| qwen: sampling, thinking off | 400 | 0.688 | 0.660 | -0.028 | -0.068 to +0.013 | 39 | 28 |
| qwen: thinking, at vendor sampling | 400 | 0.660 | 0.838 | +0.177 | +0.128 to +0.228 | 23 | 94 |
| qwen: both | 400 | 0.688 | 0.838 | +0.150 | +0.100 to +0.198 | 24 | 84 |
| e2b: sampling, thinking off | 400 | 0.917 | 0.885 | -0.033 | -0.055 to -0.013 | 16 | 3 |
| e2b: thinking, at vendor sampling | 400 | 0.885 | 0.887 | +0.003 | -0.025 to +0.030 | 16 | 17 |
| e2b: both | 400 | 0.917 | 0.887 | -0.030 | -0.058 to -0.005 | 20 | 8 |
| qwen vs e2b, greedy-off | 400 | 0.917 | 0.688 | -0.230 | -0.280 to -0.180 | 105 | 13 |
| qwen vs e2b, sampled-off | 400 | 0.885 | 0.660 | -0.225 | -0.275 to -0.177 | 103 | 13 |
| qwen vs e2b, sampled-on | 400 | 0.887 | 0.838 | -0.050 | -0.090 to -0.013 | 42 | 22 |
