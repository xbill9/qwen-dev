# Qwen3.5-2B Q4_0 on the GTX 1650 Ti: answers with and without its MTP head drafting

| task | no draft | MTP, 1 token | difference | 95% range | no draft only | MTP only | same reply text |
|---|---:|---:|---:|---|---:|---:|---:|
| gsm8k | 0.734 | 0.748 | **+0.014** | -0.007 to +0.036 | 93 | 112 | 68 of 1319 |
| bfcl_simple | 0.688 | 0.698 | **+0.010** | -0.007 to +0.028 | 5 | 9 | 303 of 400 |

| task | arm | replies at the length limit | reply tokens | wall time, min | tokens per second, all 8 slots |
|---|---|---:|---:|---:|---:|
| gsm8k | no draft | 107 | 706288 | 64.8 | 181.7 |
| gsm8k | MTP, 1 token | 94 | 684576 | 70.2 | 162.5 |
| bfcl_simple | no draft | 12 | 44127 | 9.3 | 78.8 |
| bfcl_simple | MTP, 1 token | 11 | 43669 | 10.2 | 71.6 |

Wall time, MTP / no draft: gsm8k 1.084x, bfcl_simple 1.089x
Drafted tokens accepted over the whole run (/metrics): 344756 of 382574, 0.901
