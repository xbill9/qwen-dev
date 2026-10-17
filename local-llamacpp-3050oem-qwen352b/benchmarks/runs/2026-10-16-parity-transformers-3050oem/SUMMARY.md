| test, against transformers fp32 | prompts | same chosen label | same top-1 token | label-probability difference, median / max | top-5 log-probability difference, median / max |
|---|---:|---:|---:|---|---|
| transformers bf16 (GPU) | 65 | 61 | 59 | 0.0128 / 0.563 | 0.124 / 2.98 |
| llama.cpp bf16 GGUF | 65 | 62 | 60 | 0.0027 / 0.562 | 0.050 / 3.35 |
| llama.cpp Q4_0 GGUF | 65 | 43 | 47 | 0.1136 / 0.488 | 1.638 / 5.13 |
