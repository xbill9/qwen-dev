#!/usr/bin/env bash
# First light: Qwen3.5-2B standard Q4_0 against Gemma 4 E2B exact Q4_0, same llama.cpp binary,
# same card, same day. Cells alternate models and each starts at GPU <= COOL_C, recording
# temperature, SM clock and power before and after. Usage: bench.sh <out_dir>
set -euo pipefail
OUT=${1:?out dir}; mkdir -p "$OUT"
BIN=/home/xbill/llama.cpp/build/bin/llama-bench
QWEN=/home/xbill/models/Qwen3.5-2B-gguf/Qwen3.5-2B-Q4_0.gguf
GEMMA=/home/xbill/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf
COOL_C=${COOL_C:-40}
gpu() { nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader,nounits | tr -d ' '; }
cool() { while (( $(gpu | cut -d, -f1) > COOL_C )); do sleep 5; done; }
# name | model | flags. Each model's serving config: -fa 1, f16 KV, -t 6; Gemma keeps
# per_layer_token_embd on the card as its rig serves.
CELLS=(
  "qwen-fa1|$QWEN|-fa 1 -ctk f16 -ctv f16 -t 6"
  "gemma-fa1|$GEMMA|-fa 1 -ctk f16 -ctv f16 -t 6 -ot per_layer_token_embd\.weight=CUDA0"
  "qwen-fa0|$QWEN|-fa 0 -ctk f16 -ctv f16 -t 6"
  "qwen-fa1-rep|$QWEN|-fa 1 -ctk f16 -ctv f16 -t 6"
  "gemma-fa1-rep|$GEMMA|-fa 1 -ctk f16 -ctv f16 -t 6 -ot per_layer_token_embd\.weight=CUDA0"
)
for c in "${CELLS[@]}"; do
  IFS='|' read -r name model flags <<<"$c"
  cool; g0=$(gpu)
  # shellcheck disable=SC2086
  "$BIN" -m "$model" -p 512,2048 -n 128 -r 5 -ngl 99 $flags -o json > "$OUT/$name.json" 2> "$OUT/$name.log"
  g1=$(gpu)
  echo "$name flags=[$flags] start(C,MHz,W)=$g0 end=$g1" | tee -a "$OUT/cells.txt"
done
