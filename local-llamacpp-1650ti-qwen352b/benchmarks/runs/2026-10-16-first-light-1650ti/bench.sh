#!/usr/bin/env bash
# First light: Qwen3.5-2B bf16 on a 4 GiB GTX 1650 Ti, where the 3.62 GiB of weights cannot all be resident.
# Two ways to split it, each cell started at GPU <= COOL_C:
#   cpuK  -ngl 99, the last K of the 24 blocks on the CPU via -ot, output head on the card
#   nglN  -ngl N, the plain llama.cpp split (blocks past N and the output head on the CPU)
# Plus the Q4_0 sister's file at full offload as the reference. Usage: bench.sh <out_dir>
set -euo pipefail
OUT=${1:?out dir}; mkdir -p "$OUT"
BIN=/home/xbill/llama.cpp/build/bin/llama-bench
BF16=/home/xbill/models/Qwen3.5-2B-gguf/Qwen3.5-2B-bf16.gguf
Q4=/home/xbill/models/Qwen3.5-2B-gguf/Qwen3.5-2B-Q4_0.gguf
COOL_C=${COOL_C:-50}
gpu() { nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader,nounits | tr -d ' '; }
cool() { while (( $(gpu | cut -d, -f1) > COOL_C )); do sleep 5; done; }
cpu_blocks() { local k=$1 s=""; for ((i=24-k; i<24; i++)); do s+="${s:+|}$i"; done; echo "blk\.($s)\.=CPU"; }
CELLS=()
for k in ${KS:-2 4 6 8}; do CELLS+=("cpu$k|$BF16|-ngl 99 -ot $(cpu_blocks "$k")"); done
for n in ${NS:-20}; do CELLS+=("ngl$n|$BF16|-ngl $n"); done
CELLS+=("q4_0|$Q4|-ngl 99")
for c in "${CELLS[@]}"; do
  IFS='|' read -r name model flags <<<"$c"
  cool; g0=$(gpu)
  # shellcheck disable=SC2086
  if "$BIN" -m "$model" -p 512,2048 -n 128 -r 3 $flags -fa 1 -ctk f16 -ctv f16 -t 6 -o json > "$OUT/$name.json" 2> "$OUT/$name.log"; then st=ok; else st=failed; fi
  g1=$(gpu)
  echo "$name status=$st flags=[$flags] start(C,MHz,W)=$g0 end=$g1" | tee -a "$OUT/cells.txt"
done
