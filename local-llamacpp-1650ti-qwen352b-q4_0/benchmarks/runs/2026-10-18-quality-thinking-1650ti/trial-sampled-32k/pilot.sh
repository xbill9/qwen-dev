#!/bin/bash
set -u
P=$(dirname $0); J=/home/xbill/qwen-dev/jev-tpu-v5e1; BIN=/home/xbill/llama.cpp/build/bin/llama-server
stop8080() { pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done; }
for arm in 'qwen|Qwen/Qwen3.5-2B|/home/xbill/models/Qwen3.5-2B-gguf/Qwen3.5-2B-Q4_0.gguf|{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":1.5,"repeat_penalty":1.0}' \
           'e2b|google/gemma-4-E2B-it|/home/xbill/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf|{"temperature":1.0,"top_p":0.95,"top_k":64,"min_p":0.0}'; do
  IFS='|' read -r tag M GGUF S <<<"$arm"
  stop8080
  setsid $BIN -m $GGUF --host 127.0.0.1 --port 8080 -ngl 99 -c 131072 --parallel 4 -ctk f16 -ctv f16 -fa 1 \
    -t 6 -tb 12 --reasoning on --metrics > $P/server-$tag.log 2>&1 < /dev/null &
  for i in $(seq 180); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
  echo "$tag up $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader)"
  (cd $J && JEV_SAMPLING="$S" JEV_GEN_MAX_TOKENS=32000 python3 gen_eval.py run http://127.0.0.1:8080 $M gsm8k $P/gsm8k-$tag.jsonl --limit 16 --concurrency 4)
  (cd $J && JEV_SAMPLING="$S" JEV_BFCL_MAX_TOKENS=32000 python3 gen_eval.py run http://127.0.0.1:8080 $M bfcl_simple $P/bfcl-$tag.jsonl --limit 16 --concurrency 4)
done
stop8080
