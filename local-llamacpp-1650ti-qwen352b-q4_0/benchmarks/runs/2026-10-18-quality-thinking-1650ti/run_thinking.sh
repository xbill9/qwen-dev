#!/bin/bash
# Answer quality with thinking on, Qwen3.5-2B Q4_0 against Gemma 4 E2B exact Q4_0 on the GTX 1650 Ti, to pair
# with the thinking-off runs on this card (local-llamacpp-1650ti-qwen352b-q4_0/.../2026-10-16-quality-1650ti and
# local-llamacpp-1650ti-2b-q4_0/.../2026-10-17-quality-vs-qwen-1650ti). Same llama.cpp binary (7182ef7, sm_75),
# same GGUFs. Changed from those runs:
#   - --reasoning on (thinking in reasoning_content, scored on content only);
#   - each vendor's thinking-mode sampling, since greedy thinking loops on Qwen (16-problem trial):
#     Qwen card: temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 1.5, repetition_penalty 1.0;
#     Gemma generation_config.json: temperature 1.0, top_p 0.95, top_k 64;
#   - 32,000-token limit on both tasks (Qwen card: 32,768 for most queries), so -c 131072 --parallel 4 gives
#     each slot 32,768 tokens; concurrency 4;
#   - GSM8K is a fixed random 300 of 1,319 (gsm8k-300-ids.txt, random.Random(20261018)); BFCL is all 400.
# Sampled, so a re-run gives different text. Order: Gemma BFCL, Gemma GSM8K, Qwen BFCL, Qwen GSM8K.
set -u
ROOT=/home/xbill/qwen-dev
J=$ROOT/jev-tpu-v5e1
BIN=/home/xbill/llama.cpp/build/bin/llama-server
RUN=2026-10-18-quality-thinking-1650ti
QS='{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":1.5,"repeat_penalty":1.0}'
GS='{"temperature":1.0,"top_p":0.95,"top_k":64,"min_p":0.0}'
ARMS=(
  "e2b|google/gemma-4-E2B-it|/home/xbill/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf|$ROOT/local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/$RUN|$GS"
  "qwen|Qwen/Qwen3.5-2B|/home/xbill/models/Qwen3.5-2B-gguf/Qwen3.5-2B-Q4_0.gguf|$ROOT/local-llamacpp-1650ti-qwen352b-q4_0/benchmarks/runs/$RUN|$QS"
)
stop8080() { pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done; }
for arm in "${ARMS[@]}"; do
  IFS='|' read -r tag M GGUF OUT S <<<"$arm"
  LOG=$OUT/driver.log
  state() { echo "$(date +%T) gpu=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) $*" | tee -a "$LOG"; }
  state "BEGIN $tag $(sha256sum $GGUF | cut -c1-16) $($BIN --version 2>&1 | head -1)"
  sha256sum $GGUF > "$OUT/model-sha256.txt"
  echo "$S" > "$OUT/sampling.json"
  stop8080
  setsid $BIN -m $GGUF --host 127.0.0.1 --port 8080 -ngl 99 -c 131072 --parallel 4 -ctk f16 -ctv f16 -fa 1 \
    -t 6 -tb 12 --reasoning on --metrics > "$OUT/server-$tag.log" 2>&1 < /dev/null &
  for i in $(seq 180); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
  state "server up: $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader)"
  curl -fsS http://127.0.0.1:8080/metrics > "$OUT/metrics-before-$tag.prom"
  state "START bfcl"
  (cd $J && JEV_SAMPLING="$S" JEV_BFCL_MAX_TOKENS=32000 python3 gen_eval.py run http://127.0.0.1:8080 $M bfcl_simple \
     "$OUT/bfcl-$tag.jsonl" --concurrency 4) > "$OUT/bfcl-$tag.log" 2>&1
  state "END bfcl rc=$? $(tail -1 $OUT/bfcl-$tag.log)"
  state "START gsm8k"
  (cd $J && JEV_SAMPLING="$S" JEV_GEN_MAX_TOKENS=32000 python3 gen_eval.py run http://127.0.0.1:8080 $M gsm8k \
     "$OUT/gsm8k-$tag.jsonl" --ids "$OUT/gsm8k-300-ids.txt" --concurrency 4) > "$OUT/gsm8k-$tag.log" 2>&1
  state "END gsm8k rc=$? $(tail -1 $OUT/gsm8k-$tag.log)"
  curl -fsS http://127.0.0.1:8080/metrics > "$OUT/metrics-after-$tag.prom"
  stop8080
  state "DONE $tag"
done
