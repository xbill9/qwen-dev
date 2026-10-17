#!/bin/bash
# Answer quality, Gemma 4 E2B exact Q4_0 on the GTX 1650 Ti, to pair with the Qwen3.5-2B runs on this card
# (local-llamacpp-1650ti-qwen352b{-q4_0,}/benchmarks/runs/2026-10-16-quality-1650ti): same llama.cpp binary
# (7182ef7, sm_75), same flags (-c 32768 --parallel 8, -fa 1, f16 KV, --reasoning off, no drafter), same tasks.
# per_layer_token_embd stays on the host, as this rig serves (the 3050 run put it on CUDA0).
#   1. Nimble 13-subset suite (3,880) via jev-tpu-v5e1/nimble_suite/run_suite.py through vllm_shim.py
#      on :8090, JEV_TOPK=32, concurrency 8.
#   2. GSM8K (1,319) via jev-tpu-v5e1/gen_eval.py, JEV_GEN_MAX_TOKENS=2048, concurrency 8.
#   3. BFCL simple (400) via gen_eval.py, concurrency 8.
set -u
ROOT=/home/xbill/qwen-dev
J=$ROOT/jev-tpu-v5e1
DATA=/home/xbill/data/nimble-suite/public
BIN=/home/xbill/llama.cpp/build/bin/llama-server
M=google/gemma-4-E2B-it
GGUF=/home/xbill/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf
OUT=$ROOT/local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-17-quality-vs-qwen-1650ti
LOG=$OUT/driver.log
RUN=2026-10-17-suite-1650ti-e2b-q4exact
state() { echo "$(date +%T) gpu=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) $*" | tee -a "$LOG"; }
stop8080() { pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done; }
state "BEGIN e2b $(sha256sum $GGUF | cut -c1-16) $($BIN --version 2>&1 | head -1)"
sha256sum $GGUF > "$OUT/model-sha256.txt"
stop8080
setsid $BIN -m $GGUF --host 127.0.0.1 --port 8080 -ngl 99 -c 32768 --parallel 8 -ctk f16 -ctv f16 -fa 1 \
  -t 6 -tb 12 --reasoning off --metrics > "$OUT/server-e2b.log" 2>&1 < /dev/null &
for i in $(seq 180); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
state "server up: $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader)"
curl -fsS http://127.0.0.1:8080/metrics > "$OUT/metrics-before-e2b.prom"
python3 $OUT/vllm_shim.py --listen 8090 --upstream http://127.0.0.1:8080 > "$OUT/shim-e2b.log" 2>&1 & SH=$!
sleep 1
state "START suite"
(cd $J && JEV_TOPK=32 python3 nimble_suite/run_suite.py --arm autoregressive --upstream http://127.0.0.1:8090 \
   --model $M --records $DATA --run $RUN --concurrency 8) > "$OUT/suite-e2b.log" 2>&1
state "END suite rc=$?"
kill $SH
mv $J/results/$RUN $OUT/
state "START gsm8k"
(cd $J && JEV_GEN_MAX_TOKENS=2048 python3 gen_eval.py run http://127.0.0.1:8080 $M gsm8k "$OUT/gsm8k-e2b.jsonl" --concurrency 8) > "$OUT/gsm8k-e2b.log" 2>&1
state "END gsm8k rc=$? $(tail -1 $OUT/gsm8k-e2b.log)"
state "START bfcl"
(cd $J && python3 gen_eval.py run http://127.0.0.1:8080 $M bfcl_simple "$OUT/bfcl-e2b.jsonl" --concurrency 8) > "$OUT/bfcl-e2b.log" 2>&1
state "END bfcl rc=$? $(tail -1 $OUT/bfcl-e2b.log)"
curl -fsS http://127.0.0.1:8080/metrics > "$OUT/metrics-after-e2b.prom"
stop8080
state "DONE e2b"
