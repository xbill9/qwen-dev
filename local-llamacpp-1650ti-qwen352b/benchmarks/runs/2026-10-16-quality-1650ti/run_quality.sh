#!/bin/bash
# Answer quality, Qwen3.5-2B Q4_0 and bf16 on the GTX 1650 Ti: the 3050 runs
# (local-llamacpp-3050oem-qwen352b{-q4_0,}/benchmarks/runs/2026-10-16-quality-*-3050oem) with the card changed.
# Same server flags: -c 32768 --parallel 8, -fa 1, f16 KV, --reasoning off, no drafter. bf16 adds
# -ot for the last 6 of 24 blocks on the CPU, the fewest that fits this card (its first-light run).
#   1. Nimble 13-subset suite (3,880) via jev-tpu-v5e1/nimble_suite/run_suite.py through vllm_shim.py
#      on :8090, JEV_TOPK=32, concurrency 8, --tokenizer Qwen's.
#   2. GSM8K (1,319) via jev-tpu-v5e1/gen_eval.py, JEV_GEN_MAX_TOKENS=2048, concurrency 8.
#   3. BFCL simple (400) via gen_eval.py, concurrency 8.
# Usage: run_quality.sh            (both arms, Q4_0 first)
set -u
ROOT=/home/xbill/qwen-dev
J=$ROOT/jev-tpu-v5e1
DATA=/home/xbill/data/nimble-suite/public
BIN=/home/xbill/llama.cpp/build/bin/llama-server
M=Qwen/Qwen3.5-2B
D=/home/xbill/models/Qwen3.5-2B-gguf
ARMS=(
  "q4_0|$D/Qwen3.5-2B-Q4_0.gguf|$ROOT/local-llamacpp-1650ti-qwen352b-q4_0/benchmarks/runs/2026-10-16-quality-1650ti|"
  "bf16|$D/Qwen3.5-2B-bf16.gguf|$ROOT/local-llamacpp-1650ti-qwen352b/benchmarks/runs/2026-10-16-quality-1650ti|-ot blk\.(18|19|20|21|22|23)\.=CPU"
)
stop8080() { pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done; }
for arm in "${ARMS[@]}"; do
  IFS='|' read -r tag GGUF OUT EXTRA <<<"$arm"
  LOG=$OUT/driver.log
  RUN=2026-10-16-suite-1650ti-qwen352b-$tag
  state() { echo "$(date +%T) gpu=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) $*" | tee -a "$LOG"; }
  state "BEGIN $tag $(sha256sum $GGUF | cut -c1-16) $($BIN --version 2>&1 | head -1)"
  sha256sum $GGUF > "$OUT/model-sha256.txt"
  stop8080
  # shellcheck disable=SC2086
  setsid $BIN -m $GGUF --host 127.0.0.1 --port 8080 -ngl 99 $EXTRA -c 32768 --parallel 8 -ctk f16 -ctv f16 -fa 1 \
    -t 6 -tb 12 --reasoning off --metrics > "$OUT/server-qwen.log" 2>&1 < /dev/null &
  for i in $(seq 180); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
  state "server up: $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader)"
  curl -fsS http://127.0.0.1:8080/metrics > "$OUT/metrics-before-qwen.prom"
  python3 $OUT/vllm_shim.py --listen 8090 --upstream http://127.0.0.1:8080 > "$OUT/shim-qwen.log" 2>&1 & SH=$!
  sleep 1
  state "START suite"
  (cd $J && JEV_TOPK=32 python3 nimble_suite/run_suite.py --arm autoregressive --upstream http://127.0.0.1:8090 \
     --model $M --tokenizer $M --records $DATA --run $RUN --concurrency 8) > "$OUT/suite-qwen.log" 2>&1
  state "END suite rc=$?"
  kill $SH
  mv $J/results/$RUN $OUT/
  state "START gsm8k"
  (cd $J && JEV_GEN_MAX_TOKENS=2048 python3 gen_eval.py run http://127.0.0.1:8080 $M gsm8k "$OUT/gsm8k-qwen.jsonl" --concurrency 8) > "$OUT/gsm8k-qwen.log" 2>&1
  state "END gsm8k rc=$? $(tail -1 $OUT/gsm8k-qwen.log)"
  state "START bfcl"
  (cd $J && python3 gen_eval.py run http://127.0.0.1:8080 $M bfcl_simple "$OUT/bfcl-qwen.jsonl" --concurrency 8) > "$OUT/bfcl-qwen.log" 2>&1
  state "END bfcl rc=$? $(tail -1 $OUT/bfcl-qwen.log)"
  curl -fsS http://127.0.0.1:8080/metrics > "$OUT/metrics-after-qwen.prom"
  stop8080
  state "DONE $tag"
done
