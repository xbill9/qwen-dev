#!/bin/bash
# Does Qwen3.5-2B's own MTP head change its answers? GSM8K and BFCL simple on the GTX 1650 Ti with
# --spec-type draft-mtp --spec-draft-n-max 1, otherwise the 2026-10-16-quality-1650ti Q4_0 arm exactly
# (same file, binary 7182ef7, -c 32768 --parallel 8, -fa 1, f16 KV, --reasoning off). The suite is left out:
# it reads one token's probabilities, which drafting does not touch.
set -u
ROOT=/home/xbill/qwen-dev
J=$ROOT/jev-tpu-v5e1
BIN=/home/xbill/llama.cpp/build/bin/llama-server
M=Qwen/Qwen3.5-2B
GGUF=/home/xbill/models/Qwen3.5-2B-gguf/Qwen3.5-2B-Q4_0.gguf
OUT=$ROOT/local-llamacpp-1650ti-qwen352b-q4_0/benchmarks/runs/2026-10-17-quality-mtp-1650ti
LOG=$OUT/driver.log
state() { echo "$(date +%T) gpu=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) $*" | tee -a "$LOG"; }
stop8080() { pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done; }
state "BEGIN qwen-mtp $(sha256sum $GGUF | cut -c1-16) $($BIN --version 2>&1 | head -1)"
sha256sum $GGUF > "$OUT/model-sha256.txt"
stop8080
setsid $BIN -m $GGUF --host 127.0.0.1 --port 8080 -ngl 99 -c 32768 --parallel 8 -ctk f16 -ctv f16 -fa 1 \
  -t 6 -tb 12 --reasoning off --metrics --spec-type draft-mtp --spec-draft-n-max 1 \
  > "$OUT/server-qwen.log" 2>&1 < /dev/null &
for i in $(seq 180); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
state "server up: $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader)"
curl -fsS http://127.0.0.1:8080/metrics > "$OUT/metrics-before-qwen.prom"
state "START gsm8k"
(cd $J && JEV_GEN_MAX_TOKENS=2048 python3 gen_eval.py run http://127.0.0.1:8080 $M gsm8k "$OUT/gsm8k-qwen.jsonl" --concurrency 8) > "$OUT/gsm8k-qwen.log" 2>&1
state "END gsm8k rc=$? $(tail -1 $OUT/gsm8k-qwen.log)"
state "START bfcl"
(cd $J && python3 gen_eval.py run http://127.0.0.1:8080 $M bfcl_simple "$OUT/bfcl-qwen.jsonl" --concurrency 8) > "$OUT/bfcl-qwen.log" 2>&1
state "END bfcl rc=$? $(tail -1 $OUT/bfcl-qwen.log)"
curl -fsS http://127.0.0.1:8080/metrics > "$OUT/metrics-after-qwen.prom"
stop8080
state "DONE qwen-mtp"
