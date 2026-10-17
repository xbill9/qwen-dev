#!/bin/bash
# Answer quality, Qwen3.5-2B bf16 GGUF (the unquantized reference, converted by llama.cpp 7182ef71e)
# on the RTX 3050 OEM: the Q4_0 run of local-llamacpp-3050oem-qwen352b-q4_0 with only the file changed.


# local-llamacpp-3050oem-4b-q4_0/benchmarks/runs/2026-10-13-quality-e2b-vs-e4b-3050oem.
# Same server flags as that run less Gemma's -ot (no counterpart here), no drafter:
# -c 32768 --parallel 8, -fa 1, f16 KV, --reasoning off.
#   1. Nimble 13-subset suite (3,880 records) via jev-tpu-v5e1/nimble_suite/run_suite.py
#      through vllm_shim.py on :8090, JEV_TOPK=32, concurrency 8. --tokenizer is Qwen's, so
#      the answer lead and label ids are in Qwen's vocabulary (Gemma's runs used a Gemma 4 one).
#   2. GSM8K (1,319) via jev-tpu-v5e1/gen_eval.py, JEV_GEN_MAX_TOKENS=2048, concurrency 8.
#   3. BFCL simple (400) via gen_eval.py, concurrency 8.
set -u
ROOT=/home/xbill/qwen-dev
J=$ROOT/jev-tpu-v5e1
OUT=/home/xbill/qwen-dev/local-llamacpp-3050oem-qwen352b/benchmarks/runs/2026-10-16-quality-bf16-3050oem
LOG=$OUT/driver.log
DATA=/home/xbill/data/nimble-suite/public
BIN=/home/xbill/llama.cpp/build/bin/llama-server
GGUF=/home/xbill/models/Qwen3.5-2B-gguf/Qwen3.5-2B-bf16.gguf
M=Qwen/Qwen3.5-2B
RUN=2026-10-16-suite-3050oem-qwen352b-bf16
state() { echo "$(date +%T) gpu=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) $*" | tee -a "$LOG"; }
stop8080() { pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done; }
state "BEGIN $(sha256sum $GGUF | cut -c1-16) $($BIN --version 2>&1 | head -1)"
stop8080
setsid $BIN -m $GGUF --host 127.0.0.1 --port 8080 -ngl 99 -c 32768 --parallel 8 -ctk f16 -ctv f16 -fa 1 \
  -t 6 -tb 12 --reasoning off --metrics > "$OUT/server-qwen.log" 2>&1 < /dev/null &
for i in $(seq 120); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
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
state "DONE"
