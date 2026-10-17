#!/bin/bash
# Does QAT move the needle? Gemma 4 E2B rounded to Q4_0 from Google's bf16 release (google/gemma-4-E2B-it,
# no QAT, no imatrix), the way Qwen3.5-2B's Q4_0 was built, scored on the GTX 1650 Ti with the QAT run's
# method (local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-17-quality-vs-qwen-1650ti): same binary
# (7182ef7, sm_75), same flags (-c 32768 --parallel 8, -fa 1, f16 KV, --reasoning off, no drafter), same tasks.
#   0. convert_hf_to_gguf.py --outtype bf16, then llama-quantize Q4_0 (no imatrix).
#   1. Nimble 13-subset suite (3,880) through vllm_shim.py on :8090, JEV_TOPK=32, concurrency 8.
#   2. GSM8K (1,319), JEV_GEN_MAX_TOKENS=2048, concurrency 8.
#   3. BFCL simple (400), concurrency 8.
set -u
ROOT=/home/xbill/qwen-dev
J=$ROOT/jev-tpu-v5e1
DATA=/home/xbill/data/nimble-suite/public
LC=/home/xbill/llama.cpp
BIN=$LC/build/bin/llama-server
M=google/gemma-4-E2B-it
HF=$(ls -d /home/xbill/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/*/ | head -1)
D=/home/xbill/models/gemma-4-E2B-it-gguf
BF16=$D/gemma-4-E2B-it-bf16.gguf
GGUF=$D/gemma-4-E2B-it-Q4_0.gguf
OUT=$ROOT/local-llamacpp-1650ti-2b-q4_0rtn/benchmarks/runs/2026-10-17-quality-rtn-1650ti
LOG=$OUT/driver.log
RUN=2026-10-17-suite-1650ti-e2b-q4_0rtn
state() { echo "$(date +%T) gpu=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) $*" | tee -a "$LOG"; }
stop8080() { pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done; }
mkdir -p $D
state "BEGIN convert $HF"
test -f $BF16 || python3 $LC/convert_hf_to_gguf.py $HF --outtype bf16 --outfile $BF16 > "$OUT/convert.log" 2>&1
state "END convert rc=$?"
test -f $GGUF || $LC/build/bin/llama-quantize $BF16 $GGUF Q4_0 > "$OUT/quantize.log" 2>&1
state "END quantize rc=$?"
sha256sum $BF16 $GGUF > "$OUT/model-sha256.txt"
(cd $LC && git rev-parse HEAD) > "$OUT/llama-commit.txt"
state "BEGIN e2b-rtn $(cut -c1-16 <<<"$(tail -1 $OUT/model-sha256.txt)") $($BIN --version 2>&1 | head -1)"
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
state "DONE e2b-rtn"
