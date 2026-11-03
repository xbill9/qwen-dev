#!/bin/bash
# Answer quality on the Tesla T4: Qwen3.5-2B Q4_0, and Gemma 4 E2B exact QAT Q4_0 with and without its QAT
# assistant drafter, all on one llama.cpp binary (7182ef71e, sm_75). The 1650 Ti runs' protocol
# (local-llamacpp-1650ti-qwen352b-q4_0/benchmarks/runs/2026-10-16-quality-1650ti,
#  local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-17-quality-vs-qwen-1650ti) with the card changed:
# -c 32768 --parallel 8, -fa 1, f16 KV, --reasoning off; -t 2 -tb 2 for this 2-vCPU VM. Gemma holds
# per_layer_token_embd on the card, as the 3050 run did. -cram 2048: llama-server's host prompt cache
# defaults to 8,192 MiB, more than this VM's 7.3 GiB of RAM; at the default the Qwen arm drove the VM into
# swap within two minutes (each Qwen cache entry is ~41 MiB). The prompt cache changes no score
# (local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-14-accuracy-cache-3050oem).
#   1. Nimble 13-subset suite (3,880) via jev-tpu-v5e1/nimble_suite/run_suite.py through vllm_shim.py
#      on :8090, JEV_TOPK=32, concurrency 8; --tokenizer Qwen's for Qwen, the harness default for Gemma,
#      as on the 1650 Ti.
#   2. GSM8K (1,319) via jev-tpu-v5e1/gen_eval.py, JEV_GEN_MAX_TOKENS=2048, concurrency 8.
#   3. BFCL simple (400) via gen_eval.py, concurrency 8.
# The drafted Gemma arm takes its draft length from the speed run beside this one
# (../2026-10-19-spec-mtp-vs-gemma-t4/summary.json, Gemma's fastest greedy arm), or DRAFT_N.
# Usage: run_quality.sh [arm ...]     arms: qwen-q4_0 gemma-none gemma-draft (default: all three)
set -u
ROOT=/home/xbill_glitnir_com/qwen-dev
J=$ROOT/jev-tpu-v5e1
DATA=/opt1/data/nimble-suite/public
BIN=/opt1/llama.cpp/build/bin/llama-server
OUT=$(cd "$(dirname "$0")" && pwd)
LOG=$OUT/driver.log
QWEN=/opt1/models/Qwen3.5-2B-gguf/Qwen3.5-2B-Q4_0.gguf
GEMMA=/opt1/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf
DRAFT=/opt1/models/gemma-4-E2B-it-qat-assistant/gemma-4-E2B-it-qat-assistant-q4_0-exact.gguf
PLE='-ot per_layer_token_embd\.weight=CUDA0'
if [ -z "${DRAFT_N:-}" ]; then
  DRAFT_N=$(python3 -c "
import json; g=json.load(open('$OUT/../2026-10-19-spec-mtp-vs-gemma-t4/summary.json'))['geomean_tps']
print(max((k for k in g if k.startswith('gemma/greedy/draft-n')), key=g.get).rsplit('n', 1)[1])")
fi
declare -A MODEL GGUF FLAGS TOK
MODEL[qwen-q4_0]=Qwen/Qwen3.5-2B;      GGUF[qwen-q4_0]=$QWEN;  FLAGS[qwen-q4_0]=""; TOK[qwen-q4_0]="--tokenizer Qwen/Qwen3.5-2B"
MODEL[gemma-none]=google/gemma-4-E2B-it; GGUF[gemma-none]=$GEMMA; FLAGS[gemma-none]="$PLE"
MODEL[gemma-draft]=google/gemma-4-E2B-it; GGUF[gemma-draft]=$GEMMA
FLAGS[gemma-draft]="$PLE -md $DRAFT --spec-type draft-mtp -ngld all --spec-draft-n-max $DRAFT_N"
ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=(qwen-q4_0 gemma-none gemma-draft)
state() { echo "$(date +%F\ %T) gpu=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader) $*" | tee -a "$LOG"; }
stop8080() { pkill -x llama-server; while pgrep -x llama-server >/dev/null; do sleep 1; done; }
for tag in "${ARMS[@]}"; do
  M=${MODEL[$tag]}; G=${GGUF[$tag]}; RUN=2026-10-19-suite-t4-$tag
  state "BEGIN $tag $(sha256sum $G | cut -c1-16) flags=[${FLAGS[$tag]}] $($BIN --version 2>&1 | grep version)"
  sha256sum $G > "$OUT/model-sha256-$tag.txt"
  [ "$tag" = gemma-draft ] && sha256sum $DRAFT >> "$OUT/model-sha256-$tag.txt"
  stop8080
  # shellcheck disable=SC2086
  setsid $BIN -m $G --host 127.0.0.1 --port 8080 -ngl 99 ${FLAGS[$tag]} -c 32768 --parallel 8 -ctk f16 -ctv f16 \
    -fa 1 -t 2 -tb 2 -cram 2048 --reasoning off --metrics > "$OUT/server-$tag.log" 2>&1 < /dev/null &
  for i in $(seq 300); do curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
  state "server up: $(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader)"
  curl -fsS http://127.0.0.1:8080/metrics > "$OUT/metrics-before-$tag.prom"
  python3 $OUT/vllm_shim.py --listen 8090 --upstream http://127.0.0.1:8080 > "$OUT/shim-$tag.log" 2>&1 & SH=$!
  sleep 1
  state "START suite"
  (cd $J && JEV_TOPK=32 python3 nimble_suite/run_suite.py --arm autoregressive --upstream http://127.0.0.1:8090 \
     --model $M ${TOK[$tag]:-} --records $DATA --run $RUN --concurrency 8) > "$OUT/suite-$tag.log" 2>&1
  state "END suite rc=$?"
  kill $SH
  rm -rf "$OUT/$RUN"; mv $J/results/$RUN $OUT/
  state "START gsm8k"
  (cd $J && JEV_GEN_MAX_TOKENS=2048 python3 gen_eval.py run http://127.0.0.1:8080 $M gsm8k "$OUT/gsm8k-$tag.jsonl" --concurrency 8) > "$OUT/gsm8k-$tag.log" 2>&1
  state "END gsm8k rc=$? $(tail -1 $OUT/gsm8k-$tag.log)"
  state "START bfcl"
  (cd $J && python3 gen_eval.py run http://127.0.0.1:8080 $M bfcl_simple "$OUT/bfcl-$tag.jsonl" --concurrency 8) > "$OUT/bfcl-$tag.log" 2>&1
  state "END bfcl rc=$? $(tail -1 $OUT/bfcl-$tag.log)"
  curl -fsS http://127.0.0.1:8080/metrics > "$OUT/metrics-after-$tag.prom"
  stop8080
  state "DONE $tag"
done
