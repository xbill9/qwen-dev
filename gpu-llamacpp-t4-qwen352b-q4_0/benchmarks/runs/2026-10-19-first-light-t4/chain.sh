#!/bin/bash
# The 2026-10-19 T4 runs one after another, so only one llama-server holds the card at a time:
#   0. llama-quantize Qwen3.5-2B bf16 -> Q4_0 with this binary (the rig's `make quantize`),
#   1. this directory's bench.sh (llama-bench, Qwen and Gemma E2B),
#   2. ../2026-10-19-spec-mtp-vs-gemma-t4/spec_ab.py (drafter speed, both models),
#   3. ../2026-10-19-quality-t4/run_quality.sh (Qwen Q4_0, Gemma without and with its drafter),
#   4. vLLM on the same card, Gemma E2B QAT with and without its QAT drafter
#      (~/gemma4-dev/gpu-vllm-t4-2b-w4a16/benchmarks/runs/2026-10-19-vllm-vs-llamacpp-t4/run.sh),
#   5. restart the vLLM server this VM normally serves (~/bin/vllm-t4), stopped for these runs.
set -u
# FROM=quality resumes at step 3 (the bench and spec_ab results already on disk are kept).
FROM=${FROM:-start}
B=/opt1/llama.cpp/build/bin
H=$(cd "$(dirname "$0")" && pwd)
RUNS=$(dirname "$H")
D=/opt1/models/Qwen3.5-2B-gguf
log() { echo "$(date +%F\ %T) $*" | tee -a "$H/chain.log"; }
log "START $($B/llama-server --version 2>&1 | grep version)"
if [ ! -f $D/Qwen3.5-2B-Q4_0.gguf ]; then
  $B/llama-quantize $D/Qwen3.5-2B-bf16.gguf $D/Qwen3.5-2B-Q4_0.gguf Q4_0 > "$H/quantize.log" 2>&1
  log "quantize rc=$?"
fi
sha256sum $D/Qwen3.5-2B-bf16.gguf $D/Qwen3.5-2B-Q4_0.gguf \
  /opt1/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf \
  /opt1/models/gemma-4-E2B-it-qat-assistant/gemma-4-E2B-it-qat-assistant-q4_0-exact.gguf > "$H/model-sha256.txt"
$B/llama-server --version > "$H/llama-version.txt" 2>&1
nvidia-smi -q > "$H/gpu-static.txt"
if [ "$FROM" = start ]; then
log "START bench"; bash "$H/bench.sh" "$H/results" > "$H/bench.out" 2>&1; log "END bench rc=$?"
python3 "$H/summarize.py" > /dev/null
S=$RUNS/2026-10-19-spec-mtp-vs-gemma-t4
log "START spec_ab"; python3 $S/spec_ab.py > $S/run.log 2>&1; log "END spec_ab rc=$?"
python3 $S/summarize.py > /dev/null
fi
log "START quality"; bash $RUNS/2026-10-19-quality-t4/run_quality.sh > $RUNS/2026-10-19-quality-t4/driver.out 2>&1
log "END quality rc=$?"
V=$HOME/gemma4-dev/gpu-vllm-t4-2b-w4a16/benchmarks/runs/2026-10-19-vllm-vs-llamacpp-t4
log "START vllm"; bash $V/run.sh > $V/run.out 2>&1; log "END vllm rc=$?"
log "restart vLLM: $(~/bin/vllm-t4 start nowait 2>&1 | tail -1)"
log "CHAIN DONE"
