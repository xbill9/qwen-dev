#!/bin/bash
# Runs the 2026-10-17 follow-ups one after another once the Gemma E2B QAT quality run has finished, so only
# one llama-server holds the card at a time:
#   1. this directory's spec_ab.py (MTP / drafter speed, Qwen and Gemma),
#   2. local-llamacpp-1650ti-2b-q4_0rtn/.../2026-10-17-quality-rtn-1650ti (Gemma without QAT),
#   3. .../2026-10-17-quality-mtp-1650ti (Qwen answers with its MTP head on).
set -u
R=/home/xbill/qwen-dev
G=$R/local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-17-quality-vs-qwen-1650ti/driver.log
S=$R/local-llamacpp-1650ti-qwen352b-q4_0/benchmarks/runs/2026-10-17-spec-mtp-vs-gemma-1650ti
until grep -q "DONE e2b" $G; do sleep 30; done
echo "$(date +%T) START spec_ab"
python3 $S/spec_ab.py > $S/run.log 2>&1
echo "$(date +%T) END spec_ab rc=$?"
bash $R/local-llamacpp-1650ti-2b-q4_0rtn/benchmarks/runs/2026-10-17-quality-rtn-1650ti/run_quality.sh
bash $R/local-llamacpp-1650ti-qwen352b-q4_0/benchmarks/runs/2026-10-17-quality-mtp-1650ti/run_quality.sh
echo "$(date +%T) CHAIN DONE"
