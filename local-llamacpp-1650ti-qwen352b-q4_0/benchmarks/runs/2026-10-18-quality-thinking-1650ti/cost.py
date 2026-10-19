"""Tokens and wall time per right answer, thinking on, from the records and driver logs.

    python3 cost.py > cost.txt

Wall time is gen_eval.py's own wall_s (4 requests at a time). Tokens are the server's completion_tokens.
"""

import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
G = "/home/xbill/qwen-dev/local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-18-quality-thinking-1650ti"
for name, d, tag in (("qwen", HERE, "qwen"), ("gemma", G, "e2b")):
    for task in ("gsm8k", "bfcl"):
        rows = [json.loads(line) for line in open(os.path.join(d, f"{task}-{tag}.jsonl"))]
        wall = json.loads(open(os.path.join(d, f"{task}-{tag}.log")).read().strip().splitlines()[-1])["wall_s"]
        right = sum(r["right"] for r in rows)
        toks = [r["completion_tokens"] for r in rows]
        print(f"{name} {task}: n={len(rows)} right={right} wall_s={wall} wall_min={wall / 60:.1f} wall_h={wall / 3600:.2f} "
              f"s_per_right={wall / right:.1f} tokens_total={sum(toks)} tokens_per_right={sum(toks) / right:.0f} "
              f"tokens_median={statistics.median(toks):.0f}")
