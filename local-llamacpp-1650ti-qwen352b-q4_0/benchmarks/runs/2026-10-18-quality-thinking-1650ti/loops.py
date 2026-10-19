"""Count Qwen replies that hit the token limit, and how many of those end in a loop.

    python3 loops.py > loops.txt

A reply "ends in a loop" when its last 200 characters (thinking plus answer) occur 5 or more times in it.
Reads the greedy 8k trial, the sampled 32k trial and the full thinking-on run.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
for f in ("trial-greedy-8k/gsm8k-qwen.jsonl", "trial-sampled-32k/gsm8k-qwen.jsonl", "gsm8k-qwen.jsonl"):
    rows = [json.loads(line) for line in open(os.path.join(HERE, f))]
    cut = [r for r in rows if r.get("finish") == "length"]
    loop = still = 0
    for r in cut:
        t = (r.get("reasoning") or "") + (r.get("text") or "")
        loop += t.count(t[-200:]) >= 5
        still += not (r.get("text") or "").strip()
    print(f"{f}: n={len(rows)} limit={rows[0]['max_tokens']} right={sum(r['right'] for r in rows)} "
          f"hit_limit={len(cut)} ends_in_loop={loop} still_thinking={still}")
