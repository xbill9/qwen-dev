"""Counts quoted in the thinking follow-up that no other script prints.

    python3 checks.py > checks.txt

A reply "ends in a loop" when its last 200 characters occur 5 or more times in it, as in loops.py.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.dirname(HERE)
GEMMA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(RUNS))), "local-llamacpp-1650ti-2b-q4_0", "benchmarks", "runs")


def load(path):
    return [json.loads(line) for line in open(path)]


ids = set(open(os.path.join(HERE, "gsm8k-300-ids.txt")).read().split())
greedy = [r for r in load(os.path.join(RUNS, "2026-10-16-quality-1650ti", "gsm8k-qwen.jsonl")) if r["id"] in ids]
cut = [r for r in greedy if r["finish"] == "length"]
loop = sum(t.count(t[-200:]) >= 5 for t in ((r.get("reasoning") or "") + r["text"] for r in cut))
print(f"qwen gsm8k greedy-off 300: hit_limit={len(cut)} ends_in_loop={loop} no_loop={len(cut) - loop} "
      f"best_case={(sum(r['right'] for r in greedy) + sum(not r['right'] for r in cut)) / len(greedy):.3f}")

for name in ("trial-greedy-8k", "trial-sampled-32k"):
    rows = load(os.path.join(RUNS, "2026-10-18-quality-thinking-1650ti", name, "gsm8k-qwen.jsonl"))
    print(f"qwen gsm8k {name}: {sum(r['right'] for r in rows)} of {len(rows)} right")

for tag, path in (("sampled-off", os.path.join(GEMMA, "2026-10-19-quality-sampled-nothink-1650ti", "bfcl-e2b.jsonl")),
                  ("greedy-off", os.path.join(GEMMA, "2026-10-17-quality-vs-qwen-1650ti", "bfcl-e2b.jsonl"))):
    rows = load(path)
    th = [r for r in rows if (r.get("reasoning") or "").strip()]
    print(f"gemma bfcl {tag}, --reasoning off: {len(th)} of {len(rows)} with reasoning, {sum(r['right'] for r in th)} right")
