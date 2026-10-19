"""Split the thinking-on gain into sampling and thinking, Qwen3.5-2B Q4_0 and Gemma 4 E2B on the GTX 1650 Ti.

    python3 compare.py      # writes COMPARE.md and compare.json beside this file

Three arms per model, all on the 300 GSM8K ids of gsm8k-300-ids.txt and all 400 BFCL records:
greedy thinking off (2026-10-16 / 2026-10-17 runs), vendor sampling thinking off (this run), vendor sampling
thinking on (2026-10-18). Correctness is gen_eval.py's own `right`; pairs are gen_compare.compare
(10,000 resamples, seed 0). Nothing is counted by hand.
"""

import json
import os
import statistics
import sys

sys.path[:0] = ["/home/xbill/qwen-dev/jev-tpu-v5e1"]
from gen_compare import compare as gpair  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = "/home/xbill/qwen-dev"
RUNS = {
    ("qwen", "greedy-off"): R + "/local-llamacpp-1650ti-qwen352b-q4_0/benchmarks/runs/2026-10-16-quality-1650ti",
    ("qwen", "sampled-off"): HERE,
    ("qwen", "sampled-on"): R + "/local-llamacpp-1650ti-qwen352b-q4_0/benchmarks/runs/2026-10-18-quality-thinking-1650ti",
    ("e2b", "greedy-off"): R + "/local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-17-quality-vs-qwen-1650ti",
    ("e2b", "sampled-off"): R + "/local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-19-quality-sampled-nothink-1650ti",
    ("e2b", "sampled-on"): R + "/local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-18-quality-thinking-1650ti",
}
NAME = {"qwen": "Qwen3.5-2B Q4_0", "e2b": "Gemma 4 E2B QAT Q4_0"}
IDS = set(open(os.path.join(HERE, "gsm8k-300-ids.txt")).read().split())


def rows(model, arm, task):
    out = {}
    for line in open(os.path.join(RUNS[(model, arm)], f"{task}-{model}.jsonl")):
        r = json.loads(line)
        if task == "bfcl" or r["id"] in IDS:
            out[r["id"]] = r
    return out


def right(rs):
    return {i: bool(r["right"]) for i, r in rs.items()}


def main():
    out = {"describe": {}, "pairs": {}}
    md = ["# Sampling against thinking — compare.py output", ""]
    for task in ("gsm8k", "bfcl"):
        md += [f"## {'GSM8K, fixed random 300' if task == 'gsm8k' else 'BFCL simple, 400'}", "",
               "| model | arm | n | accuracy | median tokens | hit limit | no tool call |",
               "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        data = {}
        for m in ("qwen", "e2b"):
            for arm in ("greedy-off", "sampled-off", "sampled-on"):
                rs = data[(m, arm)] = rows(m, arm, task)
                d = {
                    "n": len(rs),
                    "accuracy": sum(r["right"] for r in rs.values()) / len(rs),
                    "tokens_median": statistics.median(r["completion_tokens"] for r in rs.values()),
                    "hit_limit": sum(r.get("finish") == "length" for r in rs.values()),
                    "no_call": sum(len(r.get("calls") or []) == 0 for r in rs.values()) if task == "bfcl" else None,
                }
                out["describe"][f"{task}/{m}/{arm}"] = d
                md.append(f"| {NAME[m]} | {arm} | {d['n']} | {d['accuracy']:.3f} | {d['tokens_median']:.0f} | "
                          f"{d['hit_limit']} | {'' if d['no_call'] is None else d['no_call']} |")
        md += ["", "| pair (test vs reference) | n | reference | test | test − ref | 95% range | ref only | test only |",
               "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |"]
        for m in ("qwen", "e2b"):
            for key, ref, test in (
                ("sampling, thinking off", "greedy-off", "sampled-off"),
                ("thinking, at vendor sampling", "sampled-off", "sampled-on"),
                ("both", "greedy-off", "sampled-on"),
            ):
                c = gpair(right(data[(m, ref)]), right(data[(m, test)]))
                out["pairs"][f"{task}/{m}/{key}"] = c
                md.append(f"| {m}: {key} | {c['n']} | {c['ref']:.3f} | {c['test']:.3f} | {c['diff']:+.3f} | "
                          f"{c['lo']:+.3f} to {c['hi']:+.3f} | {c['ref_only']} | {c['test_only']} |")
        for arm in ("greedy-off", "sampled-off", "sampled-on"):
            c = gpair(right(data[("e2b", arm)]), right(data[("qwen", arm)]))
            out["pairs"][f"{task}/qwen-vs-e2b/{arm}"] = c
            md.append(f"| qwen vs e2b, {arm} | {c['n']} | {c['ref']:.3f} | {c['test']:.3f} | {c['diff']:+.3f} | "
                      f"{c['lo']:+.3f} to {c['hi']:+.3f} | {c['ref_only']} | {c['test_only']} |")
        md.append("")
    json.dump(out, open(os.path.join(HERE, "compare.json"), "w"), indent=1)
    open(os.path.join(HERE, "COMPARE.md"), "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
