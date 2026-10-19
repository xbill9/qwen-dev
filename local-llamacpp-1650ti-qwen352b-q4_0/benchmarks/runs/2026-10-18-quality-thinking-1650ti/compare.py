"""Thinking on against thinking off, Qwen3.5-2B Q4_0 and Gemma 4 E2B exact Q4_0 on the GTX 1650 Ti.

    python3 compare.py      # writes COMPARE.md and compare.json beside this file

Correctness is gen_eval.py's own `right` field. Pairs are gen_compare.compare (10,000 resamples, seed 0),
on the 300 GSM8K ids of gsm8k-300-ids.txt and all 400 BFCL records; the thinking-off records are cut to
the same ids. Token counts, cut-offs, BFCL failure reasons and the best case (every reply that hit the
limit counted right) are computed here from the records. Nothing is counted by hand.
"""

import collections
import json
import os
import statistics
import sys

J = "/home/xbill/qwen-dev/jev-tpu-v5e1"
sys.path[:0] = [J]
from gen_compare import compare as gpair  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = "/home/xbill/qwen-dev"
RUNS = {
    ("qwen", "on"): HERE,
    ("e2b", "on"): R + "/local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-18-quality-thinking-1650ti",
    ("qwen", "off"): R + "/local-llamacpp-1650ti-qwen352b-q4_0/benchmarks/runs/2026-10-16-quality-1650ti",
    ("e2b", "off"): R + "/local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-17-quality-vs-qwen-1650ti",
}
NAME = {"qwen": "Qwen3.5-2B Q4_0", "e2b": "Gemma 4 E2B QAT Q4_0"}
IDS = set(open(os.path.join(HERE, "gsm8k-300-ids.txt")).read().split())


def rows(model, think, task):
    f = os.path.join(RUNS[(model, think)], f"{'gsm8k' if task == 'gsm8k' else 'bfcl'}-{model}.jsonl")
    out = {}
    for line in open(f):
        r = json.loads(line)
        if task == "bfcl" or r["id"] in IDS:
            out[r["id"]] = r
    return out


def right(rs):
    return {i: bool(r["right"]) for i, r in rs.items()}


def describe(rs, task):
    toks = [r["completion_tokens"] for r in rs.values()]
    cut = [r for r in rs.values() if r.get("finish") == "length"]
    d = {
        "n": len(rs),
        "accuracy": sum(r["right"] for r in rs.values()) / len(rs),
        "tokens_median": statistics.median(toks),
        "tokens_mean": round(statistics.mean(toks)),
        "tokens_max": max(toks),
        "cut_off": len(cut),
        "cut_off_right": sum(r["right"] for r in cut),
        "best_case": (sum(r["right"] for r in rs.values()) + sum(not r["right"] for r in cut)) / len(rs),
        "seconds_sum": round(sum(r["seconds"] for r in rs.values())),
        "errors": sum("error" in r for r in rs.values()),
    }
    if task == "bfcl":
        d["no_call"] = sum(len(r.get("calls") or []) == 0 for r in rs.values())
        d["reasons"] = dict(collections.Counter(r["reason"].split(" ")[0] for r in rs.values() if not r["right"]))
    return d


def main():
    out = {"describe": {}, "pairs": {}}
    md = ["# Thinking on against thinking off — compare.py output", ""]
    for task in ("gsm8k", "bfcl"):
        data = {(m, t): rows(m, t, task) for m in ("qwen", "e2b") for t in ("on", "off")}
        label = "GSM8K, fixed random 300 of 1,319" if task == "gsm8k" else "BFCL simple, 400"
        md += [f"## {label}", "", "| model | thinking | n | accuracy | median tokens | mean | max | cut off (right) "
               "| best case | errors |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |"]
        for (m, t), rs in data.items():
            d = describe(rs, task)
            out["describe"][f"{task}/{m}/{t}"] = d
            md.append(f"| {NAME[m]} | {t} | {d['n']} | {d['accuracy']:.3f} | {d['tokens_median']:.0f} | "
                      f"{d['tokens_mean']} | {d['tokens_max']} | {d['cut_off']} ({d['cut_off_right']}) | "
                      f"{d['best_case']:.3f} | {d['errors']} |")
        if task == "bfcl":
            md += ["", "BFCL replies with no tool call, and failure reasons:", ""]
            for (m, t), rs in data.items():
                d = out["describe"][f"{task}/{m}/{t}"]
                md.append(f"- {NAME[m]}, thinking {t}: {d['no_call']} with no call; {d['reasons']}")
        md += ["", "| pair (test vs reference) | n | reference | test | test − ref | 95% range | ref only | test only |",
               "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |"]
        for key, ref, test in (
            ("qwen on vs off", ("qwen", "off"), ("qwen", "on")),
            ("e2b on vs off", ("e2b", "off"), ("e2b", "on")),
            ("qwen vs e2b, thinking off", ("e2b", "off"), ("qwen", "off")),
            ("qwen vs e2b, thinking on", ("e2b", "on"), ("qwen", "on")),
            ("qwen on vs e2b off", ("e2b", "off"), ("qwen", "on")),
        ):
            c = gpair(right(data[ref]), right(data[test]))
            out["pairs"][f"{task}/{key}"] = c
            md.append(f"| {key} | {c['n']} | {c['ref']:.3f} | {c['test']:.3f} | {c['diff']:+.3f} | "
                      f"{c['lo']:+.3f} to {c['hi']:+.3f} | {c['ref_only']} | {c['test_only']} |")
        md.append("")
    json.dump(out, open(os.path.join(HERE, "compare.json"), "w"), indent=1)
    open(os.path.join(HERE, "COMPARE.md"), "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
