"""Qwen3.5-2B Q4_0 on the GTX 1650 Ti with its MTP head drafting 1 token, against the same file without drafting.

    python3 compare.py      # writes COMPARE.md and compare.json beside this file

GSM8K and BFCL are gen_eval.py's own `right` field, paired by gen_compare.compare (10,000 resamples, seed 0).
Reply identity, length-limit hits, total reply tokens and wall time are counted here from the jsonl records and
the driver logs. Drafting acceptance comes from the server's /metrics counters before and after.
"""

import json
import os
import re
import sys

J = "/home/xbill/qwen-dev/jev-tpu-v5e1"
sys.path[:0] = [J]
from gen_compare import compare as gpair, load as gload  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "/home/xbill/qwen-dev/local-llamacpp-1650ti-qwen352b-q4_0/benchmarks/runs/2026-10-16-quality-1650ti"
ARMS = {"no draft": BASE, "MTP, 1 token": HERE}


def rows(d, t):
    return {r["id"]: r for r in (json.loads(line) for line in open(os.path.join(d, f"{t}-qwen.jsonl")))}


def wall(d, t):
    for line in open(os.path.join(d, "driver.log")):
        m = re.search(r'"task": "%s".*"wall_s": ([0-9.]+)' % t, line)
        if m:
            return float(m.group(1))
    return None


def metrics(d):
    out = {}
    for side in ("before", "after"):
        p = os.path.join(d, f"metrics-{side}-qwen.prom")
        for line in open(p):
            if line.startswith("llamacpp:") and " " in line:
                k, v = line.split()[:2]
                out.setdefault(k, {})[side] = float(v)
    return {k: v.get("after", 0) - v.get("before", 0) for k, v in out.items()}


res = {"gen": {}, "identical": {}, "truncated": {}, "tokens": {}, "wall_s": {}, "metrics": metrics(HERE)}
md = ["# Qwen3.5-2B Q4_0 on the GTX 1650 Ti: answers with and without its MTP head drafting", "",
      "| task | no draft | MTP, 1 token | difference | 95% range | no draft only | MTP only | same reply text |",
      "|---|---:|---:|---:|---|---:|---:|---:|"]
for t, local in (("gsm8k", "gsm8k"), ("bfcl_simple", "bfcl")):
    s = gpair(gload(os.path.join(BASE, f"{local}-qwen.jsonl")), gload(os.path.join(HERE, f"{local}-qwen.jsonl")))
    res["gen"][t] = s
    a, b = rows(BASE, local), rows(HERE, local)
    same = sum(1 for i in a if i in b and a[i].get("text") == b[i].get("text"))
    res["identical"][t] = {"same": same, "n": len(a)}
    md.append(f"| {t} | {s['ref']:.3f} | {s['test']:.3f} | **{s['diff']:+.3f}** | {s['lo']:+.3f} to {s['hi']:+.3f} | "
              f"{s['ref_only']} | {s['test_only']} | {same} of {len(a)} |")
    for k, d in ARMS.items():
        r = rows(d, local)
        res["truncated"].setdefault(t, {})[k] = sum(1 for x in r.values() if x.get("finish") == "length")
        res["tokens"].setdefault(t, {})[k] = sum(x["completion_tokens"] for x in r.values())
        res["wall_s"].setdefault(t, {})[k] = wall(d, t)

md += ["", "| task | arm | replies at the length limit | reply tokens | wall time, min | tokens per second, all 8 slots |",
       "|---|---|---:|---:|---:|---:|"]
for t in res["tokens"]:
    for k in ARMS:
        w = res["wall_s"][t][k]
        md.append(f"| {t} | {k} | {res['truncated'][t][k]} | {res['tokens'][t][k]} | {w / 60:.1f} | "
                  f"{res['tokens'][t][k] / w:.1f} |")
wt = {t: res["wall_s"][t]["MTP, 1 token"] / res["wall_s"][t]["no draft"] for t in res["wall_s"]}
res["wall_ratio"] = wt
md += ["", "Wall time, MTP / no draft: " + ", ".join(f"{t} {x:.3f}x" for t, x in wt.items())]
m = res["metrics"]
dn = m.get("llamacpp:spec_decode_num_draft_tokens_total")
da = m.get("llamacpp:spec_decode_num_accepted_tokens_total")
if dn:
    res["acceptance"] = da / dn
    md.append(f"Drafted tokens accepted over the whole run (/metrics): {da:.0f} of {dn:.0f}, {da / dn:.3f}")
else:
    md.append("No drafting counters in /metrics: " + ", ".join(sorted(k for k in m if "draft" in k)))

open(os.path.join(HERE, "COMPARE.md"), "w").write("\n".join(md) + "\n")
json.dump(res, open(os.path.join(HERE, "compare.json"), "w"), indent=1)
print("\n".join(md))
