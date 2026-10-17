"""Qwen3.5-2B Q4_0 against Gemma 4 E2B / E4B exact Q4_0, answer quality on the RTX 3050 OEM.

    python3 compare.py      # writes COMPARE.md and compare.json beside this file

Suite correctness is the harness's own (score_suite.assess on score_suite.dist, gold_key) and the
paired difference with its 2,000-draw bootstrap range is quant_compare.paired. GSM8K and BFCL are
gen_eval.py's own `right` field, paired by gen_compare.compare (10,000 resamples, seed 0). BFCL's
failure reasons are counted from gen_eval.py's own `reason` field. Nothing is counted by hand.
"""

import collections
import glob
import json
import os
import sys

J = "/home/xbill/qwen-dev/jev-tpu-v5e1"
sys.path[:0] = [J, os.path.join(J, "nimble_suite")]
from gen_compare import compare as gpair, load as gload  # noqa: E402
from quant_compare import paired  # noqa: E402
from score_suite import assess, dist, gold_key  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
G = "/home/xbill/qwen-dev/local-llamacpp-3050oem-4b-q4_0/benchmarks/runs/2026-10-13-quality-e2b-vs-e4b-3050oem"
QW, E2, E4 = "Qwen3.5-2B Q4_0", "Gemma E2B Q4", "Gemma E4B Q4"
SUITE = {
    QW: HERE + "/2026-10-16-suite-3050oem-qwen352b-q4_0",
    E2: G + "/2026-10-13-suite-3050oem-e2b-q4exact",
    E4: G + "/2026-10-13-suite-3050oem-e4b-q4exact",
}
GEN = {QW: HERE + "/{t}-qwen.jsonl", E2: G + "/{t}-e2b.jsonl", E4: G + "/{t}-e4b.jsonl"}
PAIRS = [(E2, QW), (E4, QW)]


def suite_load(d):
    out = {}
    for f in glob.glob(os.path.join(d, "*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if "reads" in r:
                full = r["reads"][0].get("labels_returned")
                out[(r["subset"], r["id"])] = (
                    assess(dist(r, "ar"), gold_key(r))["correct"],
                    None if full is None else full == len(r["keys"]),
                )
    return out


res = {"suite": {}, "suite_pairs": [], "subsets": {}, "gen": {}, "bfcl_reasons": {}}
md = ["# Answer quality: Qwen3.5-2B Q4_0 against Gemma 4 E2B / E4B exact Q4_0, RTX 3050 OEM", ""]

S = {k: suite_load(v) for k, v in SUITE.items()}
md += ["## Nimble 13-subset suite", "", "| run | records | accuracy | all labels returned |", "|---|---:|---:|---:|"]
for k, c in S.items():
    cov = [f for _, f in c.values()]
    cv = None if any(f is None for f in cov) else sum(cov) / len(cov)
    acc = sum(ok for ok, _ in c.values()) / len(c)
    res["suite"][k] = {"n": len(c), "acc": acc, "coverage": cv}
    md.append(f"| {k} | {len(c)} | {acc:.4f} | {'-' if cv is None else f'{cv:.1%}'} |")
md += [
    "",
    "Paired on the same records (difference = B - A; 95% range from 2,000 bootstrap draws):",
    "",
    "| A | B | n | A | B | difference | 95% range | A right, B wrong | A wrong, B right |",
    "|---|---|---:|---:|---:|---:|---|---:|---:|",
]
for a, b in PAIRS:
    s = paired(S[a], S[b])
    res["suite_pairs"].append({"a": a, "b": b, **s})
    md.append(
        f"| {a} | {b} | {s['n']} | {s['acc_ref']:.3f} | {s['acc_test']:.3f} | **{s['diff']:+.3f}** | "
        f"{s['diff_lo']:+.3f} to {s['diff_hi']:+.3f} | {s['ref_right_test_wrong']} | {s['ref_wrong_test_right']} |"
    )
md += [
    "",
    "Per subset, Qwen - Gemma E2B (secondary; ranges not corrected for 13 comparisons):",
    "",
    "| subset | n | E2B | Qwen | Qwen - E2B | 95% range |",
    "|---|---:|---:|---:|---:|---|",
]
for sub in sorted({s for s, _ in S[E2]}):
    g = {k: {i: v for i, v in S[k].items() if i[0] == sub} for k in (E2, QW)}
    s = paired(g[E2], g[QW])
    res["subsets"][sub] = s
    md.append(
        f"| {sub} | {s['n']} | {s['acc_ref']:.3f} | {s['acc_test']:.3f} | {s['diff']:+.3f} | "
        f"{s['diff_lo']:+.3f} to {s['diff_hi']:+.3f} |"
    )

md += [
    "",
    "## GSM8K (2,048-token limit) and BFCL simple",
    "",
    "| task | A | B | n | A | B | difference | 95% range | A only | B only |",
    "|---|---|---|---:|---:|---:|---:|---|---:|---:|",
]
for t, local in (("gsm8k", "gsm8k"), ("bfcl_simple", "bfcl")):
    for a, b in PAIRS:
        s = gpair(gload(GEN[a].format(t=local)), gload(GEN[b].format(t=local)))
        res["gen"].setdefault(t, []).append({"a": a, "b": b, **s})
        md.append(
            f"| {t} | {a} | {b} | {s['n']} | {s['ref']:.3f} | {s['test']:.3f} | **{s['diff']:+.3f}** | "
            f"{s['lo']:+.3f} to {s['hi']:+.3f} | {s['ref_only']} | {s['test_only']} |"
        )

md += [
    "",
    "BFCL failure reasons (gen_eval.py's `reason`, wrong answers only):",
    "",
    "| run | wrong | reasons |",
    "|---|---:|---|",
]
for k in (QW, E2, E4):
    rows = [json.loads(line) for line in open(GEN[k].format(t="bfcl"))]
    c = collections.Counter(("error" if r.get("error") else r.get("reason", "?")) for r in rows if not r["right"])
    res["bfcl_reasons"][k] = dict(c)
    md.append(f"| {k} | {sum(c.values())} | " + "; ".join(f"{n} × {r}" for r, n in c.most_common(6)) + " |")

open(os.path.join(HERE, "COMPARE.md"), "w").write("\n".join(md) + "\n")
json.dump(res, open(os.path.join(HERE, "compare.json"), "w"), indent=1)
print("\n".join(md))
