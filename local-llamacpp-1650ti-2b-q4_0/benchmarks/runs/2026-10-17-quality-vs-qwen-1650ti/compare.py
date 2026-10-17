"""Gemma 4 E2B exact Q4_0 on the GTX 1650 Ti, against its 3050 run and Qwen3.5-2B Q4_0 and bf16 on this card.

    python3 compare.py      # writes COMPARE.md and compare.json beside this file

The Qwen 1650 Ti run's compare.py with the arms changed. Suite correctness is the harness's own
(score_suite.assess on score_suite.dist, gold_key) and the paired difference with its 2,000-draw bootstrap
range is quant_compare.paired. GSM8K and BFCL are gen_eval.py's own `right` field, paired by
gen_compare.compare (10,000 resamples, seed 0). BFCL's failure reasons are counted from gen_eval.py's own
`reason` field. The Qwen-to-Gemma gap on each card, and how much it moves between cards, are computed here.
Nothing is counted by hand.
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
R = "/home/xbill/qwen-dev"
QW1 = R + "/local-llamacpp-1650ti-qwen352b-q4_0/benchmarks/runs/2026-10-16-quality-1650ti"
QB1 = R + "/local-llamacpp-1650ti-qwen352b/benchmarks/runs/2026-10-16-quality-1650ti"
Q3 = R + "/local-llamacpp-3050oem-qwen352b-q4_0/benchmarks/runs/2026-10-16-quality-vs-gemma-3050oem"
B3 = R + "/local-llamacpp-3050oem-qwen352b/benchmarks/runs/2026-10-16-quality-bf16-3050oem"
G3 = R + "/local-llamacpp-3050oem-4b-q4_0/benchmarks/runs/2026-10-13-quality-e2b-vs-e4b-3050oem"
E1, E3 = "Gemma E2B Q4, 1650 Ti", "Gemma E2B Q4, 3050"
QW, QB = "Qwen Q4_0, 1650 Ti", "Qwen bf16, 1650 Ti"
QW3, QB3 = "Qwen Q4_0, 3050", "Qwen bf16, 3050"
SUITE = {
    E1: HERE + "/2026-10-17-suite-1650ti-e2b-q4exact",
    E3: G3 + "/2026-10-13-suite-3050oem-e2b-q4exact",
    QW: QW1 + "/2026-10-16-suite-1650ti-qwen352b-q4_0",
    QB: QB1 + "/2026-10-16-suite-1650ti-qwen352b-bf16",
    QW3: Q3 + "/2026-10-16-suite-3050oem-qwen352b-q4_0",
    QB3: B3 + "/2026-10-16-suite-3050oem-qwen352b-bf16",
}
GEN = {E1: HERE + "/{t}-e2b.jsonl", E3: G3 + "/{t}-e2b.jsonl", QW: QW1 + "/{t}-qwen.jsonl",
       QB: QB1 + "/{t}-qwen.jsonl", QW3: Q3 + "/{t}-qwen.jsonl", QB3: B3 + "/{t}-qwen.jsonl"}
PAIRS = [(E3, E1), (E1, QW), (E1, QB), (E3, QW3), (E3, QB3)]
# Qwen − Gemma on the 1650 Ti, minus the same on the 3050: how much of the gap was the card.
SHIFT = [((E1, QW), (E3, QW3)), ((E1, QB), (E3, QB3))]


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


res = {"suite": {}, "suite_pairs": [], "subsets": {}, "gen": {}, "shift": [], "truncated": {}, "bfcl_reasons": {}}
md = ["# Answer quality: Gemma 4 E2B and Qwen3.5-2B on the GTX 1650 Ti, against the RTX 3050", ""]

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
SP = {}
for a, b in PAIRS:
    s = paired(S[a], S[b])
    SP[(a, b)] = s["diff"]
    res["suite_pairs"].append({"a": a, "b": b, **s})
    md.append(
        f"| {a} | {b} | {s['n']} | {s['acc_ref']:.3f} | {s['acc_test']:.3f} | **{s['diff']:+.3f}** | "
        f"{s['diff_lo']:+.3f} to {s['diff_hi']:+.3f} | {s['ref_right_test_wrong']} | {s['ref_wrong_test_right']} |"
    )
md += [
    "",
    "Per subset, Qwen bf16 - Gemma E2B on the 1650 Ti (secondary; ranges not corrected for 13 comparisons):",
    "",
    "| subset | n | Gemma E2B | Qwen bf16 | Qwen Q4_0 | Qwen bf16 - Gemma | 95% range |",
    "|---|---:|---:|---:|---:|---:|---|",
]
for sub in sorted({s for s, _ in S[E1]}):
    g = {k: {i: v for i, v in S[k].items() if i[0] == sub} for k in (E1, QB, QW)}
    s = paired(g[E1], g[QB])
    q4 = sum(ok for ok, _ in g[QW].values()) / len(g[QW])
    res["subsets"][sub] = {**s, "qwen_q4_0": q4}
    md.append(
        f"| {sub} | {s['n']} | {s['acc_ref']:.3f} | {s['acc_test']:.3f} | {q4:.3f} | {s['diff']:+.3f} | "
        f"{s['diff_lo']:+.3f} to {s['diff_hi']:+.3f} |"
    )

md += [
    "",
    "## GSM8K (2,048-token limit) and BFCL simple",
    "",
    "| task | A | B | n | A | B | difference | 95% range | A only | B only |",
    "|---|---|---|---:|---:|---:|---:|---|---:|---:|",
]
GP = {}
for t, local in (("gsm8k", "gsm8k"), ("bfcl_simple", "bfcl")):
    for a, b in PAIRS:
        s = gpair(gload(GEN[a].format(t=local)), gload(GEN[b].format(t=local)))
        GP[(t, a, b)] = s["diff"]
        res["gen"].setdefault(t, []).append({"a": a, "b": b, **s})
        md.append(
            f"| {t} | {a} | {b} | {s['n']} | {s['ref']:.3f} | {s['test']:.3f} | **{s['diff']:+.3f}** | "
            f"{s['lo']:+.3f} to {s['hi']:+.3f} | {s['ref_only']} | {s['test_only']} |"
        )

md += [
    "",
    "## The Qwen − Gemma gap on each card",
    "",
    "| Qwen file | task | gap, 1650 Ti | gap, 3050 | 1650 Ti − 3050 |",
    "|---|---|---:|---:|---:|",
]
for here, there in SHIFT:
    for t in ("suite", "gsm8k", "bfcl_simple"):
        h = SP[here] if t == "suite" else GP[(t, *here)]
        th = SP[there] if t == "suite" else GP[(t, *there)]
        res["shift"].append({"qwen": here[1], "task": t, "gap_1650ti": h, "gap_3050": th, "moved": h - th})
        md.append(f"| {here[1]} | {t} | {h:+.3f} | {th:+.3f} | {h - th:+.3f} |")

md += ["", "GSM8K replies that hit the 2,048-token limit (gen_eval.py's `finish` == `length`):", "", "| run | truncated |",
       "|---|---:|"]
for k in GEN:
    rows = [json.loads(line) for line in open(GEN[k].format(t="gsm8k"))]
    n = sum(1 for r in rows if r.get("finish") == "length")
    res["truncated"][k] = n
    md.append(f"| {k} | {n} |")

md += [
    "",
    "BFCL failure reasons (gen_eval.py's `reason`, wrong answers only):",
    "",
    "| run | wrong | reasons |",
    "|---|---:|---|",
]
for k in GEN:
    rows = [json.loads(line) for line in open(GEN[k].format(t="bfcl"))]
    c = collections.Counter(("error" if r.get("error") else r.get("reason", "?")) for r in rows if not r["right"])
    res["bfcl_reasons"][k] = dict(c)
    md.append(f"| {k} | {sum(c.values())} | " + "; ".join(f"{n} × {r}" for r, n in c.most_common(6)) + " |")

open(os.path.join(HERE, "COMPARE.md"), "w").write("\n".join(md) + "\n")
json.dump(res, open(os.path.join(HERE, "compare.json"), "w"), indent=1)
print("\n".join(md))
