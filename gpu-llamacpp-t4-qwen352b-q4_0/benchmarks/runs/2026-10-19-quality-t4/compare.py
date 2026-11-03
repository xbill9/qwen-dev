"""Qwen3.5-2B Q4_0 against Gemma 4 E2B exact QAT Q4_0, with and without its drafter, on the Tesla T4.

    python3 compare.py      # writes COMPARE.md and compare.json beside this file

The 1650 Ti Gemma-vs-Qwen run's compare.py with the arms changed. Suite correctness is the harness's own
(score_suite.assess on score_suite.dist, gold_key) and the paired difference with its 2,000-draw bootstrap
range is quant_compare.paired. GSM8K and BFCL are gen_eval.py's own `right` field, paired by
gen_compare.compare (10,000 resamples, seed 0). Failure reasons, cut-off replies (`finish` == "length"),
reply tokens and records that differ between arms are counted here from each record's own fields.
The Qwen-to-Gemma gap on the T4 minus the same gap on the 1650 Ti is computed here. Nothing is counted by hand.
"""

import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "../../../.."))
J = R + "/jev-tpu-v5e1"
sys.path[:0] = [J, os.path.join(J, "nimble_suite")]
from gen_compare import compare as gpair, load as gload  # noqa: E402
from quant_compare import paired  # noqa: E402
from score_suite import assess, dist, gold_key  # noqa: E402

QW1 = R + "/local-llamacpp-1650ti-qwen352b-q4_0/benchmarks/runs/2026-10-16-quality-1650ti"
E1D = R + "/local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-17-quality-vs-qwen-1650ti"
QT, GT, GD = "Qwen Q4_0, T4", "Gemma E2B, T4", "Gemma E2B + drafter, T4"
Q1, G1 = "Qwen Q4_0, 1650 Ti", "Gemma E2B, 1650 Ti"
SUITE = {
    QT: HERE + "/2026-10-19-suite-t4-qwen-q4_0",
    GT: HERE + "/2026-10-19-suite-t4-gemma-none",
    GD: HERE + "/2026-10-19-suite-t4-gemma-draft",
    Q1: QW1 + "/2026-10-16-suite-1650ti-qwen352b-q4_0",
    G1: E1D + "/2026-10-17-suite-1650ti-e2b-q4exact",
}
GEN = {
    QT: HERE + "/{t}-qwen-q4_0.jsonl",
    GT: HERE + "/{t}-gemma-none.jsonl",
    GD: HERE + "/{t}-gemma-draft.jsonl",
    Q1: QW1 + "/{t}-qwen.jsonl",
    G1: E1D + "/{t}-e2b.jsonl",
}
PAIRS = [(GT, QT), (GT, GD), (Q1, QT), (G1, GT), (G1, Q1)]
# Qwen − Gemma on the T4, minus the same on the 1650 Ti: how much of the gap was the card.
SHIFT = [((GT, QT), (G1, Q1))]
T4 = (QT, GT, GD)


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


def rows(k, t):
    return [json.loads(line) for line in open(GEN[k].format(t=t))]


res = {
    "suite": {},
    "suite_pairs": [],
    "subsets": {},
    "gen": {},
    "shift": [],
    "replies": {},
    "bfcl_reasons": {},
    "differ": {},
}
md = ["# Answer quality: Qwen3.5-2B Q4_0 against Gemma 4 E2B, with and without its drafter, on the Tesla T4", ""]

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
    SP[(a, b)] = s
    res["suite_pairs"].append({"a": a, "b": b, **s})
    md.append(
        f"| {a} | {b} | {s['n']} | {s['acc_ref']:.3f} | {s['acc_test']:.3f} | **{s['diff']:+.3f}** | "
        f"{s['diff_lo']:+.3f} to {s['diff_hi']:+.3f} | {s['ref_right_test_wrong']} | {s['ref_wrong_test_right']} |"
    )
md += [
    "",
    "Per subset, Qwen Q4_0 - Gemma E2B on the T4 (secondary; ranges not corrected for 13 comparisons):",
    "",
    "| subset | n | Gemma | Qwen | Qwen - Gemma | 95% range |",
    "|---|---:|---:|---:|---:|---|",
]
for sub in sorted({s for s, _ in S[GT]}):
    g = {k: {i: v for i, v in S[k].items() if i[0] == sub} for k in (GT, QT)}
    s = paired(g[GT], g[QT])
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
GP = {}
for t, local in (("gsm8k", "gsm8k"), ("bfcl_simple", "bfcl")):
    for a, b in PAIRS:
        s = gpair(gload(GEN[a].format(t=local)), gload(GEN[b].format(t=local)))
        GP[(local, a, b)] = s
        res["gen"].setdefault(t, []).append({"a": a, "b": b, **s})
        md.append(
            f"| {t} | {a} | {b} | {s['n']} | {s['ref']:.3f} | {s['test']:.3f} | **{s['diff']:+.3f}** | "
            f"{s['lo']:+.3f} to {s['hi']:+.3f} | {s['ref_only']} | {s['test_only']} |"
        )

md += ["", "## Qwen − Gemma gap, T4 against 1650 Ti", "", "| | T4 | 1650 Ti | moved |", "|---|---:|---:|---:|"]
for pa, pb in SHIFT:
    for name, get in (
        ("suite", lambda p: SP[p]["diff"]),
        ("gsm8k", lambda p: GP[("gsm8k",) + p]["diff"]),
        ("bfcl", lambda p: GP[("bfcl",) + p]["diff"]),
    ):
        x, y = get(pa), get(pb)
        res["shift"].append({"task": name, "t4": x, "1650ti": y, "moved": x - y})
        md.append(f"| {name} | {x:+.3f} | {y:+.3f} | {x - y:+.3f} |")

md += [
    "",
    "## Replies",
    "",
    "| run | GSM8K cut off at 2,048 | GSM8K reply tokens | BFCL no tool call | BFCL reply tokens |",
    "|---|---:|---:|---:|---:|",
]
for k in GEN:
    g, b = rows(k, "gsm8k"), rows(k, "bfcl")
    r = {
        "gsm8k_cut": sum(x.get("finish") == "length" for x in g),
        "gsm8k_tokens": sum(x.get("completion_tokens", 0) for x in g),
        "bfcl_no_call": sum(not x.get("calls") for x in b),
        "bfcl_tokens": sum(x.get("completion_tokens", 0) for x in b),
    }
    res["replies"][k] = r
    md.append(f"| {k} | {r['gsm8k_cut']} | {r['gsm8k_tokens']:,} | {r['bfcl_no_call']} | {r['bfcl_tokens']:,} |")

md += [
    "",
    "BFCL failure reasons (gen_eval.py's `reason`, wrong answers only):",
    "",
    "| run | wrong | reasons |",
    "|---|---:|---|",
]
for k in T4:
    c = collections.Counter(
        ("error" if r.get("error") else r.get("reason", "?")) for r in rows(k, "bfcl") if not r["right"]
    )
    res["bfcl_reasons"][k] = dict(c)
    md.append(f"| {k} | {sum(c.values())} | " + "; ".join(f"{n} × {r}" for r, n in c.most_common(6)) + " |")

md += [
    "",
    "## Gemma with and without its drafter, record by record",
    "",
    "| task | records | same text | same right/wrong |",
    "|---|---:|---:|---:|",
]
for t in ("gsm8k", "bfcl"):
    a = {r["id"]: r for r in rows(GT, t)}
    b = {r["id"]: r for r in rows(GD, t)}
    ids = sorted(a.keys() & b.keys())
    d = {
        "n": len(ids),
        "same_text": sum(a[i]["text"] == b[i]["text"] and a[i].get("calls") == b[i].get("calls") for i in ids),
        "same_right": sum(a[i]["right"] == b[i]["right"] for i in ids),
    }
    res["differ"][t] = d
    md.append(f"| {t} | {d['n']} | {d['same_text']} | {d['same_right']} |")
ids = sorted(S[GT].keys() & S[GD].keys())
d = {"n": len(ids), "same_right": sum(S[GT][i][0] == S[GD][i][0] for i in ids)}
res["differ"]["suite"] = d
md.append(f"| suite | {d['n']} | - | {d['same_right']} |")

open(os.path.join(HERE, "COMPARE.md"), "w").write("\n".join(md) + "\n")
json.dump(res, open(os.path.join(HERE, "compare.json"), "w"), indent=1)
print("\n".join(md))
