"""Gemma 4 E2B rounded to Q4_0 from the bf16 release, against Google's QAT Q4_0 and Qwen3.5-2B, GTX 1650 Ti.

    python3 compare.py      # writes COMPARE.md and compare.json beside this file

The Gemma 1650 Ti run's compare.py with the arms changed. Suite correctness is the harness's own
(score_suite.assess on score_suite.dist, gold_key) and the paired difference with its 2,000-draw bootstrap
range is quant_compare.paired. GSM8K and BFCL are gen_eval.py's own `right` field, paired by
gen_compare.compare (10,000 resamples, seed 0). The share of Qwen's gap that QAT accounts for is computed
here. Tensor types come from gguf-py reading both GGUFs. Nothing is counted by hand.
"""

import collections
import glob
import json
import os
import sys

J = "/home/xbill/qwen-dev/jev-tpu-v5e1"
sys.path[:0] = [J, os.path.join(J, "nimble_suite"), "/home/xbill/llama.cpp/gguf-py"]
from gen_compare import compare as gpair, load as gload  # noqa: E402
from gguf import GGUFReader  # noqa: E402
from quant_compare import paired  # noqa: E402
from score_suite import assess, dist, gold_key  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = "/home/xbill/qwen-dev"
GQ = R + "/local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-17-quality-vs-qwen-1650ti"
QW1 = R + "/local-llamacpp-1650ti-qwen352b-q4_0/benchmarks/runs/2026-10-16-quality-1650ti"
QB1 = R + "/local-llamacpp-1650ti-qwen352b/benchmarks/runs/2026-10-16-quality-1650ti"
ER, EQ, QW, QB = "Gemma E2B rounded Q4_0", "Gemma E2B QAT Q4_0", "Qwen Q4_0", "Qwen bf16"
FILES = {
    ER: "/home/xbill/models/gemma-4-E2B-it-gguf/gemma-4-E2B-it-Q4_0.gguf",
    EQ: "/home/xbill/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf",
    QW: "/home/xbill/models/Qwen3.5-2B-gguf/Qwen3.5-2B-Q4_0.gguf",
}
SUITE = {
    ER: HERE + "/2026-10-17-suite-1650ti-e2b-q4_0rtn",
    EQ: GQ + "/2026-10-17-suite-1650ti-e2b-q4exact",
    QW: QW1 + "/2026-10-16-suite-1650ti-qwen352b-q4_0",
    QB: QB1 + "/2026-10-16-suite-1650ti-qwen352b-bf16",
}
GEN = {ER: HERE + "/{t}-e2b.jsonl", EQ: GQ + "/{t}-e2b.jsonl", QW: QW1 + "/{t}-qwen.jsonl",
       QB: QB1 + "/{t}-qwen.jsonl"}
PAIRS = [(EQ, ER), (ER, QW), (ER, QB), (EQ, QW)]


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


res = {"tensors": {}, "suite": {}, "suite_pairs": [], "subsets": {}, "gen": {}, "qat_share": [], "truncated": {},
       "bfcl_reasons": {}}
md = ["# Answer quality: Gemma 4 E2B with and without QAT, against Qwen3.5-2B, GTX 1650 Ti", ""]

md += ["## What the files hold (bytes by tensor type, from gguf-py)", "", "| file | total GB | by type |", "|---|---:|---|"]
for k, p in FILES.items():
    by = collections.Counter()
    for t in GGUFReader(p).tensors:
        by[t.tensor_type.name] += int(t.n_bytes)
    tot = sum(by.values())
    res["tensors"][k] = dict(by)
    md.append(f"| {k} | {tot / 1e9:.3f} | " + ", ".join(f"{n} {b / tot:.1%}" for n, b in by.most_common()) + " |")

S = {k: suite_load(v) for k, v in SUITE.items()}
md += ["", "## Nimble 13-subset suite", "", "| run | records | accuracy | all labels returned |", "|---|---:|---:|---:|"]
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
D = {}
for a, b in PAIRS:
    s = paired(S[a], S[b])
    D[("suite", a, b)] = s["diff"]
    res["suite_pairs"].append({"a": a, "b": b, **s})
    md.append(
        f"| {a} | {b} | {s['n']} | {s['acc_ref']:.3f} | {s['acc_test']:.3f} | **{s['diff']:+.3f}** | "
        f"{s['diff_lo']:+.3f} to {s['diff_hi']:+.3f} | {s['ref_right_test_wrong']} | {s['ref_wrong_test_right']} |"
    )
md += [
    "",
    "Per subset, rounded - QAT (secondary; ranges not corrected for 13 comparisons):",
    "",
    "| subset | n | QAT | rounded | rounded - QAT | 95% range |",
    "|---|---:|---:|---:|---:|---|",
]
for sub in sorted({s for s, _ in S[EQ]}):
    g = {k: {i: v for i, v in S[k].items() if i[0] == sub} for k in (EQ, ER)}
    s = paired(g[EQ], g[ER])
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
        D[(t, a, b)] = s["diff"]
        res["gen"].setdefault(t, []).append({"a": a, "b": b, **s})
        md.append(
            f"| {t} | {a} | {b} | {s['n']} | {s['ref']:.3f} | {s['test']:.3f} | **{s['diff']:+.3f}** | "
            f"{s['lo']:+.3f} to {s['hi']:+.3f} | {s['ref_only']} | {s['test_only']} |"
        )

md += [
    "",
    "## How much of Qwen Q4_0's gap to Gemma QAT is QAT",
    "",
    "Gap = Qwen Q4_0 - Gemma QAT. QAT's part = rounded - QAT. Left after QAT = Qwen Q4_0 - Gemma rounded.",
    "",
    "| task | gap | QAT's part | left after QAT | QAT's share |",
    "|---|---:|---:|---:|---:|",
]
for t in ("suite", "gsm8k", "bfcl_simple"):
    gap, qat, left = D[(t, EQ, QW)], D[(t, EQ, ER)], D[(t, ER, QW)]
    res["qat_share"].append({"task": t, "gap": gap, "qat": qat, "left": left, "share": qat / gap})
    md.append(f"| {t} | {gap:+.3f} | {qat:+.3f} | {left:+.3f} | {qat / gap:.0%} |")

md += ["", "GSM8K replies that hit the 2,048-token limit (gen_eval.py's `finish` == `length`):", "",
       "| run | truncated |", "|---|---:|"]
for k in GEN:
    rows = [json.loads(line) for line in open(GEN[k].format(t="gsm8k"))]
    n = sum(1 for r in rows if r.get("finish") == "length")
    res["truncated"][k] = n
    md.append(f"| {k} | {n} |")

md += ["", "BFCL failure reasons (gen_eval.py's `reason`, wrong answers only):", "", "| run | wrong | reasons |",
       "|---|---:|---|"]
for k in GEN:
    rows = [json.loads(line) for line in open(GEN[k].format(t="bfcl"))]
    c = collections.Counter(("error" if r.get("error") else r.get("reason", "?")) for r in rows if not r["right"])
    res["bfcl_reasons"][k] = dict(c)
    md.append(f"| {k} | {sum(c.values())} | " + "; ".join(f"{n} × {r}" for r, n in c.most_common(6)) + " |")

open(os.path.join(HERE, "COMPARE.md"), "w").write("\n".join(md) + "\n")
json.dump(res, open(os.path.join(HERE, "compare.json"), "w"), indent=1)
print("\n".join(md))
