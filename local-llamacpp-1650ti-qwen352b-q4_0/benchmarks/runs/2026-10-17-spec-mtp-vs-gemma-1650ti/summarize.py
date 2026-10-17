"""Tables from results.jsonl, per model: median decode tok/s per prompt and arm (greedy, 3 repeats), x against
that model's no-draft arm, acceptance (accepted / drafted, summed over repeats), geometric means, the sampled
pass, Qwen against Gemma at each arm, and whether each greedy reply matches its model's no-draft reply.
Writes summary.txt and summary.json."""
import collections, json, math, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
R = [json.loads(line) for line in open(os.path.join(HERE, "results.jsonl"))]
models = list(dict.fromkeys(r["model"] for r in R))
arms = list(dict.fromkeys(r["arm"] for r in R))
prompts = list(dict.fromkeys(r["prompt"] for r in R))
cell = collections.defaultdict(list)
for r in R:
    cell[(r["model"], r["mode"], r["arm"], r["prompt"])].append(r)


def gmean(xs):
    return math.exp(st.mean(map(math.log, xs)))


med = {k: st.median(x["decode_tps"] for x in c) for k, c in cell.items()}
res = {"geomean_tps": {}, "geomean_x": {}, "acceptance": {}, "same_as_none": {}, "repeats_identical": {}}
lines = []
for m in models:
    for mode in ("greedy", "sampled"):
        lines += [f"## {m}, {mode}", "", "| prompt | " + " | ".join(arms) + " |", "|---" * (len(arms) + 1) + "|"]
        ratios = collections.defaultdict(list)
        for p in prompts:
            base = med[(m, mode, "none", p)]
            row = [p]
            for a in arms:
                c = cell[(m, mode, a, p)]
                v = med[(m, mode, a, p)]
                dn = sum(x["draft_n"] for x in c)
                acc = sum(x["draft_accepted"] for x in c) / dn if dn else None
                ratios[a].append(v / base)
                row.append(f"{v:.1f}" + ("" if a == "none" else f" ({v / base:.2f}x) {acc:.2f}"))
            lines.append("| " + " | ".join(row) + " |")
        for a in arms:
            key = f"{m}/{mode}/{a}"
            res["geomean_x"][key] = gmean(ratios[a])
            res["geomean_tps"][key] = gmean([med[(m, mode, a, p)] for p in prompts])
            dn = sum(x["draft_n"] for p in prompts for x in cell[(m, mode, a, p)])
            res["acceptance"][key] = (sum(x["draft_accepted"] for p in prompts for x in cell[(m, mode, a, p)]) / dn
                                      if dn else None)
        lines.append("| geometric mean, x | " + " | ".join(f"{res['geomean_x'][f'{m}/{mode}/{a}']:.3f}" for a in arms) + " |")
        lines.append("| geometric mean, tok/s | " + " | ".join(f"{res['geomean_tps'][f'{m}/{mode}/{a}']:.1f}" for a in arms) + " |")
        lines.append("")
    spread = max((max(x["decode_tps"] for x in c) - min(x["decode_tps"] for x in c)) / st.median(x["decode_tps"] for x in c)
                 for k, c in cell.items() if k[0] == m and k[1] == "greedy")
    lines += [f"{m}: largest within-cell spread across greedy repeats: {spread:.1%}", ""]

lines += ["## Qwen / Gemma, geometric-mean tok/s at each arm", "", "| mode | " + " | ".join(arms) + " |",
          "|---" * (len(arms) + 1) + "|"]
res["qwen_over_gemma"] = {}
for mode in ("greedy", "sampled"):
    row = []
    for a in arms:
        x = res["geomean_tps"][f"qwen/{mode}/{a}"] / res["geomean_tps"][f"gemma/{mode}/{a}"]
        res["qwen_over_gemma"][f"{mode}/{a}"] = x
        row.append(f"{x:.3f}")
    lines.append(f"| {mode} | " + " | ".join(row) + " |")
lines += ["", "Best drafted arm per model against the other model's no-draft arm (greedy):", ""]
for m, o in (("qwen", "gemma"), ("gemma", "qwen")):
    best = max((a for a in arms if a != "none"), key=lambda a: res["geomean_tps"][f"{m}/greedy/{a}"])
    v = res["geomean_tps"][f"{m}/greedy/{best}"]
    lines.append(f"- {m} {best}: {v:.1f} tok/s, {v / res['geomean_tps'][f'{o}/greedy/none']:.3f}x {o} with no draft, "
                 f"{v / res['geomean_tps'][f'{o}/greedy/{best}']:.3f}x {o} {best}")

lines += ["", "## greedy replies identical to the model's no-draft reply (rep 0), and repeats identical", "",
          "| model | arm | same as no draft | repeats identical |", "|---|---|---:|---:|"]
for m in models:
    for a in arms:
        same = sum(cell[(m, "greedy", a, p)][0]["sha"] == cell[(m, "greedy", "none", p)][0]["sha"] for p in prompts)
        rep = sum(len({x["sha"] for x in cell[(m, "greedy", a, p)]}) == 1 for p in prompts)
        res["same_as_none"][f"{m}/{a}"] = same
        res["repeats_identical"][f"{m}/{a}"] = rep
        lines.append(f"| {m} | {a} | {same} of {len(prompts)} | {rep} of {len(prompts)} |")
lines += ["", "VRAM per arm (nvidia-smi after warm-up): " + "; ".join(
    f"{m} {a} {cell[(m, 'greedy', a, prompts[0])][0]['gpu'].split(',')[-1].strip()}" for m in models for a in arms)]
open(os.path.join(HERE, "summary.txt"), "w").write("\n".join(lines) + "\n")
json.dump(res, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
print("\n".join(lines))
