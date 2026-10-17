"""Tables from results.jsonl: median decode tok/s per prompt and arm (greedy, 3 repeats), x against
no draft, acceptance (accepted / drafted, summed over repeats), geometric means, the sampled pass,
and whether each greedy reply matches the no-draft arm's. Writes summary.txt."""
import collections, json, math, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
R = [json.loads(line) for line in open(os.path.join(HERE, "results.jsonl"))]
arms = list(dict.fromkeys(r["arm"] for r in R))
prompts = list(dict.fromkeys(r["prompt"] for r in R))
cell = collections.defaultdict(list)
for r in R:
    cell[(r["mode"], r["arm"], r["prompt"])].append(r)
lines = []
for mode in ("greedy", "sampled"):
    lines += [f"## {mode}", "", "| prompt | " + " | ".join(arms) + " |", "|---" * (len(arms) + 1) + "|"]
    ratios = collections.defaultdict(list)
    for p in prompts:
        base = st.median(x["decode_tps"] for x in cell[(mode, "none", p)])
        row = [p]
        for a in arms:
            c = cell[(mode, a, p)]
            v = st.median(x["decode_tps"] for x in c)
            dn = sum(x["draft_n"] for x in c)
            acc = sum(x["draft_accepted"] for x in c) / dn if dn else None
            ratios[a].append(v / base)
            row.append(f"{v:.1f}" + ("" if a == "none" else f" ({v / base:.2f}x) {acc:.2f}"))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("| geometric mean | " + " | ".join(f"{math.exp(st.mean(map(math.log, ratios[a]))):.3f}" for a in arms) + " |")
    if mode == "greedy":
        spread = max((max(x["decode_tps"] for x in c) - min(x["decode_tps"] for x in c)) / st.median(x["decode_tps"] for x in c)
                     for k, c in cell.items() if k[0] == "greedy")
        lines += ["", f"largest within-cell spread across greedy repeats: {spread:.1%}"]
    lines.append("")
lines += ["## greedy replies identical to the no-draft arm (rep 0), and repeats identical within each arm", "",
          "| arm | same as no draft | repeats identical |", "|---|---:|---:|"]
for a in arms:
    same = sum(cell[("greedy", a, p)][0]["sha"] == cell[("greedy", "none", p)][0]["sha"] for p in prompts)
    rep = sum(len({x["sha"] for x in cell[("greedy", a, p)]}) == 1 for p in prompts)
    lines.append(f"| {a} | {same} of {len(prompts)} | {rep} of {len(prompts)} |")
lines += ["", "VRAM per arm (nvidia-smi after warm-up): " + "; ".join(f"{a} {cell[('greedy', a, prompts[0])][0]['gpu'].split(',')[-1].strip()}" for a in arms)]
open(os.path.join(HERE, "summary.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
