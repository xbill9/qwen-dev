"""Means of the two serving-config cells per model and the ratios, from results/*.json."""
import json
import os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def tps(name):
    out = {}
    for row in json.load(open(os.path.join(R, name + ".json"))):
        key = f"pp{row['n_prompt']}" if row["n_gen"] == 0 else f"tg{row['n_gen']}"
        out[key] = (row["avg_ts"], row["stddev_ts"])
    return out


c = {n: tps(n) for n in ("qwen-fa1", "gemma-fa1", "qwen-fa0", "qwen-fa1-rep", "gemma-fa1-rep")}
lines = []
for n, cell in c.items():
    lines.append(n + ": " + "  ".join(f"{k} {v[0]:.2f} ± {v[1]:.2f}" for k, v in cell.items()))
for k in ("pp512", "pp2048", "tg128"):
    q = (c["qwen-fa1"][k][0] + c["qwen-fa1-rep"][k][0]) / 2
    g = (c["gemma-fa1"][k][0] + c["gemma-fa1-rep"][k][0]) / 2
    lines.append(
        f"{k}: qwen {q:.2f}  gemma {g:.2f}  qwen/gemma {q / g:.3f}x  qwen fa1/fa0 {q / c['qwen-fa0'][k][0]:.3f}x"
        f"  qwen rep/first {c['qwen-fa1-rep'][k][0] / c['qwen-fa1'][k][0]:.3f}x"
        f"  gemma rep/first {c['gemma-fa1-rep'][k][0] / c['gemma-fa1'][k][0]:.3f}x"
    )
open(os.path.join(R, "summary.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
