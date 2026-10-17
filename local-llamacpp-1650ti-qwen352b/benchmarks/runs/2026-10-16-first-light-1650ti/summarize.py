"""Per-cell llama-bench means from results/*.json, and each bf16 cell against the Q4_0 file."""
import json
import os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def cell(name):
    p = os.path.join(R, name + ".json")
    try:
        rows = json.load(open(p))
    except (OSError, ValueError):
        return None
    out = {}
    for row in rows:
        key = f"pp{row['n_prompt']}" if row["n_gen"] == 0 else f"tg{row['n_gen']}"
        out[key] = (row["avg_ts"], row["stddev_ts"])
    return out


names = [line.split()[0] for line in open(os.path.join(R, "cells.txt"))]
cells = {n: cell(n) for n in names}
ref = cells.get("q4_0")
lines = []
for n in names:
    c = cells[n]
    if not c:
        lines.append(f"{n}: failed")
        continue
    parts = [f"{k} {v[0]:.2f} ± {v[1]:.2f}" for k, v in c.items()]
    if ref and n != "q4_0":
        parts += [f"{k}/q4_0 {c[k][0] / ref[k][0]:.3f}x" for k in c if k in ref]
    lines.append(f"{n}: " + "  ".join(parts))
open(os.path.join(R, "summary.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
