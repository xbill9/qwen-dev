"""Best-case scores for Qwen3.5-2B if every reply cut off at the GSM8K token limit, and every
BFCL reply that made no tool call, had been right; counts no-call replies whose text looks like
a tool call the parser missed; and the suite gap without the three one-sided subsets.
Run from the repo root: python3 local-llamacpp-3050oem-qwen352b-q4_0/docs/qwen-vs-gemma/evidence/best_case.py"""
import json, re

RUNS = {
    "q4_0, RTX 3050": "local-llamacpp-3050oem-qwen352b-q4_0/benchmarks/runs/2026-10-16-quality-vs-gemma-3050oem/%s-qwen.jsonl",
    "bf16, RTX 3050": "local-llamacpp-3050oem-qwen352b/benchmarks/runs/2026-10-16-quality-bf16-3050oem/%s-qwen.jsonl",
}
CALL_LIKE = re.compile(r'tool_call|<function|"name"\s*:|\{\s*"')

for arm, path in RUNS.items():
    rows = [json.loads(l) for l in open(path % "gsm8k")]
    right = sum(r["right"] for r in rows)
    cut = [r for r in rows if r["finish"] == "length"]
    cut_right = sum(r["right"] for r in cut)
    print(f"{arm} GSM8K: {right / len(rows):.4f}, {len(cut)} at the limit ({cut_right} right), "
          f"best case {(right + len(cut) - cut_right) / len(rows):.4f}")
    rows = [json.loads(l) for l in open(path % "bfcl")]
    right = sum(r["right"] for r in rows)
    nocall = [r for r in rows if not r["calls"]]
    nc_right = sum(r["right"] for r in nocall)
    like = sum(1 for r in nocall if CALL_LIKE.search(r["text"] or ""))
    print(f"{arm} BFCL: {right / len(rows):.4f}, {len(nocall)} with no tool call ({like} with call-like text), "
          f"best case {(right + len(nocall) - nc_right) / len(rows):.4f}")

# Per-subset table, 1650 Ti, from local-llamacpp-1650ti-2b-q4_0/benchmarks/runs/2026-10-17-quality-vs-qwen-1650ti/COMPARE.md
SUITE = """aegis2 250 0.744 0.692
boolq 300 0.627 0.737
civil_comments 300 0.900 0.233
helpsteer2 249 0.365 0.341
massive-de-DE 350 0.734 0.566
massive-en-US 350 0.809 0.654
multinli 299 0.696 0.331
paws 250 0.740 0.520
pubmedqa 250 0.604 0.368
squad2 299 0.686 0.816
summeval-consistency 144 0.757 0.160
summeval-relevance 240 0.296 0.246
vitaminc-dev 599 0.730 0.452"""
subsets = [(a, int(n), float(g), float(q)) for a, n, g, q in (l.split() for l in SUITE.splitlines())]
one_sided = {"civil_comments", "summeval-consistency", "multinli"}
for label, rs in (("all 13 subsets", subsets), ("without the 3 one-sided subsets", [s for s in subsets if s[0] not in one_sided])):
    n = sum(s[1] for s in rs)
    g = sum(s[1] * s[2] for s in rs) / n
    q = sum(s[1] * s[3] for s in rs) / n
    print(f"suite, 1650 Ti, {label}: n={n} Gemma {g:.3f} Qwen bf16 {q:.3f} gap {q - g:+.3f}")
