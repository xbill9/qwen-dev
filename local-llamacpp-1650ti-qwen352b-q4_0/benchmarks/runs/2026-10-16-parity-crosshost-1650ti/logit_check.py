"""Does llama.cpp compute Qwen3.5-2B the way transformers does?

Builds the suite harness's exact prompts (nimble_suite/run_suite.own_prompt, Qwen tokenizer) for the
first 5 records of each of the 13 subsets and reads the next-token top-32 log-probabilities from:

  reference  transformers on the CPU, float32 compute over Qwen's bf16 weights (cached in
             reference-fp32.jsonl)
  test       --test llama: llama-server /completion on :8080 (whichever GGUF it serves; --tag names it)
             --test hf-bf16: transformers on the GPU in bfloat16, the yardstick for how far bf16
             arithmetic alone moves the answer

Both sides are read the way the suite harness reads them: top 32 tokens, a label outside them scored at
the 32nd token's log-probability, label probabilities renormalised over the labels. Per prompt it
records the chosen label on each side, the top-1 token, the largest label-probability difference, and
the largest log-probability difference over the reference's top 5 tokens. Writes logit_check-<tag>.jsonl
and prints the summary.

    python3 logit_check.py --test llama --tag bf16
    python3 logit_check.py --test hf-bf16 --tag hf-bf16
"""

import argparse
import glob
import json
import math
import os
import sys
import urllib.request

J = "/home/xbill/qwen-dev/jev-tpu-v5e1"
sys.path[:0] = [os.path.join(J, "vendor"), os.path.join(J, "nimble_suite")]
import structured_server as ss  # noqa: E402
import run_suite as rs  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--test", choices=["llama", "hf-bf16"], required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--per-subset", type=int, default=5)
args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
HF = "/home/xbill/models/Qwen3.5-2B"
K = 32

tok = AutoTokenizer.from_pretrained(HF)
ss.init_tokenizer(tok)
rs.MODEL_TOK = tok

prompts = []
for path in sorted(glob.glob("/home/xbill/data/nimble-suite/public/*/all.jsonl")):
    sub = os.path.basename(os.path.dirname(path))
    if sub == "hs2full":
        continue
    for line in list(open(path))[: args.per_subset]:
        rec = json.loads(line)
        body = {"state": rec["input"]["state"], "questions": rec["input"]["questions"]}
        schema = ss.jev_schema(body)
        template, slots = ss.template_for(schema, ss.SCAFFOLD, "")
        ids = rs.own_prompt(ss.system_text(schema), ss.jev_state(body), template, slots[0])
        prompts.append((sub, rec["id"], ids, slots[0]["label_ids"]))

_models = {}


def hf_top(ids, device, dtype):
    key = (device, dtype)
    if key not in _models:
        _models[key] = AutoModelForCausalLM.from_pretrained(HF, dtype=dtype).to(device).eval()
    with torch.no_grad():
        logits = _models[key](torch.tensor([ids], device=device)).logits[0, -1].float().cpu()
    lp = torch.log_softmax(logits, -1)
    top = torch.topk(lp, K)
    return {int(i): float(v) for v, i in zip(top.values, top.indices)}


def llama_top(ids):
    body = {
        "prompt": ids,
        "n_predict": 1,
        "temperature": 0,
        "n_probs": K,
        "post_sampling_probs": False,
        "cache_prompt": False,
    }
    req = urllib.request.Request(
        "http://127.0.0.1:8080/completion", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    d = json.load(urllib.request.urlopen(req))
    return {t["id"]: t["logprob"] for t in d["completion_probabilities"][0]["top_logprobs"]}


def label_probs(top, labels):
    floor = min(top.values())
    lp = [top.get(i, floor) for i in labels]
    m = max(lp)
    e = [math.exp(x - m) for x in lp]
    return [x / sum(e) for x in e]


cache = os.path.join(HERE, "reference-fp32.jsonl")
ref = (
    {r["id"]: {int(k): v for k, v in r["top"].items()} for r in map(json.loads, open(cache))}
    if os.path.exists(cache)
    else {}
)

out = open(os.path.join(HERE, f"logit_check-{args.tag}.jsonl"), "w")
for sub, rid, ids, labels in prompts:
    if rid not in ref:
        ref[rid] = hf_top(ids, "cpu", torch.float32)
        with open(cache, "a") as f:
            f.write(json.dumps({"id": rid, "top": ref[rid]}) + "\n")
    r = ref[rid]
    t = llama_top(ids) if args.test == "llama" else hf_top(ids, "cuda", torch.bfloat16)
    pr, pt = label_probs(r, labels), label_probs(t, labels)
    top5 = sorted(r, key=r.get, reverse=True)[:5]
    floor_t = min(t.values())
    row = {
        "subset": sub,
        "id": rid,
        "n_prompt": len(ids),
        "n_labels": len(labels),
        "label_p_ref": pr,
        "label_p_test": pt,
        "max_abs_label_p_diff": max(abs(a - b) for a, b in zip(pr, pt)),
        "same_label": pr.index(max(pr)) == pt.index(max(pt)),
        "same_top1": max(r, key=r.get) == max(t, key=t.get),
        "top5_max_abs_logprob_diff": max(abs(r[i] - t.get(i, floor_t)) for i in top5),
    }
    out.write(json.dumps(row) + "\n")
    out.flush()

rows = [json.loads(line) for line in open(os.path.join(HERE, f"logit_check-{args.tag}.jsonl"))]


def q(key, f):
    v = sorted(x[key] for x in rows)
    return f(v)


summary = {
    "tag": args.tag,
    "test": args.test,
    "prompts": len(rows),
    "same_chosen_label": sum(x["same_label"] for x in rows),
    "same_top1_token": sum(x["same_top1"] for x in rows),
    "label_p_diff_median": q("max_abs_label_p_diff", lambda v: v[len(v) // 2]),
    "label_p_diff_max": q("max_abs_label_p_diff", lambda v: v[-1]),
    "top5_logprob_diff_median": q("top5_max_abs_logprob_diff", lambda v: v[len(v) // 2]),
    "top5_logprob_diff_max": q("top5_max_abs_logprob_diff", lambda v: v[-1]),
    "label_disagreements": [(x["subset"], x["id"]) for x in rows if not x["same_label"]],
}
json.dump(summary, open(os.path.join(HERE, f"summary-{args.tag}.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
