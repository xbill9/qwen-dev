"""Does llama-server build the same prompt as Qwen's own chat template in transformers?

Renders each prompt twice: llama-server's /apply-template (the template embedded in the GGUF, run by
llama.cpp's jinja engine, with the request's tools) and transformers' tokenizer.apply_chat_template on
the Hugging Face checkout, enable_thinking=False as llama-server's --reasoning off sets it. Covers all
400 BFCL simple records (with tools) and 100 GSM8K records. Reports how many are byte-identical and
prints the first difference. Needs llama-server on :8080 serving Qwen3.5-2B.

    python3 template_check.py > template_check.txt
"""

import json
import os
import sys
import urllib.request

J = "/home/xbill/qwen-dev/jev-tpu-v5e1"
sys.path[:0] = [J]
import gen_eval  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("/home/xbill/models/Qwen3.5-2B")


def server(body):
    req = urllib.request.Request(
        "http://127.0.0.1:8080/apply-template", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    return json.load(urllib.request.urlopen(req))["prompt"]


def load(task):
    return [json.loads(line) for line in open(os.path.join(J, "data", task + ".jsonl"))]


cases = []
for rec in load("bfcl_simple"):
    cases.append(("bfcl_simple", rec["id"], rec["question"], gen_eval.tools_for(rec["functions"])))
for rec in load("gsm8k")[:100]:
    cases.append(
        (
            "gsm8k",
            rec["id"],
            [{"role": "user", "content": gen_eval.GSM8K_PROMPT.format(question=rec["question"])}],
            None,
        )
    )

same = {}
first_diff = {}
for task, rid, msgs, tools in cases:
    body = {"messages": msgs}
    if tools:
        body["tools"] = tools
    a = server(body)
    b = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ok = a == b
    same.setdefault(task, [0, 0])
    same[task][0] += ok
    same[task][1] += 1
    if not ok and task not in first_diff:
        i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
        first_diff[task] = (rid, i, a[max(0, i - 80) : i + 120], b[max(0, i - 80) : i + 120])

for task, (ok, n) in same.items():
    print(f"{task}: {ok} of {n} prompts byte-identical")
for task, (rid, i, a, b) in first_diff.items():
    print(f"\nfirst difference, {task} {rid}, at character {i}:\n--- llama-server\n{a!r}\n--- transformers\n{b!r}")
