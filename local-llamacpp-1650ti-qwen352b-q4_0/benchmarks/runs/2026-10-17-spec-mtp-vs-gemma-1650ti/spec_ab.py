"""Speculative decoding on the GTX 1650 Ti, Qwen3.5-2B against Gemma 4 E2B on one binary: each model with
no draft and with its own drafter at 1 and 2 drafted tokens. Qwen drafts from the MTP layer inside its GGUF
(`--spec-type draft-mtp`, no -md); Gemma from Google's QAT assistant drafter stored exact Q4_0 (-md, the way
local-llamacpp-1650ti-2b-q4_0 serves it). The 3050 runs' method
(local-llamacpp-3050oem-qwen352b-q4_0/benchmarks/runs/2026-10-16-spec-mtp-3050oem/spec_ab.py): one
llama-server per arm on port 8099, -c 8192, --parallel 1, six prompts, greedy, max_tokens 256, prompt cache
off, 3 repeats, then one sampled pass (temperature 0.8, seed 1234). GPU cooled to <=50 C before each arm.
Writes results.jsonl beside this file, with each reply's text.
"""
import hashlib, json, os, signal, subprocess, time, urllib.request

OUT = os.path.dirname(os.path.abspath(__file__))
BIN = "/home/xbill/llama.cpp/build/bin/llama-server"
QWEN = "/home/xbill/models/Qwen3.5-2B-gguf/Qwen3.5-2B-Q4_0.gguf"
GEMMA = "/home/xbill/models/gemma-4-E2B-it-qat-q4_0-exact-v2/gemma-4-E2B-it-q4_0-exact.gguf"
DRAFT = "/home/xbill/models/gemma-4-E2B-it-qat-assistant/gemma-4-E2B-it-qat-assistant-q4_0-exact.gguf"
PROMPTS = {
    "code": "Write a Python function that checks whether a number is prime, with a docstring and two doctests.",
    "chat": "I just moved to a new city and feel a bit lonely. Any advice?",
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "explain": "Explain how a refrigerator works to a ten-year-old.",
    "facts": "List the planets of the solar system with one fact about each.",
    "summary": "Summarize the causes of the French Revolution in one paragraph.",
}
ARMS = [  # model, arm, gguf, extra flags
    ("qwen", "none", QWEN, []),
    ("qwen", "draft-n1", QWEN, ["--spec-type", "draft-mtp", "--spec-draft-n-max", "1"]),
    ("qwen", "draft-n2", QWEN, ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]),
    ("gemma", "none", GEMMA, []),
    ("gemma", "draft-n1", GEMMA, ["-md", DRAFT, "--spec-type", "draft-mtp", "-ngld", "all", "--spec-draft-n-max", "1"]),
    ("gemma", "draft-n2", GEMMA, ["-md", DRAFT, "--spec-type", "draft-mtp", "-ngld", "all", "--spec-draft-n-max", "2"]),
]
REPEATS = 3


def post(body):
    req = urllib.request.Request("http://127.0.0.1:8099/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))


def gpu():
    return subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,memory.used",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


def cool():
    while int(gpu().split(",")[0]) > 50:
        time.sleep(5)


out = open(os.path.join(OUT, "results.jsonl"), "w")
for model, arm, gguf, extra in ARMS:
    cmd = [BIN, "-m", gguf, "--host", "127.0.0.1", "--port", "8099",
           "-ngl", "99", "-c", "8192", "-ctk", "f16", "-ctv", "f16", "-fa", "1", "-t", "6", "-tb", "12",
           "--parallel", "1", "--reasoning", "off"] + extra
    cool()
    log = open(os.path.join(OUT, f"server-{model}-{arm}.log"), "w")
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            urllib.request.urlopen("http://127.0.0.1:8099/health", timeout=2)
            break
        except Exception:
            time.sleep(1)
    post({"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 16})  # warm-up
    vram = gpu()
    for mode in ("greedy", "sampled"):
        reps = REPEATS if mode == "greedy" else 1
        for name, prompt in PROMPTS.items():
            for r in range(reps):
                body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 256,
                        "cache_prompt": False}
                body.update({"temperature": 0} if mode == "greedy" else {"temperature": 0.8, "seed": 1234})
                d = post(body)
                t = d["timings"]
                text = d["choices"][0]["message"]["content"]
                rec = {"model": model, "arm": arm, "mode": mode, "prompt": name, "rep": r,
                       "decode_tps": t["predicted_per_second"], "n": t["predicted_n"],
                       "draft_n": t.get("draft_n", 0), "draft_accepted": t.get("draft_n_accepted", 0),
                       "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "text": text, "gpu": vram}
                out.write(json.dumps(rec) + "\n")
                out.flush()
                print(model, arm, mode, name, r, round(rec["decode_tps"], 1), rec["n"],
                      rec["draft_accepted"], "/", rec["draft_n"], vram, flush=True)
    srv.send_signal(signal.SIGTERM)
    srv.wait()
    log.close()
