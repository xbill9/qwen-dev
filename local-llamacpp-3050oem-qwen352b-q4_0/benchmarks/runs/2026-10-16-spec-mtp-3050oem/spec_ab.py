"""Speculative decoding A/B on the RTX 3050 OEM: no draft vs Qwen3.5-2B's own MTP head
(`--spec-type draft-mtp` with no -md: llama.cpp drafts from the blk.24.nextn layer inside the
served GGUF) at several --spec-draft-n-max values. Same method as the Gemma E2B drafter run
(local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-10-13-spec-mtp-3050oem/spec_ab.py): one
llama-server per arm on port 8099, -c 8192, six prompts, greedy, max_tokens 256, prompt cache off,
3 repeats, then one sampled pass (temperature 0.8, seed 1234). Writes results.jsonl, with each
reply's text so the greedy replies can be compared across arms. Run from the rig root.
"""
import hashlib, json, os, signal, subprocess, time, urllib.request

RIG = os.getcwd()
OUT = os.path.join(RIG, "benchmarks/runs/2026-10-16-spec-mtp-3050oem")
env = {}
for line in open(os.path.join(RIG, "tpu.env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        env[k] = v.strip('"')
PROMPTS = {
    "code": "Write a Python function that checks whether a number is prime, with a docstring and two doctests.",
    "chat": "I just moved to a new city and feel a bit lonely. Any advice?",
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "explain": "Explain how a refrigerator works to a ten-year-old.",
    "facts": "List the planets of the solar system with one fact about each.",
    "summary": "Summarize the causes of the French Revolution in one paragraph.",
}
ARMS = [("none", None), ("mtp-n1", "1"), ("mtp-n2", "2"), ("mtp-n3", "3"), ("mtp-n4", "4"), ("mtp-n6", "6")]
REPEATS = 3


def post(body):
    req = urllib.request.Request("http://127.0.0.1:8099/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))


def gpu():
    return subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,memory.used",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


def cool():
    while int(gpu().split(",")[0]) > 40:
        time.sleep(5)


out = open(os.path.join(OUT, "results.jsonl"), "w")
for arm, n in ARMS:
    cmd = [env["LLAMA_SERVER_BIN"], "-m", env["MODEL_PATH"], "--host", "127.0.0.1", "--port", "8099",
           "-ngl", "99", "-c", "8192", "-ctk", "f16", "-ctv", "f16", "-fa", "1", "-t", "6", "-tb", "12",
           "--parallel", "1", "--reasoning", "off"]
    if n:
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", n]
    cool()
    log = open(os.path.join(OUT, f"server-{arm}.log"), "w")
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
                rec = {"arm": arm, "mode": mode, "prompt": name, "rep": r,
                       "decode_tps": t["predicted_per_second"], "n": t["predicted_n"],
                       "draft_n": t.get("draft_n", 0), "draft_accepted": t.get("draft_n_accepted", 0),
                       "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "text": text, "gpu": vram}
                out.write(json.dumps(rec) + "\n")
                out.flush()
                print(arm, mode, name, r, round(rec["decode_tps"], 1), rec["n"],
                      rec["draft_accepted"], "/", rec["draft_n"], flush=True)
    srv.send_signal(signal.SIGTERM)
    srv.wait()
    log.close()
