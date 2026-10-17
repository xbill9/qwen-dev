"""Per-prompt comparison of this host's bf16 logits against the 3050 host's, same 65 prompts."""
import json
import os
import statistics as st

H = os.path.dirname(os.path.abspath(__file__))
a = {(r["subset"], r["id"]): r for r in map(json.loads, open(os.path.join(H, "logit_check-bf16-3050oem.jsonl")))}
b = {(r["subset"], r["id"]): r for r in map(json.loads, open(os.path.join(H, "logit_check-bf16-1650ti.jsonl")))}
ids = sorted(set(a) & set(b))
same_n = sum(a[i]["n_prompt"] == b[i]["n_prompt"] for i in ids)
argmax = lambda p: max(range(len(p)), key=p.__getitem__)  # noqa: E731
same_label = sum(argmax(a[i]["label_p_test"]) == argmax(b[i]["label_p_test"]) for i in ids)
d = [max(abs(x - y) for x, y in zip(a[i]["label_p_test"], b[i]["label_p_test"])) for i in ids]
ref_same = lambda r: sum(x["same_label"] for x in r.values())  # noqa: E731
out = {
    "prompts_in_both": len(ids),
    "same_prompt_length": same_n,
    "same_chosen_label_3050_vs_1650ti": same_label,
    "label_p_diff_3050_vs_1650ti_median": st.median(d),
    "label_p_diff_3050_vs_1650ti_max": max(d),
    "same_label_as_fp32_reference_3050": ref_same(a),
    "same_label_as_fp32_reference_1650ti": ref_same(b),
}
json.dump(out, open(os.path.join(H, "crosshost.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
