"""Re-derive the resident-vs-lazy memory split from the GGUF artifact itself.

Carried from the Gemma sibling, where per_layer_token_embd is TENSOR_READ_LAZY and
stays in the mmap. Qwen3.5 (src/models/qwen35.cpp) creates no lazy tensor, so here
the whole file is resident; the script still checks that claim against the artifact.

Reads gguf-py out of the llama.cpp checkout named in tpu.env — the vendored copy
is the primary path; the pip package is the fallback.

Usage:  python3 inspect_gguf.py [path/to/model.gguf]
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

RIG_DIR = Path(__file__).resolve().parent
load_dotenv(RIG_DIR / "tpu.env")

# Tensors llama.cpp reads lazily for this architecture: none for qwen35 at
# 7182ef71e (`grep -rn TENSOR_READ_LAZY src/` hits gemma4.cpp and qwen4exp.cpp
# only). Re-check after a llama.cpp update.
LAZY_TENSORS: set[str] = set()


def main() -> int:
    llama_dir = os.environ.get("LLAMA_CPP_DIR", "")
    if llama_dir:
        sys.path.insert(0, str(Path(llama_dir) / "gguf-py"))
    try:
        from gguf import GGUFReader
    except ImportError:
        print("gguf-py not importable. Set LLAMA_CPP_DIR in tpu.env, or: pip install gguf pyyaml")
        return 1

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("MODEL_PATH", ""))
    if not path.exists():
        print(f"not found: {path}")
        return 1

    reader = GGUFReader(str(path))
    # .name, not str() — GGMLQuantizationType is an IntEnum, and on 3.11+ str()
    # renders the number, which turns "Q6_K" into a bare "14" in the output.
    rows = [(int(t.n_bytes), t.name, t.tensor_type.name, list(map(int, t.shape)))
            for t in reader.tensors]
    rows.sort(reverse=True)

    total = sum(r[0] for r in rows)
    lazy = sum(nb for nb, name, _, _ in rows if name in LAZY_TENSORS)
    resident = total - lazy

    print(f"{path.name}")
    print(f"  tensors:  {len(rows)}")
    print(f"  total:    {total / 1e9:.3f} GB")
    print()
    print("  largest tensors:")
    for nb, name, ty, shape in rows[:6]:
        mark = "  <- LAZY, host-resident" if name in LAZY_TENSORS else ""
        print(f"    {nb / 1e6:8.1f} MB  {name:<28} {ty:<7} {shape}{mark}")
    print()
    print(f"  lazy (never on GPU):        {lazy / 1e9:.3f} GB  ({100 * lazy / total:.0f}% of file)")
    print(f"  must be resident:           {resident / 1e9:.3f} GB = {resident / 2**30:.2f} GiB")

    by_type: dict[str, int] = {}
    for nb, _, ty, _ in rows:
        by_type[ty] = by_type.get(ty, 0) + nb
    print()
    print("  by tensor type (the slot-5 token is q4_0; the file mostly is not):")
    for ty, nb in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print(f"    {ty:<8} {nb / 1e9:6.3f} GB  ({100 * nb / total:4.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
