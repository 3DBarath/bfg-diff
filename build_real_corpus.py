"""Build real multi-domain corpus for BFG-Diff distillation."""
import os
from pathlib import Path

def prepare_real_corpus(out_dir="/tmp/bfg-real-corpus", max_chars=12000000):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / "corpus.txt"

    total_chars = 0
    collected = []

    # 1. Search /usr/share/doc for markdown, text, rst
    doc_paths = sorted(Path("/usr/share/doc").rglob("*"))
    for p in doc_paths:
        if not p.is_file() or p.is_symlink():
            continue
        if p.suffix.lower() in [".txt", ".md", ".rst"] or "README" in p.name.upper():
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                if len(content) > 100:
                    collected.append(content)
                    total_chars += len(content)
                    if total_chars >= max_chars:
                        break
            except Exception:
                continue

    # 2. Add python standard library docstrings / text if needed
    if total_chars < max_chars:
        py_dirs = [Path("/usr/lib64/python3.12"), Path("/usr/lib/python3.12")]
        for d in py_dirs:
            if not d.exists():
                continue
            for p in sorted(d.rglob("*.py")):
                try:
                    content = p.read_text(encoding="utf-8", errors="ignore")
                    if len(content) > 200:
                        collected.append(content)
                        total_chars += len(content)
                        if total_chars >= max_chars:
                            break
                except Exception:
                    continue
            if total_chars >= max_chars:
                break

    with open(out_file, "w", encoding="utf-8") as out_f:
        for item in collected:
            out_f.write(item + "\n\n")

    print(f"Wrote {len(collected)} documents ({total_chars:,} characters) to {out_file}")
    return str(out_file)

if __name__ == "__main__":
    prepare_real_corpus()
