"""Window arbitrary text into (prefix, target canvas) pairs for block diffusion."""
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


class GeneralTextDataset(Dataset):
    """root_dir: directory of text files, a single file, a list of strings, or None with
    `texts`. Binary files (NUL in the first 4 KiB) are skipped."""

    def __init__(
        self, root_dir=None, tokenizer=None, prefix_len=384, canvas_len=128, texts=None, limit=None
    ):
        assert tokenizer is not None, "tokenizer is required"
        raw = texts if texts is not None else _read_texts(root_dir, limit)
        sep = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        stream = []
        for text in raw:
            stream.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
            stream.extend(sep)
        need = prefix_len + canvas_len
        self.prefix_len, self.canvas_len = prefix_len, canvas_len
        self.chunks = [
            stream[i : i + need] for i in range(0, len(stream) - need + 1, canvas_len)
        ]
        assert self.chunks, f"no {need}-token windows (got {len(stream)} tokens)"

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        chunk = self.chunks[i]
        return (
            torch.tensor(chunk[: self.prefix_len], dtype=torch.long),
            torch.tensor(chunk[self.prefix_len :], dtype=torch.long),
        )


def _read_texts(root, limit):
    if isinstance(root, (list, tuple)):
        return list(root)[:limit]
    p = Path(root)
    files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())
    texts = []
    for f in files[:limit]:
        data = f.read_bytes()
        if b"\x00" in data[:4096]:  # binary
            continue
        texts.append(data.decode("utf-8", errors="replace"))
    return texts


SYNTHETIC_WORDS = (
    "the quick brown fox jumps over a lazy dog while researchers train compact "
    "diffusion language models on plain text and document what they observe"
).split()


def synthetic_texts(n=32, words=300, seed=0):
    rng = random.Random(seed)
    return [" ".join(rng.choice(SYNTHETIC_WORDS) for _ in range(words)) for _ in range(n)]
