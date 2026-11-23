#!/usr/bin/env python3
"""Build a frequency-pruned draft vocabulary tensor (draft_vocab_ids.pt).

Counts token frequencies on an on-policy dataset (or raw sequences tensor),
selects the top V_d - K most frequent token IDs, ensures essential control tokens
(EOS, PAD, <|im_start|>, <|im_end|>, <|endoftext|>) are included, and backfills
from lowest unused BPE IDs if the dataset has fewer than V_d unique tokens.
"""

import argparse
from pathlib import Path
from typing import Set
import torch
from transformers import AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B"
DEFAULT_CONTROL_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
]


def build_draft_vocab(
    data_path: str,
    output_path: str,
    v_d: int = 32768,
    model_name: str = DEFAULT_MODEL,
):
    print(f"[build_draft_vocab] Loading dataset from {data_path}...")
    payload = torch.load(data_path, map_location="cpu")

    if isinstance(payload, dict):
        seqs = payload.get("sequences")
        seq_lens = payload.get("seq_lens")
    elif isinstance(payload, torch.Tensor):
        seqs = payload
        seq_lens = None
    else:
        raise ValueError(f"Unsupported payload type in {data_path}: {type(payload)}")

    print(f"[build_draft_vocab] Loading tokenizer {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name)
    vocab_size = tok.vocab_size

    # Collect essential control tokens
    control_ids: Set[int] = set()
    if tok.eos_token_id is not None:
        control_ids.add(tok.eos_token_id)
    if tok.pad_token_id is not None:
        control_ids.add(tok.pad_token_id)

    for c_tok in DEFAULT_CONTROL_TOKENS:
        tid = tok.convert_tokens_to_ids(c_tok)
        if tid is not None and isinstance(tid, int) and tid < vocab_size:
            control_ids.add(tid)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # Count token frequencies, respecting seq_lens when present
    print("[build_draft_vocab] Counting token frequencies across sequences...")
    token_counts = torch.zeros(vocab_size, dtype=torch.int64)

    N, S = seqs.shape
    for r in range(N):
        row = seqs[r]
        if seq_lens is not None:
            valid_len = min(int(seq_lens[r].item()), S)
            valid_tokens = row[:valid_len]
        else:
            if pad_id is not None:
                valid_tokens = row[row != pad_id]
            else:
                valid_tokens = row
        if valid_tokens.numel() > 0:
            counts = torch.bincount(valid_tokens.long(), minlength=vocab_size)
            token_counts[:len(counts)] += counts

    # Exclude control tokens from ranking pool to add them unconditionally
    for cid in control_ids:
        token_counts[cid] = -1

    # Desired total tokens
    target_v_d = min(v_d, vocab_size)
    needed_ranked = target_v_d - len(control_ids)

    # Top frequent tokens
    ranked_indices = torch.argsort(token_counts, descending=True)
    selected_from_data = ranked_indices[:needed_ranked].tolist()

    selected_ids: Set[int] = set(control_ids)
    for tid in selected_from_data:
        if token_counts[tid] >= 0:  # Valid token present in data
            selected_ids.add(tid)

    # Backfill if unique count < target_v_d (e.g. synthetic or small datasets)
    if len(selected_ids) < target_v_d:
        for tid in range(vocab_size):
            if tid not in selected_ids:
                selected_ids.add(tid)
            if len(selected_ids) >= target_v_d:
                break

    # Final sorted 1D tensor
    final_vocab = torch.tensor(sorted(list(selected_ids)), dtype=torch.long)
    assert len(final_vocab) == target_v_d, f"Expected {target_v_d} IDs, got {len(final_vocab)}"

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_vocab, out_file)
    print(f"[build_draft_vocab] Saved {len(final_vocab)} draft vocab IDs to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frequency-pruned draft vocabulary.")
    parser.add_argument("--data", type=str, default="data/onpolicy_1.5b_15k_clean.pt", help="Path to clean training data")
    parser.add_argument("--out", type=str, default="data/draft_vocab_32k.pt", help="Output tensor file")
    parser.add_argument("--v_d", type=int, default=32768, help="Draft vocabulary size (default: 32768)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name for tokenizer")
    args = parser.parse_args()

    build_draft_vocab(data_path=args.data, output_path=args.out, v_d=args.v_d, model_name=args.model)
