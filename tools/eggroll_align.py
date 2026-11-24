#!/usr/bin/env python3
"""EGGROLL Direct-Alpha Post-Training for BFG-Diff v5.2b B-Fork Pass 2 (tools/eggroll_align.py).

Implements Move 3: Rank-1 Antithetic Evolution Strategies directly maximizing discrete
Q_A=16 tree-verified mean acceptance rate (alpha) over on-policy anchors.
Follows Sarkar et al. (2025/2026) "Evolution Strategies at the Hyperscale" (EGGROLL).

Key Design Properties:
1. Strict Anti-Leakage:
   - Disjoint train and validation anchors sampled strictly from --data (NEVER benchmark prompts).
2. Zero 28-Layer Base Model Calls inside ES Loop:
   - Caches exact greedy ground truth targets (W tokens via batched base_model.generate).
   - Caches Pass-1 representations (h_tap_p2, f_ctx_last, prefix KV projections) once upfront.
   - Deletes base_model from GPU VRAM prior to starting ES generations.
3. Fast Perturbation on 2D Weights:
   - Low-rank (rank-1) antithetic perturbations (a @ b) on refine_bridge and pass2_deltas[0].
   - High-rank accumulated gradient updates in fp32 master weights.
"""

import argparse
import copy
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model import build_bfork_token_embeds, prepare_v3
from sampler import build_forked_tree
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv


def parse_args():
    p = argparse.ArgumentParser(description="EGGROLL Rank-1 Direct-Alpha Post-Training for B-Fork Pass 2")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B", help="Base model identifier")
    p.add_argument("--checkpoint", default="runs/bfg_v52b_k2w5_1.5b/last.pt", help="Path to initial v5.2b drafter checkpoint")
    p.add_argument("--data", default="data/onpolicy_1.5b_15k_v5.pt", help="Path to on-policy sequences dataset")
    p.add_argument("--draft_vocab", default="data/draft_vocab_32k_v5.pt", help="Path to draft vocabulary tensor")
    p.add_argument("--max_draft_vocab", type=int, default=16384, help="Draft vocabulary size (default 16384)")
    p.add_argument("--out", default="runs/bfg_v52b_eggroll_1.5b", help="Output directory for aligned checkpoints")
    p.add_argument("--q_a", type=int, default=16, help="Tree verifier tile size (default 16)")
    p.add_argument("--pop_size", type=int, default=32, help="Antithetic population size (must be even, default 32)")
    p.add_argument("--generations", type=int, default=150, help="Number of ES generations (default 150)")
    p.add_argument("--sigma", type=float, default=0.005, help="Perturbation scale sigma (default 0.005)")
    p.add_argument("--lr", type=float, default=0.003, help="Learning rate for ES updates (default 0.003)")
    p.add_argument("--train_anchors", type=int, default=256, help="Number of disjoint training anchors (default 256)")
    p.add_argument("--val_anchors", type=int, default=128, help="Number of disjoint validation anchors (default 128)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    p.add_argument("--device", default="cuda", help="Execution device (cuda or cpu)")
    return p.parse_args()


def load_data_sequences(data_path: str, min_len: int = 128, max_slice: int = 256) -> torch.Tensor:
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    payload = torch.load(p, map_location="cpu")
    if isinstance(payload, dict):
        seqs = payload.get("sequences")
        seq_lens = payload.get("seq_lens")
    elif isinstance(payload, torch.Tensor):
        seqs = payload
        seq_lens = None
    else:
        raise ValueError(f"Unknown payload format in {data_path}")

    filtered = []
    N, S = seqs.shape
    for i in range(N):
        row = seqs[i]
        v_len = int(seq_lens[i].item()) if seq_lens is not None else S
        if v_len >= min_len:
            # Tail-slice to max_slice tokens
            start = max(0, v_len - max_slice)
            sl = row[start:v_len]
            if len(sl) >= min_len:
                filtered.append(sl[:max_slice])

    if not filtered:
        raise ValueError(f"No sequences with valid length >= {min_len} in {data_path}")
    print(f"[EGGROLL] Loaded {len(filtered)} sequences with length >= {min_len} (tail-sliced to {max_slice}).")
    return filtered


def simulate_tree_walk(tree_tokens: List[int], tree_parents: List[int], gt_targets: List[int], q_a: int) -> int:
    """Walk tree against exact greedy ground truth targets: returns accepted tokens count k."""
    curr_slot = 0
    k = 0
    while k < len(gt_targets):
        target_tok = gt_targets[k]
        matched_child = None
        for c_slot in range(curr_slot + 1, q_a):
            if tree_parents[c_slot] == curr_slot and tree_tokens[c_slot] == target_tok:
                matched_child = c_slot
                break
        if matched_child is not None:
            k += 1
            curr_slot = matched_child
        else:
            break
    return k


def evaluate_candidate_anchor(
    drafter,
    cache_item: dict,
    q_a: int,
    k_per_step: int = 3,
) -> Tuple[float, float]:
    """Execute Pass 2 for a single cached anchor, construct forked tree, and compute alpha."""
    W_p2 = drafter.pass2_w if getattr(drafter, "pass2_w", None) is not None else drafter.max_w - 1
    K = getattr(drafter, "fork_k", 2)
    device = cache_item["h_tap_p2"].device

    # Cached Pass-1 inputs
    tok_emb_2 = cache_item["tok_emb_2"]          # (K, W_p2, D)
    h_tap_p2 = cache_item["h_tap_p2"]            # (K, W_p2, D)
    f_ctx_last = cache_item["f_ctx_last"]        # (K, 1, D)
    pos_draft_p2 = cache_item["pos_draft_p2"]    # (K, W_p2)
    attn_mask_p2 = cache_item["attn_mask_p2"]    # (K, 1, W_p2, W_p2 + S)
    k_pref_list = cache_item["k_prefix_p2"]      # List of (K, num_kv_heads, S, head_dim)
    v_pref_list = cache_item["v_prefix_p2"]      # List of (K, num_kv_heads, S, head_dim)

    # 1. Refine bridge
    h2 = drafter.refine_bridge(tok_emb_2, h_tap_p2, f_ctx_last)

    # 2. Dedicated Pass-2 layers
    p2_start = drafter.k_layers - drafter.pass2_layers
    num_heads = drafter.num_heads
    num_kv_heads = drafter.num_kv_heads
    head_dim = drafter.head_dim
    dim = drafter.dim

    for p_idx in range(drafter.pass2_layers):
        layer_idx = p2_start + p_idx
        if drafter.has_ghost_layers and getattr(drafter, "pass2_ghost_layers", None) is not None:
            ghost_l = drafter.pass2_ghost_layers[p_idx]
            norm_h2 = ghost_l.input_layernorm(h2)
            q = ghost_l.self_attn.q_proj(norm_h2).view(K, W_p2, num_heads, head_dim).transpose(1, 2)
            k_d = ghost_l.self_attn.k_proj(norm_h2).view(K, W_p2, num_kv_heads, head_dim).transpose(1, 2)
            v_d = ghost_l.self_attn.v_proj(norm_h2).view(K, W_p2, num_kv_heads, head_dim).transpose(1, 2)
        else:
            base_l = drafter.draft_layers[layer_idx]
            deltas = drafter.pass2_deltas[p_idx]
            norm_h2 = base_l.input_layernorm(h2)
            from model import apply_linear_with_delta
            q = apply_linear_with_delta(base_l.self_attn.q_proj, norm_h2, deltas.q_proj)
            q = q.view(K, W_p2, num_heads, head_dim).transpose(1, 2)
            k_d = apply_linear_with_delta(base_l.self_attn.k_proj, norm_h2, deltas.k_proj)
            k_d = k_d.view(K, W_p2, num_kv_heads, head_dim).transpose(1, 2)
            v_d = apply_linear_with_delta(base_l.self_attn.v_proj, norm_h2, deltas.v_proj)
            v_d = v_d.view(K, W_p2, num_kv_heads, head_dim).transpose(1, 2)

        cos_d, sin_d = drafter.rotary_emb(norm_h2, pos_draft_p2)
        q, k_d = apply_rotary_pos_emb(q, k_d, cos_d, sin_d, unsqueeze_dim=1)

        k_pref = k_pref_list[p_idx]
        v_pref = v_pref_list[p_idx]
        k_total = torch.cat([k_d, k_pref], dim=2)
        v_total = torch.cat([v_d, v_pref], dim=2)

        if num_heads != num_kv_heads:
            if getattr(drafter, "supports_enable_gqa", False):
                attn_out = F.scaled_dot_product_attention(q, k_total, v_total, attn_mask=attn_mask_p2, enable_gqa=True)
            else:
                n_rep = num_heads // num_kv_heads
                k_attn = repeat_kv(k_total, n_rep)
                v_attn = repeat_kv(v_total, n_rep)
                attn_out = F.scaled_dot_product_attention(q, k_attn, v_attn, attn_mask=attn_mask_p2)
        else:
            attn_out = F.scaled_dot_product_attention(q, k_total, v_total, attn_mask=attn_mask_p2)
        attn_out = attn_out.transpose(1, 2).reshape(K, W_p2, dim)

        if drafter.has_ghost_layers and getattr(drafter, "pass2_ghost_layers", None) is not None:
            h2 = h2 + ghost_l.self_attn.o_proj(attn_out)
            norm_post = ghost_l.post_attention_layernorm(h2)
            gate = ghost_l.mlp.gate_proj(norm_post)
            up = ghost_l.mlp.up_proj(norm_post)
            mlp_act = ghost_l.mlp.act_fn(gate) * up
            h2 = h2 + ghost_l.mlp.down_proj(mlp_act)
        else:
            from model import apply_linear_with_delta
            h2 = h2 + apply_linear_with_delta(base_l.self_attn.o_proj, attn_out, deltas.o_proj)
            norm_post = base_l.post_attention_layernorm(h2)
            gate = apply_linear_with_delta(base_l.mlp.gate_proj, norm_post, deltas.gate_proj)
            up = apply_linear_with_delta(base_l.mlp.up_proj, norm_post, deltas.up_proj)
            mlp_act = base_l.mlp.act_fn(gate) * up
            h2 = h2 + apply_linear_with_delta(base_l.mlp.down_proj, mlp_act, deltas.down_proj)

    # 3. Project suffix branch logits
    h2_norm = drafter.norm(h2[:, 1:])
    draft_embed_w = cache_item["draft_embed_w"]
    if draft_embed_w is not None:
        branch_logits = F.linear(h2_norm, draft_embed_w)
    else:
        branch_logits = drafter.lm_head(h2_norm)

    # 4. Synthesize forked tree
    root_logits = cache_item["root_logits"]
    root_local_ids = cache_item["root_local_ids"]
    root_global_ids = cache_item["root_global_ids"]
    draft_vocab_ids = cache_item["draft_vocab_ids"]

    t_tokens, t_parents, t_depths = build_forked_tree(
        root_logits=root_logits,
        root_local_ids=root_local_ids,
        root_global_ids=root_global_ids,
        branch_logits=branch_logits,
        q_a=q_a,
        draft_vocab_ids=draft_vocab_ids,
        k_per_step=k_per_step,
        root_hedge_k=4,
    )

    # 5. Measure accepted tokens k
    gt_targets = cache_item["gt_targets"]
    k_acc = simulate_tree_walk(t_tokens, t_parents, gt_targets, q_a)
    alpha = float(k_acc + 1)

    # Confidence margin tiebreaker
    top2_s0 = torch.topk(branch_logits[:, 0], k=2, dim=-1).values
    margin = float((top2_s0[:, 0] - top2_s0[:, 1]).mean().item())

    return alpha, margin


def main():
    args = parse_args()
    if args.pop_size % 2 != 0:
        raise ValueError(f"--pop_size must be an even integer for antithetic pairs, got {args.pop_size}")

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("=" * 80)
    print(f"BFG-Diff v5.2b EGGROLL Direct-Alpha Post-Training")
    print(f"Model: {args.model} | Device: {device} | Generations: {args.generations} | Pop: {args.pop_size}")
    print(f"Anchors: {args.train_anchors} train / {args.val_anchors} val | Sigma: {args.sigma} | LR: {args.lr}")
    print("=" * 80)

    # 1. Load initial checkpoint and drafter
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    ckpt_payload = torch.load(ckpt_path, map_location=device)
    arch = ckpt_payload.get("arch", {})
    print(f"[EGGROLL] Checkpoint architecture metadata: {arch}")

    base_dtype = torch.float16 if device == "cuda" else torch.float32
    base_model, drafter, tok, mask_id = prepare_v3(
        args.model,
        device=device,
        dtype=base_dtype,
        k_layers=arch.get("k_layers", 4),
        drafter_r=arch.get("r", 32),
        drafter_alpha=arch.get("alpha", 64.0),
        num_draft_steps=arch.get("num_draft_steps", 2),
        soft_topk=arch.get("soft_topk", 4),
        pass2_layers=arch.get("pass2_layers", 1),
        refine_bottleneck=arch.get("refine_bottleneck", 256),
        pass2_mode="bfork",
        fork_k=arch.get("fork_k", 2),
        pass2_w=arch.get("pass2_w", 5),
    )

    drafter.load_state_dict(ckpt_payload["drafter_state_dict"], strict=False)
    print(f"[EGGROLL] Loaded drafter weights from {args.checkpoint}.")
    drafter.eval()
    base_model.eval()

    # Load and slice draft vocabulary
    draft_vocab_ids = None
    draft_embed_w = None
    if args.draft_vocab and Path(args.draft_vocab).exists():
        loaded = torch.load(args.draft_vocab, map_location=device)
        d_ids = loaded["draft_vocab_ids"] if isinstance(loaded, dict) else loaded
        if args.max_draft_vocab and args.max_draft_vocab < len(d_ids):
            vocab_special = [0, 1, 2, mask_id, tok.eos_token_id]
            special_set = {s for s in vocab_special if s is not None}
            sliced_tokens = []
            seen = set()
            for s in special_set:
                sliced_tokens.append(int(s))
                seen.add(int(s))
            for tid in d_ids.tolist():
                tid = int(tid)
                if tid not in seen:
                    sliced_tokens.append(tid)
                    seen.add(tid)
                if len(sliced_tokens) >= args.max_draft_vocab:
                    break
            draft_vocab_ids = torch.tensor(sliced_tokens, dtype=torch.long, device=device)
        else:
            draft_vocab_ids = d_ids.to(device)
        print(f"[EGGROLL] Draft vocab initialized with {len(draft_vocab_ids)} tokens.")
        draft_embed_w = drafter.embed_tokens.weight[draft_vocab_ids]

    # 2. Sample Disjoint Train and Val Anchors
    sequences = load_data_sequences(args.data, min_len=128, max_slice=256)
    total_needed = args.train_anchors + args.val_anchors
    if len(sequences) < total_needed:
        # Re-sample with different anchor offsets across sequences
        pass

    indices = list(range(len(sequences)))
    random.shuffle(indices)
    train_seqs = [sequences[i] for i in indices[:args.train_anchors]]
    val_seqs = [sequences[i] for i in indices[args.train_anchors : args.train_anchors + args.val_anchors]]

    W = 7
    W_p2 = drafter.pass2_w if getattr(drafter, "pass2_w", None) is not None else 5
    K = getattr(drafter, "fork_k", 2)

    def cache_anchor_subset(seq_list: List[torch.Tensor], desc: str) -> List[dict]:
        print(f"[EGGROLL] Building exact greedy cache for {len(seq_list)} {desc} anchors...")
        cached_items = []
        batch_size = 32

        for b_start in range(0, len(seq_list), batch_size):
            b_seqs = seq_list[b_start : b_start + batch_size]
            for seq in b_seqs:
                S_len = len(seq)
                # Select random clean anchor position t
                min_t = 32
                max_t = S_len - W - 2
                if max_t <= min_t:
                    t = min_t
                else:
                    t = random.randint(min_t, max_t)

                input_prompt = seq[: t + 1].unsqueeze(0).to(device)  # (1, t + 1)
                with torch.no_grad():
                    # Generate exact greedy targets from frozen base_model
                    gen_out = base_model.generate(input_prompt, max_new_tokens=W, do_sample=False)
                    gt_tokens = gen_out[0, t + 1 : t + 1 + W].tolist()

                    # Base model forward for prefix features
                    out = base_model(input_prompt, output_hidden_states=True)
                    f_ctx = drafter.fuse_stage1_features(out.hidden_states)
                    f_last = f_ctx[:, t - 1 : t]
                    y_star = input_prompt[:, t : t + 1]

                    pos_all = torch.arange(t + 1, device=device).unsqueeze(0)
                    kp_all, vp_all = drafter.project_prefix_kv(f_ctx, pos_all)

                    # Draft Pass 1
                    y_star_emb = drafter.embed_tokens(y_star)
                    mask_tensor = torch.tensor([[mask_id]], dtype=torch.long, device=device)
                    mask_emb = drafter.embed_tokens(mask_tensor)
                    h = drafter.bridge(y_star_emb, mask_emb, f_last, w=W)
                    pos_draft = torch.arange(t, t + W, device=device).unsqueeze(0)

                    attn_mask = torch.full((1, 1, W, t + 1 + W), -1e4, dtype=f_ctx.dtype, device=device)
                    attn_mask[0, 0, :, :t] = 0.0
                    attn_mask[0, 0, :, t + 1 : t + 1 + W] = 0.0

                    h_tap = None
                    p2_start = drafter.k_layers - drafter.pass2_layers
                    for layer_idx in range(drafter.k_layers):
                        if layer_idx == p2_start:
                            h_tap = h
                        h, _, _ = drafter.forward_draft_layer(
                            layer_idx=layer_idx,
                            hidden_states=h,
                            pos_draft=pos_draft,
                            k_prefix=kp_all[layer_idx],
                            v_prefix=vp_all[layer_idx],
                            attn_mask=attn_mask,
                        )

                    # Pass 1 slot 0 prediction
                    h1_s0 = drafter.norm(h[:, 0:1])
                    if draft_embed_w is not None:
                        z1_s0 = F.linear(h1_s0, draft_embed_w)
                    else:
                        z1_s0 = drafter.lm_head(h1_s0)

                    topk_root = torch.topk(z1_s0[0, 0], k=K)
                    root_local_ids = topk_root.indices
                    root_global_ids = draft_vocab_ids[root_local_ids] if draft_vocab_ids is not None else root_local_ids
                    root_logits = z1_s0[0, 0]

                    # Conditioned Pass 2 inputs
                    d0_embs = F.embedding(root_local_ids, draft_embed_w if draft_embed_w is not None else drafter.embed_tokens.weight).unsqueeze(1)
                    tok_emb_2 = build_bfork_token_embeds(y_star_emb.expand(K, 1, -1), d0_embs, mask_emb.expand(K, 1, -1), W_p2)
                    h_tap_p2 = (h_tap if h_tap is not None else h)[:, :W_p2, :].expand(K, -1, -1)
                    f_ctx_last_exp = f_last.expand(K, -1, -1)
                    pos_draft_p2 = pos_draft[:, :W_p2].expand(K, -1)

                    attn_mask_p2 = torch.cat([attn_mask[:, :, :W_p2, :t], attn_mask[:, :, :W_p2, t + 1 : t + 1 + W_p2]], dim=-1).expand(K, -1, -1, -1)
                    k_prefix_p2 = [kp_all[l][:, :, :t, :].expand(K, -1, -1, -1) for l in range(p2_start, drafter.k_layers)]
                    v_prefix_p2 = [vp_all[l][:, :, :t, :].expand(K, -1, -1, -1) for l in range(p2_start, drafter.k_layers)]

                    cached_items.append({
                        "tok_emb_2": tok_emb_2.detach(),
                        "h_tap_p2": h_tap_p2.detach(),
                        "f_ctx_last": f_ctx_last_exp.detach(),
                        "pos_draft_p2": pos_draft_p2.detach(),
                        "attn_mask_p2": attn_mask_p2.detach(),
                        "k_prefix_p2": [kp.detach() for kp in k_prefix_p2],
                        "v_prefix_p2": [vp.detach() for vp in v_prefix_p2],
                        "root_logits": root_logits.detach(),
                        "root_local_ids": root_local_ids.detach(),
                        "root_global_ids": root_global_ids.detach(),
                        "gt_targets": gt_tokens,
                        "draft_vocab_ids": draft_vocab_ids,
                        "draft_embed_w": draft_embed_w,
                    })

        return cached_items

    train_cache = cache_anchor_subset(train_seqs, "train")
    val_cache = cache_anchor_subset(val_seqs, "validation")

    # 3. Free base model from VRAM completely!
    print("[EGGROLL] Freeing 28-layer base model from GPU VRAM...")
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[EGGROLL] VRAM after base model cleanup: {torch.cuda.memory_allocated() / (1024**2):.1f} MB.")

    # 4. Identify Target Parameters for EGGROLL Perturbations
    # Targets: refine_bridge (down.weight, up.weight) and pass2_deltas[0] (lora_A, lora_B on all projections)
    target_params: Dict[str, nn.Parameter] = {}

    if hasattr(drafter.refine_bridge, "down"):
        target_params["refine_bridge.down.weight"] = drafter.refine_bridge.down.weight
    if hasattr(drafter.refine_bridge, "up"):
        target_params["refine_bridge.up.weight"] = drafter.refine_bridge.up.weight

    if hasattr(drafter, "pass2_deltas") and drafter.pass2_deltas is not None:
        p2_delta = drafter.pass2_deltas[0]
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
            proj = getattr(p2_delta, proj_name, None)
            if proj is not None:
                if hasattr(proj, "lora_A"):
                    target_params[f"pass2_deltas.0.{proj_name}.lora_A"] = proj.lora_A
                if hasattr(proj, "lora_B"):
                    target_params[f"pass2_deltas.0.{proj_name}.lora_B"] = proj.lora_B

    print(f"[EGGROLL] Target matrices for rank-1 ES optimization: {len(target_params)} tensors:")
    for name, p in target_params.items():
        print(f"  {name}: shape {tuple(p.shape)}")

    # Master fp32 weights
    master_weights = {name: p.data.clone().float() for name, p in target_params.items()}

    # Baseline evaluation on train and val
    def evaluate_set(dataset_cache: List[dict], desc: str) -> float:
        alphas = []
        for item in dataset_cache:
            a, _ = evaluate_candidate_anchor(drafter, item, q_a=args.q_a)
            alphas.append(a)
        mean_a = sum(alphas) / len(alphas)
        print(f"[EGGROLL] Baseline {desc} Mean Alpha: {mean_a:.4f} (over {len(alphas)} anchors)")
        return mean_a

    init_train_a = evaluate_set(train_cache, "Train")
    init_val_a = evaluate_set(val_cache, "Val")
    best_val_a = init_val_a

    Path(args.out).mkdir(parents=True, exist_ok=True)

    # 5. EGGROLL Optimization Loop
    print("\n[EGGROLL] Starting Antithetic Rank-1 Evolution Generations...")
    half_pop = args.pop_size // 2

    for gen in range(1, args.generations + 1):
        t_gen_start = time.perf_counter()

        # Sample rank-1 perturbations for each target matrix
        # (a_k, b_k) with a_k ~ N(0, 1), b_k ~ N(0, 1)
        perturbations_per_param = {}
        for name, param in target_params.items():
            m, n = param.shape
            # (half_pop, m, 1) and (half_pop, 1, n)
            A = torch.randn(half_pop, m, 1, device=device)
            B = torch.randn(half_pop, 1, n, device=device)
            # Rank-1 matrix: a @ b, normalized by sqrt(m * n) for stability
            E_half = torch.bmm(A, B) / math.sqrt(n)
            # Antithetic pair: [+E, -E] -> total pop_size
            E_full = torch.cat([E_half, -E_half], dim=0)  # (pop_size, m, n)
            perturbations_per_param[name] = E_full

        # Evaluate fitness for each candidate in population
        candidate_fitness = [0.0] * args.pop_size
        candidate_alphas = [0.0] * args.pop_size

        for i in range(args.pop_size):
            # Apply perturbation i
            for name, param in target_params.items():
                p_pert = master_weights[name].to(device) + args.sigma * perturbations_per_param[name][i]
                param.data.copy_(p_pert.to(param.dtype))

            # Evaluate on train subset (sampled batch of 64 anchors per generation for speed)
            sample_anchors = random.sample(train_cache, min(64, len(train_cache)))
            gen_alphas = []
            gen_margins = []
            for item in sample_anchors:
                a, m = evaluate_candidate_anchor(drafter, item, q_a=args.q_a)
                gen_alphas.append(a)
                gen_margins.append(m)

            mean_alpha = sum(gen_alphas) / len(gen_alphas)
            mean_margin = sum(gen_margins) / len(gen_margins)
            fitness = mean_alpha + 1e-3 * mean_margin

            candidate_fitness[i] = fitness
            candidate_alphas[i] = mean_alpha

        # Compute normalized fitness weights
        fitness_tensor = torch.tensor(candidate_fitness, device=device, dtype=torch.float32)
        fit_std = fitness_tensor.std()
        if fit_std > 1e-6:
            f_hat = (fitness_tensor - fitness_tensor.mean()) / fit_std
        else:
            f_hat = fitness_tensor - fitness_tensor.mean()

        # Apply EGGROLL parameter update in master fp32 weights
        for name, param in target_params.items():
            # sum_i f_hat[i] * E_i
            E = perturbations_per_param[name]  # (pop_size, m, n)
            grad_est = (f_hat.view(-1, 1, 1) * E).mean(dim=0) / args.sigma
            master_weights[name] += args.lr * grad_est.to(master_weights[name].device)
            param.data.copy_(master_weights[name].to(device, dtype=param.dtype))

        elapsed_gen = time.perf_counter() - t_gen_start
        best_cand_a = max(candidate_alphas)
        mean_cand_a = sum(candidate_alphas) / len(candidate_alphas)

        if gen % 5 == 0 or gen == args.generations:
            print(f"Gen {gen:3d}/{args.generations} | Mean Fit Alpha: {mean_cand_a:.4f} | Best Pop Alpha: {best_cand_a:.4f} | Time: {elapsed_gen:.1f}s")

        # Validation Checkpoint
        if gen % 10 == 0 or gen == args.generations:
            val_a = evaluate_set(val_cache, f"Val @ Gen {gen}")
            if val_a >= best_val_a or gen == args.generations:
                best_val_a = val_a
                save_payload = {
                    "step": gen,
                    "drafter_state_dict": {
                        k: v for k, v in drafter.state_dict().items()
                        if k.startswith(("fusion.", "bridge.", "refine_bridge.", "layer_deltas.", "pass2_deltas."))
                    },
                    "arch": arch,
                    "val_alpha": val_a,
                }
                save_file = Path(args.out) / "last.pt"
                torch.save(save_payload, save_file)
                print(f"  [SAVED] Checkpoint saved to {save_file} (Val Alpha: {val_a:.4f}).")

    print("\n" + "=" * 80)
    print(f"EGGROLL Direct-Alpha Post-Training Complete!")
    print(f"Initial Train Alpha: {init_train_a:.4f} -> Best Val Alpha: {best_val_a:.4f}")
    print(f"Optimized Checkpoint: {Path(args.out) / 'last.pt'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
