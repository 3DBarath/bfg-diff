"""BFG-Diff v3: Random-Anchor Self-Distillation Training (train_v3.py).

Memory-Safe fp16-Base / fp32-Adapter Training:
- Frozen base model in torch.float16 (requires_grad=False).
- Trainable WeightTiedTopKDrafter in torch.float32 (requires_grad=True).
- Forward under torch.autocast("cuda", dtype=torch.float16) with GradScaler.
- DFlash-Style Random-Anchor Parallel Block Distillation:
  Top-64 KL self-distillation + Cross-Entropy with D-PACE front-loaded loss weighting.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model_v3 import WeightTiedTopKDrafter, prepare_v3

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class OnPolicySequenceDataset(Dataset):
    """Dataset of on-policy token sequences."""

    def __init__(self, data_path_or_tensor, max_len: int = 256):
        if isinstance(data_path_or_tensor, (str, Path)):
            payload = torch.load(data_path_or_tensor, map_location="cpu")
            self.seqs = payload["sequences"][:, :max_len]
        else:
            self.seqs = data_path_or_tensor[:, :max_len]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]


def create_synthetic_data(num_samples: int = 64, seq_len: int = 128, vocab_size: int = 1000) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randint(100, vocab_size - 1, (num_samples, seq_len), dtype=torch.long)


def compute_anchor_distill_loss(
    base_model: nn.Module,
    drafter: WeightTiedTopKDrafter,
    batch_seqs: torch.Tensor,       # (B, S)
    mask_token_id: int,
    w: int = 7,
    n_anchors: int = 16,
    lambda_kl: float = 0.5,
    d_pace_decay: float = 0.85,
) -> Tuple[torch.Tensor, dict]:
    """DFlash-Style Random-Anchor Parallel Block Training:
    1. Base model no_grad forward pass: extracts Top-64 teacher logits and tapped hidden states.
    2. Sample n_anchors clean anchor positions.
    3. Construct length-W draft blocks [x_a, MASK, ..., MASK].
    4. Compute loss = 0.5 * Top64_KL + 0.5 * CE weighted by D-PACE.
    """
    B, S = batch_seqs.shape
    device = batch_seqs.device

    # 1. Base model teacher forward pass
    with torch.no_grad():
        out = base_model(batch_seqs, output_hidden_states=True)
        # Teacher logits for predicting positions [1 .. S-1]: out.logits[:, :S-1]
        teacher_logits = out.logits  # (B, S, vocab)
        # Top-64 extraction to save VRAM
        top64_vals, top64_idx = torch.topk(teacher_logits, k=64, dim=-1)  # (B, S, 64)
        top64_probs = F.softmax(top64_vals.float(), dim=-1)

    # Multi-layer target feature fusion (with grad enabled for fusion parameters!)
    f_ctx = drafter.fuse_stage1_features(out.hidden_states)  # (B, S, D)

    # 2. Sample random clean anchors
    # Each anchor must have at least 1 prefix token and W target tokens
    min_anchor = 2
    max_anchor = S - w - 2
    if max_anchor <= min_anchor:
        min_anchor = 1
        max_anchor = max(2, S - w - 1)

    anchors = torch.randint(min_anchor, max_anchor, (n_anchors,), device=device).tolist()

    total_kl_loss = 0.0
    total_ce_loss = 0.0
    acc_per_offset = [0.0] * w
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    # D-PACE position weighting: [1.0, 0.85, 0.70, 0.55, 0.40, 0.30, 0.20]
    d_pace = torch.tensor([d_pace_decay ** j for j in range(w)], device=device, dtype=torch.float32)
    d_pace = d_pace / d_pace.sum()

    for a in anchors:
        y_star = batch_seqs[:, a - 1 : a]  # concrete token at a - 1
        f_last = f_ctx[:, a - 1 : a]
        targets = batch_seqs[:, a : a + w]  # targets for predicting positions a .. a+w-1

        # Prefix KV
        pos_prefix = torch.arange(a, device=device).unsqueeze(0).expand(B, a)
        kp_list, vp_list = drafter.project_prefix_kv(f_ctx[:, :a], pos_prefix)

        # Draft forward pass
        attn_mask = torch.zeros(B, 1, w, a + w, dtype=drafter.fusion.norm.weight.dtype, device=device)
        student_logits = drafter.forward_draft(
            y_star_ids=y_star,
            mask_token_id=mask_token_id,
            f_ctx_last=f_last,
            cur_pos=a,
            w=w,
            k_prefix_list=kp_list,
            v_prefix_list=vp_list,
            attn_mask=attn_mask,
        )  # (B, W, vocab)

        # Teacher targets at positions a-1 .. a+w-2 predict tokens at a .. a+w-1
        t_top64_idx = top64_idx[:, a - 1 : a + w - 1]  # (B, W, 64)
        t_top64_probs = top64_probs[:, a - 1 : a + w - 1]  # (B, W, 64)

        # Student gathered on teacher Top-64
        s_gathered = torch.gather(student_logits.float(), dim=-1, index=t_top64_idx)  # (B, W, 64)
        s_log_probs = F.log_softmax(s_gathered, dim=-1)

        kl = F.kl_div(s_log_probs, t_top64_probs, reduction="none").sum(dim=-1)  # (B, W)
        ce = F.cross_entropy(student_logits.view(-1, student_logits.shape[-1]), targets.reshape(-1), reduction="none").view(B, w)

        anchor_loss = lambda_kl * kl + (1.0 - lambda_kl) * ce  # (B, W)
        weighted_anchor_loss = (anchor_loss * d_pace.unsqueeze(0)).sum(dim=-1).mean()
        total_loss = total_loss + weighted_anchor_loss

        # Track offset accuracy
        with torch.no_grad():
            preds = torch.argmax(student_logits, dim=-1)  # (B, W)
            correct = (preds == targets).float().mean(dim=0)  # (W,)
            for j in range(w):
                acc_per_offset[j] += correct[j].item()

    loss = total_loss / max(1, len(anchors))
    stats = {
        "loss": loss.item(),
        "acc_offset1": acc_per_offset[0] / max(1, len(anchors)),
        "acc_offset2": acc_per_offset[1] / max(1, len(anchors)) if w > 1 else 0.0,
        "acc_offset3": acc_per_offset[2] / max(1, len(anchors)) if w > 2 else 0.0,
        "acc_offset4": acc_per_offset[3] / max(1, len(anchors)) if w > 3 else 0.0,
    }
    return loss, stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--data", default=None, help="path to onpolicy .pt dataset")
    p.add_argument("--out", default="runs/bfg_v3")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--w", type=int, default=7, help="draft block length W")
    p.add_argument("--n_anchors", type=int, default=16, help="random anchors per sequence")
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=500)
    args = p.parse_args()

    if args.smoke_test:
        args.steps = 12
        args.batch_size = 2
        args.n_anchors = 2
        args.warmup = 2
        args.log_every = 1
        args.out = "runs/smoke_v3"
        if args.lr == 3e-4:
            args.lr = 2e-4
    return args


def main():
    args = parse_args()
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    print(f"BFG-Diff v3 Training | Model: {args.model} | Device: {device} | Steps: {args.steps} | LR: {args.lr}")

    # Prepare model + drafter
    base_dtype = torch.float16 if device == "cuda" else torch.float32
    base_model, drafter, tok, mask_id = prepare_v3(
        args.model, device=device, dtype=base_dtype, max_w=args.w + 1
    )

    # Print parameter breakdown
    bd = drafter.parameter_breakdown()
    print("Parameter Breakdown:")
    for k, v in bd.items():
        print(f"  {k}: {v}")

    # Optimizer on drafter params only
    trainable_params = drafter.get_trainable_parameters()
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    # Cosine scheduler with warmup
    def lr_lambda(step):
        if step < args.warmup:
            return float(step + 1) / max(1, args.warmup)
        progress = float(step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_fp16 = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    # Dataset
    if args.data and Path(args.data).exists():
        dataset = OnPolicySequenceDataset(args.data)
    else:
        dataset = OnPolicySequenceDataset(create_synthetic_data(num_samples=128, seq_len=96))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    loader_iter = iter(loader)

    Path(args.out).mkdir(parents=True, exist_ok=True)

    # Probe loss for smoke test convergence check
    probe_batch = next(loader_iter).to(device)
    with torch.no_grad():
        initial_loss, _ = compute_anchor_distill_loss(
            base_model, drafter, probe_batch, mask_id, w=args.w, n_anchors=2
        )
    before_loss = initial_loss.item()
    if args.smoke_test:
        print(f"smoke probe loss (before): {before_loss:.4f}")

    start_time = time.time()
    drafter.train()

    for step in range(1, args.steps + 1):
        try:
            batch = next(loader_iter).to(device)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter).to(device)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_fp16):
            loss, stats = compute_anchor_distill_loss(
                base_model, drafter, batch, mask_id, w=args.w, n_anchors=args.n_anchors
            )

        if use_fp16:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()

        sched.step()

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - start_time
            print(
                f"step {step:5d}/{args.steps} loss {stats['loss']:.4f} "
                f"lr {sched.get_last_lr()[0]:.2e} "
                f"A_+1: {stats['acc_offset1']:.1%} A_+2: {stats['acc_offset2']:.1%} A_+3: {stats['acc_offset3']:.1%} "
                f"elapsed {elapsed:.0f}s"
            )

        if step % args.save_every == 0 or step == args.steps:
            save_path = Path(args.out) / f"ckpt-{step}.pt"
            torch.save(
                {
                    "step": step,
                    "drafter_state_dict": drafter.state_dict(),
                    "args": vars(args),
                    "bd": bd,
                },
                save_path,
            )
            # Link/copy last.pt
            torch.save(
                {
                    "step": step,
                    "drafter_state_dict": drafter.state_dict(),
                    "args": vars(args),
                    "bd": bd,
                },
                Path(args.out) / "last.pt",
            )

    if args.smoke_test:
        with torch.no_grad():
            after_loss_t, _ = compute_anchor_distill_loss(
                base_model, drafter, probe_batch, mask_id, w=args.w, n_anchors=2
            )
        after_loss = after_loss_t.item()
        ok = math.isfinite(after_loss) and after_loss < before_loss
        print(f"smoke convergence: {before_loss:.4f} -> {after_loss:.4f} {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    print(f"Finished training. Saved checkpoint to {Path(args.out) / 'last.pt'}")


if __name__ == "__main__":
    main()
