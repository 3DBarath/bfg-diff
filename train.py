"""Train Qwen2.5 as a block-autoregressive discrete diffusion LM (BD3-LM style)."""
import argparse
import math
import os
import sys
import time
from itertools import cycle
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset import GeneralTextDataset, synthetic_texts
from model import LORA_TARGETS, LORA_TARGETS_MLP, apply_lora, canvas_forward, prepare


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--data", default=None, help="text dir/file, a HF dataset is not used here")
    p.add_argument("--out", default="runs/bfg-diff")
    p.add_argument("--prefix_len", type=int, default=384)
    p.add_argument("--canvas_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=None, help="default 5e-5 full FT, 5e-4 LoRA")
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_batches", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="max source files")
    p.add_argument("--max_tokens", type=int, default=None, help="cap on corpus tokens")
    p.add_argument("--texts", type=int, default=32, help="synthetic texts when --data is unset")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"])
    p.add_argument("--threads", type=int, default=0, help="CPU threads (0 = all cores)")
    p.add_argument("--cpu", action="store_true", help="CPU-friendly preset (P=128, C=32, 1k steps)")
    p.add_argument("--lora", action="store_true", help="train LoRA adapters, freeze the base model")
    p.add_argument("--lora_mlp", action="store_true",
                   help="also adapt gate_proj/up_proj/down_proj (stage 3 ablation)")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None, help="warm-start checkpoint (e.g. runs/bfg-shifted/last.pt)")
    p.add_argument("--smoke_test", action="store_true", help="tiny synthetic run + convergence check")
    if "--cpu" in sys.argv:  # preset only changes defaults; explicit flags still win
        p.set_defaults(
            device="cpu", dtype="fp32", prefix_len=128, canvas_len=32,
            batch_size=2, grad_accum=8, steps=1000, save_every=200,
            log_every=20, eval_every=200, warmup=20, lora=True,
        )
    args = p.parse_args()
    if args.lora_mlp:
        args.lora = True
    if args.lr is None:
        args.lr = 5e-4 if args.lora else 5e-5
    if args.smoke_test:
        args.steps = 12
        args.prefix_len, args.canvas_len = 128, 64
        args.batch_size, args.grad_accum = 2, 1
        args.warmup, args.log_every = 2, 1
        args.save_every, args.eval_every = 6, 12
        args.out = "runs/smoke"
    return args


def pick_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_mask(target, mask_id):
    """Staircase-Aligned Mask Curriculum:
    - 50% Contiguous Lookahead Window Mask: target[:, :m] clean, target[:, m:] masked.
    - 30% Simulated Draft Corruption + Suffix Mask: target[:, :m] clean, 1-2 corrupted drafts at m..m+2, and target[:, m+3:] masked.
    - 20% Full or Random Scatter Mask.
    Returns: noisy_target, mask, frontiers
    """
    B, C = target.shape
    device = target.device
    noisy = target.clone()
    mask = torch.zeros_like(target, dtype=torch.bool)
    frontiers = []

    for i in range(B):
        u = torch.rand(1).item()
        if u < 0.50:
            m = torch.randint(0, max(1, C - 1), (1,)).item()
            mask[i, m:] = True
            noisy[i, m:] = mask_id
            frontiers.append(m)
        elif u < 0.80:
            m = torch.randint(0, max(1, C - 4), (1,)).item()
            frontiers.append(m)
            n_corr = min(2, C - m)
            corr_indices = torch.randperm(min(3, C - m), device=device)[:n_corr] + m
            rand_toks = torch.randint(100, 10000, (n_corr,), device=device)
            noisy[i, corr_indices] = rand_toks
            mask[i, corr_indices] = True
            m_suffix = min(C, m + 3)
            if m_suffix < C:
                mask[i, m_suffix:] = True
                noisy[i, m_suffix:] = mask_id
        else:
            if torch.rand(1).item() < 0.5:
                mask[i, :] = True
                noisy[i, :] = mask_id
                frontiers.append(0)
            else:
                ratio = 0.20 + 0.75 * torch.rand(1).item()
                n = max(1, int(ratio * C))
                perm = torch.randperm(C, device=device)[:n]
                mask[i, perm] = True
                noisy[i, perm] = mask_id
                frontiers.append(int(perm.min().item()))

    return noisy, mask, frontiers


def diffusion_loss(model, prefix, target, mask_id, lambda_kl=0.5, T_distill=1.0):
    B, C = target.shape
    P = prefix.shape[1]
    device = prefix.device
    noisy, mask, frontiers = make_mask(target, mask_id)

    # 1. Teacher Pass (Frozen Base Causal Model on clean sequence)
    clean_seq = torch.cat([prefix, target], dim=1)
    with torch.no_grad():
        if hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                t_out = model(input_ids=clean_seq, use_cache=False)
        else:
            t_out = model(input_ids=clean_seq, use_cache=False)
        Z_teacher = t_out.logits[:, P - 1 : P + C - 1, :].float()

    # 2. Student Pass (Task 2.3: 25% Self-Conditioning SC unroll + 2-step latent unroll)
    u_sc = torch.rand(1).item()
    if u_sc < 0.25:
        # Pass 1: generate logits_1 and latents_1 under no_grad
        with torch.no_grad():
            logits_1, latents_1 = canvas_forward(model, prefix, noisy, return_hidden=True, mask=mask)
            latents_1 = latents_1.detach()
            prev_pred_ids = logits_1.argmax(dim=-1)
            # Lookup predicted token embeddings from model's input embeddings
            emb_fn = model.get_input_embeddings()
            prev_pred_embeds = emb_fn(prev_pred_ids).detach()
        student_logits = canvas_forward(
            model,
            prefix,
            noisy,
            prev_latents=latents_1,
            prev_pred_embeds=prev_pred_embeds,
            return_hidden=False,
            mask=mask,
        )
    elif u_sc < 0.50:
        # Standard latent recycling unroll without self-conditioning
        with torch.no_grad():
            _, latents_1 = canvas_forward(model, prefix, noisy, return_hidden=True, mask=mask)
            latents_1 = latents_1.detach()
        student_logits = canvas_forward(
            model, prefix, noisy, prev_latents=latents_1, return_hidden=False, mask=mask
        )
    else:
        student_logits = canvas_forward(
            model, prefix, noisy, prev_latents=None, return_hidden=False, mask=mask
        )

    Z_student = student_logits.float()

    # 3. Top-64 Teacher Gather KL Divergence (Memory Invariant 4)
    topk_T = Z_teacher.topk(64, dim=-1)
    topk_vals_T = topk_T.values
    topk_idx = topk_T.indices
    topk_vals_S = Z_student.gather(dim=-1, index=topk_idx)

    p_teacher = F.softmax(topk_vals_T / T_distill, dim=-1)
    log_p_student = F.log_softmax(topk_vals_S / T_distill, dim=-1)
    kl_per_token = F.kl_div(log_p_student, p_teacher, reduction="none").sum(dim=-1) * (T_distill ** 2)

    # Cross-Entropy against true target
    ce_per_token = F.cross_entropy(Z_student.reshape(-1, Z_student.shape[-1]), target.reshape(-1), reduction="none").reshape(B, C)

    loss_per_token = lambda_kl * kl_per_token + (1.0 - lambda_kl) * ce_per_token

    # Offset position weighting: Task 1.1 Flat Frontier Plateau Weighting across offsets 0..4
    # j_rel <= 4 -> weight = 1.0; j_rel > 4 -> exp(-0.10 * (j_rel - 4))
    weights = torch.ones_like(loss_per_token)
    for i, front in enumerate(frontiers):
        if front < C:
            j_rel = torch.arange(C - front, device=device).float()
            weights[i, front:] = torch.where(
                j_rel <= 4,
                torch.ones_like(j_rel),
                torch.exp(-0.10 * (j_rel - 4.0)),
            )

    return (loss_per_token[mask] * weights[mask]).sum() / weights[mask].sum().clamp_min(1.0)


@torch.inference_mode()
def evaluate(model, loader, mask_id, device, batches):
    model.eval()
    total_loss, n = 0.0, 0
    correct_1, total_1 = 0, 0
    correct_2, total_2 = 0, 0
    correct_3, total_3 = 0, 0
    correct_4, total_4 = 0, 0
    correct_2_4, total_2_4 = 0, 0

    for i, (prefix, target) in enumerate(loader):
        if i >= batches:
            break
        prefix, target = prefix.to(device), target.to(device)
        B, C = target.shape
        P = prefix.shape[1]

        clean_seq = torch.cat([prefix, target], dim=1)
        if hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                t_out = model(input_ids=clean_seq, use_cache=False)
        else:
            t_out = model(input_ids=clean_seq, use_cache=False)
        Z_teacher = t_out.logits[:, P - 1 : P + C - 1, :].float()

        mask = torch.zeros_like(target, dtype=torch.bool)
        mask[:, 1:] = True
        noisy = target.masked_fill(mask, mask_id)

        Z_student = canvas_forward(model, prefix, noisy, mask=mask).float()

        loss = F.cross_entropy(Z_student[mask], target[mask])
        total_loss += loss.item()
        n += 1

        top1_T = Z_teacher.argmax(dim=-1)
        top1_S = Z_student.argmax(dim=-1)

        if C > 1:
            correct_1 += (top1_S[:, 1] == top1_T[:, 1]).sum().item()
            total_1 += B

        if C > 2:
            correct_2 += (top1_S[:, 2] == top1_T[:, 2]).sum().item()
            total_2 += B

        if C > 3:
            correct_3 += (top1_S[:, 3] == top1_T[:, 3]).sum().item()
            total_3 += B

        if C > 4:
            correct_4 += (top1_S[:, 4] == top1_T[:, 4]).sum().item()
            total_4 += B

        if C > 4:
            correct_2_4 += (top1_S[:, 2:5] == top1_T[:, 2:5]).sum().item()
            total_2_4 += (B * 3)

    model.train()
    acc_offset1 = correct_1 / max(1, total_1)
    acc_offset2 = correct_2 / max(1, total_2)
    acc_offset3 = correct_3 / max(1, total_3)
    acc_offset4 = correct_4 / max(1, total_4)
    acc_offset2_4 = correct_2_4 / max(1, total_2_4)
    val_loss = total_loss / max(1, n)
    return {
        "val_loss": val_loss,
        "acc_offset1": acc_offset1,
        "acc_offset2": acc_offset2,
        "acc_offset3": acc_offset3,
        "acc_offset4": acc_offset4,
        "acc_offset2_4": acc_offset2_4,
    }


def save_checkpoint(path, model, mask_id, args, step):
    """LoRA runs save ~10 MB of adapters; full fine-tunes save the whole model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ck = {"mask_id": mask_id, "model_name": args.model, "step": step, "args": vars(args)}
    if getattr(model, "bfg_adapter", None) is not None:
        ck["bfg_adapter"] = model.bfg_adapter.state_dict()
    if args.lora:
        from peft import get_peft_model_state_dict

        ck["lora"] = get_peft_model_state_dict(model)
        ck["lora_config"] = {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "targets": list(LORA_TARGETS_MLP if args.lora_mlp else LORA_TARGETS),
        }
        emb = model.get_input_embeddings().weight
        if emb.requires_grad:
            ck["mask_emb"] = emb[mask_id].detach().cpu().clone()
    else:
        ck["model"] = model.state_dict()
    torch.save(ck, path)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    n_threads = args.threads or (os.cpu_count() or 1)
    torch.set_num_threads(n_threads)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:  # already set by an earlier import
        pass
    device = pick_device(args.device)
    dtype = (
        torch.bfloat16
        if args.dtype == "bf16" or (args.dtype == "auto" and device.type == "cuda")
        else torch.float32
    )

    model, tok, mask_id = prepare(args.model, device=device, dtype=dtype)
    from model import setup_bfg_adapter
    setup_bfg_adapter(model, lora_r=args.lora_r if args.lora else 16)

    if args.lora:
        targets = LORA_TARGETS_MLP if args.lora_mlp else LORA_TARGETS
        model = apply_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout, targets, mask_id=mask_id)
        if args.resume and Path(args.resume).exists():
            print(f"resuming LoRA and mask weights from {args.resume}...")
            from peft import set_peft_model_state_dict
            ck = torch.load(args.resume, map_location="cpu")
            if "lora" in ck:
                set_peft_model_state_dict(model, ck["lora"])
            if "mask_emb" in ck:
                with torch.no_grad():
                    model.get_input_embeddings().weight[mask_id] = ck["mask_emb"].to(device)
            if "bfg_adapter" in ck and getattr(model, "bfg_adapter", None) is not None:
                model.bfg_adapter.load_state_dict(ck["bfg_adapter"])
            print(f"resumed from {args.resume} (source step {ck.get('step', '?')})")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(
            f"LoRA r={args.lora_r} alpha={args.lora_alpha} targets={len(targets)}: "
            f"{trainable:,} / {total:,} params trainable"
        )
    ds = (
        GeneralTextDataset(
            args.data, tok, args.prefix_len, args.canvas_len,
            limit=args.limit, max_tokens=args.max_tokens,
        )
        if args.data
        else GeneralTextDataset(
            tokenizer=tok,
            prefix_len=args.prefix_len,
            canvas_len=args.canvas_len,
            texts=synthetic_texts(args.texts, seed=args.seed),
        )
    )
    assert len(ds) > 1, f"dataset too small: {len(ds)} chunks"
    n_val = min(len(ds) // 20 + 1, 64, len(ds) - 1)
    assert len(ds) - n_val >= args.batch_size, "too few chunks for one batch; add data or lower --batch_size"
    val_ds, train_ds = Subset(ds, range(n_val)), Subset(ds, range(n_val, len(ds)))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    print(
        f"device={device} dtype={dtype} mask_id={mask_id} "
        f"train={len(train_ds)} val={len(val_ds)} chunks"
    )

    adapter_params = list(model.bfg_adapter.parameters()) if getattr(model, "bfg_adapter", None) is not None else []
    trainable_params = [p for n, p in model.named_parameters() if p.requires_grad and "embed_tokens" not in n]
    all_trainable = list(dict.fromkeys(trainable_params + adapter_params))
    opt = torch.optim.AdamW(
        all_trainable,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    # Track Adam moments for the single mask embedding row to avoid 136M buffer allocations
    emb_weight = model.get_input_embeddings().weight
    mask_m = torch.zeros(emb_weight.shape[1], device=device)
    mask_v = torch.zeros(emb_weight.shape[1], device=device)
    mask_t = 0

    def lr_lambda(update):
        update += 1  # first update should not use lr=0
        if update < args.warmup:
            return update / max(1, args.warmup)
        total_updates = max(1, math.ceil(args.steps / args.grad_accum))
        progress = (update - args.warmup) / max(1, total_updates - args.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    autocast = torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=(dtype == torch.bfloat16)
    )

    model.train()
    opt.zero_grad(set_to_none=True)
    if args.smoke_test:  # fixed batch + fixed 50% mask: a stable convergence probe
        probe_prefix, probe_target = next(iter(loader))
        probe_prefix, probe_target = probe_prefix.to(device), probe_target.to(device)
        probe_mask = torch.zeros_like(probe_target, dtype=torch.bool)
        probe_mask[:, ::2] = True

        def probe_loss():
            with torch.no_grad():
                logits = canvas_forward(
                    model, probe_prefix, probe_target.masked_fill(probe_mask, mask_id)
                )
                return F.cross_entropy(logits[probe_mask], probe_target[probe_mask]).item()

        before = probe_loss()
        print(f"smoke probe loss (before): {before:.4f}")
    data_iter = cycle(loader)
    running, start, last_log = 0.0, time.time(), time.time()
    tokens = 0
    logged = 0
    for step in range(1, args.steps + 1):
        prefix, target = next(data_iter)
        prefix, target = prefix.to(device), target.to(device)
        with autocast:
            loss = diffusion_loss(model, prefix, target, mask_id) / args.grad_accum
        loss.backward()
        step_loss = loss.item() * args.grad_accum
        running += step_loss
        logged += 1
        tokens += prefix.numel() + target.numel()

        if step % args.grad_accum == 0 or step == args.steps:
            torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
            opt.step()
            sched.step()  # one schedule step per optimizer update
            opt.zero_grad(set_to_none=True)

            # Update the single [MASK] row with Adam moments
            if emb_weight.requires_grad and emb_weight.grad is not None:
                g = emb_weight.grad[mask_id]
                mask_t += 1
                beta1, beta2, eps = 0.9, 0.95, 1e-8
                mask_m = beta1 * mask_m + (1.0 - beta1) * g
                mask_v = beta2 * mask_v + (1.0 - beta2) * (g ** 2)
                m_hat = mask_m / (1.0 - beta1 ** mask_t)
                v_hat = mask_v / (1.0 - beta2 ** mask_t)
                lr = sched.get_last_lr()[0]
                with torch.no_grad():
                    emb_weight[mask_id] -= lr * m_hat / (torch.sqrt(v_hat) + eps)
                emb_weight.grad.zero_()

        if step % args.log_every == 0 or step == 1:
            now = time.time()
            print(
                f"step {step:5d}/{args.steps} loss {running / max(1, logged):.4f} "
                f"lr {sched.get_last_lr()[0]:.2e} tok/s {tokens / max(now - last_log, 1e-6):,.0f} "
                f"elapsed {now - start:.0f}s"
            )
            running, tokens, logged, last_log = 0.0, 0, 0, now

        if step % args.eval_every == 0 or step == args.steps:
            val_res = evaluate(model, val_loader, mask_id, device, args.eval_batches)
            val_loss = val_res["val_loss"]
            acc1 = val_res["acc_offset1"]
            acc2 = val_res["acc_offset2"]
            acc3 = val_res["acc_offset3"]
            acc4 = val_res["acc_offset4"]
            acc2_4 = val_res["acc_offset2_4"]
            print(
                f"step {step:5d} val_loss {val_loss:.4f} "
                f"acc_offset1 {acc1:.1%} acc_offset2 {acc2:.1%} "
                f"acc_offset3 {acc3:.1%} acc_offset4 {acc4:.1%} "
                f"(acc_offset2_4 {acc2_4:.1%})"
            )

        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(Path(args.out) / "last.pt", model, mask_id, args, step)
            if step % args.save_every == 0 and step != args.steps:
                save_checkpoint(Path(args.out) / f"ckpt-{step}.pt", model, mask_id, args, step)

    if args.smoke_test:
        after = probe_loss()
        ok = math.isfinite(after) and after < before
        print(f"smoke convergence: {before:.4f} -> {after:.4f} {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit(1)
    print(f"saved {Path(args.out) / 'last.pt'}")


if __name__ == "__main__":
    main()
