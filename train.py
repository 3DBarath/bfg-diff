"""Train Qwen2.5 as a block-autoregressive discrete diffusion LM (BD3-LM style)."""
import argparse
import math
import time
from itertools import cycle
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset import GeneralTextDataset, synthetic_texts
from model import canvas_forward, prepare


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
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_batches", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="max source files")
    p.add_argument("--texts", type=int, default=32, help="synthetic texts when --data is unset")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke_test", action="store_true", help="tiny synthetic run + convergence check")
    args = p.parse_args()
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
    """Per-sample t ~ U(0.15, 0.95); mask exactly floor(t*C) canvas tokens."""
    B, C = target.shape
    mask = torch.zeros_like(target, dtype=torch.bool)
    for i in range(B):
        t = float(torch.empty(1).uniform_(0.15, 0.95))
        n = max(1, int(t * C))
        mask[i, torch.randperm(C, device=target.device)[:n]] = True
    return target.masked_fill(mask, mask_id), mask


def diffusion_loss(model, prefix, target, mask_id):
    noisy, mask = make_mask(target, mask_id)
    logits = canvas_forward(model, prefix, noisy)
    assert logits.shape[:2] == target.shape, logits.shape
    return F.cross_entropy(logits[mask], target[mask])


@torch.no_grad()
def evaluate(model, loader, mask_id, device, batches):
    model.eval()
    total, n = 0.0, 0
    for i, (prefix, target) in enumerate(loader):
        if i >= batches:
            break
        prefix, target = prefix.to(device), target.to(device)
        mask = torch.rand(target.shape, device=device) < 0.5
        mask[:, :1] = True
        logits = canvas_forward(model, prefix, target.masked_fill(mask, mask_id))
        total += F.cross_entropy(logits[mask], target[mask]).item()
        n += 1
    model.train()
    return total / max(1, n)


def save_checkpoint(path, model, mask_id, args, step):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "mask_id": mask_id,
            "model_name": args.model,
            "step": step,
            "args": vars(args),
        },
        path,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    dtype = (
        torch.bfloat16
        if args.dtype == "bf16" or (args.dtype == "auto" and device.type == "cuda")
        else torch.float32
    )

    model, tok, mask_id = prepare(args.model, device=device, dtype=dtype)
    ds = (
        GeneralTextDataset(args.data, tok, args.prefix_len, args.canvas_len, limit=args.limit)
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

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()  # one schedule step per optimizer update
            opt.zero_grad(set_to_none=True)

        if step % args.log_every == 0 or step == 1:
            now = time.time()
            print(
                f"step {step:5d}/{args.steps} loss {running / max(1, logged):.4f} "
                f"lr {sched.get_last_lr()[0]:.2e} tok/s {tokens / max(now - last_log, 1e-6):,.0f} "
                f"elapsed {now - start:.0f}s"
            )
            running, tokens, logged, last_log = 0.0, 0, 0, now

        if step % args.eval_every == 0 or step == args.steps:
            print(
                f"step {step:5d} val_loss "
                f"{evaluate(model, val_loader, mask_id, device, args.eval_batches):.4f}"
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
