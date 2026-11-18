import torch
from model import prepare, prefill_prefix, canvas_step_with_cache, setup_bfg_adapter, canvas_forward

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    print(f"Running Gate 3 verification on {device} ({dtype})...")

    # 1. Base model without adapter
    model, tok, mask_id = prepare("Qwen/Qwen2.5-0.5B", device=device, dtype=dtype)
    model.eval()

    P = 16
    prefix = torch.randint(100, 1000, (1, P), device=device)
    active = torch.cat([prefix[:, -1:], torch.full((1, 4), mask_id, device=device)], dim=1) # (1, 5)

    kv = prefill_prefix(model, prefix)
    logits_before = canvas_step_with_cache(model, active, kv, prefix_len=P, zone_a_len=1)

    # 2. Attach zero-initialized BFGDiffAdapter
    adapter = setup_bfg_adapter(model)
    kv2 = prefill_prefix(model, prefix)
    logits_after = canvas_step_with_cache(model, active, kv2, prefix_len=P, zone_a_len=1)

    diff = (logits_before - logits_after).abs().max().item()
    print(f"Zero-init adapter representation diff: {diff:.6e}")
    assert diff < 1e-6, f"Representation shock! diff={diff}"

    # 3. Test gradient backprop through adapter during a 2-step training probe
    model.train()
    target = torch.randint(100, 1000, (1, 8), device=device)
    dummy_prefix = torch.randint(100, 1000, (1, 8), device=device)
    noisy_canvas = torch.full((1, 8), mask_id, device=device)

    opt = torch.optim.SGD(adapter.parameters(), lr=0.1)

    # Step 1
    opt.zero_grad()
    prev_latents = torch.randn(1, 8, model.config.hidden_size, device=device, dtype=dtype)
    logits_step1, _ = canvas_forward(model, dummy_prefix, noisy_canvas, prev_latents=prev_latents, return_hidden=True)
    loss1 = logits_step1.sum()
    loss1.backward()

    slot_grad = adapter.slot_emb.weight.grad
    assert slot_grad is not None, "slot_emb.weight.grad is None!"
    slot_grad_norm = slot_grad.norm().item()
    print(f"Step 1 slot_emb grad norm: {slot_grad_norm:.6e}")
    assert slot_grad_norm > 0, "slot_emb gradient is zero!"

    opt.step() # Updates latent_up with non-zero gradient

    # Step 2
    opt.zero_grad()
    logits_step2, _ = canvas_forward(model, dummy_prefix, noisy_canvas, prev_latents=prev_latents, return_hidden=True)
    loss2 = logits_step2.sum()
    loss2.backward()

    latent_down_grad = adapter.latent_down.weight.grad
    assert latent_down_grad is not None, "latent_down.weight.grad is None!"
    down_grad_norm = latent_down_grad.norm().item()
    print(f"Step 2 latent_down grad norm: {down_grad_norm:.6e}")
    assert down_grad_norm > 0, "latent_down gradient is zero on step 2!"

    print("GATE 3 PASS: BFGDiffAdapter zero-init parity and gradient backprop verified!")

if __name__ == "__main__":
    main()
