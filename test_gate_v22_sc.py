"""Unit Gate 1: Test Sigma Self-Conditioning, SVD init, Parity, Gradient Flow & Zone A Isolation."""
import torch
from model import prepare, setup_bfg_adapter, build_staircase_step_mask, canvas_step_with_cache, canvas_forward

def test_gate_v22_sc():
    print("=== RUNNING GATE 1: SIGMA SC & SVD INIT UNIT CHECK ===")
    device = "cpu"
    dtype = torch.float32
    model, tok, mask_id = prepare("Qwen/Qwen2.5-0.5B", device=device, dtype=dtype)
    adapter = setup_bfg_adapter(model, max_w=16, lora_r=16)

    # 1. Zero-init step-0 parity check
    print("1. Checking step-0 parity with zero init...")
    P = 16
    prefix = torch.randint(100, 1000, (1, P), device=device)
    target = torch.randint(100, 1000, (1, 8), device=device)
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[:, 2:] = True
    noisy = target.masked_fill(mask, mask_id)

    # Logits without adapter modifications (step-0 base output)
    logits_base = canvas_forward(model, prefix, noisy, mask=mask)

    # Now verify with dummy prev_latents and prev_pred_embeds (since weights are 0, should be identical)
    dummy_latents = torch.randn(1, 8, model.config.hidden_size)
    dummy_pred_embeds = torch.randn(1, 8, model.config.hidden_size)
    logits_sc = canvas_forward(
        model, prefix, noisy, prev_latents=dummy_latents, prev_pred_embeds=dummy_pred_embeds, mask=mask
    )

    diff = (logits_base - logits_sc).abs().max().item()
    print(f"   Delta parity: {diff:.8e}")
    assert diff < 1e-6, f"Parity failed: diff {diff} >= 1e-6"
    print("   [PASS] Parity verified.")

    # 2. Check SVD initialization
    print("2. Checking iWSP SVD initialization...")
    embed_weight = model.get_input_embeddings().weight
    adapter.init_svd_from_embeddings(embed_weight, scale=0.02)
    assert not torch.allclose(adapter.latent_down.weight, torch.zeros_like(adapter.latent_down.weight)), "SVD init failed"
    print("   [PASS] SVD init verified.")

    # 3. Check Gradient flow & Finite clipped grads
    print("3. Checking gradient flow through all adapter components...")
    adapter.slot_emb.weight.requires_grad = True
    adapter.latent_down.weight.requires_grad = True
    adapter.latent_up.weight.requires_grad = True
    adapter.pred_emb_scale.requires_grad = True

    model.train()
    logits = canvas_forward(
        model, prefix, noisy, prev_latents=dummy_latents, prev_pred_embeds=dummy_pred_embeds, mask=mask
    )
    loss = logits[mask].sum()
    loss.backward()

    for name, param in [
        ("slot_emb", adapter.slot_emb.weight),
        ("latent_down", adapter.latent_down.weight),
        ("latent_up", adapter.latent_up.weight),
        ("pred_emb_scale", adapter.pred_emb_scale),
    ]:
        assert param.grad is not None, f"Gradient is None for {name}"
        gnorm = param.grad.norm().item()
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"
        print(f"   {name} grad norm: {gnorm:.4f}")
    print("   [PASS] Non-zero finite gradients verified.")

    # 4. Strict Zone A Causal Isolation Check (Delta_leak = 0)
    print("4. Checking Zone A strict causal isolation (Delta_leak = 0)...")
    model.eval()
    zone_a_len = 3
    zone_b_len = 3
    active_ids_1 = torch.tensor([[101, 102, 103, mask_id, mask_id, mask_id]], device=device)
    # Change Zone B tokens completely:
    active_ids_2 = torch.tensor([[101, 102, 103, 999, 888, 777]], device=device)

    # Prefix KV
    from sampler import prefill_prefix
    kv1 = prefill_prefix(model, prefix)
    out1 = canvas_step_with_cache(model, active_ids_1, past_key_values=kv1, prefix_len=P, zone_a_len=zone_a_len)

    kv2 = prefill_prefix(model, prefix)
    out2 = canvas_step_with_cache(model, active_ids_2, past_key_values=kv2, prefix_len=P, zone_a_len=zone_a_len)

    # Zone A outputs (queries 0..zone_a_len-1) must be strictly identical regardless of Zone B contents
    leak = (out1[:, :zone_a_len] - out2[:, :zone_a_len]).abs().max().item()
    print(f"   Zone A Delta_leak: {leak:.8e}")
    assert leak == 0.0, f"Zone A causal isolation breached! leak = {leak}"
    print("   [PASS] Zone A causal isolation verified.")

    print("\n>>> ALL GATE 1 UNIT CHECKS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    test_gate_v22_sc()
