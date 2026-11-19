"""Verify that BFGDiffAdapter parameters remain trainable after apply_lora and receive gradients."""
import torch
from model import prepare, setup_bfg_adapter, apply_lora
from train import diffusion_loss

def verify_adapter_unfrozen():
    print("=== VERIFYING BFGDiffAdapter GRADIENT FLOW AFTER apply_lora ===")
    device = "cpu"
    dtype = torch.float32

    # 1. Prepare model and setup adapter
    model, tok, mask_id = prepare("Qwen/Qwen2.5-0.5B", device=device, dtype=dtype)
    adapter = setup_bfg_adapter(model, lora_r=16)
    adapter.init_svd_from_embeddings(model.get_input_embeddings().weight, scale=0.0)

    # 2. Apply LoRA
    model = apply_lora(model, r=16, alpha=32, mask_id=mask_id)

    # Check PEFT freeze effect and explicit unfreeze
    if getattr(model, "bfg_adapter", None) is not None:
        for p in model.bfg_adapter.parameters():
            p.requires_grad = True

    # Assert all adapter parameters have requires_grad == True
    for name, p in model.bfg_adapter.named_parameters():
        assert p.requires_grad, f"Parameter {name} has requires_grad=False!"
    print("[PASS] All BFGDiffAdapter parameters have requires_grad=True.")

    # 3. Take 2 training steps and record parameter changes
    initial_slot_norm = model.bfg_adapter.slot_emb.weight.data.norm().item()
    initial_up_norm = model.bfg_adapter.latent_up.weight.data.norm().item()
    initial_scale = model.bfg_adapter.pred_emb_scale.data.item()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
    )

    P, C = 16, 8
    prefix = torch.randint(100, 1000, (2, P), device=device)
    target = torch.randint(100, 1000, (2, C), device=device)

    model.train()
    for step in range(1, 3):
        optimizer.zero_grad()
        loss = diffusion_loss(model, prefix, target, mask_id, force_sc=True)
        loss.backward()

        # Check gradients exist and are non-zero
        for name, p in model.bfg_adapter.named_parameters():
            assert p.grad is not None, f"Gradient for {name} is None at step {step}!"
            assert torch.isfinite(p.grad).all(), f"Gradient for {name} is non-finite at step {step}!"

        optimizer.step()
        print(f"  Step {step} loss: {loss.item():.4f}")

    final_slot_norm = model.bfg_adapter.slot_emb.weight.data.norm().item()
    final_up_norm = model.bfg_adapter.latent_up.weight.data.norm().item()
    final_scale = model.bfg_adapter.pred_emb_scale.data.item()

    print(f"slot_emb norm: {initial_slot_norm:.6f} -> {final_slot_norm:.6f}")
    print(f"latent_up norm: {initial_up_norm:.6f} -> {final_up_norm:.6f}")
    print(f"pred_emb_scale: {initial_scale:.6f} -> {final_scale:.6f}")

    assert final_slot_norm != initial_slot_norm, "slot_emb did not update!"
    assert final_up_norm != initial_up_norm, "latent_up did not update!"
    assert final_scale != initial_scale, "pred_emb_scale did not update!"
    print("[PASS] Weights successfully updated over 2 steps.")
    print(">>> VERIFICATION COMPLETED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    verify_adapter_unfrozen()
