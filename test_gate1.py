import torch
from model import prepare, prefill_prefix, canvas_step_with_cache, build_staircase_step_mask

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # Use fp32 for strict numerical checks
    print(f"Running Gate 1 verification on {device} ({dtype})...")

    model, tok, mask_id = prepare("Qwen/Qwen2.5-0.5B", device=device, dtype=dtype)
    model.eval()

    # Prefix of length P=16
    P = 16
    prefix = torch.randint(100, 1000, (1, P), device=device)
    
    # Zone A: 1 + V = 4 tokens (prefix_last_id + 3 drafts)
    zone_a = torch.randint(100, 1000, (1, 4), device=device)
    zone_a[:, 0] = prefix[:, -1] # Zone A token 0 is prefix_last_id (pos P-1)
    
    # Zone B: W = 4 tokens
    zone_b1 = torch.full((1, 4), mask_id, device=device)
    zone_b2 = torch.randint(100, 1000, (1, 4), device=device) # randomized tokens

    active_1 = torch.cat([zone_a, zone_b1], dim=1) # (1, 8)
    active_2 = torch.cat([zone_a, zone_b2], dim=1) # (1, 8)

    # 1. Prefill prefix[:P-1]
    kv1 = prefill_prefix(model, prefix)
    kv2 = prefill_prefix(model, prefix)

    # 2. Run canvas_step_with_cache on active_1 and active_2 with zone_a_len = 4
    logits1 = canvas_step_with_cache(model, active_1, kv1, prefix_len=P, zone_a_len=4)
    logits2 = canvas_step_with_cache(model, active_2, kv2, prefix_len=P, zone_a_len=4)

    # Check Zone A isolation
    delta_leak = (logits1[:, :4, :] - logits2[:, :4, :]).abs().max().item()
    print(f"Zone A Causal Isolation delta_leak: {delta_leak:.6e}")
    assert delta_leak < 1e-5, f"Isolation failure! delta_leak = {delta_leak}"

    # 3. Check exact parity with clean causal pass on [prefix, zone_a[1:]]
    # Note: prefix includes prefix_last (which is zone_a[0]), so sequence is [prefix, zone_a[:, 1:]]
    full_causal_ids = torch.cat([prefix, zone_a[:, 1:]], dim=1)
    full_out = model(input_ids=full_causal_ids, use_cache=False)
    # The causal logits for positions P-1 .. P+2 (predicting pos P .. P+3) are at logits[:, P-1 : P+3, :]
    causal_zone_a_logits = full_out.logits[:, P - 1 : P + 3, :]
    
    delta_causal = (logits1[:, :4, :] - causal_zone_a_logits).abs().max().item()
    print(f"Parity with causal forward pass delta_causal: {delta_causal:.6e}")
    assert delta_causal < 1e-4, f"Causal parity failure! delta_causal = {delta_causal}"

    print("GATE 1 PASS: Zone A Causal Isolation and Causal Parity verified!")

if __name__ == "__main__":
    main()
