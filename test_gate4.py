import torch
from model import prepare, prefill_prefix, canvas_step_with_cache

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    print(f"Running Gate 4 verification on {device} ({dtype})...")

    model, tok, mask_id = prepare("Qwen/Qwen2.5-0.5B", device=device, dtype=dtype)
    model.eval()

    P = 16
    prefix = torch.randint(100, 1000, (1, P), device=device)
    cur_last = prefix[:, -1:]
    
    # 2-branch candidates:
    # Branch 1: [101, 102]
    # Branch 2: [201, 202]
    b1_0, b1_1 = torch.tensor([[101]], device=device), torch.tensor([[102]], device=device)
    b2_0, b2_1 = torch.tensor([[201]], device=device), torch.tensor([[202]], device=device)

    zone_a_ids = torch.cat([cur_last, b1_0, b1_1, b2_0, b2_1], dim=1) # (1, 5)
    W = 3
    zone_b_ids = torch.full((1, W), mask_id, device=device)
    active_ids = torch.cat([zone_a_ids, zone_b_ids], dim=1) # (1, 8)
    Q = 8

    tree_mask_a = torch.tensor([
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [1, 0, 0, 1, 1],
    ], dtype=torch.bool, device=device)

    base_p = P - 1
    pos_a = torch.tensor([[base_p, base_p + 1, base_p + 2, base_p + 1, base_p + 2]], device=device)
    pos_b = (base_p + 3) + torch.arange(W, device=device).unsqueeze(0)
    pos_override = torch.cat([pos_a, pos_b], dim=1)

    kv = prefill_prefix(model, prefix)
    logits = canvas_step_with_cache(
        model,
        active_ids,
        past_key_values=kv,
        prefix_len=P,
        zone_a_len=5,
        tree_mask_a=tree_mask_a,
        position_ids_override=pos_override,
    )

    # Simulate Branch 2 winning with k=3 (both b2_0, b2_1 accepted + bonus)
    # Copy Branch 2 KV states [3, 4] into [1, 2]
    p_offset = P - 1
    for layer in kv.layers:
        layer.keys[:, :, p_offset + 1 : p_offset + 3, :] = layer.keys[:, :, p_offset + 3 : p_offset + 5, :].clone()
        layer.values[:, :, p_offset + 1 : p_offset + 3, :] = layer.values[:, :, p_offset + 3 : p_offset + 5, :].clone()

    k = 3
    kv.crop(-(Q - k))

    # Compare with clean causal prefill of [prefix, b2_0, b2_1]
    expected_full = torch.cat([prefix, b2_0, b2_1], dim=1)
    # prefill encodes all except last
    expected_kv = prefill_prefix(model, torch.cat([expected_full, torch.tensor([[999]], device=device)], dim=1))

    max_diff = 0.0
    for layer_idx in range(len(kv.layers)):
        k_sl = kv.layers[layer_idx].keys
        k_pf = expected_kv.layers[layer_idx].keys
        v_sl = kv.layers[layer_idx].values
        v_pf = expected_kv.layers[layer_idx].values
        diff = max((k_sl - k_pf).abs().max().item(), (v_sl - v_pf).abs().max().item())
        max_diff = max(max_diff, diff)

    print(f"Branch 2 KV contiguity max_diff: {max_diff:.6e}")
    assert max_diff < 1e-4, f"KV contiguity drift too high! {max_diff}"
    print("GATE 4 PASS: 2-Branch Tree verification and KV contiguity verified!")

if __name__ == "__main__":
    main()
