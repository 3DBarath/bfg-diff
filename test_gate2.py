import torch
from model import prepare, prefill_prefix
from sampler import sample_staircase_sliding

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    print(f"Running Gate 2 verification on {device} ({dtype})...")

    model, tok, mask_id = prepare("Qwen/Qwen2.5-0.5B", device=device, dtype=dtype)
    model.eval()

    P = 16
    prefix = torch.randint(100, 1000, (1, P), device=device)
    
    # 1. Run sliding staircase generation for 8 tokens
    stats = {}
    gen_tokens = sample_staircase_sliding(
        model,
        prefix,
        canvas_len=8,
        mask_id=mask_id,
        W_max=4,
        V_max=3,
        stats_out=stats,
    )
    print(f"Generated tokens: {gen_tokens.tolist()}")
    print(f"Stats: alpha={stats['alpha']:.2f}, R_kv={stats['r_kv']:.2f}, passes={stats['forward_passes']}, verified={stats['tokens_verified']}")

    assert gen_tokens.shape[1] == 8, f"Expected 8 tokens, got {gen_tokens.shape[1]}"
    assert (gen_tokens != mask_id).all(), "Zero mask_id leak invariant violated!"

    # 2. Check KV parity against clean causal prefill of [prefix, gen_tokens[:, :-1]]
    full_seq = torch.cat([prefix, gen_tokens[:, :-1]], dim=1) # total length P + 7
    expected_kv = prefill_prefix(model, torch.cat([prefix, gen_tokens], dim=1)) # prefills up to last token
    
    # Run staircase sliding with initial kv passed in
    kv = prefill_prefix(model, prefix)
    init_kv_len = kv.get_seq_length()
    assert init_kv_len == P - 1, f"Initial kv length {init_kv_len} != P-1"

    # Verify sliding maintains exact KV equality with prefill
    gen2 = sample_staircase_sliding(
        model,
        prefix,
        canvas_len=8,
        mask_id=mask_id,
        kv=kv,
        W_max=4,
        V_max=3,
    )
    final_kv_len = kv.get_seq_length()
    expected_kv_len = P + 8 - 1
    print(f"Final kv length: {final_kv_len}, expected: {expected_kv_len}")
    assert final_kv_len == expected_kv_len, f"KV length {final_kv_len} != expected {expected_kv_len}"

    # Check max difference between kv and expected_kv across all layers
    max_diff = 0.0
    for layer_idx in range(len(kv.layers)):
        layer_sliding = kv.layers[layer_idx]
        layer_prefill = expected_kv.layers[layer_idx]
        k_sliding = layer_sliding.keys
        k_prefill = layer_prefill.keys
        v_sliding = layer_sliding.values
        v_prefill = layer_prefill.values
        
        diff_k = (k_sliding - k_prefill).abs().max().item()
        diff_v = (v_sliding - v_prefill).abs().max().item()
        max_diff = max(max_diff, diff_k, diff_v)

    print(f"Multi-step KV drift max_diff: {max_diff:.6e}")
    assert max_diff < 1e-4, f"KV drift too high! {max_diff}"

    print("GATE 2 PASS: In-place KV cache slicing & length synchronization verified!")

if __name__ == "__main__":
    main()
