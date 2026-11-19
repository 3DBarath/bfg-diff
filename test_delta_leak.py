import torch
from model import prepare, canvas_forward, canvas_step_with_cache, prefill_prefix

def test_delta_leak():
    device = "cpu"
    dtype = torch.float32
    model, tok, mask_id = prepare("Qwen/Qwen2.5-0.5B", device=device, dtype=dtype)
    model.eval()

    # Create prefix and canvas
    P = 16
    C = 8
    prefix = torch.randint(100, 1000, (1, P), device=device)
    target = torch.randint(100, 1000, (1, C), device=device)

    # Let Zone A be clean (first 2 tokens of canvas), Zone B be masked (remaining 6 tokens)
    # So canvas has 2 clean tokens, then 6 [MASK] tokens
    zone_a_len = 2
    zone_b_len = C - zone_a_len
    canvas_mask = torch.zeros((1, C), dtype=torch.bool, device=device)
    canvas_mask[:, zone_a_len:] = True

    noisy_canvas = target.clone()
    noisy_canvas[:, zone_a_len:] = mask_id

    # 1. canvas_forward with hybrid training mask
    with torch.no_grad():
        logits_train = canvas_forward(
            model, prefix, noisy_canvas, mask=canvas_mask
        )
    # Zone A predictions from canvas_forward are logits for prefix[-1] and target[0]:
    # Specifically, canvas_forward outputs logits[:, P-1 : P+C-1]
    # So slot 0 predicts canvas token 0 (conditioned on prefix 0..P-1)
    # slot 1 predicts canvas token 1 (conditioned on prefix + canvas[0])
    train_zone_a_logits = logits_train[:, :zone_a_len, :]

    # 2. canvas_step_with_cache with inference staircase mask
    kv = prefill_prefix(model, prefix)
    active_ids = torch.cat([prefix[:, -1:], noisy_canvas], dim=1) # query prefix_last + canvas
    with torch.no_grad():
        step_logits = canvas_step_with_cache(
            model,
            active_ids,
            past_key_values=kv,
            prefix_len=P,
            zone_a_len=1 + zone_a_len, # prefix[-1] + 2 Zone A clean tokens
        )
    step_zone_a_logits = step_logits[:, :zone_a_len, :]

    # Check Delta_leak: max difference between train logits and inference logits on clean Zone A
    diff = (train_zone_a_logits - step_zone_a_logits).abs().max().item()
    print(f"Max difference Delta_leak on clean Zone A positions: {diff:.6e}")
    assert diff < 1e-3, f"Delta_leak too large: {diff}"
    print("PASS: Delta_leak == 0.0 (within fp precision) between train and inference masks on Zone A!")

if __name__ == "__main__":
    test_delta_leak()
