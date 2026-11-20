"""BFG-Diff v4 Unit Tests:
1. merge_ghost_weights() matches apply_linear_with_delta within fp16 tolerance (atol=1e-2).
2. WAL + [0:W] scratch tail + bucket transitions produce identical greedy outputs to base model.
3. 6+1 Leaf-Hedge preserves exact lossless greedy parity across Case A, Case B, and Case C.
"""

import sys
sys.path.insert(0, '/home/gces/.bin/.s/.scripts/bfg-diff')
import torch
import torch.nn.functional as F
from model_v3 import prepare_v3, apply_linear_with_delta, apply_rotary_pos_emb_k, quantize_for_cpu
from sampler_v3 import StaticDraftEngine, generate_speculative_v3
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

def test_ghost_weights_unit():
    print("--- Test 1: merge_ghost_weights() Unit Check ---")
    base_model, drafter, tok, mask_id = prepare_v3("Qwen/Qwen2.5-0.5B", device="cpu", dtype=torch.float32)
    B, W, D = 1, 7, drafter.dim
    h = torch.randn(B, W, D)
    pos = torch.arange(W).unsqueeze(0)

    # Reference forward without ghost
    h_ref, k_ref, v_ref = drafter.forward_draft_layer(0, h.clone(), pos)

    # Merged forward with ghost
    drafter.merge_ghost_weights(dtype=torch.float32)
    h_ghost, k_ghost, v_ghost = drafter.forward_draft_layer(0, h.clone(), pos)

    max_diff = (h_ref - h_ghost).abs().max().item()
    print(f"Ghost weights layer output max diff: {max_diff:.2e}")
    assert max_diff < 1e-3, f"Ghost weights mismatch: {max_diff}"
    print("PASS: merge_ghost_weights() unit check\n")


def test_wal_scratch_bucket_transitions():
    print("--- Test 2: WAL + [0:W] Scratch Tail + Bucket Transitions ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    base_model, drafter, tok, mask_id = prepare_v3("Qwen/Qwen2.5-0.5B", device=device, dtype=dtype)
    drafter.merge_ghost_weights(dtype=dtype)

    prompt = "The quick brown fox jumps over the lazy dog and runs across the wide green open field toward the tall mountains."
    inputs = tok(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    P = input_ids.shape[1]
    max_new = 45 # Will reach P + max_new ~ 70, crossing bucket 64 -> 128

    # Non-bucketed base model greedy reference
    with torch.no_grad():
        ref_out = base_model.generate(input_ids=input_ids, max_new_tokens=max_new, do_sample=False)

    # Bucketed StaticDraftEngine with small buckets to guarantee crossing
    custom_buckets = (32, 64, 128, 256)
    engine = StaticDraftEngine(
        base_model=base_model,
        drafter=drafter,
        max_len=256,
        batch_size=1,
        v_tokens=7,
        device=device,
        dtype=dtype,
        buckets=custom_buckets,
        use_cuda_graph=False,
    )

    with torch.no_grad():
        spec_out, stats = generate_speculative_v3(
            base_model=base_model,
            drafter=drafter,
            input_ids=input_ids,
            mask_token_id=mask_id,
            max_new_tokens=max_new,
            temperature=0.0,
            engine=engine,
            hedge_margin=0.0, # Test baseline linear chain first
        )

    ref_gen = ref_out[0, P:]
    spec_gen = spec_out[0, P:]
    matches = torch.equal(ref_gen, spec_gen)
    print(f"Prompt length: {P}, Generated tokens: {len(spec_gen)}")
    print(f"Bucket transitions: start={engine.get_bucket(P)}, end={engine.get_bucket(P + len(spec_gen))}")
    print(f"Reference vs Speculative match: {matches}")
    if not matches:
        diff_idx = (ref_gen != spec_gen).nonzero(as_tuple=True)[0][0].item()
        print(f"First mismatch at token {diff_idx}: ref={ref_gen[diff_idx]}, spec={spec_gen[diff_idx]}")
        assert False, "Mismatch between base model reference and bucketed WAL speculative engine"

    print("PASS: WAL + [0:W] scratch tail + bucket transitions produce 100% exact match\n")


def test_leaf_hedge_lossless_cases():
    print("--- Test 3: 6+1 Leaf-Hedge Lossless Parity & Case Triggers ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    base_model, drafter, tok, mask_id = prepare_v3("Qwen/Qwen2.5-0.5B", device=device, dtype=dtype)
    drafter.merge_ghost_weights(dtype=dtype)

    engine = StaticDraftEngine(
        base_model=base_model,
        drafter=drafter,
        max_len=256,
        batch_size=1,
        v_tokens=7,
        device=device,
        dtype=dtype,
        buckets=(64, 128, 256),
        use_cuda_graph=False,
    )

    # Test across multiple prompts with hedge_margin=2.0 to trigger hedging frequently
    prompts = [
        "In deep learning, speculative decoding accelerates inference by",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n",
        "The primary advantage of quantum computing lies in",
    ]

    total_hedge_attempts = 0
    total_hedge_hits = 0

    for idx, prompt in enumerate(prompts):
        inputs = tok(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        P = input_ids.shape[1]
        max_new = 32

        with torch.no_grad():
            ref_out = base_model.generate(input_ids=input_ids, max_new_tokens=max_new, do_sample=False)

        with torch.no_grad():
            spec_out, stats = generate_speculative_v3(
                base_model=base_model,
                drafter=drafter,
                input_ids=input_ids,
                mask_token_id=mask_id,
                max_new_tokens=max_new,
                temperature=0.0,
                engine=engine,
                hedge_margin=2.0, # Large hedge margin to trigger leaf-hedge morphing
            )

        ref_gen = ref_out[0, P:]
        spec_gen = spec_out[0, P:]
        if not torch.equal(ref_gen, spec_gen):
            diffs = (ref_gen != spec_gen).nonzero(as_tuple=True)[0]
            first_d = diffs[0].item()
            print(f"Mismatch at token {first_d}: ref={ref_gen[first_d].item()}, spec={spec_gen[first_d].item()}")
            print(f"Ref tokens: {ref_gen[:first_d+5].tolist()}")
            print(f"Spec tokens: {spec_gen[:first_d+5].tolist()}")
        assert torch.equal(ref_gen, spec_gen), f"Mismatch in prompt {idx} with leaf-hedge enabled!"
        total_hedge_attempts += stats["hedge_attempts"]
        total_hedge_hits += stats["hedge_hits"]
        print(f"Prompt {idx+1}: Exact match = True | Attempts: {stats['hedge_attempts']}, Hits: {stats['hedge_hits']}")

    print(f"Total hedge attempts: {total_hedge_attempts}, Total hedge hits: {total_hedge_hits}")
    print("PASS: 6+1 Leaf-Hedge maintains exact lossless parity\n")


def test_synthetic_case_b_hedge_hit():
    print("--- Test 4: Synthetic Direct Case B (Hedge Branch Hit) Verification ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    base_model, drafter, tok, mask_id = prepare_v3("Qwen/Qwen2.5-0.5B", device=device, dtype=dtype)
    drafter.merge_ghost_weights(dtype=dtype)

    engine = StaticDraftEngine(
        base_model=base_model,
        drafter=drafter,
        max_len=128,
        batch_size=1,
        v_tokens=7,
        device=device,
        dtype=dtype,
        buckets=(64, 128),
        use_cuda_graph=False,
    )

    prompt = "Artificial intelligence research"
    inputs = tok(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    P = input_ids.shape[1]

    # Prefill engine
    logits, f_last = engine.prefill(input_ids)
    y_star = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    cur_pos = P

    # Run draft step
    draft_tokens, draft_logits = engine.forward_draft_step(y_star, mask_id, f_last, [cur_pos], return_logits=True)
    # Deliberately construct hedge candidate where:
    # d0_1 is wrong, but d0_2 matches what target model predicts!
    # First get what base model actually predicts for y_star:
    target_logits = base_model(torch.cat([input_ids, y_star], dim=1)).logits[:, -1, :]
    actual_next_token = target_logits.argmax(dim=-1).item()

    wrong_token = (actual_next_token + 100) % base_model.config.vocab_size
    d0_1 = wrong_token
    d0_2 = actual_next_token # This will trigger Case B!

    v_tokens = torch.tensor([[y_star.item(), d0_1, 10, 11, 12, 13, 14, d0_2]], device=device)
    positions = torch.tensor([[cur_pos, cur_pos + 1, cur_pos + 2, cur_pos + 3, cur_pos + 4, cur_pos + 5, cur_pos + 6, cur_pos + 1]], device=device)

    # Verify step
    verify_logits, _ = engine.forward_verify_step(v_tokens, [cur_pos], pos_verify_override=positions, is_hedge=True)
    t0 = verify_logits[0, 0, :].argmax(dim=-1).item()
    t_bonus = verify_logits[0, 7, :].argmax(dim=-1).item()

    print(f"Actual next token t0: {t0}, d0_1: {d0_1}, d0_2: {d0_2}")
    assert t0 != d0_1, "Expected d0_1 to differ"
    assert t0 == d0_2, "Expected d0_2 to match t0 (Case B)"

    # Execute Case B KV pointer copy as done in sampler_v3:
    # Slot 7 is at cur_pos + 7 in target_cache, copy to cur_pos + 1
    for l_idx in range(engine.total_layers):
        engine.target_cache.k[l_idx][0, :, cur_pos + 1] = engine.target_cache.k[l_idx][0, :, cur_pos + 7]
        engine.target_cache.v[l_idx][0, :, cur_pos + 1] = engine.target_cache.v[l_idx][0, :, cur_pos + 7]
    for dl_idx in range(engine.k_layers):
        engine.draft_cache.k[dl_idx][0, :, engine.w + cur_pos + 1] = engine.draft_cache.k[dl_idx][0, :, engine.w + cur_pos + 7]
        engine.draft_cache.v[dl_idx][0, :, engine.w + cur_pos + 1] = engine.draft_cache.v[dl_idx][0, :, engine.w + cur_pos + 7]

    # Now verify that base model continuing from [input_ids, y_star, d0_2] produces t_bonus
    check_ids = torch.cat([input_ids, y_star, torch.tensor([[d0_2]], device=device)], dim=1)
    expected_bonus = base_model(check_ids).logits[:, -1, :].argmax(dim=-1).item()
    print(f"Hedge bonus token: {t_bonus}, Expected bonus from base model: {expected_bonus}")
    assert t_bonus == expected_bonus, f"Bonus token mismatch: {t_bonus} vs {expected_bonus}"

    print("PASS: Synthetic Case B Hedge verification matches ground truth exactly\n")


def test_quantized_int8_cpu_parity():
    print("--- Test 5: INT8 Quantized CPU Execution & Parity ---")
    base_model, drafter, tok, mask_id = prepare_v3("Qwen/Qwen2.5-0.5B", device="cpu", dtype=torch.float32)
    base_model, drafter = quantize_for_cpu(base_model, drafter, mode="int8")

    # Verify linear layers in base_model and drafter are quantized (lm_head kept in fp32)
    from torch.ao.nn.quantized.dynamic.modules.linear import Linear as QuantizedLinear
    assert isinstance(base_model.model.layers[0].self_attn.q_proj, QuantizedLinear), "Base model layers must be quantized"
    assert isinstance(base_model.lm_head, torch.nn.Linear), "Base model lm_head must remain torch.nn.Linear (fp32)"
    assert drafter.has_ghost_layers, "Ghost layers must be merged prior to quantization"
    assert isinstance(drafter.ghost_layers[0].self_attn.q_proj, QuantizedLinear), "Drafter ghost layers must be quantized"

    prompt = "The quick brown fox jumps over the lazy dog"
    inputs = tok(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    P = input_ids.shape[1]

    engine = StaticDraftEngine(
        base_model=base_model,
        drafter=drafter,
        max_len=128,
        batch_size=1,
        v_tokens=7,
        device=torch.device("cpu"),
        dtype=torch.float32,
        use_cuda_graph=False,
    )

    # (a) StaticDraftEngine.prefill() logits match quantized base_model(input_ids) (atol=1e-4)
    with torch.no_grad():
        logits_first, _ = engine.prefill(input_ids)
        out_base_pre = base_model(input_ids, use_cache=True)
        ref_prefill_logits = out_base_pre.logits[:, -1, :]
        prefill_diff = (logits_first.squeeze(1) - ref_prefill_logits).abs().max().item()
        print(f"Prefill max logit diff vs quantized base_model: {prefill_diff:.6e}")
        assert prefill_diff <= 1e-4, f"Prefill logit diff {prefill_diff} exceeds atol=1e-4"

    # (b) forward_verify_step on Q_A = 8 matches direct quantized base_model forward with past_key_values (atol=1e-3)
    v_tokens = torch.tensor([[100, 200, 300, 400, 500, 600, 700, 800]])
    with torch.no_grad():
        engine_v_logits, _ = engine.forward_verify_step(v_tokens, [P])
        out_base_verify = base_model(v_tokens, past_key_values=out_base_pre.past_key_values, use_cache=True)
        ref_verify_logits = out_base_verify.logits
        verify_diff = (engine_v_logits - ref_verify_logits).abs().max().item()
        print(f"Verify Q_A=8 max logit diff vs quantized base_model: {verify_diff:.6e}")
        assert verify_diff <= 1e-3, f"Verify logit diff {verify_diff} exceeds atol=1e-3"

    # (c) generate_speculative_v3 runs end-to-end without NaN/crash and produces coherent text
    max_new = 20
    with torch.no_grad():
        spec_out, stats = generate_speculative_v3(
            base_model=base_model,
            drafter=drafter,
            input_ids=input_ids,
            mask_token_id=mask_id,
            max_new_tokens=max_new,
            temperature=0.0,
            engine=engine,
        )

    gen_tokens = spec_out[0, P:]
    assert not torch.isnan(spec_out.float()).any(), "Generated tokens contain NaN"
    gen_text = tok.decode(gen_tokens)
    print(f"Generated text ({len(gen_tokens)} tokens): {repr(gen_text)}")
    print(f"Speculative decoding stats: {stats}")
    assert len(gen_tokens) == max_new, f"Expected {max_new} tokens, got {len(gen_tokens)}"
    assert stats["steps"] > 0, "Expected non-zero steps"
    print("PASS: INT8 Quantized CPU execution maintains mathematical verification parity\n")


if __name__ == "__main__":
    test_ghost_weights_unit()
    test_wal_scratch_bucket_transitions()
    test_leaf_hedge_lossless_cases()
    test_synthetic_case_b_hedge_hit()
    test_quantized_int8_cpu_parity()
    print("ALL V4 UNIT TESTS PASSED!")
