"""BFG-Diff v3: Mandatory Phase 1 Verification Gates (test_v3_gates.py).

Verifies:
- Unit Gate A: Lossless parity at T=0 vs raw model.generate() across 10 prompts (B=1 and B=4).
- Unit Gate B: Param budget <1% (<0.01) on both Qwen2.5-1.5B and Qwen2.5-3B.
- Unit Gate C: VRAM & gradient flow with fp16 base / fp32 drafter, non-zero finite gradients across
  fusion, bridge, slot_emb, and all 4-layer DraftLoRADelta weights.
- Gate 0 & Gate 1: Overhead and roofline profiling (T_verify + T_draft / T_verify(Q=1)).
"""

import argparse
import sys
import time
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig

from model_v3 import WeightTiedTopKDrafter, prepare_v3
from sampler_v3 import generate_speculative_v3

TEST_PROMPTS_10 = [
    "Once upon a time, in a small coastal village, the fishermen noticed that",
    "The old lighthouse keeper climbed the spiral staircase as the storm began to",
    "To configure the proxy, edit the settings file and set",
    "## Quick start\n\nInstall the package, then run",
    "def merge_intervals(intervals):\n    \"\"\"Merge overlapping intervals.\"\"\"\n",
    "class LRUCache:\n    def __init__(self, capacity: int):\n        self.capacity = capacity\n",
    "The primary bottleneck in distributed key-value storage occurs when",
    "During memory compaction, the garbage collector pauses application worker threads to",
    "SELECT users.id, users.email, COUNT(orders.id) AS total_orders FROM users JOIN orders ON",
    "A balanced vinaigrette needs only three things: oil, acid, and salt. Whisk two parts",
]


def test_unit_gate_b(model_names: List[str] = ["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"]) -> bool:
    """Unit Gate B: Assert trainable_params / total_params < 0.01 on both 1.5B and 3B."""
    print("\n" + "=" * 60)
    print("UNIT GATE B: Parameter Budget Check (<1% PEFT)")
    print("=" * 60)
    all_pass = True

    for m_name in model_names:
        try:
            cfg = AutoConfig.from_pretrained(m_name, local_files_only=True)
        except Exception:
            cfg = AutoConfig.from_pretrained(m_name, local_files_only=False)
        D = cfg.hidden_size
        num_heads = cfg.num_attention_heads
        num_kv = cfg.num_key_value_heads
        head_dim = getattr(cfg, "head_dim", D // num_heads)
        mlp_dim = cfg.intermediate_size
        L = cfg.num_hidden_layers
        r = 32

        # 1. Fusion: 4*D -> 512 -> D + RMSNorm(D)
        p_fusion = (4 * D * 512) + (512 * D) + D

        # 2. Bridge: 2*D -> 512 -> D + RMSNorm(D) + slot_emb(8*D)
        p_bridge = (2 * D * 512) + (512 * D) + D + (8 * D)

        # 3. 4 layers of LoRA deltas: q, k, v, o, gate, up, down
        p_delta_per_layer = (
            r * D + D * r +
            r * D + (num_kv * head_dim) * r +
            r * D + (num_kv * head_dim) * r +
            r * (num_heads * head_dim) + D * r +
            r * D + mlp_dim * r +
            r * D + mlp_dim * r +
            r * mlp_dim + D * r
        )
        p_deltas = 4 * p_delta_per_layer
        p_trainable = p_fusion + p_bridge + p_deltas

        # Total base parameters estimate
        if "1.5B" in m_name:
            est_base = 1.54e9
        elif "3B" in m_name:
            est_base = 3.09e9
        elif "0.5B" in m_name:
            est_base = 0.49e9
        else:
            est_base = 1.0e9

        ratio = p_trainable / est_base
        ratio_pct = ratio * 100.0

        print(f"\nModel: {m_name} (Layers={L}, Hidden={D}, KV_Heads={num_kv}, MLP={mlp_dim})")
        print(f"  Fusion:        {p_fusion:>10,d} ({p_fusion / 1e6:5.2f}M)")
        print(f"  Bridge:        {p_bridge:>10,d} ({p_bridge / 1e6:5.2f}M)")
        print(f"  Deltas (4L):   {p_deltas:>10,d} ({p_deltas / 1e6:5.2f}M)")
        print(f"  Total Drafter: {p_trainable:>10,d} ({p_trainable / 1e6:5.2f}M)")
        print(f"  Trainable Ratio: {ratio_pct:.3f}% (target: <1.000%, 3B target <0.850%)")

        is_ok = ratio < 0.01
        if "3B" in m_name:
            is_ok = is_ok and (ratio < 0.0085)
        print(f"  Status: {'PASS' if is_ok else 'FAIL'}")
        if not is_ok:
            all_pass = False

    return all_pass


def test_unit_gate_a(model_name: str = "Qwen/Qwen2.5-0.5B", device: str = "cpu") -> bool:
    """Unit Gate A: 100% token-identical output to raw model.generate() across 10 prompts (B=1 and B=4)."""
    print("\n" + "=" * 60)
    print(f"UNIT GATE A: Lossless Parity Check on {model_name} (device={device})")
    print("=" * 60)

    base_model, drafter, tok, mask_id = prepare_v3(
        model_name, device=device, dtype=torch.float32 if device == "cpu" else torch.float16
    )

    # 1. Test B=1 across 10 prompts
    print("\nTesting B=1 across 10 prompts (T=0, exact greedy)...")
    b1_passes = 0
    for idx, prompt in enumerate(TEST_PROMPTS_10):
        inputs = tok(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            raw_out = base_model.generate(input_ids, max_new_tokens=16, do_sample=False)
            spec_out, stats = generate_speculative_v3(
                base_model, drafter, input_ids, mask_token_id=mask_id, max_new_tokens=16
            )

        match = torch.equal(raw_out[:, : spec_out.shape[1]], spec_out)
        status = "MATCH" if match else "MISMATCH"
        print(f"  Prompt {idx + 1:2d}/10: {status} (len={spec_out.shape[1]}, alpha={stats['alpha']:.2f})")
        if match:
            b1_passes += 1

    print(f"B=1 Parity: {b1_passes}/10 passed")

    # 2. Test B=4 batched ragged generation
    print("\nTesting B=4 batched ragged generation...")
    b4_prompts = TEST_PROMPTS_10[:4]
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    inputs = tok(b4_prompts, return_tensors="pt", padding=True).to(device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        raw_out = base_model.generate(input_ids, max_new_tokens=16, do_sample=False)
        spec_out, stats = generate_speculative_v3(
            base_model, drafter, input_ids, mask_token_id=mask_id, max_new_tokens=16
        )

    b4_match = torch.equal(raw_out[:, : spec_out.shape[1]], spec_out)
    print(f"B=4 Parity Match: {b4_match} (alpha={stats['alpha']:.2f})")

    passed = (b1_passes == 10) and b4_match
    print(f"Unit Gate A Status: {'PASS' if passed else 'FAIL'}")
    return passed


def test_unit_gate_c(model_name: str = "Qwen/Qwen2.5-0.5B", device: str = "cpu") -> bool:
    """Unit Gate C: Verify grad flow across fusion, bridge, slot_emb, and all 4-layer DraftLoRADelta weights."""
    print("\n" + "=" * 60)
    print(f"UNIT GATE C: Gradient Flow Check on {model_name} (device={device})")
    print("=" * 60)

    # Base in float16 (or float32 on CPU), drafter in float32
    base_dtype = torch.float32 if device == "cpu" else torch.float16
    base_model, drafter, tok, mask_id = prepare_v3(
        model_name, device=device, dtype=base_dtype
    )

    # Dummy input sequence
    B, S = 2, 32
    W = 7
    seq = torch.randint(100, 1000, (B, S), device=device)

    # 1. Base model forward (Stage 1 verify simulation)
    with torch.no_grad():
        out = base_model(seq, output_hidden_states=True)
    f_ctx = drafter.fuse_stage1_features(out.hidden_states)

    # 2. Anchor draft pass
    anchor = 10
    y_star = seq[:, anchor : anchor + 1]
    f_ctx_last = f_ctx[:, anchor : anchor + 1]

    # Prefix KV
    pos_prefix = torch.arange(anchor + 1, device=device).unsqueeze(0).expand(B, anchor + 1)
    k_list, v_list = drafter.project_prefix_kv(f_ctx[:, : anchor + 1], pos_prefix)

    # Forward draft
    logits = drafter.forward_draft(
        y_star_ids=y_star,
        mask_token_id=mask_id,
        f_ctx_last=f_ctx_last,
        cur_pos=anchor + 1,
        w=W,
        k_prefix_list=k_list,
        v_prefix_list=v_list,
    )  # (B, W, vocab)

    # Target tokens
    targets = seq[:, anchor + 1 : anchor + 1 + W]
    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.reshape(-1))

    # Backward pass
    loss.backward()

    # Verify gradients
    all_ok = True
    print("\nChecking gradient flow through drafter submodules:")

    # Fusion gradients
    fusion_grads = [p.grad for p in drafter.fusion.parameters() if p.grad is not None]
    print(f"  Fusion: {len(fusion_grads)} grad tensors, all finite: {all(torch.isfinite(g).all() for g in fusion_grads)}")
    if not fusion_grads or not all(torch.isfinite(g).all() for g in fusion_grads):
        all_ok = False

    # Bridge gradients
    bridge_grads = [p.grad for p in drafter.bridge.parameters() if p.grad is not None]
    print(f"  Bridge: {len(bridge_grads)} grad tensors, all finite: {all(torch.isfinite(g).all() for g in bridge_grads)}")
    if not bridge_grads or not all(torch.isfinite(g).all() for g in bridge_grads):
        all_ok = False

    # Slot emb gradient
    slot_grad = drafter.bridge.slot_emb.grad
    slot_ok = slot_grad is not None and torch.isfinite(slot_grad).all() and (slot_grad.abs().sum() > 0)
    print(f"  Slot Emb: finite and non-zero: {slot_ok}")
    if not slot_ok:
        all_ok = False

    # 4-layer LoRA deltas gradients
    deltas_ok = True
    for l_idx in range(drafter.k_layers):
        ld = drafter.layer_deltas[l_idx]
        for name, delta in [
            ("q", ld.q_proj),
            ("k", ld.k_proj),
            ("v", ld.v_proj),
            ("o", ld.o_proj),
            ("gate", ld.gate_proj),
            ("up", ld.up_proj),
            ("down", ld.down_proj),
        ]:
            ga = delta.lora_A.grad
            gb = delta.lora_B.grad
            if ga is None or gb is None or not (torch.isfinite(ga).all() and torch.isfinite(gb).all()):
                deltas_ok = False
                print(f"    Layer {l_idx} {name}_proj grad issue!")

    print(f"  4-Layer LoRA Deltas (7 projections x 4 layers): all finite: {deltas_ok}")
    if not deltas_ok:
        all_ok = False

    # Verify base model has zero gradients (completely frozen)
    base_grad_count = sum(1 for p in base_model.parameters() if p.grad is not None)
    print(f"  Base Model parameters with gradients: {base_grad_count} (must be 0)")
    if base_grad_count != 0:
        all_ok = False

    print(f"Unit Gate C Status: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def test_unit_gate_d(model_name: str = "Qwen/Qwen2.5-0.5B", device: str = "cpu") -> bool:
    """Unit Gate D: Exact Numerical Identity between Training & Inference at position t.
    Enforces that train_v3's parallel block anchor forward and sampler_v3's forward_draft_step
    produce identical logits (< 1e-4) from the same prefix and inputs.
    """
    print("\n" + "=" * 60)
    print(f"UNIT GATE D: Train-Inference Numerical Parity Check ({model_name}, device={device})")
    print("=" * 60)

    base_dtype = torch.float32 if device == "cpu" else torch.float16
    base_model, drafter, tok, mask_id = prepare_v3(model_name, device=device, dtype=base_dtype)

    torch.manual_seed(42)
    S = 64
    t = 24  # anchor position t
    w = 7
    seq = torch.randint(100, 1000, (1, S), device=device)

    # 1. Training forward path for anchor position t
    with torch.no_grad():
        out = base_model(seq, output_hidden_states=True)
        f_ctx = drafter.fuse_stage1_features(out.hidden_states)
        f_last = f_ctx[:, t - 1 : t]
        y_star = seq[:, t : t + 1]

        # Training prefix KV: project full sequence once
        pos_all = torch.arange(S, device=device).unsqueeze(0)
        kp_all, vp_all = drafter.project_prefix_kv(f_ctx, pos_all)

        # Draft block bridge
        y_star_emb = drafter.embed_tokens(y_star)
        mask_emb = drafter.embed_tokens(torch.tensor([[mask_id]], device=device))
        h = drafter.bridge(y_star_emb, mask_emb, f_last, w=w)
        pos_draft = torch.arange(t, t + w, device=device).unsqueeze(0)

        # Block mask attending to prefix 0..t-1 and its own w keys
        attn_mask = torch.full((1, 1, w, S + w), -1e4, dtype=f_ctx.dtype, device=device)
        attn_mask[0, 0, :, :t] = 0.0
        attn_mask[0, 0, :, S : S + w] = 0.0

        for layer_idx in range(drafter.k_layers):
            h, _, _ = drafter.forward_draft_layer(
                layer_idx=layer_idx,
                hidden_states=h,
                pos_draft=pos_draft,
                k_prefix=kp_all[layer_idx],
                v_prefix=vp_all[layer_idx],
                attn_mask=attn_mask,
            )
        train_logits = drafter.lm_head(drafter.norm(h))  # (1, W, vocab)

    # 2. Inference forward path using StaticDraftEngine at position t
    from sampler_v3 import StaticDraftEngine
    engine = StaticDraftEngine(
        base_model=base_model,
        drafter=drafter,
        max_len=128,
        batch_size=1,
        v_tokens=w,
        device=device,
        dtype=base_dtype,
    )
    # Prefill prompt of length t (tokens 0 .. t-1)
    prompt = seq[:, :t]
    _, f_ctx_last_infer = engine.prefill(prompt)

    # Forward draft step at position t
    with torch.no_grad():
        _, infer_logits = engine.forward_draft_step(
            y_star_ids=y_star,
            mask_token_id=mask_id,
            f_ctx_last=f_ctx_last_infer,
            cur_pos_list=[t],
            return_logits=True,
        )

    thresh = 1e-4 if base_dtype == torch.float32 else 5e-2
    token_parity = torch.equal(train_logits.argmax(-1), infer_logits.argmax(-1))
    max_diff = (train_logits - infer_logits).abs().max().item()
    mean_diff = (train_logits - infer_logits).abs().mean().item()
    print(f"  Max |logits_train - logits_infer|:  {max_diff:.2e} (threshold: {thresh:.1e})")
    print(f"  Mean |logits_train - logits_infer|: {mean_diff:.2e}")
    print(f"  Argmax Token Parity: {token_parity}")

    passed = (max_diff < thresh) and token_parity
    print(f"Unit Gate D Status: {'PASS' if passed else 'FAIL'}")
    return passed


def test_gate0_gate1_profile(model_name: str = "Qwen/Qwen2.5-0.5B", device: str = "cpu") -> dict:
    """Gate 0 and Gate 1: Hardware roofline & draft cost ratio profiling with populated P=128 KV cache."""
    print("\n" + "=" * 60)
    print(f"GATE 0 & GATE 1: Overhead and Draft Cost Ratio Profile ({model_name}, device={device})")
    print("=" * 60)

    base_dtype = torch.float32 if device == "cpu" else torch.float16
    base_model, drafter, tok, mask_id = prepare_v3(model_name, device=device, dtype=base_dtype)

    from sampler_v3 import StaticDraftEngine
    P = 128
    Q_A = 8
    W = 7
    iters = 10

    def sync_cuda():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    eng = StaticDraftEngine(base_model, drafter, max_len=256, batch_size=1, v_tokens=W, device=device, dtype=base_dtype)
    prompt_ids = torch.randint(100, 1000, (1, P), device=device)
    if eng.device.type == "cuda":
        eng.capture_cuda_graphs(mask_id)
    is_graph_active = (eng.cuda_graph_verify is not None and eng.cuda_graph_draft is not None)
    print(f"  CUDA Graph Replay Active: {is_graph_active}")
    _, f_last = eng.prefill(prompt_ids)
    y_star = prompt_ids[:, -1:]
    q8_ids = torch.randint(100, 1000, (1, Q_A), device=device)
    q1_ids = prompt_ids[:, -1:]

    # Measure T_verify(Q=1) on populated P=128 cache
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            eng.forward_verify_single_token(q1_ids, cur_pos=P)
    sync_cuda()

    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            eng.forward_verify_single_token(q1_ids, cur_pos=P)
    sync_cuda()
    t_q1_ms = ((time.perf_counter() - t0) / iters) * 1000.0

    # Measure T_verify(Q_A=8) on populated P=128 cache
    for _ in range(3):
        with torch.no_grad():
            eng.forward_verify_step(q8_ids, cur_pos_list=[P])
    sync_cuda()

    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            eng.forward_verify_step(q8_ids, cur_pos_list=[P])
    sync_cuda()
    t_q8_ms = ((time.perf_counter() - t0) / iters) * 1000.0

    # Measure T_draft(4L, W=7) on populated P=128 cache
    for _ in range(3):
        with torch.no_grad():
            eng.forward_draft_step(y_star, mask_id, f_last, cur_pos_list=[P])
    sync_cuda()

    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            eng.forward_draft_step(y_star, mask_id, f_last, cur_pos_list=[P])
    sync_cuda()
    t_draft_ms = ((time.perf_counter() - t0) / iters) * 1000.0

    ratio = (t_q8_ms + t_draft_ms) / max(1e-4, t_q1_ms)

    print(f"  T_verify(Q=1, P=128):    {t_q1_ms:6.2f} ms")
    print(f"  T_verify(Q_A=8, P=128):  {t_q8_ms:6.2f} ms")
    print(f"  T_draft(4L, W=7, P=128): {t_draft_ms:6.2f} ms")
    print(f"  Draft Cost Ratio: (T_verify(8) + T_draft(7)) / T_verify(1) = {ratio:.2f}x")
    print("Gate 0 & Gate 1 profiling completed.")
    return {
        "t_q1_ms": t_q1_ms,
        "t_q8_ms": t_q8_ms,
        "t_draft_ms": t_draft_ms,
        "draft_cost_ratio": ratio,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)

    # 1. Unit Gate B (Param Budget)
    gate_b_pass = test_unit_gate_b()

    # 2. Unit Gate A (Lossless Parity)
    gate_a_pass = test_unit_gate_a(model_name=args.model, device=device)

    # 3. Unit Gate C (Gradient Flow)
    gate_c_pass = test_unit_gate_c(model_name=args.model, device=device)

    # 4. Unit Gate D (Train-Inference Parity)
    gate_d_pass = test_unit_gate_d(model_name=args.model, device=device)

    # 5. Gate 0 & 1 Roofline Profiling
    profile_stats = test_gate0_gate1_profile(model_name=args.model, device=device)

    print("\n" + "=" * 60)
    print("PHASE 1 SUMMARY:")
    print(f"  Unit Gate A (Lossless Parity):        {'PASS' if gate_a_pass else 'FAIL'}")
    print(f"  Unit Gate B (Param Budget):           {'PASS' if gate_b_pass else 'FAIL'}")
    print(f"  Unit Gate C (Gradient Flow):          {'PASS' if gate_c_pass else 'FAIL'}")
    print(f"  Unit Gate D (Train-Inference Parity): {'PASS' if gate_d_pass else 'FAIL'}")
    print("=" * 60)

    if not (gate_a_pass and gate_b_pass and gate_c_pass and gate_d_pass):
        sys.exit(1)


if __name__ == "__main__":
    main()
