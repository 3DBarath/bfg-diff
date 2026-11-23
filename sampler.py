"""BFG-Diff v4: Zero-Retraining High-Performance Speculative Decoding Engine (sampler.py).

Systems Optimizations:
1. Write-Ahead Log (WAL) draft_cache:
   - Eliminates eager commit_accepted_tokens() and full-cache concatenations in the hot loop.
   - Folded project_prefix_kv for all Q_A = 8 tokens directly into Stage 1 verification.
   - Scatters draft KV states ahead into draft_cache at W + pos_verify.
2. [0:W] In-Place Scratch Tail Layout:
   - draft_cache allocated with total sequence length W + max_len.
   - Indices [0 : W] reserved for current step's W = 7 draft KV states.
   - Prefix KV stored starting at offset W (logical pos p -> index W + p).
   - In-place writing in _step_draft_kernel completely eliminates torch.cat().
3. Power-of-2 Bucketed CUDA Graphs:
   - Captures (verify_graph, draft_graph) pairs for S_bucket in (256, 512, 1024, 2048).
   - Zero-copy shared views over single underlying target_cache and draft_cache.
   - Dynamic dispatch to smallest bucket S_b >= cur_pos + Q_A.
4. Inference-Only Merged 4-Layer Ghost Weights:
   - Pre-merges DraftLoRADelta into frozen decoder weights: W_ghost = W_base + (alpha / r) * (B @ A).
   - Bypasses 28 unfused delta additions per draft step without modifying base_model.
5. 6+1 Leaf-Hedge Mask Morphing (Q_A = 8):
   - Rescues slot-0 near-tie rejections via polymorphic static_mask_A attention geometry.
   - Evaluates slot-0 secondary candidate d0_2 in slot 7 when margin < hedge_margin.
   - 100% exact lossless parity against greedy autoregressive baseline across Case A, Case B, and Case C.
"""

import heapq
import math
import os
from typing import Callable, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import MASK_TOKEN_ID, WeightTiedTopKDrafter, apply_linear_with_delta, apply_rotary_pos_emb_k
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv


class StaticKVCache:
    """Pre-allocated static KV cache with zero crop() and in-place pointer overwriting."""

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        max_len: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.max_len = max_len
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Pre-allocate K and V tensors: List of (B, num_kv_heads, max_len, head_dim)
        self.k = [
            torch.zeros(batch_size, num_kv_heads, max_len, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.v = [
            torch.zeros(batch_size, num_kv_heads, max_len, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]

    def update(
        self,
        layer_idx: int,
        k_states: torch.Tensor,       # (B, num_kv_heads, S, head_dim)
        v_states: torch.Tensor,       # (B, num_kv_heads, S, head_dim)
        cache_positions: torch.Tensor,  # (S,) or (B, S)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Write new KV in-place at cache_positions without reallocation."""
        B, n_kv, S, d = k_states.shape
        if cache_positions.dim() == 1:
            self.k[layer_idx][:, :, cache_positions] = k_states
            self.v[layer_idx][:, :, cache_positions] = v_states
        else:
            for b in range(B):
                self.k[layer_idx][b, :, cache_positions[b]] = k_states[b]
                self.v[layer_idx][b, :, cache_positions[b]] = v_states[b]
        return self.k[layer_idx], self.v[layer_idx]

    def reset(self):
        for layer_idx in range(self.num_layers):
            self.k[layer_idx].zero_()
            self.v[layer_idx].zero_()


class StaticDraftEngine:
    """BFG-Diff v4 Speculative Decoding Inference Engine with WAL draft_cache,
    [0:W] scratch tail, power-of-2 bucketed CUDA graphs, and fused ghost weights.
    """

    DEFAULT_BUCKETS = (256, 512, 1024, 2048)

    def __init__(
        self,
        base_model: nn.Module,
        drafter: WeightTiedTopKDrafter,
        max_len: int = 2048,
        batch_size: int = 1,
        v_tokens: Optional[int] = None,  # None -> auto: 7 on CUDA, 3 on CPU
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_cuda_graph: bool = True,
        hedge_margin: float = 1.0,
        buckets: Optional[Tuple[int, ...]] = None,
        mask_token_id: int = MASK_TOKEN_ID,
        draft_vocab_ids: Optional[torch.Tensor] = None,
        q_a: Optional[int] = None,
    ):
        self.base_model = base_model
        self.drafter = drafter
        self.max_len = max_len
        self.batch_size = batch_size

        if device is None:
            device = next(base_model.parameters()).device
        if dtype is None:
            dtype = next(base_model.parameters()).dtype
        self.device = torch.device(device)
        self.dtype = dtype

        if v_tokens is None:
            v_tokens = 7 if self.device.type == "cuda" else 3
        self.v_tokens = v_tokens
        self.w = v_tokens
        if q_a is None:
            q_a = v_tokens + 1
        self.q_a = q_a
        self.use_cuda_graph = use_cuda_graph
        self.hedge_margin = float(hedge_margin)

        # Draft Vocabulary Pruning
        self.draft_vocab_ids = None
        self._draft_vocab_list = None
        self.draft_lm_head_w = None
        self.draft_lm_head_b = None
        if draft_vocab_ids is not None:
            self.draft_vocab_ids = draft_vocab_ids.to(device=self.device, dtype=torch.long).contiguous()
            self._draft_vocab_list = self.draft_vocab_ids.cpu().tolist()
            self.draft_lm_head_w = self.base_model.lm_head.weight[self.draft_vocab_ids].detach().to(device=self.device, dtype=self.dtype).contiguous()
            if getattr(self.base_model.lm_head, "bias", None) is not None:
                self.draft_lm_head_b = self.base_model.lm_head.bias[self.draft_vocab_ids].detach().to(device=self.device, dtype=self.dtype).contiguous()

        # Ensure drafter is cast to self.dtype in inference
        if self.device.type == "cuda" or self.dtype in (torch.float16, torch.bfloat16):
            self.drafter.to(dtype=self.dtype)

        # Merge ghost weights once for zero-overhead inference
        if not getattr(self.drafter, "has_ghost_layers", False):
            self.drafter.merge_ghost_weights(dtype=self.dtype)

        config = base_model.config
        self.total_layers = config.num_hidden_layers
        self.k_layers = drafter.k_layers
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)
        self.dim = config.hidden_size

        # Probe hardware/PyTorch support for enable_gqa=True with 4D float additive mask
        self.supports_enable_gqa = False
        if self.num_heads != self.num_kv_heads:
            try:
                with torch.no_grad():
                    # Deterministic non-zero tensors with non-trivial 4D mask
                    q_len, kv_len = 8, 16
                    probe_q = torch.linspace(-1, 1, self.num_heads * q_len * self.head_dim, dtype=self.dtype, device=self.device)
                    probe_q = probe_q.view(1, self.num_heads, q_len, self.head_dim)
                    probe_k = torch.linspace(-0.5, 0.5, self.num_kv_heads * kv_len * self.head_dim, dtype=self.dtype, device=self.device)
                    probe_k = probe_k.view(1, self.num_kv_heads, kv_len, self.head_dim)
                    probe_v = torch.linspace(0.1, 0.9, self.num_kv_heads * kv_len * self.head_dim, dtype=self.dtype, device=self.device)
                    probe_v = probe_v.view(1, self.num_kv_heads, kv_len, self.head_dim)

                    probe_mask = torch.zeros(1, 1, q_len, kv_len, dtype=self.dtype, device=self.device)
                    probe_mask[:, :, :, kv_len // 2:] = -1e4

                    # Reference output via repeat_kv
                    k_rep = repeat_kv(probe_k, self.num_heads // self.num_kv_heads)
                    v_rep = repeat_kv(probe_v, self.num_heads // self.num_kv_heads)
                    out_ref = F.scaled_dot_product_attention(probe_q, k_rep, v_rep, attn_mask=probe_mask)

                    # Test enable_gqa=True
                    out_gqa = F.scaled_dot_product_attention(probe_q, probe_k, probe_v, attn_mask=probe_mask, enable_gqa=True)
                    diff = (out_ref - out_gqa).abs().max().item()
                    tol = 1e-2 if self.dtype in (torch.float16, torch.bfloat16) else 1e-4
                    if diff <= tol:
                        self.supports_enable_gqa = True
            except Exception:
                self.supports_enable_gqa = False

        # Buckets for Power-of-2 CUDA Graphs (or bucketed attention views)
        cand_buckets = buckets if buckets is not None else self.DEFAULT_BUCKETS
        # max_len is always a bucket, so every position up to max_len is capturable.
        self.buckets = sorted(set(b for b in cand_buckets if b < max_len) | {max_len})

        # 1. Target Cache: full max_len across base model layers
        self.target_cache = StaticKVCache(
            num_layers=self.total_layers,
            batch_size=batch_size,
            num_kv_heads=self.num_kv_heads,
            max_len=max_len,
            head_dim=self.head_dim,
            dtype=dtype,
            device=device,
        )

        # 2. Draft Cache: [0:W] scratch tail + max_len prefix KV
        self.draft_cache = StaticKVCache(
            num_layers=self.k_layers,
            batch_size=batch_size,
            num_kv_heads=self.num_kv_heads,
            max_len=self.w + max_len,
            head_dim=self.head_dim,
            dtype=dtype,
            device=device,
        )

        # Pre-allocate static input buffers
        self.static_verify_input_ids = torch.zeros(batch_size, self.q_a, dtype=torch.long, device=device)
        self.static_pos_verify = torch.zeros(batch_size, self.q_a, dtype=torch.long, device=device)
        self.static_pos_scatter = torch.zeros(batch_size, self.q_a, dtype=torch.long, device=device)
        self.static_y_star = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        self.static_f_ctx_last = torch.zeros(batch_size, 1, self.dim, dtype=dtype, device=device)
        self.static_pos_draft = torch.zeros(batch_size, self.w, dtype=torch.long, device=device)
        self._mask_token_id = int(mask_token_id)
        self.static_mask_token = torch.tensor([[self._mask_token_id]], dtype=torch.long, device=device)

        # Pre-allocated position ranges, reused every step with an in-place add
        self.arange_qa = torch.arange(self.q_a, dtype=torch.long, device=device).unsqueeze(0)
        self.arange_w = torch.arange(self.w, dtype=torch.long, device=device).unsqueeze(0)

        # Pre-allocate master static attention masks (shared zero-copy backing buffers)
        # static_mask_A: (B, 1, Q_A, max_len)
        self.static_mask_A = torch.full((batch_size, 1, self.q_a, max_len), -1e4, dtype=dtype, device=device)
        # static_mask_B: (B, 1, W, W + max_len)
        self.static_mask_B = torch.full((batch_size, 1, self.w, self.w + max_len), -1e4, dtype=dtype, device=device)
        self.causal_qa_block = torch.triu(torch.full((self.q_a, self.q_a), -1e4, dtype=dtype, device=device), diagonal=1)
        self.causal_wxw_block = torch.triu(torch.full((self.w, self.w), -1e4, dtype=dtype, device=device), diagonal=1)
        self.causal_7x7_block = self.causal_wxw_block  # alias for backward compatibility

        # Pre-allocate static output buffers
        vocab_size = config.vocab_size
        draft_v_size = len(self.draft_vocab_ids) if self.draft_vocab_ids is not None else vocab_size
        self.static_verify_logits = torch.zeros(batch_size, self.q_a, vocab_size, dtype=dtype, device=device)
        self.static_f_ctx_verify = torch.zeros(batch_size, self.q_a, self.dim, dtype=dtype, device=device)
        self.static_draft_logits = torch.zeros(batch_size, self.w, draft_v_size, dtype=dtype, device=device)
        self.static_draft_ids = torch.zeros(batch_size, self.w, dtype=torch.long, device=device)

        # Bucketed CUDA Graph storage: dict mapping S_bucket -> (verify_graph, draft_graph)
        self.cuda_graphs = {}
        self.cuda_graph_verify = None
        self.cuda_graph_draft = None

    def get_bucket(self, cur_pos: int) -> int:
        """Find the smallest bucket that covers cur_pos + Q_A. max_len is always a bucket."""
        needed = cur_pos + self.q_a
        for b in self.buckets:
            if b >= needed:
                return b
        raise ValueError(
            f"position {cur_pos} + Q_A {self.q_a} exceeds engine max_len {self.max_len}. "
            "Create the engine with a larger max_len."
        )

    def capture_cuda_graphs(self, mask_token_id: int):
        """Warmup and capture separate power-of-2 bucketed CUDA graphs on CUDA for B=1."""
        if self.device.type != "cuda" or self.batch_size != 1 or not self.use_cuda_graph:
            return
        if not hasattr(torch.cuda, "CUDAGraph"):
            return
        if len(self.cuda_graphs) > 0:
            return

        self.static_mask_token.fill_(mask_token_id)
        self._mask_token_id = int(mask_token_id)
        self.drafter.to(dtype=self.dtype)
        if not getattr(self.drafter, "has_ghost_layers", False):
            self.drafter.merge_ghost_weights(dtype=self.dtype)

        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for S_b in self.buckets:
                    # Setup dummy inputs for bucket S_b
                    self.static_verify_input_ids.zero_()
                    self.static_pos_verify.copy_(torch.arange(0, self.q_a, device=self.device).unsqueeze(0))
                    self.static_pos_scatter.copy_(torch.arange(0, self.q_a, device=self.device).unsqueeze(0))
                    self.static_mask_A[:, :, :, :S_b].fill_(-1e4)
                    self.static_mask_A[0, 0, :, :self.q_a] = self.causal_qa_block

                    self.static_y_star.zero_()
                    self.static_f_ctx_last.zero_()
                    self.static_pos_draft.copy_(torch.arange(0, self.w, device=self.device).unsqueeze(0))
                    self.static_mask_B[:, :, :, :self.w + S_b].fill_(-1e4)
                    self.static_mask_B[0, 0, :, :self.w] = 0.0

                    # Warmup iterations
                    for _ in range(3):
                        self._step_verify_kernel(S_b)
                        self._step_draft_kernel(S_b)

                    # Capture Stage 1 verify graph for bucket S_b
                    g_verify = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g_verify):
                        self._step_verify_kernel(S_b)

                    # Capture Stage 2 draft graph for bucket S_b
                    g_draft = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g_draft):
                        self._step_draft_kernel(S_b)

                    self.cuda_graphs[S_b] = (g_verify, g_draft)

            torch.cuda.current_stream().wait_stream(s)

            if len(self.cuda_graphs) > 0:
                first_bucket = self.buckets[0]
                self.cuda_graph_verify, self.cuda_graph_draft = self.cuda_graphs[first_bucket]
                print(f"[StaticDraftEngine] Successfully captured {len(self.cuda_graphs)} bucketed CUDA Graph pairs: {self.buckets}")
        except Exception as e:
            print(f"[WARN] CUDA Graph capture failed, falling back to eager: {e}")
            self.cuda_graphs = {}
            self.cuda_graph_verify = None
            self.cuda_graph_draft = None

    def _step_verify_kernel(self, S_b: int):
        """Execution kernel for Stage 1 verify on bucket S_b with WAL write to draft_cache."""
        h = self.base_model.model.embed_tokens(self.static_verify_input_ids)
        cos, sin = self.base_model.model.rotary_emb(h, self.static_pos_verify)

        mask_A_view = self.static_mask_A[:, :, :, :S_b]
        all_hidden_states = [h]

        for l_idx, layer in enumerate(self.base_model.model.layers):
            norm_h = layer.input_layernorm(h)
            q = layer.self_attn.q_proj(norm_h).view(1, self.q_a, self.num_heads, self.head_dim).transpose(1, 2)
            k = layer.self_attn.k_proj(norm_h).view(1, self.q_a, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = layer.self_attn.v_proj(norm_h).view(1, self.q_a, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

            idx = self.static_pos_scatter.view(1, 1, self.q_a, 1).expand(1, self.num_kv_heads, self.q_a, self.head_dim)
            self.target_cache.k[l_idx].scatter_(2, idx, k)
            self.target_cache.v[l_idx].scatter_(2, idx, v)

            k_active = self.target_cache.k[l_idx][:, :, :S_b, :]
            v_active = self.target_cache.v[l_idx][:, :, :S_b, :]

            if self.num_heads != self.num_kv_heads:
                if self.supports_enable_gqa:
                    attn_out = F.scaled_dot_product_attention(q, k_active, v_active, attn_mask=mask_A_view, enable_gqa=True)
                else:
                    k_attn = repeat_kv(k_active, self.num_heads // self.num_kv_heads)
                    v_attn = repeat_kv(v_active, self.num_heads // self.num_kv_heads)
                    attn_out = F.scaled_dot_product_attention(q, k_attn, v_attn, attn_mask=mask_A_view)
            else:
                attn_out = F.scaled_dot_product_attention(q, k_active, v_active, attn_mask=mask_A_view)
            attn_out = attn_out.transpose(1, 2).reshape(1, self.q_a, self.dim)
            h = h + layer.self_attn.o_proj(attn_out)
            norm_post = layer.post_attention_layernorm(h)
            h = h + layer.mlp(norm_post)

            if l_idx == self.total_layers - 1:
                all_hidden_states.append(self.base_model.model.norm(h))
            else:
                all_hidden_states.append(h)

        h_norm = self.base_model.model.norm(h)
        logits = self.base_model.lm_head(h_norm)
        self.static_verify_logits.copy_(logits)
        f_ctx = self.drafter.fuse_stage1_features(tuple(all_hidden_states))
        self.static_f_ctx_verify.copy_(f_ctx)

        # WAL: Project and scatter draft prefix KV for all Q_A=8 tokens directly into draft_cache
        cos_d_k, sin_d_k = self.drafter.rotary_emb(f_ctx, self.static_pos_verify)
        idx_draft = (self.w + self.static_pos_scatter).view(1, 1, self.q_a, 1).expand(1, self.num_kv_heads, self.q_a, self.head_dim)

        for d_l_idx in range(self.k_layers):
            if self.drafter.has_ghost_layers:
                ghost_l = self.drafter.ghost_layers[d_l_idx]
                k_d = ghost_l.self_attn.k_proj(f_ctx)
                v_d = ghost_l.self_attn.v_proj(f_ctx)
            else:
                base_l = self.drafter.draft_layers[d_l_idx]
                deltas = self.drafter.layer_deltas[d_l_idx]
                k_d = apply_linear_with_delta(base_l.self_attn.k_proj, f_ctx, deltas.k_proj)
                v_d = apply_linear_with_delta(base_l.self_attn.v_proj, f_ctx, deltas.v_proj)

            k_d = k_d.view(1, self.q_a, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v_d = v_d.view(1, self.q_a, self.num_kv_heads, self.head_dim).transpose(1, 2)
            k_d = apply_rotary_pos_emb_k(k_d, cos_d_k, sin_d_k, unsqueeze_dim=1)

            self.draft_cache.k[d_l_idx].scatter_(2, idx_draft, k_d)
            self.draft_cache.v[d_l_idx].scatter_(2, idx_draft, v_d)

    def _step_draft_kernel(self, S_b: int):
        """Execution kernel for Stage 2 draft on bucket S_b with [0:W] in-place scratch tail."""
        y_star_emb = self.drafter.embed_tokens(self.static_y_star)
        mask_emb = self.drafter.embed_tokens(self.static_mask_token)
        h = self.drafter.bridge(y_star_emb, mask_emb, self.static_f_ctx_last, w=self.w)
        mask_B_view = self.static_mask_B[:, :, :, :self.w + S_b]

        for layer_idx in range(self.k_layers):
            if self.drafter.has_ghost_layers:
                ghost_l = self.drafter.ghost_layers[layer_idx]
                norm_h = ghost_l.input_layernorm(h)
                q = ghost_l.self_attn.q_proj(norm_h).view(1, self.w, self.num_heads, self.head_dim).transpose(1, 2)
                k_d = ghost_l.self_attn.k_proj(norm_h).view(1, self.w, self.num_kv_heads, self.head_dim).transpose(1, 2)
                v_d = ghost_l.self_attn.v_proj(norm_h).view(1, self.w, self.num_kv_heads, self.head_dim).transpose(1, 2)
            else:
                base_l = self.drafter.draft_layers[layer_idx]
                deltas = self.drafter.layer_deltas[layer_idx]
                norm_h = base_l.input_layernorm(h)
                q = apply_linear_with_delta(base_l.self_attn.q_proj, norm_h, deltas.q_proj)
                q = q.view(1, self.w, self.num_heads, self.head_dim).transpose(1, 2)
                k_d = apply_linear_with_delta(base_l.self_attn.k_proj, norm_h, deltas.k_proj)
                k_d = k_d.view(1, self.w, self.num_kv_heads, self.head_dim).transpose(1, 2)
                v_d = apply_linear_with_delta(base_l.self_attn.v_proj, norm_h, deltas.v_proj)
                v_d = v_d.view(1, self.w, self.num_kv_heads, self.head_dim).transpose(1, 2)

            cos_d, sin_d = self.drafter.rotary_emb(norm_h, self.static_pos_draft)
            q, k_d = apply_rotary_pos_emb(q, k_d, cos_d, sin_d, unsqueeze_dim=1)

            # In-place write to [0 : W] scratch tail (zero reallocation)
            self.draft_cache.k[layer_idx][:, :, :self.w, :] = k_d
            self.draft_cache.v[layer_idx][:, :, :self.w, :] = v_d

            # Zero-copy view: scratch tail [0:W] + prefix [:S_b]
            k_total = self.draft_cache.k[layer_idx][:, :, :self.w + S_b, :]
            v_total = self.draft_cache.v[layer_idx][:, :, :self.w + S_b, :]

            if self.num_heads != self.num_kv_heads:
                if self.supports_enable_gqa:
                    attn_out = F.scaled_dot_product_attention(q, k_total, v_total, attn_mask=mask_B_view, enable_gqa=True)
                else:
                    k_attn = repeat_kv(k_total, self.num_heads // self.num_kv_heads)
                    v_attn = repeat_kv(v_total, self.num_heads // self.num_kv_heads)
                    attn_out = F.scaled_dot_product_attention(q, k_attn, v_attn, attn_mask=mask_B_view)
            else:
                attn_out = F.scaled_dot_product_attention(q, k_total, v_total, attn_mask=mask_B_view)
            attn_out = attn_out.transpose(1, 2).reshape(1, self.w, self.dim)

            if self.drafter.has_ghost_layers:
                h = h + ghost_l.self_attn.o_proj(attn_out)
                norm_post = ghost_l.post_attention_layernorm(h)
                gate = ghost_l.mlp.gate_proj(norm_post)
                up = ghost_l.mlp.up_proj(norm_post)
                mlp_act = ghost_l.mlp.act_fn(gate) * up
                h = h + ghost_l.mlp.down_proj(mlp_act)
            else:
                h = h + apply_linear_with_delta(base_l.self_attn.o_proj, attn_out, deltas.o_proj)
                norm_post = base_l.post_attention_layernorm(h)
                gate = apply_linear_with_delta(base_l.mlp.gate_proj, norm_post, deltas.gate_proj)
                up = apply_linear_with_delta(base_l.mlp.up_proj, norm_post, deltas.up_proj)
                mlp_act = base_l.mlp.act_fn(gate) * up
                h = h + apply_linear_with_delta(base_l.mlp.down_proj, mlp_act, deltas.down_proj)

        h_norm = self.drafter.norm(h)
        if self.draft_lm_head_w is not None:
            logits = F.linear(h_norm, self.draft_lm_head_w, self.draft_lm_head_b)
            self.static_draft_logits.copy_(logits)
            local_ids = torch.argmax(logits, dim=-1)
            self.static_draft_ids.copy_(self.draft_vocab_ids[local_ids])
        else:
            logits = self.drafter.lm_head(h_norm)
            self.static_draft_logits.copy_(logits)
            self.static_draft_ids.copy_(torch.argmax(logits, dim=-1))

    def prefill(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stage 1 prefill on prompt tokens (length P).
        Writes target KV into target_cache at 0 .. P-1.
        Computes F_ctx (0 .. P-1) and writes prefix KV into draft_cache starting at offset W (W .. W + P - 1).
        Returns: (first_token_logits, f_ctx_last = f_ctx[:, P-1:P])
        """
        B, P = input_ids.shape
        self.target_cache.reset()
        self.draft_cache.reset()

        causal_mask = torch.tril(torch.ones(P, P, dtype=torch.bool, device=self.device))
        attn_mask = torch.where(causal_mask, 0.0, -1e4).to(self.dtype).unsqueeze(0).unsqueeze(0)
        pos_ids = torch.arange(P, device=self.device).unsqueeze(0).expand(B, P)

        h = self.base_model.model.embed_tokens(input_ids)
        cos, sin = self.base_model.model.rotary_emb(h, pos_ids)

        all_hidden_states = [h]
        for l_idx, layer in enumerate(self.base_model.model.layers):
            norm_h = layer.input_layernorm(h)
            q = layer.self_attn.q_proj(norm_h).view(B, P, self.num_heads, self.head_dim).transpose(1, 2)
            k = layer.self_attn.k_proj(norm_h).view(B, P, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = layer.self_attn.v_proj(norm_h).view(B, P, self.num_kv_heads, self.head_dim).transpose(1, 2)

            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
            self.target_cache.update(l_idx, k, v, torch.arange(P, device=self.device))

            if self.num_heads != self.num_kv_heads:
                if self.supports_enable_gqa:
                    attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=True)
                else:
                    n_rep = self.num_heads // self.num_kv_heads
                    k_attn = repeat_kv(k, n_rep)
                    v_attn = repeat_kv(v, n_rep)
                    attn_out = F.scaled_dot_product_attention(q, k_attn, v_attn, attn_mask=attn_mask)
            else:
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            attn_out = attn_out.transpose(1, 2).reshape(B, P, self.dim)
            h = h + layer.self_attn.o_proj(attn_out)

            norm_post = layer.post_attention_layernorm(h)
            h = h + layer.mlp(norm_post)
            if l_idx == self.total_layers - 1:
                all_hidden_states.append(self.base_model.model.norm(h))
            else:
                all_hidden_states.append(h)

        h_norm = self.base_model.model.norm(h[:, -1:])
        logits = self.base_model.lm_head(h_norm)  # (B, 1, vocab) predicts token P

        # Multi-layer target feature fusion on prefix (0 .. P-1)
        f_ctx = self.drafter.fuse_stage1_features(tuple(all_hidden_states))  # (B, P, D)
        f_ctx_last = f_ctx[:, -1:].clone()  # f_ctx[P-1]

        # Project and write prefix KV into draft_cache starting at offset W (W .. W + P - 1)
        k_list, v_list = self.drafter.project_prefix_kv(f_ctx, pos_ids)
        draft_pos = torch.arange(self.w, self.w + P, device=self.device)
        for d_l_idx in range(self.k_layers):
            self.draft_cache.update(d_l_idx, k_list[d_l_idx], v_list[d_l_idx], draft_pos)

        return logits, f_ctx_last

    def forward_verify_single_token(
        self,
        token: torch.Tensor,  # (B, 1)
        cur_pos: int,
    ) -> torch.Tensor:
        """Measure single-token verification on populated target_cache at position cur_pos."""
        B = token.shape[0]
        pos_id = torch.tensor([[cur_pos]], device=self.device)
        max_total_len = cur_pos + 1
        mask = torch.full((B, 1, 1, max_total_len), -1e4, dtype=self.dtype, device=self.device)
        mask[:, :, :, :max_total_len] = 0.0

        h = self.base_model.model.embed_tokens(token)
        cos, sin = self.base_model.model.rotary_emb(h, pos_id)

        all_hidden_states = [h]
        for l_idx, layer in enumerate(self.base_model.model.layers):
            norm_h = layer.input_layernorm(h)
            q = layer.self_attn.q_proj(norm_h).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
            k = layer.self_attn.k_proj(norm_h).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = layer.self_attn.v_proj(norm_h).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

            self.target_cache.k[l_idx][:, :, cur_pos : cur_pos + 1] = k
            self.target_cache.v[l_idx][:, :, cur_pos : cur_pos + 1] = v

            k_active = self.target_cache.k[l_idx][:, :, :max_total_len]
            v_active = self.target_cache.v[l_idx][:, :, :max_total_len]
            if self.num_heads != self.num_kv_heads:
                if self.supports_enable_gqa:
                    attn_out = F.scaled_dot_product_attention(q, k_active, v_active, attn_mask=mask, enable_gqa=True)
                else:
                    k_attn = repeat_kv(k_active, self.num_heads // self.num_kv_heads)
                    v_attn = repeat_kv(v_active, self.num_heads // self.num_kv_heads)
                    attn_out = F.scaled_dot_product_attention(q, k_attn, v_attn, attn_mask=mask)
            else:
                attn_out = F.scaled_dot_product_attention(q, k_active, v_active, attn_mask=mask)
            attn_out = attn_out.transpose(1, 2).reshape(B, 1, self.dim)
            h = h + layer.self_attn.o_proj(attn_out)
            norm_post = layer.post_attention_layernorm(h)
            h = h + layer.mlp(norm_post)
            if l_idx == self.total_layers - 1:
                all_hidden_states.append(self.base_model.model.norm(h))
            else:
                all_hidden_states.append(h)

        logits = self.base_model.lm_head(self.base_model.model.norm(h))

        # WAL: project and write single token draft KV ahead to draft_cache at W + cur_pos
        f_ctx = self.drafter.fuse_stage1_features(tuple(all_hidden_states))
        cos_d, sin_d = self.drafter.rotary_emb(f_ctx, pos_id)
        for d_l_idx in range(self.k_layers):
            if self.drafter.has_ghost_layers:
                ghost_l = self.drafter.ghost_layers[d_l_idx]
                k_d = ghost_l.self_attn.k_proj(f_ctx)
                v_d = ghost_l.self_attn.v_proj(f_ctx)
            else:
                base_l = self.drafter.draft_layers[d_l_idx]
                deltas = self.drafter.layer_deltas[d_l_idx]
                k_d = apply_linear_with_delta(base_l.self_attn.k_proj, f_ctx, deltas.k_proj)
                v_d = apply_linear_with_delta(base_l.self_attn.v_proj, f_ctx, deltas.v_proj)
            k_d = k_d.view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v_d = v_d.view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
            k_d = apply_rotary_pos_emb_k(k_d, cos_d, sin_d, unsqueeze_dim=1)
            self.draft_cache.k[d_l_idx][:, :, self.w + cur_pos : self.w + cur_pos + 1] = k_d
            self.draft_cache.v[d_l_idx][:, :, self.w + cur_pos : self.w + cur_pos + 1] = v_d

        return logits

    def forward_verify_step(
        self,
        verify_tokens: torch.Tensor,       # (B, Q_A) [y*, d_0, ...] or tree candidate tile
        cur_pos_list: List[int],           # list of B ints: sequence position t
        is_hedge: bool = False,
        pos_verify_override: Optional[torch.Tensor] = None,
        tree_mask_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stage 1 verify pass over Q_A tokens with WAL write-ahead logging into draft_cache.
        Uses Power-of-2 Bucketed CUDA Graphs on CUDA B=1; runs eager static fallback on CPU / B > 1.
        Supports linear causal chain, legacy leaf-hedge, or tree_mask_override (Q_A, Q_A).
        """
        B, Q_A = verify_tokens.shape
        t = cur_pos_list[0]
        S_b = self.get_bucket(t)

        if pos_verify_override is not None:
            if not isinstance(pos_verify_override, torch.Tensor):
                pos_verify_override = torch.tensor(pos_verify_override, device=self.device)
            if pos_verify_override.ndim == 1:
                pos_verify_override = pos_verify_override.unsqueeze(0)

        # CUDA Graph replay path for B=1
        if len(self.cuda_graphs) > 0 and B == 1 and S_b in self.cuda_graphs:
            self.static_verify_input_ids.copy_(verify_tokens)
            self.static_pos_scatter.copy_(self.arange_qa).add_(t)
            if pos_verify_override is not None:
                self.static_pos_verify.copy_(pos_verify_override)
            else:
                self.static_pos_verify.copy_(self.static_pos_scatter)

            mask_A_view = self.static_mask_A[:, :, :, :S_b]
            mask_A_view.fill_(-1e4)
            if t > 0:
                mask_A_view[0, 0, :, :t] = 0.0

            if tree_mask_override is not None:
                # Arbitrary static tree ancestor topology
                mask_A_view[0, 0, :, t : t + Q_A].copy_(tree_mask_override)
            elif not is_hedge:
                # Standard linear causal chain
                mask_A_view[0, 0, :, t : t + Q_A] = self.causal_qa_block
            else:
                # Leaf-Hedge Micro-Tree:
                # Slots 0..W-1: causal chain over [y*, d0_1, ..., d_{W-2}]
                mask_A_view[0, 0, :self.w, t : t + self.w] = self.causal_wxw_block
                mask_A_view[0, 0, :self.w, t + self.w] = -1e4
                # Slot Q_A-1 (d0_2): attends ONLY to prefix [0:t] + slot 0 (y*) + slot Q_A-1 (itself). Slots 1..W-1 masked out!
                mask_A_view[0, 0, self.q_a - 1, t : t + self.q_a] = -1e4
                mask_A_view[0, 0, self.q_a - 1, t] = 0.0          # slot 0 (y*)
                mask_A_view[0, 0, self.q_a - 1, t + self.w] = 0.0  # slot Q_A-1 (itself)

            g_verify, _ = self.cuda_graphs[S_b]
            self.cuda_graph_verify = g_verify
            g_verify.replay()
            return self.static_verify_logits, self.static_f_ctx_verify

        # Eager execution path (CPU or batched B > 1 on CUDA without graphs)
        max_total_len = max(cur_pos_list) + Q_A
        if self.device.type == "cuda":
            S_active = max(S_b, max_total_len)
        else:
            S_active = max_total_len  # CPU: dynamic zero-padding KV slicing

        if B == 1 and S_active <= self.max_len:
            mask = self.static_mask_A[:1, :, :, :S_active]
            pos_ids = self.static_pos_verify[:1]
        else:
            pos_ids = torch.zeros(B, Q_A, dtype=torch.long, device=self.device)
            mask = torch.full((B, 1, Q_A, S_active), -1e4, dtype=self.dtype, device=self.device)

        mask.fill_(-1e4)
        for b in range(B):
            cp = cur_pos_list[b]
            if pos_verify_override is not None and b == 0:
                pos_ids[b].copy_(pos_verify_override[0])
            else:
                pos_ids[b].copy_(torch.arange(cp, cp + Q_A, device=self.device))
            if cp > 0:
                mask[b, 0, :, :cp] = 0.0

            if tree_mask_override is not None and b == 0:
                mask[b, 0, :, cp : cp + Q_A].copy_(tree_mask_override)
            elif not is_hedge or b > 0:
                mask[b, 0, :, cp : cp + Q_A] = self.causal_qa_block
            else:
                mask[b, 0, :self.w, cp : cp + self.w] = self.causal_wxw_block
                mask[b, 0, :self.w, cp + self.w] = -1e4
                mask[b, 0, self.q_a - 1, cp : cp + self.q_a] = -1e4
                mask[b, 0, self.q_a - 1, cp] = 0.0
                mask[b, 0, self.q_a - 1, cp + self.w] = 0.0

        h = self.base_model.model.embed_tokens(verify_tokens)
        cos, sin = self.base_model.model.rotary_emb(h, pos_ids)

        all_hidden_states = [h]
        for l_idx, layer in enumerate(self.base_model.model.layers):
            norm_h = layer.input_layernorm(h)
            q = layer.self_attn.q_proj(norm_h).view(B, Q_A, self.num_heads, self.head_dim).transpose(1, 2)
            k = layer.self_attn.k_proj(norm_h).view(B, Q_A, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = layer.self_attn.v_proj(norm_h).view(B, Q_A, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

            for b in range(B):
                cp = cur_pos_list[b]
                scatter_positions = torch.arange(cp, cp + Q_A, device=self.device)
                self.target_cache.k[l_idx][b, :, scatter_positions] = k[b]
                self.target_cache.v[l_idx][b, :, scatter_positions] = v[b]

            k_active = self.target_cache.k[l_idx][:, :, :S_active]
            v_active = self.target_cache.v[l_idx][:, :, :S_active]

            if self.num_heads != self.num_kv_heads:
                if self.supports_enable_gqa:
                    attn_out = F.scaled_dot_product_attention(q, k_active, v_active, attn_mask=mask, enable_gqa=True)
                else:
                    n_rep = self.num_heads // self.num_kv_heads
                    k_attn = repeat_kv(k_active, n_rep)
                    v_attn = repeat_kv(v_active, n_rep)
                    attn_out = F.scaled_dot_product_attention(q, k_attn, v_attn, attn_mask=mask)
            else:
                attn_out = F.scaled_dot_product_attention(q, k_active, v_active, attn_mask=mask)
            attn_out = attn_out.transpose(1, 2).reshape(B, Q_A, self.dim)
            h = h + layer.self_attn.o_proj(attn_out)
            norm_post = layer.post_attention_layernorm(h)
            h = h + layer.mlp(norm_post)

            if l_idx == self.total_layers - 1:
                all_hidden_states.append(self.base_model.model.norm(h))
            else:
                all_hidden_states.append(h)

        h_norm = self.base_model.model.norm(h)
        logits = self.base_model.lm_head(h_norm)
        f_ctx = self.drafter.fuse_stage1_features(tuple(all_hidden_states))

        # WAL: Project and write draft KV states ahead to draft_cache at W + pos_ids (CUDA only)
        if self.device.type == "cuda":
            cos_d_k, sin_d_k = self.drafter.rotary_emb(f_ctx, pos_ids)
            for d_l_idx in range(self.k_layers):
                if self.drafter.has_ghost_layers:
                    ghost_l = self.drafter.ghost_layers[d_l_idx]
                    k_d = ghost_l.self_attn.k_proj(f_ctx)
                    v_d = ghost_l.self_attn.v_proj(f_ctx)
                else:
                    base_l = self.drafter.draft_layers[d_l_idx]
                    deltas = self.drafter.layer_deltas[d_l_idx]
                    k_d = apply_linear_with_delta(base_l.self_attn.k_proj, f_ctx, deltas.k_proj)
                    v_d = apply_linear_with_delta(base_l.self_attn.v_proj, f_ctx, deltas.v_proj)

                k_d = k_d.view(B, Q_A, self.num_kv_heads, self.head_dim).transpose(1, 2)
                v_d = v_d.view(B, Q_A, self.num_kv_heads, self.head_dim).transpose(1, 2)
                k_d = apply_rotary_pos_emb_k(k_d, cos_d_k, sin_d_k, unsqueeze_dim=1)

                for b in range(B):
                    cp = cur_pos_list[b]
                    draft_positions = self.w + torch.arange(cp, cp + Q_A, device=self.device)
                    self.draft_cache.k[d_l_idx][b, :, draft_positions] = k_d[b]
                    self.draft_cache.v[d_l_idx][b, :, draft_positions] = v_d[b]

        return logits, f_ctx

    def forward_draft_step(
        self,
        y_star_ids: torch.Tensor,    # (B, 1) concrete bonus token at position t
        mask_token_id: int,          # [MASK] token id, from setup_tokenizer
        f_ctx_last: torch.Tensor,    # (B, 1, D) F_ctx of token at t - 1
        cur_pos_list: List[int],     # list of B ints: sequence position t
        return_logits: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Stage 2 draft pass: produces W = 7 draft tokens predicting [t+1 .. t+7].
        Writes draft KV states in-place into scratch tail [0 : W] of draft_cache.
        Uses Power-of-2 Bucketed CUDA Graphs on CUDA B=1; runs eager static fallback on CPU / B > 1.
        """
        B = y_star_ids.shape[0]
        W = self.w
        t = cur_pos_list[0]
        S_b = self.get_bucket(t)

        # The [MASK] row is a graph input buffer. Refill it only when the id changes.
        if self._mask_token_id != mask_token_id:
            self.static_mask_token.fill_(mask_token_id)
            self._mask_token_id = int(mask_token_id)

        # CUDA Graph replay path for B=1
        if len(self.cuda_graphs) > 0 and B == 1 and S_b in self.cuda_graphs:
            self.static_y_star.copy_(y_star_ids)
            self.static_f_ctx_last.copy_(f_ctx_last)
            self.static_pos_draft.copy_(self.arange_w).add_(t)

            mask_B_view = self.static_mask_B[:, :, :, :self.w + S_b]
            mask_B_view.fill_(-1e4)
            # 1. Attends to W draft tokens in scratch tail [0:W]
            mask_B_view[0, 0, :, :W] = 0.0
            # 2. Attends to valid prefix in [W : W + t]
            if t > 0:
                mask_B_view[0, 0, :, W : W + t] = 0.0

            _, g_draft = self.cuda_graphs[S_b]
            self.cuda_graph_draft = g_draft
            g_draft.replay()
            if return_logits:
                return self.static_draft_ids, self.static_draft_logits
            return self.static_draft_ids

        # Eager execution path (CPU or batched B > 1 on CUDA without graphs)
        y_star_emb = self.drafter.embed_tokens(y_star_ids)
        mask_emb = self.drafter.embed_tokens(self.static_mask_token)

        h = self.drafter.bridge(y_star_emb, mask_emb, f_ctx_last, w=W)
        max_prefix_len = max(cur_pos_list)
        if self.device.type == "cuda":
            S_active = max(S_b, max_prefix_len)
        else:
            S_active = max_prefix_len  # CPU: dynamic zero-padding KV slicing

        if B == 1 and (W + S_active) <= (self.w + self.max_len):
            pos_draft = self.static_pos_draft[:1]
            attn_mask = self.static_mask_B[:1, :, :, :W + S_active]
        else:
            pos_draft = torch.zeros(B, W, dtype=torch.long, device=self.device)
            attn_mask = torch.full((B, 1, W, W + S_active), -1e4, dtype=self.dtype, device=self.device)

        attn_mask.fill_(-1e4)
        for b in range(B):
            cp = cur_pos_list[b]
            pos_draft[b].copy_(torch.arange(cp, cp + W, device=self.device))
            attn_mask[b, 0, :, :W] = 0.0
            if cp > 0:
                attn_mask[b, 0, :, W : W + cp] = 0.0

        for layer_idx in range(self.k_layers):
            if self.drafter.has_ghost_layers:
                ghost_l = self.drafter.ghost_layers[layer_idx]
                norm_h = ghost_l.input_layernorm(h)
                q = ghost_l.self_attn.q_proj(norm_h).view(B, W, self.num_heads, self.head_dim).transpose(1, 2)
                k_d = ghost_l.self_attn.k_proj(norm_h).view(B, W, self.num_kv_heads, self.head_dim).transpose(1, 2)
                v_d = ghost_l.self_attn.v_proj(norm_h).view(B, W, self.num_kv_heads, self.head_dim).transpose(1, 2)
            else:
                base_l = self.drafter.draft_layers[layer_idx]
                deltas = self.drafter.layer_deltas[layer_idx]
                norm_h = base_l.input_layernorm(h)
                q = apply_linear_with_delta(base_l.self_attn.q_proj, norm_h, deltas.q_proj)
                q = q.view(B, W, self.num_heads, self.head_dim).transpose(1, 2)
                k_d = apply_linear_with_delta(base_l.self_attn.k_proj, norm_h, deltas.k_proj)
                k_d = k_d.view(B, W, self.num_kv_heads, self.head_dim).transpose(1, 2)
                v_d = apply_linear_with_delta(base_l.self_attn.v_proj, norm_h, deltas.v_proj)
                v_d = v_d.view(B, W, self.num_kv_heads, self.head_dim).transpose(1, 2)

            cos_d, sin_d = self.drafter.rotary_emb(norm_h, pos_draft)
            q, k_d = apply_rotary_pos_emb(q, k_d, cos_d, sin_d, unsqueeze_dim=1)

            # In-place write to [0 : W] scratch tail
            self.draft_cache.k[layer_idx][:, :, :W, :] = k_d
            self.draft_cache.v[layer_idx][:, :, :W, :] = v_d

            k_total = self.draft_cache.k[layer_idx][:, :, :W + S_active, :]
            v_total = self.draft_cache.v[layer_idx][:, :, :W + S_active, :]

            if self.num_heads != self.num_kv_heads:
                if self.supports_enable_gqa:
                    attn_out = F.scaled_dot_product_attention(q, k_total, v_total, attn_mask=attn_mask, enable_gqa=True)
                else:
                    n_rep = self.num_heads // self.num_kv_heads
                    k_attn = repeat_kv(k_total, n_rep)
                    v_attn = repeat_kv(v_total, n_rep)
                    attn_out = F.scaled_dot_product_attention(q, k_attn, v_attn, attn_mask=attn_mask)
            else:
                attn_out = F.scaled_dot_product_attention(q, k_total, v_total, attn_mask=attn_mask)
            attn_out = attn_out.transpose(1, 2).reshape(B, W, self.dim)

            if self.drafter.has_ghost_layers:
                h = h + ghost_l.self_attn.o_proj(attn_out)
                norm_post = ghost_l.post_attention_layernorm(h)
                gate = ghost_l.mlp.gate_proj(norm_post)
                up = ghost_l.mlp.up_proj(norm_post)
                mlp_act = ghost_l.mlp.act_fn(gate) * up
                h = h + ghost_l.mlp.down_proj(mlp_act)
            else:
                h = h + apply_linear_with_delta(base_l.self_attn.o_proj, attn_out, deltas.o_proj)
                norm_post = base_l.post_attention_layernorm(h)
                gate = apply_linear_with_delta(base_l.mlp.gate_proj, norm_post, deltas.gate_proj)
                up = apply_linear_with_delta(base_l.mlp.up_proj, norm_post, deltas.up_proj)
                mlp_act = base_l.mlp.act_fn(gate) * up
                h = h + apply_linear_with_delta(base_l.mlp.down_proj, mlp_act, deltas.down_proj)

        h_norm = self.drafter.norm(h)
        if self.draft_lm_head_w is not None:
            logits = F.linear(h_norm, self.draft_lm_head_w, self.draft_lm_head_b)
            local_ids = torch.argmax(logits, dim=-1)
            draft_ids = self.draft_vocab_ids[local_ids]
        else:
            logits = self.drafter.lm_head(h_norm)
            draft_ids = torch.argmax(logits, dim=-1)

        if return_logits:
            return draft_ids, logits
        return draft_ids

    def commit_accepted_tokens(
        self,
        f_ctx_accepted: torch.Tensor,
        pos_accepted: Union[torch.Tensor, List[int]],
        scatter_pos: Optional[Union[torch.Tensor, List[int]]] = None,
    ):
        """CPU-Only Lazy Draft KV Commit.
        Projects and writes draft KV only for the accepted tokens into draft_cache.
        On CUDA, this is a zero-op because WAL writes ahead during forward_verify_step.
        """
        if self.device.type == "cuda":
            return

        B = f_ctx_accepted.shape[0]
        N = f_ctx_accepted.shape[1]

        if isinstance(pos_accepted, list):
            # Backward-compatibility signature: (f_ctx_new, cur_pos_prev_list, k_list)
            cp = pos_accepted[0]
            k_val = scatter_pos[0] if isinstance(scatter_pos, list) else N
            pos_tensor = torch.arange(cp, cp + k_val, device=self.device).unsqueeze(0).expand(B, -1)
            scatter_tensor = (self.w + pos_tensor)
        else:
            pos_tensor = pos_accepted if isinstance(pos_accepted, torch.Tensor) else torch.tensor(pos_accepted, device=self.device)
            if pos_tensor.ndim == 1:
                pos_tensor = pos_tensor.unsqueeze(0).expand(B, -1)

            if scatter_pos is None:
                scatter_tensor = self.w + pos_tensor
            else:
                scatter_tensor = scatter_pos if isinstance(scatter_pos, torch.Tensor) else torch.tensor(scatter_pos, device=self.device)
                if scatter_tensor.ndim == 1:
                    scatter_tensor = scatter_tensor.unsqueeze(0).expand(B, -1)

        cos_d_k, sin_d_k = self.drafter.rotary_emb(f_ctx_accepted, pos_tensor)
        for d_l_idx in range(self.k_layers):
            if self.drafter.has_ghost_layers:
                ghost_l = self.drafter.ghost_layers[d_l_idx]
                k_d = ghost_l.self_attn.k_proj(f_ctx_accepted)
                v_d = ghost_l.self_attn.v_proj(f_ctx_accepted)
            else:
                base_l = self.drafter.draft_layers[d_l_idx]
                deltas = self.drafter.layer_deltas[d_l_idx]
                k_d = apply_linear_with_delta(base_l.self_attn.k_proj, f_ctx_accepted, deltas.k_proj)
                v_d = apply_linear_with_delta(base_l.self_attn.v_proj, f_ctx_accepted, deltas.v_proj)

            k_d = k_d.view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v_d = v_d.view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
            k_d = apply_rotary_pos_emb_k(k_d, cos_d_k, sin_d_k, unsqueeze_dim=1)

            for b in range(B):
                sp = scatter_tensor[b]
                self.draft_cache.k[d_l_idx][b, :, sp] = k_d[b]
                self.draft_cache.v[d_l_idx][b, :, sp] = v_d[b]


def build_topk_tree(
    draft_logits: torch.Tensor,       # (W, V_d) or (W, V) logits
    q_a: int = 16,
    draft_vocab_list: Optional[List[int]] = None,
    draft_vocab_ids: Optional[torch.Tensor] = None,
    k_per_step: int = 3,
) -> Tuple[List[int], List[int], List[int]]:
    """Build a static tree candidate tile of size Q_A from 1D/2D draft logits.

    Returns:
        tree_tokens: List of length Q_A (slot 0 is a placeholder 0 for concrete y*, filled at step)
        tree_parents: List of length Q_A (parent slot index in 0..Q_A-1, with parent[0] = -1)
        tree_depths: List of length Q_A (depth relative to t: slot 0 has depth 0, children have depth >= 1)
    """
    W = draft_logits.shape[0]
    log_probs = F.log_softmax(draft_logits.float(), dim=-1)

    # Pre-extract top-k candidates for each depth step 0 .. W-1
    # Each slot j corresponds to draft step j (depth j + 1 in tree relative to t)
    topk_vals, topk_inds = torch.topk(log_probs, k=min(k_per_step, log_probs.shape[-1]), dim=-1)
    topk_vals = topk_vals.tolist()
    topk_inds = topk_inds.tolist()

    # Remap local vocab indices to global token IDs
    topk_global = []
    for step in range(W):
        step_tokens = []
        for local_idx in topk_inds[step]:
            if draft_vocab_list is not None:
                step_tokens.append(draft_vocab_list[local_idx])
            elif draft_vocab_ids is not None:
                step_tokens.append(int(draft_vocab_ids[local_idx].item()))
            else:
                step_tokens.append(local_idx)
        topk_global.append(step_tokens)

    # Priority queue: (-cum_logprob, depth, parent_slot, token_id, step_idx, rank)
    heap = []
    # Seed queue with top-k candidates for depth 1 (from draft step 0)
    for rank in range(len(topk_global[0])):
        val = topk_vals[0][rank]
        tok = topk_global[0][rank]
        heapq.heappush(heap, (-val, 1, 0, tok, 0, rank))

    tree_tokens = [0]       # slot 0 placeholder for y*
    tree_parents = [-1]    # slot 0 has no parent
    tree_depths = [0]      # slot 0 is at depth 0 (pos t)

    while len(tree_tokens) < q_a and heap:
        neg_p, depth, p_slot, tok, step_idx, rank = heapq.heappop(heap)
        cur_slot = len(tree_tokens)
        tree_tokens.append(tok)
        tree_parents.append(p_slot)
        tree_depths.append(depth)

        # Expand child candidates at next draft step if within window W
        next_step = step_idx + 1
        if next_step < W:
            for next_rank in range(len(topk_global[next_step])):
                next_val = -neg_p + topk_vals[next_step][next_rank]
                next_tok = topk_global[next_step][next_rank]
                heapq.heappush(heap, (-next_val, depth + 1, cur_slot, next_tok, next_step, next_rank))

    # Pad if tree didn't reach q_a (fallback)
    while len(tree_tokens) < q_a:
        tree_tokens.append(0)
        tree_parents.append(0)
        tree_depths.append(1)

    return tree_tokens, tree_parents, tree_depths


def generate_speculative_v3(
    base_model: nn.Module,
    drafter: WeightTiedTopKDrafter,
    input_ids: torch.Tensor,
    mask_token_id: int = MASK_TOKEN_ID,
    max_new_tokens: int = 64,
    v_tokens: Optional[int] = None,
    temperature: float = 0.0,
    eos_token_id: Optional[int] = None,
    stop_token_ids: Optional[Union[int, Set[int], List[int]]] = None,
    use_cuda_graph: bool = True,
    engine: Optional[StaticDraftEngine] = None,
    hedge_margin: float = 1.0,
    step_callback: Optional[Callable[[dict], None]] = None,
    tree_mode: str = "auto",
    q_a: Optional[int] = None,
) -> Tuple[torch.Tensor, dict]:
    """BFG-Diff v4 Speculative Decoding:
    - 100% Lossless mathematical parity to greedy base_model autoregressive decoding.
    - Write-Ahead Log (WAL) draft_cache on CUDA; CPU-only lazy draft KV commit.
    - [0:W] scratch tail: zero torch.cat() in draft step.
    - Power-of-2 Bucketed CUDA Graphs: 256, 512, 1024, 2048.
    - Fused Ghost Weights: 28 fewer kernel launches per draft step.
    - (W-1)+1 Leaf-Hedge Mask Morphing: rescues near-tie slot-0 misses with zero graph recompilation.
    """
    B, P = input_ids.shape
    device = input_ids.device
    dtype = next(base_model.parameters()).dtype

    if v_tokens is None:
        v_tokens = 7 if device.type == "cuda" else 3

    if engine is None or getattr(engine, "max_len", 0) < (P + max_new_tokens + 32) or getattr(engine, "batch_size", 0) != B or getattr(engine, "v_tokens", None) != v_tokens or (q_a is not None and getattr(engine, "q_a", None) != q_a):
        engine = StaticDraftEngine(
            base_model=base_model,
            drafter=drafter,
            max_len=P + max_new_tokens + 64,
            batch_size=B,
            v_tokens=v_tokens,
            q_a=q_a,
            device=device,
            dtype=dtype,
            use_cuda_graph=use_cuda_graph,
            hedge_margin=hedge_margin,
            draft_vocab_ids=getattr(engine, "draft_vocab_ids", None),
        )

    v_tokens = engine.v_tokens

    stop_set = set()
    if eos_token_id is not None:
        stop_set.add(eos_token_id)
    if stop_token_ids is not None:
        if isinstance(stop_token_ids, int):
            stop_set.add(stop_token_ids)
        else:
            stop_set.update(stop_token_ids)

    def _truncate_at_stop(tokens: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Cut a committed block right after its first stop token (inclusive)."""
        if not stop_set:
            return tokens, False
        for i, tid in enumerate(tokens.tolist()):
            if tid in stop_set:
                return tokens[: i + 1], True
        return tokens, False

    # 1. Warm up & capture CUDA graphs BEFORE prefill if not yet captured
    if engine.device.type == "cuda" and engine.batch_size == 1 and engine.use_cuda_graph:
        engine.capture_cuda_graphs(mask_token_id)

    # 2. Prefill prompt (resets target_cache and draft_cache cleanly)
    logits_0, f_ctx_last = engine.prefill(input_ids)
    if temperature == 0.0:
        next_token = torch.argmax(logits_0[:, -1:], dim=-1)  # (B, 1) token at position P
    else:
        probs = F.softmax(logits_0[:, -1:] / max(1e-4, temperature), dim=-1)
        next_token = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(B, 1)

    generated_per_row = [[input_ids[b], next_token[b]] for b in range(B)]
    cur_pos_list = [P for _ in range(B)]
    done = [False] * B
    for b in range(B):
        if int(next_token[b, 0]) in stop_set:
            done[b] = True

    # Acceptance tracking
    total_draft_tokens = 0
    accepted_draft_tokens = 0
    steps = 0
    accept_counts = [0] * (v_tokens + 2)
    a_plus = [0] * min(4, v_tokens)
    hedge_attempts = 0
    hedge_hits = 0

    if step_callback is not None and not done[0]:
        step_callback({
            "step": 0,
            "committed_tokens": [int(next_token[0, 0])],
            "accepted": 0,
            "is_hedge": False,
            "hedge_hit": False,
            "q_a": engine.q_a,
            "total_generated": 1,
        })

    # Pre-allocated reusable buffers for hot loop (zero per-step heap allocations)
    engine_dtype = getattr(engine, "dtype", dtype)
    verify_input_buf = torch.zeros(B, engine.q_a, dtype=torch.long, device=engine.device)
    pos_override_buf = torch.zeros(B, engine.q_a, dtype=torch.long, device=engine.device)
    tree_mask_buf = torch.full((engine.q_a, engine.q_a), -1e4, dtype=engine_dtype, device=engine.device)
    tree_mask_cpu = torch.full((engine.q_a, engine.q_a), -1e4, dtype=engine_dtype, device="cpu")

    while not all(done) and any((cur_pos_list[b] - P + 1) < max_new_tokens for b in range(B)):
        y_star = next_token  # (B, 1) concrete bonus token at position t
        t = cur_pos_list[0]
        generated_before = cur_pos_list[0] - P + 1
        step_committed = []
        step_accepted = 0
        hedge_hit = False

        # Stage 2: Draft W tokens
        draft_ids, draft_logits = engine.forward_draft_step(
            y_star_ids=y_star,
            mask_token_id=mask_token_id,
            f_ctx_last=f_ctx_last,
            cur_pos_list=cur_pos_list,
            return_logits=True,
        )

        use_tree = False
        is_hedge = False
        d0_1, d0_2 = None, None

        if B == 1:
            if tree_mode == "tree" or (tree_mode == "auto" and engine.q_a > engine.w + 1):
                use_tree = True
            elif tree_mode != "linear" and hedge_margin > 0.0 and engine.q_a == engine.w + 1:
                # Leaf-Hedge Margin Check at Slot 0
                top2 = torch.topk(draft_logits[0, 0], k=2)
                top2_vals = top2.values.tolist()
                top2_inds = top2.indices.tolist()
                v1, v2 = top2_vals[0], top2_vals[1]
                raw_d0_1, raw_d0_2 = top2_inds[0], top2_inds[1]
                draft_v_list = getattr(engine, "_draft_vocab_list", None)
                if draft_v_list is not None:
                    d0_1 = draft_v_list[raw_d0_1]
                    d0_2 = draft_v_list[raw_d0_2]
                elif getattr(engine, "draft_vocab_ids", None) is not None:
                    d0_1 = int(engine.draft_vocab_ids[raw_d0_1].item())
                    d0_2 = int(engine.draft_vocab_ids[raw_d0_2].item())
                else:
                    d0_1 = raw_d0_1
                    d0_2 = raw_d0_2
                margin = v1 - v2
                if margin < hedge_margin:
                    is_hedge = True
                    hedge_attempts += 1

        # Stage 1: Verify batch Q_A at positions [t .. t + Q_A - 1]
        if use_tree:
            # Build top-k best-first tree topology of size Q_A
            t_tokens, t_parents, t_depths = build_topk_tree(
                draft_logits=draft_logits[0],
                q_a=engine.q_a,
                draft_vocab_list=getattr(engine, "_draft_vocab_list", None),
                draft_vocab_ids=getattr(engine, "draft_vocab_ids", None),
                k_per_step=3,
            )
            t_tokens[0] = int(y_star[0, 0].item())

            verify_input_buf[0].copy_(torch.tensor(t_tokens, dtype=torch.long, device=engine.device))
            pos_override_buf[0].copy_(torch.tensor([t + d for d in t_depths], dtype=torch.long, device=engine.device))

            # Build ancestor mask in tree_mask_cpu and transfer once to GPU (zero micro-kernel launches)
            tree_mask_cpu.fill_(-1e4)
            for i in range(engine.q_a):
                curr = i
                while curr != -1:
                    tree_mask_cpu[i, curr] = 0.0
                    curr = t_parents[curr]
            tree_mask_buf.copy_(tree_mask_cpu, non_blocking=True)

            verify_logits, f_ctx_verify = engine.forward_verify_step(
                verify_input_buf,
                cur_pos_list=cur_pos_list,
                is_hedge=False,
                pos_verify_override=pos_override_buf,
                tree_mask_override=tree_mask_buf,
            )
            verify_input = verify_input_buf
        elif is_hedge:
            # Polymorphic Micro-Tree Tile: [y*, d0_1, ..., d0_2]
            verify_input_buf[0, 0] = y_star[0, 0]
            verify_input_buf[0, 1] = d0_1
            if engine.w > 2:
                verify_input_buf[0, 2 : engine.w].copy_(draft_ids[0, 1 : engine.w - 1])
            verify_input_buf[0, engine.q_a - 1] = d0_2

            pos_override_buf[0, 0 : engine.w].copy_(torch.arange(t, t + engine.w, device=engine.device))
            pos_override_buf[0, engine.q_a - 1] = t + 1
            verify_logits, f_ctx_verify = engine.forward_verify_step(
                verify_input_buf, cur_pos_list=cur_pos_list, is_hedge=True, pos_verify_override=pos_override_buf
            )
            verify_input = verify_input_buf
        else:
            # Standard Linear Chain Tile: [y*, d0, d1, ...]
            verify_input_buf[:, 0:1].copy_(y_star)
            verify_input_buf[:, 1 : engine.w + 1].copy_(draft_ids)
            if engine.q_a > engine.w + 1:
                verify_input_buf[:, engine.w + 1 :].zero_()
            verify_logits, f_ctx_verify = engine.forward_verify_step(
                verify_input_buf, cur_pos_list=cur_pos_list, is_hedge=False
            )
            verify_input = verify_input_buf

        # Verification matching logic
        if temperature == 0.0:
            target_tokens = torch.argmax(verify_logits, dim=-1)
        else:
            probs = F.softmax(verify_logits / max(1e-4, temperature), dim=-1)
            target_tokens = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(B, engine.q_a)

        if use_tree:
            # Walk tree from root slot 0 along greedy verified path
            # Extract target predictions once to CPU to avoid per-node GPU-CPU syncs
            t_preds = target_tokens[0].tolist()
            accepted_slots = [0]
            curr_slot = 0
            while True:
                t_pred = t_preds[curr_slot]
                matched_child = None
                for c_slot in range(curr_slot + 1, engine.q_a):
                    if t_parents[c_slot] == curr_slot and t_tokens[c_slot] == t_pred:
                        matched_child = c_slot
                        break
                if matched_child is not None:
                    accepted_slots.append(matched_child)
                    curr_slot = matched_child
                else:
                    break

            k = len(accepted_slots) - 1  # number of accepted draft tokens
            for offset_idx in range(min(k, len(a_plus))):
                a_plus[offset_idx] += 1

            num_committed = k + 1
            accept_counts[num_committed] += 1
            total_draft_tokens += v_tokens
            accepted_draft_tokens += k

            # Track tree hedge rescues (non-linear branch accepted)
            hedge_attempts += 1
            if k > 0 and any(accepted_slots[m] != m for m in range(1, len(accepted_slots))):
                hedge_hits += 1
                hedge_hit = True

            # Final bonus token is predicted by last accepted slot
            last_slot = accepted_slots[-1]
            bonus_token = target_tokens[0, last_slot : last_slot + 1]

            # Committed tokens: candidate tokens along accepted path + bonus token
            # Slot 0 token is y* (already committed in previous step/prefill), so path tokens are slots 1..k
            path_draft_tokens = [t_tokens[s] for s in accepted_slots[1:]]
            if path_draft_tokens:
                committed_tokens = torch.cat([
                    torch.tensor(path_draft_tokens, device=engine.device, dtype=torch.long),
                    bonus_token,
                ], dim=0)
            else:
                committed_tokens = bonus_token

            step_accepted = k
            step_committed = [int(x) for x in committed_tokens]

            # Ascending KV compaction: compact accepted slot n_m into sequence position t + m
            # Topological ordering (parent[i] < i) guarantees n_m >= m, making ascending copy overwrite-safe!
            for m in range(1, len(accepted_slots)):
                src_slot = accepted_slots[m]
                dst_pos = t + m
                src_pos = t + src_slot
                if src_pos != dst_pos:
                    for l_idx in range(engine.total_layers):
                        engine.target_cache.k[l_idx][0, :, dst_pos] = engine.target_cache.k[l_idx][0, :, src_pos]
                        engine.target_cache.v[l_idx][0, :, dst_pos] = engine.target_cache.v[l_idx][0, :, src_pos]
                    if engine.device.type == "cuda":
                        for dl_idx in range(engine.k_layers):
                            engine.draft_cache.k[dl_idx][0, :, engine.w + dst_pos] = engine.draft_cache.k[dl_idx][0, :, engine.w + src_pos]
                            engine.draft_cache.v[dl_idx][0, :, engine.w + dst_pos] = engine.draft_cache.v[dl_idx][0, :, engine.w + src_pos]

            # CPU Lazy Draft KV Commit for accepted path tokens
            if engine.device.type != "cuda":
                pos_acc = torch.arange(t, t + k + 1, device=engine.device)
                f_ctx_path = torch.stack([f_ctx_verify[0, s] for s in accepted_slots], dim=0).unsqueeze(0)
                engine.commit_accepted_tokens(
                    f_ctx_path,
                    pos_acc,
                    engine.w + pos_acc,
                )

            committed_tokens, hit_stop = _truncate_at_stop(committed_tokens)
            generated_per_row[0].append(committed_tokens)

            next_token = bonus_token.unsqueeze(0)
            f_ctx_last = f_ctx_verify[0:1, last_slot : last_slot + 1]
            cur_pos_list[0] += num_committed

            if hit_stop:
                done[0] = True
            if (cur_pos_list[0] - P + 1) >= max_new_tokens:
                done[0] = True

        elif is_hedge:
            # Leaf-Hedge Verification Branching
            t0 = target_tokens[0, 0].item()
            total_draft_tokens += v_tokens

            if t0 == d0_1:
                # Case A: Primary branch hits! Verify sequential draft slots
                k = 1
                for j in range(1, engine.w - 1):
                    if draft_ids[0, j].item() == target_tokens[0, j].item():
                        k += 1
                    else:
                        break

                for offset_idx in range(min(k, len(a_plus))):
                    a_plus[offset_idx] += 1

                num_committed = k + 1
                accept_counts[num_committed] += 1
                accepted_draft_tokens += k

                committed_tokens = torch.cat([verify_input[0, 1 : k + 1], target_tokens[0, k : k + 1]], dim=0)
                step_accepted = k
                step_committed = [int(x) for x in committed_tokens]

                # CPU Lazy Draft KV Commit
                if engine.device.type != "cuda":
                    pos_acc = torch.arange(t, t + k + 1, device=engine.device)
                    engine.commit_accepted_tokens(
                        f_ctx_verify[0:1, 0 : k + 1],
                        pos_acc,
                        engine.w + pos_acc,
                    )

                committed_tokens, hit_stop = _truncate_at_stop(committed_tokens)
                generated_per_row[0].append(committed_tokens)

                next_token = target_tokens[0, k : k + 1].unsqueeze(0)
                f_ctx_last = f_ctx_verify[0:1, k : k + 1]
                cur_pos_list[0] += num_committed

                if hit_stop:
                    done[0] = True
                if (cur_pos_list[0] - P + 1) >= max_new_tokens:
                    done[0] = True

            elif t0 == d0_2:
                # Case B: Hedge branch hits! Accept d0_2 + verified bonus token from hedge slot
                hedge_hits += 1
                t_bonus = target_tokens[0, (engine.q_a - 1) : engine.q_a]
                committed_tokens = torch.cat([torch.tensor([d0_2], device=engine.device), t_bonus], dim=0)
                step_accepted = 1
                hedge_hit = True
                step_committed = [int(x) for x in committed_tokens]

                # Commit hedge slot KV to slot 1 (position t + 1) in target_cache
                for l_idx in range(engine.total_layers):
                    engine.target_cache.k[l_idx][0, :, t + 1] = engine.target_cache.k[l_idx][0, :, t + (engine.q_a - 1)]
                    engine.target_cache.v[l_idx][0, :, t + 1] = engine.target_cache.v[l_idx][0, :, t + (engine.q_a - 1)]

                if engine.device.type == "cuda":
                    for dl_idx in range(engine.k_layers):
                        engine.draft_cache.k[dl_idx][0, :, engine.w + t + 1] = engine.draft_cache.k[dl_idx][0, :, engine.w + t + (engine.q_a - 1)]
                        engine.draft_cache.v[dl_idx][0, :, engine.w + t + 1] = engine.draft_cache.v[dl_idx][0, :, engine.w + t + (engine.q_a - 1)]
                else:
                    # CPU Lazy Draft KV Commit for [slot 0, slot hedge] -> [t, t+1]
                    f_ctx_hedge = torch.cat([f_ctx_verify[0:1, 0:1], f_ctx_verify[0:1, (engine.q_a - 1) : engine.q_a]], dim=1)
                    pos_hedge = torch.tensor([t, t + 1], device=engine.device)
                    engine.commit_accepted_tokens(
                        f_ctx_hedge,
                        pos_hedge,
                        engine.w + pos_hedge,
                    )

                committed_tokens, hit_stop = _truncate_at_stop(committed_tokens)
                generated_per_row[0].append(committed_tokens)

                cur_pos_list[0] += 2
                next_token = t_bonus.unsqueeze(0)
                f_ctx_last = f_ctx_verify[0:1, (engine.q_a - 1) : engine.q_a]
                accepted_draft_tokens += 1
                a_plus[0] += 1
                accept_counts[2] += 1

                if hit_stop:
                    done[0] = True
                if (cur_pos_list[0] - P + 1) >= max_new_tokens:
                    done[0] = True

            else:
                # Case C: Neither candidate matches slot 0. Emit t0 as bonus token
                committed_tokens = target_tokens[0, 0:1]
                step_accepted = 0
                step_committed = [int(x) for x in committed_tokens]

                # CPU Lazy Draft KV Commit for slot 0 at t
                if engine.device.type != "cuda":
                    pos_acc = torch.tensor([t], device=engine.device)
                    engine.commit_accepted_tokens(
                        f_ctx_verify[0:1, 0:1],
                        pos_acc,
                        engine.w + pos_acc,
                    )

                committed_tokens, hit_stop = _truncate_at_stop(committed_tokens)
                generated_per_row[0].append(committed_tokens)
                cur_pos_list[0] += 1
                next_token = committed_tokens.unsqueeze(0)
                f_ctx_last = f_ctx_verify[0:1, 0:1]
                accept_counts[1] += 1

                if hit_stop:
                    done[0] = True
                if (cur_pos_list[0] - P + 1) >= max_new_tokens:
                    done[0] = True

        else:
            # Standard Linear Chain Verification
            next_token_list = []
            f_ctx_last_list = []

            for b in range(B):
                if done[b] or (cur_pos_list[b] - P + 1) >= max_new_tokens:
                    next_token_list.append(y_star[b : b + 1])
                    f_ctx_last_list.append(f_ctx_last[b : b + 1])
                    continue

                k = 0
                for j in range(v_tokens):
                    if draft_ids[b, j] == target_tokens[b, j]:
                        k += 1
                    else:
                        break

                for offset_idx in range(min(k, len(a_plus))):
                    a_plus[offset_idx] += 1

                num_committed = k + 1
                accept_counts[num_committed] += 1
                total_draft_tokens += v_tokens
                accepted_draft_tokens += k

                if k > 0:
                    committed_tokens = torch.cat([draft_ids[b, :k], target_tokens[b, k : k + 1]], dim=0)
                else:
                    committed_tokens = target_tokens[b, 0:1]

                if b == 0:
                    step_accepted = k
                    step_committed = [int(x) for x in committed_tokens]

                cp = cur_pos_list[b]
                # CPU Lazy Draft KV Commit for accepted tokens [cp .. cp + k]
                if engine.device.type != "cuda":
                    pos_acc = torch.arange(cp, cp + k + 1, device=engine.device)
                    engine.commit_accepted_tokens(
                        f_ctx_verify[b : b + 1, 0 : k + 1],
                        pos_acc,
                        engine.w + pos_acc,
                    )

                committed_tokens, hit_stop = _truncate_at_stop(committed_tokens)
                generated_per_row[b].append(committed_tokens)

                b_token = target_tokens[b, k : k + 1].unsqueeze(0)
                next_token_list.append(b_token)
                f_ctx_last_list.append(f_ctx_verify[b : b + 1, k : k + 1])
                cur_pos_list[b] += num_committed

                if hit_stop:
                    done[b] = True
                if (cur_pos_list[b] - P + 1) >= max_new_tokens:
                    done[b] = True

            next_token = torch.cat(next_token_list, dim=0)
            f_ctx_last = torch.cat(f_ctx_last_list, dim=0)

        steps += 1
        if step_callback is not None:
            cb_tokens = []
            for tid in step_committed:
                if tid in stop_set:
                    break
                cb_tokens.append(tid)
            cb_tokens = cb_tokens[: max(0, max_new_tokens - generated_before)]
            step_callback({
                "step": steps,
                "committed_tokens": cb_tokens,
                "accepted": step_accepted,
                "is_hedge": is_hedge,
                "hedge_hit": hedge_hit,
                "q_a": engine.q_a,
                "total_generated": generated_before + len(cb_tokens),
            })

    clipped = [torch.cat(generated_per_row[b], dim=-1)[: P + max_new_tokens] for b in range(B)]
    max_row_len = max(int(r.shape[0]) for r in clipped)
    pad_id = eos_token_id if eos_token_id is not None else 0
    final_outputs = []
    for r in clipped:
        n_pad = max_row_len - int(r.shape[0])
        if n_pad > 0:
            filler = torch.full((n_pad,), pad_id, dtype=r.dtype, device=r.device)
            r = torch.cat([r, filler], dim=0)
        final_outputs.append(r)
    output_ids = torch.stack(final_outputs, dim=0)

    n_queries = max(1, steps * B)
    stats = {
        "steps": steps,
        "lengths": [int(r.shape[0]) for r in clipped],
        "total_draft_tokens": total_draft_tokens,
        "accepted_draft_tokens": accepted_draft_tokens,
        "alpha": (accepted_draft_tokens + steps * B) / n_queries,
        "accept_distribution": accept_counts,
        "a_plus_1": a_plus[0] / n_queries if len(a_plus) > 0 else 0.0,
        "a_plus_2": a_plus[1] / n_queries if len(a_plus) > 1 else 0.0,
        "a_plus_3": a_plus[2] / n_queries if len(a_plus) > 2 else 0.0,
        "a_plus_4": a_plus[3] / n_queries if len(a_plus) > 3 else 0.0,
        "hedge_attempts": hedge_attempts,
        "hedge_hits": hedge_hits,
        "hedge_hit_rate": (hedge_hits / max(1, hedge_attempts)) if hedge_attempts > 0 else 0.0,
    }
    return output_ids, stats
