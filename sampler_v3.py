"""BFG-Diff v3: Static-Shape Speculative Inference Engine (sampler_v3.py).

Core Features:
1. Zero-crop(), Pointer-Indexed StaticCache:
   - Target cache across all L layers: (B, n_kv_heads, MAX_LEN, head_dim)
   - Draft cache across top 4 layers: (B, n_kv_heads, MAX_LEN, head_dim)
   - In-place pointer indexing overwrites rejected tail in-place.
2. Fixed Active Shapes:
   - Stage 1 (verify): fixed Q_A = 8 ([y*, d_0, ..., d_6], V=7).
   - Stage 2 (draft): fixed W = 7 draft tokens.
3. Mathematical Train-Inference Invariant:
   - When drafting from concrete bonus token y* at sequence position t (y* = x_t):
     - y* has RoPE position t, and draft tokens have RoPE positions [t, t+1, ..., t+W-1],
       predicting [x_{t+1}, x_{t+2}, ..., x_{t+W}].
     - f_ctx_last MUST be f_ctx[t-1] (feature of token immediately preceding y*).
     - Draft prefix KV cache contains strictly positions 0 .. t-1 (length t).
   - Verified length t starts at P (prompt length) after prefill.
4. Support for both CUDA-Graph captured execution (on CUDA B=1) and eager static fallback (CPU/B=4).
5. Batched ragged B=1 and B=4 support.
"""

import math
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_v3 import WeightTiedTopKDrafter, apply_linear_with_delta
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
    """Encapsulates static-shape Stage 1 verification and Stage 2 drafting with CUDA Graph capture."""

    def __init__(
        self,
        base_model: nn.Module,
        drafter: WeightTiedTopKDrafter,
        max_len: int = 2048,
        batch_size: int = 1,
        v_tokens: int = 7,  # Q_A = 8
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_cuda_graph: bool = True,
    ):
        self.base_model = base_model
        self.drafter = drafter
        self.max_len = max_len
        self.batch_size = batch_size
        self.v_tokens = v_tokens
        self.q_a = v_tokens + 1  # 8
        self.w = v_tokens        # 7
        self.use_cuda_graph = use_cuda_graph

        if device is None:
            device = next(base_model.parameters()).device
        if dtype is None:
            dtype = next(base_model.parameters()).dtype
        self.device = torch.device(device)
        self.dtype = dtype

        # In inference mode on CUDA (or when dtype is fp16), ensure drafter is cast to self.dtype
        # so apply_linear_with_delta runs purely in fp16 with zero fp32 casts
        if self.device.type == "cuda" or self.dtype == torch.float16:
            self.drafter.to(dtype=self.dtype)

        config = base_model.config
        self.total_layers = config.num_hidden_layers
        self.k_layers = drafter.k_layers
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)
        self.dim = config.hidden_size

        # Pre-allocate static caches
        self.target_cache = StaticKVCache(
            num_layers=self.total_layers,
            batch_size=batch_size,
            num_kv_heads=self.num_kv_heads,
            max_len=max_len,
            head_dim=self.head_dim,
            dtype=dtype,
            device=device,
        )
        self.draft_cache = StaticKVCache(
            num_layers=self.k_layers,
            batch_size=batch_size,
            num_kv_heads=self.num_kv_heads,
            max_len=max_len,
            head_dim=self.head_dim,
            dtype=dtype,
            device=device,
        )

        # Pre-allocate static input buffers
        self.static_verify_input_ids = torch.zeros(batch_size, self.q_a, dtype=torch.long, device=device)
        self.static_pos_verify = torch.zeros(batch_size, self.q_a, dtype=torch.long, device=device)
        self.static_y_star = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        self.static_f_ctx_last = torch.zeros(batch_size, 1, self.dim, dtype=dtype, device=device)
        self.static_pos_draft = torch.zeros(batch_size, self.w, dtype=torch.long, device=device)
        self.static_mask_token = torch.tensor([[151665]], dtype=torch.long, device=device)

        # Pre-allocate static attention masks
        # static_mask_A: (B, 1, Q_A, MAX_LEN)
        self.static_mask_A = torch.full((batch_size, 1, self.q_a, max_len), -1e4, dtype=dtype, device=device)
        # static_mask_B: (B, 1, W, MAX_LEN + W)
        self.static_mask_B = torch.full((batch_size, 1, self.w, max_len + self.w), -1e4, dtype=dtype, device=device)

        # Pre-allocate static output buffers
        vocab_size = config.vocab_size
        self.static_verify_logits = torch.zeros(batch_size, self.q_a, vocab_size, dtype=dtype, device=device)
        self.static_f_ctx_verify = torch.zeros(batch_size, self.q_a, self.dim, dtype=dtype, device=device)
        self.static_draft_logits = torch.zeros(batch_size, self.w, vocab_size, dtype=dtype, device=device)
        self.static_draft_ids = torch.zeros(batch_size, self.w, dtype=torch.long, device=device)

        self.cuda_graph_verify = None
        self.cuda_graph_draft = None

    def capture_cuda_graphs(self, mask_token_id: int):
        """Warmup and capture CUDA graphs on CUDA for B=1."""
        if self.device.type != "cuda" or self.batch_size != 1 or not self.use_cuda_graph:
            return
        if not hasattr(torch.cuda, "CUDAGraph"):
            return

        self.static_mask_token = torch.tensor([[mask_token_id]], dtype=torch.long, device=self.device)
        self.drafter.to(dtype=self.dtype)

        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._step_verify_kernel()
                    self._step_draft_kernel()
            torch.cuda.current_stream().wait_stream(s)

            # Capture Stage 1 verify graph
            self.cuda_graph_verify = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.cuda_graph_verify):
                self._step_verify_kernel()

            # Capture Stage 2 draft graph
            self.cuda_graph_draft = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.cuda_graph_draft):
                self._step_draft_kernel()
        except Exception as e:
            # Fall back to eager static execution if CUDA graph capture fails on host
            self.cuda_graph_verify = None
            self.cuda_graph_draft = None

    def _step_verify_kernel(self):
        """Execution kernel for Stage 1 verify with fixed static shapes."""
        h = self.base_model.model.embed_tokens(self.static_verify_input_ids)
        cos, sin = self.base_model.model.rotary_emb(h, self.static_pos_verify)

        all_hidden_states = [h]
        for l_idx, layer in enumerate(self.base_model.model.layers):
            norm_h = layer.input_layernorm(h)
            q = layer.self_attn.q_proj(norm_h).view(1, self.q_a, self.num_heads, self.head_dim).transpose(1, 2)
            k = layer.self_attn.k_proj(norm_h).view(1, self.q_a, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = layer.self_attn.v_proj(norm_h).view(1, self.q_a, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

            idx = self.static_pos_verify.view(1, 1, self.q_a, 1).expand(1, self.num_kv_heads, self.q_a, self.head_dim)
            self.target_cache.k[l_idx].scatter_(2, idx, k)
            self.target_cache.v[l_idx].scatter_(2, idx, v)

            if self.num_heads != self.num_kv_heads:
                k_attn = repeat_kv(self.target_cache.k[l_idx], self.num_heads // self.num_kv_heads)
                v_attn = repeat_kv(self.target_cache.v[l_idx], self.num_heads // self.num_kv_heads)
            else:
                k_attn, v_attn = self.target_cache.k[l_idx], self.target_cache.v[l_idx]

            attn_out = F.scaled_dot_product_attention(q, k_attn, v_attn, attn_mask=self.static_mask_A)
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

    def _step_draft_kernel(self):
        """Execution kernel for Stage 2 draft with fixed static shapes."""
        y_star_emb = self.drafter.embed_tokens(self.static_y_star)
        mask_emb = self.drafter.embed_tokens(self.static_mask_token)
        h = self.drafter.bridge(y_star_emb, mask_emb, self.static_f_ctx_last, w=self.w)

        for layer_idx in range(self.k_layers):
            base_layer = self.drafter.draft_layers[layer_idx]
            deltas = self.drafter.layer_deltas[layer_idx]

            norm_h = base_layer.input_layernorm(h)
            q = apply_linear_with_delta(base_layer.self_attn.q_proj, norm_h, deltas.q_proj)
            q = q.view(1, self.w, self.num_heads, self.head_dim).transpose(1, 2)
            k_d = apply_linear_with_delta(base_layer.self_attn.k_proj, norm_h, deltas.k_proj)
            k_d = k_d.view(1, self.w, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v_d = apply_linear_with_delta(base_layer.self_attn.v_proj, norm_h, deltas.v_proj)
            v_d = v_d.view(1, self.w, self.num_kv_heads, self.head_dim).transpose(1, 2)

            cos_d, sin_d = self.drafter.rotary_emb(norm_h, self.static_pos_draft)
            q, k_d = apply_rotary_pos_emb(q, k_d, cos_d, sin_d, unsqueeze_dim=1)

            k_total = torch.cat([self.draft_cache.k[layer_idx], k_d], dim=2)
            v_total = torch.cat([self.draft_cache.v[layer_idx], v_d], dim=2)

            if self.num_heads != self.num_kv_heads:
                k_attn = repeat_kv(k_total, self.num_heads // self.num_kv_heads)
                v_attn = repeat_kv(v_total, self.num_heads // self.num_kv_heads)
            else:
                k_attn, v_attn = k_total, v_total

            attn_out = F.scaled_dot_product_attention(q, k_attn, v_attn, attn_mask=self.static_mask_B)
            attn_out = attn_out.transpose(1, 2).reshape(1, self.w, self.dim)
            h = h + apply_linear_with_delta(base_layer.self_attn.o_proj, attn_out, deltas.o_proj)

            norm_post = base_layer.post_attention_layernorm(h)
            gate = apply_linear_with_delta(base_layer.mlp.gate_proj, norm_post, deltas.gate_proj)
            up = apply_linear_with_delta(base_layer.mlp.up_proj, norm_post, deltas.up_proj)
            mlp_act = base_layer.mlp.act_fn(gate) * up
            h = h + apply_linear_with_delta(base_layer.mlp.down_proj, mlp_act, deltas.down_proj)

        h_norm = self.drafter.norm(h)
        logits = self.drafter.lm_head(h_norm)
        self.static_draft_logits.copy_(logits)
        self.static_draft_ids.copy_(torch.argmax(logits, dim=-1))

    def prefill(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stage 1 prefill on prompt tokens (length P).
        Writes target KV into target_cache at 0 .. P-1.
        Computes F_ctx (0 .. P-1) and writes prefix KV into draft_cache at 0 .. P-1.
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
                n_rep = self.num_heads // self.num_kv_heads
                k_attn = repeat_kv(k, n_rep)
                v_attn = repeat_kv(v, n_rep)
            else:
                k_attn, v_attn = k, v

            attn_out = F.scaled_dot_product_attention(q, k_attn, v_attn, attn_mask=attn_mask)
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

        # Project and write prefix KV into draft_cache at 0 .. P-1
        k_list, v_list = self.drafter.project_prefix_kv(f_ctx, pos_ids)
        for d_l_idx in range(self.k_layers):
            self.draft_cache.update(d_l_idx, k_list[d_l_idx], v_list[d_l_idx], torch.arange(P, device=self.device))

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
                k_attn = repeat_kv(k_active, self.num_heads // self.num_kv_heads)
                v_attn = repeat_kv(v_active, self.num_heads // self.num_kv_heads)
            else:
                k_attn, v_attn = k_active, v_active

            attn_out = F.scaled_dot_product_attention(q, k_attn, v_attn, attn_mask=mask)
            attn_out = attn_out.transpose(1, 2).reshape(B, 1, self.dim)
            h = h + layer.self_attn.o_proj(attn_out)
            norm_post = layer.post_attention_layernorm(h)
            h = h + layer.mlp(norm_post)

        logits = self.base_model.lm_head(self.base_model.model.norm(h))
        return logits

    def forward_verify_step(
        self,
        verify_tokens: torch.Tensor,       # (B, Q_A=8) [y*, d_0, ..., d_6]
        cur_pos_list: List[int],           # list of B ints: sequence position t
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stage 1 verify pass over Q_A = 8 tokens at positions [t .. t+7].
        Writes target KV into target_cache at [t .. t+7].
        Uses CUDA Graph replay on CUDA when B=1, otherwise runs eager static step.
        """
        B, Q_A = verify_tokens.shape
        t = cur_pos_list[0]

        # CUDA Graph replay path for B=1
        if self.cuda_graph_verify is not None and B == 1:
            self.static_verify_input_ids.copy_(verify_tokens)
            self.static_pos_verify.copy_(torch.arange(t, t + Q_A, device=self.device).unsqueeze(0))
            self.static_mask_A.fill_(-1e4)
            if t > 0:
                self.static_mask_A[0, 0, :, :t] = 0.0
            causal_block = torch.triu(torch.full((Q_A, Q_A), -1e4, dtype=self.dtype, device=self.device), diagonal=1)
            self.static_mask_A[0, 0, :, t : t + Q_A] = causal_block

            self.cuda_graph_verify.replay()
            return self.static_verify_logits, self.static_f_ctx_verify

        # Eager static execution path (CPU or B > 1)
        max_total_len = max(cur_pos_list) + Q_A
        pos_ids = torch.zeros(B, Q_A, dtype=torch.long, device=self.device)
        mask = torch.full((B, 1, Q_A, max_total_len), -1e4, dtype=self.dtype, device=self.device)

        for b in range(B):
            cp = cur_pos_list[b]
            pos_ids[b] = torch.arange(cp, cp + Q_A, device=self.device)
            if cp > 0:
                mask[b, 0, :, :cp] = 0.0
            causal_block = torch.triu(torch.full((Q_A, Q_A), -1e4, dtype=self.dtype, device=self.device), diagonal=1)
            mask[b, 0, :, cp : cp + Q_A] = causal_block

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
                cache_positions = pos_ids[b]
                self.target_cache.k[l_idx][b, :, cache_positions] = k[b]
                self.target_cache.v[l_idx][b, :, cache_positions] = v[b]

            k_active = self.target_cache.k[l_idx][:, :, :max_total_len]
            v_active = self.target_cache.v[l_idx][:, :, :max_total_len]

            if self.num_heads != self.num_kv_heads:
                n_rep = self.num_heads // self.num_kv_heads
                k_attn = repeat_kv(k_active, n_rep)
                v_attn = repeat_kv(v_active, n_rep)
            else:
                k_attn, v_attn = k_active, v_active

            attn_out = F.scaled_dot_product_attention(q, k_attn, v_attn, attn_mask=mask)
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
        return logits, f_ctx

    def forward_draft_step(
        self,
        y_star_ids: torch.Tensor,    # (B, 1) concrete bonus token at position t
        mask_token_id: int,          # 151665
        f_ctx_last: torch.Tensor,    # (B, 1, D) F_ctx of token at t - 1
        cur_pos_list: List[int],     # list of B ints: sequence position t
        return_logits: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Stage 2 draft pass: produces W = 7 draft tokens predicting [t+1 .. t+7].
        Prefix KV cache sliced to length t (positions 0 .. t-1, zero garbage slots).
        Uses CUDA Graph replay on CUDA when B=1, otherwise runs eager static step.
        """
        B = y_star_ids.shape[0]
        W = self.w
        t = cur_pos_list[0]

        # CUDA Graph replay path for B=1
        if self.cuda_graph_draft is not None and B == 1:
            self.static_y_star.copy_(y_star_ids)
            self.static_f_ctx_last.copy_(f_ctx_last)
            self.static_pos_draft.copy_(torch.arange(t, t + W, device=self.device).unsqueeze(0))
            self.static_mask_B.fill_(-1e4)
            if t > 0:
                self.static_mask_B[0, 0, :, :t] = 0.0
            self.static_mask_B[0, 0, :, self.max_len : self.max_len + W] = 0.0

            self.cuda_graph_draft.replay()
            if return_logits:
                return self.static_draft_ids, self.static_draft_logits
            return self.static_draft_ids

        # Eager static execution path (CPU or B > 1)
        max_prefix_len = max(cur_pos_list)
        kp_list = [self.draft_cache.k[i][:, :, :max_prefix_len] for i in range(self.k_layers)]
        vp_list = [self.draft_cache.v[i][:, :, :max_prefix_len] for i in range(self.k_layers)]

        attn_mask = torch.full((B, 1, W, max_prefix_len + W), -1e4, dtype=self.dtype, device=self.device)
        pos_draft = torch.zeros(B, W, dtype=torch.long, device=self.device)
        for b in range(B):
            cp = cur_pos_list[b]
            if cp > 0:
                attn_mask[b, 0, :, :cp] = 0.0
            attn_mask[b, 0, :, max_prefix_len : max_prefix_len + W] = 0.0
            pos_draft[b] = torch.arange(cp, cp + W, device=self.device)

        y_star_emb = self.drafter.embed_tokens(y_star_ids)
        mask_tensor = torch.tensor([[mask_token_id]], dtype=torch.long, device=self.device)
        mask_emb = self.drafter.embed_tokens(mask_tensor)

        h = self.drafter.bridge(y_star_emb, mask_emb, f_ctx_last, w=W)

        for layer_idx in range(self.k_layers):
            h, _, _ = self.drafter.forward_draft_layer(
                layer_idx=layer_idx,
                hidden_states=h,
                pos_draft=pos_draft,
                k_prefix=kp_list[layer_idx],
                v_prefix=vp_list[layer_idx],
                attn_mask=attn_mask,
            )

        h_norm = self.drafter.norm(h)
        logits = self.drafter.lm_head(h_norm)
        draft_ids = torch.argmax(logits, dim=-1)

        if return_logits:
            return draft_ids, logits
        return draft_ids

    def commit_accepted_tokens(
        self,
        f_ctx_new: torch.Tensor,      # (B, Q_A, D) newly committed F_ctx vectors from verify
        cur_pos_prev_list: List[int], # start position t per row
        k_list: List[int],            # number of newly committed tokens (num_committed = k + 1)
    ):
        """Append num_committed F_ctx vectors into draft_cache at positions [t .. t + k]."""
        B = f_ctx_new.shape[0]
        for b in range(B):
            num_committed = k_list[b]
            if num_committed <= 0:
                continue
            start_t = cur_pos_prev_list[b]
            pos_ids = torch.arange(start_t, start_t + num_committed, device=self.device).unsqueeze(0)
            cache_positions = torch.arange(start_t, start_t + num_committed, device=self.device)
            f_slice = f_ctx_new[b : b + 1, :num_committed]

            k_sub, v_sub = self.drafter.project_prefix_kv(f_slice, pos_ids)
            for d_l_idx in range(self.k_layers):
                self.draft_cache.k[d_l_idx][b, :, cache_positions] = k_sub[d_l_idx][0]
                self.draft_cache.v[d_l_idx][b, :, cache_positions] = v_sub[d_l_idx][0]


def generate_speculative_v3(
    base_model: nn.Module,
    drafter: WeightTiedTopKDrafter,
    input_ids: torch.Tensor,
    mask_token_id: int = 151665,
    max_new_tokens: int = 64,
    v_tokens: int = 7,  # Q_A = 8
    temperature: float = 0.0,
    eos_token_id: Optional[int] = None,
    use_cuda_graph: bool = True,
) -> Tuple[torch.Tensor, dict]:
    """100% Lossless Speculative Sampling with BFG-Diff v3 Drafter.
    Guarantees exact mathematical match to base_model greedy autoregressive decoding at T=0.
    """
    B, P = input_ids.shape
    device = input_ids.device
    dtype = next(base_model.parameters()).dtype

    engine = StaticDraftEngine(
        base_model=base_model,
        drafter=drafter,
        max_len=P + max_new_tokens + 32,
        batch_size=B,
        v_tokens=v_tokens,
        device=device,
        dtype=dtype,
        use_cuda_graph=use_cuda_graph,
    )

    # 1. Prefill prompt
    logits_0, f_ctx_last = engine.prefill(input_ids)
    if temperature == 0.0:
        next_token = torch.argmax(logits_0[:, -1:], dim=-1)  # (B, 1) token at position P
    else:
        probs = F.softmax(logits_0[:, -1:] / temperature, dim=-1)
        next_token = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(B, 1)

    # Capture CUDA graphs on CUDA for B=1 after prefill
    if engine.device.type == "cuda" and engine.batch_size == 1 and engine.use_cuda_graph:
        engine.capture_cuda_graphs(mask_token_id)

    generated_per_row = [[input_ids[b], next_token[b]] for b in range(B)]
    # verified_len tracks current sequence position t (starts at P)
    cur_pos_list = [P for _ in range(B)]
    done = [False] * B

    # Acceptance tracking
    total_draft_tokens = 0
    accepted_draft_tokens = 0
    steps = 0
    accept_counts = [0] * (v_tokens + 2)
    a_plus = [0] * min(4, v_tokens)

    while not all(done) and any((cur_pos_list[b] - P) < max_new_tokens for b in range(B)):
        y_star = next_token  # (B, 1) concrete bonus token at position t

        # 2. Stage 2: Draft V tokens (W = 7)
        draft_ids = engine.forward_draft_step(
            y_star_ids=y_star,
            mask_token_id=mask_token_id,
            f_ctx_last=f_ctx_last,
            cur_pos_list=cur_pos_list,
        )  # (B, V=7)

        # 3. Stage 1: Verify batch Q_A = 8: [y*, d_0, ..., d_6] at positions [t .. t+7]
        verify_input = torch.cat([y_star, draft_ids], dim=-1)  # (B, 8)
        verify_logits, f_ctx_verify = engine.forward_verify_step(verify_input, cur_pos_list=cur_pos_list)

        # 4. Verify match loop
        target_tokens = torch.argmax(verify_logits, dim=-1)  # (B, 8)

        k_list = []
        cur_pos_prev_list = list(cur_pos_list)
        next_token_list = []
        f_ctx_last_list = []

        for b in range(B):
            if done[b] or (cur_pos_list[b] - P) >= max_new_tokens:
                k_list.append(0)
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

            num_committed = k + 1  # tokens at positions [t .. t+k]
            accept_counts[num_committed] += 1
            total_draft_tokens += v_tokens
            accepted_draft_tokens += k

            if k > 0:
                committed_tokens = torch.cat([draft_ids[b, :k], target_tokens[b, k : k + 1]], dim=0)
            else:
                committed_tokens = target_tokens[b, 0:1]

            generated_per_row[b].append(committed_tokens)

            # Next bonus token at position t + num_committed
            b_token = target_tokens[b, k : k + 1].unsqueeze(0)  # (1, 1)
            next_token_list.append(b_token)

            # Feature of token at position t + k (which is t_new - 1)
            f_ctx_last_list.append(f_ctx_verify[b : b + 1, k : k + 1])

            k_list.append(num_committed)
            cur_pos_list[b] += num_committed

            if eos_token_id is not None and (committed_tokens == eos_token_id).any():
                done[b] = True
            if (cur_pos_list[b] - P) >= max_new_tokens:
                done[b] = True

        next_token = torch.cat(next_token_list, dim=0)
        f_ctx_last = torch.cat(f_ctx_last_list, dim=0)

        # Commit accepted vectors into draft_cache at positions [t .. t+k]
        engine.commit_accepted_tokens(
            f_ctx_new=f_ctx_verify,
            cur_pos_prev_list=cur_pos_prev_list,
            k_list=k_list,
        )
        steps += 1

    final_outputs = []
    for b in range(B):
        row_tokens = torch.cat(generated_per_row[b], dim=-1)[: P + max_new_tokens]
        final_outputs.append(row_tokens)
    output_ids = torch.stack(final_outputs, dim=0)

    n_queries = max(1, steps * B)
    stats = {
        "steps": steps,
        "total_draft_tokens": total_draft_tokens,
        "accepted_draft_tokens": accepted_draft_tokens,
        "alpha": (accepted_draft_tokens + steps * B) / n_queries,
        "accept_distribution": accept_counts,
        "a_plus_1": a_plus[0] / n_queries if len(a_plus) > 0 else 0.0,
        "a_plus_2": a_plus[1] / n_queries if len(a_plus) > 1 else 0.0,
        "a_plus_3": a_plus[2] / n_queries if len(a_plus) > 2 else 0.0,
        "a_plus_4": a_plus[3] / n_queries if len(a_plus) > 3 else 0.0,
    }
    return output_ids, stats
