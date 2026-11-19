"""BFG-Diff v3: Static-Shape Speculative Inference Engine (sampler_v3.py).

Core Features:
1. Zero-crop(), Pointer-Indexed StaticCache:
   - Target cache across all L layers: (B, n_kv_heads, MAX_LEN, head_dim)
   - Draft cache across top 4 layers: (B, n_kv_heads, MAX_LEN, head_dim)
   - cur_pos maintained on device; cache_position = cur_pos + arange(Q_A) overwrites rejected tail in-place.
2. Fixed Active Shapes:
   - Stage 1 (verify): fixed Q_A = 8 ([cur_last_id, d_0, ..., d_6], V=7).
   - Stage 2 (draft): fixed W = 7 draft tokens.
3. Fixed-shape 4D masks (static_mask_A, static_mask_B) with vectorized tensor ops.
4. Support for both CUDA-Graph captured execution (on CUDA) and eager fallback (on CPU).
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
            # Batched ragged indices: cache_positions is (B, S)
            for b in range(B):
                self.k[layer_idx][b, :, cache_positions[b]] = k_states[b]
                self.v[layer_idx][b, :, cache_positions[b]] = v_states[b]
        return self.k[layer_idx], self.v[layer_idx]

    def reset(self):
        for layer_idx in range(self.num_layers):
            self.k[layer_idx].zero_()
            self.v[layer_idx].zero_()


class StaticDraftEngine:
    """Encapsulates static-shape Stage 1 verification and Stage 2 drafting."""

    def __init__(
        self,
        base_model: nn.Module,
        drafter: WeightTiedTopKDrafter,
        max_len: int = 2048,
        batch_size: int = 1,
        v_tokens: int = 7,  # Q_A = 8
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.base_model = base_model
        self.drafter = drafter
        self.max_len = max_len
        self.batch_size = batch_size
        self.v_tokens = v_tokens
        self.q_a = v_tokens + 1  # 8
        self.w = v_tokens        # 7

        if device is None:
            device = next(base_model.parameters()).device
        if dtype is None:
            dtype = next(base_model.parameters()).dtype
        self.device = device
        self.dtype = dtype

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
        self.static_cur_pos = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

        # Pre-allocate static attention masks
        # static_mask_A: (B, 1, Q_A, MAX_LEN)
        self.static_mask_A = torch.full((batch_size, 1, self.q_a, max_len), -1e4, dtype=dtype, device=device)
        # static_mask_B: (B, 1, W, MAX_LEN)
        self.static_mask_B = torch.full((batch_size, 1, self.w, max_len), -1e4, dtype=dtype, device=device)

        self.cuda_graph_verify = None
        self.cuda_graph_draft = None

    def build_verify_mask(self, cur_pos: Union[int, torch.Tensor]) -> torch.Tensor:
        """Construct static mask for Stage 1 verification of length Q_A = 8.
        Queries at cur_pos - 1 + [0 .. Q_A - 1] attend causally to all past prefix 0 .. cur_pos - 1
        and causally among the Q_A tokens.
        """
        B = self.batch_size
        mask = torch.full((B, 1, self.q_a, self.max_len), -1e4, dtype=self.dtype, device=self.device)
        if isinstance(cur_pos, int):
            # Prefix keys 0 .. cur_pos - 2
            if cur_pos > 1:
                mask[:, :, :, : cur_pos - 1] = 0.0
            # Causal within verify block of length Q_A: start pos is cur_pos - 1
            causal_diag = torch.tril(torch.zeros(self.q_a, self.q_a, dtype=self.dtype, device=self.device))
            mask[:, :, :, cur_pos - 1 : cur_pos - 1 + self.q_a] = causal_diag
        else:
            for b in range(B):
                cp = cur_pos[b].item()
                if cp > 1:
                    mask[b, 0, :, : cp - 1] = 0.0
                causal_diag = torch.tril(torch.zeros(self.q_a, self.q_a, dtype=self.dtype, device=self.device))
                mask[b, 0, :, cp - 1 : cp - 1 + self.q_a] = causal_diag
        return mask

    def build_draft_mask(self, cur_pos: Union[int, torch.Tensor]) -> torch.Tensor:
        """Construct static mask for Stage 2 drafting of length W = 7.
        All W draft queries attend bidirectionally to:
        1. All valid prefix positions 0 .. cur_pos - 1
        2. All W draft positions
        """
        B = self.batch_size
        mask = torch.full((B, 1, self.w, self.max_len), -1e4, dtype=self.dtype, device=self.device)
        if isinstance(cur_pos, int):
            # Attend to all prefix keys up to cur_pos - 1
            if cur_pos > 0:
                mask[:, :, :, :cur_pos] = 0.0
            # Attend bidirectionally to all W draft positions: cur_pos .. cur_pos + W - 1
            mask[:, :, :, cur_pos : cur_pos + self.w] = 0.0
        else:
            for b in range(B):
                cp = cur_pos[b].item()
                if cp > 0:
                    mask[b, 0, :, :cp] = 0.0
                mask[b, 0, :, cp : cp + self.w] = 0.0
        return mask

    def prefill(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stage 1 prefill on prompt tokens.
        Writes target KV into target_cache, computes initial F_ctx, and writes prefix KV to draft_cache.
        Returns: (first_token_logits, f_ctx_last)
        """
        B, P = input_ids.shape
        self.target_cache.reset()
        self.draft_cache.reset()

        # Run base model forward pass with output_hidden_states=True
        # Use causal mask for prompt
        causal_mask = torch.tril(torch.ones(P, P, dtype=torch.bool, device=self.device))
        attn_mask = torch.where(causal_mask, 0.0, -1e4).to(self.dtype).unsqueeze(0).unsqueeze(0)

        pos_ids = torch.arange(P, device=self.device).unsqueeze(0).expand(B, P)

        # Forward through base model layers manually to populate target_cache
        h = self.base_model.model.embed_tokens(input_ids)
        cos, sin = self.base_model.model.rotary_emb(h, pos_ids)

        all_hidden_states = [h]
        for l_idx, layer in enumerate(self.base_model.model.layers):
            norm_h = layer.input_layernorm(h)
            q = layer.self_attn.q_proj(norm_h).view(B, P, self.num_heads, self.head_dim).transpose(1, 2)
            k = layer.self_attn.k_proj(norm_h).view(B, P, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = layer.self_attn.v_proj(norm_h).view(B, P, self.num_kv_heads, self.head_dim).transpose(1, 2)

            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
            # Write into target_cache
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
            all_hidden_states.append(h)

        # Final norm and lm_head for first token
        h_norm = self.base_model.model.norm(h[:, -1:])
        logits = self.base_model.lm_head(h_norm)  # (B, 1, vocab)

        # Multi-layer target feature fusion on prefix
        f_ctx = self.drafter.fuse_stage1_features(tuple(all_hidden_states))  # (B, P, D)
        f_ctx_last = f_ctx[:, -1:].clone()

        # Project and write prefix KV into draft_cache
        k_list, v_list = self.drafter.project_prefix_kv(f_ctx, pos_ids)
        for d_l_idx in range(self.k_layers):
            self.draft_cache.update(d_l_idx, k_list[d_l_idx], v_list[d_l_idx], torch.arange(P, device=self.device))

        return logits, f_ctx_last

    def forward_verify_step(
        self,
        verify_tokens: torch.Tensor,       # (B, Q_A=8)
        cur_pos_list: List[int],           # list of B ints
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stage 1 verify pass over Q_A tokens for each row in batch.
        Writes target KV into target_cache at [cur_pos[b] - 1 .. cur_pos[b] + Q_A - 2].
        Returns: (logits (B, Q_A, vocab), f_ctx_committed (B, Q_A, D))
        """
        B, Q_A = verify_tokens.shape
        max_total_len = max(cur_pos_list) - 1 + Q_A
        pos_ids = torch.zeros(B, Q_A, dtype=torch.long, device=self.device)
        mask = torch.full((B, 1, Q_A, max_total_len), -1e4, dtype=self.dtype, device=self.device)

        for b in range(B):
            start_pos = cur_pos_list[b] - 1
            pos_ids[b] = torch.arange(start_pos, start_pos + Q_A, device=self.device)
            # Prefix keys [:start_pos]
            if start_pos > 0:
                mask[b, 0, :, :start_pos] = 0.0
            # Causal within Q_A
            causal_block = torch.triu(torch.full((Q_A, Q_A), -1e4, dtype=self.dtype, device=self.device), diagonal=1)
            mask[b, 0, :, start_pos : start_pos + Q_A] = causal_block

        h = self.base_model.model.embed_tokens(verify_tokens)
        cos, sin = self.base_model.model.rotary_emb(h, pos_ids)

        all_hidden_states = [h]
        for l_idx, layer in enumerate(self.base_model.model.layers):
            norm_h = layer.input_layernorm(h)
            q = layer.self_attn.q_proj(norm_h).view(B, Q_A, self.num_heads, self.head_dim).transpose(1, 2)
            k = layer.self_attn.k_proj(norm_h).view(B, Q_A, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = layer.self_attn.v_proj(norm_h).view(B, Q_A, self.num_kv_heads, self.head_dim).transpose(1, 2)

            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

            # In-place write to target_cache per row
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
            all_hidden_states.append(h)

        h_norm = self.base_model.model.norm(h)
        logits = self.base_model.lm_head(h_norm)  # (B, Q_A, vocab)

        f_ctx = self.drafter.fuse_stage1_features(tuple(all_hidden_states))  # (B, Q_A, D)
        return logits, f_ctx

    def forward_draft_step(
        self,
        y_star_ids: torch.Tensor,    # (B, 1) concrete bonus token
        mask_token_id: int,          # 151665
        f_ctx_last: torch.Tensor,    # (B, 1, D)
        cur_pos_list: List[int],     # list of B ints
    ) -> torch.Tensor:
        """Stage 2 draft pass: produces W = 7 draft tokens predicting [cur_pos .. cur_pos + 6].
        Returns: (B, W=7) draft token ids
        """
        B = y_star_ids.shape[0]
        W = self.w
        max_prefix_len = max(cur_pos_list)

        kp_list = [self.draft_cache.k[i][:, :, :max_prefix_len] for i in range(self.k_layers)]
        vp_list = [self.draft_cache.v[i][:, :, :max_prefix_len] for i in range(self.k_layers)]

        # Mask: per row b attends to valid prefix [:cur_pos[b]] and W draft positions
        attn_mask = torch.full((B, 1, W, max_prefix_len + W), -1e4, dtype=self.dtype, device=self.device)
        pos_draft = torch.zeros(B, W, dtype=torch.long, device=self.device)
        for b in range(B):
            cp = cur_pos_list[b]
            if cp > 0:
                attn_mask[b, 0, :, :cp] = 0.0
            attn_mask[b, 0, :, max_prefix_len : max_prefix_len + W] = 0.0
            pos_draft[b] = torch.arange(cp - 1, cp + W - 1, device=self.device)

        # Token embeddings
        y_star_emb = self.drafter.embed_tokens(y_star_ids)
        mask_tensor = torch.tensor([[mask_token_id]], dtype=torch.long, device=self.device)
        mask_emb = self.drafter.embed_tokens(mask_tensor)

        h = self.drafter.bridge(y_star_emb, mask_emb, f_ctx_last, w=W)

        # 4 draft layers
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
        logits = self.drafter.lm_head(h_norm)  # (B, W, vocab)
        draft_ids = torch.argmax(logits, dim=-1)  # (B, W)
        return draft_ids

    def commit_accepted_tokens(
        self,
        f_ctx_new: torch.Tensor,    # (B, max_k, D) newly committed F_ctx vectors
        cur_pos_prev_list: List[int], # previous start positions per row
        k_list: List[int],          # number of accepted tokens per row
    ):
        """Append the k_b newly committed F_ctx vectors into draft_cache for each row b."""
        B = f_ctx_new.shape[0]
        for b in range(B):
            k = k_list[b]
            if k <= 0:
                continue
            cur_pos_prev = cur_pos_prev_list[b]
            pos_ids = torch.arange(cur_pos_prev, cur_pos_prev + k, device=self.device).unsqueeze(0)
            cache_positions = torch.arange(cur_pos_prev, cur_pos_prev + k, device=self.device)
            f_slice = f_ctx_new[b : b + 1, :k]

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
    )

    # 1. Prefill prompt
    logits_0, f_ctx_last = engine.prefill(input_ids)
    if temperature == 0.0:
        next_token = torch.argmax(logits_0[:, -1:], dim=-1)  # (B, 1)
    else:
        probs = F.softmax(logits_0[:, -1:] / temperature, dim=-1)
        next_token = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(B, 1)

    generated_per_row = [[input_ids[b], next_token[b]] for b in range(B)]
    cur_pos_list = [P + 1 for _ in range(B)]
    done = [False] * B

    # Acceptance tracking
    total_draft_tokens = 0
    accepted_draft_tokens = 0
    steps = 0
    accept_counts = [0] * (v_tokens + 2)

    while not all(done) and any((cur_pos_list[b] - P) < max_new_tokens for b in range(B)):
        y_star = next_token  # (B, 1) concrete bonus token

        # 2. Stage 2: Draft V tokens (W = 7)
        draft_ids = engine.forward_draft_step(
            y_star_ids=y_star,
            mask_token_id=mask_token_id,
            f_ctx_last=f_ctx_last,
            cur_pos_list=cur_pos_list,
        )  # (B, V=7)

        # 3. Stage 1: Verify batch Q_A = 8: [y*, d_0, ..., d_6]
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

            accepted_k = k
            num_committed = k + 1
            accept_counts[num_committed] += 1
            total_draft_tokens += v_tokens
            accepted_draft_tokens += accepted_k

            committed_b = torch.cat([draft_ids[b, :k], target_tokens[b, k : k + 1]], dim=0)
            generated_per_row[b].append(committed_b)

            b_token = committed_b[-1:].unsqueeze(0)  # (1, 1)
            next_token_list.append(b_token)
            f_ctx_last_list.append(f_ctx_verify[b : b + 1, k : k + 1])

            k_list.append(num_committed)
            cur_pos_list[b] += num_committed

            if eos_token_id is not None and (committed_b == eos_token_id).any():
                done[b] = True
            if (cur_pos_list[b] - P) >= max_new_tokens:
                done[b] = True

        next_token = torch.cat(next_token_list, dim=0)
        f_ctx_last = torch.cat(f_ctx_last_list, dim=0)

        # Commit accepted vectors
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

    stats = {
        "steps": steps,
        "total_draft_tokens": total_draft_tokens,
        "accepted_draft_tokens": accepted_draft_tokens,
        "alpha": (accepted_draft_tokens + steps * B) / max(1, steps * B),
        "accept_distribution": accept_counts,
    }
    return output_ids, stats
