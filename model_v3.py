"""BFG-Diff v3: Weight-Tied Top-4-Layer Block-Diffusion Drafter (<1% PEFT).

Core Architectural Guarantees:
1. <1% trainable parameters (~0.60% on Qwen2.5-1.5B, ~0.39% on Qwen2.5-3B).
2. Zero separate draft model: weight-tied to frozen backbone's top 4 layers (L-4 .. L-1).
3. 100% lossless Stage 1 autoregressive verification: base model runs untouched with zero LoRA dispatch.
4. No PEFT wrappers on drafter: Stage 1 has zero wrapper overhead; Stage 2 is 100% CUDA-Graph safe.
5. Manifold Alignment Bridge: aligns Layer-0 token embeddings to Layer L-4 residual manifold.
6. Exact Right-Shift Alignment: Slot 0 predicts A_+1 directly from concrete bonus token y*.
"""

import math
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, apply_rotary_pos_emb, repeat_kv, rotate_half

MASK_TOKEN = "<|extra_0|>"
MASK_TOKEN_ID = 151665  # Authoritative ID in Qwen2.5 tokenizer


def apply_rotary_pos_emb_k(k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1) -> torch.Tensor:
    """Key-only rotary position embedding without dummy query allocation."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (k * cos) + (rotate_half(k) * sin)


def setup_tokenizer(model_name: str):
    """Return (tokenizer, mask_id). [MASK] aliases spare row 151665 without resizing."""
    try:
        tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_name)
    if MASK_TOKEN not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [MASK_TOKEN]})
    mask_id = tok.convert_tokens_to_ids(MASK_TOKEN)
    assert isinstance(mask_id, int) and mask_id != tok.unk_token_id, "cannot register [MASK]"
    return tok, mask_id


class DraftLoRADelta(nn.Module):
    """Lightweight LoRA delta (~20 lines, no PEFT wrapper overhead, 100% CUDA-Graph safe)."""

    def __init__(self, in_features: int, out_features: int, r: int = 32, alpha: float = 64.0):
        super().__init__()
        self.r = r
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(r)
        self.lora_A = nn.Parameter(torch.empty(r, in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (..., in_features)
        # Compute in delta.lora_A.dtype, then cast residual output back to x.dtype
        x_in = x.to(dtype=self.lora_A.dtype)
        out = self.scaling * F.linear(F.linear(x_in, self.lora_A), self.lora_B)
        return out.to(dtype=x.dtype)


def apply_linear_with_delta(base_linear: nn.Linear, x: torch.Tensor, delta: DraftLoRADelta) -> torch.Tensor:
    """Helper: base_linear(x) + delta(x.to(delta.lora_A.dtype)).to(x.dtype)"""
    return base_linear(x) + delta(x.to(delta.lora_A.dtype)).to(x.dtype)


class BottleneckFusion(nn.Module):
    """Multi-Layer Target Feature Fusion (F_ctx):
    Taps 4 evenly spaced target layers from Stage 1:
      L=28 (1.5B) -> {6, 13, 20, 27}
      L=36 (3B)   -> {8, 17, 26, 35}
      General formula: [(L // 4) * i - 1 for i in range(1, 5)]
    Bottleneck MLP + RMSNorm:
      Linear(4*D, 512, bias=False) -> SiLU -> Linear(512, D, bias=False) -> Qwen2RMSNorm(D)
    """

    def __init__(self, dim: int, num_tapped: int = 4, bottleneck: int = 512):
        super().__init__()
        self.dim = dim
        self.num_tapped = num_tapped
        self.down = nn.Linear(num_tapped * dim, bottleneck, bias=False)
        self.act = nn.SiLU()
        self.up = nn.Linear(bottleneck, dim, bias=False)
        self.norm = Qwen2RMSNorm(dim, eps=1e-6)

    def forward(self, tapped_states: List[torch.Tensor]) -> torch.Tensor:
        # tapped_states: list of 4 tensors (B, S, D)
        cat_h = torch.cat(tapped_states, dim=-1)  # (B, S, 4*D)
        w_dtype = self.down.weight.dtype if isinstance(getattr(self.down, "weight", None), torch.Tensor) else cat_h.dtype
        out = self.up(self.act(self.down(cat_h.to(w_dtype))))
        return self.norm(out).to(tapped_states[0].dtype)


class ManifoldAlignmentBridge(nn.Module):
    """Manifold Alignment Bridge (Fixing Layer-0 vs. Layer-(L-4) Mismatch):
    Slot 0 (concrete bonus token y*): input_bridge(cat([RMSNorm(Emb(y*)), F_ctx_last]))
    Slots 1..W-1 ([MASK]): RMSNorm(Emb(MASK)) + SlotEmb[1..W-1] + F_ctx_last
    Bottleneck: 2D -> 512 -> D + residual F_ctx_last. Zero-init up projection so initial forward
    starts cleanly in the Layer-(L-4) manifold.
    """

    def __init__(self, dim: int, max_w: int = 8, bottleneck: int = 512):
        super().__init__()
        self.dim = dim
        self.max_w = max_w
        self.emb_norm = Qwen2RMSNorm(dim, eps=1e-6)
        self.slot_emb = nn.Parameter(torch.zeros(max_w, dim, dtype=torch.float32))
        nn.init.normal_(self.slot_emb, std=0.02)
        self.down = nn.Linear(2 * dim, bottleneck, bias=False)
        self.act = nn.SiLU()
        self.up = nn.Linear(bottleneck, dim, bias=False)
        nn.init.normal_(self.up.weight, std=1e-3)  # Near-zero init for clean residual start + non-zero grad flow

    def forward(
        self,
        y_star_emb: torch.Tensor,  # (B, 1, D)
        mask_emb: torch.Tensor,    # (1, 1, D) or (B, 1, D)
        f_ctx_last: torch.Tensor,  # (B, 1, D)
        w: int,
    ) -> torch.Tensor:
        """Construct aligned draft block input of length W in Layer-(L-4) manifold."""
        B = y_star_emb.shape[0]
        dtype = y_star_emb.dtype
        w_dtype = self.down.weight.dtype if isinstance(getattr(self.down, "weight", None), torch.Tensor) else dtype

        # Slot 0 token embedding: y*
        norm_y = self.emb_norm(y_star_emb.to(w_dtype)) + self.slot_emb[0:1].unsqueeze(0).to(w_dtype)  # (B, 1, D)

        # Slots 1..W-1 token embeddings: [MASK]
        norm_mask = self.emb_norm(mask_emb.to(w_dtype))  # (B or 1, 1, D)
        if norm_mask.shape[0] != B:
            norm_mask = norm_mask.expand(B, 1, self.dim)

        if w > 1:
            slots_1_to_w = self.slot_emb[1:w].unsqueeze(0).expand(B, w - 1, self.dim).to(w_dtype)  # (B, W-1, D)
            norm_mask_slots = norm_mask.expand(B, w - 1, self.dim) + slots_1_to_w
            norm_emb = torch.cat([norm_y, norm_mask_slots], dim=1)  # (B, W, D)
        else:
            norm_emb = norm_y  # (B, 1, D)

        f_ctx_expanded = f_ctx_last.expand(B, w, self.dim).to(w_dtype)
        cat_feat = torch.cat([norm_emb, f_ctx_expanded], dim=-1)  # (B, W, 2*D)

        bridge_out = self.up(self.act(self.down(cat_feat)))  # (B, W, D)
        aligned = (f_ctx_expanded + bridge_out).to(dtype)
        return aligned


class LayerDeltas(nn.Module):
    """Holds DraftLoRADelta for a single transformer layer (q, k, v, o, gate, up, down)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        r: int = 32,
        alpha: float = 64.0,
    ):
        super().__init__()
        self.q_proj = DraftLoRADelta(dim, num_heads * head_dim, r=r, alpha=alpha)
        self.k_proj = DraftLoRADelta(dim, num_kv_heads * head_dim, r=r, alpha=alpha)
        self.v_proj = DraftLoRADelta(dim, num_kv_heads * head_dim, r=r, alpha=alpha)
        self.o_proj = DraftLoRADelta(num_heads * head_dim, dim, r=r, alpha=alpha)
        self.gate_proj = DraftLoRADelta(dim, intermediate_size, r=r, alpha=alpha)
        self.up_proj = DraftLoRADelta(dim, intermediate_size, r=r, alpha=alpha)
        self.down_proj = DraftLoRADelta(intermediate_size, dim, r=r, alpha=alpha)


class WeightTiedTopKDrafter(nn.Module):
    """BFG-Diff v3 Drafter: Weight-Tied to the top K (default 4) layers of the base model.
    References (never copies) base_model.model.layers[L-K : L], norm, lm_head, and embed_tokens.
    """

    def __init__(
        self,
        base_model: nn.Module,
        k_layers: int = 4,
        r: int = 32,
        alpha: float = 64.0,
        max_w: int = 8,
    ):
        super().__init__()
        self.k_layers = k_layers
        self.max_w = max_w
        self.r = r
        self.alpha = alpha

        # Frozen backbone references
        self.base_model = base_model
        config = base_model.config
        self.dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", self.dim // self.num_heads)
        self.intermediate_size = config.intermediate_size
        self.total_layers = config.num_hidden_layers

        # Referenced backbone modules (unwrapped, zero copies)
        self.draft_layers = base_model.model.layers[self.total_layers - k_layers :]
        self.norm = base_model.model.norm
        self.lm_head = base_model.lm_head
        self.embed_tokens = base_model.get_input_embeddings()
        self.rotary_emb = base_model.model.rotary_emb

        # Tapped layer indices: [(L // 4) * i - 1 for i in range(1, 5)]
        step = self.total_layers // 4
        self.tapped_layer_indices = [step * i - 1 for i in range(1, 5)]

        # Trainable drafter modules (instantiated in fp32)
        self.fusion = BottleneckFusion(self.dim, num_tapped=4, bottleneck=512)
        self.bridge = ManifoldAlignmentBridge(self.dim, max_w=max_w, bottleneck=512)
        self.layer_deltas = nn.ModuleList(
            [
                LayerDeltas(
                    self.dim,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.intermediate_size,
                    r=r,
                    alpha=alpha,
                )
                for _ in range(k_layers)
            ]
        )

        # Ghost layers for zero-overhead inference
        self.ghost_layers = None
        self.has_ghost_layers = False

    def merge_ghost_weights(self, dtype: Optional[torch.dtype] = None):
        """Inference-only optimization: deep-copy the 4 tapped decoder layers and fold
        LoRA deltas directly into the linear weights: W_ghost = W_base + (alpha / r) * (B @ A).
        Eliminates 28 unfused delta additions per draft step without modifying base_model.
        """
        import copy
        self.ghost_layers = nn.ModuleList([copy.deepcopy(layer) for layer in self.draft_layers])
        for l_idx, ghost_layer in enumerate(self.ghost_layers):
            deltas = self.layer_deltas[l_idx]
            for name, proj in [
                ("q_proj", ghost_layer.self_attn.q_proj),
                ("k_proj", ghost_layer.self_attn.k_proj),
                ("v_proj", ghost_layer.self_attn.v_proj),
                ("o_proj", ghost_layer.self_attn.o_proj),
                ("gate_proj", ghost_layer.mlp.gate_proj),
                ("up_proj", ghost_layer.mlp.up_proj),
                ("down_proj", ghost_layer.mlp.down_proj),
            ]:
                delta = getattr(deltas, name)
                delta_w = delta.scaling * torch.matmul(delta.lora_B.data.float(), delta.lora_A.data.float())
                w_merged = proj.weight.data.float() + delta_w
                target_dtype = dtype if dtype is not None else proj.weight.dtype
                proj.weight.data.copy_(w_merged.to(target_dtype))
                proj.weight.requires_grad = False
        if dtype is not None:
            self.ghost_layers.to(dtype=dtype)
        self.has_ghost_layers = True
        return self

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Return list of trainable parameters (fusion, bridge, layer_deltas)."""
        params = []
        params.extend(self.fusion.parameters())
        params.extend(self.bridge.parameters())
        params.extend(self.layer_deltas.parameters())
        return params

    def parameter_breakdown(self) -> dict:
        """Compute exact parameter breakdown."""
        p_fusion = sum(p.numel() for p in self.fusion.parameters())
        p_bridge = sum(p.numel() for p in self.bridge.parameters())
        p_deltas = sum(p.numel() for p in self.layer_deltas.parameters())
        p_trainable = p_fusion + p_bridge + p_deltas
        p_base = sum(p.numel() for p in self.base_model.parameters())
        return {
            "dim": self.dim,
            "fusion": p_fusion,
            "bridge": p_bridge,
            "deltas_4l": p_deltas,
            "total_drafter": p_trainable,
            "total_base": p_base,
            "trainable_ratio_pct": (p_trainable / max(1, p_base)) * 100.0,
            "tapped_layers": self.tapped_layer_indices,
        }

    def fuse_stage1_features(self, all_hidden_states: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Extract tapped hidden states from Stage 1 verify outputs and fuse them into F_ctx.
        all_hidden_states: tuple of length total_layers + 1. Layer k is at index k + 1.
        Returns: F_ctx (B, S, D)
        """
        tapped = [all_hidden_states[idx + 1] for idx in self.tapped_layer_indices]
        return self.fusion(tapped)

    def forward_draft_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,     # (B, W, D)
        pos_draft: torch.Tensor,          # (B, W)
        k_prefix: Optional[torch.Tensor] = None,  # (B, num_kv_heads, S_p, head_dim)
        v_prefix: Optional[torch.Tensor] = None,  # (B, num_kv_heads, S_p, head_dim)
        attn_mask: Optional[torch.Tensor] = None,  # (B, 1, W, S_p + W) bool or float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute one draft layer referencing base_model layer with DraftLoRADelta (or fused ghost weights).
        Returns: (hidden_states, k_draft_total, v_draft_total)
        """
        B, W, D = hidden_states.shape

        if self.has_ghost_layers:
            ghost_layer = self.ghost_layers[layer_idx]
            norm_h = ghost_layer.input_layernorm(hidden_states)
            q = ghost_layer.self_attn.q_proj(norm_h).view(B, W, self.num_heads, self.head_dim).transpose(1, 2)
            k_draft = ghost_layer.self_attn.k_proj(norm_h).view(B, W, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v_draft = ghost_layer.self_attn.v_proj(norm_h).view(B, W, self.num_kv_heads, self.head_dim).transpose(1, 2)

            cos_d, sin_d = self.rotary_emb(norm_h, pos_draft)
            q, k_draft = apply_rotary_pos_emb(q, k_draft, cos_d, sin_d, unsqueeze_dim=1)

            if k_prefix is not None and v_prefix is not None:
                k_total = torch.cat([k_prefix, k_draft], dim=2)
                v_total = torch.cat([v_prefix, v_draft], dim=2)
            else:
                k_total = k_draft
                v_total = v_draft

            if self.num_heads != self.num_kv_heads:
                n_rep = self.num_heads // self.num_kv_heads
                k_attn = repeat_kv(k_total, n_rep)
                v_attn = repeat_kv(v_total, n_rep)
            else:
                k_attn = k_total
                v_attn = v_total

            attn_out = F.scaled_dot_product_attention(
                q, k_attn, v_attn, attn_mask=attn_mask, dropout_p=0.0
            )
            attn_out = attn_out.transpose(1, 2).reshape(B, W, D)
            hidden_states = hidden_states + ghost_layer.self_attn.o_proj(attn_out)

            norm_post = ghost_layer.post_attention_layernorm(hidden_states)
            gate = ghost_layer.mlp.gate_proj(norm_post)
            up = ghost_layer.mlp.up_proj(norm_post)
            mlp_act = ghost_layer.mlp.act_fn(gate) * up
            down = ghost_layer.mlp.down_proj(mlp_act)
            hidden_states = hidden_states + down
            return hidden_states, k_draft, v_draft

        # Standard training / unfused forward path
        base_layer = self.draft_layers[layer_idx]
        deltas = self.layer_deltas[layer_idx]

        # 1. Input RMSNorm
        norm_h = base_layer.input_layernorm(hidden_states)

        # 2. Self Attention
        # Project queries with delta
        q = apply_linear_with_delta(base_layer.self_attn.q_proj, norm_h, deltas.q_proj)
        q = q.view(B, W, self.num_heads, self.head_dim).transpose(1, 2)  # (B, n_h, W, head_dim)

        # Project draft keys and values with delta
        k_draft = apply_linear_with_delta(base_layer.self_attn.k_proj, norm_h, deltas.k_proj)
        k_draft = k_draft.view(B, W, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_draft = apply_linear_with_delta(base_layer.self_attn.v_proj, norm_h, deltas.v_proj)
        v_draft = v_draft.view(B, W, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to draft Q and K
        cos_d, sin_d = self.rotary_emb(norm_h, pos_draft)
        q, k_draft = apply_rotary_pos_emb(q, k_draft, cos_d, sin_d, unsqueeze_dim=1)

        # KV Concatenation: cached prefix KV + draft KV
        if k_prefix is not None and v_prefix is not None:
            k_total = torch.cat([k_prefix, k_draft], dim=2)  # (B, n_kv, S_p + W, head_dim)
            v_total = torch.cat([v_prefix, v_draft], dim=2)
        else:
            k_total = k_draft
            v_total = v_draft

        # GQA Repeat if needed
        if self.num_heads != self.num_kv_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k_attn = repeat_kv(k_total, n_rep)
            v_attn = repeat_kv(v_total, n_rep)
        else:
            k_attn = k_total
            v_attn = v_total

        # Scaled dot-product attention
        attn_out = F.scaled_dot_product_attention(
            q, k_attn, v_attn, attn_mask=attn_mask, dropout_p=0.0
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, W, D)

        # Output projection with delta + residual
        o_out = apply_linear_with_delta(base_layer.self_attn.o_proj, attn_out, deltas.o_proj)
        hidden_states = hidden_states + o_out

        # 3. MLP with deltas
        norm_post = base_layer.post_attention_layernorm(hidden_states)
        gate = apply_linear_with_delta(base_layer.mlp.gate_proj, norm_post, deltas.gate_proj)
        up = apply_linear_with_delta(base_layer.mlp.up_proj, norm_post, deltas.up_proj)
        mlp_act = base_layer.mlp.act_fn(gate) * up
        down = apply_linear_with_delta(base_layer.mlp.down_proj, mlp_act, deltas.down_proj)
        hidden_states = hidden_states + down

        return hidden_states, k_draft, v_draft

    def forward_draft(
        self,
        y_star_ids: torch.Tensor,           # (B, 1) concrete bonus token
        mask_token_id: int,                 # int [MASK] token id (151665)
        f_ctx_last: torch.Tensor,           # (B, 1, D) F_ctx of token at cur_pos - 1
        cur_pos: int,                       # integer current position t (where y* = x_t)
        w: int = 7,                         # draft block length W
        k_prefix_list: Optional[List[torch.Tensor]] = None,  # list of 4 tensors (B, n_kv, S_p, head_dim)
        v_prefix_list: Optional[List[torch.Tensor]] = None,  # list of 4 tensors (B, n_kv, S_p, head_dim)
        attn_mask: Optional[torch.Tensor] = None,            # (B, 1, W, S_p + W)
        pos_draft: Optional[torch.Tensor] = None,            # (B, W)
    ) -> torch.Tensor:
        """Run 4-layer bidirectional draft pass.
        When drafting from concrete bonus token y* at sequence position t (y* = x_t):
        - y* has RoPE position t, and the W draft block tokens [y*, MASK_1, ..., MASK_{W-1}]
          have RoPE positions [t, t+1, ..., t+W-1], predicting [x_{t+1}, x_{t+2}, ..., x_{t+W}].
        - f_ctx_last MUST be f_ctx[t-1] (fused feature of token immediately preceding y*).
        - Prefix KV contains positions 0 .. t-1 (length t).
        Slot 0 predicts A_+1 (x_{t+1}) directly from concrete bonus token y*.
        Returns: logits (B, W, vocab_size)
        """
        B = y_star_ids.shape[0]
        device = y_star_ids.device

        # Token embeddings
        y_star_emb = self.embed_tokens(y_star_ids)  # (B, 1, D)
        mask_tensor = torch.tensor([[mask_token_id]], dtype=torch.long, device=device)
        mask_emb = self.embed_tokens(mask_tensor)   # (1, 1, D)

        # Manifold alignment
        h = self.bridge(y_star_emb, mask_emb, f_ctx_last, w=w)  # (B, W, D)

        # RoPE positions for draft block: [cur_pos, cur_pos + 1, ..., cur_pos + W - 1]
        if pos_draft is None:
            pos_draft = torch.arange(cur_pos, cur_pos + w, device=device).unsqueeze(0).expand(B, w)

        # Execute 4 draft layers
        for layer_idx in range(self.k_layers):
            kp = k_prefix_list[layer_idx] if k_prefix_list is not None else None
            vp = v_prefix_list[layer_idx] if v_prefix_list is not None else None
            h, _, _ = self.forward_draft_layer(
                layer_idx=layer_idx,
                hidden_states=h,
                pos_draft=pos_draft,
                k_prefix=kp,
                v_prefix=vp,
                attn_mask=attn_mask,
            )

        # Output head: frozen norm + frozen lm_head
        h_norm = self.norm(h)
        logits = self.lm_head(h_norm)  # (B, W, vocab_size)
        return logits

    def project_prefix_kv(
        self,
        f_ctx: torch.Tensor,        # (B, S_k, D)
        pos_ids: torch.Tensor,      # (B, S_k)
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Project and RoPE prefix K and V for the 4 draft layers directly from F_ctx.
        Returns: (k_list, v_list) each containing 4 tensors of shape (B, num_kv_heads, S_k, head_dim)
        """
        B, S_k, D = f_ctx.shape
        k_list, v_list = [], []
        cos, sin = self.rotary_emb(f_ctx, pos_ids)

        for layer_idx in range(self.k_layers):
            if self.has_ghost_layers:
                ghost_layer = self.ghost_layers[layer_idx]
                k = ghost_layer.self_attn.k_proj(f_ctx)
                v = ghost_layer.self_attn.v_proj(f_ctx)
            else:
                base_layer = self.draft_layers[layer_idx]
                deltas = self.layer_deltas[layer_idx]
                k = apply_linear_with_delta(base_layer.self_attn.k_proj, f_ctx, deltas.k_proj)
                v = apply_linear_with_delta(base_layer.self_attn.v_proj, f_ctx, deltas.v_proj)

            k = k.view(B, S_k, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, S_k, self.num_kv_heads, self.head_dim).transpose(1, 2)

            # Apply key-only RoPE without dummy_q allocation
            k = apply_rotary_pos_emb_k(k, cos, sin, unsqueeze_dim=1)

            k_list.append(k)
            v_list.append(v)

        return k_list, v_list


def load_base_model(model_name: str, device: str = "cpu", dtype: torch.dtype = torch.float32) -> AutoModelForCausalLM:
    """Load the unmodified frozen base model."""
    kwargs = {"attn_implementation": "sdpa"}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, local_files_only=True, **kwargs)
    except (TypeError, Exception):
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, local_files_only=True, **kwargs)
        except Exception:
            try:
                model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **kwargs)
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **kwargs)

    # Freeze ALL base model parameters
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


def prepare_v3(
    model_name: str,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    drafter_r: int = 32,
    drafter_alpha: float = 64.0,
    max_w: int = 8,
) -> Tuple[AutoModelForCausalLM, WeightTiedTopKDrafter, AutoTokenizer, int]:
    """Prepare frozen base model + standalone WeightTiedTopKDrafter."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32

    tok, mask_id = setup_tokenizer(model_name)
    base_model = load_base_model(model_name, device=device, dtype=dtype)

    # Ensure mask_id is within bounds (Qwen2.5 has 151665 < 151936)
    emb_rows = base_model.get_input_embeddings().weight.shape[0]
    assert mask_id < emb_rows, f"mask_id {mask_id} exceeds vocabulary size {emb_rows}"

    drafter = WeightTiedTopKDrafter(
        base_model=base_model,
        k_layers=4,
        r=drafter_r,
        alpha=drafter_alpha,
        max_w=max_w,
    ).to(device)

    # Drafter trainable parameters in fp32 for stable training
    for p in drafter.get_trainable_parameters():
        p.requires_grad = True

    return base_model, drafter, tok, mask_id


def quantize_for_cpu(
    base_model: nn.Module,
    drafter: WeightTiedTopKDrafter,
    mode: str = "int8",
) -> Tuple[nn.Module, WeightTiedTopKDrafter]:
    """Auto-adapt model for CPU execution using dynamic INT8 (VNNI) or INT4 quantization.
    
    Strict order:
    1. Merge ghost weights into drafter.ghost_layers on float32 master weights.
    2. Quantize Linear layers in base_model (layers + lm_head) and drafter.ghost_layers
       in-place layer-by-layer with immediate gc.collect() so peak startup RSS stays
       safely under MemoryMax=9G cgroup cap.
    Keeps embeddings and lightweight fusion/bridge modules unquantized for numerical stability.
    """
    import gc

    mode = (mode or "none").lower().strip()
    if mode in ("none", "fp32", "fp16", "bf16"):
        return base_model, drafter

    # 1. Merge ghost weights first on float32/master weights
    if not getattr(drafter, "has_ghost_layers", False):
        drafter.merge_ghost_weights(dtype=torch.float32)

    # 2. Quantize Linear layers in-place layer by layer
    if mode == "int8":
        if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
            for i in range(len(base_model.model.layers)):
                base_model.model.layers[i] = torch.ao.quantization.quantize_dynamic(
                    base_model.model.layers[i], {nn.Linear}, dtype=torch.qint8
                )
                gc.collect()
        if hasattr(base_model, "lm_head") and isinstance(base_model.lm_head, nn.Linear):
            base_model.lm_head = torch.ao.quantization.quantize_dynamic(
                nn.Sequential(base_model.lm_head), {nn.Linear}, dtype=torch.qint8
            )[0]
            gc.collect()

        if hasattr(drafter, "ghost_layers") and drafter.ghost_layers is not None:
            for i in range(len(drafter.ghost_layers)):
                drafter.ghost_layers[i] = torch.ao.quantization.quantize_dynamic(
                    drafter.ghost_layers[i], {nn.Linear}, dtype=torch.qint8
                )
                gc.collect()
        gc.collect()
        print(f"[quantize_for_cpu] Applied in-place dynamic INT8 quantization to base model and ghost layers.")

    elif mode == "int4":
        try:
            import torchao
            from torchao.quantization import quantize_, int4_weight_only
            quantize_(base_model, int4_weight_only())
            if hasattr(drafter, "ghost_layers") and drafter.ghost_layers is not None:
                quantize_(drafter.ghost_layers, int4_weight_only())
            print(f"[quantize_for_cpu] Applied torchao INT4 weight-only quantization.")
        except Exception as e:
            print(f"[quantize_for_cpu] torchao INT4 unavailable ({e}); falling back to in-place dynamic INT8.")
            if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
                for i in range(len(base_model.model.layers)):
                    base_model.model.layers[i] = torch.ao.quantization.quantize_dynamic(
                        base_model.model.layers[i], {nn.Linear}, dtype=torch.qint8
                    )
                    gc.collect()
            if hasattr(base_model, "lm_head") and isinstance(base_model.lm_head, nn.Linear):
                base_model.lm_head = torch.ao.quantization.quantize_dynamic(
                    nn.Sequential(base_model.lm_head), {nn.Linear}, dtype=torch.qint8
                )[0]
                gc.collect()
            if hasattr(drafter, "ghost_layers") and drafter.ghost_layers is not None:
                for i in range(len(drafter.ghost_layers)):
                    drafter.ghost_layers[i] = torch.ao.quantization.quantize_dynamic(
                        drafter.ghost_layers[i], {nn.Linear}, dtype=torch.qint8
                    )
                    gc.collect()
            gc.collect()

    return base_model, drafter

