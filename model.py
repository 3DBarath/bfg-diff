"""Qwen2.5 + BFG-Diff: hybrid (prefix-causal, canvas-bidirectional) attention."""
import torch

MASK_TOKEN = "<|extra_0|>"
MASK_TOKEN_ID = 151665  # <|extra_0|> in Qwen2.5 (authoritative id comes from setup_tokenizer)


def setup_tokenizer(model_name):
    """Return (tokenizer, mask_id). [MASK] aliases an unused special-token slot, so the
    embedding matrix and LM head stay tied and no weights are reinitialised."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if MASK_TOKEN not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [MASK_TOKEN]})
    mask_id = tok.convert_tokens_to_ids(MASK_TOKEN)
    assert isinstance(mask_id, int) and mask_id != tok.unk_token_id, "cannot register [MASK]"
    return tok, mask_id


def load_model(model_name, device=None, dtype=None):
    from transformers import AutoModelForCausalLM

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    kwargs = {"attn_implementation": "sdpa"}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **kwargs)
    except TypeError:  # transformers < 4.56
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **kwargs)
    return model.to(device)


def prepare(model_name, device=None, dtype=None):
    """Model + tokenizer + mask id, resizing only if the tokenizer outgrew the embeddings."""
    tok, mask_id = setup_tokenizer(model_name)
    model = load_model(model_name, device, dtype)
    emb = model.get_input_embeddings().weight.shape[0]
    if mask_id >= emb:  # e.g. GPT-2 smoke tests; Qwen2 has 271 spare rows
        model.resize_token_embeddings(max(len(tok), mask_id + 1))
    assert mask_id < model.get_input_embeddings().weight.shape[0]
    return model, tok, mask_id


def apply_lora(model, r=16, alpha=32, dropout=0.05, targets=("q_proj", "k_proj", "v_proj", "o_proj"), mask_id=None):
    """Wrap the frozen model with trainable low-rank adapters (PEFT LoRA).
    If `mask_id` is provided, unfreezes only the `mask_id` row of the input embedding matrix."""
    from peft import LoraConfig, get_peft_model

    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            target_modules=list(targets),
            task_type="CAUSAL_LM",
        ),
    )
    if mask_id is not None:
        emb = peft_model.get_input_embeddings().weight
        emb.requires_grad = True

        def hook(grad):
            g = torch.zeros_like(grad)
            g[mask_id] = grad[mask_id]
            return g

        emb.register_hook(hook)

    return peft_model


LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
LORA_TARGETS_MLP = LORA_TARGETS + ("gate_proj", "up_proj", "down_proj")


def build_hybrid_attention_mask(
    prefix_len, canvas_len, batch_size=1, dtype=torch.bfloat16, device="cuda", prefix_keep=None
):
    """Additive mask (B, 1, P+C, P+C): prefix queries 0..P-2 see only past prefix;
    prefix query P-1 and canvas queries P..P+C-1 see all past prefix and all canvas.
    `prefix_keep` (B, P) bool hides left-padding keys."""
    T = prefix_len + canvas_len
    allow = torch.zeros(batch_size, T, T, dtype=torch.bool, device=device)
    causal = torch.tril(torch.ones(prefix_len, prefix_len, dtype=torch.bool, device=device))
    allow[:, :prefix_len, :prefix_len] = causal
    allow[:, prefix_len - 1 :, :] = True
    if prefix_keep is not None:
        keep = prefix_keep.to(device=device, dtype=torch.bool).view(batch_size, 1, prefix_len)
        allow[:, :prefix_len, :prefix_len] &= keep
        allow[:, prefix_len - 1 :, :prefix_len] &= keep
    return torch.zeros(batch_size, 1, T, T, dtype=dtype, device=device).masked_fill_(
        ~allow.unsqueeze(1), torch.finfo(dtype).min
    )


def mask_position_ids(prefix_keep, canvas_len):
    """Position ids that stay gap-free when the prefix is left-padded."""
    B, P = prefix_keep.shape
    keep = prefix_keep.long()
    prefix_pos = (keep.cumsum(dim=1) - 1).clamp_min(0)
    real_len = keep.sum(dim=1, keepdim=True)
    canvas_pos = real_len + torch.arange(canvas_len, device=prefix_keep.device).unsqueeze(0)
    return torch.cat([prefix_pos, canvas_pos], dim=1)


def prefill_prefix(model, prefix_ids, prefix_keep=None):
    """Run prefix_ids[:, :-1] once with a KV cache; canvas steps then reuse it."""
    B, P = prefix_ids.shape
    if P <= 1:
        # If prefix is only 1 token, initialize an empty DynamicCache
        from transformers import DynamicCache
        return DynamicCache()
    pos = (
        torch.arange(P - 1, device=prefix_ids.device).unsqueeze(0).expand(B, P - 1)
        if prefix_keep is None
        else mask_position_ids(prefix_keep, 0)[:, : P - 1]
    )
    keep = prefix_keep[:, : P - 1] if prefix_keep is not None else None
    return model(
        input_ids=prefix_ids[:, :-1],
        attention_mask=keep,
        position_ids=pos,
        use_cache=True,
    ).past_key_values


class BFGDiffAdapter(torch.nn.Module):
    """Pillar 3: Slot-Offset Embeddings & Latent Residual Recycling.

    - slot_emb: Learnable offset embeddings for Zone B positions +1..+W
    - latent_norm + latent_down + latent_up: Residual projection recycling
      previous step's top-layer hidden states h^(L) into Zone B lookahead slots.
    Zero-initialized so that legacy checkpoints and step 0 produce 100% identical outputs.
    """
    def __init__(self, hidden_size, max_w=16, lora_r=16):
        super().__init__()
        self.max_w = max_w
        self.slot_emb = torch.nn.Embedding(max_w, hidden_size)
        torch.nn.init.zeros_(self.slot_emb.weight)

        try:
            from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
            self.latent_norm = Qwen2RMSNorm(hidden_size, eps=1e-6)
        except Exception:
            self.latent_norm = torch.nn.LayerNorm(hidden_size)

        self.latent_down = torch.nn.Linear(hidden_size, lora_r, bias=False)
        self.latent_up = torch.nn.Linear(lora_r, hidden_size, bias=False)
        torch.nn.init.zeros_(self.latent_up.weight)

    def forward_offset(self, j_indices):
        return self.slot_emb(j_indices)

    def forward_latent(self, prev_latents):
        normed = self.latent_norm(prev_latents)
        return self.latent_up(torch.nn.functional.silu(self.latent_down(normed)))

    def apply_to_embeds(
        self, embeds, zone_a_len=0, zone_b_len=0, prev_latents=None, mask=None, prefix_len=None
    ):
        out_embeds = embeds.clone()
        B = embeds.shape[0]
        device = embeds.device

        # Training / full sequence mode with explicit mask (B, C)
        if mask is not None and prefix_len is not None:
            for b in range(B):
                masked_idx = torch.where(mask[b])[0]
                if len(masked_idx) > 0:
                    j_idx = torch.arange(len(masked_idx), device=device).clamp_max(self.max_w - 1)
                    seq_pos = prefix_len + masked_idx
                    out_embeds[b, seq_pos] = out_embeds[b, seq_pos] + self.slot_emb(j_idx)
                    if prev_latents is not None:
                        w_use = min(len(masked_idx), prev_latents.shape[1])
                        if w_use > 0:
                            rec = self.forward_latent(prev_latents[b : b + 1, :w_use])
                            out_embeds[b, seq_pos[:w_use]] = out_embeds[b, seq_pos[:w_use]] + rec[0]
            return out_embeds

        if zone_b_len <= 0:
            return out_embeds

        # Step / inference mode: Zone A (clean) followed by Zone B (masks)
        j_idx = torch.arange(zone_b_len, device=device).clamp_max(self.max_w - 1)
        offset_embs = self.slot_emb(j_idx).unsqueeze(0)

        out_embeds[:, zone_a_len : zone_a_len + zone_b_len] = (
            out_embeds[:, zone_a_len : zone_a_len + zone_b_len] + offset_embs
        )

        if prev_latents is not None:
            w_use = min(zone_b_len, prev_latents.shape[1])
            if w_use > 0:
                rec_embs = self.forward_latent(prev_latents[:, :w_use])
                out_embeds[:, zone_a_len : zone_a_len + w_use] = (
                    out_embeds[:, zone_a_len : zone_a_len + w_use] + rec_embs
                )

        return out_embeds

    def get_inputs_embeds(self, model, active_ids, zone_a_len, zone_b_len, prev_latents=None):
        base_embeds = model.get_input_embeddings()(active_ids)
        return self.apply_to_embeds(
            base_embeds, zone_a_len=zone_a_len, zone_b_len=zone_b_len, prev_latents=prev_latents
        )


def setup_bfg_adapter(model, max_w=16, lora_r=16):
    """Attach BFGDiffAdapter to model matching its dtype and device."""
    hidden_size = getattr(model.config, "hidden_size", 896)
    adapter = BFGDiffAdapter(hidden_size, max_w=max_w, lora_r=lora_r)
    emb = model.get_input_embeddings()
    dtype = emb.weight.dtype
    device = emb.weight.device
    adapter.to(device=device, dtype=dtype)
    model.bfg_adapter = adapter
    return adapter


def build_staircase_step_mask(
    batch_size,
    prefix_len,
    zone_a_len,
    zone_b_len,
    dtype=torch.bfloat16,
    device="cuda",
    prefix_keep=None,
    tree_mask_a=None,
):
    """Additive attention mask for fused staircase verification + lookahead drafting pass.

    Zone A (queries 0..zone_a_len-1):
      - Causal attention to cached prefix keys 0..prefix_len-2 (respecting prefix_keep)
      - Strictly causal attention to Zone A keys (or tree_mask_a)
      - ZERO attention to Zone B keys (strict causal isolation, Delta_leak = 0)
    Zone B (queries zone_a_len..Q-1):
      - Bidirectional attention across cached prefix, all Zone A keys, and all Zone B keys.
    """
    Q = zone_a_len + zone_b_len
    cached_prefix = max(0, prefix_len - 1)
    K = cached_prefix + Q
    allow = torch.zeros(batch_size, 1, Q, K, dtype=torch.bool, device=device)

    # 1. Prefix visibility
    if cached_prefix > 0:
        allow[:, :, :, :cached_prefix] = True
        if prefix_keep is not None:
            p_init = min(cached_prefix, prefix_keep.shape[1] - 1)
            if p_init > 0:
                keep = prefix_keep[:, :p_init].to(device=device, dtype=torch.bool).view(batch_size, 1, 1, p_init)
                allow[:, :, :, :p_init] &= keep

    # 2. Zone A queries
    if zone_a_len > 0:
        if tree_mask_a is not None:
            t_mask = tree_mask_a.to(device=device, dtype=torch.bool)
            if t_mask.dim() == 2:
                t_mask = t_mask.unsqueeze(0).unsqueeze(0)
            elif t_mask.dim() == 3:
                t_mask = t_mask.unsqueeze(1)
            allow[:, :, :zone_a_len, cached_prefix : cached_prefix + zone_a_len] = t_mask
        else:
            causal_a = torch.tril(torch.ones(zone_a_len, zone_a_len, dtype=torch.bool, device=device))
            allow[:, :, :zone_a_len, cached_prefix : cached_prefix + zone_a_len] = causal_a
        # Causal isolation: Zone A cannot attend to Zone B keys
        allow[:, :, :zone_a_len, cached_prefix + zone_a_len :] = False

    # 3. Zone B queries
    if zone_b_len > 0:
        allow[:, :, zone_a_len :, cached_prefix : cached_prefix + zone_a_len] = True
        allow[:, :, zone_a_len :, cached_prefix + zone_a_len :] = True

    return torch.zeros(batch_size, 1, Q, K, dtype=dtype, device=device).masked_fill_(
        ~allow, torch.finfo(dtype).min
    )


def canvas_step_with_cache(
    model,
    active_ids_or_prefix_last,
    past_key_values_or_canvas_ids=None,
    past_key_values=None,
    prefix_len=None,
    zone_a_len=None,
    prefix_keep=None,
    tree_mask_a=None,
    position_ids_override=None,
    return_hidden=False,
    inputs_embeds=None,
    prev_latents=None,
):
    """Logits for staircase step pass.

    Supports both:
    1. Legacy v1 signature:
       canvas_step_with_cache(model, prefix_last_id, canvas_ids, past_key_values, prefix_len, prefix_keep=None)
       -> returns logits[:, :C, :] of shape (B, C, vocab)
    2. v2 fused staircase signature:
       canvas_step_with_cache(model, active_ids, past_key_values, prefix_len, zone_a_len=1+V, ...)
       -> returns logits of shape (B, Q, vocab) (and hidden states if return_hidden=True)
    """
    if isinstance(past_key_values_or_canvas_ids, torch.Tensor):
        # Legacy v1 call
        prefix_last_id = active_ids_or_prefix_last
        canvas_ids = past_key_values_or_canvas_ids
        active_ids = torch.cat([prefix_last_id, canvas_ids], dim=1) if inputs_embeds is None else None
        pkv = past_key_values
        p_len = prefix_len
        p_keep = zone_a_len  # 6th positional arg was prefix_keep in v1
        z_a_len = 1
        z_b_len = canvas_ids.shape[1]
        is_v1_call = True
    else:
        # v2 staircase call
        active_ids = active_ids_or_prefix_last
        pkv = past_key_values_or_canvas_ids if past_key_values is None else past_key_values
        p_len = prefix_len
        z_a_len = 1 if zone_a_len is None else zone_a_len
        total_len = active_ids.shape[1] if active_ids is not None else inputs_embeds.shape[1]
        z_b_len = max(0, total_len - z_a_len)
        p_keep = prefix_keep
        is_v1_call = False

    B = active_ids.shape[0] if active_ids is not None else inputs_embeds.shape[0]
    Q = active_ids.shape[1] if active_ids is not None else inputs_embeds.shape[1]
    device = active_ids.device if active_ids is not None else inputs_embeds.device
    dtype = model.get_input_embeddings().weight.dtype

    attn = build_staircase_step_mask(
        batch_size=B,
        prefix_len=p_len,
        zone_a_len=z_a_len,
        zone_b_len=z_b_len,
        dtype=dtype,
        device=device,
        prefix_keep=p_keep,
        tree_mask_a=tree_mask_a,
    )

    if position_ids_override is not None:
        pos = position_ids_override
    elif p_keep is not None:
        pad_count = p_keep.shape[1] - p_keep.to(device=device, dtype=torch.long).sum(dim=1, keepdim=True)
        pos = (p_len - 1 - pad_count) + torch.arange(Q, device=device).unsqueeze(0)
    else:
        pos = torch.arange(p_len - 1, p_len - 1 + Q, device=device).unsqueeze(0).expand(B, Q)

    if inputs_embeds is None and getattr(model, "bfg_adapter", None) is not None:
        base_embeds = model.get_input_embeddings()(active_ids)
        inputs_embeds = model.bfg_adapter.apply_to_embeds(
            base_embeds, z_a_len, z_b_len, prev_latents=prev_latents
        )

    kwargs = {
        "attention_mask": attn,
        "position_ids": pos,
        "past_key_values": pkv,
        "use_cache": True,
    }
    if return_hidden:
        kwargs["output_hidden_states"] = True

    if inputs_embeds is not None:
        kwargs["inputs_embeds"] = inputs_embeds
    else:
        kwargs["input_ids"] = active_ids

    out = model(**kwargs)

    logits = out.logits
    res = logits[:, :z_b_len, :] if is_v1_call else logits

    if return_hidden:
        hidden = out.hidden_states[-1]
        return res, hidden
    return res


def canvas_forward(
    model, prefix_ids, canvas_ids, prefix_keep=None, prev_latents=None, return_hidden=False, mask=None
):
    """Logits for the canvas block only: (B, C, vocab), aligned with Qwen's k -> k+1 prediction."""
    B, P = prefix_ids.shape
    C = canvas_ids.shape[1]
    device = prefix_ids.device
    input_ids = torch.cat([prefix_ids, canvas_ids], dim=1)
    dtype = model.get_input_embeddings().weight.dtype
    attn = build_hybrid_attention_mask(P, C, B, dtype, device, prefix_keep)
    pos = (
        torch.arange(P + C, device=device).unsqueeze(0).expand(B, P + C)
        if prefix_keep is None
        else mask_position_ids(prefix_keep, C)
    )
    kwargs = {"attention_mask": attn, "position_ids": pos, "use_cache": False}
    if return_hidden:
        kwargs["output_hidden_states"] = True

    if getattr(model, "bfg_adapter", None) is not None:
        base_embeds = model.get_input_embeddings()(input_ids)
        mask_to_use = mask if mask is not None else (canvas_ids == getattr(model, "mask_id", 151665))
        inputs_embeds = model.bfg_adapter.apply_to_embeds(
            base_embeds, mask=mask_to_use, prefix_len=P, prev_latents=prev_latents
        )
        kwargs["inputs_embeds"] = inputs_embeds
    else:
        kwargs["input_ids"] = input_ids

    out = model(**kwargs)
    logits = out.logits[:, P - 1 : P + C - 1, :]
    if return_hidden:
        hidden = out.hidden_states[-1][:, P - 1 : P + C - 1, :]
        return logits, hidden
    return logits
