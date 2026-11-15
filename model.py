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
        # If prefix is only 1 token, there is no past to prefill
        return None
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


def canvas_step_with_cache(model, prefix_last_id, canvas_ids, past_key_values, prefix_len, prefix_keep=None):
    """Logits (B, C, vocab) for canvas positions using cached prefix[:P-1] and active [prefix_last, canvas].

    Slot 0 (canvas[0]) is predicted by prefix_last_id (pos P-1) attending to all prefix and canvas.
    Slot j (canvas[j]) is predicted by canvas[j-1] (pos P-1+j) attending to all prefix and canvas.
    """
    B, C = canvas_ids.shape
    device = canvas_ids.device
    dtype = model.get_input_embeddings().weight.dtype
    active_ids = torch.cat([prefix_last_id, canvas_ids], dim=1)  # length 1 + C
    T = prefix_len + C

    allow = torch.ones(B, 1, 1 + C, T, dtype=torch.bool, device=device)
    if prefix_keep is not None and prefix_len > 1:
        allow[..., : prefix_len - 1] &= prefix_keep[:, : prefix_len - 1].to(device=device, dtype=torch.bool).view(B, 1, 1, prefix_len - 1)

    attn = torch.zeros(B, 1, 1 + C, T, dtype=dtype, device=device).masked_fill_(
        ~allow, torch.finfo(dtype).min
    )
    pos = (
        torch.arange(prefix_len - 1, prefix_len + C, device=device).unsqueeze(0).expand(B, 1 + C)
        if prefix_keep is None
        else mask_position_ids(prefix_keep, C)[:, prefix_len - 1 :]
    )
    out = model(
        input_ids=active_ids,
        attention_mask=attn,
        position_ids=pos,
        past_key_values=past_key_values,
        use_cache=True,
    )
    return out.logits[:, :C, :]


def canvas_forward(model, prefix_ids, canvas_ids, prefix_keep=None):
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
    out = model(input_ids=input_ids, attention_mask=attn, position_ids=pos, use_cache=False)
    return out.logits[:, P - 1 : P + C - 1, :]
