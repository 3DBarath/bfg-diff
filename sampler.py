"""Iterative non-autoregressive canvas decoding, highest Top-1 margin first."""
import torch

from model import MASK_TOKEN_ID, canvas_forward, canvas_step_with_cache, prefill_prefix


@torch.no_grad()
def sample_canvas_block(
    model,
    prefix_ids,
    canvas_len=128,
    steps=8,
    mask_id=MASK_TOKEN_ID,
    prefix_keep=None,
    canvas=None,
    margin_floor=0.0,
    temperature=0.0,
    position_decay=0.0,
    top_k=50,
):
    """Return concrete canvas token ids (B, C). Pass `canvas` pre-filled with mask_id at
    unknown slots to do reconstruction; otherwise the canvas starts fully masked.

    `margin_floor` is an absolute log-prob margin (log p1 - log p2): a slot is only
    unmasked when its top-1 is at least this much more likely than its runner-up.
    Slots below the floor stay masked until a later step (or the final argmax fill).

    `temperature > 0` samples tokens with Gumbel noise from the `top_k` most likely
    logits instead of taking the clean argmax; the margin ranking still uses the
    clean log-probs. `position_decay > 0` multiplies step-1 margins by
    `exp(-lambda * i / C)`, anchoring the canvas start to the prefix.
    """
    assert steps >= 1 and canvas_len >= 1
    B = prefix_ids.shape[0]
    if canvas is None:
        canvas = torch.full(
            (B, canvas_len), mask_id, dtype=torch.long, device=prefix_ids.device
        )
    canvas_len = canvas.shape[1]
    assert canvas.shape == (B, canvas_len)

    decay = None
    if position_decay > 0.0:
        decay = torch.exp(
            -position_decay
            * torch.arange(canvas_len, device=prefix_ids.device).float()
            / canvas_len
        )

    prev_quota = 0
    prefix_len = prefix_ids.shape[1]
    kv = prefill_prefix(model, prefix_ids, prefix_keep)  # prefix encoded once
    for s in range(1, steps + 1):
        logits = canvas_step_with_cache(model, canvas, kv, prefix_len, prefix_keep)
        kv.crop(-canvas_len)  # drop the canvas rows; keep the prefix cache
        assert logits.shape[:2] == (B, canvas_len), logits.shape
        z = logits.float()  # log p(i) - log p(j) == z(i) - z(j): softmax shift cancels
        top2 = z.topk(2, dim=-1)
        margin = top2.values[..., 0] - top2.values[..., 1]  # clean confidence
        score = margin * decay if (decay is not None and s == 1) else margin
        score = score.masked_fill(canvas != mask_id, -float("inf"))
        quota = min(canvas_len, int(s / steps * canvas_len))
        budget = quota - prev_quota
        prev_quota = quota
        eligible = canvas == mask_id
        if margin_floor > 0.0 and s < steps:
            eligible &= margin >= margin_floor

        if temperature > 0.0:
            vals, idx = z.topk(min(top_k or z.shape[-1], z.shape[-1]), dim=-1)
            u = torch.rand_like(vals).clamp(1e-9, 1.0 - 1e-9)
            pick = (vals / temperature - torch.log(-torch.log(u))).argmax(-1, keepdim=True)
            sampled = idx.gather(-1, pick).squeeze(-1)
        else:
            sampled = top2.indices[..., 0]

        for i in range(B):  # per-sample count: a pre-filled canvas may have fewer slots
            n = min(budget, int(eligible[i].sum()))
            if n:
                idx = score[i].topk(n).indices
                canvas[i, idx] = sampled[i, idx]

    leftover = canvas == mask_id  # only if steps/canvas_len rounding left holes
    if leftover.any():
        fill = canvas_forward(model, prefix_ids, canvas, prefix_keep).argmax(-1)
        canvas = torch.where(leftover, fill, canvas)
    return canvas


@torch.no_grad()
def sample_canvas_text(model, tokenizer, prefix_ids, **kwargs):
    """Convenience wrapper: (decoded_text, canvas_ids) for a single prompt."""
    canvas = sample_canvas_block(model, prefix_ids, **kwargs)
    return [tokenizer.decode(c, skip_special_tokens=True) for c in canvas], canvas
