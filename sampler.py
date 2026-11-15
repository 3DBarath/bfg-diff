"""Iterative non-autoregressive canvas decoding, highest Top-1 margin first."""
import torch
import torch.nn.functional as F

from model import MASK_TOKEN_ID, canvas_forward


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
):
    """Return concrete canvas token ids (B, C). Pass `canvas` pre-filled with mask_id at
    unknown slots to do reconstruction; otherwise the canvas starts fully masked.

    `margin_floor` is an absolute log-prob margin (log p1 - log p2): a slot is only
    unmasked when its top-1 is at least this much more likely than its runner-up.
    Slots below the floor stay masked until a later step (or the final argmax fill).
    """
    assert steps >= 1 and canvas_len >= 1
    B = prefix_ids.shape[0]
    if canvas is None:
        canvas = torch.full(
            (B, canvas_len), mask_id, dtype=torch.long, device=prefix_ids.device
        )
    canvas_len = canvas.shape[1]
    assert canvas.shape == (B, canvas_len)

    prev_quota = 0
    for s in range(1, steps + 1):
        logits = canvas_forward(model, prefix_ids, canvas, prefix_keep)
        assert logits.shape[:2] == (B, canvas_len), logits.shape
        lp = F.log_softmax(logits.float(), dim=-1)
        top2 = lp.topk(2, dim=-1)
        margin = (top2.values[..., 0] - top2.values[..., 1]).masked_fill(
            canvas != mask_id, -float("inf")
        )
        quota = min(canvas_len, int(s / steps * canvas_len))
        budget = quota - prev_quota
        prev_quota = quota
        eligible = canvas == mask_id
        if margin_floor > 0.0 and s < steps:
            eligible &= margin >= margin_floor
        for i in range(B):  # per-sample count: a pre-filled canvas may have fewer slots
            n = min(budget, int(eligible[i].sum()))
            if n:
                idx = margin[i].topk(n).indices
                canvas[i, idx] = top2.indices[i, idx, 0]

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
