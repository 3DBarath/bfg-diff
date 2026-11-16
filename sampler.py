"""Iterative non-autoregressive canvas decoding, highest Top-1 margin first."""
import math

import torch

from model import MASK_TOKEN_ID, canvas_forward, canvas_step_with_cache, mask_position_ids, prefill_prefix


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
    remask_ratio=0.0,
    suppress_adjacent=True,
    carrier_priority=True,
    schedule="linear",
    anchor_thresh=2.5,
    kv=None,
    step_callback=None,
    stop_token_ids=None,
    formula=None,
    tau=0.8,
    gamma=0.5,
    settle_margin=2.0,
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

    `remask_ratio > 0` lets the weakest fraction of already-decoded slots re-enter
    the competition: they lose their protected status and can be reset to the mask
    if a still-masked slot outscores them. After step `s` the canvas holds
    `n_frozen + floor(s/K * (C - n_frozen))` unmasked slots, so pre-filled slots
    never starve the schedule. Re-masking changes which slots hold the quota, not
    how many are held.

    `suppress_adjacent` removes adjacent identical tokens. Two mechanisms:
    cross-step logit repulsion (an open slot next to a committed token has that
    token removed from its logits, so even the final step picks a distinct
    continuation) and same-step margin deferral (two open neighbours that propose
    the same token hold the weaker one back for a step). Neither revokes a
    committed token.
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

    frozen = canvas != mask_id  # pre-filled input slots: never touched
    decoded = torch.zeros_like(frozen)  # slots this sampler has filled (draft or verified)
    verified = frozen.clone()  # slots with verified carriers; permanent
    prefix_len = prefix_ids.shape[1]
    if kv is None:
        kv = prefill_prefix(model, prefix_ids, prefix_keep)  # prefix[:-1] encoded once
    for s in range(1, steps + 1):
        prev_canvas = canvas.clone()
        prev_verified = verified.clone()

        # 1. Construct feed_canvas with verified tokens + current 1-token lookahead draft only.
        # Distant slots beyond the active lookahead draft are kept masked to avoid polluting self-attention.
        feed_canvas = torch.full_like(canvas, mask_id)
        for i in range(B):
            m_front = 0
            while m_front < canvas_len and verified[i, m_front]:
                m_front += 1
            feed_canvas[i, :m_front] = canvas[i, :m_front]
            # Retain active speculative draft at m_front if already drafted from previous step
            if m_front < canvas_len and canvas[i, m_front] != mask_id:
                feed_canvas[i, m_front] = canvas[i, m_front]

        if kv is None:
            logits = canvas_forward(model, prefix_ids, feed_canvas, prefix_keep)
        else:
            logits = canvas_step_with_cache(model, prefix_ids[:, -1:], feed_canvas, kv, prefix_len, prefix_keep)
            kv.crop(-(1 + canvas_len))
        assert logits.shape[:2] == (B, canvas_len), logits.shape
        z = logits.float()
        top2 = z.topk(2, dim=-1)
        margin = top2.values[..., 0] - top2.values[..., 1]

        if temperature > 0.0:
            vals, idx = z.topk(min(top_k or z.shape[-1], z.shape[-1]), dim=-1)
            u = torch.rand_like(vals).clamp(1e-9, 1.0 - 1e-9)
            pick = (vals / temperature - torch.log(-torch.log(u))).argmax(-1, keepdim=True)
            sampled = idx.gather(-1, pick).squeeze(-1)
        else:
            sampled = top2.indices[..., 0]

        for i in range(B):
            row = canvas[i].clone()

            # Find first unverified slot m
            m = 0
            while m < canvas_len and verified[i, m]:
                m += 1
            if m >= canvas_len:
                continue

            # 1. Slot m is anchored by verified carrier m-1 and is verified
            new_m = int(sampled[i, m])
            prev_tok = int(row[m - 1]) if m > 0 else (int(prefix_ids[i, -1]) if prefix_len else None)
            if suppress_adjacent and prev_tok is not None and new_m == prev_tok:
                cand_z = z[i, m].clone()
                cand_z[prev_tok] = -float("inf")
                new_m = int(cand_z.argmax())

            row[m] = new_m
            verified[i, m] = True

            # Check for early exit on stop tokens
            if stop_token_ids and new_m in stop_token_ids:
                break

            # 2. Speculative Leap: Verify slot m+1 via NCM Energy Bound (tau = 0.8)
            if m + 1 < canvas_len:
                draft_c1 = int(feed_canvas[i, m + 1])
                top_z, top_tok = z[i, m + 1].max(dim=-1)
                if draft_c1 != mask_id and draft_c1 != new_m:
                    draft_z = z[i, m + 1, draft_c1]
                    accepted = False
                    if draft_c1 == int(top_tok):
                        accepted = True
                    elif (top_z - draft_z).item() <= (tau if formula == "formula1" else 0.8):
                        accepted = True

                    if accepted:
                        row[m + 1] = draft_c1
                        verified[i, m + 1] = True
                        m += 1
                        if stop_token_ids and draft_c1 in stop_token_ids:
                            canvas[i] = row
                            break

            # 3. Clean up unverified tail beyond m and draft next lookahead token at m+1 if confident
            for j in range(m + 1, canvas_len):
                row[j] = mask_id

            if m + 1 < canvas_len:
                top2_c1 = z[i, m + 1].topk(2).values
                margin_c1 = (top2_c1[0] - top2_c1[1]).item()
                if margin_c1 >= 1.0:
                    draft_tok = int(sampled[i, m + 1])
                    if suppress_adjacent and draft_tok == int(row[m]):
                        cand_z = z[i, m + 1].clone()
                        cand_z[int(row[m])] = -float("inf")
                        draft_tok = int(cand_z.argmax())
                    if draft_tok != int(row[m]):
                        row[m + 1] = draft_tok

            canvas[i] = row
            decoded[i] = (row != mask_id) & ~frozen[i]

        # Rule 3: Only display verified tokens + active frontier draft in the UI trace (no visual flicker)
        if step_callback is not None:
            display_canvas = canvas.clone()
            for i in range(B):
                # Mask out any non-contiguous wild guesses: keep only verified and immediately adjacent draft
                m_frontier = 0
                while m_frontier < canvas_len and verified[i, m_frontier]:
                    m_frontier += 1
                for j in range(m_frontier + 2, canvas_len):
                    display_canvas[i, j] = mask_id
            step_callback(s, display_canvas)

        # Early exit 1: All tokens verified in this sub-block
        if verified.all():
            break

        # Early exit 2: Stop token verified in this sub-block
        if stop_token_ids:
            stop_set = set(stop_token_ids)
            if any(int(canvas[i, j]) in stop_set for i in range(B) for j in range(canvas_len) if verified[i, j]):
                break

        # Early exit 3: Formula 5 (Sub-Block Frontier Settling)
        # Early exit from sample_canvas_block prior to step K when all active unverified slots satisfy min margin >= settle_margin
        if formula == "formula5" and s < steps:
            settled = True
            for i in range(B):
                m_chk = 0
                while m_chk < canvas_len and verified[i, m_chk]:
                    m_chk += 1
                if m_chk < canvas_len:
                    # check all remaining slots up to canvas_len
                    active_margins = margin[i, m_chk:]
                    if (active_margins < settle_margin).any():
                        settled = False
                        break
            if settled:
                # Accept all settled positions
                for i in range(B):
                    m_chk = 0
                    while m_chk < canvas_len and verified[i, m_chk]:
                        m_chk += 1
                    for j in range(m_chk, canvas_len):
                        canvas[i, j] = sampled[i, j]
                        verified[i, j] = True
                break

    # When reconstructing or infilling (pre-filled canvas with frozen tokens), return full canvas_len.
    # When free-generating across sliding sub-blocks, truncate block to only verified tokens so
    # unverified mask holes or hallucinated argmax guesses never leak across sub-block boundaries!
    if frozen.any():
        leftover = canvas == mask_id
        if leftover.any():
            fill = canvas_forward(model, prefix_ids, canvas, prefix_keep).argmax(-1)
            canvas = torch.where(leftover, fill, canvas)
        return canvas

    m_end = 0
    while m_end < canvas_len and verified[0, m_end]:
        m_end += 1

    if m_end == 0:
        # Fallback: at least slot 0 is verified from step 1
        m_end = 1

    return canvas[:, :m_end]


def sample_canvas(
    model,
    prefix_ids,
    canvas_len=32,
    sub_canvas_len=None,
    steps=8,
    sub_steps=None,
    prefix_keep=None,
    carrier_priority=True,
    block_callback=None,
    stop_token_ids=None,
    **kwargs,
):
    """Generate canvas_len tokens either in one block or sliding through sub-blocks.

    When `sub_canvas_len` is provided (e.g. sub_canvas_len=16 or 8), splits the generation
    into successive sub-blocks. Each completed sub-block is appended to the prefix context,
    providing strong causal anchors while taking fewer FLOPs per step.
    """
    if sub_canvas_len is None or sub_canvas_len >= canvas_len:
        return sample_canvas_block(
            model,
            prefix_ids,
            canvas_len=canvas_len,
            steps=steps,
            prefix_keep=prefix_keep,
            carrier_priority=carrier_priority,
            **kwargs,
        )

    k_sub = sub_steps if sub_steps is not None else max(1, math.ceil(steps * (sub_canvas_len / canvas_len)))
    cur_prefix = prefix_ids
    cur_keep = prefix_keep
    generated_blocks = []
    rem = canvas_len

    # Pre-encode prefix[: -1] into KV cache once for all sub-blocks
    kv = prefill_prefix(model, cur_prefix, cur_keep)

    block_idx = 0
    while rem > 0:
        c_sub = min(sub_canvas_len, rem)

        def step_cb(s, canvas_tensor):
            if block_callback is not None:
                block_callback(block_idx, s, canvas_tensor)

        block = sample_canvas_block(
            model,
            cur_prefix,
            canvas_len=c_sub,
            steps=k_sub,
            prefix_keep=cur_keep,
            carrier_priority=carrier_priority,
            kv=kv,
            step_callback=step_cb,
            stop_token_ids=stop_token_ids,
            **kwargs,
        )
        c_actual = block.shape[1]
        generated_blocks.append(block)
        block_idx += 1
        rem -= c_actual

        # Early exit across sub-blocks if a stop token was generated
        if stop_token_ids:
            stop_set = set(stop_token_ids)
            if any(int(tok) in stop_set for tok in block[0]):
                break

        if rem > 0 and kv is not None:
            # Advance KV cache by appending [cur_prefix[:, -1:], block[:, :-1]] (exactly c_actual tokens).
            # This advances the cached prefix up to the new prefix[:-1] for the next sub-block,
            # leaving block[:, -1:] as prefix_last_id!
            advance_tokens = torch.cat([cur_prefix[:, -1:], block[:, :-1]], dim=1)
            p_cached = kv.get_seq_length()
            new_prefix = torch.cat([cur_prefix, block], dim=1)
            new_keep = torch.cat([cur_keep, torch.ones_like(block, dtype=torch.bool)], dim=1) if cur_keep is not None else None
            pos_advance = (
                torch.arange(p_cached, p_cached + c_actual, device=advance_tokens.device).unsqueeze(0).expand(cur_prefix.shape[0], c_actual)
                if new_keep is None
                else mask_position_ids(new_keep, 0)[:, p_cached : p_cached + c_actual]
            )
            attn_mask = new_keep[:, : p_cached + c_actual] if new_keep is not None else None
            model(
                input_ids=advance_tokens,
                past_key_values=kv,
                position_ids=pos_advance,
                attention_mask=attn_mask,
                use_cache=True,
            )

        cur_prefix = torch.cat([cur_prefix, block], dim=1)
        if cur_keep is not None:
            cur_keep = torch.cat([cur_keep, torch.ones_like(block, dtype=torch.bool)], dim=1)

    return torch.cat(generated_blocks, dim=1)


@torch.no_grad()
def sample_canvas_text(model, tokenizer, prefix_ids, **kwargs):
    """Convenience wrapper: (decoded_text, canvas_ids) for a single prompt."""
    canvas = sample_canvas(model, prefix_ids, **kwargs)
    return [tokenizer.decode(c, skip_special_tokens=True) for c in canvas], canvas
