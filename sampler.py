"""Iterative non-autoregressive canvas decoding, highest Top-1 margin first."""
import math

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
    remask_ratio=0.0,
    suppress_adjacent=True,
    carrier_priority=True,
    schedule="linear",
    anchor_thresh=4.0,
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
    decoded = torch.zeros_like(frozen)  # slots this sampler has filled
    prefix_len = prefix_ids.shape[1]
    kv = prefill_prefix(model, prefix_ids, prefix_keep)  # prefix[:-1] encoded once
    for s in range(1, steps + 1):
        if kv is None:
            logits = canvas_forward(model, prefix_ids, canvas, prefix_keep)
        else:
            logits = canvas_step_with_cache(model, prefix_ids[:, -1:], canvas, kv, prefix_len, prefix_keep)
            kv.crop(-(1 + canvas_len))  # drop active [prefix_last, canvas] tokens; keep prefix cache
        assert logits.shape[:2] == (B, canvas_len), logits.shape
        z = logits.float()  # log p(i) - log p(j) == z(i) - z(j): softmax shift cancels
        top2 = z.topk(2, dim=-1)
        margin = top2.values[..., 0] - top2.values[..., 1]  # clean confidence

        # Slots whose existing token survives: frozen input plus the confident
        # fraction of already-decoded slots. The rest of `decoded` competes again.
        # On the last step every decoded token is an anchor.
        keep_token = frozen.clone()
        if s < steps:
            for i in range(B):
                idx = decoded[i].nonzero().flatten()
                if idx.numel():
                    n = math.ceil((1.0 - remask_ratio) * idx.numel())
                    keep_token[i, idx[margin[i, idx].topk(n).indices]] = True
        else:
            keep_token |= decoded

        # Cross-step repulsion: an open slot next to a committed token cannot
        # predict that token, so its top-1 is already a distinct continuation and
        # its margin is measured between the two best *valid* choices. Without
        # this, holding the slot back as [MASK] just delays the stutter.
        if suppress_adjacent:
            left = keep_token[:, :-1]
            right = keep_token[:, 1:]
            idx = torch.where(left, canvas[:, :-1], torch.zeros_like(canvas[:, :-1]))
            rep = z[:, 1:].scatter(-1, idx.unsqueeze(-1), -float("inf"))
            z[:, 1:] = torch.where(left.unsqueeze(-1), rep, z[:, 1:])
            idx = torch.where(right, canvas[:, 1:], torch.zeros_like(canvas[:, 1:]))
            rep = z[:, :-1].scatter(-1, idx.unsqueeze(-1), -float("inf"))
            z[:, :-1] = torch.where(right.unsqueeze(-1), rep, z[:, :-1])
            if prefix_len:
                last = prefix_ids[:, -1:]
                rep = z[:, :1].scatter(-1, last.unsqueeze(-1), -float("inf"))
                z[:, :1] = torch.where(~keep_token[:, :1].unsqueeze(-1), rep, z[:, :1])
            top2 = z.topk(2, dim=-1)
            margin = top2.values[..., 0] - top2.values[..., 1]

        rank = margin * decay if (decay is not None and s == 1) else margin
        if carrier_priority and s < steps:
            # Right-shifted geometry: slot j's logits sit on top of input token canvas[j-1].
            # For j=0, carrier is prefix[-1] which is always concrete.
            # For j>0, carrier is concrete if keep_token[:, j-1] is True.
            has_carrier = torch.zeros_like(frozen)
            has_carrier[:, 0] = True
            has_carrier[:, 1:] = keep_token[:, :-1] | frozen[:, :-1]
            # Prioritize slots with real carrier, OR strong structural anchors
            anchor = margin >= anchor_thresh
            is_priority = has_carrier | anchor
            # Priority offset (+10.0) ensures carrier-anchored slots unmask before orphan masks
            rank = rank + torch.where(is_priority, 10.0, 0.0)

        if margin_floor > 0.0 and s < steps:
            rank = rank.masked_fill((canvas == mask_id) & (margin < margin_floor), -float("inf"))

        if temperature > 0.0:
            vals, idx = z.topk(min(top_k or z.shape[-1], z.shape[-1]), dim=-1)
            u = torch.rand_like(vals).clamp(1e-9, 1.0 - 1e-9)
            pick = (vals / temperature - torch.log(-torch.log(u))).argmax(-1, keepdim=True)
            sampled = idx.gather(-1, pick).squeeze(-1)
        else:
            sampled = top2.indices[..., 0]

        if s < steps:
            rank = rank.masked_fill(keep_token, float("inf"))

            # Two open neighbours predicting the same token would commit a
            # stutter. Only slots that are still masked can be held back; next
            # step the winner is an anchor, so repulsion above forces the loser
            # onto a distinct continuation.
            if suppress_adjacent:
                eff = torch.where(keep_token, canvas, sampled)  # token that would stand
                same = (eff[:, :-1] == eff[:, 1:]) & ~(frozen[:, :-1] & frozen[:, 1:])
                open_l = ~keep_token[:, :-1]  # still masked, may commit this step
                open_r = ~keep_token[:, 1:]
                ml, mr = margin[:, :-1], margin[:, 1:]
                kill = torch.zeros_like(frozen)
                kill[:, :-1] = same & open_l & (~open_r | (ml < mr))
                kill[:, 1:] = same & open_r & (~open_l | (mr <= ml))
                rank = rank.masked_fill(kill, -float("inf"))

        for i in range(B):
            n_frozen = int(frozen[i].sum())
            if s == steps:  # final step: fill every remaining mask, keep the rest
                row = canvas[i].clone()
                for j in (canvas[i] == mask_id).nonzero().flatten().tolist():
                    tok = int(sampled[i, j])
                    if suppress_adjacent:
                        prev = int(row[j - 1]) if j > 0 else (int(prefix_ids[i, -1]) if prefix_len else None)
                        if prev is not None and tok == prev:
                            cand = z[i, j].clone()
                            cand[prev] = -float("inf")
                            tok = int(cand.argmax())
                    row[j] = tok
            else:
                # Interpolate strictly over the slots this sampler can fill:
                # pre-filled slots hold their own share of the canvas for free.
                n_mutable = canvas_len - n_frozen
                if schedule == "geometric":
                    # Geometric unmasking quota ending cleanly at n_mutable
                    budget = min(n_mutable, max(1, math.ceil(n_mutable * (2.0 ** -(steps - s)))))
                else:
                    # Linear quota
                    budget = min(max(0, int(s / steps * n_mutable)), n_mutable)
                if not budget:
                    continue
                # Protected decoded slots keep their token; the rest of the budget
                # is filled from the strongest remaining slots. Frozen slots never
                # compete: they already hold quota and are restored below.
                keep = keep_token[i] & ~frozen[i]
                need = budget - int(keep.sum())
                pick = set(keep.nonzero().flatten().tolist())

                if need > 0 and carrier_priority:
                    # Carrier-Island Frontier selection:
                    # 1. First, pick all open slots that already have a real carrier (has_carrier == True)
                    has_carrier_i = has_carrier[i]
                    cand_carrier = [
                        j for j in range(canvas_len)
                        if has_carrier_i[j] and j not in pick and not frozen[i, j] and rank[i, j] > -float("inf")
                    ]
                    cand_carrier.sort(key=lambda j: float(rank[i, j]), reverse=True)
                    for j in cand_carrier:
                        if len(pick) < budget:
                            pick.add(j)

                    # 2. If budget still needs slots (parallel unmasking > available carriers),
                    # pick island anchors that are NOT immediately adjacent to newly picked slots
                    # (j - 1 not in pick), so that next step the island becomes a second carrier frontier!
                    while len(pick) < budget:
                        cand_island = [
                            j for j in range(canvas_len)
                            if j not in pick and not keep_token[i, j] and not frozen[i, j]
                            and rank[i, j] > -float("inf")
                            and not (j > 0 and (j - 1) in pick)
                        ]
                        if not cand_island:
                            # Fallback if canvas is tight
                            cand_island = [
                                j for j in range(canvas_len)
                                if j not in pick and not keep_token[i, j] and not frozen[i, j]
                                and rank[i, j] > -float("inf")
                            ]
                        if not cand_island:
                            break
                        cand_island.sort(key=lambda j: float(rank[i, j]), reverse=True)
                        pick.add(cand_island[0])
                    pick_tensor = torch.tensor(sorted(list(pick)), dtype=torch.long, device=canvas.device)
                elif need > 0:
                    cand = (rank[i] > -float("inf")) & ~frozen[i] & ~keep
                    cand = cand.nonzero().flatten()
                    pick_tensor = keep.nonzero().flatten()
                    if cand.numel():
                        order = rank[i, cand].topk(min(need, cand.numel())).indices
                        pick_tensor = torch.cat([pick_tensor, cand[order]])
                else:
                    pick_tensor = keep.nonzero().flatten()

                old = canvas[i].clone()
                row = torch.full_like(old, mask_id)
                row[frozen[i]] = old[frozen[i]]
                if pick_tensor.numel():
                    row[pick_tensor] = torch.where(keep_token[i, pick_tensor], old[pick_tensor], sampled[i, pick_tensor])
            canvas[i] = row
            decoded[i] = (row != mask_id) & ~frozen[i]

    leftover = canvas == mask_id  # only if steps/canvas_len rounding left holes
    if leftover.any():
        fill = canvas_forward(model, prefix_ids, canvas, prefix_keep).argmax(-1)
        canvas = torch.where(leftover, fill, canvas)
    return canvas


def sample_canvas(
    model,
    prefix_ids,
    canvas_len=32,
    sub_canvas_len=None,
    steps=8,
    sub_steps=None,
    prefix_keep=None,
    carrier_priority=True,
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

    while rem > 0:
        c_sub = min(sub_canvas_len, rem)
        block = sample_canvas_block(
            model,
            cur_prefix,
            canvas_len=c_sub,
            steps=k_sub,
            prefix_keep=cur_keep,
            carrier_priority=carrier_priority,
            **kwargs,
        )
        generated_blocks.append(block)
        cur_prefix = torch.cat([cur_prefix, block], dim=1)
        if cur_keep is not None:
            cur_keep = torch.cat([cur_keep, torch.ones_like(block, dtype=torch.bool)], dim=1)
        rem -= c_sub

    return torch.cat(generated_blocks, dim=1)


@torch.no_grad()
def sample_canvas_text(model, tokenizer, prefix_ids, **kwargs):
    """Convenience wrapper: (decoded_text, canvas_ids) for a single prompt."""
    canvas = sample_canvas(model, prefix_ids, **kwargs)
    return [tokenizer.decode(c, skip_special_tokens=True) for c in canvas], canvas
