"""Iterative non-autoregressive canvas decoding, highest Top-1 margin first."""
import math

import torch

from model import MASK_TOKEN_ID, canvas_forward, canvas_step_with_cache, mask_position_ids, prefill_prefix


@torch.no_grad()
def sample_staircase_sliding(
    model,
    prefix_ids,
    canvas_len=128,
    mask_id=MASK_TOKEN_ID,
    prefix_keep=None,
    temperature=0.0,
    top_k=50,
    suppress_adjacent=True,
    tau=0.8,
    settle_margin=2.0,
    draft_margin_thresh=0.0,
    offset_thresholds=(0.0, 0.35, 0.75),
    W_max=4,
    V_max=3,
    kv=None,
    step_callback=None,
    stop_token_ids=None,
    stats_out=None,
    enable_branching=False,
    branch_thresh=1.0,
    adapter=None,
    **kwargs,
):
    """Fused Verify-and-Diffuse Staircase Engine (BFG-Diff v2).

    Simultaneously verifies Zone A (1 + V tokens) under strictly causal attention
    and drafts Zone B (W tokens) under bidirectional attention. Progressively slices
    KV cache in-place with zero token recomputations.
    """
    assert canvas_len >= 1
    B = prefix_ids.shape[0]
    device = prefix_ids.device
    cur_prefix_len = prefix_ids.shape[1]

    if adapter is None:
        adapter = getattr(model, "bfg_adapter", None)

    if kv is None:
        kv = prefill_prefix(model, prefix_ids, prefix_keep)

    if stats_out is not None:
        if "forward_passes" not in stats_out:
            stats_out["forward_passes"] = 0
            stats_out["active_tokens_evaluated"] = 0
            stats_out["tokens_verified"] = 0
            stats_out["accept_histogram"] = {}

    cur_last_id = prefix_ids[:, -1:]
    draft_ids = None
    tree_branches = None  # (b1_tokens, b2_tokens) if 2-branch micro-tree active
    committed_tokens = []
    prev_latents = None
    prev_pred_embeds = None

    step = 0
    while len(committed_tokens) < canvas_len:
        step += 1
        rem_tokens = canvas_len - len(committed_tokens)

        tree_mask_a = None
        pos_override = None

        if tree_branches is not None and rem_tokens >= 2:
            # Ambiguous Fork Regime: 2-Branch Zone A of length 5
            b1_toks, b2_toks = tree_branches
            zone_a_ids = torch.cat([cur_last_id, b1_toks, b2_toks], dim=1) # (B, 5)
            zone_a_len = 5
            tree_mask_a = torch.tensor([
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 0, 0, 1, 0],
                [1, 0, 0, 1, 1],
            ], dtype=torch.bool, device=device)

            W = min(W_max, max(1, rem_tokens - 2))
            pad_count = prefix_keep.shape[1] - prefix_keep.to(device=device, dtype=torch.long).sum(dim=1, keepdim=True) if prefix_keep is not None else 0
            base_p = cur_prefix_len - 1 - pad_count
            pos_a = base_p + torch.tensor([[0, 1, 2, 1, 2]], device=device).expand(B, -1)
            pos_b = (base_p + 3) + torch.arange(W, device=device).unsqueeze(0).expand(B, -1)
            pos_override = torch.cat([pos_a, pos_b], dim=1)
            is_tree_step = True
        else:
            is_tree_step = False
            V_cur = draft_ids.shape[1] if (draft_ids is not None and draft_ids.numel() > 0) else 0
            if V_cur >= rem_tokens:
                V_cur = max(0, rem_tokens - 1)
                draft_ids = draft_ids[:, :V_cur] if V_cur > 0 else None

            # 1. Zone A: [cur_last_id, draft_ids] of length 1 + V_cur
            if V_cur > 0:
                zone_a_ids = torch.cat([cur_last_id, draft_ids], dim=1)
            else:
                zone_a_ids = cur_last_id
            zone_a_len = 1 + V_cur
            W = min(W_max, max(1, rem_tokens - V_cur))

        # 2. Zone B: W masks
        zone_b_ids = torch.full((B, W), mask_id, dtype=torch.long, device=device)

        # 3. Active IDs
        active_ids = torch.cat([zone_a_ids, zone_b_ids], dim=1)
        Q = active_ids.shape[1]

        inputs_embeds = None
        if adapter is not None and hasattr(adapter, "get_inputs_embeds"):
            inputs_embeds = adapter.get_inputs_embeds(
                model, active_ids, zone_a_len, W, prev_latents=prev_latents, prev_pred_embeds=prev_pred_embeds
            )

        if stats_out is not None:
            stats_out["forward_passes"] += 1
            stats_out["active_tokens_evaluated"] += Q

        return_hidden = (adapter is not None)
        out_step = canvas_step_with_cache(
            model,
            active_ids,
            past_key_values=kv,
            prefix_len=cur_prefix_len,
            zone_a_len=zone_a_len,
            prefix_keep=prefix_keep,
            tree_mask_a=tree_mask_a,
            position_ids_override=pos_override,
            return_hidden=return_hidden,
            inputs_embeds=inputs_embeds,
            prev_latents=prev_latents,
            prev_pred_embeds=prev_pred_embeds,
        )
        if return_hidden and isinstance(out_step, tuple):
            logits, latents = out_step
        else:
            logits = out_step
            latents = None

        z = logits.float()

        # Repetition penalty over recent context
        if suppress_adjacent and (cur_prefix_len > 0 or len(committed_tokens) > 0):
            hist = []
            if cur_prefix_len > 0:
                hist.append(prefix_ids[0, max(0, cur_prefix_len - 128):])
            if len(committed_tokens) > 0:
                hist.append(torch.tensor(committed_tokens, device=device))
            if hist:
                recent = torch.cat(hist)
                valid_recent = recent[recent != mask_id]
                counts = torch.bincount(valid_recent, minlength=z.shape[-1])
                repeated = counts > 0
                if repeated.any():
                    pen = counts[repeated].float() * 0.35
                    for qa in range(zone_a_len):
                        z[:, qa, repeated] -= pen

        # Anti-stutter for slot 0 against cur_last_id
        if suppress_adjacent:
            prev_tok = int(cur_last_id[0, 0])
            z[:, 0, prev_tok] = -float("inf")

        # 5. Verify Zone A
        accepted_drafts = []
        r = 0
        if is_tree_step:
            # 2-Branch verification
            b1_toks, b2_toks = tree_branches
            y_star_0 = int(z[0, 0].argmax()) # true causal prediction for pos P
            # Branch 1 candidates: index 1 and 2
            b1_0 = int(b1_toks[0, 0])
            b1_1 = int(b1_toks[0, 1])
            # Branch 2 candidates: index 3 and 4
            b2_0 = int(b2_toks[0, 0])
            b2_1 = int(b2_toks[0, 1])

            # Check Branch 1 streak
            b1_match0 = (b1_0 == y_star_0) or ((z[0, 0, y_star_0] - z[0, 0, b1_0]).item() <= tau)
            b1_streak = 0
            if b1_match0:
                b1_streak = 1
                y_star_b1_1 = int(z[0, 1].argmax())
                if (b1_1 == y_star_b1_1) or ((z[0, 1, y_star_b1_1] - z[0, 1, b1_1]).item() <= tau):
                    b1_streak = 2

            # Check Branch 2 streak
            b2_match0 = (b2_0 == y_star_0) or ((z[0, 0, y_star_0] - z[0, 0, b2_0]).item() <= tau)
            b2_streak = 0
            if b2_match0:
                b2_streak = 1
                y_star_b2_1 = int(z[0, 3].argmax())
                if (b2_1 == y_star_b2_1) or ((z[0, 3, y_star_b2_1] - z[0, 3, b2_1]).item() <= tau):
                    b2_streak = 2

            # Select winner
            if b2_streak > b1_streak:
                winner = 2
                streak = b2_streak
            elif b1_streak > 0:
                winner = 1
                streak = b1_streak
            else:
                winner = 0
                streak = 0

            if winner == 2:
                # Branch 2 won! Copy Branch 2 KV states [3, 4] into [1, 2]
                p_offset = cur_prefix_len - 1
                for layer in kv.layers:
                    layer.keys[:, :, p_offset + 1 : p_offset + 3, :] = layer.keys[:, :, p_offset + 3 : p_offset + 5, :].clone()
                    layer.values[:, :, p_offset + 1 : p_offset + 3, :] = layer.values[:, :, p_offset + 3 : p_offset + 5, :].clone()
                if streak == 2:
                    bonus_tok = int(z[0, 4].argmax())
                    new_tokens = [b2_0, b2_1, bonus_tok]
                else:
                    override_tok = int(z[0, 3].argmax())
                    new_tokens = [b2_0, override_tok]
            elif winner == 1:
                if streak == 2:
                    bonus_tok = int(z[0, 2].argmax())
                    new_tokens = [b1_0, b1_1, bonus_tok]
                else:
                    override_tok = int(z[0, 1].argmax())
                    new_tokens = [b1_0, override_tok]
            else:
                new_tokens = [y_star_0]

            k = len(new_tokens)
            new_last_id = torch.full((B, 1), new_tokens[-1], dtype=torch.long, device=device)
            r = streak
            V_cur = 2
            tree_branches = None
        elif V_cur == 0:
            if temperature > 0.0:
                k_eval = min(top_k or z.shape[-1], z.shape[-1])
                vals, idx = z[:, 0].topk(k_eval, dim=-1)
                u = torch.rand_like(vals).clamp(1e-9, 1.0 - 1e-9)
                pick = (vals / temperature - torch.log(-torch.log(u))).argmax(-1, keepdim=True)
                y_0 = idx.gather(-1, pick).squeeze(-1)
            else:
                y_0 = z[:, 0].argmax(dim=-1)
            new_tokens = [int(y_0[0])]
            new_last_id = y_0.view(B, 1)
            k = 1
        else:
            curr_prev = int(cur_last_id[0, 0])
            # Vectorized evaluation for Zone A drafts
            d_toks = draft_ids[0, :V_cur]
            z_eval = z[0, :V_cur]  # (V_cur, vocab)
            top_toks = z_eval.argmax(dim=-1)  # (V_cur,)
            top_z = z_eval.gather(1, top_toks.unsqueeze(1)).squeeze(1)
            draft_z = z_eval.gather(1, d_toks.unsqueeze(1)).squeeze(1)

            exact_match = (d_toks == top_toks)
            margin_ok = (top_z - draft_z) <= tau
            if suppress_adjacent:
                prev_check = torch.cat([cur_last_id[0], d_toks[:-1]])
                no_stutter = (d_toks != prev_check)
                accept_mask = exact_match | (margin_ok & no_stutter)
            else:
                accept_mask = exact_match | margin_ok

            # First false index is the rejection point
            rejection_indices = torch.where(~accept_mask)[0]
            if len(rejection_indices) == 0:
                # All accepted
                accepted_drafts = d_toks.tolist()
                r = V_cur
                top_bonus = int(z[0, V_cur].argmax())
                if suppress_adjacent and top_bonus == (accepted_drafts[-1] if accepted_drafts else curr_prev):
                    cz = z[0, V_cur].clone()
                    cz[accepted_drafts[-1] if accepted_drafts else curr_prev] = -float("inf")
                    top_bonus = int(cz.argmax())
                new_tokens = accepted_drafts + [top_bonus]
                new_last_id = torch.full((B, 1), top_bonus, dtype=torch.long, device=device)
                k = V_cur + 1
            else:
                rej_j = rejection_indices[0].item()
                accepted_drafts = d_toks[:rej_j].tolist()
                r = rej_j
                override_tok = int(top_toks[rej_j].item())
                prev_tok_here = accepted_drafts[-1] if accepted_drafts else curr_prev
                if suppress_adjacent and override_tok == prev_tok_here:
                    cz = z[0, rej_j].clone()
                    cz[prev_tok_here] = -float("inf")
                    override_tok = int(cz.argmax())
                new_tokens = accepted_drafts + [override_tok]
                new_last_id = torch.full((B, 1), override_tok, dtype=torch.long, device=device)
                k = r + 1

        if k > rem_tokens:
            k = rem_tokens
            new_tokens = new_tokens[:k]
            new_last_id = torch.full((B, 1), new_tokens[-1], dtype=torch.long, device=device)

        # 6. In-Place KV Cache Slice
        kv.crop(-(Q - k))
        cur_prefix_len += k
        committed_tokens.extend(new_tokens)
        cur_last_id = new_last_id

        if stats_out is not None:
            stats_out["accept_histogram"][k] = stats_out["accept_histogram"].get(k, 0) + 1

        # Check early exit on stop tokens
        stop_hit = False
        if stop_token_ids:
            stop_set = set(stop_token_ids)
            for idx, tok in enumerate(new_tokens):
                if tok in stop_set:
                    cut = len(committed_tokens) - len(new_tokens) + idx
                    committed_tokens = committed_tokens[:cut]
                    stop_hit = True
                    break

        if stop_hit or len(committed_tokens) >= canvas_len:
            if step_callback:
                disp = torch.tensor([committed_tokens], device=device)
                step_callback(step, disp)
            break

        # 7. Extract new drafts from Zone B (Multi-Exit Lookahead Drafting)
        draft_ids = None
        tree_branches = None
        if not stop_hit and (canvas_len - len(committed_tokens)) >= 1:
            if V_cur == r:
                # Primary Zone B lookahead: full Zone A was accepted
                zone_b_z = z[0, zone_a_len : zone_a_len + W]
                top2_0 = zone_b_z[0].topk(2)
                margin_0 = (top2_0.values[0] - top2_0.values[1]).item()

                if enable_branching and margin_0 < branch_thresh and W >= 2 and (canvas_len - len(committed_tokens)) >= 2:
                    # Fork 2-Branch Zone A for next step
                    b1_tok0 = int(top2_0.indices[0])
                    b2_tok0 = int(top2_0.indices[1])
                    top2_1 = zone_b_z[1].topk(2)
                    b1_tok1 = int(top2_1.indices[0])
                    b2_tok1 = int(top2_1.indices[1])
                    b1_t = torch.tensor([[b1_tok0, b1_tok1]], dtype=torch.long, device=device).expand(B, -1)
                    b2_t = torch.tensor([[b2_tok0, b2_tok1]], dtype=torch.long, device=device).expand(B, -1)
                    tree_branches = (b1_t, b2_t)
                else:
                    candidate_drafts = []
                    c_prev = int(cur_last_id[0, 0])
                    # Vectorized extraction of top-2 tokens and margins for Zone B lookahead
                    W_eval = min(V_max, W)
                    top2_all = zone_b_z[:W_eval].topk(2, dim=-1)  # values: (W_eval, 2), indices: (W_eval, 2)
                    top_vals = top2_all.values
                    top_inds = top2_all.indices
                    margins = top_vals[:, 0] - top_vals[:, 1]
                    best_toks = top_inds[:, 0]
                    alt_toks = top_inds[:, 1]

                    for jb in range(W_eval):
                        tok_b = int(best_toks[jb].item())
                        if suppress_adjacent and tok_b == c_prev:
                            tok_b = int(alt_toks[jb].item())
                        thresh = offset_thresholds[jb] if jb < len(offset_thresholds) else 0.75
                        eff_thresh = max(draft_margin_thresh, thresh) if draft_margin_thresh > 0.0 else thresh
                        if jb == 0 or margins[jb].item() >= eff_thresh:
                            candidate_drafts.append(tok_b)
                            c_prev = tok_b
                        else:
                            break
                    if candidate_drafts:
                        draft_ids = torch.tensor([candidate_drafts], dtype=torch.long, device=device).expand(B, -1)
                        if latents is not None:
                            prev_latents = latents[:, zone_a_len : zone_a_len + len(candidate_drafts)]
                        else:
                            prev_latents = None
                        # Sigma SC: extract predicted embeddings for Zone B candidate drafts
                        emb_fn = model.get_input_embeddings()
                        prev_pred_embeds = emb_fn(draft_ids).detach()
                    else:
                        prev_latents = None
                        prev_pred_embeds = None
            else:
                # Partial rejection fallback: r < V_cur.
                # Avoid resetting V_cur = 0 by extracting lookahead draft from Zone B slot 0
                zone_b_z = z[0, zone_a_len : zone_a_len + W]
                top2_0 = zone_b_z[0].topk(2)
                margin_0 = (top2_0.values[0] - top2_0.values[1]).item()
                c_prev = int(cur_last_id[0, 0])
                fb_tok = int(top2_0.indices[0])
                if suppress_adjacent and fb_tok == c_prev:
                    fb_tok = int(top2_0.indices[1])
                # Always draft at least 1 token if margin >= 0.25 (or if margin >= 0.0 to prevent ping-pong reset)
                if margin_0 >= 0.20 or W > 0:
                    draft_ids = torch.tensor([[fb_tok]], dtype=torch.long, device=device).expand(B, -1)
                    if latents is not None and latents.shape[1] > zone_a_len:
                        prev_latents = latents[:, zone_a_len : zone_a_len + 1]
                    else:
                        prev_latents = None
                    emb_fn = model.get_input_embeddings()
                    prev_pred_embeds = emb_fn(draft_ids).detach()
                else:
                    prev_latents = None
                    prev_pred_embeds = None
        else:
            prev_latents = None
            prev_pred_embeds = None

        if step_callback:
            disp_tokens = list(committed_tokens)
            if draft_ids is not None:
                disp_tokens.extend(draft_ids[0].tolist())
            elif tree_branches is not None:
                disp_tokens.extend(tree_branches[0][0].tolist())
            disp = torch.tensor([disp_tokens], device=device).expand(B, -1)
            step_callback(step, disp)

    ret = torch.tensor([committed_tokens], dtype=torch.long, device=device).expand(B, -1) if committed_tokens else torch.empty((B, 0), dtype=torch.long, device=device)
    if stats_out is not None:
        stats_out["tokens_verified"] += ret.shape[1]
        fp = max(1, stats_out["forward_passes"])
        tv = max(1, stats_out["tokens_verified"])
        stats_out["alpha"] = stats_out["tokens_verified"] / fp
        stats_out["r_kv"] = (stats_out["active_tokens_evaluated"] / tv) - 1.0

    return ret


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
    stats_out=None,
    **kwargs,
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
    frozen = (canvas != mask_id) if canvas is not None else None
    if frozen is None or not frozen.any():
        # Free-generation: use high-efficiency fused staircase engine with in-place KV slicing
        return sample_staircase_sliding(
            model,
            prefix_ids,
            canvas_len=canvas_len,
            mask_id=mask_id,
            prefix_keep=prefix_keep,
            temperature=temperature,
            top_k=top_k,
            suppress_adjacent=suppress_adjacent,
            kv=kv,
            step_callback=step_callback,
            stop_token_ids=stop_token_ids,
            stats_out=stats_out,
            tau=tau,
            settle_margin=settle_margin,
            **kwargs,
        )

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
    if stats_out is not None:
        if "forward_passes" not in stats_out:
            stats_out["forward_passes"] = 0
            stats_out["active_tokens_evaluated"] = 0
            stats_out["tokens_verified"] = 0
            stats_out["accept_histogram"] = {}
    for s in range(1, steps + 1):
        prev_canvas = canvas.clone()
        prev_verified = verified.clone()
        n_ver_before = int(verified[0].sum().item())

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
            act_eval = feed_canvas.shape[1] + prefix_ids.shape[1]
        else:
            logits = canvas_step_with_cache(model, prefix_ids[:, -1:], feed_canvas, kv, prefix_len, prefix_keep)
            kv.crop(-(1 + canvas_len))
            act_eval = 1 + canvas_len
        if stats_out is not None:
            stats_out["forward_passes"] += 1
            stats_out["active_tokens_evaluated"] += act_eval
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

            # Strict boundary anti-echo at slot 0 against last token of prefix before selecting slot 0
            if m == 0 and prefix_len > 0 and suppress_adjacent:
                last_prefix_tok = int(prefix_ids[i, -1])
                z[i, 0, last_prefix_tok] = -float("inf")

            # Soft frequency/presence penalty over prefix history + current canvas (applied before sampling)
            if suppress_adjacent and (prefix_len > 0 or m > 0):
                hist_tokens = []
                if prefix_len > 0:
                    hist_tokens.append(prefix_ids[i, max(0, prefix_len - 128):])
                if m > 0:
                    hist_tokens.append(row[:m])
                if hist_tokens:
                    recent = torch.cat(hist_tokens)
                    valid_recent = recent[recent != mask_id]
                    counts = torch.bincount(valid_recent, minlength=z.shape[-1])
                    repeated = counts > 0
                    if repeated.any():
                        z[i, m, repeated] -= (counts[repeated].float() * 0.35)

            # Sample slot m with temperature/top_k or greedy argmax after penalty
            if temperature > 0.0:
                k_eval = min(top_k or z.shape[-1], z.shape[-1])
                vals, idx = z[i, m].topk(k_eval)
                u = torch.rand_like(vals).clamp(1e-9, 1.0 - 1e-9)
                pick = (vals / temperature - torch.log(-torch.log(u))).argmax()
                new_m = int(idx[pick])
            else:
                new_m = int(z[i, m].argmax())

            # 1. Slot m is anchored by verified carrier m-1 and is verified
            prev_tok = int(row[m - 1]) if m > 0 else (int(prefix_ids[i, -1]) if prefix_len else None)
            if suppress_adjacent and prev_tok is not None and new_m == prev_tok:
                cand_z = z[i, m].clone()
                cand_z[prev_tok] = -float("inf")
                new_m = int(cand_z.argmax())

            row[m] = new_m
            verified[i, m] = True

            # Check for early exit on stop tokens
            if stop_token_ids and new_m in stop_token_ids:
                # Clear unverified slots beyond m to mask_id
                for j in range(m + 1, canvas_len):
                    row[j] = mask_id
                canvas[i] = row
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
                            for j in range(m + 1, canvas_len):
                                row[j] = mask_id
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

        n_ver_after = int(verified[0].sum().item())
        k_step = max(0, n_ver_after - n_ver_before)
        if stats_out is not None:
            stats_out["accept_histogram"][k_step] = stats_out["accept_histogram"].get(k_step, 0) + 1

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
    # When free-generating across sliding sub-blocks, truncate block strictly to the contiguous verified
    # prefix so that unverified mask holes (mask_id) never leak across sub-block boundaries!
    if frozen.any():
        leftover = canvas == mask_id
        if leftover.any():
            fill = canvas_forward(model, prefix_ids, canvas, prefix_keep).argmax(-1)
            canvas = torch.where(leftover, fill, canvas)
        return canvas

    # Find strictly contiguous verified prefix, ensuring NO mask_id can ever be included
    m_end = 0
    while m_end < canvas_len and verified[0, m_end] and int(canvas[0, m_end]) != mask_id:
        m_end += 1

    if m_end == 0:
        # Fallback: at least slot 0 is verified from step 1
        m_end = 1
        verified[0, 0] = True
        if int(canvas[0, 0]) == mask_id:
            # Emergency fallback: predict slot 0 directly
            cand_tok = int(sampled[0, 0]) if int(sampled[0, 0]) != mask_id else int(prefix_ids[0, -1])
            canvas[0, 0] = cand_tok

    # Truncate cleanly at m_end: ZERO mask_id tokens allowed!
    ret = canvas if frozen.any() else canvas[:, :m_end]
    if stats_out is not None:
        stats_out["tokens_verified"] += ret.shape[1]
        fp = max(1, stats_out["forward_passes"])
        tv = max(1, stats_out["tokens_verified"])
        stats_out["alpha"] = stats_out["tokens_verified"] / fp
        stats_out["r_kv"] = (stats_out["active_tokens_evaluated"] / tv) - 1.0
    return ret


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
    stats_out=None,
    use_v1=False,
    **kwargs,
):
    """Generate canvas_len tokens using the fused staircase sliding engine with in-place KV slicing."""
    if use_v1:
        sub_c_len = sub_canvas_len or 8
        sub_s = sub_steps or max(1, steps // max(1, canvas_len // sub_c_len))
        cur_prefix = prefix_ids
        cur_keep = prefix_keep
        all_blocks = []
        for blk_start in range(0, canvas_len, sub_c_len):
            cur_len = min(sub_c_len, canvas_len - blk_start)
            block_out = sample_canvas_block(
                model,
                cur_prefix,
                canvas_len=cur_len,
                steps=sub_s,
                prefix_keep=cur_keep,
                carrier_priority=carrier_priority,
                stop_token_ids=stop_token_ids,
                stats_out=stats_out,
                **kwargs,
            )
            all_blocks.append(block_out)
            cur_prefix = torch.cat([cur_prefix, block_out], dim=1)
            if cur_keep is not None:
                cur_keep = torch.cat(
                    [cur_keep, torch.ones_like(block_out, dtype=cur_keep.dtype)], dim=1
                )
        return torch.cat(all_blocks, dim=1)

    def cb(s, disp_tensor):
        if block_callback is not None:
            block_callback(0, s, disp_tensor)

    B = prefix_ids.shape[0]
    if B > 1:
        chunks = []
        for b in range(B):
            p_keep_b = prefix_keep[b : b + 1] if prefix_keep is not None else None
            out_b = sample_staircase_sliding(
                model,
                prefix_ids[b : b + 1],
                canvas_len=canvas_len,
                prefix_keep=p_keep_b,
                step_callback=cb if b == 0 else None,
                stop_token_ids=stop_token_ids,
                stats_out=stats_out,
                **kwargs,
            )
            chunks.append(out_b)
        return torch.cat(chunks, dim=0)

    return sample_staircase_sliding(
        model,
        prefix_ids,
        canvas_len=canvas_len,
        prefix_keep=prefix_keep,
        step_callback=cb,
        stop_token_ids=stop_token_ids,
        stats_out=stats_out,
        **kwargs,
    )


@torch.no_grad()
def sample_canvas_text(model, tokenizer, prefix_ids, **kwargs):
    """Convenience wrapper: (decoded_text, canvas_ids) for a single prompt."""
    canvas = sample_canvas(model, prefix_ids, **kwargs)
    return [tokenizer.decode(c, skip_special_tokens=True) for c in canvas], canvas
