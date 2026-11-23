# Research notes

## Objective

Turn a pretrained causal LM into a block-diffusion LM:

- prefix `x[0:P]` is clean context,
- canvas `x[P:P+C]` is the generation target,
- the model learns `p(x_i | prefix, canvas \ {i})` for masked positions,
- generation starts from a fully masked canvas and iteratively fills it.

Nothing is language- or domain-specific: training and sampling operate on raw
BPE ids.

## Mask token without touching the embedding table

Qwen2.5's embedding matrix has 151936 rows, but its tokenizer only emits ids up
to 151664. The rows above that are unused during pretraining.

`setup_tokenizer` adds `<|extra_0|>` to the tokenizer, which assigns the next
free id, 151665. That id addresses an existing embedding row, so:

- `model.config.vocab_size` and the tensor shapes are unchanged,
- no `resize_token_embeddings`, therefore no reinitialisation of any existing
  row,
- Qwen2.5 ties input embeddings and the LM head, so the `[MASK]` vector and its
  output projection are automatically the same parameter.

If a model has no spare embedding row (`mask_id >= vocab_size`), `prepare()`
falls back to resizing, which is only relevant for non-Qwen smoke models.

## Hybrid 4D mask

```
        keys:  prefix P        canvas C
query prefix:   causal           blocked
query canvas:   all              all
```

- Prefix stays a normal causal LM so its hidden states are valid context.
- Canvas is bidirectional, which lets a token use both left and right unmasked
  neighbours.
- Blocking prefix→canvas prevents canvas content from changing the prefix
  representation and keeps training consistent with inference.

The mask is additive: `0.0` where attention is allowed,
`torch.finfo(dtype).min` where it is not. Shape `(B, 1, T, T)`, broadcast over
heads. This is the format PyTorch SDPA accepts. FlashAttention's arbitrary-mask
mode is not portable across versions, so `attn_implementation="sdpa"` is pinned.

Position ids are contiguous. With left-padded prompts, padded prefix tokens get
position 0 and real tokens start at 0; canvas positions start at each row's real
prefix length. This keeps prompt batches aligned without embedding a padding
offset into the canvas.

## Corruption and loss

For every sample, 20% of batches are fully masked (`t = 1.0`); the remaining
batches draw `t ~ U(0.20, 1.0)` per sample and mask exactly `floor(t*C)`
positions. Full-mask batches teach from-scratch generation — without them the
model never sees the all-mask canvas it actually starts from at inference.
Exact counts matter for a predictable difficulty schedule; independent
per-position Bernoulli draws would only match `t*C` in expectation.

Loss is plain cross-entropy on masked positions only. No weighting, no
reconstruction of unmasked tokens. The model is not trained to copy clean
tokens, so the objective at `t → 0` is undefined by construction; the 0.20
floor avoids wasting compute on near-clean canvases.

## Sampler

Non-autoregressive decoding can commit early to a wrong token and then repeat
it or split a subword. The margin

```
margin_i = log p_i(token_1) - log p_i(token_2)
```

is a cheap uncertainty proxy: a slot whose top-1 token dominates is unlikely to
change once neighbours are revealed.

Budget after step `s`: `k = floor(s / K * C)` total unmasked slots. Slots are
ranked by margin, and the highest-margin `k` slots keep their top-1 token.
Already revealed tokens are frozen (their margin is set to `-inf`). With a
pre-filled canvas (reconstruction), the number of newly revealed slots is
capped per sample at the number still masked; otherwise `topk` could pull in
`-inf` entries and overwrite frozen tokens.

The final step has `k = C`, so the canvas is fully concrete. A last argmax pass
covers any rounding leftovers.

### Re-masking

Parallel unmasking commits tokens that later steps cannot revisit: a slot decoded
at step 1 never sees the tokens decoded at step 2. `remask_ratio` (default 0)
re-opens the weakest fraction of already-decoded slots at every interior step.

The quota runs over the mutable slots only: after step `s` the canvas holds
`n_frozen + floor(s/K * (C - n_frozen))` unmasked slots. Pre-filled
reconstruction slots are frozen, so they neither compete on margin nor consume
the schedule — a 50%-masked canvas with `K=8` still gets 8 real steps
(`16,14,12,10,8,6,4,2` masks before each forward), not 4. A masked slot that is
not re-selected goes back to `[MASK]`.

At each interior step, for every batch row:

1. Score all `C` positions by the current margin (logit top-1 minus top-2).
2. Protect the strongest `ceil((1 - remask_ratio) * n_decoded)` decoded slots:
   they keep their token and count against the quota but are not re-selected.
3. The remaining decoded slots compete with the masked slots on their raw
   margin, so a stale commitment can lose to a confident new position.
4. Fill the remaining budget `floor(s/K * (C - n_frozen)) - n_protected` with the
   highest-scoring candidates.
5. Rebuild the row: frozen slots keep their token, non-selected slots return to
   `[MASK]`, protected slots keep theirs, and newly selected slots take the
   freshly sampled token.

The final step fills every remaining mask, so the canvas is always complete.
`margin_floor` only blocks slots; it never adds catch-up slots, so a floor can
leave the canvas under quota for a step and those slots are filled at the end.

### Adjacent-token suppression

Two slots decoded in the same step cannot see each other's choice, so they can
both commit the same high-prior token and produce `not not` / `the the`. Before
the budget is spent, adjacent slots whose effective token (stored for committed
slots, sampled otherwise) is identical are compared on margin: the weaker one is
scored `-inf` for this step, so only the stronger anchor commits and the loser
re-predicts next step with its neighbour in the canvas.

Only still-masked slots can be held back — a token that is already committed is
never revoked here, so the quota stays monotone. Two consequences: the loser
costs one step of latency at the tail (with `C=32, K=8`, 50% masked, the mask
schedule becomes `16,14,12,10,8,6,4,4` instead of `...,4,2`), and a collision
between two tokens committed in *different* steps is out of scope, because the
two slots never predicted simultaneously.

### Margin floor

A rigid linear quota forces the model to commit to `C/K` tokens per step even
when it is unsure. Early wrong commitments cannot be revised, and repeated
phrases are the usual symptom.

`margin_floor` adds an absolute gate: a masked slot is eligible for unmasking
only if `margin >= floor`. Held-back slots stay masked until a later step or the
final fill. This trades a little latency for fewer early mistakes. A high floor
can leave most of the canvas to the final fill, so the floor is meant to be
small (roughly 0.5-2.0 in log-prob units) and tuned in `benchmark.py --floors`.

## CPU execution

The verification machine is a 10-core / 20-thread Xeon Gold 5215 (Cascade
Lake). It has AVX-512 but no native bf16 or AMX, so bf16 would be emulated;
fp32 with MKLDNN is faster. `train.py` therefore keeps fp32 on CPU, sets
`torch.set_num_threads` (explicit `--threads` or all cores), uses
`torch.inference_mode()` for evaluation and benchmarking, and caps the corpus
with `--max_tokens` because `GeneralTextDataset` materialises overlapping
windows eagerly.

On hyper-threaded Xeons, setting `--threads` to the physical core count can beat
the default all-cores setting.

## Stochastic token choice and position prior

Greedy argmax in parallel unmasking commits confidently to one token per slot
and tends to collapse into a repeated phrase. Two optional, sampler-only knobs
address this without retraining:

- **Gumbel temperature.** When `temperature > 0`, the token written to each
  selected slot is `argmax(logits / T + Gumbel(0,1))`, which is equivalent to
  sampling from `softmax(logits / T)`. The rank order (which slots to unmask)
  still comes from the clean log-probs, so confidence estimates are not
  distorted by noise. `T = 0` keeps the previous greedy behavior.
- **Top-k truncation.** Qwen2.5 has 151,936 output tokens. Standard Gumbel noise
  is unbounded, and a token sitting 10 nats below the top-1 can win the argmax
  when its noise draw is extreme; over 32 slots this happens often enough to
  produce punctuation and subword salad. With `top_k = 50`, logits below the
  50th largest per slot are set to `-inf` before the noise is added, so sampling
  stays inside the plausible head of the distribution. Truncation is applied
  only when `temperature > 0`.
- **Step-1 position decay.** Token 0 of the canvas directly follows the prefix;
  token 25 is separated from it by 25 masks. On the first step only, margins
  are multiplied by `exp(-lambda * i / C)`, so the left boundary settles first
  and later steps condition on concrete causal anchors. `lambda = 1.5` makes
  the first 2-4 slots strongly preferred for `C = 32`.

Both knobs leave frozen/pre-filled tokens untouched, so reconstruction and
partial-canvas decoding keep their guarantees.

## Prefix KV cache

The prefix is clean context: its hidden states and therefore its key/value
tensors are identical at every diffusion step. Recomputing them is pure waste,
and at `P = 128`, `C = 32` it is 80% of the per-step token count.

`prefill_prefix` runs the prefix once with `use_cache=True` and keeps the
returned `DynamicCache`. `canvas_step_with_cache` then feeds only the `C`
canvas ids with a custom 4D mask of shape `(B, 1, C, P+C)`: every canvas query
sees all prefix keys and all canvas keys, which is exactly the canvas row of the
hybrid mask. Canvas RoPE positions still start at the real prefix length, so
they are identical to the full-sequence pass. After each step the canvas rows
are cropped from the cache (`crop(-C)`), leaving the prefix cache reusable.

Measured fp32 difference against a full `(P+C)` forward is ~3e-5 in logits and
token-identical argmax. A diffusion step therefore costs `C` tokens instead of
`P+C`.

## LoRA training

Full fp32 fine-tuning updates 496M parameters and stores two AdamW moments for
each, which dominates both compute and the 1.9 GB checkpoints. `--lora` freezes
the base model and trains rank-16 adapters on `q_proj, k_proj, v_proj, o_proj`
(2.16M parameters, 0.4%), alpha 32, dropout 0.05, bias none. Only the adapter
matrices receive gradients and optimizer state, so checkpoints shrink to a few
MB and the backward pass skips the frozen MLP blocks.

The adapter starts at zero (`B = 0`), so the initial forward pass is the
pretrained model and the loss starts where the base model is. LoRA tolerates a
higher learning rate than full fine-tuning; `--lr` defaults to `5e-4` with
`--lora` and `5e-5` without.

`--lora_mlp` extends the targets to `gate_proj, up_proj, down_proj`, the MLP
blocks that hold most of the weights. Attention-only adapts 2.16M parameters
(0.43%); adding the MLP adapts 8.80M (1.75%). That is the stage-3 ablation knob:
if attention-only re-masking still repeats, the frozen MLP is the next suspect.

## Cost model

A diffusion step with the cache is one forward over `C` tokens plus a one-off
`P`-token prefill, so `K` steps cost `P + K*C` token evaluations. Causal
generation costs `P + C` but strictly sequentially: 32 kernel launches of one
token each. The diffusion decoder is therefore cheaper per token at small `K`
and parallel across the canvas, at the price of needing a model fine-tuned for
bidirectional denoising. Logits for a canvas step are `(B, C, vocab)`, which is
the dominant memory term at large `C`.

`--prefix_sweep` measures that trade-off directly: it fixes `C`, pads or
truncates a real document to exactly `P = 32, 128, 256, 512` tokens, and times
causal `generate()` against cached `k=4` and `k=8` at batch 1 and 4. Prompt
length is the variable that matters: with `P` around 15 (the default generation
prompts), the prefix is a rounding error and the cache cannot help.

## Right-Shifted Coordinate Alignment & Carrier-Frontier Diffusion

Pretrained causal transformers (including Qwen2.5) are hardwired so that query position $k$ predicts token $k+1$. In standard causal training:
$$\text{logits}[k] \to \text{target}[k+1]$$
When turning this into a hybrid prefix-canvas architecture:
1. **Right-Shifted Coordinates:** Slicing `out.logits[:, P-1 : P+C-1, :]` ensures that position $P-1$ (the prompt boundary token) predicts `canvas[0]`, and active canvas position $P-1+j$ predicts `canvas[j]`. This aligns 100% of the transformer's native positional representations with the diffusion objective.
2. **Trainable `[MASK]` Row:** The unused spare row `151665` (`<|extra_0|>`) is unfrozen via an Adam gradient filter hook during LoRA training. The base embedding table remains frozen.
3. **Carrier-Frontier & Island Anchoring:** Because slot $j$'s logits sit on top of input token $j-1$'s residual stream, unmasking slot $j$ when $j-1$ is still `[MASK]` forces the model to guess without a carrier token. The sampler prioritizes slots with concrete left carriers (`has_carrier == True`), while unmasking non-adjacent island anchors ($j \ge 2$) when the step budget exceeds available carriers, enabling independent frontiers to double carriers cleanly ($1 \to 2 \to 4 \to 8$) in parallel.
4. **Sliding Sub-Blocks & Stop Token Early Termination:** To generate sequences of arbitrary length without quadratic memory or unanchored degradation, `sample_canvas` slides across sub-blocks ($C_{\text{sub}}=8$ or $16$) that are committed into the causal prefix KV cache as they finish. When a stop token (e.g. `<|im_end|>`, `<|endoftext|>`) is carrier-verified within a sub-block, the sampler terminates future steps and trailing sub-blocks immediately, cutting generation latency by up to 50–75%.

## BFG-Diff v2: Fused Verify-and-Diffuse Staircase Engine

BFG-Diff v2 upgrades the right-shifted discrete block-diffusion architecture into a unified, single-pass speculative verification and diffusion engine. It eliminates redundant sub-block recomputations and raises verification throughput via five core architectural pillars:

### Pillar 1: Fused Staircase Attention (Zone A + Zone B)

In BFG-Diff v1, verification and drafting operated in separate or unisolated attention structures where unverified `[MASK]` tokens could pollute prompt queries. In BFG-Diff v2, each active forward step evaluates a unified window of size $Q = 1 + V + W$ tokens:
- **Token 0:** The prompt boundary token (`prefix_last_id` at coordinate $P-1$).
- **Zone A ($1 + V$ tokens):** Prefix last token plus $V$ speculative draft tokens under a **strictly causal mask**. Zone A queries attend causally to the prefix and prior Zone A tokens, but attention to Zone B keys is strictly masked to $-\infty$ (`torch.finfo(dtype).min`). This guarantees complete causal isolation: Zone A representations are numerically identical to a true causal forward pass ($\Delta_{\text{causal}} < 10^{-4}$, $\Delta_{\text{leak}} = 0.0$).
- **Zone B ($W$ tokens):** Lookahead diffusion slots filled with `[MASK]` tokens (`<|extra_0|>`). Zone B queries attend bidirectionally to Prefix, all of Zone A, and all of Zone B.
In every single forward pass, Zone A verifies $1 \dots V+1$ speculative tokens while Zone B simultaneously diffuses drafts for the next $W$ tokens.

### Pillar 2: Progressive In-Place KV Cache Slicing

Because Zone A queries never attend rightward to Zone B, the key-value states computed for Zone A in `DynamicCache` are mathematically valid causal KV states.
- At the start of step $s$, the prefix KV cache has length `cur_prefix_len - 1`.
- Step $s$ forwards $Q = 1 + V + W$ tokens, appending $Q$ keys and values to the cache.
- The sampler verifies $k \in [1, 1+V]$ contiguous tokens in Zone A.
- Rather than discarding the entire sub-block cache and recomputing prefix states via separate `advance_tokens` passes, the sampler performs in-place slicing:
  $$\text{kv.crop}(-(Q - k))$$
- This keeps the first $k$ tokens permanently in the cache, synchronizes `kv.get_seq_length() == cur_prefix_len - 1 + k`, and crops only the unverified Zone A tail and Zone B diffusion states. Redundant token evaluation ratio drops from $R_{\text{kv}} \approx 4.5$ down to near theoretical minimum.

### Pillar 3: BFGDiffAdapter (Slot-Offset Embeddings & Latent Recycling)

Pretrained causal transformers rely on positional embeddings (RoPE) to distinguish tokens. When multiple identical `[MASK]` tokens are passed in Zone B, diffusion drafting lacks explicit structural distance signals. `BFGDiffAdapter` introduces:
1. **Learnable Slot-Offset Table (`slot_emb`):** Shape $(W_{\text{max}}, D)$. Adds explicit distance embeddings for offsets $+1 \dots +W$ relative to the active frontier.
2. **Latent Residual Recycling (`latent_proj`):** A low-rank bottleneck projection ($W_{\text{down}}: D \to r$, $W_{\text{up}}: r \to D$, rank $r=64$) mapping top-layer hidden states $h^{(L)}$ of unverified slots from step $s-1$ into step $s$ inputs:
   $$E_{\text{Zone B}}[j] = \text{Emb}(\text{mask\_id}) + \text{SlotEmb}[j] + W_{\text{up}}(\text{GELU}(W_{\text{down}}(h^{(L)}_{\text{prev}}[j])))$$
3. **Zero-Initialization Compatibility:** `slot_emb.weight` and `latent_up.weight` are initialized to zero, ensuring exact mathematical backward compatibility with base and v1 checkpoints ($0.000000$ drift on step 0).

### Pillar 4: Confidence-Gated 2-Branch Micro-Tree Verification

When the primary draft token in Zone B exhibits low confidence (top-2 logit margin $z_{(1)} - z_{(2)} < \theta_{\text{branch}}$), single-linear verification risks immediate rejection. BFG-Diff v2 forks Zone A into a 2-branch micro-tree:
- Zone A contains two parallel hypotheses: branch 1 (`[top1, draft_c1]`) and branch 2 (`[top2, draft_c2]`).
- A 2D tree attention mask prevents branch 1 from seeing branch 2 and vice versa, while both attend to the causal root.
- When branch 2 is accepted, the KV cache layers are updated in-place by copying branch 2 states over branch 1 positions before cropping.

### Pillar 5: On-Policy Teacher-Student Self-Distillation KL & Suffix Curriculum

Instead of hard cross-entropy against fixed training tokens, `train.py` trains the adapter via self-distillation against the frozen base causal model:
- **Teacher Pass:** Computed causally using `with model.disable_adapter():`.
- **Top-64 Gathered KL Divergence:** Gathers student logits for the teacher's top 64 tokens, avoiding memory materialization of full $151,936$-dim vocabulary distributions.
- **Suffix Curriculum:** `make_mask` draws 50% contiguous lookahead suffixes (mimicking decoding state), 30% simulated corrupted drafts, and 20% scatter masks.
- **Distance Weighting:** Loss is weighted by $w(j) = 1.0 / (1.0 + 0.3 \cdot j)$, prioritizing immediate frontier accuracy $A_{+1}$.

## BFG-Diff v2 Surgical Bug Fixes & Alignment (2026-11-18)

During dual-checkpoint verification on real weights (`runs/bfg-polished/last.pt` and `runs/bfg-instruct/last.pt`), five surgical alignment bugs in the initial implementation were identified and resolved:

### 1. Lookahead Draft Proposal & Offset Alignment in `sampler.py`
- **Issue:** In `sample_staircase_sliding`, candidate lookahead drafts were gated behind `margin_b >= draft_margin_thresh` with a high default threshold `0.8`. For undertrained or standard models, lookahead drafts were almost always suppressed (`draft_ids = None`), preventing Zone A speculative leaps. In addition, offset indexing between Zone A output logits and Zone B lookahead drafting required strict verification against carrier tokens.
- **Resolution:** Set `draft_margin_thresh = 0.0` by default so lookahead tokens are always proposed for verification. Zone B mask index `0` (at `zone_a_len`) directly predicts the token following the Zone A bonus token, ensuring seamless speculative pipeline execution.

### 2. Full Adapter Wire-Up During Training and Inference
- **Issue:** `BFGDiffAdapter` was disconnected at both training and inference time:
  1. During training, `apply_to_embeds(base_embeds, P, C)` mistakenly treated `P` as `zone_a_len` and added slot offsets to clean prompt/target tokens rather than unmasked slots starting at frontier $P+m$.
  2. During inference, `sampler.py` defaulted `adapter=None` and checked `hasattr(adapter, "get_inputs_embeds")` while the adapter method was named `apply_to_embeds`, causing latent recycling to never execute.
- **Resolution:** 
  - Updated `apply_to_embeds` in `model.py` to accept `mask` and `prefix_len` tensors, applying zero offset to clean context and adding `slot_emb[0..]` strictly to masked canvas slots relative to each sample's mask frontier.
  - Added `get_inputs_embeds` convenience wrapper in `BFGDiffAdapter`.
  - In `sampler.py`, auto-detected `adapter = getattr(model, "bfg_adapter", None)`, enabled `return_hidden = (adapter is not None)`, and recycled top-layer hidden states `latents[:, zone_a_len : zone_a_len + len(candidate_drafts)]` into subsequent forward passes.

### 3. Removal of Synthetic Placeholders in Ablation Benchmarks
- **Issue:** `benchmark.py` (`run_ablation_table`) previously used hardcoded placeholders (`tps_v1 * 0.72`, `"alpha": 1.00, "r_kv": 4.25`) instead of executing real sub-block diffusion.
- **Resolution:** Replaced all synthetic placeholders with actual execution passes via `use_v1=True` and real sliding sub-block loops (`sample_canvas_block`), measuring genuine hardware throughput and acceptance rates on GPU.

### 4. Batch Isolation in `sample_canvas`
- **Issue:** `sample_staircase_sliding` indexed batch element 0 (`z[0, j]`) and returned `committed_tokens` expanded to batch size $B$, discarding elements $1 \dots B-1$ and corrupting corpus diversity metrics ($Dist1, Dist2$) when $B > 1$.
- **Resolution:** Updated `sample_canvas` to chunk multi-prompt batches into independent prompt passes ($B=1$), isolating cache and committed states and eliminating cross-batch overwrite.

### 5. CLI Device Selection in `chat.py`
- **Issue:** `chat.py` hardcoded `device="cpu"` and `dtype=torch.float32`.
- **Resolution:** Introduced dynamic `--device` CLI argument (`auto`, `cuda`, `cpu`) with automatic GPU acceleration when CUDA is available.

---

## BFG-Diff v2.1 Architecture & Bottleneck Resolution

### 1. Multi-Exit Lookahead Drafting (Ping-Pong Elimination)
- **Problem:** When `draft_margin_thresh = 0.0` was combined with fixed `V_cur = 3`, offset $+3$ (which only has ~10% accuracy) caused verification rejection (`r < V_cur`) on ~90% of forward passes. Previously, `sampler.py` only extracted new drafts when `V_cur == r`, causing the engine to reset `V_cur = 0` on every rejection step and alternating between $(1, 2, 1, 2)$ tokens per step ($\alpha = 1.50$).
- **Fix:** 
  1. Offset-conditioned confidence gating: Offset $+1$ drafts at $\tau \ge 0.0$, Offset $+2$ requires margin $\ge 0.35$, Offset $+3$ requires margin $\ge 0.75$.
  2. Fallback lookahead drafting: When $r < V_{\text{cur}}$, the engine extracts the next step's draft candidate directly from Zone B slot $0$ (which attended to the verified prefix and sees the causal continuation) with fallback margin gating ($\ge 0.20$), completely preventing the reset to $V_{\text{cur}} = 0$.

### 2. Fair Apples-to-Apples Batch Benchmarking
- **Problem:** `run_causal_generation` ran at batch size $B=4$ in a single matrix pass while `sample_canvas` sequentially iterated over $B=1$, hiding single-stream speculative speedups.
- **Fix:** In `benchmark.py`, `run_ablation_table` evaluates causal baseline and BFG-Diff methods at identical effective batch size ($B=1$ interactive single-stream, and $B=4$), reporting explicit batch indicators in table outputs.

### 3. Mask-Only Slot Embedding Injection
- **Problem:** In `train.py`'s simulated draft curriculum, corrupted concrete draft tokens were treated as masked, adding `slot_emb` to clean token embeddings.
- **Fix:** `model.py` and `train.py` enforce that `BFGDiffAdapter.apply_to_embeds` ONLY injects `slot_emb` and latent recycling onto actual `[MASK]` tokens (`canvas_ids == mask_id`). Clean and draft token embeddings are preserved untouched.

### 4. Multi-Domain Real Corpus Distillation
- **Problem:** Default fallback to 21-word synthetic word salad caused catastrophic forgetting during extended training.
- **Fix:** Added `build_real_corpus.py` harvesting >1,000 real technical and markdown documents (>12M characters, zero benchmark leakage) and ChatML dialogues for production training of `runs/bfg-v2-base/last.pt` and `runs/bfg-v2-instruct/last.pt`.

---

## BFG-Diff v2.2 Architecture: Unstarving Gradients & Sigma Self-Conditioning

### 1. Flat Frontier Plateau Weighting in Diffusion Loss
- **Problem:** Previously, `diffusion_loss` decayed target position weights exponentially immediately starting at the mask frontier: $w_j = \exp(-0.15 \cdot j_{\text{rel}})$. This starved multi-offset lookahead positions $+2, +3, +4$ of supervisory gradients, locking $A_{+2..4}$ at $12.5\% \text{--} 16.7\%$.
- **Fix:** Implemented a flat plateau across lookahead positions $0 \dots 4$ ($w_j = 1.0$), with gentle decay only past offset $4$: $w_j = \exp(-0.10 \cdot (j_{\text{rel}} - 4))$.

### 2. Sigma Predicted-Embedding Self-Conditioning & Analytical SVD Init
- **Concept:** Following Sigma (*arXiv:2610.02665*) and iWSP (*arXiv:2611.18114*), Zone B mask slots receive concrete lexical guidance to resolve multi-token semantic ambiguity without breaking Zone A causal isolation.
- **Formulation:**
  $$E_{\text{Zone B}}[j] = \text{Emb}(\text{mask\_id}) + \text{SlotEmb}[j] + W_{\text{up}}\big(\text{SiLU}(W_{\text{down}}(\text{Norm}(h^{(L)}_{\text{prev}}[j])))\big) + \alpha_{\text{sc}} \cdot \text{Norm}\big(\text{Emb}(\hat{y}^{(s-1)}_j)\big)$$
- **Zero-Init Parity:** $\alpha_{\text{sc}}$ (`pred_emb_scale`) is initialized to $0.0$, guaranteeing $100\%$ numerical identity with step-0 and legacy checkpoints ($\Delta < 10^{-6}$).
- **Training Unroll:** On 25% of training steps, an unroll pass generates $\hat{y}$ predictions under `torch.no_grad()`, passing token embeddings into pass 2.
- **Inference Recycling:** In `sample_staircase_sliding`, both top-layer hidden states `latents` and predicted token embeddings `prev_pred_embeds` are recycled into Zone B for the subsequent step.

---

## BFG-Diff v3: Weight-Tied Top-4-Layer Block-Diffusion Drafter (<1% PEFT)

BFG-Diff v3 upgrades the speculative architecture to break the "blind skip-1" ceiling and eliminate draft model deployment overhead by weight-tying the speculative drafter directly to the frozen backbone's own top 4 transformer layers ($L-4 \dots L-1$).

```
Stage 1 (Frozen Backbone):
Raw Causal Forward -> Tap 4 Evenly Spaced Layers -> Multi-Layer Feature Fusion (F_ctx)
                                                            |
Stage 2 (Weight-Tied Drafter):                              v
Concrete Token y* + [MASK] x (W-1) -> Manifold Bridge -> 4 Tied Layers (L-4..L-1) + LoRA -> Draft Tokens [d_0..d_6]
                                                            ^
                                           Draft StaticCache (Prefix KV from F_ctx)
```

### 1. Minimal `DraftLoRADelta` (Zero `peft` Wrappers)
- Base `Qwen2DecoderLayer` modules are never wrapped, patched, or mutated.
- Stage 1 (`forward_verify`) invokes the raw frozen base model untouched with zero LoRA dispatch overhead.
- Stage 2 (`forward_draft`) explicitly calls the frozen layers `model.model.layers[L-4 : L]` with lightweight `DraftLoRADelta(in_features, out_features, r=32, alpha=64)` applied to `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.
- Helper `apply_linear_with_delta(base_linear, x, delta)` evaluates `base_linear(x) + delta(x.to(delta.lora_A.dtype)).to(x.dtype)`. Parameters remain in `fp32` during training and cast to `fp16` during inference, guaranteeing 100% CUDA-Graph safety.

### 2. Multi-Layer Target Feature Fusion (`F_ctx`)
- Taps 4 evenly spaced target layers from Stage 1:
  - For $L=28$ (`Qwen2.5-1.5B`): $\{6, 13, 20, 27\}$
  - For $L=36$ (`Qwen2.5-3B`): $\{8, 17, 26, 35\}$
  - General formula: `[(L // 4) * i - 1 for i in range(1, 5)]`
- Fused via bottleneck MLP + RMSNorm:
  $$\text{Linear}(4D \to 512, \text{bias}=\text{False}) \to \text{SiLU} \to \text{Linear}(512 \to D, \text{bias}=\text{False}) \to \text{Qwen2RMSNorm}(D)$$

### 3. Manifold Alignment Bridge (Fixing Layer-0 vs. Layer-`L-4` Mismatch)
- Raw token embeddings $\text{Emb}(y^*)$ live in Layer-0 embedding space, whereas Layer $L-4$ expects Layer $L-4$'s residual stream.
- The draft block of length $W$ (`[y*, MASK_1, ..., MASK_{W-1}]`) is aligned as:
  - Slot 0 (concrete bonus token $y^*$): `bridge(cat([RMSNorm(Emb(y*)), F_ctx_last])) + F_ctx_last`
  - Slots $1 \dots W-1$ (`[MASK]`): `RMSNorm(Emb(MASK)) + SlotEmb[1..W-1] + F_ctx_last` projected through the bottleneck bridge.
  - Bridge up-projection is initialized near zero (`std=1e-3`), ensuring Step 0 starts cleanly in the Layer-($L-4$) manifold while allowing gradient backpropagation to all bridge and slot embedding parameters.

### 4. Exact Right-Shift Alignment ($k \to k+1$)
- Under Qwen2's native shift, feeding $W$ tokens $[y^*, \text{MASK}_1, \dots, \text{MASK}_{W-1}]$ at RoPE positions $[cur\_pos - 1, cur\_pos, \dots, cur\_pos + W - 2]$ produces $W$ output logits predicting $[x_{cur\_pos}, x_{cur\_pos+1}, \dots, x_{cur\_pos+W-1}]$.
- Slot 0 predicts $A_{+1}$ directly from concrete bonus token $y^*$, eliminating the blind skip-1 penalty.

### 5. Zero-`crop()`, Pointer-Indexed `StaticCache` & Active Shapes
- Static active shapes (multiples of 8 for Tensor Cores):
  - Stage 1 verify: fixed $Q_A = 8$ ($[y^*, d_0, \dots, d_6]$). Writes target KV in-place at $[t .. t+7]$.
  - Stage 2 draft: fixed $W = 7$ draft tokens.
- No `kv.crop()`: When $k$ tokens are accepted ($num\_committed = k + 1$), $t$ simply advances by $num\_committed$. Subsequent verification steps automatically overwrite rejected tail slots in-place.
- Supports ragged $B=4$ batches with per-row position indexing.

### 6. Exact Train–Inference Invariant & Final Layer Normalization
- Enforces strict mathematical identity between `train_v3.py` and `sampler_v3.py`:
  - When drafting from concrete bonus token $y^*$ at sequence position $t$ ($y^* = x_t$):
    - $y^*$ has RoPE position $t$, and the $W$ draft block tokens $[y^*, \text{MASK}_1, \dots, \text{MASK}_{W-1}]$ have RoPE positions $[t, t+1, \dots, t+W-1]$, predicting $[x_{t+1}, x_{t+2}, \dots, x_{t+W}]$.
    - $f_{\text{ctx\_last}}$ MUST be $f_{\text{ctx}}[t-1]$ (the Stage-1 fused feature of the token immediately preceding $y^*$, NOT $f_{\text{ctx}}[t]$).
    - Draft prefix KV cache contains strictly positions $0 \dots t-1$ (length $t$, zero unwritten zero slots).
    - Targets are $\text{seq}[:, t+1 : t+1+W]$, and teacher Top-64 logits are $\text{top64}[:, t : t+W]$ (since $\text{teacher\_logits}[:, t]$ predicts $\text{seq}[:, t+1]$).
  - Tapped Final Hidden State Normalization: Transformers `AutoModelForCausalLM` decorates `forward()` with `@capture_outputs(tie_last_hidden_states=True)`, which overwrites `out.hidden_states[-1]` with `outputs.last_hidden_state` (post-`RMSNorm`). In `sampler_v3.py`, `all_hidden_states` explicitly appends `model.norm(h)` for layer $L-1$, guaranteeing exact numerical parity ($2.77 \times 10^{-5} < 10^{-4}$ in Unit Gate D).

### 7. Vectorized Parallel Random-Anchor Distillation (`train_v3.py`)
- Full sequence prefix projection: `drafter.project_prefix_kv(f_ctx, pos_all)` is evaluated once for the full sequence $(B, S, D)$.
- Packed blocks: $N$ anchor blocks ($M = N \times W$ tokens) are packed into a single tensor $(B, M, D)$ with RoPE positions `pos_draft` $(B, M)$.
- Block-diagonal attention mask: A $(1, 1, M, S + M)$ mask allows each block $m$ (anchor $t_m$) to attend exclusively to prefix keys $0 \dots t_m - 1$ and bidirectionally to its own $W$ block keys, executing all $N$ blocks in ONE single 4-layer forward pass.

### 8. Production Multi-Model Dynamic Serving & Web UI Architecture (`chat.py`)
- **`V3ModelManager` Thread-Safe Dynamic Caching**:
  - Implements lazy loading and thread-safe instance caching for multiple model sizes (`1.5B` headline production and `0.5B` prototype).
  - Pre-allocates dedicated `StaticDraftEngine` with static Stage-1 ($Q_A=8$) and Stage-2 ($W=7$) pointer-indexed KV caches.
  - Automatic offline initialization: Sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to guarantee instant local weight loading without remote metadata timeouts.
- **Speculative Streaming Protocol (`generate_v3_stream`)**:
  - Emits real-time SSE event stream (`chat_step`, `bfg_step`) containing accepted draft token chips, per-step acceptance counts ($k/7$), cumulative response text, live TPS, and step latency.
  - Tracks running speculative speedup and mean acceptance length $\alpha = (\sum k + \text{steps}) / \text{steps}$.
- **Web UI Header Dropdown & Model Hot-Swapping**:
  - Integrated `.preset-dropdown` selector in the top navigation bar enabling client-side switching between `1.5B` (15k bundle) and `0.5B` without reloading the page.
  - Transparently dispatches `/api/chat` and `/api/generate` requests with the active `model` parameter.
  - Added `/api/models` metadata endpoint returning active registry configs, parameter counts, and checkpoint paths.

---

## 9. BFG-Diff v4: Zero-Retraining Inference Engine (`sampler_v3.py` / `model_v3.py`)

BFG-Diff v4 upgrades the speculative decoding runtime with four zero-retraining systems optimizations. It is 100% backward compatible with existing v3 checkpoints (e.g. `bfg_v3_1.5b_15k/last.pt`), maintains fixed active shapes ($Q_A = 8, W = 7$) for CUDA Graphs, and preserves strict lossless verification parity against the frozen base model.

### 1. Write-Ahead Log (WAL) `draft_cache` + `[0:W]` In-Place Scratch Tail + Key-Only RoPE
- **Key-Only RoPE Helper (`apply_rotary_pos_emb_k`)**:
  - Slices `cos, sin` rotary embeddings and rotates key states directly without allocating dummy queries:
    $$\text{apply\_rotary\_pos\_emb\_k}(k, \cos, \sin) = (k \odot \cos) + (\text{rotate\_half}(k) \odot \sin)$$
- **Write-Ahead Log (WAL) inside `_step_verify_kernel`**:
  - Projects and scatters all $Q_A = 8$ verified draft KV states ahead to `W + static_pos_scatter` directly within the Stage 1 Verify CUDA Graph.
  - Completely eliminates eager Python `commit_accepted_tokens()` from the hot loop. On acceptance of $k$ tokens, only integer pointer `cur_pos += (k + 1)` advances; unaccepted speculative slots are masked by `static_mask_B` and overwritten in-place on the next verify pass.
- **`[0:W]` Scratch Tail Buffer Layout**:
  - `draft_cache` allocated with sequence capacity $W + \text{max\_len}$.
  - Indices $[0 : W]$ reserved exclusively for the current step's $W=7$ draft KV states. Committed prefix KV begins at offset $W$ (logical position $p \to W + p$).
  - `_step_draft_kernel` writes $k_d, v_d$ directly into `draft_cache[:, :, 0:W, :]` in-place, eliminating dynamic `torch.cat` operations during drafting.

### 2. Power-of-2 Bucketed CUDA Graphs
- **Zero-Copy Shared Buffers**: Slices master `target_cache` and `draft_cache` without dynamic reallocation.
- **4 Bucketed Graph Pairs**: Captures separate `(verify_graph, draft_graph)` pairs for $S_{\text{bucket}} \in (256, 512, 1024, 2048)$.
  - Verify views `target_cache[:, :, :S_bucket, :]` and `static_mask_A[:, :, :, :S_bucket]`.
  - Draft views `draft_cache[:, :, :W + S_bucket, :]` and `static_mask_B[:, :, :, :W + S_bucket]`.
- **Dynamic Dispatch**: Automatically replays the smallest bucket $S_{\text{bucket}} \ge \text{cur\_pos} + Q_A$, eliminating attention bandwidth waste over unpopulated sequence slots.

### 3. Inference-Only Merged 4-Layer Ghost Weights (`drafter.merge_ghost_weights()`)
- Deep-copies only the 4 tapped decoder layers into `fp16` ghost layers and folds each LoRA delta directly into its weight matrix:
  $$W_{\text{ghost}} = W_{\text{base}} + \frac{\alpha}{r} (B A)$$
- Eliminates 28 unfused `base_linear(x) + delta(x)` kernel launches per draft step while keeping the frozen base model weights 100% untouched for Stage 1 verify.

### 5. Empirical Validation & Device Auto-Adaptation

#### A. Systems Isolation (`hedge_margin = 0.0`)
- On NVIDIA Tesla T4 GPU (`Qwen2.5-1.5B`, 50 prompts, $C=128$, $B=1$, checkpoint `runs/bfg_v3_1.5b_15k/last.pt`), pure systems optimizations (WAL `draft_cache` + `[0:W]` scratch tail + key-only RoPE + merged 4-layer `fp16` ghost weights) reduced draft latency $T_{\text{draft}}$ by **33.5%** ($11.95\text{ ms} \to 7.95\text{ ms}$) and verification latency $T_{\text{verify}}$ by $2.90\text{ ms}$ ($36.49\text{ ms} \to 33.59\text{ ms}$).
- Increased T4 throughput from **61.6 TPS to 68.5 TPS** ($+6.9\text{ TPS}$, $2.64\times$ speedup vs. baseline) while maintaining identical mean acceptance $\alpha = 1.72$ and zero retraining.

#### B. `6+1` Leaf-Hedge Mask Morphing (`hedge_margin = 1.0`)
- In 50-prompt T4 GPU evaluation, leaf-hedging rescued **308 out of 1,768** low-margin slot-0 misses (**17.4% hit rate**).
- Lifted $A_{+1}$ acceptance rate by **+7.9 percentage points** ($49.8\% \to 57.7\%$), increased mean acceptance $\alpha$ from $1.72 \to 1.80$, and raised overall throughput to **70.3 TPS ($2.71\times$ speedup)** with zero extra draft forward passes.

#### C. Hardware Auto-Adaptation (`device.type`)
- Clean hardware split in `sampler_v3.py`:
  - **CUDA (`device.type == "cuda"`):** Strictly uses `fp16`, fused `fp16` ghost weights, WAL `draft_cache`, `[0:W]` in-place scratch tail, and Power-of-2 bucketed CUDA Graphs (`256, 512, 1024, 2048`) for minimal kernel launch overhead.
  - **CPU (`device.type == "cpu"`):** Strictly avoids empty static padding. Uses dynamic zero-padding KV slicing ($S_{\text{active}} = \text{max\_total\_len}$), pre-allocated static mask views, and in-place layer-by-layer INT8 quantization (`quantize_for_cpu`) with `gc.collect()`. Accelerates x86 Xeon throughput from 1.6 TPS to 5.5 TPS (3.44× speedup) and verification latency from 504.5 ms to 92.2 ms (5.47× faster) while staying comfortably under the `MemoryMax=9G` cgroup cap.

#### D. Production Promotion
- BFG-Diff v4 is promoted as the headline production inference engine across both GPU and CPU serving configurations. All existing v3 checkpoints load seamlessly with zero weight fine-tuning required.

#### E. CPU Lazy Draft KV Commit & Unified Speculative Streaming
- **CPU Lazy Commit (`commit_accepted_tokens`)**:
  - On CUDA, WAL writes all $Q_A$ candidates directly inside the verify CUDA graph for zero-overhead execution.
  - On CPU, WAL inside `forward_verify_step` is skipped. Instead, `commit_accepted_tokens(f_ctx_accepted, pos_accepted, scatter_pos)` projects draft K/V strictly for accepted tokens ($k + 1$ tokens on linear/Case A, $[y^*, d_{0, 2}]$ on Case B hedge hits, or $y^*$ on Case C misses), eliminating unneeded projections on CPU.
- **Auto-Adapted Draft Window $W$ ($v\_tokens$)**:
  - Parameterized `StaticDraftEngine` masks dynamically for arbitrary $W$:
    - CUDA defaults to $W = 7$ ($Q_A = 8$, $6+1$ Leaf-Hedge).
    - CPU defaults to $W = 3$ ($Q_A = 4$, $2+1$ Leaf-Hedge), cutting verification/draft FLOPs in half ($(4 \times 3 + 28 \times 4) = 124$ vs. 252 layer-tokens) and accelerating CPU serving throughput by +64% ($2.2 \to 3.6$ TPS).
- **Unified Streaming & Sampling Parity**:
  - `generate_v3_stream` directly invokes `generate_speculative_v3`, unifying streaming Web UI / CLI serving with full Leaf-Hedge support and stop delimiter handling.
  - Both speculative decoding and causal baseline share consistent temperature decoding (exact greedy `argmax` at $T=0.0$, multinomial sampling at $T>0.0$).
- **Modernized Web UI**:
  - Replaced legacy v2 parameters (`sub_canvas_len`, `sub_steps`, `tau`, `branching`) in settings modal and presets with active v4 parameters (`canvas_len`, `temperature`, `hedge_margin`, `v_tokens`).
  - Updated Manual tab documentation to detail the v4 speculative architecture.

### 6. Architectural and Runtime Hardening (v4 Post-Review)

#### A. Speculative Budgeting & Stop Contracts
- **Strict Token Budget Bounds**:
  - Off-by-one budget leak resolved: `cur_pos - P + 1` is strictly enforced across loop termination and per-row completion checks.
  - Sliced callback generation strictly bounds total emissions to `max_new_tokens`.
  - In-block stop token handling: speculative multi-token blocks truncate immediately at the first emitted stop delimiter across all hedge branches. Batched sequences ($B > 1$) right-pad with EOS to ensure uniform stacking.
- **Dynamic Mask ID & Zero-Allocation Graph Kernels**:
  - `StaticDraftEngine` accepts variable `mask_token_id` without breaking graph capture pointers, updating buffers in-place only upon integer divergence.
  - Eliminates per-step temporary allocations in the graph path via static pre-allocated index buffers and in-place `copy_().add_()` position offsets.

#### B. Memory Isolation & Backbone Registration
- **Backbone Submodule Decoupling**:
  - Base model submodules (`base_model`, `norm`, `lm_head`, `embed_tokens`, `rotary_emb`) are decoupled from PyTorch `nn.Module` tracking via `object.__setattr__`.
  - Drafter-level mutations (`.to()`, `.train()`, `.state_dict()`) strictly touch the 4-layer speculative adapter parameters (~78M parameters) without cloning or re-casting the 1.54B base model backbone.
  - `draft_layers` is structured as a dynamic `@property` over the live base layers, preventing stale references or memory leaks during CPU INT8 quantization.

#### C. Concurrency & Re-entrant Serving
- **Thread-Safe Re-entrant Generation**:
  - `ThreadingHTTPServer` endpoints (`/api/chat`, `/api/generate`) synchronize across model access and token generation using `threading.RLock()`.
  - Strict input validation clamps `canvas_len`, `messages`, `v_tokens`, `temperature`, and `hedge_margin` before acquiring the lock, rejecting malformed requests immediately.
  - Multi-engine capacity caching tracks `(v_tokens, capacity)` to eliminate re-initialization overhead across conversation turns.

#### D. Training Data Sanitation & Exact Gradient Accumulation
- **Sequence Boundary Masking**:
  - `generate_onpolicy.py` strips prompt left-padding, truncates sequences at the first generated EOS, and exports true sequence lengths (`seq_lens`).
  - `train.py` builds 2D `long` attention masks and constrains anchor sampling within strictly unpadded token spans (`hi = min_len - w_eff`).
- **Exact Gradient Accumulation & Full-Vocab Distillation**:
  - Nested micro-batch accumulation decouples gradient step increments from optimizer updates, ensuring the cosine learning rate schedule completes.
  - Shared full-vocab `logsumexp` unifies top-64 Kullback-Leibler divergence with Cross-Entropy loss under exact mathematical normalization.

### 8. BFG-Diff v5 Path B: Weight-Tied 2-Layer, 2-Step Differentiable Soft-Embedding Block Diffusion

#### A. Architectural Motivation & Budget
- **Layer & Parameter Reduction**: Shrinks the drafter stack from $K=4$ layers ($r=32$) to $K=2$ layers ($r=32$), reducing parameter footprint from ~6.6M to ~3.9M parameters while cutting single-pass draft forward compute in half.
- **Two-Step Differentiable Diffusion Cycle**:
  1. **Pass 1**: The bridge synthesizes initial draft representations $h_1 \in \mathbb{R}^{B \times W \times D}$ from $y^*$ and mask embeddings, which pass through the $K=2$ draft layers attending to prefix KV and their own $W$ block tokens.
  2. **Soft Refinement Bridge**: Soft predicted token distributions from Pass 1 (slots $0 \dots W-2$) are shifted by one position ($+1$) to align with candidate slots $1 \dots W-1$. Differentiable soft expected embeddings are computed via top-$k$ normalized softmax over the tied embedding matrix:
     $$\tilde{e}_j = \sum_{m=1}^k \text{Softmax}\left(\frac{z_{j-1}^{(m)}}{\tau}\right) W_\text{emb}[c_m]$$
     where slot 0 receives $y^*$ embedding.
  3. `SoftRefinementBridge` combines normalized shifted embeddings with slot embeddings and top-layer Pass 1 representations via an SiLU-activated bottleneck MLP ($\Delta h = W_\text{up} \cdot \text{SiLU}(W_\text{down} [ \text{norm}(e_2), \text{norm}(h_1) ])$ initialized near zero) added to $f_\text{ctx}^\text{last}$.
  4. **Pass 2**: Draft representations $h_2$ re-enter the identical $K=2$ draft layers, overwriting the `[0:W]` scratch tail in `draft_cache`.
- **Zero Additional CUDA Graph Overhead**: Pass 2 operates inside the exact same captured bucketed CUDA Graph on the identical `[0:W]` scratch memory, keeping host-side launch overhead at 0.

#### B. Distillation Training & Checkpoint Compatibility
- **Two-Pass Joint Loss Objective**:
  $$\mathcal{L}_\text{joint} = \lambda_\text{pass1} \mathcal{L}_\text{pass1} + \mathcal{L}_\text{pass2}$$
  with default $\lambda_\text{pass1} = 0.3$.
- **Chunked FP32 Anchor Distillation**: Anchor Top-64 KL and Cross-Entropy loss is computed over chunks of $N$ anchors (`_chunked_distill_loss`) in FP32, preventing memory spikes during batched 2-pass training.
- **Backward Compatibility**: Checkpoints save architecture metadata under `"arch"` (`k_layers`, `r`, `alpha`, `num_draft_steps`, `soft_topk`). Checkpoints without `"arch"` default cleanly to v4 configuration ($K=4, r=32, \text{steps}=1$).


#### A. Scaled Autocast Gradients
- `train.py` scales micro-batch gradients via `scaler.scale(loss / args.grad_accum).backward()` when `use_fp16=True` (CUDA), guaranteeing safe FP16 dynamic loss scaling across inner accumulation steps.

#### B. Frequency-Pruned Draft Vocabulary ($V_d = 32{,}768$)
- `tools/build_draft_vocab.py` extracts top frequent token IDs from on-policy rollouts while preserving essential control tokens (EOS, PAD, `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`), with deterministic backfill for small corpora.
- Slices drafter output projection to $V_d$ dimensions (`F.linear(h_norm, draft_lm_head_w)`).
- Local prediction indices in $[0, V_d - 1]$ remap cleanly back to global vocabulary token IDs via `engine.draft_vocab_ids[local_ids]` across both standard draft and leaf-hedge branching paths.
- Stage 1 verification retains the full 151,936 base model vocabulary projection, preserving exact lossless speculative decoding.

#### C. Hardware-Validated SDPA GQA Attention
- Replaces repetitive KV tensor expansions (`repeat_kv`) with PyTorch native Grouped-Query Attention (`enable_gqa=True`) in `F.scaled_dot_product_attention`.
- Guarded by runtime probe checking device, dtype, head configuration, and 4D additive float attention masks (`attn_mask`) against reference outputs (`max_abs_diff <= 1e-2`), automatically falling back to `repeat_kv` on non-compliant backends.

#### D. Generic Static-Shape Tree Verifier ($Q_A \in \{8, 16\}$)
- Decouples verification tile size $Q_A$ from the draft sequence window $W$ (`q_a: Optional[int] = None`, defaulting to $W + 1$). Allows high-throughput $Q_A = 16$ tree verification tiles on CUDA while keeping $W = 7$ drafter architecture intact (`slot_emb` size 8 untouched).
- Dynamic topological tree generator (`build_topk_tree`) expands top-$k$ branch candidates from draft logits using best-first priority queue search, constructing an arbitrary tree candidate tile of size $Q_A$ with strict topological ordering (`parent[i] < i`).
- Invariant CUDA Graph Replay: because positional RoPE embedding in `_step_verify_kernel` depends on `static_pos_verify` ($t + \text{depth}$) while KV cache scatter uses $t + \text{slot}$, the exact same captured bucketed CUDA Graph executes any tree topology of size $Q_A$ without recompilation.
- Overwrite-Safe Ascending KV Compaction: accepted root-to-leaf paths $[0, n_1, \dots, n_k]$ satisfy $n_m \ge m$. Compacting KV cache entries in ascending order ($t + n_m \to t + m$) guarantees zero overwriting of uncompacted slots, updating both target KV cache and CUDA draft KV cache in place.

#### E. T4 Benchmark Rigor & Margin Diagnostics
- Implemented `GraphedCausalRunner`: base-model-only bucketed CUDA Graph runner ($Q=1$) using identical static KV cache, bucket dispatch, and memory layout to report true system speculative speedup without python/framework dispatch overheads.
- First-divergence diagnostic logger: on greedy baseline mismatches, extracts context up to the divergence step, evaluates base model logits, and outputs the top-2 logit margin to identify floating-point boundary sensitivity.
- Exposed `--q_a` and `--tree_mode` (`auto`, `linear`, `tree`) CLI options in `benchmark.py`.






