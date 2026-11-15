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
4. **Sliding Sub-Blocks:** To generate sequences of arbitrary length without quadratic memory or unanchored degradation, `sample_canvas` slides across sub-blocks ($C_{\text{sub}}=8$ or $16$) that are committed into the causal prefix KV cache as they finish.

