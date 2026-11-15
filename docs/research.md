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

### Margin floor

A rigid linear quota forces the model to commit to `C/K` tokens per step even
when it is unsure. Early wrong commitments cannot be revised, and repeated
phrases are the usual symptom.

`margin_floor` adds an absolute gate: a masked slot is eligible for unmasking
only if `margin >= floor`. The per-step budget stays linear
(`quota_s - quota_{s-1}`), so held-back slots are not dumped later; they remain
masked until the final argmax fill. This trades a little latency for fewer
early mistakes. A high floor can leave most of the canvas to the final fill, so
the floor is meant to be small (roughly 0.5-2.0 in log-prob units) and tuned in
`benchmark.py --floors`.

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

## Cost model

A diffusion step with the cache is one forward over `C` tokens plus a one-off
`P`-token prefill, so `K` steps cost `P + K*C` token evaluations. Causal
generation costs `P + C` but strictly sequentially: 32 kernel launches of one
token each. The diffusion decoder is therefore cheaper per token at small `K`
and parallel across the canvas, at the price of needing a model fine-tuned for
bidirectional denoising. Logits for a canvas step are `(B, C, vocab)`, which is
the dominant memory term at large `C`.
