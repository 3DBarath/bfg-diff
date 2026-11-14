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

For every sample independently, draw `t ~ U(0.15, 0.95)` and mask exactly
`floor(t*C)` positions. Exact counts matter for a predictable difficulty
schedule; independent per-position Bernoulli draws would only match `t*C` in
expectation.

Loss is plain cross-entropy on masked positions only. No weighting, no
reconstruction of unmasked tokens. The model is not trained to copy clean
tokens, so the objective at `t → 0` is undefined by construction; the 0.15
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

## Cost model

A diffusion step is one forward over `P+C` tokens and produces `C` positions at
once, so `K` steps cost `K` forwards. Autoregressive generation costs one
forward per new token, growing its context each time. The diffusion decoder is
therefore much cheaper per token at small `K`, at the price of needing a model
fine-tuned for bidirectional denoising. Logits for a canvas step are
`(B, C, vocab)`, which is the dominant memory term at large `C`.
