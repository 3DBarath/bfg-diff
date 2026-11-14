# Findings

All measurements are from 2026-11-15 on a CPU-only box: 20 cores, 31 GB RAM,
no CUDA device, `torch 2.14.1+cpu`, `transformers 5.20.2`, Python 3 from
`AD/.venv`. The package is written for CUDA bf16, but nothing below exercises
that path.

## Environment facts confirmed

- `Qwen/Qwen2.5-0.5B`: 24 layers, hidden size 896, `vocab_size = 151936`.
- Its tokenizer emits ids up to 151664 (22 added special tokens, 151665 entries
  total). `<|extra_0|>` is not present in `tokenizer.json`.
- After aliasing, `<|extra_0|>` resolves to **151665**, i.e. the first spare row
  of the existing embedding matrix. `len(tokenizer) = 151666`, vocabulary size
  stays 151936, no resize and no reinitialised weights.
- Padding token is `<|endoftext|>`, so left-padded benchmark prompts use a valid
  id; padded prefix keys are masked out separately in the attention mask.

## Training smoke test

Command: `python train.py --smoke_test` (12 steps, synthetic word-salad text,
`P=128`, `C=64`, batch 2, fp32, 140 train / 8 val chunks). Runtime ~95 s.

```
smoke probe loss (before): 5.7692
step     1/12 loss 14.5084 lr 5.00e-05
step     2/12 loss  4.9111 lr 4.88e-05
step     3/12 loss  8.3438 lr 4.52e-05
step     4/12 loss  7.2797 lr 3.97e-05
step     5/12 loss  5.8685 lr 3.27e-05
step     6/12 loss  4.6176 lr 2.50e-05
step     7/12 loss  4.4995 lr 1.73e-05
step     8/12 loss  3.9444 lr 1.03e-05
step     9/12 loss  4.1271 lr 4.77e-06
step    10/12 loss  3.7695 lr 1.22e-06
step    11/12 loss  3.5346 lr 0.00e+00
step    12/12 loss  3.7620 lr 0.00e+00
step    12 val_loss 3.7360
smoke convergence: 5.7692 -> 3.8388 PASS
```

Notes:

- The per-step loss swings because `t` is redrawn per sample each step. It is
  not a convergence signal.
- The fixed-batch, fixed-50%-mask probe is the signal: 5.7692 → 3.8388.
- An earlier run with an 8-step schedule ended at 3.49 loss / 3.51 val, so the
  drop is reproducible within this toy setting.
- CPU throughput was 65–76 tokens/s at `B=2`, `T=192`, including optimizer time.
  One step costs roughly 5–9 s once the process is warm.

## Benchmark smoke test

Command: `python benchmark.py --smoke_test` (2 prefixes, `C=16`, k=2, 12-step
checkpoint). Runtime ~17 s.

```
Reconstruction
bfg-diff k=2     recovery 5.3%   16.4 TPS

Generation
bfg-diff k=2     4-gram H 0.532   repeat 46.2%   38.1 TPS
causal generate  4-gram H 1.000   repeat  0.0%   13.3 TPS
```

How to read this:

- 5.3% recovery is far above the ~0.0007% chance rate, but a 12-step checkpoint
  has not learned denoising. Do not use these numbers as quality results.
- The repeated 4-grams and low entropy of the k=2 sampler are exactly the
  degeneration the margin ordering is meant to fight; two steps are too few and
  the model is undertrained. The k=4/k=8 defaults on a trained model are the
  intended setting.
- TPS includes prompt tokenisation and batch padding. Causal generation is
  slower by ~2.9x here even though each diffusion step itself is a full
  `P+C` forward.
- Extrapolating the measured rates (optimistically, since causal context grows):
  100 prompts x 128 tokens is roughly 16 min for `generate()` and 5–6 min for
  k=4 diffusion on this CPU. Use a GPU or lower `--num_prompts`/`--canvas_len`
  for exploratory runs.

## Bugs found and fixed

1. **Frozen tokens overwritten on a partially filled canvas.**
   `sample_canvas_block` computed `topk(quota)` over all `C` slots. With a
   pre-filled canvas (reconstruction), `quota` could exceed the number of
   masked slots, so `-inf` entries (already-unmasked tokens) were selected and
   replaced. Fix: per sample, take `n = min(quota, masked_count)` and scatter
   only those positions. A unit test now asserts that pre-filled slots keep
   their exact values through sampling.

2. **LR scheduler warning and schedule skew under gradient accumulation.**
   Calling `scheduler.step()` every micro-batch before `optimizer.step()`
   triggered PyTorch's ordering warning, and cosine progress was tied to
   micro-steps rather than updates. Fix: scheduler steps once per optimizer
   update; total updates are `ceil(steps / grad_accum)`; the lambda uses
   `update + 1` so the first update does not run at lr 0.

3. **Noisy smoke convergence check.**
   Comparing the mean loss of the first and last training steps initially
   failed or passed by luck because `t` changes every step. Fix: a probe batch
   with a fixed 50% mask is evaluated before and after training; the smoke test
   exits non-zero unless the probe loss falls.

4. **CPU dtype.** Auto-selecting bf16 on CPU makes the smoke run slower and
   tests an emulated path; the loader now chooses fp32 on CPU and bf16 on CUDA
   unless overridden with `--dtype`.

## Open questions / not verified

- bf16 numerics, SDPA mask handling, and memory use on a real GPU.
- Whether `recovery` and `repeat` separate meaningfully after long training;
  the smoke run only proves the plumbing.
- Reference texts repeat every 10 prompts in the 100-prompt benchmark, so
  per-prefix scores are correlated.
- No comparison against standard block-autoregressive decoding or other
  diffusion samplers; only HF greedy `generate()`.
- The 0.15–0.95 corruption range is fixed as specified, not tuned.
