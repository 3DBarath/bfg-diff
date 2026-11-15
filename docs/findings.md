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

## CPU sprint: preset, corpus, floor

Environment: same CPU-only box, Intel Xeon Gold 5215 (10 cores / 20 threads).
`avx512f` only — no `avx512_bf16`, no AMX — so fp32 is the fast path and bf16
would be emulation.

- Corpus: `/usr/share/doc` filtered to `README*`, `*.md`, `*.txt`, 2 KB–200 KB,
  depth 3, symlinked into `/tmp/bfg-corpus` (1006 files, 13 MB). Capped at
  2,000,000 tokens → 62,432 windows at `P=128, C=32`.
- Threads: 4 steps took 31.8 s with `--threads 10` and 31.6 s with
  `--threads 20`, so there is no benefit from hyper-threads here. Peak RSS for
  model + optimizer + data was 10.2 GB.
- Per-step time is about 3 s at `B=2, T=160` before gradient accumulation, i.e.
  roughly 45–55 min for the 1000-step preset including five checkpoints.
- `margin_floor` unit test: with `floor=1e9`, pre-filled slots are untouched and
  the rest are completed by the final argmax fill; with `floor=0.0`, behavior
  matches the old rigid quota.

The full 1000-step run and the k/floor sweep are reported below.

### 1000-step CPU run

```bash
python train.py --cpu --threads 10 --data /tmp/bfg-corpus \
  --max_tokens 2000000 --out runs/bfg-diff
```

- Corpus: 1006 filtered documentation files, capped at 2,000,000 tokens
  (62,432 overlapping windows, 64 held out for validation).
- Wall time: 3195 s (53 min) at ~100 tokens/s; 10.5 GB peak RSS.
- Train loss (20-step mean, noisy because `t` changes): 5.23 at step 100,
  5.20 at 200, 4.73 at 500, 4.52 at 1000.
- Validation loss: 4.97 (200), 4.85 (400), 3.94 (600), 3.64 (800), 4.06 (1000).
  The validation set is only 64 windows and each evaluation draws a fresh 50%
  mask, so the last two points are within noise of each other; `ckpt-800.pt` is
  not obviously worse than `last.pt`.
- Checkpoints: `ckpt-200/400/600/800.pt` and `last.pt`, 1.9 GB each (fp32).

### Benchmark of the 1000-step checkpoint

```bash
python benchmark.py --checkpoint runs/bfg-diff/last.pt --num_prompts 12 \
  --prefix_len 128 --canvas_len 32 --batch_size 2 \
  --steps 2 4 8 --floors 0.0 1.0 --show 3 --threads 10
```

Reconstruction (50% of the canvas masked, 12 prefixes):

| method | recovery | TPS |
|---|---|---|
| bfg-diff k=2 | 63.9% | 33.4 |
| bfg-diff k=4 | 81.4% | 16.9 |
| bfg-diff k=8 | 90.2% | 8.5 |

Generation (12 prompts, 32 tokens each):

| method | H4 | repeat | Dist-1 | Dist-2 | TPS |
|---|---|---|---|---|---|
| bfg-diff k=2 | 0.523 | 56.0% | 0.083 | 0.183 | 92.9 |
| bfg-diff k=4 | 0.548 | 54.6% | 0.099 | 0.196 | 46.7 |
| bfg-diff k=8 | 0.757 | 37.6% | 0.141 | 0.288 | 23.4 |
| bfg-diff k=2, f=1 | 0.523 | 56.9% | 0.096 | 0.191 | 63.5 |
| bfg-diff k=4, f=1 | 0.508 | 57.2% | 0.104 | 0.196 | 37.9 |
| bfg-diff k=8, f=1 | 0.571 | 51.1% | 0.115 | 0.218 | 20.9 |
| causal generate | 0.938 | 13.5% | 0.362 | 0.578 | 25.6 |

Interpretation:

- **Infilling works, free generation does not.** Recovery rises 63.9% → 90.2%
  with more denoising steps, so the model learned to use prefix and unmasked
  canvas context. But full-canvas generation collapses to "the the the"
  (repeat 56% at k=2). Example: `"the fish of the the the the..."`; causal
  generation produced a coherent sentence. Training on 2M tokens for 1000
  steps, with `t` up to 0.95, rewards local copying rather than long-range
  planning.
- **More steps help.** k=8 is the best diffusion setting (H4 0.757, repeat
  37.6%, Dist-2 0.288), still well below causal (H4 0.938, repeat 13.5%).
- **The margin floor did not help at 1.0.** Repeat rate moved 54.6% → 57.2% at
  k=4 and 37.6% → 51.1% at k=8; Dist-1 was flat or slightly worse. It also
  costs an extra forward for the final argmax fill (TPS drops ~20%). The
  bottleneck here is the learned conditional distribution, not the rigid quota.
  A higher floor or a stronger checkpoint would be needed before concluding the
  mechanism is useless.
- **Throughput depends on canvas/prompt length.** At C=32, k=2 and k=4 beat
  causal `generate()` (92.9 and 46.7 vs 25.6 TPS); k=8 is slower. The earlier
  C=16 smoke had k=2 at ~2.9x causal.

### Sampler-only fixes on `last.pt` (no retraining)

Sweep at k=8 (12 prompts, C=32), adding Gumbel temperature and the step-1
position-decay prior:

| config | H4 | rep4 | Dist-1 | Dist-2 | TPS |
|---|---|---|---|---|---|
| T=0, d=0 (greedy) | 0.757 | 37.6% | 0.141 | 0.288 | 23.2 |
| T=0, d=1.5 | 0.762 | 38.2% | 0.130 | 0.277 | 23.2 |
| T=0.7, d=0 | 1.000 | 0.0% | 0.385 | 0.849 | 17.8 |
| T=0.7, d=1.5 | 0.996 | 0.9% | 0.354 | 0.820 | 17.8 |
| causal generate | 0.938 | 13.5% | 0.362 | 0.578 | 25.8 |

Decoded samples for `Once upon a time, ... the fishermen noticed that`:

- greedy: `the fish of the sea the the sea of the sea of the sea ...`
- Gumbel T=0.7: `the fish the day, the a fish of.The,, deep kept, it that, ...`
- decay + Gumbel: `a new of the fish the fish are more exposed to a a new species ...`
- causal: `the fish they caught were smaller than the ones they caught in the sea.`

Interpretation:

- **Gumbel removes repetition but not the underlying problem.** The metrics look
  excellent (rep 0%, Dist-2 0.85) because sampling near-flat distributions
  produces mostly unique tokens; the text is word salad. Diversity metrics
  cannot be read without the decoded samples — random tokens score best.
- **Position decay alone does nothing here** (38.2% vs 37.6% repeat). With
  C=32, k=8 only four slots are picked on step 1, and the model already prefers
  early positions. Combined with Gumbel it changes the sample but does not
  improve it.
- **Gumbel costs throughput** (23.2 → 17.8 TPS) because it materialises a
  full `(B, C, vocab)` noise tensor each step.
- The bottleneck is the checkpoint's conditional distribution, not the
  unmasking order. The implemented 20%-`t=1` training change is the right next
  experiment; sampler tweaks alone will not produce coherent prose.

## Open questions / not verified

- bf16 numerics, SDPA mask handling, and memory use on a real GPU.
- Infilling separates from generation strongly (90% recovery vs 56% repeat),
  but 1000 steps on 2M tokens is not enough for coherent free generation.
  Longer training, larger `C`, or a curriculum on `t` are the obvious next
  levers.
- The margin floor needs a sweep above 1.0 (2.0, 3.0) and a stronger checkpoint
  before judging the idea; at 1.0 it did not beat the rigid quota.
- Gumbel sampling below T=0.7 (e.g. 0.3-0.5) might trade some diversity for
  coherence; only 0.7 was measured, and it produced word salad.
- The 20%-full-mask training change is implemented but not yet trained.
- Reference texts repeat every 10 prompts in the benchmark, so per-prefix scores
  are correlated; only 12 prompts were used in the timed sweep.
- No comparison against standard block-autoregressive decoding or other
  diffusion samplers; only HF greedy `generate()`.
- The 0.15–0.95 corruption range is fixed as specified, not tuned.
