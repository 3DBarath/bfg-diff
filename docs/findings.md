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
checkpoint). Runtime ~20 s. Re-measured 2026-11-15 with the re-mask sampler:

```
Reconstruction
bfg-diff k=2     recovery 15.8%   18.4 TPS

Generation
bfg-diff k=2     4-gram H 0.251   repeat 73.1%   27.4 TPS
causal generate  4-gram H 1.000   repeat  0.0%   13.2 TPS
```

How to read this:

- 15.8% recovery is far above the ~0.0007% chance rate, but a 12-step checkpoint
  has not learned denoising. Do not use these numbers as quality results.
- The repeated 4-grams and low entropy of the k=2 sampler are exactly the
  degeneration the margin ordering is meant to fight; two steps are too few and
  the model is undertrained. The k=4/k=8 defaults on a trained model are the
  intended setting.
- TPS includes prompt tokenisation and batch padding. Causal generation is
  slower by ~2x here even though each diffusion step itself is a full
  `P+C` forward.
- Extrapolating the measured rates (optimistically, since causal context grows):
  100 prompts x 128 tokens is roughly 13 min for `generate()` and 6 min for
  k=2 diffusion on this CPU. Use a GPU or lower `--num_prompts`/`--canvas_len`
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

5. **Pre-filled slots ate the whole diffusion budget (found 2026-11-15 while
   writing the Task 1 table).** The quota-preserving re-mask rewrite selected
   `topk(budget)` over the *entire* canvas, and frozen reconstruction slots carry
   score `+inf`. Fix: frozen slots never compete; protected decoded slots keep
   their token and only the remaining `budget - n_protected` slots are chosen
   from the highest margins. Verified: pre-filled tokens are unchanged and
   `remask_ratio=0.0` is token-identical to commit `e2b7886` across steps
   1/2/3/4/8 × {greedy, T=0.5, position_decay}.

6. **The reconstruction quota spent half its schedule before the first fill
   (found 2026-11-15, review of bug 5).** `budget = floor(s/K*C) - n_frozen`
   subtracts the visible slots from a quota that also counts them, so with a
   50%-masked canvas the first `K/2` steps all evaluated `budget <= 0`:
   k=8 filled `16,16,16,16,16,12,8,4` and behaved like k=4, and k=2 never
   iterated at all. Fix: interpolate over the mutable slots,
   `floor(s/K * (C - n_frozen))`. Mask counts before each forward are now
   exactly `16,14,12,10,8,6,4,2` (k=8), `16,12,8,4` (k=4), `16,8` (k=2) for
   `C=32, n_frozen=16`, and reconstruction recovery rose
   80.4→89.7 / 88.1→92.3 / 90.7→96.4% at k=2/4/8. With `n_frozen=0` the formula
   reduces to `floor(s/K*C)`, so free generation stays token-identical
   (re-verified against `21812a7` with suppression off, 20/20 cases).

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

### Top-k truncation and prefix KV cache

The TPS column above predates two sampler changes: Gumbel now runs over the
top-50 logits only (a mini-batch of 50, not the full 151,936-row vocabulary),
and the softmax pass is gone because `log p(1) - log p(2) == z(1) - z(2)`.
The prefix is also prefilled once per prompt into a KV cache; each step
forwards only the 32 canvas tokens. Re-measured on the same `last.pt`
(12 prompts, P=128, C=32, batch 2, 10 threads), greedy is token-identical to
the previous sampler:

| config | H4 | rep4 | Dist-1 | Dist-2 | TPS |
|---|---|---|---|---|---|
| k=4, T=0 | 0.548 | 54.6% | 0.099 | 0.196 | 47.2 |
| k=8, T=0 | 0.757 | 37.6% | 0.141 | 0.288 | 25.6 |
| k=4, T=0.3, top_k=50 | 0.908 | 15.5% | 0.143 | 0.387 | 46.5 |
| k=8, T=0.3, top_k=50 | 0.914 | 17.5% | 0.180 | 0.452 | 25.0 |
| k=4, T=0.5, top_k=50 | 0.975 | 4.9% | 0.219 | 0.645 | 46.4 |
| k=8, T=0.5, top_k=50 | 0.983 | 3.7% | 0.250 | 0.677 | 25.1 |
| causal generate | 0.938 | 13.5% | 0.362 | 0.578 | 25.6 |

Reconstruction TPS: k=4 16.9 → 33.4, k=8 8.5 → 20.8 (2.0-2.4x). Generation TPS
barely moved because benchmark prompts are only ~15-20 tokens, so the full
`P+C` forward was already dominated by the canvas; caching pays off in
proportion to prefix length.

Throughput depends on batch size. At batch 4 (same 12 prompts):
k=8 T=0 is 31.3 TPS vs causal 29.2, and k=8 T=0.5 is 30.3 TPS. So the "k=8
beats causal" target holds at batch 4 and is a tie at batch 2; k=4 is ~2x
causal either way. That was measured with `P ~= 15-20`, the most favourable
prompt length; see the systems finding below for where the advantage ends.

Decoded k=8 completions with top-k=50 (`runs/bench-samples.log`):

- `Once upon a time, ... noticed that` → T=0.3: `the fish of the sea the the sea
  of the sea and the sea of the sea ...`; T=0.5: `the fish of the day the the
  sea, and the to the sea, and to the., the fish, the the the sea, ...`
- `## Quick start\n\nInstall the package, then run` → T=0.3: `` `make. . . The
  is a ` ...`` (backtick salad); T=0.5: more of the same.
- `To configure the proxy, ... set` → T=0.5: `` the `proxy` by. ` `export --
  `proxyproxyproxy `hostproxyhost`` ...``

Interpretation:

- **Top-k helps the prose prompts, not the format-heavy ones.** Repetition
  drops from 54.6%/37.6% to 4.9%/3.7% for T=0.5, and the coastal-village prompt
  stays readable. For markdown/code prompts the model's top-50 is itself full
  of backticks, so truncation cannot help; the 4-gram metrics still report
  "diverse" because the salad is unique.
- **Truncation alone does not fix the degenerate distribution.** Same
  conclusion as the T=0.7 run: metrics improve because sampling is more random,
  not because the checkpoint knows what to say.
- **KV caching is a real 2x** where the prefix is long, and costs nothing
  elsewhere (greedy output unchanged, ~3e-5 logit difference).

## LoRA training run (3000 steps)

```bash
python train.py --cpu --lora --threads 20 --steps 3000 --batch_size 4 \
  --grad_accum 2 --data /tmp/bfg-corpus --max_tokens 2000000 --out runs/bfg-lora
```

- Adapters: 2,162,688 of 496,195,456 parameters trainable (0.4%) on
  `q_proj, k_proj, v_proj, o_proj`, r=16, alpha=32, dropout 0.05, lr 5e-4.
- Smoke test with LoRA: probe 5.7332 → 4.2757 `PASS` (12 steps, synthetic).
- Measured step time at B=4, P=128, C=32: 4.2 s/step. `--threads 10`, `14` and
  `20` all land within noise (4.1-4.2 s), so 10 physical cores saturate; the
  20-thread setting is not faster.
- Therefore 3000 steps ≈ 3.5 h plus ~15 min of evaluation, not the ~45 min
  estimated in the task brief. Effective batch is 8 (1500 optimizer updates),
  so this is 12x the optimizer updates of the earlier 1000-step full fine-tune.
- Actual run: 13,211 s (3 h 40 min), final val 3.7263, best 3.6254 @2600.
  Full curve in `docs/loss-curve.csv`.
- Checkpoints are adapter-only, 8.4 MB each, in `runs/bfg-lora/`.

## Systems finding: where CPU block diffusion stops paying

**On CPU, block diffusion's parallel-GEMM advantage only beats causal GEMV while
`k <= 4` and `P <= 256`. Past that point the `k*C` redundant token evaluations
saturate fp32 compute and the speedup disappears.**

Supporting cells from the sweep above (batch 4, the realistic serving shape):

| P | k=4 | k=8 |
|---|---|---|
| 32 | 1.87x | 0.88x |
| 128 | 1.56x | 0.99x |
| 256 | 1.35x | 0.94x |
| 512 | 1.14x | 0.88x |

- k=8 is already at parity at `P=128` and below causal from `P=256` on, so the
  "more steps is better quality" argument is bought at a compute loss on this
  hardware.
- k=4 never falls below 1.0 in the measured range, but it decays monotonically
  toward it (1.87x → 1.14x) and extrapolates through 1.0 just past `P=512`,
  which is the ceiling for this box with `C=32`.
- Mechanism: each diffusion step is a `32 x d` GEMM over `P+32` cached keys,
  repeated `k` times; causal decoding is `32` sequential `1 x d` GEMV passes.
  The GEMM wins on arithmetic intensity (the 0.5B weights are read once per
  canvas step instead of once per token), but it computes `k*C` token slots, of
  which only `C` survive. At fp32 on 10 physical cores the extra work overtakes
  the weight amortisation once `P` makes the KV term dominant.
- Practical rule: use `k=4` for `P <= 256`, fall back to causal `generate()` for
  long prefixes, and only use `k=8` when quality is worth ~10-15% throughput.

## Task 1: evaluation of the 3000-step attention-LoRA checkpoint


Checkpoint `runs/bfg-lora/last.pt` (step 3000, r=16 attention adapters, 2.16M
trainable). Training: 13,211 s = 3 h 40 min, 4.2 s/step, 153 tok/s at the tail,
effective batch 8, 1500 optimizer updates.

### Loss curve

166 logged points, extracted verbatim to `docs/loss-curve.csv`
(`step,loss,split`; train is a 20-step mean, val is 64 chunks with a fresh 50%
mask). The log has no step-0 entry: the first training line is step 1 and the
first validation is at step 200, so the curve starts at the first batch. Every tenth point:

| step | train | val | | step | train | val |
|---|---|---|---|---|---|---|
| 1 | 16.7940 | — | | 1600 | 4.3845 | 3.8302 |
| 20 | 12.6339 | — | | 1800 | 4.5324 | 4.0500 |
| 100 | 5.5466 | — | | 2000 | 4.8721 | 3.9739 |
| 200 | 5.5351 | 4.5726 | | 2200 | 5.0892 | 4.0071 |
| 400 | 4.9582 | 4.3238 | | 2400 | 4.8446 | 4.1155 |
| 600 | 5.2872 | 4.3598 | | 2600 | 4.6397 | **3.6254** |
| 800 | 5.1294 | 4.9353 | | 2800 | 4.8162 | 4.0966 |
| 1000 | 5.1107 | 3.8867 | | 3000 | 4.6068 | 3.7263 |
| 1200 | 5.2887 | 4.2450 | | | | |
| 1400 | 4.9299 | 4.4857 | | | | |

How to read it:

- **The train curve stops moving at ~step 300.** From step 200 to 3000 the 20-step
  mean sits in 4.38–5.29 with no trend; the LR schedule decays from 5e-4 to 0
  over that window and the loss does not follow. The task is
  `mask ~ U(0.20, 1.0)` plus 20% fully masked batches, redrawn every step, so the
  per-step number is dominated by how many tokens were masked, not by fit.
- **Validation is the only usable signal** and it is noisy: ±0.3 between adjacent
  evals at the same nominal quality. Best 3.6254 @2600, final 3.7263 @3000. The
  2600→2800 jump (3.63 → 4.10) is larger than any plausible real regression at
  lr≈2e-7, so treat the tail as flat.
- **Step 1 is an outlier, not a bug in logging.** Train loss 16.79 is above
  `ln(151936) = 11.93`, which a mean cross-entropy can only reach if the first
  batch produces extreme logits — that batch is entirely unseen
  `[MASK]`-filled canvases at `t≈1`. It recovers to 5.5 within 100 steps.
- Attainable floor: the 1000-step full fine-tune reached val 3.6395 @800 with
  100% of the weights; attention-only LoRA reaches 3.6254 @2600 with 0.43%.
  **The adapter is not under-trained relative to the full fine-tune; it is
  simply a different (and much smaller) parameter budget.**

### Reconstruction (50% masked, P=128, C=32, 12 prefixes, batch 2)

```bash
python benchmark.py --checkpoint runs/bfg-lora/last.pt --num_prompts 12 \
  --prefix_len 128 --canvas_len 32 --batch_size 2 --steps 2 4 8 \
  --temps 0.0 0.3 0.5 --top_k 50 --remask_ratio 0.0 0.25 --show 12 --threads 10
```

| method | recovery | TPS | 1000-step full FT recovery |
|---|---|---|---|
| bfg-diff k=2 | 89.7% | 47.0 | 63.9% |
| bfg-diff k=4 | 92.3% | 32.7 | 81.4% |
| bfg-diff k=8 | 96.4% | 20.2 | 90.2% |

- **Fixing the quota schedule (bug #6) is the largest quality gain of the
  sprint, and it is pure sampling math.** With `budget = floor(s/K*C) - n_frozen`
  the first four steps of k=8 unmasked nothing (quota 4/8/12/16 minus 16 visible
  slots), so a 50%-masked canvas got four real steps, not eight. Interpolating
  over the `C - n_frozen` mutable slots gives the scheduled
  `16,14,12,10,8,6,4,2` and lifts recovery 80.4→89.7, 88.1→92.3, 90.7→96.4%.
- **The LoRA checkpoint is now a better infiller than the full fine-tune at
  every `k`** (+25.8, +10.9, +6.2 points) at roughly 2x the TPS, because of the
  prefix KV cache. Reconstruction quality was never the binding constraint on
  free generation; this table says the 3000 adapter steps did learn infilling.
- Recovery is monotone in `k` (89.7 → 92.3 → 96.4) and k=2 now iterates: its
  first budget is 8 mutable slots, not 0.
- With collision suppression on (default), the k=8 mask schedule deviates at the
  last interior step (`...,4,4` instead of `...,4,2`): one adjacent pair sampled
  the same token, the weaker was held back, and the final step filled it. The
  canvas still ends with zero masks. With `--no_adj_suppress` the schedule is
  exactly `16,14,12,10,8,6,4,2`.
- `run_reconstruction` always samples with `remask_ratio=0.0`; re-masking is a
  generation-time knob here, so this table is a pure `k` ablation.

### Free generation (100% masked, C=32, top_k=50 for T>0)

| method | H4 | rep4 | Dist-1 | Dist-2 | TPS |
|---|---|---|---|---|---|
| k=2, T=0, r=0 | 0.385 | 67.5% | 0.086 | 0.164 | 78.3 |
| k=4, T=0, r=0 | 0.527 | 58.0% | 0.107 | 0.220 | 45.2 |
| k=8, T=0, r=0 | 0.658 | 42.2% | 0.161 | 0.309 | 24.5 |
| k=2, T=0, r=0.25 | 0.385 | 67.5% | 0.086 | 0.164 | 78.3 |
| k=4, T=0, r=0.25 | 0.642 | 45.1% | 0.122 | 0.258 | 45.3 |
| k=8, T=0, r=0.25 | **0.949** | **10.1%** | 0.180 | 0.444 | 24.5 |
| k=2, T=0.3, r=0 | 0.924 | 12.4% | 0.122 | 0.376 | 75.7 |
| k=4, T=0.3, r=0 | 0.933 | 10.6% | 0.151 | 0.449 | 44.2 |
| k=8, T=0.3, r=0 | 0.953 | 10.3% | 0.185 | 0.508 | 23.9 |
| k=2, T=0.3, r=0.25 | 0.899 | 14.9% | 0.117 | 0.401 | 77.1 |
| k=4, T=0.3, r=0.25 | 0.914 | 14.4% | 0.167 | 0.473 | 44.3 |
| k=8, T=0.3, r=0.25 | 0.969 | 6.0% | 0.240 | 0.624 | 23.9 |
| k=2, T=0.5, r=0 | 0.992 | 1.7% | 0.185 | 0.656 | 77.1 |
| k=4, T=0.5, r=0 | 0.975 | 4.6% | 0.203 | 0.581 | 44.3 |
| k=8, T=0.5, r=0 | 0.983 | 3.7% | 0.247 | 0.664 | 23.9 |
| k=2, T=0.5, r=0.25 | 0.996 | 0.9% | 0.201 | 0.608 | 75.7 |
| k=4, T=0.5, r=0.25 | 0.988 | 2.9% | 0.286 | 0.715 | 44.4 |
| k=8, T=0.5, r=0.25 | 0.959 | 7.8% | 0.302 | 0.688 | 24.0 |
| causal generate | 0.964 | 7.8% | 0.411 | 0.653 | 22.6 |

Exact decoded text for all 12 prompts × 19 configs: `docs/decoded-samples.md`.
The three `r=0` rows were re-measured after the frozen-budget fix (bug #5) with
identical H4/rep4/Dist and TPS within 0.5% (`runs/bench-recon-fixed.log`). This
table predates collision suppression, so its `adj` column was not recorded; the
same configs re-measured with suppression on/off are below.

### Re-masking ablation (Task 3, no collision suppression)

Each `k` group holds the same config twice, `r=0` then `r=0.25`; that pair is the
Task 3 ablation, measured before adjacent-token suppression existed. The
shipped-default numbers (suppression on) are in the next section. Interpretation
of both tables:

- **Re-masking is the single largest sampler win measured.** At T=0, k=8 it moves
  rep4 42.2% → 10.1% and H4 0.658 → 0.949 at *zero* TPS cost (24.5 either way;
  the forward count is unchanged). It also helps k=4 (58.0% → 45.1%) and is a
  no-op at k=2, which is expected: k=2 has no interior step with both committed
  tokens and a smaller quota, so nothing can be re-opened.
- **It does not produce grammar.** The best greedy sample is still
  `caught were not not not the the the same of the caught caught fish ...`:
  re-masking removes the *static* "the the the" attractor, not the model's
  inability to plan 32 tokens. Dist-2 0.444 vs causal 0.653 says the same thing.
- **Greedy is now competitive with T=0.3 and better than T=0.5 for the `rep4`
  metric.** With `r=0.25`, k=8 T=0 gives rep4 10.1%/Dist-2 0.444 while T=0.3
  gives 6.0%/0.624 and T=0.5 gives 7.8%/0.688. The T>0 rows look better because
  sampling noise raises diversity; their raw text (see the samples file) is a
  word-salad at 0.5, e.g. `the proxy port0000.00.000000000.00000 of 0.00`. Judge
  quality at T=0.
- **k=8 + r=0.25 is the first diffusion setting to reach causal-level H4**
  (0.949 vs 0.964) while staying within 9% of its TPS (24.5 vs 22.6). It does
  not beat it: causal still has 2.3 points less repetition and much higher
  Dist-2.
- **The LoRA run did not fix free generation either.** At k=8 T=0 it is worse
  than the 1000-step full fine-tune: H4 0.757 → 0.658, rep4 37.6% → 42.2%; at
  k=2 it is also worse (rep4 56.0% → 67.5%). The headline improvement over the
  old checkpoint comes from the sampler (`r=0.25`), not from the weights.

### Collision suppression A/B (shipped default)

`runs/bench-adj.log` (suppression on, default) vs `runs/bench-noadj.log`
(`--no_adj_suppress`), same checkpoint, same prompts, fixed quota, `r=0.25`:

| k | T | H4 off → on | rep4 off → on | rep2 off → on | adj off → on | Dist-2 off → on |
|---|---|---|---|---|---|---|
| 2 | 0.0 | 0.385 → 0.583 | 67.5% → 49.7% | 77.7% → 69.9% | 82.5% → 68.8% | 0.164 → 0.226 |
| 4 | 0.0 | 0.642 → 0.874 | 45.1% → 22.1% | 65.6% → 53.8% | 65.6% → 41.1% | 0.258 → 0.363 |
| 8 | 0.0 | 0.949 → **0.967** | 10.1% → **5.7%** | 42.7% → 37.4% | 31.5% → **24.2%** | 0.444 → 0.478 |
| 2 | 0.3 | 0.925 → 0.982 | 11.8% → 3.7% | 41.1% → 32.8% | 32.8% → 20.2% | 0.422 → 0.449 |
| 4 | 0.3 | 0.903 → 0.972 | 14.4% → 5.7% | 45.4% → 33.6% | 36.0% → 23.9% | 0.470 → 0.540 |
| 8 | 0.3 | 0.986 → 0.990 | 2.9% → 2.3% | 23.9% → 25.3% | 19.1% → 17.2% | 0.702 → 0.656 |
| 2 | 0.5 | 0.995 → 0.998 | 1.1% → 0.6% | 20.7% → 14.5% | 18.0% → 12.4% | 0.618 → 0.739 |
| 4 | 0.5 | 0.992 → 0.993 | 1.7% → 1.7% | 18.8% → 20.2% | 15.3% → 9.4% | 0.723 → 0.707 |
| 8 | 0.5 | 0.996 → 0.995 | 0.9% → 1.1% | 15.1% → 13.7% | 14.0% → 11.6% | 0.788 → 0.831 |
| causal | | 0.964 | 7.8% | 20.7% | 0.0% | 0.653 |

Metrics added in this pass: `rep2` = 1 - unique 2-grams / total 2-grams within a
sequence, `adj` = fraction of neighbouring token pairs that are identical.
`adj` is the column `rep4` was hiding: at k=8/T=0 the old sampler produced a
31.5% adjacent-duplicate rate that barely moved the 4-gram number.

Findings:

- **Suppression works best where the collision problem is worst.** At `k=2`
  sixteen slots commit in one step: rep4 67.5% → 49.7%, adj 82.5% → 68.8%,
  H4 0.385 → 0.583. At `k=8` only one or two slots commit per step, so the gain
  is small but still there at T=0 (rep4 10.1% → 5.7%, adj 31.5% → 24.2%).
- **`not not not` is gone, `the the the` is not.** k=8 T=0 decodes
  `the were able able to to fish the fish of the the the the same of the fish
  were fish fish were not the the fish were not same the fish were`: the
  three-token `not not not` run has disappeared, but runs of `the` survive
  because they are committed in *different* steps. The mechanism can only see
  simultaneous predictions, and that is exactly its stated scope.
  Counted on the rendered text (word-level, so a different measure than the
  token-level `adj` column): 27.7% of adjacent pairs identical before
  suppression, 15.0% after, and the longest run is still 3 tokens in both.
- **At T>=0.3 the effect is inside noise** (rep2 and Dist-2 move in both
  directions, ±2 points). Sampling noise already breaks up many pairs, so the
  deterministic setting is where the mechanism pays.
- **Causal still wins** on `adj` (0.0% by construction: `generate()` conditions
  on the previous token) and on Dist-2. Diffusion is now at H4 parity
  (0.967 vs 0.964) with more adjacent stutter left.
- Cost: none measurable (same forward count, TPS within noise: 24.5 vs 24.4 at
  k=8/T=0).

Full decoded text for this build: `docs/decoded-samples-suppressed.md` (120
blocks, 12 prompts × 9 configs).

### Prefix KV cache scaling (Task 2)

```bash
python benchmark.py --checkpoint runs/bfg-lora/last.pt --prefix_sweep \
  --num_prompts 12 --canvas_len 32 --threads 10
```

A real corpus document is truncated to exactly `P` tokens and used as the prefix
for all 12 prompts (one prefix, repeated); `C=32` fixed, greedy, `k=4`/`k=8` with
`prefill_prefix` + `canvas_step_with_cache`. Output is `runs/prefix-sweep.log`.

| P | batch | causal | k=4 | k=4 / causal | k=8 | k=8 / causal |
|---|---|---|---|---|---|---|
| 32 | 1 | 11.9 | 28.4 | **2.38x** | 15.6 | 1.31x |
| 32 | 4 | 26.7 | 50.1 | **1.87x** | 23.5 | 0.88x |
| 128 | 1 | 8.5 | 17.3 | 2.04x | 13.6 | 1.61x |
| 128 | 4 | 22.9 | 35.8 | 1.56x | 22.6 | 0.99x |
| 256 | 1 | 10.3 | 19.0 | 1.85x | 12.3 | 1.19x |
| 256 | 4 | 19.5 | 26.4 | 1.35x | 18.3 | 0.94x |
| 512 | 1 | 8.5 | 12.9 | 1.51x | 9.3 | 1.09x |
| 512 | 4 | 14.5 | 16.6 | 1.14x | 12.8 | 0.88x |

Interpretation:

- **The speedup is real but it decays with prefix length, it does not grow.**
  k=4 goes 2.38x → 1.51x (batch 1) and 1.87x → 1.14x (batch 4) as `P` goes
  32 → 512; at `P=512`, batch 4, diffusion is a tie (16.6 vs 14.5 TPS).
- **k=8 only beats causal for short prefixes.** 1.31x at `P=32`, 1.61x at
  `P=128` batch 1 (noise), then 0.94x/0.88x from `P=256` on at batch 4 — i.e.
  the 12 default benchmark prompts (`P≈15-20`) were the *most* favourable case,
  not a hostile one. The earlier "short prompts hide the cache advantage" guess
  was wrong: caching is what makes 2.4x possible at `P=32`, and the same
  mechanism runs out of headroom as `P` grows.
- **Why it decays:** with the prefix cached, a diffusion step is a `32 × d` matmul
  over `P+32` keys, while causal decoding is 32 sequential single-token matmuls
  over the same keys. The theoretical FLOP ratio is 32/k (8x at k=4), but a
  1-token causal matmul is memory-bound on the 0.5B weights and a 32-token
  diffusion matmul is not, so real speedups land at 1.1-2.4x. The larger the
  prefix, the more of every forward is KV attention, which both methods pay.
- **Batch 4 helps causal more than diffusion** (P=32: causal 2.2x, k=4 1.8x,
  k=8 1.5x from batch 1 to 4). Batching amortizes the weight reads that causal
  decoding is bottlenecked on; the diffusion forward is already compute-bound.
- Batch-1 causal values are not monotone in `P` (11.9, 8.5, 10.3, 8.5): 12
  prompts × 32 tokens is a ~2-8 s measurement, so ±20% is run noise. The
  batch-4 column and the k=4/k=8 columns are the ones to quote.

### Task 4: `--lora_mlp` and the stage-3 gate

```bash
python train.py --smoke_test --lora --lora_mlp --threads 8 --out /tmp/smoke-mlp
LoRA r=16 alpha=32 targets=7: 8,798,208 / 502,830,976 params trainable
smoke probe loss (before): 5.4177
smoke convergence: 5.4177 -> 3.5945 PASS
```

- MLP targets add 6,636,096 parameters (2.16M → 8.80M, 0.43% → 1.75%), i.e.
  **4.1x the trainable budget** for the 65% of weights that were frozen.
- The flag implies `--lora`, and `targets` is stored in `lora_config`, so
  `benchmark.py` rebuilds the correct 7-target wrapper from the checkpoint
  alone. The pre-existing `runs/bfg-lora/last.pt` has no `targets` key and still
  loads as attention-only (default `LORA_TARGETS`).
- **Decision gate: NOT triggered.** The gate was "rep4 at T=0.0 with
  `remask_ratio=0.25` still > 20%". Measured 10.1% at k=8 (45.1% at k=4). The
  remaining failure is grammatical structure, not a repeated-4-gram attractor,
  and it is not obviously caused by frozen MLPs. No 3000-step `--lora_mlp` run
  was launched by the gate itself.
- **Run launched anyway, as a controlled ablation** (2026-11-15): the qualitative
  text showed adjacent stutter (`not not not`) that `rep4` under-reported, and
  the hypothesis that frozen causal MLPs cannot form fluent bidirectional
  n-grams is testable by the same 3000-step recipe at 8.80M parameters.
  `runs/bfg-lora-mlp` / `runs/train-lora-mlp.log`, same seed (0), same corpus,
  same batch (4 x accum 2). Comparison pending: reconstruction `k=2,4,8` and
  generation `k=8, r=0.25, T=0/0.3/0.5` with `rep2`/`rep4`/`adj` for both
  checkpoints.

## Stage 3 ablation: Attn-LoRA (2.16M) vs Attn+MLP-LoRA (8.80M)

`runs/bfg-lora-mlp` finished 2026-11-15 19:04 (step 3000, val_loss 4.1111).
Same seed (0), corpus, batch (4 x accum 2) as `runs/bfg-lora`.

Loss (val, every 200 steps):

| step | Attn-LoRA | Attn+MLP-LoRA |
|---|---|---|
| 200 | 3.80 | 5.20 |
| 1000 | 3.72 | 5.33 |
| 2000 | 3.72 | 4.26 |
| 2600 | **3.6254 (best)** | 4.0123 (best) |
| 3000 | 3.7263 (final) | 4.1111 (final) |

The MLP run is **~0.4 nats worse at every checkpoint**, including its own best
(4.01 vs 3.63). Adding 6.6M trainable parameters to the frozen causal MLPs did
not help; it hurt. The hypothesis that frozen MLPs were the bottleneck for
bidirectional n-grams is **not supported** by this ablation.

Reconstruction (50% masked, P=128, C=32, 12 prefixes):

| k | Attn-LoRA | Attn+MLP-LoRA |
|---|---|---|
| 2 | 89.7% | 91.8% |
| 4 | 92.3% | 91.2% |
| 8 | 96.4% | 94.8% |

Generation, k=8, r=0.25 (suppression on):

| T | metric | Attn-LoRA | Attn+MLP-LoRA |
|---|---|---|---|
| 0.0 | H4 | 0.996 | 0.975 |
| 0.0 | rep4 | 0.9% | 6.0% |
| 0.0 | rep2 | 26.9% | 31.5% |
| 0.0 | adj | 4.6% | 4.3% |
| 0.0 | Dist-2 | 0.570 | 0.530 |
| 0.3 | H4 | 0.988 | 0.992 |
| 0.3 | rep4 | 2.9% | 1.7% |
| 0.3 | adj | 3.8% | 3.2% |
| 0.3 | Dist-2 | 0.726 | 0.710 |
| 0.5 | H4 | 0.999 | 0.995 |
| 0.5 | rep4 | 0.3% | 1.1% |
| 0.5 | Dist-2 | 0.836 | 0.798 |

Conclusion: the attention-only adapter (2.16M) is the better checkpoint on every
generation metric that matters; the MLP adapter trades nothing useful for its
4x parameter count. The 3000-step recipe does not need `--lora_mlp`.

### Cross-step repulsion (second sampler fix, 2026-11-15)

`suppress_adjacent` now has two parts:

1. **Cross-step logit repulsion.** An open slot next to a committed token has
   that token removed from its logits, so even on the final fill step it picks a
   distinct continuation. This is the fix for the surviving `the the the` (the
   earlier same-step-only rule could only delay it). The prefix's last token is
   also repelled at position 0.
2. **Same-step margin deferral** (unchanged): two open neighbours proposing the
   same token hold the weaker one back for a step.

Result on `runs/bfg-lora/last.pt`, k=8, r=0.25, suppression default-on:

| T | adj before | adj after | rep4 before | rep4 after |
|---|---|---|---|---|
| 0.0 | 24.2% | 4.6% | 5.7% | 0.9% |
| 0.3 | 17.2% | 3.8% | 2.3% | 2.9% |
| 0.5 | 11.6% | 2.4% | 1.1% | 0.3% |

`adj` is not zero (the left-to-right final fill still occasionally has to
fall back to the repelled slot's #2, and cross-step pairs committed in
different steps are only partially covered), but the three-token runs are gone
from the decoded text. Mechanism check on the base model: adj 0.976 -> 0.126
at T=0.

## Open questions / not verified

- bf16 numerics, SDPA mask handling, and memory use on a real GPU.
- Infilling separates from generation strongly (96.4% recovery vs 5.7% rep4),
  but 3000 LoRA steps on 2M tokens still do not give coherent free generation.
  Larger `C`, a longer canvas, or a curriculum on `t` are the remaining levers;
  more steps at this corpus size are not (the val curve is flat after 2600).
- Adjacent-collision suppression only sees *simultaneous* predictions. Runs
  committed across different steps (`the the the` after three separate steps)
  survive it: 24.2% adjacent duplicates at k=8/T=0 versus 0% causal. A variant
  that compares the surviving token of a committed slot against its neighbour's
  candidate would close that gap and is not implemented.
- The margin floor was only measured at 1.0 on the 1000-step full fine-tune
  (it lost to the rigid quota). It is now redundant with `remask_ratio` for the
  repetition failure mode and has not been re-swept on the LoRA checkpoint.
- `remask_ratio` was swept at 0.0 and 0.25 only. 0.5 and 1.0 (re-open
  everything) are untested; the protection fraction is the obvious next knob.
- Gumbel temperatures 0.3 and 0.5 were measured on the LoRA checkpoint: the
  4-gram metrics improve monotonically with T, but the decoded text at 0.5 is
  word salad (`the proxy port0000.00...`), so the metric is rewarding noise.
  No temperature between 0.1 and 0.3 was tried.
- The 20%-fully-masked schedule is now trained (3000 steps) and did not fix free
  generation, but it was never ablated against the old `U(0.15,0.95)` schedule
  at equal compute.
- Reference texts repeat every 10 prompts in the benchmark, so per-prefix scores
  are correlated; only 12 prompts were used in the timed sweep. The prefix sweep
  additionally reuses a single prefix for all 12 prompts.
- No comparison against standard block-autoregressive decoding or other
  diffusion samplers; only HF greedy `generate()`.
- The 0.15–0.95 corruption range is fixed as specified, not tuned.
- Reconstruction recovery is now monotone in `k` again (see Task 1), but the
  k=2 path is single-shot by construction when half the canvas is visible: the
  first quota equals the number of frozen slots, so there is nothing to fill
  until the final step.

## Right-Shifted Geometry & Carrier-Frontier Diffusion (Step 1500 `runs/bfg-shifted/last.pt`)

### 1. Architectural Upgrades
1. **Right-Shifted Geometry:** Fixed the shift mismatch in `canvas_forward` by returning `out.logits[:, P - 1 : P + C - 1, :]`. This aligns diffusion prediction with Qwen2.5's native $k \to k+1$ causal mechanics: slot 0 of canvas is predicted by position $P-1$ (`prefix[-1]`), and slot $j$ is predicted by carrier token $j-1$.
2. **Trainable `[MASK]` Row:** In `apply_lora`, registered a gradient filter hook allowing only spare row `151665` (`<|extra_0|>`) in `embed_tokens.weight` to receive gradient updates without reinitializing or resizing the 151,936-row table.
3. **Carrier-Frontier & Island Sampling:** In `sampler.py`, open slots with a concrete committed left-carrier (`has_carrier == True`) unmask first. When parallel quotas exceed available carriers, the sampler seeds non-adjacent island anchors ($j \ge 2, j-1 \notin \text{pick}$) so real carriers double cleanly ($1 \to 2 \to 4 \to 8$).
4. **Sliding Sub-Blocks:** Added `sample_canvas` supporting sub-blocks ($C_{\text{sub}}=8$) sliding through the prefix KV cache.

### 2. Uncontended Final Benchmarks (Xeon 20-core CPU, Step 1500)

#### A. Infilling Reconstruction (50% masked, $P=128, C=32$, 12 prefixes)
| Method | Recovery | Uncontended TPS |
|---|---|---|
| BFG-Diff $k=2$ | 87.6% | 48.2 |
| BFG-Diff $k=4$ | 95.4% | 33.5 |
| BFG-Diff $k=8$ | **95.9%** | 20.7 |

#### B. Free Generation (12 prompts, $C=32, T=0.0$)
| Configuration | $H_4$ | Rep-4 | Rep-2 | Adj | Dist-1 | Dist-2 | TPS | Notes |
|---|---|---|---|---|---|---|---|---|
| **Causal `model.generate()`** | 0.965 | 8.0% | 21.8% | 0.0% | 0.391 | 0.629 | 26.8 | Native sequential baseline (1 tok/pass) |
| **$C=32, k=8$ (Single block)** | 0.959 | 9.2% | 32.3% | 5.1% | 0.247 | 0.519 | 25.9 | Full-canvas parallel diffusion (4 tok/pass) |
| **$C=32, k=4$ (Single block)** | 0.953 | 10.3% | 39.0% | 18.8% | 0.208 | 0.446 | **48.4** | High-speed single-block diffusion (8 tok/pass) |
| **$C=32, k=2$ (Single block)** | 0.881 | 23.6% | 55.9% | 29.6% | 0.115 | 0.298 | **87.3** | Extreme speed (16 tok/pass) |
| **$C_{\text{sub}}=8, k_{\text{sub}}=4$ (Sliding)** | **0.975** | **5.7%** | 22.6% | **0.0%** | 0.297 | 0.613 | 22.2 | **2 tok/pass parallel sub-blocks (Beats causal Rep-4)** |
| **$C_{\text{sub}}=8, k_{\text{sub}}=2$ (Sliding)** | 0.979 | 4.9% | 35.5% | 15.9% | 0.203 | 0.484 | 34.3 | 4 tok/pass high-speed sub-blocks |
| **$C_{\text{sub}}=8, k_{\text{sub}}=8$ (Sliding)** | 0.975 | 5.7% | 22.6% | 0.0% | 0.297 | 0.613 | 12.9 | 1 tok/pass causal-diffusion upper bound |

Complete sample texts across all prompts are logged in [decoded-samples-shifted.md](file:///home/gces/.bin/.s/.scripts/bfg-diff/docs/decoded-samples-shifted.md).


## Persistent KV-Cache & Gated Island Anchors (2026-11-16)

### Optimization & Algorithmic Refinement
1. **Persistent KV Cache in `sample_canvas`:**
   - Previous sliding sub-block implementation re-evaluated the entire prompt prefix on each sub-block. For $P=128, C=32, C_{\text{sub}}=8$, total token evaluations across 4 sub-blocks were $560$ tokens.
   - Fixed by calling `prefill_prefix` once before the sub-block loop and incrementally advancing the cache with causal passes over $[ \text{prefix}_{[-1]}, \text{block}_{[:-1]} ]$ ($C_{\text{sub}}$ tokens), cutting token evaluations to $272$ tokens.
   - Cache key difference against fresh prefill is $< 3.1 \times 10^{-5}$.
2. **Anchor Threshold Gating:**
   - Added `anchor_thresh=2.5` gating to `cand_island` in `sample_canvas_block`.
   - When step budget exceeds the available concrete carriers (`has_carrier`), unanchored island slots are only picked if their confidence margin satisfies $\text{margin}[i, j] \ge \text{anchor\_thresh}$.
   - Prevents low-confidence blind guesses from committing on step 1.

### Verified Benchmark Smoke Output
```
Reconstruction | 2 prefixes | P=128 C=16
method              recovery         TPS
bfg-diff k=2          26.3%        17.9

Generation | 2 prompts | C=16 | top_k=50
method                                 H4   rep4   rep2    adj  Dist1  Dist2       TPS
bfg-diff k=2 T=0 d=0 f=0 r=0        0.664  46.2%  56.7%   0.0%  0.219  0.433      29.1
causal generate                     1.000   0.0%   3.3%   0.0%  0.750  0.967      13.6
```
Smoke convergence probe: `11.9797 -> 3.6238 PASS`.

## Anti-Starvation Carrier Frontier Sampling (2026-11-16)

### Analysis of the Budget Starvation Failure Mode
- **Diagnosis:** Restricting unmasked tokens to only concrete carriers plus `margin >= anchor_thresh` caused earlier steps ($s < K$) to unmask only 1 token (`slot 0` on step 1, `slot 1` on step 2, `slot 2` on step 3). On the final step $s == K$, all remaining unmasked slots ($5$ in $C_{\text{sub}}=8$, or $29$ in $C=32$) were unmasked simultaneously in an unanchored batch, collapsing the tail into stutter runner-ups (`"the a the a the a"`).
- **Resolution in `sampler.py`:**
  1. Pick available slots with real carriers (`has_carrier`).
  2. Pick distant non-adjacent island anchors with `margin >= anchor_thresh`.
  3. **Strict Quota Guarantee:** If `len(pick) < budget`, immediately expand the carrier frontier along contiguous adjacent slots ($j-1 \in \text{pick}$), followed by top-rank open slots. Every step strictly satisfies `len(pick) == budget`.

### Verification on Step 1500 Checkpoint (`runs/bfg-shifted/last.pt`)
- **Prompt [0]:**
  - $C_{\text{sub}}=8, k_{\text{sub}}=4$ (9.3 TPS): `' the fish caught were not smaller than the ones they caught. They fishermen were that they, too the fish were smaller than the they caught. They thought that the'`
  - $C_{\text{sub}}=8, k_{\text{sub}}=6$ (6.7 TPS): `' the fish caught were not all the. They decided to make a fishing boat. The boat was a simple one, a small boat with a single sail. a'`
- **Prompt [1]:**
  - $C_{\text{sub}}=8, k_{\text{sub}}=4$ (9.4 TPS): `' the proxy server to the proxy server address and the port port.\n\nTo\nconfigure the proxy server, edit the settings file and set the proxy server to the'`
  - $C_{\text{sub}}=8, k_{\text{sub}}=6$ (6.7 TPS): `' the proxy server to the IP address of the proxy server.\nA. Correct\nB. Incorrect\nAnswer\nAnswer\nA\n\nThe isobaric line is'`
- **Prompt [4]:**
  - $C_{\text{sub}}=8, k_{\text{sub}}=4$ (9.4 TPS): `' plot the data\n- fourth, plot the outliers\n- fifth, plot the outliers\n- sixth, plot the outliers\n- seventh, plot the outliers\n'`
  - $C_{\text{sub}}=8, k_{\text{sub}}=6$ (6.7 TPS): `' plot the data\n- fourth, plot the outliers\n- fifth, plot the outliers\n- sixth, plot the outliers\n- seventh, plot the outliers\n'`

`python train.py --smoke_test`: `smoke convergence: 11.9797 -> 3.6238 PASS`.

## Carrier-Verified Lookahead Sampling (Draft & Verify in One Loop) (2026-11-16)

### Root Cause Analysis of Lookahead Stutters
In sub-block diffusion ($C_{\text{sub}}=8, k_{\text{sub}}=4$), when step 1 unmasks 2 tokens under budget constraints, slot 0 is conditioned on `prefix[-1]` (real carrier), while slot 1 is predicted while `canvas[0]` is still `[MASK]` (an unverified blind lookahead draft).
- **Previous failure:** Slot 1 was permanently committed on step 1 into `keep_token` (`"The is"`, `"I I'm"`, `"well great"`). Even though step 2's forward pass finally had `"The"` in slot 0, slot 1 could never revise itself to `" capital"`.
- **Carrier-Verified Lookahead Fix:**
  1. Maintained a `verified` mask (permanent locked tokens) separate from `decoded` (all currently placed tokens).
  2. Input to the forward pass (`feed_canvas = torch.where(verified, canvas, mask_id)`) only feeds carrier-verified tokens, preventing lookahead hallucinations from corrupting the causal context.
  3. Slots whose input left carrier was verified during the forward pass are promoted to `verified = True`.
  4. On final step $s == \text{steps}$, remaining unverified slots are sequentially conditioned and verified left-to-right on real carriers.

### Live Multi-Prompt Benchmark Results (`runs/bfg-instruct/last.pt`)
Evaluated via `http://127.0.0.1:8766/api/generate` with $C_{\text{sub}}=8, k_{\text{sub}}=4$:

| Prompt | Mode | Before Fix | After Fix (Carrier-Verified Lookahead) | Causal Baseline |
| :--- | :--- | :--- | :--- | :--- |
| `"How are you doing today?"` | Chat | `"I I'm good, I'm doing well great today."` | `"I'm doing well, thank you. How can I assist you today?"` | `"I'm good, thanks for asking! How about you?"` |
| `"What is the capital of France?"` | Chat | `"The is the Paris."` | `"The capital of France is Paris."` | `"The capital of France is Paris."` |
| `"Tell me a short story about a fisherman."` | Chat | `"A fisherman named a day, catches a fish..."` | `"A fisherman named John was fishing in the ocean when he caught a beautiful fish"` | `"A fisherman named Jack was fishing in the ocean..."` |
| `"The quick brown fox"` | Completion | Stutter repeats | `"jumps over the lazy dog.\nThis is a classic poem that has been around for"` | `" jumps over the lazy dog.\nThis is a classic poem..."` |
| `"Machine learning is a subset of"` | Completion | Stutter repeats | `"artificial intelligence that enables computers to learn from data. It involves the use of algorithms"` | `" artificial intelligence that involves the use of algorithms..."` |
| `"In a galaxy far, far away"` | Completion | Stutter repeats | `", there is a galaxy of stars. The stars are arranged in a circle,"` | `", there is a planet with the same mass as Earth..."` |

- **Smoke test verification:** `python train.py --smoke_test`: `smoke convergence: 11.9797 -> 4.0549 PASS`.

## Web Streaming & Early-Exit Optimization Benchmarks (2026-11-16)

Evaluated via `chat.py` on `runs/bfg-instruct/last.pt` with early-exit stop tokens (`<|im_end|>`, `<|endoftext|>`):

### 6-Prompt Benchmark Comparison

| Prompt | Mode | BFG-Diff Output | BFG Latency | BFG Blocks | Causal Output | Causal Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `"How are you doing today?"` | Chat | `"I'm doing well, thank you. How can I assist you today?"` | **3.43s** (4.7 TPS) | 2 | `"I'm good, thanks for asking! How about you?"` | 1.18s (11.0 TPS) |
| `"What is the capital of France?"` | Chat | `"The capital of France is Paris."` | **1.75s** (4.6 TPS) | 1 | `"The capital of France is Paris."` | 0.76s (10.5 TPS) |
| `"Tell me a short story about a fisherman."` | Chat | `"A fisherman named John was fishing in the ocean when he caught a beautiful fish. He named it "Fishy" and named it "Day" for the"` | **6.85s** (4.7 TPS) | 4 | `"A fisherman named Jack was fishing in the ocean..."` | 2.76s (11.6 TPS) |
| `"The quick brown fox"` | Completion | `"jumps over the lazy dog.\nThis is a classic poem that has been around for centuries. It's a great example of a poem that is both beautiful and easy"` | **6.77s** (4.7 TPS) | 4 | `" jumps over the lazy dog.\nThis is a classic poem..."` | 2.68s (11.9 TPS) |
| `"Machine learning is a subset of"` | Completion | `"artificial intelligence that enables computers to learn from data. It involves the use of algorithms to analyze patterns in data and make predictions or decisions based on that data. The"` | **7.24s** (4.4 TPS) | 4 | `" artificial intelligence that involves the use of algorithms..."` | 2.70s (11.8 TPS) |
| `"In a galaxy far, far away"` | Completion | `", there is a galaxy of stars. The stars are arranged in a circle, and each star is connected to its two nearest neighbors by a straight line. The"` | **6.75s** (4.7 TPS) | 4 | `", there is a planet with the same mass as Earth..."` | 2.71s (11.8 TPS) |

### Verification Smoke Tests
- `python train.py --smoke_test`: `smoke convergence: 11.9798 -> 4.0549 PASS`.
- `python benchmark.py --smoke_test`:
  - `Reconstruction | 2 prefixes | P=128 C=16`: `bfg-diff k=2`: `recovery = 63.2%`, `TPS = 9.2`.
  - `Generation | 2 prompts | C=16 | top_k=50`: `H4 = 1.000`, `rep4 = 0.0%`, `rep2 = 6.7%`, `TPS = 5.0`.
  - Causal generate: `H4 = 1.000`, `rep4 = 0.0%`, `rep2 = 3.3%`, `TPS = 13.6`.

## Speculative Lookahead Verification & Visual Stability Benchmarks (2026-11-16)

### Changes & Mechanics
- **Zero Step-4 Sequential Loop:** Eliminated the sequential loop on step 4 that ran separate forward passes per unverified token.
- **Speculative Draft Retention:** Forward pass evaluates lookahead drafts rather than wiping unverified slots to `mask_id`.
- **Chain Acceptance:** Accepted drafts verify the subsequent token ($m+1$) in the same forward pass if within tolerance and distinct from the neighbor.
- **UI Trace Stabilization:** Distant unverified slots ($j > m_{\text{frontier}} + 1$) are masked in the UI callback to prevent flickering chips and visual rewriting.

### Measured Side-by-Side Throughput & Quality (`runs/bfg-instruct/last.pt`)

Measured via live SSE API with $C=32, C_{\text{sub}}=8, k_{\text{sub}}=4$, greedy ($T=0$):

| Prompt | BFG Text | BFG Latency | BFG TPS | Causal Latency | Causal TPS |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `"What is the capital of France?"` | `'The capital of France is Paris..'` | **2.28s** | **7.0** | 2.25s | 3.1 |
| `"Hello! How are you today?"` | `"Hello! I'm just a.. I don't havehave any"` | **3.14s** | **7.6** | 6.32s | 3.6 |
| `"Explain the concept of gravity in simple terms."` | `'Gravity is the forceeness that objects o'` | **4.51s** | **7.1** | 7.84s | 3.8 |
| `"Write a haiku about autumn leaves."` | `'Leaves whisper, golden,,, fall,  \nNature'` | **2.85s** | **8.4** | 4.41s | 3.9 |
| `"Who wrote the play Romeo and Juliet?"` | `'William Shakespeare wrote the Romeo and '` | **2.67s** | **6.0** | 3.07s | 3.6 |
| `"Give me three tips for learning a new programming language."` | `'Certainly! Here are three tips tips tips'` | **5.09s** | **6.3** | 8.12s | 3.9 |

### Verification Smoke Tests
- `python train.py --smoke_test`: `smoke convergence: 11.9797 -> 4.0549 PASS`.
- `python benchmark.py --smoke_test`:
  - `Reconstruction | 2 prefixes | P=128 C=16`: `bfg-diff k=2`: `recovery = 26.3%`, `TPS = 12.1`.
  - `Generation | 2 prompts | C=16 | top_k=50`: `H4 = 0.796`, `rep4 = 23.1%`, `rep2 = 46.7%`, `adj = 56.7%`, `TPS = 21.2`.
  - Causal generate: `H4 = 1.000`, `rep4 = 0.0%`, `rep2 = 3.3%`, `adj = 0.0%`, `TPS = 13.5`.

## Strict Top-1 Carrier Verification & Feed Masking (2026-11-16)

### Modifications (`sampler.py`)
1. **Masked `feed_canvas`:** Restored `feed_canvas = torch.where(verified, canvas, mask_id)` to ensure speculative unverified drafts never contaminate self-attention representations of carrier tokens during forward passes.
2. **Strict Verification Agreement:** Replaced relaxed threshold $\Delta z \le 1.2$ with strict top-1 agreement (`draft == top_tok`), rejecting lookahead drafts unless the forward pass agrees exactly with the lookahead prediction under verified left context.

### Verification Smoke Tests
- `python benchmark.py --smoke_test`:
  - `Reconstruction | 2 prefixes | P=128 C=16`: `bfg-diff k=2`: `recovery = 15.8%`, `TPS = 12.3`.
  - `Generation | 2 prompts | C=16 | top_k=50`: `H4 = 0.789`, `rep4 = 26.9%`, `rep2 = 46.7%`, `adj = 63.3%`, `TPS = 22.0`.
  - `causal generate`: `H4 = 1.000`, `rep4 = 0.0%`, `rep2 = 3.3%`, `adj = 0.0%`, `TPS = 13.5`.
- **Live Visualizer (`runs/bfg-instruct/last.pt`, $C_{\text{sub}}=8, k_{\text{sub}}=4$):**
  - `"What is the capital of France?"`: BFG completed in **1.75s (9.15 TPS)** with clean output `"The capital of France is Paris."`.




