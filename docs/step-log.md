# Step log

Chronological record of building and verifying the package. Commands are run
from `/home/gces/.bin/.s/.scripts/bfg-diff` with `../AD/.venv/bin/python`.

## 1. Environment

- `python -c "import torch"` → not installed in system Python.
- `../AD/.venv/bin/python` → `torch 2.14.1+cpu`, CUDA unavailable,
  `transformers 5.20.2`.
- `nvidia-smi` absent, 20 cores, 31 GB RAM, 452 GB free on `/home`.

Decision: keep CUDA/bf16 code paths, verify on CPU fp32.

## 2. Model download

- First downloaded `Qwen/Qwen2.5-Coder-0.5B` for the initial code-oriented
  spec.
- Spec was revised to general-purpose text; downloaded `Qwen/Qwen2.5-0.5B`
  (7 files, ~988 MB safetensors) and removed the first smoke checkpoints.

## 3. First implementation

- Wrote `model.py`, `dataset.py` (code windows), `sampler.py`, `train.py`,
  `benchmark.py` (AST parse rate, collisions).
- `py_compile` clean.
- Mask unit check: shape `(2,1,8,8)`, prefix never sees canvas, canvas sees all,
  padded keys blocked, position ids gap-free.
- Training smoke (8 steps): loss 17.76 → 8.75, val 8.64, 117 s. Passed with a
  first-vs-last comparison.

## 4. Revision to general-purpose text

- Default model → `Qwen/Qwen2.5-0.5B`; no code-specific tokenisation or AST.
- `dataset.py` rewritten as `GeneralTextDataset` (dir/file/string list, binary
  files skipped via NUL-byte sniff) plus `synthetic_texts()`.
- `sampler.py` gained an optional pre-filled canvas for reconstruction and a
  `sample_canvas_text` wrapper.
- `benchmark.py` rewritten: 50%-mask recovery at k=4/8, 4-gram entropy and
  repeat rate, TPS vs `generate()`.
- Training: weight decay 0.01, probe-based smoke convergence check.

## 5. Unit checks

```
mask builder        OK (causal prefix, bidirectional canvas, padded keys hidden)
tokenizer alias     <|extra_0|> -> 151665, vocab 151936, no resize
dataset             dir + string list OK, binary-only dir rejected
sampler             full-mask, partial-init, batched left-pad OK
```

A partial-init sampler test failed: frozen tokens were overwritten. Root cause
was `topk(quota)` selecting `-inf` already-filled slots. Fixed with a per-sample
`min(quota, masked_count)` cap; the test then passed.

## 6. Training smoke (final code)

```
time python train.py --smoke_test
```

- 12 steps, 12/12 eval and checkpoint written, 95 s.
- Probe 5.7692 → 3.8388 PASS; val 3.7360.
- Scheduler ordering warning from the accumulation experiment was fixed before
  this run (schedule now advances per optimizer update).

## 7. Benchmark smoke (final code)

```
time python benchmark.py --smoke_test
```

- 17 s, loads `runs/smoke/last.pt`.
- Recovery 5.3%; diffusion k=2 38.1 TPS vs causal 13.3 TPS; entropy 0.532 vs
  1.000 and repeat 46.2% vs 0% — expected for a 12-step checkpoint and two
  sampling steps, not a quality verdict.

## 8. CPU optimization sprint

- Added `--cpu` preset (P=128, C=32, batch 2 x accum 8, 1000 steps, warmup 20
  updates, checkpoints every 200), `--threads`, `--max_tokens`, in-loop
  `torch.set_num_threads`, and `torch.inference_mode()` evaluation.
- `sampler.py`: added `margin_floor`, per-step new-slot budget, and a unit test
  that a huge floor leaves pre-filled slots intact.
- `benchmark.py`: k sweep (2/4/8) crossed with margin floors, Dist-1/Dist-2,
  4-gram entropy/repeat, side-by-side decoded completions.
- Speed trial: 4 steps and a 2 GB checkpoint in 31.6 s, 10.2 GB RSS, corpus
  62,432 windows. 1000-step run launched on the filtered documentation corpus.

## 9. 1000-step CPU run

- Command: `train.py --cpu --threads 10 --data /tmp/bfg-corpus
  --max_tokens 2000000 --out runs/bfg-diff`; ran in tmux session `bfg-train`.
- 3195 s (53 min), ~100 tok/s, val loss 4.97 → 3.64 (step 800) → 4.06 (1000).
  Checkpoints 200/400/600/800/1000 at 1.9 GB each.
- Restarted once at step 60 to move the run into tmux (a live process cannot be
  attached without `reptyr`); cost ~2 min.

## 10. Benchmark sweep of the trained checkpoint

- 12 prompts, P=128, C=32, k in {2,4,8}, margin floors {0.0, 1.0}, 3 examples.
- Recovery 63.9% / 81.4% / 90.2% for k=2/4/8.
- Generation stays repetitive ("the the the"); k=8 is best (repeat 37.6% vs
  13.5% causal). Margin floor 1.0 did not reduce repetition.
- k=2 and k=4 are faster than causal `generate()` at this canvas size
  (92.9/46.7 vs 25.6 TPS); k=8 is slower.

## 11. Sampler-only fixes

- Added Gumbel temperature sampling (clean margins still rank slots) and a
  step-1 position-decay prior; both default off.
- Swept k=8 on `last.pt`: decay alone is a no-op; Gumbel T=0.7 removes 4-gram
  repetition (0%) but the decoded text is incoherent, so the metric win is an
  artifact of near-random sampling. Gumbel costs ~23% TPS.
- Changed the training mask schedule to 20% fully masked batches and
  `U(0.20, 1.0)` otherwise; not yet trained.

## 12. Top-k sampling, KV cache, LoRA

- `sampler.py`: Gumbel now samples from a `top_k` mini-batch (default 50) instead
  of the full vocabulary; margins use raw logits (`z1 - z2`), removing the
  151k-row `log_softmax`. Verified greedy output is token-identical to the
  `log_softmax` version, top-k is respected, and the margin identity is exact.
- `model.py`: `prefill_prefix` + `canvas_step_with_cache` cache prefix keys and
  values; each step forwards only `C` tokens. Cached logits differ from a full
  `(P+C)` forward by ~3e-5 and argmax is identical (checked with and without
  left padding). Sampler crops the canvas rows after each step.
- Measured: reconstruction TPS k=4 16.9 → 33.4, k=8 8.5 → 20.8; generation
  unchanged (short prompts). k=8 at batch 4 is 31.3 TPS vs causal 29.2; at
  batch 2 it is 25.6 vs 25.6.
- `train.py`: `--lora` (PEFT), adapter-only checkpoints, `--lr` 5e-4 default
  with LoRA, `torch.set_num_interop_threads(2)`, CPU preset enables LoRA.
  `benchmark.py` loads adapter checkpoints and gained `--top_k`.
- `train.py --smoke_test --lora` PASS 5.7332 → 4.2757; `benchmark.py
  --smoke_test` runs. 3000-step LoRA run started in tmux `bfg-lora`, 4.2 s/step
  (~3.5 h).

## 13. Cleanup

- Removed `runs/smoke/ckpt-*.pt` (numbered smoke checkpoints) and
  `__pycache__`; kept `runs/smoke/last.pt` so the benchmark smoke runs
  immediately.
- Added `README.md`, `docs/research.md`, `docs/findings.md`, this log.

## 14. Strict evaluation sprint: LoRA checkpoint, prefix sweep, re-masking

### Code

- `sampler.py`: added `remask_ratio` (default 0.0) implementing the
  quota-preserving re-mask spec. Frozen pre-filled slots are counted against the
  quota and never compete for budget; the strongest `ceil((1-r) * n_decoded)`
  decoded slots are protected; the rest of the budget goes to the highest-margin
  candidates; the final step fills every remaining mask. `margin_floor` now only
  blocks slots instead of adding catch-up slots (documented behaviour change).
- Fixed bug #5 found during this sprint: frozen slots with `+inf` score were
  winning the whole `topk`, collapsing reconstruction to a single final fill.
  Mask counts before each forward (`C=32`, 16 frozen, `k=8`) are now
  `16,16,16,16,16,12,8,4`.
- `benchmark.py`: `--remask_ratio` (sweepable list), `--prefix_sweep` with
  `run_prefix_sweep` (warm-up, then causal vs cached `k=4`/`k=8` at
  `P=32/128/256/512` x batch 1/4, Markdown table), adapter reload now honours
  `lora_config["targets"]`.
- `train.py` / `model.py`: `--lora_mlp` (implies `--lora`) targeting
  `q,k,v,o,gate,up,down`; `targets` stored in `lora_config`.

### Verification

- `remask_ratio=0.0` is token-identical to commit `e2b7886` for steps
  1/2/3/4/8 x {greedy, T=0.5, position_decay=1.5}. The only intended difference
  is `margin_floor`, where the quota no longer over-fills.
- Quota invariant (`C=8`, `k=4`): masks before each forward `8,6,4,2`, then no
  masks; pre-filled slots unchanged; `r=0.25` changes the output at `k=4/k=8`
  and is a no-op at `k=2` (no interior step).
- `train.py --smoke_test`: `5.7695 -> 3.7964 PASS`.
- `benchmark.py --smoke_test`: runs (recovery 15.8%, k=2), garbage numbers by
  design.
- `train.py --smoke_test --lora --lora_mlp`: `targets=7`,
  `8,798,208 / 502,830,976` trainable, `5.4177 -> 3.5945 PASS`.

### Measurements (all on the 3000-step LoRA checkpoint)

- Loss curve: `docs/loss-curve.csv` (166 points, extracted from
  `runs/train-lora.log`). Run finished in 13,211 s, final val 3.7263, best
  3.6254 @2600.
- Task 1/3 benchmark: `runs/bench-lora.log` — reconstruction (fixed sampler)
  80.4 / 88.1 / 90.7% at k=2/4/8; generation k=8 T=0 rep4 42.2% -> 10.1% with
  `remask_ratio=0.25`, TPS unchanged; causal H4 0.964 / rep4 7.8% / 22.6 TPS.
- Task 2 prefix sweep: `runs/prefix-sweep.log` — k=4 speedup over causal 2.38x
  at `P=32` decaying to 1.14x at `P=512` (batch 4); k=8 wins only for short
  prefixes.
- Task 1 decoded text: `docs/decoded-samples.md` (12 prompts x 19 configs,
  verbatim).
- Re-ran reconstruction after the frozen-budget fix
  (`runs/bench-recon-fixed.log`); greedy generation metrics reproduced exactly,
  confirming the fix only touched the pre-filled path.

### Decision

Stage-3 gate was "launch `--lora_mlp` only if rep4 at `T=0`, `r=0.25` is still
> 20%". Measured 10.1% (k=8). **No `--lora_mlp` run launched.** The residual
failure is grammatical, not a repeated-4-gram attractor.

## 15. Second pass: mutable-slot quota, collision suppression, CPU crossover

### Code

- `sampler.py`: quota now interpolates over the mutable slots,
  `floor(s/K * (C - n_frozen))`, instead of `floor(s/K*C) - n_frozen`. On a
  50%-masked canvas the old form had `budget <= 0` for the first `K/2` steps, so
  k=8 behaved like k=4 and k=2 was single-shot.
- `sampler.py`: adjacent-token collision suppression (`suppress_adjacent=True`).
  Adjacent slots whose effective token is identical are compared on margin; the
  weaker one is scored `-inf` for that step and re-predicts next step with its
  neighbour committed. Only still-masked slots are held back, so a committed
  token is never revoked and the quota stays monotone.
- `benchmark.py`: `rep2` (2-gram repeat rate) and `adj` (adjacent identical
  tokens) columns, plus `--no_adj_suppress` for the A/B.

### Verification

- Mask counts before each forward, `C=32`, 16 frozen, suppression off:
  k=8 `16,14,12,10,8,6,4,2`, k=4 `16,12,8,4`, k=2 `16,8` — exactly the
  requested schedules. With suppression on, k=8 ends `...,4,4` (one pair
  collided at the last interior step); the final canvas still has 0 masks.
- `n_frozen=0` still reduces to `floor(s/K*C)`: token-identical to `21812a7` for
  steps 1/2/3/4/8 x {greedy, T=0.5, decay, floor}, suppression off (20/20).
- Mechanism check on the base model: `adj` 0.976 -> 0.898 at T=0 and
  0.961 -> 0.843 at T=0.3.
- `train.py --smoke_test`: `5.7694 -> 3.7964 PASS`. `benchmark.py --smoke_test`
  runs with the new columns.

### Measurements

- Reconstruction with the fixed quota: **89.7 / 92.3 / 96.4%** at k=2/4/8
  (`runs/bench-adj.log`), up from 80.4 / 88.1 / 90.7.
- Suppression A/B at k=8, r=0.25: T=0 rep4 10.1% -> 5.7%, adj 31.5% -> 24.2%,
  H4 0.949 -> 0.967, TPS unchanged; at T>=0.3 the deltas are inside noise.
  At k=4 T=0: rep4 45.1% -> 22.1%, adj 65.6% -> 41.1%.
- `not not not` disappears from the k=8/T=0 sample; runs of `the` committed in
  different steps survive.
- Systems result recorded in `docs/findings.md`: CPU block diffusion only beats
  causal GEMV for `k <= 4` and `P <= 256`; beyond that the `k*C` redundant
  evaluations saturate fp32 compute (k=4 1.87x -> 1.14x, k=8 0.88x at P=512).
- Full decoded text of the shipped sampler: `docs/decoded-samples-suppressed.md`.

### Stage 3 launched

```bash
python train.py --cpu --lora --lora_mlp --threads 20 --steps 3000 \
  --batch_size 4 --grad_accum 2 --data /tmp/bfg-corpus --max_tokens 2000000 \
  --out runs/bfg-lora-mlp > runs/train-lora-mlp.log
```

Same seed (default 0), batch 4 x accum 2, same corpus as `runs/bfg-lora`;
8.80M trainable params. Comparison benchmark pending its completion.

The run first started as a harness background shell at 13:49 (step 1 loss
14.6225) and was moved into tmux session `bfg-lora-mlp` at 14:09 for durability
(`reptyr` could not steal the process, so it restarted from step 1; step-1 loss
is identical, confirming the same seed and data order). Steady rate is 5.85 s/step
(111 tok/s), so 3000 steps is ~4.9 h, ETA ~19:05. A watcher fires when the tmux
session exits.

## 16. Cross-step repulsion + Stage 3 result (2026-11-15)

### Cross-step repulsion

`sampler.py`: `suppress_adjacent` now repels committed neighbours from an open
slot's logits (cross-step), in addition to the existing same-step margin
deferral. Prefix's last token is repelled at position 0. Final-step fill goes
left-to-right so a collision still resolves to a distinct token.

Verified:
- `n_frozen=0` still token-identical to `21812a7` (suppression off, 20/20).
- Quota schedules unchanged: k=8 `16,14,12,10,8,6,4,2`.
- Mechanism on base model: adj 0.976 -> 0.126 at T=0.
- Real checkpoint, k=8, r=0.25: adj 24.2% -> 4.6% at T=0, 17.2% -> 3.8% at
  T=0.3; rep4 5.7% -> 0.9% at T=0.
- `train.py --smoke_test`: 5.7695 -> 3.7964 PASS.

### Stage 3 result

`runs/bfg-lora-mlp` finished 19:04, val_loss 4.1111 (best 4.0123 @2600) vs
`runs/bfg-lora` 3.7263 (best 3.6254). The MLP adapter is ~0.4 nats worse at
every checkpoint and worse on generation (k=8/T=0 rep4 0.9% -> 6.0%, Dist-2
0.570 -> 0.530). Hypothesis not supported. `--lora_mlp` is not needed.

Benchmarked `runs/bfg-lora-mlp/last.pt` (recon + gen, suppression on) ->
`runs/bench-mlp.log`. Full table in `docs/findings.md`.

## 17. Right-Shifted Geometry & Carrier-Frontier Diffusion (2026-11-15)

1. **Shift Mismatch Resolved:**
   - Modified `model.py` so `canvas_forward` slices `out.logits[:, P - 1 : P + C - 1, :]`. Query position $P-1$ attends bidirectionally to canvas and predicts `canvas[0]`; query $P+j-1$ predicts `canvas[j]`.
   - Updated `prefill_prefix` to cache `prefix[:, :-1]` and `canvas_step_with_cache` to evaluate active `[prefix[:, -1:], canvas]`. Logit diff against full forward pass is `0.0`.
2. **Trainable Mask Vector:**
   - Added `mask_id` unfreezing hook in `apply_lora` (`model.py`). Only row 151665 receives gradients.
   - Updated `train.py` with direct Adam moment tracking for the single 896-dim vector, saving `ck["mask_emb"]`.
3. **Training Execution (`runs/bfg-shifted`):**
   - Ran 1500 steps on 2M token corpus (`batch_size=4, grad_accum=2, threads=20`).
   - Train loss dropped from 15.09 to 4.88; validation loss converged to 4.0259 (best 3.6565 at step 1000).
   - Saved `runs/bfg-shifted/last.pt` (8.4 MB).
4. **Geometric Carrier-Frontier Sampler:**
   - Updated `sampler.py` to implement geometric quota budgeting ending at `n_mutable`.
   - Carrier-aware frontier logic unmasks slots with concrete left carriers first, and seeds non-adjacent island anchors ($j \ge 2, j-1 \notin \text{pick}$) when unmasking in parallel.
   - Added `sample_canvas` sliding sub-block wrapper.
5. **Verification & Smoke Tests:**
   - `python train.py --smoke_test`: `11.9797 -> 3.6238 PASS`.
   - `python benchmark.py --smoke_test`: `recovery 26.3%, TPS 18.8`.
   - Final uncontended benchmarks on `runs/bfg-shifted/last.pt` (12 prompts, $P=128, C=32$):
     - Infilling recovery: $k=2 \to 87.6\%$, $k=4 \to 95.4\%$, $k=8 \to 95.9\%$.
     - Sliding sub-blocks $C_{\text{sub}}=8, k_{\text{sub}}=4$ (2 tok/pass parallel): Rep-4 **5.7%** (beating causal baseline's 8.0%), Adj **0.0%**, TPS 22.2.
     - Single block $C=32, k=4$ (8 tok/pass): TPS **48.4** (1.8x faster than causal).
     - Generated texts logged in `docs/decoded-samples-shifted.md`.


## 18. Persistent KV-Cache & Island Anchor Margin Gating (2026-11-16)

1. **Persistent KV-Cache Across Sub-Blocks (`sampler.py`):**
   - Eliminated prompt re-prefill across sliding blocks in `sample_canvas`.
   - Calls `kv = prefill_prefix(model, cur_prefix, cur_keep)` once before the block loop.
   - After each completed sub-block of length $c_{\text{sub}}$, advances `kv` by evaluating $[ \text{prefix}_{[-1]}, \text{block}_{[:-1]} ]$ with causal position IDs matching `mask_position_ids(new_keep, 0)[:, p_cached : p_cached + c_sub]`.
   - Leaves `block[:, -1]` as `prefix_last_id` for the next sub-block.
2. **Margin-Gated Island Anchors (`sampler.py`):**
   - Gated `cand_island` by `margin[i, j] >= anchor_thresh` (default `2.5`).
   - When step budget exceeds concrete carriers, island anchors are only selected if they meet high-confidence structural threshold; otherwise, blind unanchored guesses are prevented from committing on step 1.
3. **Verification:**
   - `python train.py --smoke_test`: `smoke convergence: 11.9797 -> 3.6238 PASS`.
   - `python benchmark.py --smoke_test`: `Reconstruction 26.3% TPS 17.9, Generation TPS 29.1 PASS`.

## 19. Anti-Starvation Carrier Frontier & Verified Generations (2026-11-16)

1. **Resolution of Budget Starvation in `sampler.py`:**
   - In `sample_canvas_block`, after picking slots with `has_carrier` and high-margin non-adjacent island anchors (`margin >= anchor_thresh`), any remaining quota deficit (`len(pick) < budget`) is filled by advancing contiguous frontier slots ($j - 1 \in \text{pick}$), followed by highest-rank open slots.
   - Prevents step-starvation where earlier passes only unmask 1 token and dump all remaining masks on the final step.
2. **Evaluation on `runs/bfg-shifted/last.pt`:**
   - Evaluated Prompts 0, 1, 4 for $C_{\text{sub}}=8, k_{\text{sub}}=4$ and $k_{\text{sub}}=6$.
   - Output completely eliminated `"the a the a"` stutters, restoring rich coherent narrative, technical configs, and structured ordinal lists.
   - Smoke tests verified: `smoke convergence: 11.9797 -> 3.6238 PASS`.

## 20. Suffix-Mask Curriculum, Fine-Tuning & Interactive Visualizer Web UI (2026-11-16)

1. **Hybrid Suffix + Random Mask Curriculum (`train.py`):**
   - 20% full mask ($t=1.0$) for initial denoising.
   - 40% contiguous suffix mask $[m:]$ to match free-generation carrier frontier, directly training 2-step and 3-step lookahead without right-hand context leaks.
   - 40% random scatter mask ($t \sim U(0.20, 0.95)$) to preserve infilling recovery.
   - Added `--resume` to warm-start LoRA and mask embedding weights.
2. **Warm-Start Training Run (`runs/bfg-polished`):**
   - Launched 600-step polish run in `tmux bfg-polish` from `runs/bfg-shifted/last.pt`.
3. **Interactive Visualizer & Chat CLI (`chat.py`):**
   - Created standalone [chat.py](file:///home/gces/.bin/.s/.scripts/bfg-diff/chat.py).
   - Features: step-by-step unmasking visualization (`░` $\to$ tokens), side-by-side TPS and text comparison against causal baseline, live `/mode`, `/vis`, `/compare` toggling.
   - Built lightweight HTML/Web visualizer serving on port `8766`, routed via `bfg-dff.gces.net.in`.

## 21. Carrier-Verified Lookahead Sampling (2026-11-16)

1. **Root Cause Analysis:**
   - Diagnosed that picking 2 tokens on step 1 ($C_{\text{sub}}=8, k_{\text{sub}}=4$) caused slot 1 to draft tokens without an unmasked carrier at slot 0 during that pass.
   - The previous implementation permanently committed all decoded tokens into `keep_token`, locking `"The is"` and `"I I'm"` into the canvas permanently.
2. **Carrier-Verified Lookahead Architecture (`sampler.py`):**
   - Introduced `verified` boolean mask (permanent locked tokens) separate from `decoded` (all currently placed tokens).
   - Forward pass inputs are masked to only verified tokens (`feed_canvas = torch.where(verified, canvas, mask_id)`), ensuring that lookahead drafts do not pollute subsequent forward passes.
   - Slots whose carrier was verified during the forward pass are promoted to `verified = True`.
   - On the final step ($s == \text{steps}$), any remaining unverified draft or mask slots are conditioned and verified left-to-right on real carriers.
3. **Live Verification & Smoke Tests:**
   - `python train.py --smoke_test`: `smoke convergence: 11.9797 -> 4.0549 PASS`.
   - Tested against live API at `http://127.0.0.1:8766/api/generate` with 6 multi-prompt test cases (3 chat, 3 completion).
   - Traces confirmed zero lookahead draft stutters: `"What is the capital of France?"` cleanly generated `"The capital of France is Paris."`, and `"How are you doing today?"` generated `"I'm doing well, thank you. How can I assist you today?"`.

## 22. Web UI Streaming Polish & Stop Token Early Termination (2026-11-16)

1. **Web UI SSE Streaming Fixes (`chat.py`):**
   - Converted `HTML_PAGE` to a raw Python triple-quoted string (`r"""..."""`) and used `String.fromCharCode(10)` / `String.fromCharCode(13)` in client JavaScript to prevent CRLF parsing syntax errors. Verified with `node --check` (0 errors).
   - Set HTTP response headers `Cache-Control: no-cache, no-transform` and `X-Accel-Buffering: no`.
   - Appended 2048-byte comment padding (`": " + (" " * 2048) + "\n\n"`) to each SSE chunk, forcing Cloudflare's 4KB edge chunk buffer to flush each diffusion step immediately.
   - Reordered streaming execution to run BFG Discrete Diffusion first, delivering instant animation.
2. **Stop Token Early Exit (`sampler.py`):**
   - Added `stop_token_ids` parameter to `sample_canvas` and `sample_canvas_block`.
   - Enabled early `break` on step loop when all tokens are verified (`verified.all()`) or when any verified token in the canvas belongs to `stop_token_ids`.
   - Enabled early `break` across sub-blocks in `sample_canvas` when a block contains a stop token, dropping 16-token latency on short chat answers from 3.42s to 1.75s (50% reduction).
3. **Verification:**
   - `python train.py --smoke_test`: `smoke convergence: 11.9798 -> 4.0549 PASS`.
   - `python benchmark.py --smoke_test`: `Reconstruction 63.2% TPS 9.2, Generation TPS 5.0 PASS`.
   - Live endpoints verified locally (`http://127.0.0.1:8766`) and through Cloudflare tunnel (`https://bfg-dff.gces.net.in`).

## 23. Speculative Lookahead Verification & UI Display Masking (2026-11-16)

1. **True Speculative Lookahead Verification (`sampler.py`):**
   - Eliminated the sequential step-4 forward-pass loop that ran up to 5 additional forward passes on unverified trailing slots.
   - Preserved lookahead draft predictions in `feed_canvas` so the forward pass evaluates whether the speculative draft token at $m+1$ is verified in the same step.
   - Implemented speculative chain verification: when slot $m$ is verified, if slot $m+1$'s draft matches the top sampled token or is within relaxed NCM energy tolerance ($z[m+1, \text{top}] - z[m+1, \text{draft}] \le 1.2$) and satisfies anti-stutter checks, slot $m+1$ is accepted and verified in the same forward pass.
   - Cleans up trailing slots beyond the active frontier to avoid wild unverified drafts.
2. **Visual Flicker Elimination in UI Trace (`sampler.py`):**
   - Updated `step_callback` to mask distant unverified slots beyond the active lookahead frontier ($m_{\text{frontier}} + 1$).
   - The UI display trace shows only verified tokens plus the active 1-token frontier draft, preventing visual chip fluttering and rewriting in the Web visualizer.
3. **Verification:**
   - `python train.py --smoke_test`: `smoke convergence: 11.9797 -> 4.0549 PASS`.
   - `python benchmark.py --smoke_test`: `Reconstruction recovery 26.3% TPS 12.1; Generation TPS 21.2`.
   - Live visualizer service verified on port 8766.

## 25. Active-Draft Speculative Verification & Sub-Block Boundary Truncation (2026-11-16)

1. **Active Lookahead Drafting in `feed_canvas`:**
   - Modified [sampler.py](file:///home/gces/.bin/.s/.scripts/bfg-diff/sampler.py) to feed verified tokens plus exactly the single active lookahead draft token at $m_{\text{frontier}}$, masking distant slots.
   - Evaluates slot $m$ and checks speculative acceptance of slot $m+1$ (`draft_c1 == top_c1`) in the same forward pass.
2. **Sub-Block Truncation at Verified Frontier:**
   - In `sample_canvas_block`, free-generation sub-blocks truncate at the contiguous verified boundary (`canvas[:, :m_end]`), leaving infilled reconstruction full length.
   - `sample_canvas` dynamically decrements remaining quota and advances KV cache by $c_{\text{actual}}$, eliminating boundary duplication and unanchored emergency argmax tail passes.
3. **Verification & Tests:**
   - `python train.py --smoke_test`: `smoke convergence: 11.9797 -> 4.0549 PASS`.
   - `python benchmark.py --smoke_test`: `recovery = 21.1%`, `rep4 = 0.0%`, `adj = 0.0%`, `Dist2 = 1.000`, `TPS = 29.9`.
   - Live endpoint test on `"What is a while loop in Python?"`:
     Clean text generated in 6.71s with 0 stutters and 0 boundary duplicates (`"A while loop in Python is a control structure that repeatedly executes a block of code as long as a certain condition is true. It is often used in situations where"`).
   - Live service restarted on port 8766.





