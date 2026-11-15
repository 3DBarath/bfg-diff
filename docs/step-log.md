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

## 11. Cleanup

- Removed `runs/smoke/ckpt-*.pt` (numbered smoke checkpoints) and
  `__pycache__`; kept `runs/smoke/last.pt` so the benchmark smoke runs
  immediately.
- Added `README.md`, `docs/research.md`, `docs/findings.md`, this log.
