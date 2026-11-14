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

## 8. Cleanup

- Removed `runs/smoke/ckpt-*.pt` (numbered smoke checkpoints) and
  `__pycache__`; kept `runs/smoke/last.pt` so the benchmark smoke runs
  immediately.
- Added `README.md`, `docs/research.md`, `docs/findings.md`, this log.
