# BFG-Diff

Block-autoregressive discrete diffusion on top of a pretrained causal LM.
Qwen2.5 is fine-tuned to denoise a 128-token canvas conditioned on a 384-token
prefix, then sampled non-autoregressively in 4 or 8 steps.

Pure PyTorch + Transformers. No custom vocabulary, no grammar parsers, no
task-specific tokens.

## Files

| Path | Purpose |
|---|---|
| `model.py` | Tokenizer/`[MASK]` aliasing, model loading, 4D hybrid attention mask, canvas forward |
| `dataset.py` | `GeneralTextDataset` over text dirs/files/string lists; synthetic text for smoke runs |
| `sampler.py` | Top-1-margin iterative unmasking (`sample_canvas_block`) |
| `train.py` | Training loop, dynamic masking, checkpointing, `--smoke_test` |
| `benchmark.py` | Reconstruction, repetition/entropy, and TPS vs causal `generate()` |
| `docs/research.md` | Design notes and rationale |
| `docs/findings.md` | Measured results, bugs found, limitations |
| `docs/step-log.md` | Chronological build/verification log |

## Requirements

```bash
pip install torch transformers
```

CUDA is optional. On CUDA the model loads and trains in `bfloat16`; on CPU it
falls back to `float32` (CPU bf16 training is not exercised).

First run downloads `Qwen/Qwen2.5-0.5B` (~1 GB).

## Quickstart

```bash
python train.py --smoke_test
python benchmark.py --smoke_test
```

The smoke run uses synthetic text, 148 chunks of 128 prefix + 64 canvas tokens,
12 steps, and passes only if a fixed probe loss decreases. It writes
`runs/smoke/last.pt`.

## Real training

```bash
python train.py --data /path/to/texts --steps 4000 --batch_size 4 --grad_accum 4
```

`--data` accepts a directory (recursively reads files, skipping binaries), a
single file, or is omitted to use synthetic text. Any text format is fine
(`.txt`, `.md`, `.json`, source code).

Defaults: `P=384`, `C=128`, lr `5e-5`, warmup 500 optimizer updates, cosine
decay, weight decay `0.01`, AdamW, checkpoint every 500 steps to
`runs/bfg-diff/`.

## Benchmark

```bash
python benchmark.py --checkpoint runs/bfg-diff/last.pt --num_prompts 100
```

Reports, for k=4 and k=8:

- masked-token recovery on 50%-masked real text,
- 4-gram entropy and repeat rate of free generations,
- wall-clock tokens/s, alongside Hugging Face causal `generate()`.

The default checkpoint is `runs/bfg-diff/last.pt`. A missing checkpoint is
reported and the base model is benchmarked instead.

## Mechanics

1. **Mask token.** `<|extra_0|>` is added to the tokenizer and resolves to id
   `151665`, an unused row inside Qwen2.5's existing `151936`-row embedding
   table. No embedding resize, no reinitialisation; the LM head is tied so both
   sides use the same row.
2. **Hybrid mask.** Prefix queries see only earlier prefix tokens. Canvas
   queries see the whole prefix and the whole canvas block. Prefix queries never
   see canvas tokens. The mask is a `(B, 1, P+C, P+C)` additive float tensor
   (`finfo.min` for blocked, `0.0` for allowed), passed through SDPA.
3. **Training.** Per sample, `t ~ U(0.15, 0.95)` and exactly `floor(t*C)` canvas
   tokens are replaced with the mask. Cross-entropy is computed only on those
   positions.
4. **Sampling.** Start from an all-mask canvas. Each step, score every masked
   slot by `log p(1) - log p(2)`, unmask the `floor(s/K*C)` highest-margin
   slots, and freeze them. With a partially filled canvas only the remaining
   masked slots are eligible.

See `docs/research.md` for the reasoning and `docs/findings.md` for measured
numbers, including the CPU smoke-test results.
