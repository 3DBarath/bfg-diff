# BFG-Diff

Block-autoregressive discrete diffusion on top of a pretrained causal LM.
Qwen2.5 is fine-tuned to denoise a 128-token canvas conditioned on a 384-token
prefix, then sampled non-autoregressively in 4 or 8 steps.

Pure PyTorch + Transformers. No custom vocabulary, no grammar parsers, no
task-specific tokens.

## Files

| Path | Purpose |
|---|---|
| `model.py` | Tokenizer/`[MASK]` aliasing, model loading, 4D hybrid mask, canvas forward, prefix KV cache, LoRA wrapper |
| `dataset.py` | `GeneralTextDataset` over text dirs/files/string lists; synthetic text for smoke runs |
| `sampler.py` | Top-1-margin iterative unmasking (`sample_canvas_block`) |
| `train.py` | Training loop, dynamic masking, checkpointing, `--smoke_test` |
| `benchmark.py` | Reconstruction, repetition/entropy, and TPS vs causal `generate()` |
| `docs/research.md` | Design notes and rationale |
| `docs/findings.md` | Measured results, bugs found, limitations |
| `docs/step-log.md` | Chronological build/verification log |

## Requirements

```bash
pip install torch transformers peft
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
(`.txt`, `.md`, `.json`, source code). `--limit` caps files and `--max_tokens`
caps the corpus.

Defaults: `P=384`, `C=128`, lr `5e-5`, warmup 500 optimizer updates, cosine
decay, weight decay `0.01`, AdamW, checkpoint every 500 steps to
`runs/bfg-diff/`.

### CPU preset

`--cpu` sets `P=128`, `C=32`, batch 2 x gradient accumulation 8 (effective 16),
1000 steps, checkpoints every 200 steps, `float32`, `--device cpu`, and enables
`--lora`. Explicit flags override the preset. Threads default to all cores; on
a 10-core/20-thread Xeon, `--threads 10` matches `--threads 20` in measured
throughput, so either is fine.

```bash
python train.py --cpu --threads 20 --lora \
  --steps 3000 --batch_size 4 --grad_accum 2 \
  --data /path/to/docs --limit 2000 --max_tokens 2000000
```

CPU training is fp32 on purpose: bf16/AMX is emulated on pre-Sapphire-Rapids
Xeons and is slower than fp32 there.

### LoRA

`--lora` freezes the base model and trains rank-16 adapters on
`q_proj, k_proj, v_proj, o_proj` (`--lora_r`, `--lora_alpha 32`,
`--lora_dropout 0.05`), 2.16M of 496M parameters. `--lora_mlp` adds
`gate_proj, up_proj, down_proj` (8.80M parameters) for the stage-3 ablation.
`--lr` defaults to `5e-4` with `--lora` and `5e-5` without. Checkpoints then
hold only the adapter (`~9-20 MB` instead of 1.9 GB) plus `lora_config`;
`benchmark.py` rebuilds the PEFT wrapper when it sees a `lora` key, and still
reads full-model checkpoints.

## Benchmark

```bash
python benchmark.py --checkpoint runs/bfg-diff/last.pt --num_prompts 100 \
  --canvas_len 32 --steps 4 8 --temps 0.0 0.3 0.5 --top_k 50 --show 3
```

Reports, for every step-count x margin-floor x temperature x decay combination:

- masked-token recovery on 50%-masked real text,
- 4-gram entropy, 4-gram repeat rate, **2-gram repeat rate** and **`adj`**
  (fraction of neighbouring tokens that are identical), Dist-1 and Dist-2,
- wall-clock tokens/s, alongside Hugging Face causal `generate()`.

`--top_k` truncates the stochastic sampler to the `k` most likely tokens per
slot (default 50); it has no effect at `--temps 0.0`. `--remask_ratio 0.25`
re-opens the weakest 25% of decoded slots every step, keeping the quota exact.
Adjacent-token collision suppression is on by default; `--no_adj_suppress`
turns it off for the A/B comparison. `--prefix_sweep` instead prints a Markdown
TPS table for `P = 32, 128, 256, 512` x batch 1/4, comparing causal `generate()`
with cached `k=4` and `k=8`.

It also prints side-by-side completions for the first `--show` prompts
(diffusion vs causal), so repetition and phrasing can be inspected directly.
`--floors` is the confidence-margin sweep; `0.0` is the original rigid quota.

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
3. **Training.** 20% of batches are fully masked (`t=1`); the rest draw
   `t ~ U(0.20, 1.0)` per sample and replace exactly `floor(t*C)` canvas tokens
   with the mask. Cross-entropy is computed only on those positions.
4. **Sampling.** Start from an all-mask canvas. Each step, score every masked
   slot by `log p(1) - log p(2)`, unmask the `floor(s/K*C)` highest-margin
   slots, and freeze them. With a partially filled canvas the schedule runs over
   the `C - n_frozen` mutable slots only, so pre-filled slots never starve it.
   `margin_floor` optionally holds back slots whose top-1 is not clearly ahead
   of the runner-up; held-back slots are filled by argmax at the end.
   `temperature > 0` samples tokens with Gumbel noise over the `top_k` most
   likely logits (ranking still uses clean margins), and `position_decay > 0`
   biases step 1 toward the canvas start. `remask_ratio` lets the weakest
   decoded slots re-compete against masked ones each step without changing the
   quota. Adjacent slots that sample the same token in the same step keep only
   the more confident one, so the weaker slot re-predicts with its neighbour in
   context.
5. **Prefix KV cache.** The prefix is prefilled once per prompt; each diffusion
   step forwards only the `C` canvas tokens against the cached prefix keys and
   values, so a step costs `C` tokens instead of `P+C`.

See `docs/research.md` for the reasoning and `docs/findings.md` for measured
numbers, including the CPU smoke-test results.
