"""Benchmark BFG-Diff vs causal generation: reconstruction, diversity, throughput."""
import argparse
import math
import time
from collections import Counter
from pathlib import Path

import torch

from model import prepare
from sampler import sample_canvas_block

# Long reference documents (prose, docs, markdown, JSON) for masked reconstruction.
REFERENCE_TEXTS = [
    "The river begins as a thin thread of meltwater high in the mountains. It gathers "
    "strength from a hundred unnamed streams, cuts a narrow channel through the granite, "
    "and finally spreads into a wide, slow delta where herons stand in the shallows. "
    "People have built towns along its banks for centuries, drawn by the water and the "
    "rich soil it leaves behind after every spring flood.",
    "## Installing the service\n\n"
    "The service ships as a single static binary. Download the release archive for your "
    "platform, verify the checksum, and unpack it into any directory on your PATH.\n\n"
    "1. Fetch the archive with curl.\n"
    "2. Compare the SHA-256 digest against the published value.\n"
    "3. Run `service --version` to confirm the installation.\n\n"
    "On first launch the service writes a default configuration file to "
    "`/etc/service/config.yaml` and listens on port 8080.\n",
    '{\n  "id": "prod_1042",\n  "name": "Standing desk",\n  "price": 349.0,\n'
    '  "currency": "USD",\n  "in_stock": true,\n  "tags": ["office", "furniture"],\n'
    '  "dimensions": {"width_cm": 140, "depth_cm": 70, "height_cm": 120},\n'
    '  "reviews": [{"score": 5, "text": "Solid and quiet."}, {"score": 4, "text": "Assembly took an hour."}]\n}\n',
    "Diffusion language models corrupt a sequence with random masks and then learn to "
    "recover the original tokens. Unlike autoregressive models, every token in the "
    "corrupted block can attend to every other token, so the model can revise its earlier "
    "predictions. At sampling time the least ambiguous positions are decoded first, which "
    "gives the remaining positions more context and reduces the chance of repeating a "
    "phrase or splitting a word in the wrong place.\n",
    "- Set up a virtual environment before installing dependencies.\n"
    "- Pin every dependency to an exact version in the lock file.\n"
    "- Run the test suite on every supported Python version.\n"
    "- Cache large downloads between CI jobs to keep builds fast.\n"
    "- Delete build artifacts after publishing the release.\n",
    "The experiment used a single eight-core machine with no GPU. Each configuration ran "
    "three times, and the reported numbers are medians. Throughput was measured after a "
    "warm-up pass so that lazy initialisation and page faults did not inflate the timing. "
    "Latency includes tokenisation but excludes model loading.\n",
    "Dear team,\n\n"
    "The migration is complete. All customer records now live in the new database, and "
    "the old cluster is read-only. Please update your dashboards to point at the new "
    "endpoints before Friday. If you notice missing rows, open a ticket with the data "
    "platform group and include the affected account ids.\n\n"
    "Thanks for your patience.\n",
    "A balanced vinaigrette needs only three things: oil, acid, and salt. Whisk two parts "
    "olive oil with one part vinegar, add a pinch of salt, and taste. If it feels sharp, "
    "add another spoon of oil; if it feels flat, add a few drops of vinegar. Mustard helps "
    "the emulsion hold together and adds a pleasant edge.\n",
    "### Release notes 2.4.0\n\n"
    "- Added incremental decoding for long documents.\n"
    "- Fixed a crash when the prompt exceeded the context window.\n"
    "- Reduced cold-start memory by 18 percent.\n"
    "- Deprecated the legacy `--causal` flag; use `--mode causal` instead.\n"
    "- Updated the bundled tokenizer to revision 2024-11.\n",
]

# Short openers spanning prose, docs, markdown, JSON and code.
GEN_PROMPTS = [
    "Once upon a time, in a small coastal village, the fishermen noticed that",
    "To configure the proxy, edit the settings file and set",
    "## Quick start\n\nInstall the package, then run",
    "{\n  \"task\": \"summarise\",\n  \"input\": \"Quarterly revenue rose",
    "- first, collect the measurements\n- second, clean the outliers\n- third,",
    "The committee reviewed the proposal and concluded that",
    "def merge_intervals(intervals):\n    \"\"\"Merge overlapping intervals.\"\"\"\n",
    "In 1969, engineers at the laboratory succeeded in",
    "Recipe: heat the pan, add the onions, and cook until",
    "The report argues that remote work changes urban economies because",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--num_prompts", type=int, default=100)
    p.add_argument("--prefix_len", type=int, default=384)
    p.add_argument("--canvas_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--steps", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--floors", type=float, nargs="+", default=[0.0],
                   help="margin floors tested for every step count")
    p.add_argument("--show", type=int, default=2, help="side-by-side prompts to print")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--smoke_test", action="store_true")
    args = p.parse_args()
    if args.smoke_test:
        args.checkpoint = args.checkpoint or "runs/smoke/last.pt"
        args.num_prompts, args.canvas_len, args.prefix_len = 2, 16, 128
        args.batch_size, args.steps, args.floors, args.show = 1, [2], [0.0], 1
    else:
        args.checkpoint = args.checkpoint or "runs/bfg-diff/last.pt"
    return args


def left_pad(seqs, pad_id, device):
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long, device=device)
    keep = torch.zeros((len(seqs), width), dtype=torch.bool, device=device)
    for i, s in enumerate(seqs):
        ids[i, width - len(s) :] = torch.tensor(s, device=device)
        keep[i, width - len(s) :] = True
    return ids, keep


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def tokenize(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def ngram_entropy(seqs, n=4):
    vals = []
    for s in seqs:
        grams = Counter(tuple(s[i : i + n]) for i in range(len(s) - n + 1))
        total = sum(grams.values())
        vals.append(
            -sum((c / total) * math.log2(c / total) for c in grams.values()) / math.log2(total)
            if total > 1
            else 0.0
        )
    return sum(vals) / len(vals)


def repeat_rate(seqs, n=4):
    vals = []
    for s in seqs:
        grams = [tuple(s[i : i + n]) for i in range(max(0, len(s) - n + 1))]
        vals.append(1 - len(set(grams)) / len(grams) if grams else 0.0)
    return sum(vals) / len(vals)


def distinct(seqs, n):
    """Corpus-level Dist-n: unique n-grams / total n-grams."""
    total, unique = 0, set()
    for s in seqs:
        grams = [tuple(s[i : i + n]) for i in range(max(0, len(s) - n + 1))]
        total += len(grams)
        unique.update(grams)
    return len(unique) / total if total else 0.0


@torch.inference_mode()
def run_reconstruction(model, tok, args, mask_id, device, steps):
    n = args.num_prompts
    prefixes, golds = [], []
    for i in range(n):
        ids = tokenize(tok, REFERENCE_TEXTS[i % len(REFERENCE_TEXTS)])
        need = args.prefix_len + args.canvas_len
        if len(ids) < need:
            ids = (ids * (need // max(1, len(ids)) + 1))[:need]
        ids = ids[:need]
        prefixes.append(torch.tensor(ids[: args.prefix_len]))
        golds.append(torch.tensor(ids[args.prefix_len :]))
    prefix_all = torch.stack(prefixes).to(device)
    gold_all = torch.stack(golds).to(device)
    mask = torch.rand(gold_all.shape, generator=torch.Generator().manual_seed(0)) < 0.5

    correct = total = 0
    token_ids = []
    sync(device)
    start = time.perf_counter()
    for i in range(0, n, args.batch_size):
        prefix, gold = prefix_all[i : i + args.batch_size], gold_all[i : i + args.batch_size]
        m = mask[i : i + args.batch_size].to(device)
        canvas = sample_canvas_block(
            model, prefix, canvas_len=args.canvas_len, steps=steps, mask_id=mask_id,
            canvas=gold.masked_fill(m, mask_id),
        )
        correct += (canvas[m] == gold[m]).sum().item()
        total += m.sum().item()
        token_ids.extend(canvas.tolist())
    sync(device)
    return correct / total, len(token_ids) * args.canvas_len / (time.perf_counter() - start)


@torch.inference_mode()
def run_sampler_generation(model, tok, prompt_ids, args, mask_id, device, steps, margin_floor):
    token_ids = []
    sync(device)
    start = time.perf_counter()
    for i in range(0, len(prompt_ids), args.batch_size):
        ids, keep = left_pad(prompt_ids[i : i + args.batch_size], tok.pad_token_id, device)
        canvas = sample_canvas_block(
            model, ids, canvas_len=args.canvas_len, steps=steps, mask_id=mask_id,
            prefix_keep=keep, margin_floor=margin_floor,
        )
        token_ids.extend(canvas.tolist())
    sync(device)
    elapsed = time.perf_counter() - start
    texts = [tok.decode(t, skip_special_tokens=True) for t in token_ids]
    return token_ids, texts, len(token_ids) * args.canvas_len / elapsed


@torch.inference_mode()
def run_causal_generation(model, tok, prompt_ids, args, device):
    token_ids = []
    sync(device)
    start = time.perf_counter()
    for i in range(0, len(prompt_ids), args.batch_size):
        ids, keep = left_pad(prompt_ids[i : i + args.batch_size], tok.pad_token_id, device)
        out = model.generate(
            input_ids=ids,
            attention_mask=keep.long(),
            max_new_tokens=args.canvas_len,
            do_sample=False,
            use_cache=True,
            pad_token_id=tok.pad_token_id,
        )
        token_ids.extend(out[:, ids.shape[1] :].tolist())
    sync(device)
    elapsed = time.perf_counter() - start
    texts = [tok.decode(t, skip_special_tokens=True) for t in token_ids]
    return token_ids, texts, sum(len(s) for s in token_ids) / elapsed


def main():
    args = parse_args()
    torch.set_num_threads(args.threads or (torch.get_num_threads()))
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        ("cpu" if args.device == "auto" else args.device)
    )
    dtype = (
        torch.bfloat16
        if args.dtype == "bf16" or (args.dtype == "auto" and device.type == "cuda")
        else torch.float32
    )
    model, tok, mask_id = prepare(args.model, device=device, dtype=dtype)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if Path(args.checkpoint).exists():
        ck = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ck["model"])
        mask_id = ck.get("mask_id", mask_id)
    else:
        print(f"warning: checkpoint {args.checkpoint!r} not found; benchmarking base model")
    model.eval()

    prompts = [GEN_PROMPTS[i % len(GEN_PROMPTS)] for i in range(args.num_prompts)]
    prompt_ids = [tokenize(tok, p) for p in prompts]

    print(f"\nReconstruction | {args.num_prompts} prefixes | P={args.prefix_len} C={args.canvas_len}")
    print(f"{'method':<18}{'recovery':>10}{'TPS':>12}")
    for steps in args.steps:
        rec, tps = run_reconstruction(model, tok, args, mask_id, device, steps)
        print(f"{f'bfg-diff k={steps}':<18}{rec:>9.1%}{tps:>12,.1f}")

    print(f"\nGeneration | {args.num_prompts} prompts | C={args.canvas_len}")
    print(f"{'method':<22}{'H4':>7}{'rep4':>7}{'Dist1':>7}{'Dist2':>7}{'TPS':>10}")
    outputs = {}
    for floor in args.floors:
        for steps in args.steps:
            ids, texts, tps = run_sampler_generation(
                model, tok, prompt_ids, args, mask_id, device, steps, floor
            )
            name = f"bfg-diff k={steps}" + (f" f={floor:g}" if floor else "")
            outputs[name] = texts
            print(
                f"{name:<22}{ngram_entropy(ids):>7.3f}{repeat_rate(ids):>7.1%}"
                f"{distinct(ids, 1):>7.3f}{distinct(ids, 2):>7.3f}{tps:>10,.1f}"
            )
    ids, texts, tps = run_causal_generation(model, tok, prompt_ids, args, device)
    outputs["causal generate"] = texts
    print(
        f"{'causal generate':<22}{ngram_entropy(ids):>7.3f}{repeat_rate(ids):>7.1%}"
        f"{distinct(ids, 1):>7.3f}{distinct(ids, 2):>7.3f}{tps:>10,.1f}"
    )

    if args.show:
        print("\nSide-by-side completions")
        for i in range(min(args.show, len(prompts))):
            print(f"\n[{i}] {prompts[i]}")
            for name, texts in outputs.items():
                print(f"  {name}: {texts[i]}")


if __name__ == "__main__":
    main()
