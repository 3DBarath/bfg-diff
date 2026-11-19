"""Benchmark BFG-Diff vs causal generation: reconstruction, diversity, throughput."""
import argparse
import math
import time
from collections import Counter
from pathlib import Path

import torch

from model import LORA_TARGETS, apply_lora, canvas_forward, canvas_step_with_cache, prefill_prefix, prepare
from sampler import sample_canvas, sample_canvas_block

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

# 50 unique, diverse openers spanning prose, technical docs, code, JSON, reasoning, mathematics.
GEN_PROMPTS = [
    # Prose & Literature
    "Once upon a time, in a small coastal village, the fishermen noticed that",
    "The old lighthouse keeper climbed the spiral staircase as the storm began to",
    "Through the dense morning fog, the outline of the abandoned fortress slowly",
    "She opened the dusty leather journal and read the very first entry, which stated that",
    "Beneath the roots of the ancient oak tree lay an iron chest containing",
    "The autumn wind scattered dry leaves across the stone courtyard while the travelers",
    "Deep in the desert canyon, archaeologists discovered carved symbols indicating",
    "The astronomer adjusted the brass telescope and observed a strange flare emanating from",
    "Across the silent frozen lake, the wolves howled under the pale green light of",
    "Every year during the spring equinox, the villagers gathered at the summit to",

    # Technical Docs & Systems
    "To configure the proxy, edit the settings file and set",
    "## Quick start\n\nInstall the package, then run",
    "- first, collect the measurements\n- second, clean the outliers\n- third,",
    "The primary bottleneck in distributed key-value storage occurs when",
    "### Authentication Architecture\n\nAll incoming client requests must include a valid bearer token in the",
    "To scale database write throughput across multiple read replicas, implement",
    "During memory compaction, the garbage collector pauses application worker threads to",
    "The network load balancer distributes inbound TCP connections evenly by using",
    "When a worker node experiences a kernel panic or loses heartbeat contact with the coordinator,",
    "To ensure zero downtime deployments, traffic is gradually shifted from the blue cluster to",

    # Software Engineering & Code
    "def merge_intervals(intervals):\n    \"\"\"Merge overlapping intervals.\"\"\"\n",
    "async def fetch_user_profile(user_id: str, db_pool) -> dict:\n    \"\"\"Query user metadata from database.\"\"\"\n",
    "class LRUCache:\n    def __init__(self, capacity: int):\n        self.capacity = capacity\n",
    "def binary_search(arr, target):\n    \"\"\"Find target index in sorted array.\"\"\"\n    left, right = 0, len(arr) - 1\n",
    "import torch\nimport torch.nn as nn\n\nclass SelfAttentionBlock(nn.Module):\n    def __init__(self, d_model, n_heads):\n",
    "const handleFormSubmission = async (event: React.FormEvent) => {\n    event.preventDefault();\n",
    "fn parse_http_headers(raw_request: &[u8]) -> Result<HeaderMap, ParseError> {\n",
    "SELECT users.id, users.email, COUNT(orders.id) AS total_orders FROM users JOIN orders ON",
    "def dijkstra_shortest_path(graph, start_node):\n    \"\"\"Compute shortest distance to all reachable nodes.\"\"\"\n",
    "docker-compose build --parallel && docker-compose up -d && echo \"Services started successfully on",

    # JSON & Structured Data
    "{\n  \"task\": \"summarise\",\n  \"input\": \"Quarterly revenue rose",
    "{\n  \"status\": 200,\n  \"data\": {\n    \"cluster_name\": \"us-east-prod-4\",\n    \"healthy_nodes\":",
    "[\n  {\"id\": 101, \"title\": \"Database Migration\", \"priority\": \"critical\", \"assigned_to\":",
    "{\n  \"model\": \"qwen-2.5-0.5b\",\n  \"temperature\": 0.7,\n  \"messages\": [\n    {\"role\": \"user\", \"content\":",
    "{\n  \"event_type\": \"payment_completed\",\n  \"transaction\": {\n    \"amount_cents\": 4999,\n    \"currency\": \"USD\",\n",
    "{\n  \"query\": \"SELECT name, price FROM inventory WHERE in_stock = true\",\n  \"execution_time_ms\":",
    "{\n  \"error\": null,\n  \"result\": {\n    \"embeddings\": [0.0412, -0.1983, 0.8841,\n",
    "{\n  \"device\": \"cuda:0\",\n  \"total_memory_mb\": 15360,\n  \"allocated_memory_mb\":",
    "[\n  {\"epoch\": 1, \"train_loss\": 4.120, \"val_loss\": 3.891},\n  {\"epoch\": 2, \"train_loss\":",
    "{\n  \"service\": \"auth-gateway\",\n  \"uptime_seconds\": 864000,\n  \"p99_latency_ms\":",

    # Reasoning, Analysis & Mathematics
    "The committee reviewed the proposal and concluded that",
    "In 1969, engineers at the laboratory succeeded in",
    "Recipe: heat the pan, add the onions, and cook until",
    "The report argues that remote work changes urban economies because",
    "To prove that the square root of 2 is irrational, assume by contradiction that",
    "Analyzing the economic impact of tariffs on consumer electronics indicates that",
    "The central limit theorem states that the arithmetic mean of a large number of independent random variables",
    "Comparing transformer attention complexity with recurrent neural networks reveals that",
    "In quantum mechanics, Heisenberg's uncertainty principle sets a fundamental limit on",
    "The historical consensus regarding the decline of the Western Roman Empire emphasizes",
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
    p.add_argument("--temps", type=float, nargs="+", default=[0.0],
                   help="Gumbel temperatures (0 = greedy)")
    p.add_argument("--pos_decay", type=float, nargs="+", default=[0.0],
                   help="step-1 position decay lambdas")
    p.add_argument("--top_k", type=int, default=50,
                   help="truncate stochastic sampling to the top-k logits")
    p.add_argument("--remask_ratio", type=float, nargs="+", default=[0.0],
                   help="fraction of decoded slots that may be re-masked per step")
    p.add_argument("--sub_canvas_len", type=int, default=None,
                   help="sub-block canvas length for sliding generation (e.g. 16 or 8)")
    p.add_argument("--sub_steps", type=int, default=None,
                   help="steps per sub-block (defaults to proportional steps)")
    p.add_argument("--no_carrier_priority", action="store_true",
                   help="disable carrier-aware margin bonus in sampler")
    p.add_argument("--schedule", default="linear", choices=["linear", "geometric"],
                   help="unmasking schedule quota: linear or geometric")
    p.add_argument("--prefix_sweep", action="store_true",
                   help="TPS vs prefix length for causal and bfg-diff at batch 1 and 4")
    p.add_argument("--no_adj_suppress", action="store_true",
                   help="disable adjacent-token collision suppression")
    p.add_argument("--roofline", action="store_true",
                   help="measure median step latency and gamma_L across active token counts")
    p.add_argument("--offset_probe", action="store_true",
                   help="measure multi-offset lookahead accuracy against causal ground truth")
    p.add_argument("--ablation", action="store_true",
                   help="run comparative ablation table across stages")
    p.add_argument("--enable_branching", action="store_true",
                   help="enable confidence-gated 2-branch micro-tree verification")
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
        args.temps, args.pos_decay = [0.0], [0.0]
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


def adj_dup(seqs):
    """Fraction of neighbouring token pairs that are the same token."""
    hits = pairs = 0
    for s in seqs:
        for a, b in zip(s, s[1:]):
            pairs += 1
            hits += a == b
    return hits / pairs if pairs else 0.0


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
def run_sampler_generation(
    model, tok, prompt_ids, args, mask_id, device, steps, margin_floor, temperature, position_decay,
    remask_ratio=0.0, use_v1=False,
):
    token_ids = []
    stats_total = {
        "forward_passes": 0,
        "active_tokens_evaluated": 0,
        "tokens_verified": 0,
        "accept_histogram": {},
    }
    sync(device)
    start = time.perf_counter()
    for i in range(0, len(prompt_ids), args.batch_size):
        ids, keep = left_pad(prompt_ids[i : i + args.batch_size], tok.pad_token_id, device)
        batch_stats = {}
        canvas = sample_canvas(
            model, ids, canvas_len=args.canvas_len, sub_canvas_len=args.sub_canvas_len,
            steps=steps, sub_steps=args.sub_steps, mask_id=mask_id,
            prefix_keep=keep, margin_floor=margin_floor,
            temperature=temperature, position_decay=position_decay, top_k=args.top_k,
            remask_ratio=remask_ratio, suppress_adjacent=not args.no_adj_suppress,
            carrier_priority=not args.no_carrier_priority,
            schedule=args.schedule,
            enable_branching=getattr(args, "enable_branching", False),
            stats_out=batch_stats,
            use_v1=use_v1,
        )
        token_ids.extend(canvas.tolist())
        stats_total["forward_passes"] += batch_stats.get("forward_passes", 0)
        stats_total["active_tokens_evaluated"] += batch_stats.get("active_tokens_evaluated", 0)
        stats_total["tokens_verified"] += batch_stats.get("tokens_verified", 0)
        for k, v in batch_stats.get("accept_histogram", {}).items():
            stats_total["accept_histogram"][k] = stats_total["accept_histogram"].get(k, 0) + v
    sync(device)
    elapsed = time.perf_counter() - start
    texts = [tok.decode(t, skip_special_tokens=True) for t in token_ids]
    total_generated = sum(len(s) for s in token_ids)
    fp = max(1, stats_total["forward_passes"])
    tv = max(1, stats_total["tokens_verified"])
    stats_total["alpha"] = stats_total["tokens_verified"] / fp
    stats_total["r_kv"] = (stats_total["active_tokens_evaluated"] / tv) - 1.0
    return token_ids, texts, total_generated / max(1e-5, elapsed), stats_total


@torch.inference_mode()
def run_hardware_roofline(model, tok, device, num_warmup=3, num_iters=10):
    P = 128
    prefix = torch.zeros((1, P), dtype=torch.long, device=device)
    kv = prefill_prefix(model, prefix)
    assert kv is not None, "Failed to prefill prefix for roofline"

    n_list = [1, 2, 4, 6, 8, 12, 16]
    latencies = {}

    for N in n_list:
        active_ids = torch.zeros((1, N), dtype=torch.long, device=device)
        pos = torch.arange(P - 1, P - 1 + N, device=device).unsqueeze(0)
        T = (P - 1) + N
        allow = torch.ones(1, 1, N, T, dtype=torch.bool, device=device)
        dtype = model.get_input_embeddings().weight.dtype
        attn = torch.zeros(1, 1, N, T, dtype=dtype, device=device).masked_fill_(
            ~allow, torch.finfo(dtype).min
        )

        for _ in range(num_warmup):
            model(input_ids=active_ids, attention_mask=attn, position_ids=pos, past_key_values=kv, use_cache=True)
            kv.crop(-N)
        sync(device)

        times = []
        for _ in range(num_iters):
            sync(device)
            t0 = time.perf_counter()
            model(input_ids=active_ids, attention_mask=attn, position_ids=pos, past_key_values=kv, use_cache=True)
            sync(device)
            t1 = time.perf_counter()
            kv.crop(-N)
            times.append((t1 - t0) * 1000.0)

        times.sort()
        latencies[N] = times[len(times) // 2]

    t1 = latencies[1]
    results = {}
    print(f"\nHardware Roofline Profile | P={P} | device={device.type}")
    print(f"{'N_active':<10}{'Median (ms)':>14}{'gamma_L(N)':>14}")
    for N in n_list:
        gamma = latencies[N] / max(1e-6, t1)
        results[N] = (latencies[N], gamma)
        print(f"{N:<10}{latencies[N]:>14.2f}{gamma:>13.2f}x")
    return results


@torch.inference_mode()
def run_offset_accuracy_probe(model, tok, prompt_ids, mask_id, device, W=4, tau=0.8):
    exact_matches = {k: 0 for k in range(W + 1)}
    tau_matches = {k: 0 for k in range(W + 1)}
    total_prompts = 0

    eval_prompts = prompt_ids[:min(50, len(prompt_ids))]
    for p_ids in eval_prompts:
        if len(p_ids) < 4:
            continue
        prefix = torch.tensor([p_ids], device=device)
        P = prefix.shape[1]

        causal_tokens = []
        causal_logits_list = []
        curr_input = prefix
        for k in range(W + 1):
            out = model(input_ids=curr_input, use_cache=False)
            last_logit = out.logits[:, -1, :].float()
            tok_top = last_logit.argmax(dim=-1, keepdim=True)
            causal_logits_list.append(last_logit[0])
            causal_tokens.append(int(tok_top[0, 0]))
            curr_input = torch.cat([curr_input, tok_top], dim=1)

        canvas_masks = torch.full((1, W), mask_id, dtype=torch.long, device=device)
        active_ids = torch.cat([prefix[:, -1:], canvas_masks], dim=1)
        kv = prefill_prefix(model, prefix)
        step_logits = canvas_step_with_cache(
            model, active_ids, past_key_values=kv, prefix_len=P, zone_a_len=1
        )
        z_masked = step_logits[0].float()

        for k in range(W + 1):
            pred_tok = int(z_masked[k].argmax())
            gold_tok = causal_tokens[k]
            gold_logits = causal_logits_list[k]

            if pred_tok == gold_tok:
                exact_matches[k] += 1
                tau_matches[k] += 1
            else:
                margin = (gold_logits[gold_tok] - gold_logits[pred_tok]).item()
                if margin <= tau:
                    tau_matches[k] += 1
        total_prompts += 1

    print(f"\nOffset Accuracy Probe | {total_prompts} prompts | W={W} | tau={tau}")
    print(f"{'Offset':<10}{'A_exact (%)':>14}{'A_tau (%)':>14}")
    results = {}
    for k in range(W + 1):
        pct_exact = (exact_matches[k] / max(1, total_prompts)) * 100.0
        pct_tau = (tau_matches[k] / max(1, total_prompts)) * 100.0
        results[k] = (pct_exact, pct_tau)
        print(f"{f'+{k}':<10}{pct_exact:>13.1f}%{pct_tau:>13.1f}%")
    return results


@torch.inference_mode()
def run_ablation_table(model, tok, prompt_ids, mask_id, device, args):
    eff_bs = getattr(args, "batch_size", 1)
    print(f"\nRunning Comparative Ablation Study: BFG-Diff v1 vs v2 (batch_size={eff_bs})...")

    orig_adapter = getattr(model, "bfg_adapter", None)

    # 1. Run probe for Stage 1 (adapter OFF)
    model.bfg_adapter = None
    probe_results_s1 = run_offset_accuracy_probe(model, tok, prompt_ids, mask_id, device, W=4, tau=0.8)
    a1_s1 = probe_results_s1.get(1, (0.0, 0.0))[1]
    a2_s1 = probe_results_s1.get(2, (0.0, 0.0))[1]

    # 2. Run probe for Stage 2 (adapter ON)
    model.bfg_adapter = orig_adapter
    if orig_adapter is not None:
        probe_results_s2 = run_offset_accuracy_probe(model, tok, prompt_ids, mask_id, device, W=4, tau=0.8)
        a1_s2 = probe_results_s2.get(1, (0.0, 0.0))[1]
        a2_s2 = probe_results_s2.get(2, (0.0, 0.0))[1]
    else:
        a1_s2, a2_s2 = a1_s1, a2_s1

    # Ensure Causal and Sampler use the EXACT same effective batch size for fair apples-to-apples evaluation
    causal_tps = getattr(args, "baseline_causal_tps", None)
    if causal_tps is None:
        c_ids, c_texts, causal_tps = run_causal_generation(model, tok, prompt_ids, args, device)
        h4_c = ngram_entropy(c_ids)
        rep4_c = repeat_rate(c_ids)
        adj_c = adj_dup(c_ids)
    else:
        h4_c = getattr(args, "baseline_h4", 0.9)
        rep4_c = getattr(args, "baseline_rep4", 0.0)
        adj_c = getattr(args, "baseline_adj", 0.0)

    rows = []
    rows.append({
        "method": f"Causal Baseline (generate, B={eff_bs})",
        "tps": causal_tps,
        "speedup": 1.00,
        "alpha": 1.00,
        "r_kv": 0.00,
        "a1": f"{100.0:.1f}%",
        "a2": "N/A",
        "h4": h4_c,
        "rep4": rep4_c,
        "adj": adj_c,
    })

    ids_v1, texts_v1, tps_v1, stats_v1 = run_sampler_generation(
        model, tok, prompt_ids, args, mask_id, device, steps=args.steps[0],
        margin_floor=0.0, temperature=0.0, position_decay=0.0, use_v1=True,
    )
    rows.append({
        "method": f"BFG-Diff v1 (Sub-block recompute, cap 2, B={eff_bs})",
        "tps": tps_v1,
        "speedup": tps_v1 / max(1e-5, causal_tps),
        "alpha": stats_v1.get("alpha", 1.00),
        "r_kv": stats_v1.get("r_kv", 4.00),
        "a1": f"{a1_s1:.1f}%",
        "a2": f"{a2_s1:.1f}%",
        "h4": ngram_entropy(ids_v1),
        "rep4": repeat_rate(ids_v1),
        "adj": adj_dup(ids_v1),
    })

    model.bfg_adapter = None
    ids_s1, texts_s1, tps_s1, stats_s1 = run_sampler_generation(
        model, tok, prompt_ids, args, mask_id, device, steps=args.steps[0],
        margin_floor=0.0, temperature=0.0, position_decay=0.0,
    )
    rows.append({
        "method": f"BFG-Diff v2 - Stage 1 (Staircase Zone A/B + In-Place KV Slice, B={eff_bs})",
        "tps": tps_s1,
        "speedup": tps_s1 / max(1e-5, causal_tps),
        "alpha": stats_s1.get("alpha", 1.0),
        "r_kv": stats_s1.get("r_kv", 1.2),
        "a1": f"{a1_s1:.1f}%",
        "a2": f"{a2_s1:.1f}%",
        "h4": ngram_entropy(ids_s1),
        "rep4": repeat_rate(ids_s1),
        "adj": adj_dup(ids_s1),
    })
    model.bfg_adapter = orig_adapter

    ids_s2, texts_s2, tps_s2, stats_s2 = run_sampler_generation(
        model, tok, prompt_ids, args, mask_id, device, steps=args.steps[0],
        margin_floor=0.0, temperature=0.0, position_decay=0.0,
    )
    rows.append({
        "method": f"BFG-Diff v2 - Stage 2 (+ SlotEmb + Latent Recycling + Self-Distill, B={eff_bs})",
        "tps": tps_s2,
        "speedup": tps_s2 / max(1e-5, causal_tps),
        "alpha": stats_s2.get("alpha", 1.0),
        "r_kv": stats_s2.get("r_kv", 1.1),
        "a1": f"{a1_s2:.1f}%",
        "a2": f"{a2_s2:.1f}%",
        "h4": ngram_entropy(ids_s2),
        "rep4": repeat_rate(ids_s2),
        "adj": adj_dup(ids_s2),
    })

    saved_branching = getattr(args, "enable_branching", False)
    args.enable_branching = True
    ids_s3, texts_s3, tps_s3, stats_s3 = run_sampler_generation(
        model, tok, prompt_ids, args, mask_id, device, steps=args.steps[0],
        margin_floor=0.0, temperature=0.0, position_decay=0.0,
    )
    args.enable_branching = saved_branching
    rows.append({
        "method": f"BFG-Diff v2 - Stage 3 (+ 2-Branch Micro-Tree Verification, B={eff_bs})",
        "tps": tps_s3,
        "speedup": tps_s3 / max(1e-5, causal_tps),
        "alpha": stats_s3.get("alpha", 1.0),
        "r_kv": stats_s3.get("r_kv", 1.1),
        "a1": f"{a1_s2:.1f}%",
        "a2": f"{a2_s2:.1f}%",
        "h4": ngram_entropy(ids_s3),
        "rep4": repeat_rate(ids_s3),
        "adj": adj_dup(ids_s3),
    })

    print(f"\n### Comparative Ablation Study: BFG-Diff v1 vs v2 (batch_size={eff_bs})")
    print("| Method | TPS | Speedup (x) | Alpha (tok/pass) | R_kv | A_+1 (%) | A_+2 (%) | H4 | rep4 (%) | adj_dup (%) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['method']} | {r['tps']:.1f} | {r['speedup']:.2f}x | {r['alpha']:.2f} | "
            f"{r['r_kv']:.2f} | {r['a1']} | {r['a2']} | {r['h4']:.3f} | {r['rep4']:.1%} | {r['adj']:.1%} |"
        )


@torch.inference_mode()
def run_causal_generation(model, tok, prompt_ids, args, device):
    # Run a 1-prompt unmeasured warmup pass before timing so kernel/allocator init does not bias baseline
    if len(prompt_ids) > 0:
        w_ids, w_keep = left_pad(prompt_ids[:1], tok.pad_token_id, device)
        model.generate(
            input_ids=w_ids,
            attention_mask=w_keep.long(),
            max_new_tokens=min(16, args.canvas_len),
            do_sample=False,
            use_cache=True,
            pad_token_id=tok.pad_token_id,
        )
        sync(device)

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


@torch.inference_mode()
def run_prefix_sweep(model, tok, args, mask_id, device):
    """TPS vs prefix length: causal generate() against cached diffusion steps."""
    text_ids = tokenize(tok, " ".join(REFERENCE_TEXTS))
    assert len(text_ids) >= 512, f"need 512 tokens of reference text, have {len(text_ids)}"
    prompts = args.num_prompts
    warm = [text_ids[:32]] * min(2, prompts)
    run_sampler_generation(model, tok, warm, args, mask_id, device, 2, 0.0, 0.0, 0.0)
    run_causal_generation(model, tok, warm, args, device)

    print(f"\nPrefix sweep | C={args.canvas_len} | {prompts} prompts | tokens/s")
    print("| P | batch | causal | k=4 | k=4 x causal | k=8 | k=8 x causal |")
    print("|---|---|---|---|---|---|---|")
    for plen in (32, 128, 256, 512):
        prompt_ids = [text_ids[:plen]] * prompts
        for bs in (1, 4):
            args.batch_size = bs
            _, _, causal = run_causal_generation(model, tok, prompt_ids, args, device)
            _, _, k4 = run_sampler_generation(
                model, tok, prompt_ids, args, mask_id, device, 4, 0.0, 0.0, 0.0
            )
            _, _, k8 = run_sampler_generation(
                model, tok, prompt_ids, args, mask_id, device, 8, 0.0, 0.0, 0.0
            )
            print(
                f"| {plen} | {bs} | {causal:,.1f} | {k4:,.1f} | {k4 / causal:.2f}x "
                f"| {k8:,.1f} | {k8 / causal:.2f}x |"
            )


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
    # Determine if running with an Instruct model to use ChatML prompts if applicable
    is_instruct = "instruct" in args.model.lower() or (args.checkpoint and "instruct" in args.checkpoint.lower())
    if is_instruct:
        chatml_prompts = [
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
            for p in GEN_PROMPTS
        ]
        prompts = [chatml_prompts[i % len(chatml_prompts)] for i in range(args.num_prompts)]
    else:
        prompts = [GEN_PROMPTS[i % len(GEN_PROMPTS)] for i in range(args.num_prompts)]
    prompt_ids = [tokenize(tok, p) for p in prompts]

    # Constraint 1: Measure Causal Baseline on RAW unadapted base model BEFORE applying LoRA
    c_ids_base, c_texts_base, causal_tps_base = run_causal_generation(model, tok, prompt_ids, args, device)
    args.baseline_causal_tps = causal_tps_base
    args.baseline_h4 = ngram_entropy(c_ids_base)
    args.baseline_rep4 = repeat_rate(c_ids_base)
    args.baseline_adj = adj_dup(c_ids_base)

    if Path(args.checkpoint).exists():
        ck = torch.load(args.checkpoint, map_location="cpu")
        if ck.get("lora") is not None:
            from peft import set_peft_model_state_dict

            cfg = ck.get("lora_config", {})
            model = apply_lora(
                model, cfg.get("r", 16), cfg.get("alpha", 32), cfg.get("dropout", 0.05),
                cfg.get("targets", LORA_TARGETS),
            )
            set_peft_model_state_dict(model, ck["lora"])
            # Merge LoRA back into base model
            model = model.merge_and_unload()
            if "mask_emb" in ck:
                with torch.no_grad():
                    model.get_input_embeddings().weight[mask_id] = ck["mask_emb"].to(device)
        else:
            model.load_state_dict(ck["model"], strict=False)

        if "bfg_adapter" in ck:
            from model import setup_bfg_adapter
            if getattr(model, "bfg_adapter", None) is None:
                setup_bfg_adapter(model)
            model.bfg_adapter.load_state_dict(ck["bfg_adapter"], strict=False)
        mask_id = ck.get("mask_id", mask_id)
    else:
        print(f"warning: checkpoint {args.checkpoint!r} not found; benchmarking base model")
    model.eval()

    if args.prefix_sweep:
        run_prefix_sweep(model, tok, args, mask_id, device)
        return

    if args.ablation:
        run_ablation_table(model, tok, prompt_ids, mask_id, device, args)
        return

    if args.roofline or not args.smoke_test:
        run_hardware_roofline(
            model, tok, device,
            num_warmup=2 if args.smoke_test else 3,
            num_iters=3 if args.smoke_test else 10,
        )

    if args.offset_probe or not args.smoke_test:
        run_offset_accuracy_probe(model, tok, prompt_ids, mask_id, device, W=4, tau=0.8)

    print(f"\nReconstruction | {args.num_prompts} prefixes | P={args.prefix_len} C={args.canvas_len}")
    print(f"{'method':<18}{'recovery':>10}{'TPS':>12}")
    for steps in args.steps:
        rec, tps = run_reconstruction(model, tok, args, mask_id, device, steps)
        print(f"{f'bfg-diff k={steps}':<18}{rec:>9.1%}{tps:>12,.1f}")

    print(f"\nGeneration | {args.num_prompts} prompts | C={args.canvas_len} | top_k={args.top_k}")
    print(
        f"{'method':<34}{'H4':>7}{'rep4':>7}{'rep2':>7}{'adj':>7}"
        f"{'Dist1':>7}{'Dist2':>7}{'alpha':>7}{'R_kv':>7}{'speedup':>9}{'TPS':>10}"
    )
    outputs = {}
    causal_tps = args.baseline_causal_tps
    outputs["causal generate"] = c_texts_base
    print(
        f"{'causal generate (unadapted)':<34}{args.baseline_h4:>7.3f}{args.baseline_rep4:>7.1%}"
        f"{repeat_rate(c_ids_base, 2):>7.1%}{args.baseline_adj:>7.1%}"
        f"{distinct(c_ids_base, 1):>7.3f}{distinct(c_ids_base, 2):>7.3f}"
        f"{1.00:>7.2f}{0.00:>7.2f}{'1.00x':>9}{causal_tps:>10,.1f}"
    )

    for floor in args.floors:
        for temp in args.temps:
            for decay in args.pos_decay:
                for remask in args.remask_ratio:
                    for steps in args.steps:
                        ids, texts, tps, stats = run_sampler_generation(
                            model, tok, prompt_ids, args, mask_id, device, steps,
                            floor, temp, decay, remask,
                        )
                        label = f"k={steps} T={temp:g} d={decay:g} f={floor:g} r={remask:g}"
                        name = f"bfg-diff {label}"
                        outputs[name] = texts
                        alpha = stats.get("alpha", 1.0)
                        r_kv = stats.get("r_kv", 0.0)
                        speedup = tps / max(1e-5, causal_tps)
                        print(
                            f"{name:<34}{ngram_entropy(ids):>7.3f}{repeat_rate(ids):>7.1%}"
                            f"{repeat_rate(ids, 2):>7.1%}{adj_dup(ids):>7.1%}"
                            f"{distinct(ids, 1):>7.3f}{distinct(ids, 2):>7.3f}"
                            f"{alpha:>7.2f}{r_kv:>7.2f}{f'{speedup:.2f}x':>9}{tps:>10,.1f}"
                        )

    if args.show:
        print("\nSide-by-side completions")
        for i in range(min(args.show, len(prompts))):
            print(f"\n[{i}] {prompts[i]}")
            for name, texts in outputs.items():
                print(f"  {name}: {texts[i]}")


if __name__ == "__main__":
    main()
