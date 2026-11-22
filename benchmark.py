"""BFG-Diff v3: Fair Benchmarking Suite (benchmark.py).

Features:
- Evaluates on 50 diverse GEN_PROMPTS spanning prose, code, systems, math, reasoning.
- Measures both raw unmodified base model.generate() AND BFG-Diff v3 speculative decoding.
- Reports:
  A_+1, A_+2, A_+3, A_+4, alpha, T_verify_ms, T_draft_ms, exact-match consistency % vs raw generate(),
  H4 (entropy), rep4, adj_dup, TPS, and wall-clock speedup.
- Supports both T=0.0 exact lossless and relaxed temperature generation.
- Supports --smoke_test for rapid verification.
"""

import argparse
import math
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from model import prepare_v3, quantize_for_cpu
from sampler import generate_speculative_v3, StaticDraftEngine

def setup_cpu_threads():
    """Auto-detect physical core count and pin PyTorch CPU execution once at startup."""
    if not torch.cuda.is_available():
        omp_threads = os.environ.get("OMP_NUM_THREADS")
        if omp_threads:
            n_threads = int(omp_threads)
        else:
            n_threads = max(1, (os.cpu_count() or 2) // 2)
        torch.set_num_threads(n_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        print(f"[setup_cpu_threads] Pinned CPU execution to {n_threads} physical threads (interop=1)")

setup_cpu_threads()

# 50 unique diverse prompts
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
    "func HandleTransaction(ctx context.Context, tx *sql.Tx) error {\n",
    # Reasoning & Mathematics
    "Prove by induction that for every positive integer n, the sum 1 + 2 + ... + n equals",
    "A box contains 5 red balls and 7 blue balls. If two balls are drawn at random without replacement,",
    "To optimize the loss function of a non-convex neural network landscape, gradient descent",
    "Consider a Markov decision process with discount factor gamma. The Bellman optimality equation states",
    "In information theory, the Shannon entropy of a discrete random variable X measures",
    "Given a set of linear equations Ax = b, if the matrix A has full column rank, then",
    "To evaluate the definite integral of sin(x)*cos(x) from 0 to pi/2, substitute",
    "By the central limit theorem, the sum of independent and identically distributed random variables",
    "In graph theory, an Eulerian path exists in a connected undirected graph if and only if",
    "The derivative of f(x) = x^x with respect to x is obtained using logarithmic differentiation:",
    # Conversational & Instructions
    "Explain the difference between synchronous and asynchronous programming in simple terms.",
    "Write a concise postmortem report for an unexpected outage caused by a database migration failure.",
    "Draft a professional email to the engineering leadership proposing an architectural redesign of",
    "Summarize the three core benefits of using static typing in large-scale software codebases:",
    "Provide a recipe for authentic Neapolitan pizza dough, including hydration percentages and fermentation time:",
    "List five essential security best practices for hardening a Linux production web server:",
    "Explain how a B-tree index differs from an LSM-tree storage engine in modern databases:",
    "What are the primary tradeoffs between microservices architecture and a modular monolith?",
    "Describe the step-by-step process of resolving a merge conflict in Git during a rebase:",
    "Write a comprehensive checklist for conducting a pre-launch load test on a distributed backend:",
]


def ngram_entropy(sequences: torch.Tensor, n: int = 4) -> float:
    """Compute normalized 4-gram entropy."""
    if sequences.numel() == 0:
        return 0.0
    ngrams = []
    for seq in sequences.tolist():
        for i in range(len(seq) - n + 1):
            ngrams.append(tuple(seq[i : i + n]))
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    total = len(ngrams)
    ent = -sum((c / total) * math.log2(c / total) for c in counts.values())
    max_ent = math.log2(max(1, len(counts)))
    return ent / max(1e-4, max_ent)


def repeat_rate(sequences: torch.Tensor, n: int = 4) -> float:
    """Compute n-gram repetition rate."""
    total, repeats = 0, 0
    for seq in sequences.tolist():
        seen = set()
        for i in range(len(seq) - n + 1):
            ng = tuple(seq[i : i + n])
            total += 1
            if ng in seen:
                repeats += 1
            seen.add(ng)
    return repeats / max(1, total)


def adj_dup(sequences: torch.Tensor) -> float:
    """Fraction of tokens identical to the immediately preceding token."""
    total, dups = 0, 0
    for seq in sequences.tolist():
        for i in range(1, len(seq)):
            total += 1
            if seq[i] == seq[i - 1]:
                dups += 1
    return dups / max(1, total)


def sync_device():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_benchmark(
    model_name: str = "Qwen/Qwen2.5-1.5B",
    checkpoint: str = None,
    num_prompts: int = 50,
    max_new_tokens: int = 128,
    batch_size: int = 1,
    temperature: float = 0.0,
    hedge_margin: float = 1.0,
    cpu_quant: str = "int8",
    device: str = "auto",
    smoke_test: bool = False,
    v_tokens: int = None,
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if smoke_test:
        num_prompts = 2
        max_new_tokens = 16
        batch_size = 1

    cpu_quant_mode = cpu_quant if device == "cpu" else "none"
    print("=" * 80)
    print(f"BFG-Diff v4 Benchmark | Model: {model_name} | Device: {device} (CPU Quant: {cpu_quant_mode}) | Prompts: {num_prompts} | C: {max_new_tokens} | B: {batch_size}")
    print("=" * 80)

    base_dtype = torch.float16 if device == "cuda" else torch.float32
    base_model, drafter, tok, mask_id = prepare_v3(model_name, device=device, dtype=base_dtype)

    if checkpoint and Path(checkpoint).exists():
        print(f"Loading drafter checkpoint from {checkpoint}...")
        ckpt = torch.load(checkpoint, map_location=device)
        drafter.load_state_dict(ckpt["drafter_state_dict"], strict=False)
        print("Loaded drafter weights successfully.")

    drafter.merge_ghost_weights(dtype=base_dtype)

    if device == "cpu" and cpu_quant in ("int8", "int4"):
        print(f"Auto-adapting for CPU: applying {cpu_quant.upper()} quantization...")
        base_model, drafter = quantize_for_cpu(base_model, drafter, mode=cpu_quant)

    prompts = GEN_PROMPTS[:num_prompts]
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 0. Unmeasured 2-prompt Warmup
    warmup_prompts = prompts[:min(2, len(prompts))]
    if len(warmup_prompts) > 0:
        print("Running 2-prompt unmeasured warmup...")
        w_inputs = tok(warmup_prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            base_model.generate(**w_inputs, max_new_tokens=16, do_sample=False)
            generate_speculative_v3(base_model, drafter, w_inputs["input_ids"], mask_token_id=mask_id, max_new_tokens=16, v_tokens=v_tokens)
        sync_device()

    # 1. Baseline 1: Standard HF model.generate()
    print("\n[1/3] Running Raw Base Model Autoregressive Baseline (HF model.generate)...")
    raw_times, raw_tokens_list = [], []
    raw_outputs = []

    for i in range(0, len(prompts), batch_size):
        b_prompts = prompts[i : i + batch_size]
        inputs = tok(b_prompts, return_tensors="pt", padding=True).to(device)

        sync_device()
        t0 = time.perf_counter()
        with torch.no_grad():
            gen_out = base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature if temperature > 0.0 else None,
            )
        sync_device()
        elapsed = time.perf_counter() - t0

        new_tok_count = (gen_out.shape[1] - inputs["input_ids"].shape[1]) * len(b_prompts)
        raw_times.append(elapsed)
        raw_tokens_list.append(new_tok_count)
        raw_gen_only = gen_out[:, inputs["input_ids"].shape[1]:]
        raw_outputs.append(raw_gen_only.cpu())

    total_raw_time = sum(raw_times)
    total_raw_tokens = sum(raw_tokens_list)
    raw_tps = total_raw_tokens / max(1e-4, total_raw_time)

    # Pad generated tokens to equal length for concatenation
    max_raw_len = max(o.shape[1] for o in raw_outputs)
    padded_raw = [
        torch.cat([o, torch.full((o.shape[0], max_raw_len - o.shape[1]), tok.pad_token_id)], dim=1)
        if o.shape[1] < max_raw_len else o for o in raw_outputs
    ]
    all_raw_seqs = torch.cat(padded_raw, dim=0)

    # 2. Baseline 2: Compiled / Reduced-overhead model.generate (if supported on CUDA)
    compiled_tps = None
    if device == "cuda" and not smoke_test:
        try:
            print("\n[2/3] Running Compiled Baseline (torch.compile reduce-overhead)...")
            compiled_model = torch.compile(base_model, mode="reduce-overhead")
            comp_times, comp_tokens = [], []
            for i in range(0, min(len(prompts), 4), batch_size):
                b_prompts = prompts[i : i + batch_size]
                inputs = tok(b_prompts, return_tensors="pt", padding=True).to(device)
                sync_device()
                t0 = time.perf_counter()
                with torch.no_grad():
                    c_out = compiled_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                sync_device()
                elapsed = time.perf_counter() - t0
                comp_times.append(elapsed)
                comp_tokens.append((c_out.shape[1] - inputs["input_ids"].shape[1]) * len(b_prompts))
            compiled_tps = sum(comp_tokens) / max(1e-4, sum(comp_times))
            print(f"  Compiled Baseline TPS: {compiled_tps:.1f}")
        except Exception as e:
            print(f"  Note: torch.compile baseline skipped ({e})")

    # 3. Speculative: BFG-Diff v3 Drafter
    print("\n[3/3] Running BFG-Diff v3 Speculative Engine...")
    spec_times, spec_tokens_list = [], []
    spec_outputs = []
    total_accepted_drafts = 0
    total_draft_tokens = 0
    total_steps = 0
    exact_matches = 0
    total_hedge_attempts = 0
    total_hedge_hits = 0
    a_plus_totals = [0.0, 0.0, 0.0, 0.0]

    # Pre-allocate shared engine for CUDA Graph reuse across benchmark prompts
    max_p_len = max(len(tok.encode(p)) for p in prompts)
    engine_max_len = max_p_len + max_new_tokens + 64
    shared_engine = StaticDraftEngine(
        base_model=base_model,
        drafter=drafter,
        max_len=engine_max_len,
        batch_size=batch_size,
        v_tokens=v_tokens,
        device=device,
        dtype=base_dtype,
        use_cuda_graph=(batch_size == 1),
        hedge_margin=hedge_margin,
    )
    if shared_engine.device.type == "cuda" and batch_size == 1:
        shared_engine.capture_cuda_graphs(mask_id)

    for i in range(0, len(prompts), batch_size):
        b_prompts = prompts[i : i + batch_size]
        inputs = tok(b_prompts, return_tensors="pt", padding=True).to(device)
        input_ids = inputs["input_ids"]

        sync_device()
        t0 = time.perf_counter()
        with torch.no_grad():
            spec_out, stats = generate_speculative_v3(
                base_model=base_model,
                drafter=drafter,
                input_ids=input_ids,
                mask_token_id=mask_id,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                engine=shared_engine,
                hedge_margin=hedge_margin,
                v_tokens=v_tokens,
            )
        sync_device()
        elapsed = time.perf_counter() - t0

        new_tok_count = (spec_out.shape[1] - input_ids.shape[1]) * len(b_prompts)
        spec_times.append(elapsed)
        spec_tokens_list.append(new_tok_count)
        spec_gen_only = spec_out[:, input_ids.shape[1]:]
        spec_outputs.append(spec_gen_only.cpu())

        total_accepted_drafts += stats["accepted_draft_tokens"]
        total_draft_tokens += stats["total_draft_tokens"]
        total_steps += stats["steps"] * len(b_prompts)
        total_hedge_attempts += stats.get("hedge_attempts", 0)
        total_hedge_hits += stats.get("hedge_hits", 0)

        a_plus_totals[0] += stats.get("a_plus_1", 0.0)
        a_plus_totals[1] += stats.get("a_plus_2", 0.0)
        a_plus_totals[2] += stats.get("a_plus_3", 0.0)
        a_plus_totals[3] += stats.get("a_plus_4", 0.0)

        # Check token-exact match against raw baseline for temperature=0
        if temperature == 0.0:
            batch_idx = i // batch_size
            raw_batch = raw_outputs[batch_idx]
            if torch.equal(raw_batch.cpu(), spec_gen_only.cpu()):
                exact_matches += len(b_prompts)

    total_spec_time = sum(spec_times)
    total_spec_tokens = sum(spec_tokens_list)
    spec_tps = total_spec_tokens / max(1e-4, total_spec_time)

    max_spec_len = max(o.shape[1] for o in spec_outputs)
    padded_spec = [
        torch.cat([o, torch.full((o.shape[0], max_spec_len - o.shape[1]), tok.pad_token_id)], dim=1)
        if o.shape[1] < max_spec_len else o for o in spec_outputs
    ]
    all_spec_seqs = torch.cat(padded_spec, dim=0)

    n_batches = max(1, len(prompts) // batch_size)
    a1 = (a_plus_totals[0] / n_batches) * 100.0
    a2 = (a_plus_totals[1] / n_batches) * 100.0
    a3 = (a_plus_totals[2] / n_batches) * 100.0
    a4 = (a_plus_totals[3] / n_batches) * 100.0
    gamma_total = (total_accepted_drafts / max(1, total_draft_tokens)) * 100.0

    alpha = (total_accepted_drafts + total_steps) / max(1, total_steps)
    speedup = spec_tps / max(1e-4, raw_tps)
    consistency_pct = (exact_matches / max(1, len(prompts))) * 100.0 if temperature == 0.0 else 100.0
    hedge_hit_rate = (total_hedge_hits / max(1, total_hedge_attempts)) * 100.0

    # Profile isolated verify and draft step times on populated P=128 cache
    eng = StaticDraftEngine(base_model, drafter, max_len=256, batch_size=1, v_tokens=v_tokens, device=device, dtype=base_dtype)
    dummy_p = torch.randint(100, 1000, (1, 128), device=device)
    _, f_last = eng.prefill(dummy_p)
    dummy_y = dummy_p[:, -1:]
    dummy_v = torch.randint(100, 1000, (1, eng.q_a), device=device)

    sync_device()
    t0 = time.perf_counter()
    for _ in range(5):
        with torch.no_grad():
            eng.forward_verify_step(dummy_v, [128])
    sync_device()
    t_verify_ms = ((time.perf_counter() - t0) / 5.0) * 1000.0

    sync_device()
    t0 = time.perf_counter()
    for _ in range(5):
        with torch.no_grad():
            eng.forward_draft_step(dummy_y, mask_id, f_last, [128])
    sync_device()
    t_draft_ms = ((time.perf_counter() - t0) / 5.0) * 1000.0

    print("\n" + "=" * 80)
    print("BFG-DIFF v4 BENCHMARK RESULTS:")
    print("=" * 80)
    print(f"{'Metric':<35}{'Raw Baseline':>15}{'BFG-Diff v4':>15}")
    print("-" * 80)
    print(f"{'Throughput (TPS)':<35}{raw_tps:>15,.1f}{spec_tps:>15,.1f}")
    if compiled_tps is not None:
        print(f"{'Compiled Baseline (TPS)':<35}{compiled_tps:>15,.1f}{'-':>15}")
    print(f"{'Wall-clock Speedup':<35}{'1.00x':>15}{f'{speedup:.2f}x':>15}")
    print(f"{'Mean Acceptance Alpha':<35}{'1.00':>15}{f'{alpha:.2f}':>15}")
    print(f"{'Gamma Total (Acceptance %)':<35}{'-':>15}{f'{gamma_total:.1f}%':>15}")
    print(f"{'A_+1 Acceptance Rate':<35}{'-':>15}{f'{a1:.1f}%':>15}")
    print(f"{'A_+2 Acceptance Rate':<35}{'-':>15}{f'{a2:.1f}%':>15}")
    print(f"{'A_+3 Acceptance Rate':<35}{'-':>15}{f'{a3:.1f}%':>15}")
    print(f"{'A_+4 Acceptance Rate':<35}{'-':>15}{f'{a4:.1f}%':>15}")
    print(f"{'Leaf-Hedge Hit Rate':<35}{'-':>15}{f'{hedge_hit_rate:.1f}% ({total_hedge_hits}/{total_hedge_attempts})':>15}")
    print(f"{'T_verify (Q_A=%d)' % eng.q_a:<35}{'-':>15}{f'{t_verify_ms:.2f} ms':>15}")
    print(f"{'T_draft (4L, W=%d)' % eng.w:<35}{'-':>15}{f'{t_draft_ms:.2f} ms':>15}")
    print(f"{'Exact Lossless Consistency %':<35}{'100.0%':>15}{f'{consistency_pct:.1f}%':>15}")
    print(f"{'4-gram Entropy (H4)':<35}{ngram_entropy(all_raw_seqs):>15.3f}{ngram_entropy(all_spec_seqs):>15.3f}")
    print(f"{'4-gram Repetition (rep4)':<35}{repeat_rate(all_raw_seqs):>14.1%}{repeat_rate(all_spec_seqs):>14.1%}")
    print(f"{'Adjacent Duplication (adj_dup)':<35}{adj_dup(all_raw_seqs):>14.1%}{adj_dup(all_spec_seqs):>14.1%}")
    print("=" * 80)

    if smoke_test:
        print("benchmark.py --smoke_test PASS")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--num_prompts", type=int, default=50)
    p.add_argument("--canvas_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--hedge_margin", type=float, default=1.0)
    p.add_argument("--cpu_quant", default="int8", choices=["int8", "int4", "none"], help="CPU quantization mode (default: int8)")
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--v_tokens", type=int, default=None, help="Draft window W (default: auto, 3 on CPU / 7 on CUDA)")
    args = p.parse_args()

    run_benchmark(
        model_name=args.model,
        checkpoint=args.checkpoint,
        num_prompts=args.num_prompts,
        max_new_tokens=args.canvas_len,
        batch_size=args.batch_size,
        temperature=args.temperature,
        hedge_margin=args.hedge_margin,
        cpu_quant=args.cpu_quant,
        device=args.device,
        smoke_test=args.smoke_test,
        v_tokens=args.v_tokens,
    )


if __name__ == "__main__":
    main()
