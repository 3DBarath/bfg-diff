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
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import prepare_v3, quantize_for_cpu
from sampler import generate_speculative_v3, StaticDraftEngine, StaticKVCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv


class GraphedCausalRunner:
    """Base-model-only bucketed CUDA graph causal runner (Q=1).

    Executes isolated base model autoregressive decoding using the exact same
    StaticKVCache, power-of-2 buckets, and CUDA graph machinery as StaticDraftEngine,
    without executing any drafter operations. Used as the fair graphed Q=1 baseline.
    """

    def __init__(
        self,
        base_model: nn.Module,
        max_len: int = 2048,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        buckets: Optional[List[int]] = None,
    ):
        self.base_model = base_model
        self.max_len = max_len
        if device is None:
            device = next(base_model.parameters()).device
        if dtype is None:
            dtype = next(base_model.parameters()).dtype
        self.device = torch.device(device)
        self.dtype = dtype

        config = base_model.config
        self.total_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)
        self.dim = config.hidden_size

        cand_buckets = buckets if buckets is not None else [256, 512, 1024, 2048]
        self.buckets = sorted(set(b for b in cand_buckets if b < max_len) | {max_len})

        self.cache = StaticKVCache(
            num_layers=self.total_layers,
            batch_size=1,
            num_kv_heads=self.num_kv_heads,
            max_len=max_len,
            head_dim=self.head_dim,
            dtype=dtype,
            device=device,
        )

        self.static_token = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.static_pos = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.static_mask = torch.full((1, 1, 1, max_len), -1e4, dtype=dtype, device=device)
        self.static_logits = torch.zeros(1, 1, config.vocab_size, dtype=dtype, device=device)

        # Check enable_gqa
        self.supports_enable_gqa = False
        if self.num_heads != self.num_kv_heads:
            try:
                with torch.no_grad():
                    q_len, kv_len = 8, 16
                    probe_q = torch.linspace(-1, 1, self.num_heads * q_len * self.head_dim, dtype=self.dtype, device=self.device)
                    probe_q = probe_q.view(1, self.num_heads, q_len, self.head_dim)
                    probe_k = torch.linspace(-0.5, 0.5, self.num_kv_heads * kv_len * self.head_dim, dtype=self.dtype, device=self.device)
                    probe_k = probe_k.view(1, self.num_kv_heads, kv_len, self.head_dim)
                    probe_v = torch.linspace(0.1, 0.9, self.num_kv_heads * kv_len * self.head_dim, dtype=self.dtype, device=self.device)
                    probe_v = probe_v.view(1, self.num_kv_heads, kv_len, self.head_dim)

                    probe_mask = torch.zeros(1, 1, q_len, kv_len, dtype=self.dtype, device=self.device)
                    probe_mask[:, :, :, kv_len // 2:] = -1e4

                    k_rep = repeat_kv(probe_k, self.num_heads // self.num_kv_heads)
                    v_rep = repeat_kv(probe_v, self.num_heads // self.num_kv_heads)
                    out_ref = F.scaled_dot_product_attention(probe_q, k_rep, v_rep, attn_mask=probe_mask)

                    out_gqa = F.scaled_dot_product_attention(probe_q, probe_k, probe_v, attn_mask=probe_mask, enable_gqa=True)
                    diff = (out_ref - out_gqa).abs().max().item()
                    tol = 1e-2 if self.dtype in (torch.float16, torch.bfloat16) else 1e-4
                    if diff <= tol:
                        self.supports_enable_gqa = True
            except Exception:
                self.supports_enable_gqa = False

        self.cuda_graphs = {}
        if self.device.type == "cuda":
            self._capture_cuda_graphs()

    def get_bucket(self, pos: int) -> int:
        for b in self.buckets:
            if b >= pos + 1:
                return b
        if pos + 1 > self.max_len:
            raise ValueError(f"Position {pos + 1} exceeds max_len {self.max_len}")
        return self.buckets[-1]

    def _step_kernel(self, S_b: int):
        h = self.base_model.model.embed_tokens(self.static_token)
        cos, sin = self.base_model.model.rotary_emb(h, self.static_pos)
        mask_view = self.static_mask[:, :, :, :S_b]

        for l_idx, layer in enumerate(self.base_model.model.layers):
            norm_h = layer.input_layernorm(h)
            q = layer.self_attn.q_proj(norm_h).view(1, 1, self.num_heads, self.head_dim).transpose(1, 2)
            k = layer.self_attn.k_proj(norm_h).view(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = layer.self_attn.v_proj(norm_h).view(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

            idx = self.static_pos.view(1, 1, 1, 1).expand(1, self.num_kv_heads, 1, self.head_dim)
            self.cache.k[l_idx].scatter_(2, idx, k)
            self.cache.v[l_idx].scatter_(2, idx, v)

            k_active = self.cache.k[l_idx][:, :, :S_b, :]
            v_active = self.cache.v[l_idx][:, :, :S_b, :]

            if self.num_heads != self.num_kv_heads:
                if self.supports_enable_gqa:
                    attn_out = F.scaled_dot_product_attention(q, k_active, v_active, attn_mask=mask_view, enable_gqa=True)
                else:
                    k_attn = repeat_kv(k_active, self.num_heads // self.num_kv_heads)
                    v_attn = repeat_kv(v_active, self.num_heads // self.num_kv_heads)
                    attn_out = F.scaled_dot_product_attention(q, k_attn, v_attn, attn_mask=mask_view)
            else:
                attn_out = F.scaled_dot_product_attention(q, k_active, v_active, attn_mask=mask_view)

            attn_out = attn_out.transpose(1, 2).reshape(1, 1, self.dim)
            h = h + layer.self_attn.o_proj(attn_out)
            norm_post = layer.post_attention_layernorm(h)
            h = h + layer.mlp(norm_post)

        h_norm = self.base_model.model.norm(h)
        logits = self.base_model.lm_head(h_norm)
        self.static_logits.copy_(logits)

    def _capture_cuda_graphs(self):
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for S_b in self.buckets:
                    self.static_token.zero_()
                    self.static_pos.zero_()
                    self.static_mask[:, :, :, :S_b].fill_(-1e4)
                    self.static_mask[0, 0, 0, 0] = 0.0

                    for _ in range(3):
                        self._step_kernel(S_b)

                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        self._step_kernel(S_b)
                    self.cuda_graphs[S_b] = g
            torch.cuda.current_stream().wait_stream(s)
            print(f"[GraphedCausalRunner] Captured {len(self.cuda_graphs)} CUDA graphs.")
        except Exception as e:
            print(f"[GraphedCausalRunner] CUDA Graph capture skipped/failed: {e}")
            self.cuda_graphs = {}

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.cache.reset()
        B, P = input_ids.shape
        pos_ids = torch.arange(P, device=self.device).unsqueeze(0)
        mask = torch.triu(torch.full((1, 1, P, P), -1e4, dtype=self.dtype, device=self.device), diagonal=1)

        h = self.base_model.model.embed_tokens(input_ids)
        cos, sin = self.base_model.model.rotary_emb(h, pos_ids)

        for l_idx, layer in enumerate(self.base_model.model.layers):
            norm_h = layer.input_layernorm(h)
            q = layer.self_attn.q_proj(norm_h).view(B, P, self.num_heads, self.head_dim).transpose(1, 2)
            k = layer.self_attn.k_proj(norm_h).view(B, P, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = layer.self_attn.v_proj(norm_h).view(B, P, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

            self.cache.k[l_idx][0, :, :P] = k[0]
            self.cache.v[l_idx][0, :, :P] = v[0]

            if self.num_heads != self.num_kv_heads:
                if self.supports_enable_gqa:
                    attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
                else:
                    k_rep = repeat_kv(k, self.num_heads // self.num_kv_heads)
                    v_rep = repeat_kv(v, self.num_heads // self.num_kv_heads)
                    attn_out = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=mask)
            else:
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

            attn_out = attn_out.transpose(1, 2).reshape(B, P, self.dim)
            h = h + layer.self_attn.o_proj(attn_out)
            norm_post = layer.post_attention_layernorm(h)
            h = h + layer.mlp(norm_post)

        h_last = self.base_model.model.norm(h[:, -1:])
        logits = self.base_model.lm_head(h_last)
        return logits

    def step(self, token: torch.Tensor, cur_pos: int) -> torch.Tensor:
        S_b = self.get_bucket(cur_pos)
        if cur_pos in self.cuda_graphs or S_b in self.cuda_graphs:
            g = self.cuda_graphs.get(S_b)
            if g is not None:
                self.static_token.copy_(token)
                self.static_pos.fill_(cur_pos)
                mask_view = self.static_mask[:, :, :, :S_b]
                mask_view.fill_(-1e4)
                mask_view[0, 0, 0, :cur_pos + 1] = 0.0
                g.replay()
                return self.static_logits

        # Fallback eager step
        self.static_token.copy_(token)
        self.static_pos.fill_(cur_pos)
        self.static_mask[:, :, :, :cur_pos + 1] = 0.0
        self._step_kernel(cur_pos + 1)
        return self.static_logits

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
    draft_vocab: Optional[str] = None,
    q_a: Optional[int] = None,
    tree_mode: str = "auto",
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

    # Match chat.py's runtime dtype so server and benchmark numbers are comparable.
    if device == "cuda":
        base_dtype = torch.float16
    elif cpu_quant_mode in ("int8", "int4"):
        base_dtype = torch.float32
    else:
        base_dtype = torch.bfloat16
    base_model, drafter, tok, mask_id = prepare_v3(model_name, device=device, dtype=base_dtype)

    draft_vocab_ids = None
    if draft_vocab and Path(draft_vocab).exists():
        print(f"Loading draft vocab IDs from {draft_vocab}...")
        loaded = torch.load(draft_vocab, map_location=device)
        draft_vocab_ids = loaded["draft_vocab_ids"] if isinstance(loaded, dict) else loaded
        print(f"Loaded {len(draft_vocab_ids)} draft vocab IDs.")

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
            print("\n[2/4] Running Compiled Baseline (torch.compile reduce-overhead)...")
            compiled_model = torch.compile(base_model, mode="reduce-overhead")
            # Untimed warmup: the first call pays the whole graph compilation cost.
            w_inputs = tok(prompts[:1], return_tensors="pt", padding=True).to(device)
            sync_device()
            with torch.no_grad():
                compiled_model.generate(**w_inputs, max_new_tokens=8, do_sample=False)
            sync_device()
            comp_times, comp_tokens = [], []
            for i in range(0, len(prompts), batch_size):
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

    # 3. Baseline 3: Base-Model-Only Graphed Causal Runner (Q=1)
    graphed_q1_tps = None
    if device == "cuda" and batch_size == 1:
        try:
            print("\n[3/4] Running Base-Model-Only Graphed Causal Baseline (Q=1)...")
            max_p_len_base = max(len(tok.encode(p)) for p in prompts)
            graphed_runner = GraphedCausalRunner(
                base_model=base_model,
                max_len=max_p_len_base + max_new_tokens + 64,
                device=device,
                dtype=base_dtype,
            )
            graphed_times, graphed_tokens = [], []
            for i in range(len(prompts)):
                b_prompt = prompts[i]
                inputs = tok(b_prompt, return_tensors="pt").to(device)
                p_ids = inputs["input_ids"]
                P_len = p_ids.shape[1]

                sync_device()
                t0 = time.perf_counter()
                with torch.no_grad():
                    logits = graphed_runner.prefill(p_ids)
                    if temperature == 0.0:
                        cur_tok = torch.argmax(logits, dim=-1)
                    else:
                        probs = F.softmax(logits / max(1e-4, temperature), dim=-1)
                        cur_tok = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(1, 1)

                    for step_idx in range(1, max_new_tokens):
                        cur_pos = P_len + step_idx - 1
                        logits_step = graphed_runner.step(cur_tok, cur_pos)
                        if temperature == 0.0:
                            cur_tok = torch.argmax(logits_step, dim=-1)
                        else:
                            probs = F.softmax(logits_step / max(1e-4, temperature), dim=-1)
                            cur_tok = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(1, 1)
                sync_device()
                elapsed = time.perf_counter() - t0
                graphed_times.append(elapsed)
                graphed_tokens.append(max_new_tokens)
            graphed_q1_tps = sum(graphed_tokens) / max(1e-4, sum(graphed_times))
            print(f"  Graphed Q=1 Baseline TPS: {graphed_q1_tps:.1f}")
        except Exception as e:
            print(f"  Note: Graphed Q=1 baseline skipped ({e})")

    # 4. Speculative: BFG-Diff v4 Drafter
    print("\n[4/4] Running BFG-Diff v4 Speculative Engine...")
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
        q_a=q_a,
        device=device,
        dtype=base_dtype,
        use_cuda_graph=(batch_size == 1),
        hedge_margin=hedge_margin,
        draft_vocab_ids=draft_vocab_ids,
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
                q_a=q_a,
                tree_mode=tree_mode,
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

        # Check token-exact match against raw baseline for greedy B=1
        if temperature == 0.0 and batch_size == 1:
            batch_idx = i // batch_size
            raw_batch = raw_outputs[batch_idx]
            if torch.equal(raw_batch.cpu(), spec_gen_only.cpu()):
                exact_matches += len(b_prompts)
            else:
                # Logit-margin first divergence diagnostic
                try:
                    r_toks = raw_batch[0]
                    s_toks = spec_gen_only[0]
                    min_len = min(len(r_toks), len(s_toks))
                    div_idx = None
                    for k_idx in range(min_len):
                        if r_toks[k_idx] != s_toks[k_idx]:
                            div_idx = k_idx
                            break
                    if div_idx is not None:
                        ctx_ids = torch.cat([input_ids[0].cpu(), r_toks[:div_idx]], dim=0).unsqueeze(0).to(device)
                        with torch.no_grad():
                            base_out = base_model(ctx_ids)
                            div_logits = base_out.logits[0, -1].float()
                            top2_vals = torch.topk(div_logits, 2).values
                            margin = (top2_vals[0] - top2_vals[1]).item()
                        print(f"  [Divergence Diagnostic] Prompt {i}: first mismatch at step {div_idx} | raw={int(r_toks[div_idx])} vs spec={int(s_toks[div_idx])} | base logit margin={margin:.4f}")
                except Exception as ex:
                    print(f"  [Divergence Diagnostic] Warning: {ex}")

    total_spec_time = sum(spec_times)
    total_spec_tokens = sum(spec_tokens_list)
    spec_tps = total_spec_tokens / max(1e-4, total_spec_time)

    max_spec_len = max(o.shape[1] for o in spec_outputs)
    padded_spec = [
        torch.cat([o, torch.full((o.shape[0], max_spec_len - o.shape[1]), tok.pad_token_id)], dim=1)
        if o.shape[1] < max_spec_len else o for o in spec_outputs
    ]
    all_spec_seqs = torch.cat(padded_spec, dim=0)

    n_batches = max(1, math.ceil(len(prompts) / batch_size))
    a1 = (a_plus_totals[0] / n_batches) * 100.0
    a2 = (a_plus_totals[1] / n_batches) * 100.0
    a3 = (a_plus_totals[2] / n_batches) * 100.0
    a4 = (a_plus_totals[3] / n_batches) * 100.0
    gamma_total = (total_accepted_drafts / max(1, total_draft_tokens)) * 100.0

    alpha = (total_accepted_drafts + total_steps) / max(1, total_steps)
    speedup = spec_tps / max(1e-4, raw_tps)
    if temperature == 0.0 and batch_size == 1:
        consistency_str = f"{(exact_matches / max(1, len(prompts))) * 100.0:.1f}%"
    else:
        consistency_str = "N/A"
    hedge_hit_rate = (total_hedge_hits / max(1, total_hedge_attempts)) * 100.0

    # Profile isolated verify and draft step times on populated P=128 cache.
    # Capture the CUDA graphs first: capture runs dummy kernels that overwrite both caches.
    eng = StaticDraftEngine(
        base_model, drafter, max_len=256, batch_size=1, v_tokens=v_tokens,
        q_a=q_a, device=device, dtype=base_dtype, use_cuda_graph=(device == "cuda"),
        draft_vocab_ids=draft_vocab_ids,
    )
    if eng.device.type == "cuda":
        eng.capture_cuda_graphs(mask_id)
    dummy_p = torch.randint(100, 1000, (1, 128), device=device)
    _, f_last = eng.prefill(dummy_p)
    dummy_y = dummy_p[:, -1:]
    dummy_v = torch.randint(100, 1000, (1, eng.q_a), device=device)

    for _ in range(3):
        with torch.no_grad():
            eng.forward_verify_step(dummy_v, [128])
            eng.forward_draft_step(dummy_y, mask_id, f_last, [128])

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
    if graphed_q1_tps is not None:
        print(f"{'Graphed Q=1 Baseline (TPS)':<35}{graphed_q1_tps:>15,.1f}{'-':>15}")
        spec_vs_graphed = spec_tps / max(1e-4, graphed_q1_tps)
        print(f"{'Speculative Speedup (vs Q=1)':<35}{'1.00x':>15}{f'{spec_vs_graphed:.2f}x':>15}")
    print(f"{'Wall-clock Speedup (vs HF)':<35}{'1.00x':>15}{f'{speedup:.2f}x':>15}")
    print(f"{'Mean Acceptance Alpha':<35}{'1.00':>15}{f'{alpha:.2f}':>15}")
    print(f"{'Gamma Total (Acceptance %)':<35}{'-':>15}{f'{gamma_total:.1f}%':>15}")
    print(f"{'A_+1 Acceptance Rate':<35}{'-':>15}{f'{a1:.1f}%':>15}")
    print(f"{'A_+2 Acceptance Rate':<35}{'-':>15}{f'{a2:.1f}%':>15}")
    print(f"{'A_+3 Acceptance Rate':<35}{'-':>15}{f'{a3:.1f}%':>15}")
    print(f"{'A_+4 Acceptance Rate':<35}{'-':>15}{f'{a4:.1f}%':>15}")
    print(f"{'Leaf-Hedge Hit Rate':<35}{'-':>15}{f'{hedge_hit_rate:.1f}% ({total_hedge_hits}/{total_hedge_attempts})':>15}")
    print(f"{'T_verify (Q_A=%d)' % eng.q_a:<35}{'-':>15}{f'{t_verify_ms:.2f} ms':>15}")
    print(f"{'T_draft (4L, W=%d)' % eng.w:<35}{'-':>15}{f'{t_draft_ms:.2f} ms':>15}")
    print(f"{'Exact Lossless Consistency %':<35}{'100.0%' if consistency_str != 'N/A' else 'N/A':>15}{consistency_str:>15}")
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
    p.add_argument("--draft_vocab", type=str, default=None, help="Path to draft_vocab_ids tensor")
    p.add_argument("--q_a", type=int, default=None, help="Verification tile size Q_A (e.g. 8 or 16, default: v_tokens + 1)")
    p.add_argument("--tree_mode", type=str, default="auto", choices=["auto", "linear", "tree"], help="Tree verification mode")
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
        draft_vocab=args.draft_vocab,
        q_a=args.q_a,
        tree_mode=args.tree_mode,
    )


if __name__ == "__main__":
    main()
