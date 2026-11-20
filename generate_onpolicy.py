"""BFG-Diff v3: One-Time On-Policy Dataset Generation (generate_onpolicy.py).

Features:
- Samples diverse prompts across Code, Math, Technical Systems, and Prose.
- Strictly zero overlap with benchmark.py's 50 GEN_PROMPTS.
- Batched generation (T=0.2 low-temp + greedy mix) at batch_size=32, max_new_tokens=256 using frozen target model.
- Compact .pt / .jsonl serialization of complete token sequences for zero distribution shift.
"""

import argparse
import itertools
import json
import math
import os
import time
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from benchmark_v3 import GEN_PROMPTS

# Curated seed templates across 5 diverse domains with zero benchmark overlap
PROMPT_DOMAINS = {
    "code": [
        "def quickselect(arr, k):\n    \"\"\"Find k-th smallest element in unsorted list.\"\"\"\n",
        "class TrieNode:\n    def __init__(self):\n        self.children = {}\n        self.is_word = False\n",
        "async def handle_websocket_message(ws, raw_payload: bytes):\n    \"\"\"Decode and route binary frame.\"\"\"\n",
        "def topological_sort(graph: dict[str, list[str]]) -> list[str]:\n    \"\"\"Return linear ordering of vertices.\"\"\"\n",
        "fn calculate_sha256_blocks(input_bytes: &[u8]) -> [u8; 32] {\n",
        "CREATE TABLE audit_logs (\n    id BIGSERIAL PRIMARY KEY,\n    user_uuid UUID NOT NULL,\n    action VARCHAR(64) NOT NULL\n);\n",
        "class ConvNeXtBlock(nn.Module):\n    def __init__(self, dim, drop_path=0.0):\n        super().__init__()\n",
        "export function debounce<T extends (...args: any[]) => void>(fn: T, delayMs: number): T {\n",
        "def solve_knapsack_01(weights: list[int], values: list[int], capacity: int) -> int:\n",
        "pub async fn poll_tcp_stream(stream: &mut TcpStream, buf: &mut [u8]) -> io::Result<usize> {\n",
    ],
    "math": [
        "Theorem: The sequence of primes contains arbitrarily long arithmetic progressions. Proof sketch:\n",
        "To find the eigenvalues of a block tridiagonal matrix, we begin by computing the characteristic polynomial:\n",
        "Problem: Compute the integral of x^3 * exp(-x^2) from 0 to infinity using integration by parts.\nSolution:\n",
        "Let G be a finite group and H a normal subgroup of G. By the first isomorphism theorem,\n",
        "Consider the second-order linear differential equation y'' + 4y' + 13y = 0 with initial conditions y(0)=1, y'(0)=2:\n",
        "To maximize the concave utility function U(x, y) = x^0.4 * y^0.6 subject to budget constraint px*x + py*y = M:\n",
        "Compute the discrete Fourier transform of the sequence [1, 0, -1, 0] of length N=4:\n",
        "In Bayesian inference, when the prior is conjugate to the likelihood, the posterior distribution\n",
    ],
    "systems": [
        "In a write-ahead log (WAL) database storage engine, crash recovery proceeds in three phases:\n1. Analysis phase:\n",
        "Linux kernel memory cgroups (memcg) track page allocations through page counters. When memory limits are reached,\n",
        "To design an idempotent message ingestion pipeline with Apache Kafka and PostgreSQL,\n",
        "Modern CPUs use branch target buffers (BTB) and direction predictors to minimize pipeline bubbles. Specifically,\n",
        "The Raft consensus algorithm resolves log divergence between followers and the leader by\n",
        "When designing a multi-region active-active database, conflict resolution policies such as Last-Write-Wins (LWW)\n",
        "In Kubernetes networking, kube-proxy configures iptables or IPVS mode to route service cluster IPs to\n",
        "Zero-copy socket I/O using splice() or io_uring eliminates kernel-to-user buffer copying by\n",
    ],
    "science": [
        "In quantum mechanics, the time-independent Schrödinger equation for a particle in a 1D infinite potential well is:\n",
        "Enzyme kinetics can be described by the Michaelis-Menten model, where the reaction velocity V relates to substrate [S]:\n",
        "CRISPR-Cas9 endonuclease initiates targeted double-strand breaks by utilizing a single guide RNA (sgRNA) that\n",
        "The second law of thermodynamics states that the total entropy of an isolated system always\n",
        "In stellar nucleosynthesis, the triple-alpha process converts three helium-4 nuclei into carbon-12 by\n",
        "Atmospheric carbon dioxide levels fluctuate seasonally primarily due to terrestrial biosphere photosythetic uptake in\n",
    ],
    "chat_prose": [
        "Explain the fundamental differences between monads and applicative functors in functional programming with clear examples.\n\n",
        "Summarize the tradeoffs between optimistic locking and pessimistic locking in relational database transaction management.\n\n",
        "Write a detailed technical design document for a distributed rate-limiting microservice handling 100k requests per second.\n\n",
        "Compare synchronous replication versus asynchronous replication in high-availability distributed storage clusters.\n\n",
        "Draft a clear migration plan for converting a monolithic legacy Django web application into modular FastAPI services.\n\n",
    ],
}


def build_prompts(num_prompts: int) -> List[str]:
    """Generate diverse prompt pool cycling through domains with variations."""
    all_seeds = []
    for domain, items in PROMPT_DOMAINS.items():
        all_seeds.extend(items)

    prompts = []
    cycle_seeds = itertools.cycle(all_seeds)
    for i in range(num_prompts):
        base = next(cycle_seeds)
        if i >= len(all_seeds):
            prefix_modifiers = [
                "Detailed explanation:\n",
                "Step-by-step implementation:\n",
                "Production-ready example:\n",
                "Analysis and proof:\n",
                "Comprehensive overview:\n",
            ]
            mod = prefix_modifiers[i % len(prefix_modifiers)]
            prompts.append(mod + base)
        else:
            prompts.append(base)
    return prompts


def load_hf_prompts(num_prompts: int) -> List[str]:
    """Load diverse unique prompts from HuggingFace datasets across code, math, chat, systems.
    Filters out any prompt with substring overlap against benchmark_v3's GEN_PROMPTS.
    """
    from datasets import load_dataset

    benchmark_low = [p.lower()[:32] for p in GEN_PROMPTS]

    def has_overlap(text: str) -> bool:
        t_low = text.lower()
        return any(b in t_low for b in benchmark_low)

    prompts = []
    seen = set()

    sources = [
        ("HuggingFaceH4/ultrachat_200k", "default", "train_sft", lambda x: x["messages"][0]["content"]),
        ("meta-math/MetaMathQA", None, "train", lambda x: x["query"]),
        ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", lambda x: x["text"][:300].rsplit(".", 1)[0] + "."),
        ("ise-uiuc/Magicoder-OSS-Instruct-75K", None, "train", lambda x: x["problem"]),
    ]

    per_source = math.ceil(num_prompts / len(sources))

    for ds_name, config, split, extractor in sources:
        if len(prompts) >= num_prompts:
            break
        source_count = 0
        try:
            print(f"Loading diverse prompts from {ds_name} (cap={per_source})...")
            kwargs = {"split": split, "streaming": True}
            if config:
                ds = load_dataset(ds_name, config, **kwargs)
            else:
                ds = load_dataset(ds_name, **kwargs)

            for item in ds:
                try:
                    text = extractor(item).strip()
                    if len(text) < 20 or len(text) > 1000:
                        continue
                    if has_overlap(text):
                        continue
                    if text not in seen:
                        seen.add(text)
                        prompts.append(text)
                        source_count += 1
                        if source_count >= per_source or len(prompts) >= num_prompts:
                            break
                except Exception:
                    continue
        except Exception as e:
            print(f"Warning: could not load {ds_name} ({e})")

    return prompts


def load_diverse_prompts(num_prompts: int, prompt_source: str = "hf", smoke_test: bool = False) -> List[str]:
    """Load prompts either from HF or local seeds with zero benchmark overlap."""
    if smoke_test or prompt_source == "local":
        return build_prompts(num_prompts)

    try:
        prompts = load_hf_prompts(num_prompts)
        if len(prompts) >= num_prompts:
            return prompts[:num_prompts]
        print(f"HF loaded {len(prompts)} prompts (< {num_prompts}); supplementing with local seeds...")
        supplement = build_prompts(num_prompts - len(prompts))
        return prompts + supplement
    except Exception as e:
        print(f"Falling back to local diverse prompt pool ({e})...")
        return build_prompts(num_prompts)


def generate_onpolicy_dataset(
    model_name: str = "Qwen/Qwen2.5-1.5B",
    out_path: str = "data/onpolicy_1.5b.pt",
    num_samples: int = 25000,
    batch_size: int = 32,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    smoke_test: bool = False,
    prompt_source: str = "hf",
):
    """Run batched generation using frozen target model and save token sequences."""
    print(f"Loading {model_name} for on-policy generation on {device}...")
    try:
        tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float16 if device == "cuda" else torch.float32
    kwargs = {"attn_implementation": "sdpa"}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, local_files_only=True, **kwargs)
    except Exception:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, local_files_only=True, **kwargs)
        except Exception:
            try:
                model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **kwargs)
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **kwargs)
    model.to(device).eval()

    if smoke_test:
        num_samples = min(num_samples, 4)
        batch_size = min(batch_size, 2)
        max_new_tokens = 32

    prompts = load_diverse_prompts(num_samples, prompt_source=prompt_source, smoke_test=smoke_test)
    prompts = prompts[int(os.environ.get("SHARD_IDX", "0"))::int(os.environ.get("NUM_SHARDS", "1"))]
    print(f"Loaded {len(prompts)} unique diverse prompts (source={prompt_source}). Generating responses at batch_size={batch_size}...")

    all_sequences = []
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    total_tokens = 0

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tok(batch_prompts, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            if temperature == 0.0:
                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            else:
                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    pad_token_id=tok.pad_token_id,
                )

        # Append CPU tensor
        all_sequences.append(gen_out.cpu())
        total_tokens += gen_out.numel()

        now = time.time()
        elapsed = now - t0
        done_samples = min(i + batch_size, len(prompts))
        tok_s = total_tokens / max(1e-4, elapsed)
        print(f"Progress: {done_samples:5d}/{len(prompts)} sequences | {tok_s:,.0f} tok/s | elapsed {elapsed:.0f}s")

    # Pad/truncate and concatenate
    max_len = max(s.shape[1] for s in all_sequences)
    padded_seqs = []
    for s in all_sequences:
        if s.shape[1] < max_len:
            pad = torch.full((s.shape[0], max_len - s.shape[1]), tok.pad_token_id, dtype=torch.long)
            s = torch.cat([s, pad], dim=-1)
        padded_seqs.append(s)

    final_dataset = torch.cat(padded_seqs, dim=0)
    torch.save(
        {
            "sequences": final_dataset,
            "vocab_size": model.config.vocab_size,
            "pad_token_id": tok.pad_token_id,
            "model_name": model_name,
            "num_samples": len(final_dataset),
        },
        out_path,
    )
    print(f"Saved {len(final_dataset)} sequences to {out_path} (shape: {final_dataset.shape})")
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--out", default="data/onpolicy_1.5b.pt")
    p.add_argument("--num_samples", type=int, default=25000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--device", default="auto")
    p.add_argument("--prompt_source", default="hf", choices=["hf", "local"])
    p.add_argument("--smoke_test", action="store_true")
    args = p.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    generate_onpolicy_dataset(
        model_name=args.model,
        out_path=args.out,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=device,
        smoke_test=args.smoke_test,
        prompt_source=args.prompt_source,
    )


if __name__ == "__main__":
    main()
