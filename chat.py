"""Interactive CLI and Visualizer for BFG-Diff models.

Supports live unmasking step-by-step visualization, side-by-side comparison
against base causal generation, completion and chat modes, and an optional Web UI.
"""
import argparse
import json
import math
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch
from peft import set_peft_model_state_dict

from model import LORA_TARGETS, apply_lora, prepare
from sampler import sample_canvas


def load_bfg_model(checkpoint_path, model_name="Qwen/Qwen2.5-0.5B", device=None):
    """Load base Qwen model and apply checkpoint adapters."""
    torch.set_num_threads(10)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    ck_file = Path(checkpoint_path)
    if not ck_file.exists():
        fallback = Path("runs/bfg-polished/last.pt") if Path("runs/bfg-polished/last.pt").exists() else Path("runs/bfg-shifted/last.pt")
        if fallback.exists():
            print(f"Notice: {checkpoint_path} not found. Falling back to {fallback}")
            ck_file = fallback
        elif Path("runs/smoke/last.pt").exists():
            ck_file = Path("runs/smoke/last.pt")
        else:
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")

    print(f"Loading checkpoint from {ck_file}...")
    ck = torch.load(ck_file, map_location="cpu")
    model_name = ck.get("model_name", model_name)

    model, tok, mask_id = prepare(model_name, device=device, dtype=torch.float32)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if "lora" in ck:
        cfg = ck.get("lora_config", {})
        model = apply_lora(
            model,
            cfg.get("r", 16),
            cfg.get("alpha", 32),
            cfg.get("dropout", 0.05),
            cfg.get("targets", LORA_TARGETS),
        )
        set_peft_model_state_dict(model, ck["lora"])
        if "mask_emb" in ck:
            with torch.no_grad():
                model.get_input_embeddings().weight[mask_id] = ck["mask_emb"].to(device)
    elif "model" in ck:
        model.load_state_dict(ck["model"], strict=False)

    if "bfg_adapter" in ck:
        from model import setup_bfg_adapter
        if getattr(model, "bfg_adapter", None) is None:
            setup_bfg_adapter(model)
        model.bfg_adapter.load_state_dict(ck["bfg_adapter"])
    model.eval()
    return model, tok, mask_id


def generate_bfg_stream(
    model,
    tok,
    mask_id,
    prompt_text,
    canvas_len=32,
    sub_canvas_len=8,
    sub_steps=4,
    temperature=0.0,
    carrier_priority=True,
    stop_tokens=None,
    step_callback=None,
):
    """Generate tokens using sample_canvas while streaming step-by-step updates via callback."""
    torch.set_num_threads(10)
    with torch.no_grad():
        prefix_ids = tok.encode(prompt_text, return_tensors="pt").to(model.device)

    start_time = time.perf_counter()
    tokens_generated = 0

    def block_trace_cb(b_idx, s, canvas_tensor):
        nonlocal tokens_generated
        elapsed = time.perf_counter() - start_time
        # Convert canvas tokens
        toks = [tok.decode([int(t)]) if int(t) != mask_id else "░" for t in canvas_tensor[0]]
        # Number of unmasked tokens so far in this block
        unmasked = sum(1 for t in canvas_tensor[0] if int(t) != mask_id)
        current_total = b_idx * sub_canvas_len + unmasked
        tps = current_total / max(1e-5, elapsed)
        if step_callback:
            step_callback({
                "type": "bfg_step",
                "block_idx": b_idx,
                "step": s,
                "tokens": toks,
                "text": "".join(toks),
                "unmasked": unmasked,
                "tps": tps,
                "elapsed": elapsed,
            })

    stop_token_ids = []
    if stop_tokens:
        for st in stop_tokens:
            tid = tok.convert_tokens_to_ids(st)
            if tid is not None and tid != tok.unk_token_id:
                stop_token_ids.append(tid)

    stats = {}
    canvas_out = sample_canvas(
        model,
        prefix_ids,
        canvas_len=canvas_len,
        sub_canvas_len=sub_canvas_len,
        steps=sub_steps * (canvas_len // sub_canvas_len),
        sub_steps=sub_steps,
        temperature=temperature,
        carrier_priority=carrier_priority,
        block_callback=block_trace_cb,
        stop_token_ids=stop_token_ids,
        stats_out=stats,
    )

    elapsed = time.perf_counter() - start_time
    total_tokens = canvas_out.shape[1]
    raw_text = tok.decode(canvas_out[0], skip_special_tokens=False)
    # Strip mask token completely if present
    raw_text = raw_text.replace("<|extra_0|>", "")

    if stop_tokens:
        for st in stop_tokens:
            if st in raw_text:
                raw_text = raw_text.split(st)[0]
    full_text = raw_text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
    tps = total_tokens / max(1e-5, elapsed)
    alpha = stats.get("alpha", 1.0)
    r_kv = stats.get("r_kv", 0.0)

    return {
        "text": full_text,
        "tokens": total_tokens,
        "elapsed": elapsed,
        "tps": tps,
        "alpha": alpha,
        "r_kv": r_kv,
        "badge": f"{tps:.1f} TPS | α={alpha:.2f} tok/step",
    }


def generate_bfg_traced(
    model,
    tok,
    mask_id,
    prompt_text,
    canvas_len=32,
    sub_canvas_len=8,
    sub_steps=4,
    temperature=0.0,
    carrier_priority=True,
    stop_tokens=None,
):
    """Generate tokens using unified sampler.py while capturing step-by-step traces."""
    torch.set_num_threads(10)
    with torch.no_grad():
        prefix_ids = tok.encode(prompt_text, return_tensors="pt").to(model.device)

    start_time = time.perf_counter()
    trace_blocks = {}

    def block_trace_cb(b_idx, s, canvas_tensor):
        if b_idx not in trace_blocks:
            # Initial step 0
            init_tokens = [tok.decode([int(t)]) if int(t) != mask_id else "░" for t in canvas_tensor[0]]
            trace_blocks[b_idx] = [{"step": 0, "tokens": init_tokens, "text": "".join(init_tokens)}]
        step_tokens = [tok.decode([int(t)]) if int(t) != mask_id else "░" for t in canvas_tensor[0]]
        trace_blocks[b_idx].append({"step": s, "tokens": step_tokens, "text": "".join(step_tokens)})

    stop_token_ids = []
    if stop_tokens:
        for st in stop_tokens:
            tid = tok.convert_tokens_to_ids(st)
            if tid is not None and tid != tok.unk_token_id:
                stop_token_ids.append(tid)

    stats = {}
    canvas_out = sample_canvas(
        model,
        prefix_ids,
        canvas_len=canvas_len,
        sub_canvas_len=sub_canvas_len,
        steps=sub_steps * (canvas_len // sub_canvas_len),
        sub_steps=sub_steps,
        temperature=temperature,
        carrier_priority=carrier_priority,
        block_callback=block_trace_cb,
        stop_token_ids=stop_token_ids,
        stats_out=stats,
    )

    elapsed = time.perf_counter() - start_time
    total_tokens = canvas_out.shape[1]
    raw_text = tok.decode(canvas_out[0], skip_special_tokens=False)
    raw_text = raw_text.replace("<|extra_0|>", "")

    # In chat mode, truncate cleanly at stop tokens
    if stop_tokens:
        for st in stop_tokens:
            if st in raw_text:
                raw_text = raw_text.split(st)[0]
    full_text = raw_text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
    tps = total_tokens / max(1e-5, elapsed)
    alpha = stats.get("alpha", 1.0)
    r_kv = stats.get("r_kv", 0.0)

    trace = []
    for b_idx in sorted(trace_blocks.keys()):
        steps_list = trace_blocks[b_idx]
        final_text = steps_list[-1]["text"].replace("░", "")
        trace.append({
            "block_idx": b_idx,
            "steps": steps_list,
            "final_text": final_text,
        })

    return {
        "text": full_text,
        "tokens": total_tokens,
        "elapsed": elapsed,
        "tps": tps,
        "alpha": alpha,
        "r_kv": r_kv,
        "badge": f"{tps:.1f} TPS | α={alpha:.2f} tok/step",
        "trace": trace,
    }


def generate_causal_stream(model, tok, prompt_text, max_new_tokens=32, callback=None):
    """Generate baseline response using causal next-token loop with streaming callback."""
    prefix_ids = tok.encode(prompt_text, return_tensors="pt").to(model.device)
    start_time = time.perf_counter()
    generated_ids = []
    curr_ids = prefix_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(curr_ids)
            next_id = int(out.logits[0, -1].argmax())
            if next_id == tok.eos_token_id or (tok.pad_token_id and next_id == tok.pad_token_id):
                break
            generated_ids.append(next_id)
            tok_text = tok.decode([next_id], skip_special_tokens=True)
            elapsed = time.perf_counter() - start_time
            tps = len(generated_ids) / max(1e-5, elapsed)
            if callback:
                callback(tok_text, len(generated_ids), tps, elapsed)
            curr_ids = torch.cat([curr_ids, torch.tensor([[next_id]], device=model.device)], dim=1)

    elapsed = time.perf_counter() - start_time
    full_text = tok.decode(generated_ids, skip_special_tokens=True)
    tps = len(generated_ids) / max(1e-5, elapsed)
    return {"text": full_text, "tokens": len(generated_ids), "elapsed": elapsed, "tps": tps}


def generate_causal(model, tok, prompt_text, max_new_tokens=32):
    """Generate baseline response using causal generate()."""
    prefix_ids = tok.encode(prompt_text, return_tensors="pt").to(model.device)
    start_time = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            prefix_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    elapsed = time.perf_counter() - start_time
    gen_tokens = out.shape[1] - prefix_ids.shape[1]
    text = tok.decode(out[0, prefix_ids.shape[1] :], skip_special_tokens=True)
    tps = gen_tokens / max(1e-5, elapsed)
    return {"text": text, "tokens": gen_tokens, "elapsed": elapsed, "tps": tps}


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>BFG-Diff Engine</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500;600&family=Geist:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #000000;
    --surface: #0a0a0a;
    --surface-elevated: #111111;
    --border: #222222;
    --border-hover: #333333;
    --text-primary: #ededed;
    --text-secondary: #888888;
    --text-tertiary: #555555;
    --accent: #ffffff;
    --accent-dim: #262626;
    --mono-font: "Geist Mono", SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    --sans-font: "Geist", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }

  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
    -webkit-tap-highlight-color: transparent;
  }

  body {
    background-color: var(--bg);
    color: var(--text-primary);
    font-family: var(--sans-font);
    font-size: 14px;
    line-height: 1.5;
    letter-spacing: -0.01em;
    padding: 20px 16px;
    display: flex;
    justify-content: center;
    min-height: 100vh;
    -webkit-font-smoothing: antialiased;
  }

  .shell {
    max-width: 880px;
    width: 100%;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  /* Top Navigation Bar */
  .top-bar {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
  }
  @media (min-width: 640px) {
    .top-bar {
      flex-direction: row;
      justify-content: space-between;
      align-items: center;
    }
  }

  .brand {
    display: flex;
    align-items: baseline;
    gap: 10px;
  }
  .brand-name {
    font-size: 15px;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--text-primary);
  }
  .brand-version {
    font-family: var(--mono-font);
    font-size: 11px;
    color: var(--text-tertiary);
  }

  .segmented-nav {
    display: inline-flex;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 2px;
  }
  .nav-item {
    background: transparent;
    border: none;
    color: var(--text-secondary);
    font-family: var(--sans-font);
    font-size: 12px;
    font-weight: 500;
    padding: 6px 14px;
    border-radius: 4px;
    cursor: pointer;
    transition: color 0.15s, background-color 0.15s;
    white-space: nowrap;
  }
  .nav-item:hover {
    color: var(--text-primary);
  }
  .nav-item.active {
    background: var(--surface-elevated);
    color: var(--text-primary);
    box-shadow: 0 1px 2px rgba(0,0,0,0.4);
  }

  /* Parameter Strip */
  .config-strip {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .preset-row {
    display: flex;
    align-items: center;
    gap: 8px;
    overflow-x: auto;
    scrollbar-width: none;
    -webkit-overflow-scrolling: touch;
    padding-bottom: 2px;
  }
  .preset-row::-webkit-scrollbar { display: none; }
  .preset-label {
    font-size: 11px;
    font-family: var(--mono-font);
    color: var(--text-tertiary);
    text-transform: uppercase;
    white-space: nowrap;
    letter-spacing: 0.04em;
  }
  .preset-btn {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text-secondary);
    font-family: var(--sans-font);
    font-size: 12px;
    padding: 4px 10px;
    border-radius: 4px;
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.15s;
  }
  .preset-btn:hover {
    border-color: var(--border-hover);
    color: var(--text-primary);
  }
  .preset-btn.active {
    background: var(--accent);
    color: var(--bg);
    border-color: var(--accent);
    font-weight: 500;
  }

  .params-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    padding-top: 10px;
    border-top: 1px solid var(--border);
  }
  @media (min-width: 640px) {
    .params-grid {
      grid-template-columns: repeat(5, 1fr);
      gap: 12px;
    }
  }

  .field-group {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .field-label {
    font-family: var(--mono-font);
    font-size: 10px;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .field-input, .field-select {
    background: var(--surface-elevated);
    border: 1px solid var(--border);
    color: var(--text-primary);
    font-family: var(--mono-font);
    font-size: 12px;
    padding: 6px 8px;
    border-radius: 4px;
    outline: none;
    transition: border-color 0.15s;
    width: 100%;
  }
  .field-input:focus, .field-select:focus {
    border-color: var(--border-hover);
  }

  /* Buttons */
  .btn-primary {
    background: var(--accent);
    color: var(--bg);
    border: none;
    font-family: var(--sans-font);
    font-size: 13px;
    font-weight: 500;
    padding: 8px 16px;
    border-radius: 5px;
    cursor: pointer;
    transition: opacity 0.15s;
    width: 100%;
  }
  @media (min-width: 640px) {
    .btn-primary { width: auto; }
  }
  .btn-primary:hover { opacity: 0.9; }
  .btn-primary:disabled { opacity: 0.4; cursor: not-allowed; }

  .btn-ghost {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text-secondary);
    font-family: var(--sans-font);
    font-size: 12px;
    padding: 5px 10px;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.15s;
  }
  .btn-ghost:hover {
    border-color: var(--border-hover);
    color: var(--text-primary);
  }

  /* Workspace Cards */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }
  .card-header {
    padding: 10px 14px;
    background: var(--surface-elevated);
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .card-title {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-secondary);
    letter-spacing: -0.01em;
  }
  .telemetry-chip {
    font-family: var(--mono-font);
    font-size: 11px;
    color: var(--text-tertiary);
  }

  .prompt-box {
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .prompt-textarea {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text-primary);
    font-family: var(--sans-font);
    font-size: 13.5px;
    line-height: 1.5;
    padding: 10px 12px;
    resize: vertical;
    min-height: 80px;
    outline: none;
    transition: border-color 0.15s;
    width: 100%;
  }
  .prompt-textarea:focus { border-color: var(--border-hover); }

  /* Output Comparison */
  .comparison-split {
    display: grid;
    grid-template-columns: 1fr;
    gap: 12px;
  }
  @media (min-width: 768px) {
    .comparison-split { grid-template-columns: 1fr 1fr; }
  }
  .output-view {
    padding: 14px;
    font-size: 13.5px;
    line-height: 1.55;
    color: var(--text-primary);
    min-height: 120px;
    white-space: pre-wrap;
    word-break: break-word;
  }

  /* Execution Trace */
  .trace-container {
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .trace-sub-block {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 8px 10px;
  }
  .trace-header {
    font-family: var(--mono-font);
    font-size: 10px;
    text-transform: uppercase;
    color: var(--text-tertiary);
    margin-bottom: 6px;
    letter-spacing: 0.04em;
  }
  .trace-row {
    font-family: var(--mono-font);
    font-size: 11.5px;
    display: flex;
    gap: 8px;
    align-items: baseline;
    flex-wrap: wrap;
    line-height: 1.8;
  }
  .chip {
    padding: 1px 5px;
    background: var(--surface-elevated);
    border: 1px solid var(--border);
    border-radius: 3px;
    color: var(--text-secondary);
    font-size: 11px;
  }

  /* Multi-Turn Chat View */
  .chat-wrapper {
    height: calc(100vh - 200px);
    min-height: 480px;
    max-height: 720px;
    display: flex;
    flex-direction: column;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }
  .chat-feed {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .message-unit {
    display: flex;
    flex-direction: column;
    max-width: 86%;
  }
  .message-unit.user {
    align-self: flex-end;
  }
  .message-unit.assistant {
    align-self: flex-start;
  }
  .message-meta {
    font-family: var(--mono-font);
    font-size: 10px;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
  }
  .message-unit.user .message-meta {
    text-align: right;
  }
  .message-body {
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 13.5px;
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .message-unit.user .message-body {
    background: var(--surface-elevated);
    border: 1px solid var(--border);
    color: var(--text-primary);
  }
  .message-unit.assistant .message-body {
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--text-primary);
  }
  .message-telemetry {
    font-family: var(--mono-font);
    font-size: 10.5px;
    color: var(--text-tertiary);
    margin-top: 4px;
  }

  .chat-input-deck {
    padding: 12px 14px;
    border-top: 1px solid var(--border);
    background: var(--surface-elevated);
    display: flex;
    gap: 10px;
    align-items: flex-end;
  }
  .chat-textarea {
    flex: 1;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 5px;
    color: var(--text-primary);
    font-family: var(--sans-font);
    font-size: 13.5px;
    line-height: 1.45;
    padding: 8px 12px;
    resize: none;
    min-height: 40px;
    max-height: 110px;
    outline: none;
  }
  .chat-textarea:focus { border-color: var(--border-hover); }

  .chat-placeholder {
    margin: auto;
    text-align: center;
    color: var(--text-tertiary);
    font-size: 12.5px;
    font-family: var(--mono-font);
    padding: 20px;
  }
</style>
</head>
<body>

<div class="shell">
  <!-- Minimal Top Header -->
  <header class="top-bar">
    <div class="brand">
      <span class="brand-name">BFG-Diff</span>
      <span class="brand-version">v2 · Staircase Diffusion</span>
    </div>
    <nav class="segmented-nav">
      <button class="nav-item active" id="navVisualizer" onclick="selectTab('visualizer')">Visualizer</button>
      <button class="nav-item" id="navChat" onclick="selectTab('chat')">Chat</button>
    </nav>
  </header>

  <!-- Clean Config Strip -->
  <section class="config-strip">
    <div class="preset-row">
      <span class="preset-label">Preset:</span>
      <button class="preset-btn active" onclick="loadPreset('balanced')">Balanced</button>
      <button class="preset-btn" onclick="loadPreset('creative')">Creative</button>
      <button class="preset-btn" onclick="loadPreset('fast')">Fast Lookahead</button>
      <button class="preset-btn" onclick="loadPreset('greedy')">Deterministic</button>
    </div>
    <div class="params-grid">
      <div class="field-group">
        <label class="field-label">Mode</label>
        <select class="field-select" id="mode">
          <option value="chat" selected>ChatML</option>
          <option value="completion">Raw</option>
        </select>
      </div>
      <div class="field-group">
        <label class="field-label">Tokens</label>
        <input class="field-input" type="number" id="canvas_len" value="32" min="8" max="256" step="8">
      </div>
      <div class="field-group">
        <label class="field-label">Block Size</label>
        <input class="field-input" type="number" id="sub_canvas_len" value="8" min="4" max="32" step="4">
      </div>
      <div class="field-group">
        <label class="field-label">Steps / Block</label>
        <input class="field-input" type="number" id="sub_steps" value="4" min="1" max="16" step="1">
      </div>
      <div class="field-group" style="grid-column: span 1;">
        <label class="field-label">Temperature</label>
        <input class="field-input" type="number" id="temp" value="0.3" min="0.0" max="1.5" step="0.1">
      </div>
    </div>
  </section>

  <!-- VIEW 1: VISUALIZER -->
  <main id="paneVisualizer">
    <div class="card" style="margin-bottom: 16px;">
      <div class="prompt-box">
        <textarea class="prompt-textarea" id="prompt" rows="3">Explain the difference between a process and a thread in one short paragraph.</textarea>
        <div style="display: flex; justify-content: flex-end;">
          <button class="btn-primary" id="runBtn" onclick="executeGeneration()">Execute & Compare</button>
        </div>
      </div>
    </div>

    <div class="comparison-split" style="margin-bottom: 16px;">
      <div class="card">
        <div class="card-header">
          <span class="card-title">BFG-Diff (Discrete Diffusion)</span>
          <span class="telemetry-chip" id="bfgTelemetry">0.0 TPS</span>
        </div>
        <div class="output-view" id="bfgOutput">Awaiting execution...</div>
      </div>
      <div class="card">
        <div class="card-header">
          <span class="card-title">Causal Baseline (generate)</span>
          <span class="telemetry-chip" id="causalTelemetry">0.0 TPS</span>
        </div>
        <div class="output-view" id="causalOutput">Awaiting execution...</div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <span class="card-title">Block Unmasking Trace</span>
      </div>
      <div class="trace-container" id="traceContainer">
        <div class="chat-placeholder">Execute a prompt to inspect speculative lookahead stages.</div>
      </div>
    </div>
  </main>

  <!-- VIEW 2: MULTI-TURN CHAT -->
  <main id="paneChat" style="display: none;">
    <div class="chat-wrapper">
      <div class="card-header">
        <span class="card-title">Session Memory</span>
        <button class="btn-ghost" onclick="resetChat()">Reset Context</button>
      </div>
      <div class="chat-feed" id="chatFeed">
        <div class="chat-placeholder" id="chatPlaceholder">
          Context retained across turns. Send a message to begin.
        </div>
      </div>
      <div class="chat-input-deck">
        <textarea class="chat-textarea" id="chatInput" placeholder="Write a message..."></textarea>
        <button class="btn-primary" id="chatSubmitBtn" onclick="submitMessage()" style="width:auto; padding:8px 14px;">Send</button>
      </div>
    </div>
  </main>
</div>

<script>
let activeTab = 'visualizer';
let sessionContext = [
  { role: "system", content: "You are a concise, helpful assistant powered by BFG-Diff v2 discrete diffusion." }
];

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function selectTab(tab) {
  activeTab = tab;
  document.getElementById('navVisualizer').className = 'nav-item' + (tab === 'visualizer' ? ' active' : '');
  document.getElementById('navChat').className = 'nav-item' + (tab === 'chat' ? ' active' : '');
  document.getElementById('paneVisualizer').style.display = tab === 'visualizer' ? 'block' : 'none';
  document.getElementById('paneChat').style.display = tab === 'chat' ? 'block' : 'none';
}

function loadPreset(name) {
  document.querySelectorAll('.preset-btn').forEach(btn => btn.classList.remove('active'));
  if (event && event.target) event.target.classList.add('active');

  const canvasLen = document.getElementById('canvas_len');
  const subCanvas = document.getElementById('sub_canvas_len');
  const subSteps = document.getElementById('sub_steps');
  const temp = document.getElementById('temp');

  if (name === 'balanced') {
    canvasLen.value = 32;
    subCanvas.value = 8;
    subSteps.value = 4;
    temp.value = 0.3;
  } else if (name === 'creative') {
    canvasLen.value = 48;
    subCanvas.value = 8;
    subSteps.value = 4;
    temp.value = 0.7;
  } else if (name === 'fast') {
    canvasLen.value = 32;
    subCanvas.value = 16;
    subSteps.value = 2;
    temp.value = 0.1;
  } else if (name === 'greedy') {
    canvasLen.value = 32;
    subCanvas.value = 8;
    subSteps.value = 4;
    temp.value = 0.0;
  }
}

async function executeGeneration() {
  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  btn.innerText = 'Processing...';

  const bfgOutput = document.getElementById('bfgOutput');
  const causalOutput = document.getElementById('causalOutput');
  const bfgTelemetry = document.getElementById('bfgTelemetry');
  const causalTelemetry = document.getElementById('causalTelemetry');
  const traceContainer = document.getElementById('traceContainer');

  bfgOutput.innerText = 'Unmasking active blocks...';
  causalOutput.innerText = 'Generating token stream...';
  bfgTelemetry.innerText = '0.0 TPS';
  causalTelemetry.innerText = '0.0 TPS';
  traceContainer.innerHTML = '';

  const payload = {
    prompt: document.getElementById('prompt').value,
    mode: document.getElementById('mode').value,
    canvas_len: parseInt(document.getElementById('canvas_len').value),
    sub_canvas_len: parseInt(document.getElementById('sub_canvas_len').value),
    sub_steps: parseInt(document.getElementById('sub_steps').value),
    temp: parseFloat(document.getElementById('temp').value),
  };

  const blockMap = {};
  let causalBuffer = '';

  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error("Server returned HTTP " + res.status);

    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let raw = '';

    const NL = String.fromCharCode(10);
    const CR = String.fromCharCode(13);
    const DBL_NL = NL + NL;

    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      raw += decoder.decode(chunk.value, { stream: true });

      const normalized = raw.split(CR + NL).join(NL);
      const items = normalized.split(DBL_NL);
      raw = items.pop();

      for (const item of items) {
        const line = item.trim();
        if (!line.startsWith('data:')) continue;
        const body = line.substring(5).trim();
        if (!body) continue;

        const evt = JSON.parse(body);

        if (evt.type === 'causal_token') {
          causalBuffer += evt.token;
          causalOutput.innerText = causalBuffer;
          causalTelemetry.innerText = evt.tps.toFixed(1) + " TPS (" + evt.elapsed.toFixed(2) + "s)";
        } else if (evt.type === 'causal_done') {
          causalOutput.innerText = evt.causal.text;
          causalTelemetry.innerText = evt.causal.tps.toFixed(1) + " TPS (" + evt.causal.elapsed.toFixed(2) + "s)";
        } else if (evt.type === 'bfg_step') {
          bfgOutput.innerText = evt.text;
          bfgTelemetry.innerText = evt.tps.toFixed(1) + " TPS (" + evt.elapsed.toFixed(2) + "s)";

          if (!blockMap[evt.block_idx]) {
            const blockEl = document.createElement('div');
            blockEl.className = 'trace-sub-block';
            blockEl.innerHTML = '<div class="trace-header">Sub-Block ' + (evt.block_idx + 1) + '</div>';
            traceContainer.appendChild(blockEl);
            blockMap[evt.block_idx] = blockEl;
          }

          const blockEl = blockMap[evt.block_idx];
          const row = document.createElement('div');
          row.className = 'trace-row';
          const chips = evt.tokens.map(t => '<span class="chip">' + escapeHtml(t) + '</span>').join(' ');
          row.innerHTML = '<span style="color:var(--text-tertiary); min-width:44px;">P' + evt.step + '</span>' + chips;
          blockEl.appendChild(row);
        } else if (evt.type === 'bfg_done') {
          bfgOutput.innerText = evt.bfg.text;
          bfgTelemetry.innerText = evt.bfg.tps.toFixed(1) + " TPS (" + evt.bfg.elapsed.toFixed(2) + "s)";
        }
      }
    }
  } catch (err) {
    console.error("Execution error:", err);
    bfgOutput.innerText = "Error: " + err.message;
  } finally {
    btn.disabled = false;
    btn.innerText = 'Execute & Compare';
  }
}

function resetChat() {
  sessionContext = [
    { role: "system", content: "You are a concise, helpful assistant powered by BFG-Diff v2 discrete diffusion." }
  ];
  document.getElementById('chatFeed').innerHTML = `
    <div class="chat-placeholder" id="chatPlaceholder">
      Context retained across turns. Send a message to begin.
    </div>
  `;
}

async function submitMessage() {
  const input = document.getElementById('chatInput');
  const text = input.value.trim();
  if (!text) return;

  const btn = document.getElementById('chatSubmitBtn');
  btn.disabled = true;
  input.disabled = true;

  const placeholder = document.getElementById('chatPlaceholder');
  if (placeholder) placeholder.remove();

  sessionContext.push({ role: "user", content: text });
  input.value = '';

  const feed = document.getElementById('chatFeed');

  const userUnit = document.createElement('div');
  userUnit.className = 'message-unit user';
  userUnit.innerHTML = `
    <div class="message-meta">You</div>
    <div class="message-body">${escapeHtml(text)}</div>
  `;
  feed.appendChild(userUnit);

  const assistantUnit = document.createElement('div');
  assistantUnit.className = 'message-unit assistant';
  const bodyId = 'replyBody_' + Date.now();
  const telemetryId = 'replyTelem_' + Date.now();
  assistantUnit.innerHTML = `
    <div class="message-meta">BFG-Diff</div>
    <div class="message-body" id="${bodyId}">Diffusing...</div>
    <div class="message-telemetry" id="${telemetryId}"></div>
  `;
  feed.appendChild(assistantUnit);
  feed.scrollTop = feed.scrollHeight;

  const bodyEl = document.getElementById(bodyId);
  const telemEl = document.getElementById(telemetryId);

  const payload = {
    messages: sessionContext,
    canvas_len: parseInt(document.getElementById('canvas_len').value) || 32,
    sub_canvas_len: parseInt(document.getElementById('sub_canvas_len').value) || 8,
    sub_steps: parseInt(document.getElementById('sub_steps').value) || 4,
    temp: parseFloat(document.getElementById('temp').value) || 0.3,
  };

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error("HTTP " + res.status);

    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let raw = '';
    let finalAnswer = '';

    const NL = String.fromCharCode(10);
    const CR = String.fromCharCode(13);
    const DBL_NL = NL + NL;

    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      raw += decoder.decode(chunk.value, { stream: true });

      const normalized = raw.split(CR + NL).join(NL);
      const items = normalized.split(DBL_NL);
      raw = items.pop();

      for (const item of items) {
        const line = item.trim();
        if (!line.startsWith('data:')) continue;
        const body = line.substring(5).trim();
        if (!body) continue;

        const evt = JSON.parse(body);
        if (evt.type === 'chat_step') {
          bodyEl.innerText = evt.text;
          telemEl.innerText = evt.tps.toFixed(1) + " TPS (" + evt.elapsed.toFixed(2) + "s)";
          feed.scrollTop = feed.scrollHeight;
        } else if (evt.type === 'chat_done') {
          finalAnswer = evt.text;
          bodyEl.innerText = evt.text;
          telemEl.innerText = `${evt.tps.toFixed(1)} TPS · ${evt.tokens} tokens · ${evt.elapsed.toFixed(2)}s`;
          sessionContext.push({ role: "assistant", content: finalAnswer });
          feed.scrollTop = feed.scrollHeight;
        }
      }
    }
  } catch (err) {
    console.error("Chat failure:", err);
    bodyEl.innerText = "Error: " + err.message;
  } finally {
    btn.disabled = false;
    input.disabled = false;
    input.focus();
  }
}

document.getElementById('chatInput').addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    submitMessage();
  }
});
</script>
</body>
</html>
"""


class VisualizerHandler(BaseHTTPRequestHandler):
    model = None
    tok = None
    mask_id = None

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/chat":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            req = json.loads(body.decode("utf-8"))

            messages = req.get("messages", [])
            canvas_len = req.get("canvas_len", 32)
            sub_canvas_len = req.get("sub_canvas_len", 8)
            sub_steps = req.get("sub_steps", 4)
            temp = req.get("temp", 0.3)

            # Apply Jinja2/ChatML template if available, else format cleanly
            try:
                formatted_prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                formatted_prompt = ""
                for m in messages:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                formatted_prompt += "<|im_start|>assistant\n"

            stop_tokens = ["<|im_end|>", "<|endoftext|>"]

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            pad = ": " + (" " * 2048) + "\n\n"
            def send_chat_sse(evt_data):
                msg = (f"data: {json.dumps(evt_data)}\n\n" + pad).encode("utf-8")
                try:
                    self.wfile.write(msg)
                    self.wfile.flush()
                except Exception:
                    pass

            # Stream response using BFG discrete diffusion with stop tokens
            bfg_res = generate_bfg_stream(
                self.model,
                self.tok,
                self.mask_id,
                formatted_prompt,
                canvas_len=canvas_len,
                sub_canvas_len=sub_canvas_len,
                sub_steps=sub_steps,
                temperature=temp,
                stop_tokens=stop_tokens,
                step_callback=lambda evt: send_chat_sse({
                    "type": "chat_step",
                    "text": evt.get("text", ""),
                    "tps": evt.get("tps", 0.0),
                    "elapsed": evt.get("elapsed", 0.0),
                }),
            )
            clean_reply = bfg_res.get("text", "")
            for st in stop_tokens:
                clean_reply = clean_reply.replace(st, "")
            clean_reply = clean_reply.strip()

            send_chat_sse({
                "type": "chat_done",
                "text": clean_reply,
                "tokens": bfg_res.get("tokens", 0),
                "tps": bfg_res.get("tps", 0.0),
                "elapsed": bfg_res.get("elapsed", 0.0),
            })
            return
        if self.path == "/api/generate":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            req = json.loads(body.decode("utf-8"))

            prompt = req.get("prompt", "")
            mode = req.get("mode", "completion")
            canvas_len = req.get("canvas_len", 32)
            sub_canvas_len = req.get("sub_canvas_len", 8)
            sub_steps = req.get("sub_steps", 4)
            temp = req.get("temp", 0.0)

            if mode == "chat" and "<|im_start|>" not in prompt:
                prompt = (
                    f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                )

            stop_tokens = ["<|im_end|>", "<|endoftext|>"] if mode == "chat" else None

            # Streaming SSE endpoint
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            # 2KB comment padding to force immediate flush through proxy buffers (e.g. Cloudflare)
            pad = ": " + (" " * 2048) + "\n\n"

            def send_sse(evt_data):
                msg = (f"data: {json.dumps(evt_data)}\n\n" + pad).encode("utf-8")
                try:
                    self.wfile.write(msg)
                    self.wfile.flush()
                except Exception:
                    pass

            # 1. Stream BFG Discrete Diffusion step-by-step FIRST for instant visual response
            bfg_res = generate_bfg_stream(
                self.model,
                self.tok,
                self.mask_id,
                prompt,
                canvas_len=canvas_len,
                sub_canvas_len=sub_canvas_len,
                sub_steps=sub_steps,
                temperature=temp,
                stop_tokens=stop_tokens,
                step_callback=lambda evt: send_sse(evt),
            )
            send_sse({"type": "bfg_done", "bfg": bfg_res})

            # 2. Stream Causal Baseline token by token in real-time
            causal_res = generate_causal_stream(
                self.model,
                self.tok,
                prompt,
                max_new_tokens=canvas_len,
                callback=lambda tok_str, total, tps, elap: send_sse({
                    "type": "causal_token",
                    "token": tok_str,
                    "total": total,
                    "tps": tps,
                    "elapsed": elap,
                }),
            )
            send_sse({"type": "causal_done", "causal": causal_res})
        else:
            self.send_response(404)
            self.end_headers()


def start_server(model, tok, mask_id, port=8766):
    VisualizerHandler.model = model
    VisualizerHandler.tok = tok
    VisualizerHandler.mask_id = mask_id
    server = ThreadingHTTPServer(("0.0.0.0", port), VisualizerHandler)
    print(f"\n[Visualizer Web UI] Live at http://localhost:{port} and http://bfg-dff.gces.net.in")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down visualizer server...")


def run_cli_interactive(model, tok, mask_id, args):
    """Interactive terminal chat / completion loop."""
    mode = args.mode
    canvas_len = args.canvas_len
    sub_canvas_len = args.sub_canvas_len
    sub_steps = args.sub_steps
    temp = args.temp
    visualize = args.visualize
    compare = args.compare

    print("=" * 70)
    print(" BFG-Diff Interactive Terminal Playground")
    print(f" Checkpoint: {args.checkpoint} | Device: {model.device}")
    print(" Commands: /mode chat|complete, /vis, /compare, /c <len>, /k <steps>, /temp <val>, /quit")
    print("=" * 70)

    while True:
        try:
            prompt = input(f"\n[{mode.upper()}] >>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not prompt:
            continue
        if prompt in ["/quit", "/exit"]:
            break
        if prompt.startswith("/mode"):
            parts = prompt.split()
            if len(parts) > 1 and parts[1].lower() in ["chat", "completion", "complete"]:
                mode = "chat" if parts[1].lower() == "chat" else "completion"
                print(f"Mode set to: {mode}")
            continue
        if prompt == "/vis":
            visualize = not visualize
            print(f"Step visualization: {visualize}")
            continue
        if prompt == "/compare":
            compare = not compare
            print(f"Causal comparison: {compare}")
            continue
        if prompt.startswith("/c "):
            canvas_len = int(prompt.split()[1])
            print(f"Canvas length: {canvas_len}")
            continue
        if prompt.startswith("/k "):
            sub_steps = int(prompt.split()[1])
            print(f"Sub-steps: {sub_steps}")
            continue
        if prompt.startswith("/temp "):
            temp = float(prompt.split()[1])
            print(f"Temperature: {temp}")
            continue

        formatted_prompt = prompt
        if mode == "chat" and "<|im_start|>" not in prompt:
            formatted_prompt = (
                f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            )

        print("-" * 70)
        res = generate_bfg_traced(
            model,
            tok,
            mask_id,
            formatted_prompt,
            canvas_len=canvas_len,
            sub_canvas_len=sub_canvas_len,
            sub_steps=sub_steps,
            temperature=temp,
            stop_tokens=["<|im_end|>", "<|endoftext|>"] if mode == "chat" else None,
        )

        if visualize:
            print("\n[Step-by-Step Parallel Unmasking Trace]")
            for block in res["trace"]:
                print(
                    f"--- Sub-Block {block['block_idx']+1} (Final: {repr(block['final_text'])}) ---"
                )
                for st in block["steps"]:
                    print(f"  Step {st['step']}: {' '.join(st['tokens'])}")
            print()

        print(f"BFG-Diff ({res['tps']:.1f} TPS, {res['elapsed']:.2f}s):")
        print(res["text"])

        if compare:
            c_res = generate_causal(model, tok, formatted_prompt, max_new_tokens=canvas_len)
            print(f"\nCausal Base ({c_res['tps']:.1f} TPS, {c_res['elapsed']:.2f}s):")
            print(c_res["text"])
        print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description="BFG-Diff Interactive Chat & Visualizer")
    parser.add_argument("--checkpoint", default="runs/bfg-polished/last.pt")
    parser.add_argument("--mode", choices=["chat", "completion"], default="completion")
    parser.add_argument("--canvas_len", type=int, default=32)
    parser.add_argument("--sub_canvas_len", type=int, default=8)
    parser.add_argument("--sub_steps", type=int, default=4)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--visualize", action="store_true", help="Print step-by-step unmasking")
    parser.add_argument("--compare", action="store_true", default=True)
    parser.add_argument("--no_compare", action="store_false", dest="compare")
    parser.add_argument("--server", action="store_true", help="Launch web visualizer server")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--prompt", type=str, default=None, help="Single prompt execution non-interactively"
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"], help="Compute device"
    )
    args = parser.parse_args()

    # Load model
    device = torch.device(
        "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    )
    model, tok, mask_id = load_bfg_model(args.checkpoint, device=device)

    if args.server:
        start_server(model, tok, mask_id, port=args.port)
    elif args.prompt:
        res = generate_bfg_traced(
            model,
            tok,
            mask_id,
            args.prompt,
            canvas_len=args.canvas_len,
            sub_canvas_len=args.sub_canvas_len,
            sub_steps=args.sub_steps,
            temperature=args.temp,
        )
        if args.visualize:
            print("\n[Step-by-Step Parallel Unmasking Trace]")
            for block in res["trace"]:
                print(
                    f"--- Sub-Block {block['block_idx']+1} (Final: {repr(block['final_text'])}) ---"
                )
                for st in block["steps"]:
                    print(f"  Step {st['step']}: {' '.join(st['tokens'])}")
            print()
        print(f"BFG-Diff ({res['tps']:.1f} TPS, {res['elapsed']:.2f}s):")
        print(res["text"])
        if args.compare:
            c_res = generate_causal(model, tok, args.prompt, max_new_tokens=args.canvas_len)
            print(f"\nCausal Base ({c_res['tps']:.1f} TPS, {c_res['elapsed']:.2f}s):")
            print(c_res["text"])
    else:
        run_cli_interactive(model, tok, mask_id, args)


if __name__ == "__main__":
    main()
