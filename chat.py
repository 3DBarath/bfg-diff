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
<title>BFG-Diff v2 - Visualizer & Multi-Turn Chat</title>
<style>
  :root {
    --bg: #090d16;
    --card: #131c2e;
    --card-hover: #1a263d;
    --border: #23324d;
    --primary: #38bdf8;
    --primary-glow: rgba(56, 189, 248, 0.2);
    --text: #f1f5f9;
    --muted: #94a3b8;
    --mask: #334155;
    --accent: #a855f7;
    --user-msg: #1e293b;
    --bot-msg: #0f1f38;
    --success: #10b981;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
  body {
    background-color: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    padding: 10px;
    display: flex;
    justify-content: center;
    -webkit-font-smoothing: antialiased;
    min-height: 100vh;
  }
  @media (min-width: 600px) {
    body { padding: 16px; }
  }
  .container {
    max-width: 960px;
    width: 100%;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
  }
  header {
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-bottom: 14px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 12px;
  }
  @media (min-width: 640px) {
    header {
      flex-direction: row;
      justify-content: space-between;
      align-items: center;
    }
  }
  .header-left h1 {
    font-size: 18px;
    color: var(--primary);
    display: flex;
    align-items: center;
    gap: 8px;
    letter-spacing: -0.3px;
  }
  @media (min-width: 600px) {
    .header-left h1 { font-size: 22px; }
  }
  .header-left h1 span.badge {
    font-size: 10px;
    background: var(--primary-glow);
    color: var(--primary);
    border: 1px solid var(--primary);
    padding: 2px 6px;
    border-radius: 4px;
    text-transform: uppercase;
    font-weight: 700;
  }
  p.subtitle { margin-top: 3px; color: var(--muted); font-size: 12px; }

  /* Navigation Tabs */
  .tabs-nav {
    display: flex;
    gap: 4px;
    background: var(--card);
    padding: 3px;
    border-radius: 8px;
    border: 1px solid var(--border);
    width: 100%;
  }
  @media (min-width: 640px) {
    .tabs-nav { width: auto; }
  }
  .tab-btn {
    flex: 1;
    background: transparent;
    border: none;
    color: var(--muted);
    font-weight: 600;
    font-size: 12px;
    padding: 8px 12px;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.2s;
    text-align: center;
    white-space: nowrap;
  }
  @media (min-width: 600px) {
    .tab-btn { font-size: 13px; padding: 8px 16px; flex: initial; }
  }
  .tab-btn.active {
    background: var(--primary);
    color: #090d16;
    box-shadow: 0 2px 8px var(--primary-glow);
    font-weight: 700;
  }

  /* Preset Pills */
  .preset-bar {
    display: flex;
    align-items: center;
    gap: 6px;
    overflow-x: auto;
    padding-bottom: 6px;
    margin-bottom: 12px;
    scrollbar-width: none;
    -webkit-overflow-scrolling: touch;
  }
  .preset-bar::-webkit-scrollbar { display: none; }
  .preset-label {
    font-size: 11px;
    color: var(--muted);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    white-space: nowrap;
  }
  .preset-pill {
    background: var(--card);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 5px 11px;
    border-radius: 16px;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
    flex-shrink: 0;
  }
  .preset-pill:hover, .preset-pill.active {
    border-color: var(--primary);
    color: var(--primary);
    background: var(--primary-glow);
  }

  /* Parameter Controls */
  .controls {
    background: var(--card);
    padding: 12px;
    border-radius: 8px;
    border: 1px solid var(--border);
    margin-bottom: 14px;
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
  }
  @media (min-width: 640px) {
    .controls {
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 12px;
      padding: 14px;
    }
  }
  .control-group {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  label { font-size: 11px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
  input, select, textarea, button {
    background: #090d16;
    border: 1px solid var(--border);
    color: var(--text);
    padding: 8px 10px;
    border-radius: 6px;
    font-size: 14px;
    box-sizing: border-box;
    width: 100%;
  }
  input:focus, select:focus, textarea:focus {
    outline: none;
    border-color: var(--primary);
  }
  button.action-btn {
    background: var(--primary);
    color: #090d16;
    font-weight: 700;
    cursor: pointer;
    border: none;
    padding: 10px 18px;
    border-radius: 6px;
    font-size: 14px;
    transition: all 0.2s;
    width: 100%;
  }
  @media (min-width: 600px) {
    button.action-btn { width: auto; }
  }
  button.action-btn:hover { opacity: 0.9; }
  button.action-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  button.secondary-btn {
    background: transparent;
    color: var(--muted);
    border: 1px solid var(--border);
    font-weight: 600;
    cursor: pointer;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 12px;
    transition: all 0.2s;
    width: auto;
  }
  button.secondary-btn:hover {
    color: var(--text);
    border-color: var(--muted);
  }

  /* Visualizer Cards */
  .prompt-card {
    background: var(--card);
    padding: 12px;
    border-radius: 8px;
    border: 1px solid var(--border);
    margin-bottom: 14px;
  }
  .results { display: flex; flex-direction: column; gap: 14px; }
  .comparison-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 12px;
  }
  @media (min-width: 768px) {
    .comparison-grid { grid-template-columns: 1fr 1fr; }
  }
  .model-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }
  .model-title {
    background: rgba(0,0,0,0.25);
    padding: 10px 12px;
    font-size: 13px;
    font-weight: 700;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .metric-badge {
    font-family: monospace;
    font-size: 11px;
    background: rgba(56, 189, 248, 0.15);
    color: var(--primary);
    padding: 2px 6px;
    border-radius: 4px;
    border: 1px solid rgba(56, 189, 248, 0.3);
  }
  .output-box {
    padding: 12px;
    min-height: 90px;
    line-height: 1.5;
    font-size: 13px;
    white-space: pre-wrap;
    word-break: break-word;
    color: #e2e8f0;
  }
  .trace-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
  }
  .trace-title {
    font-size: 12px;
    font-weight: 700;
    color: var(--muted);
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .trace-block {
    background: #090d16;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px;
    margin-bottom: 8px;
  }
  .trace-step {
    font-family: monospace;
    font-size: 11px;
    margin-bottom: 4px;
    display: flex;
    gap: 6px;
    align-items: baseline;
    flex-wrap: wrap;
  }
  .token-chip {
    padding: 1px 4px;
    border-radius: 3px;
    background: #1e293b;
    border: 1px solid #334155;
    color: #cbd5e1;
    font-size: 11px;
  }

  /* Multi-Turn Chat UI */
  .chat-pane {
    display: flex;
    flex-direction: column;
    height: calc(100vh - 220px);
    min-height: 420px;
    max-height: 680px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }
  .chat-header {
    padding: 10px 14px;
    background: rgba(0, 0, 0, 0.2);
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
  }
  .chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .msg-row {
    display: flex;
    flex-direction: column;
    max-width: 90%;
  }
  @media (min-width: 600px) {
    .msg-row { max-width: 82%; }
  }
  .msg-row.user {
    align-self: flex-end;
  }
  .msg-row.assistant {
    align-self: flex-start;
  }
  .msg-sender {
    font-size: 10px;
    font-weight: 700;
    color: var(--muted);
    margin-bottom: 3px;
    text-transform: uppercase;
  }
  .msg-row.user .msg-sender { text-align: right; color: var(--primary); }
  .msg-bubble {
    padding: 9px 13px;
    border-radius: 8px;
    font-size: 13.5px;
    line-height: 1.45;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .msg-row.user .msg-bubble {
    background: var(--user-msg);
    border: 1px solid var(--border);
    color: #fff;
    border-top-right-radius: 2px;
  }
  .msg-row.assistant .msg-bubble {
    background: var(--bot-msg);
    border: 1px solid rgba(56, 189, 248, 0.25);
    color: #e2e8f0;
    border-top-left-radius: 2px;
  }
  .msg-stats {
    font-size: 10.5px;
    font-family: monospace;
    color: var(--muted);
    margin-top: 3px;
  }
  .chat-input-bar {
    padding: 10px 12px;
    border-top: 1px solid var(--border);
    background: rgba(0, 0, 0, 0.25);
    display: flex;
    gap: 8px;
    align-items: flex-end;
  }
  .chat-input-bar textarea {
    flex: 1;
    min-height: 42px;
    max-height: 100px;
    resize: none;
    line-height: 1.4;
    padding: 8px 10px;
    font-size: 13.5px;
  }
  .chat-input-bar button {
    width: auto;
    padding: 10px 16px;
    flex-shrink: 0;
  }
  .empty-state {
    text-align: center;
    color: var(--muted);
    margin: auto;
    font-size: 13px;
    padding: 20px;
    line-height: 1.5;
  }
</style>
</head>
<body>

<div class="container">
  <header>
    <div class="header-left">
      <h1>BFG-Diff v2 <span class="badge">Live</span></h1>
      <p class="subtitle">Staircase Verify-and-Diffuse Engine</p>
    </div>
    <div class="tabs-nav">
      <button class="tab-btn active" id="tabVisualizer" onclick="switchTab('visualizer')">Visualizer</button>
      <button class="tab-btn" id="tabChat" onclick="switchTab('chat')">Chat Session</button>
    </div>
  </header>

  <!-- Presets Bar -->
  <div class="preset-bar">
    <span class="preset-label">Presets:</span>
    <button class="preset-pill active" onclick="applyPreset('balanced')">Balanced</button>
    <button class="preset-pill" onclick="applyPreset('creative')">Creative (T=0.7)</button>
    <button class="preset-pill" onclick="applyPreset('fast')">Fast Lookahead</button>
    <button class="preset-pill" onclick="applyPreset('greedy')">Greedy (T=0.0)</button>
  </div>

  <!-- Manual Controls -->
  <div class="controls" id="controlsPanel">
    <div class="control-group">
      <label>Mode</label>
      <select id="mode">
        <option value="chat" selected>Chat (ChatML)</option>
        <option value="completion">Raw Completion</option>
      </select>
    </div>
    <div class="control-group">
      <label>Canvas (Tokens)</label>
      <input type="number" id="canvas_len" value="32" min="8" max="256" step="8">
    </div>
    <div class="control-group">
      <label>Sub-Block Size</label>
      <input type="number" id="sub_canvas_len" value="8" min="4" max="32" step="4">
    </div>
    <div class="control-group">
      <label>Sub-Steps</label>
      <input type="number" id="sub_steps" value="4" min="1" max="16" step="1">
    </div>
    <div class="control-group" style="grid-column: span 2;">
      <label>Temperature</label>
      <input type="number" id="temp" value="0.3" min="0.0" max="1.5" step="0.1">
    </div>
  </div>

  <!-- VIEW 1: DIFFUSION VISUALIZER -->
  <div id="visualizerView">
    <div class="prompt-card">
      <label style="display:block; margin-bottom:8px;">Input Prompt</label>
      <textarea id="prompt" rows="3">Explain the difference between a process and a thread in one short paragraph.</textarea>
      <div style="margin-top: 10px; display: flex; justify-content: flex-end;">
        <button class="action-btn" id="runBtn" onclick="runGeneration()">Generate & Compare</button>
      </div>
    </div>

    <div class="results" id="resultsSection">
      <div class="comparison-grid">
        <div class="model-card">
          <div class="model-title">
            <span>BFG-Diff (Verify & Diffuse)</span>
            <span class="metric-badge" id="bfgMetrics">0.0 TPS</span>
          </div>
          <div class="output-box" id="bfgOutput">Ready to generate.</div>
        </div>
        <div class="model-card">
          <div class="model-title">
            <span>Causal Baseline</span>
            <span class="metric-badge" id="causalMetrics">0.0 TPS</span>
          </div>
          <div class="output-box" id="causalOutput">Ready to generate.</div>
        </div>
      </div>

      <div class="trace-card" id="traceContainer">
        <div class="trace-title">Block Denoising Trace</div>
        <div id="traceLogs"><span style="color:var(--muted); font-size:12px;">Click "Generate & Compare" to view unmasking trace.</span></div>
      </div>
    </div>
  </div>

  <!-- VIEW 2: MULTI-TURN CONVERSATIONAL CHAT -->
  <div id="chatView" style="display:none;">
    <div class="chat-pane">
      <div class="chat-header">
        <div>
          <strong style="font-size:13px; color:var(--primary);">Session</strong>
          <span style="font-size:11px; color:var(--muted); margin-left:6px;">Context retained</span>
        </div>
        <button class="secondary-btn" onclick="clearChat()">Clear History</button>
      </div>
      <div class="chat-messages" id="chatMessages">
        <div class="empty-state" id="emptyChatNotice">
          Start a conversation. The model remembers past turns in this session.
        </div>
      </div>
      <div class="chat-input-bar">
        <textarea id="chatInput" placeholder="Message assistant..."></textarea>
        <button class="action-btn" id="chatSendBtn" onclick="sendChatMessage()">Send</button>
      </div>
    </div>
  </div>
</div>

<script>
let currentTab = 'visualizer';
let chatHistory = [
  { role: "system", content: "You are a concise, helpful assistant powered by BFG-Diff v2 discrete diffusion." }
];

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function switchTab(tab) {
  currentTab = tab;
  document.getElementById('tabVisualizer').className = 'tab-btn' + (tab === 'visualizer' ? ' active' : '');
  document.getElementById('tabChat').className = 'tab-btn' + (tab === 'chat' ? ' active' : '');
  document.getElementById('visualizerView').style.display = tab === 'visualizer' ? 'block' : 'none';
  document.getElementById('chatView').style.display = tab === 'chat' ? 'block' : 'none';
}

function applyPreset(name) {
  document.querySelectorAll('.preset-pill').forEach(el => el.classList.remove('active'));
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

// Visualizer Generator
async function runGeneration() {
  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  btn.innerText = 'Streaming...';

  const bfgOutput = document.getElementById('bfgOutput');
  const causalOutput = document.getElementById('causalOutput');
  const bfgMetrics = document.getElementById('bfgMetrics');
  const causalMetrics = document.getElementById('causalMetrics');
  const traceLogs = document.getElementById('traceLogs');

  bfgOutput.innerText = 'Parallel unmasking...';
  causalOutput.innerText = 'Waiting for causal stream...';
  bfgMetrics.innerText = '0.0 TPS';
  causalMetrics.innerText = '0.0 TPS';
  traceLogs.innerHTML = '';

  const payload = {
    prompt: document.getElementById('prompt').value,
    mode: document.getElementById('mode').value,
    canvas_len: parseInt(document.getElementById('canvas_len').value),
    sub_canvas_len: parseInt(document.getElementById('sub_canvas_len').value),
    sub_steps: parseInt(document.getElementById('sub_steps').value),
    temp: parseFloat(document.getElementById('temp').value),
  };

  const blockContainers = {};
  let causalAccum = '';

  try {
    const response = await fetch('/api/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });

    if (!response.ok) throw new Error("Server returned HTTP " + response.status);

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    const NL = String.fromCharCode(10);
    const CR = String.fromCharCode(13);
    const DBL_NL = NL + NL;

    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true });

      const normalized = buffer.split(CR + NL).join(NL);
      const parts = normalized.split(DBL_NL);
      buffer = parts.pop();

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith('data:')) continue;
        const jsonStr = line.substring(5).trim();
        if (!jsonStr) continue;

        const evt = JSON.parse(jsonStr);

        if (evt.type === 'causal_token') {
          causalAccum += evt.token;
          causalOutput.innerText = causalAccum;
          causalMetrics.innerText = evt.tps.toFixed(1) + " TPS (" + evt.elapsed.toFixed(2) + "s)";
        } else if (evt.type === 'causal_done') {
          causalOutput.innerText = evt.causal.text;
          causalMetrics.innerText = evt.causal.tps.toFixed(1) + " TPS (" + evt.causal.elapsed.toFixed(2) + "s)";
        } else if (evt.type === 'bfg_step') {
          bfgOutput.innerText = evt.text;
          bfgMetrics.innerText = evt.tps.toFixed(1) + " TPS (" + evt.elapsed.toFixed(2) + "s)";

          if (!blockContainers[evt.block_idx]) {
            const bDiv = document.createElement('div');
            bDiv.className = 'trace-block';
            bDiv.innerHTML = '<div class="trace-step" style="font-weight:700; color:var(--primary); margin-bottom:6px;">Sub-Block ' + (evt.block_idx + 1) + '</div>';
            traceLogs.appendChild(bDiv);
            blockContainers[evt.block_idx] = bDiv;
          }

          const bDiv = blockContainers[evt.block_idx];
          const sRow = document.createElement('div');
          sRow.className = 'trace-step';
          const chips = evt.tokens.map(t => '<span class="token-chip">' + escapeHtml(t) + '</span>').join(' ');
          sRow.innerHTML = '<span style="color:var(--muted); min-width:44px;">Step ' + evt.step + ':</span>' + chips;
          bDiv.appendChild(sRow);
        } else if (evt.type === 'bfg_done') {
          bfgOutput.innerText = evt.bfg.text;
          bfgMetrics.innerText = evt.bfg.tps.toFixed(1) + " TPS (" + evt.bfg.elapsed.toFixed(2) + "s)";
        }
      }
    }
  } catch (err) {
    console.error("Stream error:", err);
    bfgOutput.innerText = 'Error: ' + err.message;
  } finally {
    btn.disabled = false;
    btn.innerText = 'Generate & Compare';
  }
}

// Conversational Chat Functions
function clearChat() {
  chatHistory = [
    { role: "system", content: "You are a concise, helpful assistant powered by BFG-Diff v2 discrete diffusion." }
  ];
  document.getElementById('chatMessages').innerHTML = `
    <div class="empty-state" id="emptyChatNotice">
      Start a conversation. The model remembers past turns in this session.
    </div>
  `;
}

async function sendChatMessage() {
  const inputEl = document.getElementById('chatInput');
  const userText = inputEl.value.trim();
  if (!userText) return;

  const sendBtn = document.getElementById('chatSendBtn');
  sendBtn.disabled = true;
  inputEl.disabled = true;

  const notice = document.getElementById('emptyChatNotice');
  if (notice) notice.remove();

  chatHistory.push({ role: "user", content: userText });
  inputEl.value = '';

  const messagesContainer = document.getElementById('chatMessages');

  const userRow = document.createElement('div');
  userRow.className = 'msg-row user';
  userRow.innerHTML = `
    <div class="msg-sender">You</div>
    <div class="msg-bubble">${escapeHtml(userText)}</div>
  `;
  messagesContainer.appendChild(userRow);

  const botRow = document.createElement('div');
  botRow.className = 'msg-row assistant';
  const botBubbleId = 'botMsg_' + Date.now();
  const botStatsId = 'botStats_' + Date.now();
  botRow.innerHTML = `
    <div class="msg-sender">BFG-Diff Assistant</div>
    <div class="msg-bubble" id="${botBubbleId}">Diffusing...</div>
    <div class="msg-stats" id="${botStatsId}"></div>
  `;
  messagesContainer.appendChild(botRow);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;

  const botBubble = document.getElementById(botBubbleId);
  const botStats = document.getElementById(botStatsId);

  const payload = {
    messages: chatHistory,
    canvas_len: parseInt(document.getElementById('canvas_len').value) || 32,
    sub_canvas_len: parseInt(document.getElementById('sub_canvas_len').value) || 8,
    sub_steps: parseInt(document.getElementById('sub_steps').value) || 4,
    temp: parseFloat(document.getElementById('temp').value) || 0.3,
  };

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });

    if (!response.ok) throw new Error("HTTP error " + response.status);

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let finalAssistantText = '';

    const NL = String.fromCharCode(10);
    const CR = String.fromCharCode(13);
    const DBL_NL = NL + NL;

    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true });

      const normalized = buffer.split(CR + NL).join(NL);
      const parts = normalized.split(DBL_NL);
      buffer = parts.pop();

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith('data:')) continue;
        const jsonStr = line.substring(5).trim();
        if (!jsonStr) continue;

        const evt = JSON.parse(jsonStr);
        if (evt.type === 'chat_step') {
          botBubble.innerText = evt.text;
          botStats.innerText = evt.tps.toFixed(1) + " TPS (" + evt.elapsed.toFixed(2) + "s)";
          messagesContainer.scrollTop = messagesContainer.scrollHeight;
        } else if (evt.type === 'chat_done') {
          finalAssistantText = evt.text;
          botBubble.innerText = evt.text;
          botStats.innerText = `${evt.tps.toFixed(1)} TPS · ${evt.tokens} tokens · ${evt.elapsed.toFixed(2)}s`;
          chatHistory.push({ role: "assistant", content: finalAssistantText });
          messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
      }
    }
  } catch (err) {
    console.error("Chat error:", err);
    botBubble.innerText = "Error: " + err.message;
  } finally {
    sendBtn.disabled = false;
    inputEl.disabled = false;
    inputEl.focus();
  }
}

document.getElementById('chatInput').addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendChatMessage();
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
