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
    enable_branching=False,
    tau=0.8,
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
        enable_branching=enable_branching,
        tau=tau,
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
    
    # Calculate true overall average TPS based on actual emitted response tokens
    actual_tokens = len(tok.encode(full_text, add_special_tokens=False)) if full_text else 0
    reported_tokens = max(1, actual_tokens) if full_text else total_tokens
    tps = reported_tokens / max(1e-5, elapsed)
    alpha = stats.get("alpha", 1.0)
    r_kv = stats.get("r_kv", 0.0)

    return {
        "text": full_text,
        "tokens": reported_tokens,
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
        enable_branching=enable_branching,
        tau=tau,
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

    actual_tokens = len(tok.encode(full_text, add_special_tokens=False)) if full_text else 0
    reported_tokens = max(1, actual_tokens) if full_text else total_tokens
    tps = reported_tokens / max(1e-5, elapsed)
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
        "tokens": reported_tokens,
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
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, interactive-widget=resizes-content">
<title>BFG-Diff Engine</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500;600&family=Geist:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #09090b;
    --surface: #111114;
    --surface-elevated: #18181c;
    --surface-hover: #1f1f24;
    --border: #242429;
    --border-hover: #36363e;
    --border-subtle: #19191d;
    --text-primary: #f4f4f5;
    --text-secondary: #a1a1aa;
    --text-tertiary: #52525b;
    --accent: #ffffff;
    --accent-contrast: #000000;
    --token-bg: #141418;
    --token-border: #27272f;
    --token-highlight: #38bdf8;
    --mono-font: "Geist Mono", SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    --sans-font: "Geist", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }

  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
    -webkit-tap-highlight-color: transparent;
  }

  html, body {
    height: 100%;
    width: 100%;
    overflow: hidden;
    background-color: var(--bg);
    color: var(--text-primary);
    font-family: var(--sans-font);
    font-size: 14px;
    line-height: 1.5;
    letter-spacing: -0.01em;
    -webkit-font-smoothing: antialiased;
  }

  .app-frame {
    display: flex;
    flex-direction: column;
    height: 100%;
    width: 100%;
    position: relative;
  }

  /* Fixed Header */
  .top-header {
    height: 52px;
    min-height: 52px;
    padding: 0 16px;
    border-bottom: 1px solid var(--border-subtle);
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--bg);
    z-index: 20;
  }
  @media (min-width: 768px) {
    .top-header { padding: 0 24px; }
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .brand-title {
    font-size: 14.5px;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--text-primary);
  }
  .brand-tag {
    font-family: var(--mono-font);
    font-size: 10.5px;
    color: var(--text-tertiary);
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 2px 6px;
    border-radius: 4px;
  }

  /* Preset Dropdown in Header */
  .preset-select-wrapper {
    position: relative;
    display: flex;
    align-items: center;
  }
  .preset-dropdown {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text-secondary);
    font-family: var(--sans-font);
    font-size: 12px;
    font-weight: 500;
    padding: 5px 24px 5px 10px;
    border-radius: 5px;
    cursor: pointer;
    outline: none;
    appearance: none;
    -webkit-appearance: none;
    transition: all 0.15s ease;
  }
  .preset-dropdown:hover, .preset-dropdown:focus {
    border-color: var(--border-hover);
    color: var(--text-primary);
  }
  .preset-caret {
    position: absolute;
    right: 8px;
    pointer-events: none;
    font-size: 9px;
    color: var(--text-tertiary);
  }

  .header-actions {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  @media (min-width: 640px) {
    .header-actions { gap: 10px; }
  }

  /* Segmented Navigation */
  .segmented-control {
    display: inline-flex;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 2px;
  }
  .seg-btn {
    background: transparent;
    border: none;
    color: var(--text-secondary);
    font-family: var(--sans-font);
    font-size: 12px;
    font-weight: 500;
    padding: 5px 11px;
    border-radius: 4px;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: all 0.15s ease;
  }
  .seg-btn:hover { color: var(--text-primary); }
  .seg-btn.active {
    background: var(--surface-elevated);
    color: var(--text-primary);
    box-shadow: 0 1px 3px rgba(0,0,0,0.4);
  }

  .icon-btn {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text-secondary);
    font-size: 12px;
    padding: 6px;
    border-radius: 5px;
    cursor: pointer;
    font-family: var(--sans-font);
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.15s ease;
  }
  .icon-btn:hover {
    border-color: var(--border-hover);
    color: var(--text-primary);
    background: var(--surface-hover);
  }

  
  /* Custom Styled Menu (Replaces Native Android/iOS Picker) */
  .custom-menu-wrapper {
    position: relative;
    width: 100%;
  }
  .custom-menu-trigger {
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--text-primary);
    font-family: var(--sans-font);
    font-size: 13px;
    padding: 8px 12px;
    border-radius: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: pointer;
    user-select: none;
    transition: border-color 0.15s ease;
  }
  .custom-menu-trigger:hover, .custom-menu-trigger.active {
    border-color: var(--border-hover);
  }
  .custom-menu-options {
    position: absolute;
    top: calc(100% + 4px);
    left: 0;
    width: 100%;
    background: var(--surface-elevated);
    border: 1px solid var(--border-hover);
    border-radius: 6px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.7);
    z-index: 200;
    display: none;
    flex-direction: column;
    max-height: 220px;
    overflow-y: scroll;
    overflow-x: hidden;
    -webkit-overflow-scrolling: touch;
    touch-action: pan-y;
    overscroll-behavior: contain;
  }
  .custom-menu-options.open {
    display: flex;
    animation: menuFadeIn 0.12s ease-out;
  }
  @keyframes menuFadeIn {
    from { opacity: 0; transform: translateY(-4px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .custom-menu-item {
    padding: 9px 12px;
    font-size: 12.5px;
    color: var(--text-secondary);
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: background 0.15s, color 0.15s;
    user-select: none;
  }
  .custom-menu-item:hover {
    background: var(--surface-hover);
    color: var(--text-primary);
  }
  .custom-menu-item.selected {
    color: var(--text-primary);
    font-weight: 500;
    background: rgba(255, 255, 255, 0.04);
  }
  .custom-menu-item .item-sub {
    font-family: var(--mono-font);
    font-size: 10.5px;
    color: var(--text-tertiary);
  }

  /* Modal Popup (Clean Dialog) */
  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: rgba(0, 0, 0, 0.65);
    backdrop-filter: blur(4px);
    z-index: 100;
    display: none;
    align-items: center;
    justify-content: center;
    padding: 16px;
  }
  .modal-overlay.open {
    display: flex;
    animation: modalFadeIn 0.15s ease-out;
  }
  @keyframes modalFadeIn {
    from { opacity: 0; transform: scale(0.98); }
    to { opacity: 1; transform: scale(1); }
  }

  .modal-card {
    background: var(--surface);
    border: 1px solid var(--border-hover);
    border-radius: 10px;
    width: 100%;
    max-width: 440px;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.6);
    overflow: visible;
    position: relative;
  }
  .modal-header {
    padding: 14px 16px;
    background: var(--surface-elevated);
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .modal-title {
    font-size: 13.5px;
    font-weight: 600;
    color: var(--text-primary);
  }
  .modal-close-btn {
    background: transparent;
    border: none;
    color: var(--text-tertiary);
    font-size: 16px;
    cursor: pointer;
    padding: 2px 6px;
    border-radius: 4px;
    line-height: 1;
  }
  .modal-close-btn:hover { color: var(--text-primary); }

  .modal-body {
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .modal-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
  }
  .modal-field {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .modal-field label {
    font-family: var(--mono-font);
    font-size: 10.5px;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .modal-field input, .modal-field select {
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--text-primary);
    font-family: var(--mono-font);
    font-size: 12.5px;
    padding: 7px 10px;
    border-radius: 5px;
    outline: none;
    transition: border-color 0.15s;
    width: 100%;
  }
  .modal-field input:focus, .modal-field select:focus {
    border-color: var(--border-hover);
  }
  .modal-footer {
    padding: 12px 16px;
    background: var(--surface-elevated);
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  /* -------------------------------------------------------------
     FULLSCREEN CHAT INTERFACE (ChatGPT Style)
     ------------------------------------------------------------- */
  #tabChatView {
    flex: 1;
    display: flex;
    flex-direction: column;
    height: calc(100% - 52px);
    width: 100%;
    position: relative;
    overflow: hidden;
  }

  .chat-stream-viewport {
    flex: 1;
    overflow-y: auto;
    overflow-x: hidden;
    padding: 20px 16px 120px 16px;
    display: flex;
    flex-direction: column;
    align-items: center;
    scroll-behavior: smooth;
  }
  @media (min-width: 768px) {
    .chat-stream-viewport { padding: 32px 24px 140px 24px; }
  }

  .chat-conversation-flow {
    max-width: 768px;
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 24px;
  }

  .message-block {
    display: flex;
    width: 100%;
    gap: 14px;
    animation: fadeIn 0.2s ease-out;
  }
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .message-block.user { justify-content: flex-end; }
  .message-block.assistant { justify-content: flex-start; }

  .message-content {
    display: flex;
    flex-direction: column;
    max-width: 88%;
  }
  @media (min-width: 768px) {
    .message-content { max-width: 82%; }
  }

  .user-bubble {
    background: var(--surface-elevated);
    border: 1px solid var(--border);
    color: var(--text-primary);
    padding: 10px 16px;
    border-radius: 18px;
    font-size: 14px;
    line-height: 1.55;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .assistant-stream {
    color: var(--text-primary);
    font-size: 14.5px;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
    padding: 4px 0;
  }

  .message-stats {
    font-family: var(--mono-font);
    font-size: 11px;
    color: var(--text-tertiary);
    margin-top: 6px;
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .chat-welcome {
    margin: auto;
    text-align: center;
    max-width: 480px;
    padding: 40px 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
  }
  .welcome-logo {
    width: 44px;
    height: 44px;
    border-radius: 10px;
    background: var(--surface);
    border: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--mono-font);
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
  }
  .welcome-title {
    font-size: 17px;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--text-primary);
  }
  .welcome-sub {
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.5;
  }

  /* Floating Bottom Composer Deck */
  .chat-composer-floating {
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    padding: 12px 16px 20px 16px;
    display: flex;
    justify-content: center;
    background: linear-gradient(180deg, transparent 0%, var(--bg) 28%);
    pointer-events: none;
    z-index: 10;
  }
  @media (min-width: 768px) {
    .chat-composer-floating { padding: 16px 24px 28px 24px; }
  }

  .composer-capsule {
    max-width: 768px;
    width: 100%;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 8px 10px 8px 18px;
    display: flex;
    align-items: flex-end;
    gap: 10px;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.5);
    pointer-events: auto;
    transition: border-color 0.15s ease;
  }
  .composer-capsule:focus-within { border-color: var(--border-hover); }

  .chat-prompt-field {
    flex: 1;
    background: transparent;
    border: none;
    color: var(--text-primary);
    font-family: var(--sans-font);
    font-size: 14px;
    line-height: 1.45;
    padding: 6px 0;
    resize: none;
    min-height: 24px;
    max-height: 160px;
    outline: none;
  }
  .chat-prompt-field::placeholder { color: var(--text-tertiary); }

  .submit-circle-btn {
    width: 34px;
    height: 34px;
    min-width: 34px;
    border-radius: 50%;
    background: var(--accent);
    color: var(--accent-contrast);
    border: none;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: opacity 0.15s ease, transform 0.1s ease;
  }
  .submit-circle-btn:hover { opacity: 0.9; }
  .submit-circle-btn:active { transform: scale(0.96); }
  .submit-circle-btn:disabled { opacity: 0.2; cursor: not-allowed; }
  .submit-circle-btn svg { width: 14px; height: 14px; fill: currentColor; }

  /* -------------------------------------------------------------
     VISUALIZER TAB INTERFACE (With Enhanced Block Trace)
     ------------------------------------------------------------- */
  #tabVisualizerView {
    flex: 1;
    overflow-y: auto;
    overflow-x: hidden;
    padding: 16px 14px 60px 14px;
    display: none;
    -webkit-overflow-scrolling: touch;
    height: calc(100% - 52px);
    width: 100%;
  }
  @media (min-width: 640px) {
    #tabVisualizerView { padding: 24px 20px 60px 20px; }
  }

  .vis-container {
    max-width: 900px;
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 16px;
    margin: 0 auto;
    align-items: stretch;
    height: fit-content;
  }

  .vis-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    width: 100%;
    flex-shrink: 0;
  }
  .vis-header {
    padding: 10px 14px;
    background: var(--surface-elevated);
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
  }
  .vis-title {
    font-size: 12.5px;
    font-weight: 500;
    color: var(--text-secondary);
  }
  .vis-meta {
    font-family: var(--mono-font);
    font-size: 11px;
    color: var(--text-tertiary);
    flex-shrink: 0;
  }

  .vis-prompt-deck {
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .vis-textarea {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text-primary);
    font-family: var(--sans-font);
    font-size: 13.5px;
    line-height: 1.5;
    padding: 10px 12px;
    resize: vertical;
    min-height: 84px;
    max-height: 240px;
    outline: none;
    width: 100%;
    flex-shrink: 0;
    box-sizing: border-box;
    transition: border-color 0.15s;
  }
  .vis-textarea:focus { border-color: var(--border-hover); }

  .vis-split {
    display: grid;
    grid-template-columns: 1fr;
    gap: 14px;
    width: 100%;
    flex-shrink: 0;
  }
  @media (min-width: 768px) {
    .vis-split {
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }
  }

  .vis-output {
    padding: 14px;
    font-size: 13.5px;
    line-height: 1.55;
    color: var(--text-primary);
    min-height: 110px;
    max-height: 380px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
    box-sizing: border-box;
  }

  /* Enhanced Block-by-Block Visualization Trace */
  .trace-feed {
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    max-height: 540px;
    overflow-y: auto;
  }
  .trace-block {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    animation: fadeIn 0.2s ease-out;
  }
  .trace-block-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-subtle);
  }
  .trace-block-tag {
    font-family: var(--mono-font);
    font-size: 11px;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: 0.02em;
  }
  .trace-block-stats {
    font-family: var(--mono-font);
    font-size: 11px;
    color: var(--text-tertiary);
  }

  .trace-steps-container {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .trace-step-row {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 4px 0;
  }
  .trace-step-label {
    font-family: var(--mono-font);
    font-size: 11px;
    font-weight: 500;
    color: var(--text-tertiary);
    min-width: 48px;
    padding-top: 3px;
    flex-shrink: 0;
  }
  .token-cluster {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
  }
  .token-pill {
    padding: 3px 8px;
    background: var(--token-bg);
    border: 1px solid var(--token-border);
    border-radius: 4px;
    font-family: var(--mono-font);
    font-size: 12px;
    color: var(--text-secondary);
    display: inline-flex;
    align-items: center;
    white-space: pre-wrap;
    word-break: break-all;
    transition: all 0.15s ease;
  }
  .token-pill.verified {
    color: var(--text-primary);
    border-color: rgba(255, 255, 255, 0.2);
    background: var(--surface-elevated);
  }
  .token-pill.active-frontier {
    border-color: rgba(56, 189, 248, 0.5);
    color: #38bdf8;
    background: rgba(56, 189, 248, 0.08);
  }

  .btn-execute {
    background: var(--accent);
    color: var(--accent-contrast);
    border: none;
    font-family: var(--sans-font);
    font-size: 13px;
    font-weight: 500;
    padding: 10px 18px;
    min-height: 38px;
    border-radius: 5px;
    cursor: pointer;
    flex-shrink: 0;
    align-self: stretch;
    transition: opacity 0.15s ease;
  }
  @media (min-width: 640px) {
    .btn-execute { align-self: flex-end; width: auto; }
  }
  .btn-execute:hover { opacity: 0.9; }
  .btn-execute:disabled { opacity: 0.35; cursor: not-allowed; }
</style>
</head>
<body>

<div class="app-frame">
  <!-- Top Navigation Header -->
  <header class="top-header">
    <div class="header-left">
      <div class="header-brand" style="display:flex; align-items:center; gap:8px;">
        <span class="brand-title">BFG-Diff</span>
        <span class="brand-tag">v2</span>
      </div>
    </div>

    <div class="header-actions">
      <nav class="segmented-control">
        <button class="seg-btn active" id="btnTabChat" onclick="navigateTab('chat')" title="Interactive Chat">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
          <span>Chat</span>
        </button>
        <button class="seg-btn" id="btnTabVisualizer" onclick="navigateTab('visualizer')" title="Block Trace Visualizer">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
          <span>Visualizer</span>
        </button>
        <button class="seg-btn" id="btnTabManual" onclick="navigateTab('manual')" title="Architecture & Settings Manual">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path></svg>
          <span>Manual</span>
        </button>
      </nav>
      <button class="icon-btn" onclick="openSettingsModal()" title="Settings" aria-label="Settings">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
      </button>
    </div>
  </header>

  <!-- Clean Modal Settings Popup -->
  <div class="modal-overlay" id="settingsModal" onclick="onModalOverlayClick(event)">
    <div class="modal-card">
      <div class="modal-header">
        <span class="modal-title">Engine Parameters</span>
        <button class="modal-close-btn" onclick="closeSettingsModal()">✕</button>
      </div>
      <div class="modal-body">
        <div class="modal-field">
          <label>Engine Preset</label>
          <div class="custom-menu-wrapper" id="customPresetWrapper">
            <div class="custom-menu-trigger" id="customPresetTrigger" onclick="toggleCustomPresetMenu()">
              <span id="customPresetSelectedLabel">Balanced</span>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
            </div>
            <div class="custom-menu-options" id="customPresetOptions">
              <div class="custom-menu-item selected" onclick="pickCustomPreset('balanced', 'Balanced Standard', this)">
                <div>
                  <div style="font-weight:500;">Balanced Standard</div>
                  <div class="item-sub">T=0.3 · 4 passes · calibrated lookahead</div>
                </div>
              </div>
              <div class="custom-menu-item" onclick="pickCustomPreset('snappy', 'Snappy Low-Latency', this)">
                <div>
                  <div style="font-weight:500;">Snappy Low-Latency</div>
                  <div class="item-sub">T=0.2 · 2 passes · immediate stop termination</div>
                </div>
              </div>
              <div class="custom-menu-item" onclick="pickCustomPreset('branching', 'Multi-Branch Speculative', this)">
                <div>
                  <div style="font-weight:500;">Multi-Branch Speculative</div>
                  <div class="item-sub">2-branch micro-tree · parallel runner-up exploration</div>
                </div>
              </div>
              <div class="custom-menu-item" onclick="pickCustomPreset('code', 'Code & Strict Syntax', this)">
                <div>
                  <div style="font-weight:500;">Code & Strict Syntax</div>
                  <div class="item-sub">T=0.0 · strict tau=0.4 · zero hallucinations</div>
                </div>
              </div>
              <div class="custom-menu-item" onclick="pickCustomPreset('infill', 'High-Throughput Infilling', this)">
                <div>
                  <div style="font-weight:500;">High-Throughput Infilling</div>
                  <div class="item-sub">C=48 · sub-blocks=16 · parallel span completion</div>
                </div>
              </div>
              <div class="custom-menu-item" onclick="pickCustomPreset('creative', 'Creative Narrative', this)">
                <div>
                  <div style="font-weight:500;">Creative Narrative</div>
                  <div class="item-sub">T=0.7 · anti-loop frequency suppression</div>
                </div>
              </div>
              <div class="custom-menu-item" onclick="pickCustomPreset('greedy', 'Pure Deterministic', this)">
                <div>
                  <div style="font-weight:500;">Pure Argmax</div>
                  <div class="item-sub">T=0.0 · exact greedy decoding for benchmarks</div>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div class="modal-grid">
          <div class="modal-field">
            <label>Mode</label>
            <select id="cfgMode">
              <option value="chat" selected>ChatML</option>
              <option value="completion">Raw</option>
            </select>
          </div>
          <div class="modal-field">
            <label>Temperature</label>
            <input type="number" id="cfgTemp" value="0.3" min="0.0" max="1.5" step="0.1">
          </div>
          <div class="modal-field">
            <label>Canvas Tokens</label>
            <input type="number" id="cfgCanvas" value="32" min="8" max="256" step="8">
          </div>
          <div class="modal-field">
            <label>Sub-Block Size</label>
            <input type="number" id="cfgSubCanvas" value="8" min="4" max="32" step="4">
          </div>
          <div class="modal-field">
            <label>Diffusion Sub-Steps</label>
            <input type="number" id="cfgSubSteps" value="4" min="1" max="16" step="1">
          </div>
          <div class="modal-field">
            <label>Lookahead Tau</label>
            <input type="number" id="cfgTau" value="0.8" min="0.1" max="2.0" step="0.1">
          </div>
          <div class="modal-field" style="grid-column: span 2; display:flex; flex-direction:row; align-items:center; justify-content:space-between; padding-top:4px;">
            <label style="cursor:pointer;" for="cfgBranching">2-Branch Micro-Tree (Pillar 4)</label>
            <input type="checkbox" id="cfgBranching" style="width:18px; height:18px; cursor:pointer; accent-color:var(--accent);">
          </div>
        </div>
      </div>
      <div class="modal-footer">
        <button class="icon-btn" onclick="clearConversation()" style="color:#ef4444; border-color:#3f1414;">Reset Chat History</button>
        <button class="btn-execute" onclick="closeSettingsModal()" style="align-self:center; min-height:32px; padding:6px 14px;">Done</button>
      </div>
    </div>
  </div>

  <!-- TAB 1: FULLSCREEN CHAT (ChatGPT Style) -->
  <section id="tabChatView">
    <div class="chat-stream-viewport" id="chatViewport">
      <div class="chat-conversation-flow" id="conversationFlow">
        <div class="chat-welcome" id="chatWelcome">
          <div class="welcome-logo">BD</div>
          <div class="welcome-title">BFG-Diff Discrete Diffusion</div>
          <div class="welcome-sub">Ask a question or brainstorm ideas. Previous turns are remembered across messages.</div>
        </div>
      </div>
    </div>

    <!-- Floating Bottom Composer -->
    <div class="chat-composer-floating">
      <div class="composer-capsule">
        <textarea class="chat-prompt-field" id="chatInputText" placeholder="Message BFG-Diff..." rows="1"></textarea>
        <button class="submit-circle-btn" id="chatSubmitCircleBtn" onclick="transmitChatMessage()">
          <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
        </button>
      </div>
    </div>
  </section>

  <!-- TAB 2: VISUALIZER (Clean Block Unmasking) -->
  <section id="tabVisualizerView">
    <div class="vis-container">
      <div class="vis-card">
        <div class="vis-prompt-deck">
          <textarea class="vis-textarea" id="visPrompt" rows="3">Explain the difference between a process and a thread in one short paragraph.</textarea>
          <button class="btn-execute" id="btnRunVis" onclick="runVisualizerComparison()">Execute & Compare</button>
        </div>
      </div>

      <div class="vis-split">
        <div class="vis-card">
          <div class="vis-header">
            <span class="vis-title">BFG-Diff (Verify & Diffuse)</span>
            <span class="vis-meta" id="visBfgMeta">0.0 TPS</span>
          </div>
          <div class="vis-output" id="visBfgOutput">Awaiting execution...</div>
        </div>
        <div class="vis-card">
          <div class="vis-header">
            <span class="vis-title">Causal Baseline (generate)</span>
            <span class="vis-meta" id="visCausalMeta">0.0 TPS</span>
          </div>
          <div class="vis-output" id="visCausalOutput">Awaiting execution...</div>
        </div>
      </div>

      <div class="vis-card">
        <div class="vis-header">
          <span class="vis-title">Block Unmasking Trace</span>
        </div>
        <div class="trace-feed" id="visTraceFeed">
          <div style="font-family:var(--mono-font); font-size:12px; color:var(--text-tertiary); text-align:center; padding:16px;">
            Execute a prompt to inspect speculative lookahead stages.
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- TAB 3: SYSTEM MANUAL & ARCHITECTURE REFERENCE -->
  <section id="tabManualView" style="display: none; flex: 1; overflow-y: auto; padding: 20px 16px 80px 16px; justify-content: center;">
    <div style="max-width: 860px; width: 100%; margin: 0 auto; display: flex; flex-direction: column; gap: 20px;">

      <div class="vis-card">
        <div class="vis-header">
          <span class="vis-title">Architecture Overview</span>
          <span class="vis-meta">BFG-Diff v2 · Fused Staircase Engine</span>
        </div>
        <div style="padding: 18px 20px; font-size: 13.5px; line-height: 1.65; color: var(--text-primary); display: flex; flex-direction: column; gap: 12px;">
          <p>
            Standard language models generate text strictly sequentially (one forward pass per token). 
            <strong>BFG-Diff</strong> transforms a pretrained causal transformer into a discrete block-diffusion engine, evaluating and verifying multiple tokens in parallel blocks across 2 to 4 forward steps.
          </p>
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; margin-top: 6px;">
            <div style="background: var(--bg); border: 1px solid var(--border); padding: 12px; border-radius: 6px;">
              <div style="font-family: var(--mono-font); font-size: 11px; font-weight: 600; color: #38bdf8; margin-bottom: 4px;">Zone A: Causal Verification</div>
              <p style="font-size: 12.5px; color: var(--text-secondary);">
                Evaluates prior draft tokens under a strictly causal attention mask. Completely isolated from trailing mask tokens (zero attention leakage).
              </p>
            </div>
            <div style="background: var(--bg); border: 1px solid var(--border); padding: 12px; border-radius: 6px;">
              <div style="font-family: var(--mono-font); font-size: 11px; font-weight: 600; color: #a855f7; margin-bottom: 4px;">Zone B: Lookahead Diffusion</div>
              <p style="font-size: 12.5px; color: var(--text-secondary);">
                Simultaneously diffuses future tokens on lookahead positions using bidirectional attention in the exact same forward pass.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div class="vis-card">
        <div class="vis-header">
          <span class="vis-title">Engine Parameters & Controls Guide</span>
          <span class="vis-meta">Fine-Tuning Decoding Mechanics</span>
        </div>
        <div style="padding: 16px 20px; font-size: 13px; line-height: 1.6; color: var(--text-secondary); display: flex; flex-direction: column; gap: 16px;">
          <div>
            <div style="font-family: var(--mono-font); font-size: 12px; font-weight: 600; color: var(--text-primary); margin-bottom: 4px;">
              1. Canvas Tokens (<code style="color:#38bdf8;">canvas_len</code>)
            </div>
            <p>
              The maximum total length of output text to generate (e.g., 24, 32, 48 tokens). When stop tokens like <code>&lt;|im_end|&gt;</code> are carrier-verified in chat mode, generation terminates early to save latency.
            </p>
          </div>

          <div>
            <div style="font-family: var(--mono-font); font-size: 12px; font-weight: 600; color: var(--text-primary); margin-bottom: 4px;">
              2. Sub-Block Size (<code style="color:#38bdf8;">sub_canvas_len</code>)
            </div>
            <p>
              The size of the sliding parallel attention window. A size of <strong>8</strong> offers tight left-carrier grounding and lowest repetition. Larger windows (<strong>16</strong> or <strong>32</strong>) increase throughput for span infilling by batching more positions into GPU matrix multiplications.
            </p>
          </div>

          <div>
            <div style="font-family: var(--mono-font); font-size: 12px; font-weight: 600; color: var(--text-primary); margin-bottom: 4px;">
              3. Diffusion Sub-Steps (<code style="color:#38bdf8;">sub_steps</code>)
            </div>
            <p>
              The number of iterative forward passes allowed per sub-block. 
              <strong>4 steps</strong> provides maximum token precision and grammar stability. 
              <strong>2 steps</strong> cuts latency in half for snappy, sub-second responses.
            </p>
          </div>

          <div>
            <div style="font-family: var(--mono-font); font-size: 12px; font-weight: 600; color: var(--text-primary); margin-bottom: 4px;">
              4. Lookahead Tau (<code style="color:#38bdf8;">tau</code>)
            </div>
            <p>
              Calibrated energy difference tolerance: <code>(z_top - z_draft) &le; &tau;</code>. 
              Controls how confident the model must be to accept speculative lookahead tokens. Lower values (<strong>0.4</strong>) enforce strict conservative acceptance (ideal for code). Higher values (<strong>0.8 &ndash; 1.0</strong>) accept more tokens in parallel.
            </p>
          </div>

          <div>
            <div style="font-family: var(--mono-font); font-size: 12px; font-weight: 600; color: var(--text-primary); margin-bottom: 4px;">
              5. 2-Branch Micro-Tree Verification (<code style="color:#38bdf8;">enable_branching</code> &middot; Pillar 4)
            </div>
            <p>
              When the primary draft token has a narrow margin over runner-up candidates, the engine forks Zone A into two parallel hypothesis branches evaluated under a 2D tree mask. If branch 2 wins, its KV states are copied in-place, preventing draft rejections.
            </p>
          </div>
        </div>
      </div>

      <div class="vis-card">
        <div class="vis-header">
          <span class="vis-title">Specialized Presets Reference</span>
          <span class="vis-meta">Pre-Calibrated Profiles</span>
        </div>
        <div style="padding: 16px 20px; font-size: 13px; color: var(--text-secondary); display: flex; flex-direction: column; gap: 14px;">
          <table style="width: 100%; border-collapse: collapse; font-size: 12px; text-align: left;">
            <thead>
              <tr style="border-bottom: 1px solid var(--border); color: var(--text-primary); font-family: var(--mono-font);">
                <th style="padding: 8px 6px;">Preset</th>
                <th style="padding: 8px 6px;">Temp</th>
                <th style="padding: 8px 6px;">Steps</th>
                <th style="padding: 8px 6px;">&tau;</th>
                <th style="padding: 8px 6px;">Best Use Case</th>
              </tr>
            </thead>
            <tbody>
              <tr style="border-bottom: 1px solid var(--border-subtle);">
                <td style="padding: 8px 6px; font-weight: 500; color: var(--text-primary);">Balanced Standard</td>
                <td style="padding: 8px 6px; font-family: var(--mono-font);">0.3</td>
                <td style="padding: 8px 6px; font-family: var(--mono-font);">4</td>
                <td style="padding: 8px 6px; font-family: var(--mono-font);">0.8</td>
                <td style="padding: 8px 6px;">General dialogue, high fluency, low repetition.</td>
              </tr>
              <tr style="border-bottom: 1px solid var(--border-subtle);">
                <td style="padding: 8px 6px; font-weight: 500; color: var(--text-primary);">Snappy Low-Latency</td>
                <td style="padding: 8px 6px; font-family: var(--mono-font);">0.2</td>
                <td style="padding: 8px 6px; font-family: var(--mono-font);">2</td>
                <td style="padding: 8px 6px; font-family: var(--mono-font);">0.8</td>
                <td style="padding: 8px 6px;">Sub-second answers with aggressive stop termination.</td>
              </tr>
              <tr style="border-bottom: 1px solid var(--border-subtle);">
                <td style="padding: 8px 6px; font-weight: 500; color: var(--text-primary);">Multi-Branch (Pillar 4)</td>
                <td style="padding: 8px 6px; font-family: var(--mono-font);">0.2</td>
                <td style="padding: 8px 6px; font-family: var(--mono-font);">4</td>
                <td style="padding: 8px 6px; font-family: var(--mono-font);">0.9</td>
                <td style="padding: 8px 6px;">Speculative 2-branch micro-tree exploration.</td>
              </tr>
              <tr style="border-bottom: 1px solid var(--border-subtle);">
                <td style="padding: 8px 6px; font-weight: 500; color: var(--text-primary);">Code & Strict Syntax</td>
                <td style="padding: 8px 6px; font-family: var(--mono-font);">0.0</td>
                <td style="padding: 8px 6px; font-family: var(--mono-font);">4</td>
                <td style="padding: 8px 6px; font-family: var(--mono-font);">0.4</td>
                <td style="padding: 8px 6px;">JSON, Python, brackets, and deterministic code logic.</td>
              </tr>
              <tr style="border-bottom: 1px solid var(--border-subtle);">
                <td style="padding: 8px 6px; font-weight: 500; color: var(--text-primary);">High-Throughput Infilling</td>
                <td style="padding: 8px 6px; font-family: var(--mono-font);">0.1</td>
                <td style="padding: 8px 6px; font-family: var(--mono-font);">4</td>
                <td style="padding: 8px 6px; font-family: var(--mono-font);">0.8</td>
                <td style="padding: 8px 6px;">Parallel span completion (200+ TPS on GPU).</td>
              </tr>
              <tr style="border-bottom: 1px solid var(--border-subtle);">
                <td style="padding: 8px 6px; font-weight: 500; color: var(--text-primary);">Creative Narrative</td>
                <td style="padding: 8px 6px; font-family: var(--mono-font);">0.7</td>
                <td style="padding: 8px 6px; font-family: var(--mono-font);">4</td>
                <td style="padding: 8px 6px; font-family: var(--mono-font);">1.0</td>
                <td style="padding: 8px 6px;">Maximum lexical variety and open-ended writing.</td>
              </tr>
              <tr>
                <td style="padding: 8px 6px; font-weight: 500; color: var(--text-primary);">Pure Argmax</td>
                <td style="padding: 8px 6px; font-family: var(--mono-font);">0.0</td>
                <td style="padding: 8px 6px; font-family: var(--mono-font);">4</td>
                <td style="padding: 8px 6px; font-family: var(--mono-font);">0.8</td>
                <td style="padding: 8px 6px;">Exact greedy decoding for formal benchmarks.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

    </div>
  </section>
</div>

<script>
let currentActiveTab = 'chat';
let sessionChatHistory = [
  { role: "system", content: "You are a concise, helpful assistant powered by BFG-Diff v2 discrete diffusion." }
];

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function navigateTab(tab) {
  currentActiveTab = tab;
  document.getElementById('btnTabChat').className = 'seg-btn' + (tab === 'chat' ? ' active' : '');
  document.getElementById('btnTabVisualizer').className = 'seg-btn' + (tab === 'visualizer' ? ' active' : '');
  document.getElementById('btnTabManual').className = 'seg-btn' + (tab === 'manual' ? ' active' : '');
  document.getElementById('tabChatView').style.display = tab === 'chat' ? 'flex' : 'none';
  document.getElementById('tabVisualizerView').style.display = tab === 'visualizer' ? 'block' : 'none';
  document.getElementById('tabManualView').style.display = tab === 'manual' ? 'flex' : 'none';
}

function openSettingsModal() {
  document.getElementById('settingsModal').classList.add('open');
}

function closeSettingsModal() {
  document.getElementById('settingsModal').classList.remove('open');
}

function onModalOverlayClick(e) {
  if (e.target.id === 'settingsModal') {
    closeSettingsModal();
  }
}


function toggleCustomPresetMenu() {
  const opts = document.getElementById('customPresetOptions');
  const trigger = document.getElementById('customPresetTrigger');
  opts.classList.toggle('open');
  trigger.classList.toggle('active');
}

function pickCustomPreset(val, label, el) {
  document.getElementById('customPresetSelectedLabel').innerText = label;
  document.querySelectorAll('.custom-menu-item').forEach(item => item.classList.remove('selected'));
  if (el) el.classList.add('selected');
  document.getElementById('customPresetOptions').classList.remove('open');
  document.getElementById('customPresetTrigger').classList.remove('active');
  onSelectPreset(val);
}

// Close custom menu if clicking outside
document.addEventListener('click', function(e) {
  const wrapper = document.getElementById('customPresetWrapper');
  if (wrapper && !wrapper.contains(e.target)) {
    const opts = document.getElementById('customPresetOptions');
    const trigger = document.getElementById('customPresetTrigger');
    if (opts) opts.classList.remove('open');
    if (trigger) trigger.classList.remove('active');
  }
});

function onSelectPreset(preset) {
  const cLen = document.getElementById('cfgCanvas');
  const scLen = document.getElementById('cfgSubCanvas');
  const scSteps = document.getElementById('cfgSubSteps');
  const scTemp = document.getElementById('cfgTemp');
  const scTau = document.getElementById('cfgTau');
  const scBranch = document.getElementById('cfgBranching');

  if (preset === 'balanced') {
    cLen.value = 32; scLen.value = 8; scSteps.value = 4; scTemp.value = 0.3; scTau.value = 0.8; scBranch.checked = false;
  } else if (preset === 'snappy') {
    cLen.value = 24; scLen.value = 8; scSteps.value = 2; scTemp.value = 0.2; scTau.value = 0.8; scBranch.checked = false;
  } else if (preset === 'branching') {
    cLen.value = 32; scLen.value = 8; scSteps.value = 4; scTemp.value = 0.2; scTau.value = 0.9; scBranch.checked = true;
  } else if (preset === 'code') {
    cLen.value = 32; scLen.value = 8; scSteps.value = 4; scTemp.value = 0.0; scTau.value = 0.4; scBranch.checked = false;
  } else if (preset === 'infill') {
    cLen.value = 48; scLen.value = 16; scSteps.value = 4; scTemp.value = 0.1; scTau.value = 0.8; scBranch.checked = false;
  } else if (preset === 'creative') {
    cLen.value = 48; scLen.value = 8; scSteps.value = 4; scTemp.value = 0.7; scTau.value = 1.0; scBranch.checked = false;
  } else if (preset === 'greedy') {
    cLen.value = 32; scLen.value = 8; scSteps.value = 4; scTemp.value = 0.0; scTau.value = 0.8; scBranch.checked = false;
  }
}

function clearConversation() {
  sessionChatHistory = [
    { role: "system", content: "You are a concise, helpful assistant powered by BFG-Diff v2 discrete diffusion." }
  ];
  document.getElementById('conversationFlow').innerHTML = `
    <div class="chat-welcome" id="chatWelcome">
      <div class="welcome-logo">BD</div>
      <div class="welcome-title">BFG-Diff Discrete Diffusion</div>
      <div class="welcome-sub">Ask a question or brainstorm ideas. Previous turns are remembered across messages.</div>
    </div>
  `;
  closeSettingsModal();
}

// Auto-grow textarea
const chatInputArea = document.getElementById('chatInputText');
chatInputArea.addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 160) + 'px';
});

chatInputArea.addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    transmitChatMessage();
  }
});

async function transmitChatMessage() {
  const inputEl = document.getElementById('chatInputText');
  const userText = inputEl.value.trim();
  if (!userText) return;

  const btn = document.getElementById('chatSubmitCircleBtn');
  btn.disabled = true;
  inputEl.disabled = true;

  const welcomeNotice = document.getElementById('chatWelcome');
  if (welcomeNotice) welcomeNotice.remove();

  sessionChatHistory.push({ role: "user", content: userText });
  inputEl.value = '';
  inputEl.style.height = 'auto';

  const flow = document.getElementById('conversationFlow');
  const viewport = document.getElementById('chatViewport');

  const userBlock = document.createElement('div');
  userBlock.className = 'message-block user';
  userBlock.innerHTML = `
    <div class="message-content">
      <div class="user-bubble">${escapeHtml(userText)}</div>
    </div>
  `;
  flow.appendChild(userBlock);

  const botBlock = document.createElement('div');
  botBlock.className = 'message-block assistant';
  const replySpanId = 'replySpan_' + Date.now();
  const replyMetaId = 'replyMeta_' + Date.now();
  botBlock.innerHTML = `
    <div class="message-content" style="max-width:100%;">
      <div class="assistant-stream" id="${replySpanId}">Diffusing...</div>
      <div class="message-stats" id="${replyMetaId}"></div>
    </div>
  `;
  flow.appendChild(botBlock);
  viewport.scrollTop = viewport.scrollHeight;

  const replySpan = document.getElementById(replySpanId);
  const replyMeta = document.getElementById(replyMetaId);

  const payload = {
    messages: sessionChatHistory,
    canvas_len: parseInt(document.getElementById('cfgCanvas').value) || 32,
    sub_canvas_len: parseInt(document.getElementById('cfgSubCanvas').value) || 8,
    sub_steps: parseInt(document.getElementById('cfgSubSteps').value) || 4,
    temp: parseFloat(document.getElementById('cfgTemp').value) || 0.3,
    branching: document.getElementById('cfgBranching').checked,
    tau: parseFloat(document.getElementById('cfgTau').value) || 0.8,
  };

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error("HTTP error " + res.status);

    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let raw = '';
    let completedReply = '';

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
        const bodyStr = line.substring(5).trim();
        if (!bodyStr) continue;

        const evt = JSON.parse(bodyStr);
        if (evt.type === 'chat_step') {
          replySpan.innerText = evt.text;
          replyMeta.innerText = evt.tps.toFixed(1) + " TPS · " + evt.elapsed.toFixed(2) + "s";
          viewport.scrollTop = viewport.scrollHeight;
        } else if (evt.type === 'chat_done') {
          completedReply = evt.text;
          replySpan.innerText = evt.text;
          replyMeta.innerText = `${evt.tps.toFixed(1)} TPS · ${evt.tokens} tokens · ${evt.elapsed.toFixed(2)}s`;
          sessionChatHistory.push({ role: "assistant", content: completedReply });
          viewport.scrollTop = viewport.scrollHeight;
        }
      }
    }
  } catch (err) {
    console.error("Chat failure:", err);
    replySpan.innerText = "Error: " + err.message;
  } finally {
    btn.disabled = false;
    inputEl.disabled = false;
    inputEl.focus();
  }
}

// Visualizer Handler
async function runVisualizerComparison() {
  const btn = document.getElementById('btnRunVis');
  btn.disabled = true;
  btn.innerText = 'Executing...';

  const bfgOut = document.getElementById('visBfgOutput');
  const causalOut = document.getElementById('visCausalOutput');
  const bfgMeta = document.getElementById('visBfgMeta');
  const causalMeta = document.getElementById('visCausalMeta');
  const traceFeed = document.getElementById('visTraceFeed');

  bfgOut.innerText = 'Unmasking active blocks...';
  causalOut.innerText = 'Generating token stream...';
  bfgMeta.innerText = '0.0 TPS';
  causalMeta.innerText = '0.0 TPS';
  traceFeed.innerHTML = '';

  const payload = {
    prompt: document.getElementById('visPrompt').value,
    mode: document.getElementById('cfgMode').value,
    canvas_len: parseInt(document.getElementById('cfgCanvas').value),
    sub_canvas_len: parseInt(document.getElementById('cfgSubCanvas').value),
    sub_steps: parseInt(document.getElementById('cfgSubSteps').value),
    temp: parseFloat(document.getElementById('cfgTemp').value),
    branching: document.getElementById('cfgBranching').checked,
    tau: parseFloat(document.getElementById('cfgTau').value) || 0.8,
  };

  const blockMap = {};
  let causalBuffer = '';

  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error("HTTP " + res.status);

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
          causalOut.innerText = causalBuffer;
          causalMeta.innerText = evt.tps.toFixed(1) + " TPS (" + evt.elapsed.toFixed(2) + "s)";
        } else if (evt.type === 'causal_done') {
          causalOut.innerText = evt.causal.text;
          causalMeta.innerText = evt.causal.tps.toFixed(1) + " TPS (" + evt.causal.elapsed.toFixed(2) + "s)";
        } else if (evt.type === 'bfg_step') {
          bfgOut.innerText = evt.text;
          bfgMeta.innerText = evt.tps.toFixed(1) + " TPS (" + evt.elapsed.toFixed(2) + "s)";

          if (!blockMap[evt.block_idx]) {
            const bDiv = document.createElement('div');
            bDiv.className = 'trace-block';
            bDiv.innerHTML = `
              <div class="trace-block-header">
                <span class="trace-block-tag">Sub-Block ${evt.block_idx + 1}</span>
                <span class="trace-block-stats" id="blockStats_${evt.block_idx}"></span>
              </div>
              <div class="trace-steps-container" id="stepsContainer_${evt.block_idx}"></div>
            `;
            traceFeed.appendChild(bDiv);
            blockMap[evt.block_idx] = bDiv;
          }

          const stepsBox = document.getElementById('stepsContainer_' + evt.block_idx);
          const statsBox = document.getElementById('blockStats_' + evt.block_idx);
          if (statsBox) {
            statsBox.innerText = `${evt.tps.toFixed(1)} TPS · ${evt.elapsed.toFixed(2)}s`;
          }

          const row = document.createElement('div');
          row.className = 'trace-step-row';

          const chipsHtml = evt.tokens.map((tok, idx) => {
            const isFrontier = (idx === evt.tokens.length - 1);
            const cls = isFrontier ? 'token-pill active-frontier' : 'token-pill verified';
            return `<span class="${cls}">${escapeHtml(tok)}</span>`;
          }).join('');

          row.innerHTML = `
            <span class="trace-step-label">Pass ${evt.step}</span>
            <div class="token-cluster">${chipsHtml}</div>
          `;
          stepsBox.appendChild(row);
        } else if (evt.type === 'bfg_done') {
          bfgOut.innerText = evt.bfg.text;
          bfgMeta.innerText = evt.bfg.tps.toFixed(1) + " TPS (" + evt.bfg.elapsed.toFixed(2) + "s)";
        }
      }
    }
  } catch (err) {
    console.error("Visualizer error:", err);
    bfgOut.innerText = "Error: " + err.message;
  } finally {
    btn.disabled = false;
    btn.innerText = 'Execute & Compare';
  }
}
// Initialize default preset on page load
window.addEventListener('DOMContentLoaded', () => {
  onSelectPreset('balanced');
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
            branching = bool(req.get("branching", False))
            tau = float(req.get("tau", 0.8))

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
                enable_branching=branching,
                tau=tau,
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
            branching = bool(req.get("branching", False))
            tau = float(req.get("tau", 0.8))

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
                enable_branching=branching,
                tau=tau,
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
