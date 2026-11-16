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
from sampler import sample_canvas_block


def load_bfg_model(checkpoint_path, device="cpu"):
    """Load base Qwen model and apply checkpoint adapters."""
    device = torch.device(device)
    model, tok, mask_id = prepare("Qwen/Qwen2.5-0.5B", device=device, dtype=torch.float32)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    ck_file = Path(checkpoint_path)
    if not ck_file.exists():
        fallback = Path("runs/bfg-shifted/last.pt")
        if fallback.exists():
            print(f"Notice: {checkpoint_path} not found. Falling back to {fallback}")
            ck_file = fallback
        else:
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")

    print(f"Loading checkpoint from {ck_file}...")
    ck = torch.load(ck_file, map_location="cpu")
    cfg = ck.get("lora_config", {})
    model = apply_lora(
        model,
        cfg.get("r", 16),
        cfg.get("alpha", 32),
        cfg.get("dropout", 0.05),
        cfg.get("targets", LORA_TARGETS),
    )
    if "lora" in ck:
        set_peft_model_state_dict(model, ck["lora"])
    if "mask_emb" in ck:
        with torch.no_grad():
            model.get_input_embeddings().weight[mask_id] = ck["mask_emb"].to(device)
    model.eval()
    return model, tok, mask_id


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
    """Generate tokens using BFG-Diff while capturing block-by-block and step-by-step traces."""
    with torch.no_grad():
        prefix_ids = tok.encode(prompt_text, return_tensors="pt").to(model.device)
    k_sub = sub_steps
    cur_prefix = prefix_ids
    cur_keep = None
    generated_blocks = []
    rem = canvas_len
    trace = []
    block_idx = 0
    start_time = time.perf_counter()

    # Pre-encode prefix into KV cache
    from model import mask_position_ids, prefill_prefix

    kv = prefill_prefix(model, cur_prefix, cur_keep)

    while rem > 0:
        c_sub = min(sub_canvas_len, rem)
        B = cur_prefix.shape[0]
        canvas = torch.full((B, c_sub), mask_id, dtype=torch.long, device=cur_prefix.device)
        block_steps_trace = []

        # Run step-by-step decoding within sub-block
        frozen = canvas != mask_id
        decoded = torch.zeros_like(frozen)
        prefix_len = cur_prefix.shape[1]

        # Initial all-mask state
        init_tokens = [tok.decode([int(t)]) if int(t) != mask_id else "░" for t in canvas[0]]
        block_steps_trace.append({"step": 0, "tokens": init_tokens, "text": "".join(init_tokens)})

        for s in range(1, k_sub + 1):
            from model import canvas_step_with_cache

            logits = canvas_step_with_cache(
                model, cur_prefix[:, -1:], canvas, kv, prefix_len, cur_keep
            )
            kv.crop(-(1 + c_sub))
            z = logits.float()
            top2 = z.topk(2, dim=-1)
            margin = top2.values[..., 0] - top2.values[..., 1]

            keep_token = frozen.clone()
            if s == k_sub:
                keep_token |= decoded

            if carrier_priority and s < k_sub:
                has_carrier = torch.zeros_like(frozen)
                has_carrier[:, 0] = True
                has_carrier[:, 1:] = keep_token[:, :-1] | frozen[:, :-1]
                anchor = margin >= 2.5
                is_priority = has_carrier | anchor
                rank = margin + torch.where(is_priority, 10.0, 0.0)
            else:
                rank = margin.clone()

            # Cross-step repulsion
            left = keep_token[:, :-1]
            right = keep_token[:, 1:]
            idx_rep = torch.where(left, canvas[:, :-1], torch.zeros_like(canvas[:, :-1]))
            rep = z[:, 1:].scatter(-1, idx_rep.unsqueeze(-1), -float("inf"))
            z[:, 1:] = torch.where(left.unsqueeze(-1), rep, z[:, 1:])
            idx_rep = torch.where(right, canvas[:, 1:], torch.zeros_like(canvas[:, 1:]))
            rep = z[:, :-1].scatter(-1, idx_rep.unsqueeze(-1), -float("inf"))
            z[:, :-1] = torch.where(right.unsqueeze(-1), rep, z[:, :-1])
            if prefix_len:
                last_t = cur_prefix[:, -1:]
                rep = z[:, :1].scatter(-1, last_t.unsqueeze(-1), -float("inf"))
                z[:, :1] = torch.where(~keep_token[:, :1].unsqueeze(-1), rep, z[:, :1])

            if temperature > 0.0:
                vals, top_idx = z.topk(min(50, z.shape[-1]), dim=-1)
                u = torch.rand_like(vals).clamp(1e-9, 1.0 - 1e-9)
                pick_samp = (vals / temperature - torch.log(-torch.log(u))).argmax(-1, keepdim=True)
                sampled = top_idx.gather(-1, pick_samp).squeeze(-1)
            else:
                sampled = top2.indices[..., 0]

            if s == k_sub:
                row = canvas[0].clone()
                for j in (canvas[0] == mask_id).nonzero().flatten().tolist():
                    tok_id = int(sampled[0, j])
                    prev_id = (
                        int(row[j - 1])
                        if j > 0
                        else (int(cur_prefix[0, -1]) if prefix_len else None)
                    )
                    if prev_id is not None and tok_id == prev_id:
                        cand = z[0, j].clone()
                        cand[prev_id] = -float("inf")
                        tok_id = int(cand.argmax())
                    row[j] = tok_id
                canvas[0] = row
            else:
                n_mutable = c_sub - int(frozen[0].sum())
                budget = min(max(0, int(s / k_sub * n_mutable)), n_mutable)
                keep = keep_token[0] & ~frozen[0]
                pick = set(keep.nonzero().flatten().tolist())

                if carrier_priority:
                    # 1. Concrete carriers
                    has_carrier_0 = has_carrier[0]
                    cand_c = [
                        j
                        for j in range(c_sub)
                        if has_carrier_0[j] and j not in pick and not frozen[0, j]
                    ]
                    cand_c.sort(key=lambda j: float(rank[0, j]), reverse=True)
                    for j in cand_c:
                        if len(pick) < budget:
                            pick.add(j)

                    # 2. Distant island anchors
                    while len(pick) < budget:
                        cand_is = [
                            j
                            for j in range(c_sub)
                            if j not in pick
                            and not frozen[0, j]
                            and margin[0, j] >= 2.5
                            and not (j > 0 and (j - 1) in pick)
                        ]
                        if not cand_is:
                            break
                        cand_is.sort(key=lambda j: float(rank[0, j]), reverse=True)
                        pick.add(cand_is[0])

                    # 3. Frontier advance
                    while len(pick) < budget:
                        cand_front = [
                            j
                            for j in range(c_sub)
                            if j not in pick and not frozen[0, j] and (j > 0 and (j - 1) in pick)
                        ]
                        if cand_front:
                            cand_front.sort(key=lambda j: float(rank[0, j]), reverse=True)
                            pick.add(cand_front[0])
                            continue
                        cand_rem = [j for j in range(c_sub) if j not in pick and not frozen[0, j]]
                        if not cand_rem:
                            break
                        cand_rem.sort(key=lambda j: float(rank[0, j]), reverse=True)
                        pick.add(cand_rem[0])
                else:
                    cand = (rank[0] > -float("inf")) & ~frozen[0] & ~keep
                    cand = cand.nonzero().flatten()
                    if cand.numel():
                        order = rank[0, cand].topk(min(budget - len(pick), cand.numel())).indices
                        for idx_c in cand[order].tolist():
                            pick.add(idx_c)

                row = torch.full_like(canvas[0], mask_id)
                pick_list = sorted(list(pick))
                if pick_list:
                    pick_t = torch.tensor(pick_list, dtype=torch.long, device=canvas.device)
                    row[pick_t] = sampled[0, pick_t]
                canvas[0] = row
                decoded[0] = row != mask_id

            step_tokens = [
                tok.decode([int(t)]) if int(t) != mask_id else "░" for t in canvas[0]
            ]
            block_steps_trace.append(
                {"step": s, "tokens": step_tokens, "text": "".join(step_tokens)}
            )

        block = canvas
        generated_blocks.append(block)
        trace.append(
            {
                "block_idx": block_idx,
                "steps": block_steps_trace,
                "final_text": tok.decode(block[0], skip_special_tokens=True),
            }
        )
        block_idx += 1
        rem -= c_sub

        # Advance KV
        if rem > 0:
            advance_tokens = torch.cat([cur_prefix[:, -1:], block[:, :-1]], dim=1)
            p_cached = kv.get_seq_length()
            pos_advance = torch.arange(
                p_cached, p_cached + c_sub, device=advance_tokens.device
            ).unsqueeze(0)
            model(
                input_ids=advance_tokens,
                past_key_values=kv,
                position_ids=pos_advance,
                use_cache=True,
            )

        cur_prefix = torch.cat([cur_prefix, block], dim=1)

        # Check for stop tokens if in chat mode
        decoded_so_far = tok.decode(torch.cat(generated_blocks, dim=1)[0])
        if stop_tokens and any(st in decoded_so_far for st in stop_tokens):
            break

    elapsed = time.perf_counter() - start_time
    total_tokens = sum(b.shape[1] for b in generated_blocks)
    full_output_ids = torch.cat(generated_blocks, dim=1)[0]
    full_text = tok.decode(full_output_ids, skip_special_tokens=True)
    tps = total_tokens / max(1e-5, elapsed)

    return {
        "text": full_text,
        "tokens": total_tokens,
        "elapsed": elapsed,
        "tps": tps,
        "trace": trace,
    }


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


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BFG-Diff Diffusion Visualizer & Chat</title>
<style>
  :root {
    --bg: #0f172a;
    --card: #1e293b;
    --border: #334155;
    --primary: #38bdf8;
    --text: #f1f5f9;
    --muted: #94a3b8;
    --mask: #475569;
    --accent: #a855f7;
  }
  body {
    background-color: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin: 0;
    padding: 20px;
    display: flex;
    justify-content: center;
  }
  .container {
    max-width: 900px;
    width: 100%;
  }
  header {
    margin-bottom: 24px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 16px;
  }
  h1 { margin: 0 0 8px 0; font-size: 24px; color: var(--primary); }
  p.subtitle { margin: 0; color: var(--muted); font-size: 14px; }
  .controls {
    background: var(--card);
    padding: 16px;
    border-radius: 8px;
    border: 1px solid var(--border);
    margin-bottom: 20px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
  }
  .control-group {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  label { font-size: 12px; color: var(--muted); font-weight: 600; }
  input, select, textarea, button {
    background: #090d16;
    border: 1px solid var(--border);
    color: var(--text);
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 14px;
  }
  textarea { resize: vertical; min-height: 70px; width: 100%; box-sizing: border-box; }
  button {
    background: var(--primary);
    color: #090d16;
    font-weight: bold;
    cursor: pointer;
    border: none;
    transition: opacity 0.2s;
  }
  button:hover { opacity: 0.9; }
  .prompt-card {
    background: var(--card);
    padding: 16px;
    border-radius: 8px;
    border: 1px solid var(--border);
    margin-bottom: 20px;
  }
  .results { display: flex; flex-direction: column; gap: 20px; }
  .comparison-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }
  @media (max-width: 768px) {
    .comparison-grid { grid-template-columns: 1fr; }
  }
  .model-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
  }
  .model-title {
    font-size: 15px;
    font-weight: 600;
    color: var(--primary);
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
  }
  .metric-badge {
    background: rgba(56, 189, 248, 0.1);
    color: var(--primary);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 12px;
  }
  .output-box {
    background: #090d16;
    padding: 12px;
    border-radius: 6px;
    border: 1px solid var(--border);
    font-family: monospace;
    font-size: 14px;
    white-space: pre-wrap;
    line-height: 1.5;
  }
  .trace-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-top: 20px;
  }
  .trace-title { font-size: 16px; font-weight: 600; color: var(--accent); margin-bottom: 12px; }
  .block-trace { margin-bottom: 16px; border-left: 3px solid var(--accent); padding-left: 12px; }
  .step-line {
    font-family: monospace;
    font-size: 13px;
    padding: 4px 0;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .step-num { color: var(--muted); width: 60px; font-weight: bold; }
  .tokens-canvas { display: flex; gap: 4px; flex-wrap: wrap; }
  .token-chip {
    background: #1e293b;
    border: 1px solid var(--border);
    padding: 2px 6px;
    border-radius: 4px;
  }
  .token-chip.mask {
    background: rgba(71, 85, 105, 0.4);
    color: #64748b;
    border: 1px dashed var(--mask);
  }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>BFG-Diff Interactive Visualizer</h1>
    <p class="subtitle">Right-Shifted Discrete Diffusion vs Causal Autoregression (Qwen2.5-0.5B)</p>
  </header>

  <div class="controls">
    <div class="control-group">
      <label>Mode</label>
      <select id="mode">
        <option value="completion" selected>Completion</option>
        <option value="chat">Chat (<|im_start|>)</option>
      </select>
    </div>
    <div class="control-group">
      <label>Total Canvas (C)</label>
      <input type="number" id="canvas_len" value="32" min="8" max="128" step="8">
    </div>
    <div class="control-group">
      <label>Sub-Block (C_sub)</label>
      <input type="number" id="sub_canvas_len" value="8" min="4" max="32" step="4">
    </div>
    <div class="control-group">
      <label>Steps / Block (k_sub)</label>
      <input type="number" id="sub_steps" value="4" min="1" max="16">
    </div>
    <div class="control-group">
      <label>Temperature</label>
      <input type="number" id="temp" value="0.0" min="0.0" max="1.0" step="0.1">
    </div>
  </div>

  <div class="prompt-card">
    <label style="display:block; margin-bottom:8px;">Input Prompt</label>
    <textarea id="prompt">Once upon a time, in a small coastal village, the fishermen noticed that</textarea>
    <div style="margin-top: 12px; display: flex; justify-content: flex-end;">
      <button id="runBtn" onclick="runGeneration()">Generate & Compare</button>
    </div>
  </div>

  <div class="results" id="resultsSection" style="display: none;">
    <div class="comparison-grid">
      <div class="model-card">
        <div class="model-title">
          <span>BFG-Diff (Discrete Diffusion)</span>
          <span class="metric-badge" id="bfgMetrics">0 TPS</span>
        </div>
        <div class="output-box" id="bfgOutput">Generating...</div>
      </div>
      <div class="model-card">
        <div class="model-title">
          <span>Causal Baseline (generate)</span>
          <span class="metric-badge" id="causalMetrics">0 TPS</span>
        </div>
        <div class="output-box" id="causalOutput">Generating...</div>
      </div>
    </div>

    <div class="trace-card" id="traceContainer">
      <div class="trace-title">Block-by-Block Parallel Denoising Trace</div>
      <div id="traceLogs"></div>
    </div>
  </div>
</div>

<script>
async function runGeneration() {
  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  btn.innerText = 'Denoising...';

  const resultsSection = document.getElementById('resultsSection');
  resultsSection.style.display = 'block';

  const bfgOutput = document.getElementById('bfgOutput');
  const causalOutput = document.getElementById('causalOutput');
  const bfgMetrics = document.getElementById('bfgMetrics');
  const causalMetrics = document.getElementById('causalMetrics');
  const traceLogs = document.getElementById('traceLogs');

  bfgOutput.innerText = 'Running parallel carrier-frontier unmasking...';
  causalOutput.innerText = 'Running causal autoregression...';
  traceLogs.innerHTML = '';

  const payload = {
    prompt: document.getElementById('prompt').value,
    mode: document.getElementById('mode').value,
    canvas_len: parseInt(document.getElementById('canvas_len').value),
    sub_canvas_len: parseInt(document.getElementById('sub_canvas_len').value),
    sub_steps: parseInt(document.getElementById('sub_steps').value),
    temp: parseFloat(document.getElementById('temp').value),
  };

  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    bfgOutput.innerText = data.bfg.text;
    bfgMetrics.innerText = `${data.bfg.tps.toFixed(1)} TPS (${data.bfg.elapsed.toFixed(2)}s)`;

    causalOutput.innerText = data.causal.text;
    causalMetrics.innerText = `${data.causal.tps.toFixed(1)} TPS (${data.causal.elapsed.toFixed(2)}s)`;

    // Render trace
    let traceHtml = '';
    data.bfg.trace.forEach((block, bIdx) => {
      traceHtml += `<div class="block-trace">`;
      traceHtml += `<div style="font-size:13px; font-weight:600; color:var(--muted); margin-bottom:6px;">Sub-Block ${bIdx + 1}: Final: "${block.final_text}"</div>`;
      block.steps.forEach(step => {
        traceHtml += `<div class="step-line">`;
        traceHtml += `<span class="step-num">Step ${step.step}</span>`;
        traceHtml += `<div class="tokens-canvas">`;
        step.tokens.forEach(tok => {
          const isMask = tok === '░';
          traceHtml += `<span class="token-chip ${isMask ? 'mask' : ''}">${isMask ? '░' : tok}</span>`;
        });
        traceHtml += `</div></div>`;
      });
      traceHtml += `</div>`;
    });
    traceLogs.innerHTML = traceHtml;
  } catch (err) {
    bfgOutput.innerText = 'Error: ' + err.message;
  } finally {
    btn.disabled = false;
    btn.innerText = 'Generate & Compare';
  }
}
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

            # BFG generation
            bfg_res = generate_bfg_traced(
                self.model,
                self.tok,
                self.mask_id,
                prompt,
                canvas_len=canvas_len,
                sub_canvas_len=sub_canvas_len,
                sub_steps=sub_steps,
                temperature=temp,
                stop_tokens=stop_tokens,
            )

            # Causal generation
            causal_res = generate_causal(self.model, self.tok, prompt, max_new_tokens=canvas_len)

            resp_data = {"bfg": bfg_res, "causal": causal_res}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp_data).encode("utf-8"))
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
    args = parser.parse_args()

    # Load model
    model, tok, mask_id = load_bfg_model(args.checkpoint, device="cpu")

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
