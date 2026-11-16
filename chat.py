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


def load_bfg_model(checkpoint_path, model_name="Qwen/Qwen2.5-0.5B-Instruct", device="cpu"):
    """Load base Qwen model and apply checkpoint adapters."""
    torch.set_num_threads(10)
    device = torch.device(device)
    model, tok, mask_id = prepare(model_name, device=device, dtype=torch.float32)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    ck_file = Path(checkpoint_path)
    if not ck_file.exists():
        fallback = Path("runs/bfg-polished/last.pt") if Path("runs/bfg-polished/last.pt").exists() else Path("runs/bfg-shifted/last.pt")
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
    )

    elapsed = time.perf_counter() - start_time
    total_tokens = canvas_out.shape[1]
    raw_text = tok.decode(canvas_out[0], skip_special_tokens=False)

    # In chat mode, truncate cleanly at stop tokens
    if stop_tokens:
        for st in stop_tokens:
            if st in raw_text:
                raw_text = raw_text.split(st)[0]
    full_text = raw_text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
    tps = total_tokens / max(1e-5, elapsed)

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
    padding: 12px;
    display: flex;
    justify-content: center;
    box-sizing: border-box;
    -webkit-font-smoothing: antialiased;
  }
  .container {
    max-width: 900px;
    width: 100%;
    margin: 0 auto;
  }
  header {
    margin-bottom: 16px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 12px;
  }
  h1 { margin: 0 0 6px 0; font-size: 20px; color: var(--primary); }
  p.subtitle { margin: 0; color: var(--muted); font-size: 13px; }
  .controls {
    background: var(--card);
    padding: 12px;
    border-radius: 8px;
    border: 1px solid var(--border);
    margin-bottom: 16px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 10px;
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
    max-width: 100%;
  }
  textarea { resize: vertical; min-height: 80px; width: 100%; font-family: inherit; line-height: 1.4; }
  button {
    background: var(--primary);
    color: #090d16;
    font-weight: bold;
    cursor: pointer;
    border: none;
    transition: all 0.2s;
    padding: 10px 16px;
    border-radius: 6px;
    font-size: 14px;
    width: 100%;
  }
  @media (min-width: 600px) {
    body { padding: 20px; }
    h1 { font-size: 24px; }
    button { width: auto; }
  }
  button:hover { opacity: 0.9; transform: translateY(-1px); }
  button:active { transform: translateY(0); }
  .prompt-card {
    background: var(--card);
    padding: 12px;
    border-radius: 8px;
    border: 1px solid var(--border);
    margin-bottom: 16px;
  }
  .results { display: flex; flex-direction: column; gap: 16px; }
  .comparison-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 12px;
  }
  @media (min-width: 768px) {
    .comparison-grid { grid-template-columns: 1fr 1fr; gap: 16px; }
  }
  .model-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .model-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--primary);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 6px;
  }
  .metric-badge {
    background: rgba(56, 189, 248, 0.15);
    color: var(--primary);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
  }
  .output-box {
    background: #090d16;
    padding: 10px;
    border-radius: 6px;
    border: 1px solid var(--border);
    font-family: monospace;
    font-size: 13px;
    white-space: pre-wrap;
    word-break: break-word;
    line-height: 1.5;
    min-height: 60px;
  }
  .trace-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
    margin-top: 4px;
    overflow-x: auto;
  }
  .trace-title { font-size: 15px; font-weight: 600; color: var(--accent); margin-bottom: 10px; }
  .block-trace { margin-bottom: 14px; border-left: 3px solid var(--accent); padding-left: 10px; }
  .step-line {
    font-family: monospace;
    font-size: 12px;
    padding: 3px 0;
    display: flex;
    align-items: flex-start;
    gap: 6px;
    flex-wrap: wrap;
  }
  .step-num { color: var(--muted); min-width: 50px; font-weight: bold; flex-shrink: 0; }
  .tokens-canvas { display: flex; gap: 4px; flex-wrap: wrap; }
  .token-chip {
    background: #1e293b;
    border: 1px solid var(--border);
    padding: 2px 5px;
    border-radius: 4px;
    font-size: 11px;
    white-space: pre;
    word-break: break-all;
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
        <option value="chat" selected>Chat (&lt;|im_start|&gt;)</option>
        <option value="completion">Completion</option>
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
    <textarea id="prompt">How are you doing today?</textarea>
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
