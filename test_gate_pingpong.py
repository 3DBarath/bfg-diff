"""Gate A Verification: Verify multi-exit lookahead drafting eliminates the ping-pong wipeout."""
from pathlib import Path
import torch
from peft import set_peft_model_state_dict

from benchmark import GEN_PROMPTS, tokenize
from model import LORA_TARGETS, apply_lora, prepare, setup_bfg_adapter
from sampler import sample_staircase_sliding


def run_gate_pingpong(model_name="Qwen/Qwen2.5-0.5B", ckpt="runs/bfg-polished-v2/last.pt", device="cpu"):
    torch.set_num_threads(10)
    dev = torch.device(device)
    model, tok, mask_id = prepare(model_name, device=dev, dtype=torch.float32)

    if ckpt and Path(ckpt).exists():
        print(f"Loading checkpoint {ckpt}...")
        ck = torch.load(ckpt, map_location=dev)
        if "lora" in ck:
            cfg = ck.get("lora_config", {})
            model = apply_lora(
                model,
                r=cfg.get("r", 16),
                alpha=cfg.get("alpha", 32),
                dropout=cfg.get("dropout", 0.05),
                targets=cfg.get("targets", LORA_TARGETS),
            )
            set_peft_model_state_dict(model, ck["lora"])
            if "mask_emb" in ck:
                with torch.no_grad():
                    model.get_input_embeddings().weight[mask_id] = ck["mask_emb"].to(dev)
        else:
            model.load_state_dict(ck["model"], strict=False)

        if "bfg_adapter" in ck:
            if getattr(model, "bfg_adapter", None) is None:
                setup_bfg_adapter(model)
            model.bfg_adapter.load_state_dict(ck["bfg_adapter"])
        mask_id = ck.get("mask_id", mask_id)
    else:
        print("No checkpoint found; running on base model")

    model.eval()

    test_prompts = GEN_PROMPTS[:5]
    print(f"\n--- Gate A: Multi-Exit Lookahead Trace on {len(test_prompts)} Prompts ---")

    total_committed = 0
    total_passes = 0

    for idx, p in enumerate(test_prompts):
        p_ids = tokenize(tok, p)
        prefix_ids = torch.tensor([p_ids], device=dev)
        stats = {}

        out_tokens = sample_staircase_sliding(
            model,
            prefix_ids,
            canvas_len=32,
            mask_id=mask_id,
            stats_out=stats,
            adapter=getattr(model, "bfg_adapter", None),
        )

        n_tok = out_tokens.shape[1]
        fp = stats.get("forward_passes", 1)
        alpha = n_tok / max(1, fp)
        total_committed += n_tok
        total_passes += fp
        hist = stats.get("accept_histogram", {})
        print(f"Prompt {idx+1}: tokens={n_tok}, passes={fp}, alpha={alpha:.2f}, hist={hist}")

    mean_alpha = total_committed / max(1, total_passes)
    print(f"\nMean Alpha across 5 prompts: {mean_alpha:.2f} tok/pass")

    assert mean_alpha >= 1.40, f"Expected mean alpha >= 1.40, got {mean_alpha:.2f}"
    print("GATE A: PASS")
    return mean_alpha


if __name__ == "__main__":
    run_gate_pingpong()
