"""
evaluate_structured.py
======================
AEK v4 — Structured Pruning eval script.

Kullanım:
  python experiments/structured/evaluate_structured.py --model Qwen/Qwen2.5-0.5B-Instruct
  python experiments/structured/evaluate_structured.py --model Qwen/Qwen2.5-7B-Instruct --skip_lmeval
"""

import argparse
import json
import os
import sys
import time

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.structured import collect_activation_stats, calibrate_layer_eps, full_compress_v4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",         default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--eps",           type=float, default=0.35)
    p.add_argument("--eps_min",       type=float, default=0.10)
    p.add_argument("--eps_max",       type=float, default=0.60)
    p.add_argument("--flat_eps",      action="store_true", help="Disable adaptive ε")
    p.add_argument("--no_remove",     action="store_true", help="Zero heads instead of removing")
    p.add_argument("--calib_samples", type=int, default=32)
    p.add_argument("--ppl_samples",   type=int, default=50)
    p.add_argument("--ppl_stride",    type=int, default=512)
    p.add_argument("--skip_lmeval",   action="store_true")
    p.add_argument("--output",        default="results/structured/v4_results.json")
    return p.parse_args()


def sliding_ppl(model, tokenizer, n_samples=50, stride=512, max_length=1024, device="cuda"):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"][0]

    total_nll, count = 0.0, 0
    positions = list(range(0, min(len(input_ids) - 1, n_samples * stride), stride))
    model.eval()
    with torch.no_grad():
        for start in positions:
            end = min(start + max_length, len(input_ids) - 1)
            if end <= start:
                continue
            chunk = input_ids[start:end + 1].unsqueeze(0).to(device)
            out = model(chunk, labels=chunk.clone())
            nll = out.loss.item()
            if np.isfinite(nll):
                total_nll += nll
                count += 1
    return float(np.exp(total_nll / count)) if count > 0 else float("inf")


def make_calib_dataloader(tokenizer, n_samples=32, seq_len=512, device="cuda"):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"][0]

    batches = []
    for i in range(n_samples):
        start = i * seq_len
        end = start + seq_len
        if end > len(input_ids):
            break
        batches.append({"input_ids": input_ids[start:end].unsqueeze(0).to(device)})
    return batches


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    print(f"\n[1/4] Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}  (~{n_params*2/1e9:.2f} GB BF16)")

    print(f"\n[2/4] PPL original...")
    ppl_orig = sliding_ppl(model, tokenizer, args.ppl_samples, args.ppl_stride, device=device)
    print(f"  PPL original = {ppl_orig:.4f}")

    activation_stats = None
    if not args.flat_eps:
        print(f"\n[3/4] Calibrating layer-adaptive ε ({args.calib_samples} samples)...")
        calib = make_calib_dataloader(tokenizer, args.calib_samples, device=device)
        activation_stats = collect_activation_stats(model, calib, args.calib_samples, device)
        eps_map = calibrate_layer_eps(activation_stats, args.eps, args.eps_min, args.eps_max)
        eps_vals = list(eps_map.values())
        if eps_vals:
            print(f"  ε range: [{min(eps_vals):.3f}, {max(eps_vals):.3f}], mean={np.mean(eps_vals):.3f}")
    else:
        print(f"\n[3/4] Using flat ε={args.eps} (--flat_eps)")

    print(f"\n[4/4] AEK v4 compress (remove_heads={not args.no_remove})...")
    t0 = time.time()
    results = full_compress_v4(
        model,
        eps=args.eps,
        activation_stats=activation_stats,
        eps_min=args.eps_min,
        eps_max=args.eps_max,
        remove_heads=not args.no_remove,
        output_path=args.output.replace(".json", "_decisions.json"),
    )
    compress_time = time.time() - t0
    print(f"  Compress time: {compress_time:.1f}s")

    n_params_after = sum(p.numel() for p in model.parameters())
    disk_before = n_params * 2 / 1e9
    disk_after = n_params_after * 2 / 1e9
    print(f"  Params: {n_params:,} → {n_params_after:,}")
    print(f"  Disk:   {disk_before:.3f} → {disk_after:.3f} GB")

    print(f"\n  PPL after compress...")
    ppl_aek = sliding_ppl(model, tokenizer, args.ppl_samples, args.ppl_stride, device=device)
    ppl_delta = (ppl_aek - ppl_orig) / ppl_orig * 100
    print(f"  PPL AEK v4 = {ppl_aek:.4f} ({ppl_delta:+.4f}%)")

    output = {
        "version": "v4_structured",
        "model": args.model,
        "eps": args.eps,
        "adaptive_eps": not args.flat_eps,
        "remove_heads": not args.no_remove,
        "compress": results,
        "compress_time_s": round(compress_time, 1),
        "disk": {
            "before_gb": round(disk_before, 3),
            "after_gb": round(disk_after, 3),
            "reduction_x": round(disk_before / max(disk_after, 1e-6), 3),
        },
        "ppl": {
            "original": ppl_orig,
            "aek_v4": ppl_aek,
            "delta_pct": round(ppl_delta, 4),
        },
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results: {args.output}")


if __name__ == "__main__":
    main()
