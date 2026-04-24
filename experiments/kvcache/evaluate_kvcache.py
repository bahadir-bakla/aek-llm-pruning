"""
evaluate_kvcache.py
===================
AEK-KV Cache eval script.

Ölçüler:
  - PPL: full context vs budget-limited (simulates KV eviction)
  - Cache stats: eviction count, theorem check
  - Memory proxy: tokens kept vs total

Kullanım:
  python experiments/kvcache/evaluate_kvcache.py --model Qwen/Qwen2.5-0.5B-Instruct
  python experiments/kvcache/evaluate_kvcache.py --budget 256 --eps 0.05
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
from src.kvcache import AEKKVCache


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--budget",      type=int, default=256, help="Max KV cache tokens")
    p.add_argument("--eps",         type=float, default=0.10)
    p.add_argument("--keep_recent", type=int, default=32)
    p.add_argument("--n_samples",   type=int, default=20)
    p.add_argument("--stride",      type=int, default=256)
    p.add_argument("--output",      default="results/kvcache/kv_results.json")
    return p.parse_args()


def sliding_ppl_full(model, tokenizer, n_samples=20, stride=256,
                     max_length=512, device="cpu"):
    """Standard full-context PPL baseline."""
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


def sliding_ppl_budget(model, tokenizer, budget, n_samples=20, stride=256,
                       max_length=512, device="cpu"):
    """
    PPL with budget-limited context window.
    Simulates KV cache eviction by only feeding the most recent `budget` tokens
    as context. This is the conservative bound on AEK-KV performance.
    """
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
            chunk = input_ids[start:end + 1]
            # Limit context to last `budget` tokens
            if len(chunk) > budget:
                chunk = chunk[-budget:]
            chunk = chunk.unsqueeze(0).to(device)
            out = model(chunk, labels=chunk.clone())
            nll = out.loss.item()
            if np.isfinite(nll):
                total_nll += nll
                count += 1
    return float(np.exp(total_nll / count)) if count > 0 else float("inf")


def simulate_aek_eviction(budget, eps, keep_recent, n_tokens, hidden_dim):
    """
    Simulate AEK-KV eviction stats without running the model.
    Returns: n_evictions, E_total, theorem_satisfied.
    """
    cache = AEKKVCache(budget=budget, eps=eps, hidden_dim=hidden_dim,
                       keep_recent=keep_recent)

    # Simulate token-by-token with synthetic KV
    for layer_idx in range(4):  # simulate 4 layers
        for t in range(n_tokens):
            dummy_k = torch.randn(1, 1, 1, 64)
            dummy_v = torch.randn(1, 1, 1, 64)
            cache.update(layer_idx, dummy_k, dummy_v)

    return cache.stats()


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

    print(f"\n[1/3] Loading: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True)
    model.eval()
    hidden_dim = model.config.hidden_size
    print(f"  hidden_dim={hidden_dim}, budget={args.budget}, eps={args.eps}")

    print(f"\n[2/3] PPL baseline (full context, max_length=512)...")
    t0 = time.time()
    ppl_full = sliding_ppl_full(model, tokenizer, args.n_samples, args.stride,
                                 max_length=512, device=device)
    print(f"  PPL full    = {ppl_full:.4f}  ({time.time()-t0:.1f}s)")

    print(f"\n[3/3] PPL budget-limited (context={args.budget} tokens)...")
    t0 = time.time()
    ppl_budget = sliding_ppl_budget(model, tokenizer, args.budget, args.n_samples,
                                     args.stride, max_length=512, device=device)
    ppl_delta = (ppl_budget - ppl_full) / ppl_full * 100
    print(f"  PPL budget  = {ppl_budget:.4f}  ({time.time()-t0:.1f}s)")
    print(f"  PPL Δ       = {ppl_delta:+.4f}%")

    # Simulate eviction stats for different sequence lengths
    print(f"\n  Eviction simulation (eps={args.eps}, budget={args.budget}):")
    eviction_results = []
    for seq_len in [256, 512, 1024, 2048]:
        stats = simulate_aek_eviction(args.budget, args.eps, args.keep_recent,
                                       seq_len, hidden_dim)
        theorem = "✓" if stats["theorem_satisfied"] else "✗"
        print(f"    seq={seq_len}: evictions={stats['n_evictions']}, "
              f"E={stats['E_total']:.4f}, theorem={theorem}")
        eviction_results.append({"seq_len": seq_len, **stats})

    output = {
        "version": "aek_kv_v1",
        "model": args.model,
        "budget": args.budget,
        "eps": args.eps,
        "keep_recent": args.keep_recent,
        "ppl_full": ppl_full,
        "ppl_budget": ppl_budget,
        "ppl_delta_pct": round(ppl_delta, 4),
        "eviction_sim": eviction_results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  AEK-KV ÖZET")
    print(f"{'='*55}")
    print(f"  Model         : {args.model}")
    print(f"  Budget        : {args.budget} tokens")
    print(f"  ε             : {args.eps}")
    print(f"  PPL full      : {ppl_full:.4f}")
    print(f"  PPL budget    : {ppl_budget:.4f}")
    print(f"  PPL Δ         : {ppl_delta:+.4f}%")
    print(f"  Memory ratio  : {args.budget}/512 = {args.budget/512:.2f}x")
    print(f"{'='*55}")
    print(f"  Results: {args.output}")


if __name__ == "__main__":
    main()
