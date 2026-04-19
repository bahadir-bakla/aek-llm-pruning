"""
evaluate_7b_v2.py
=================
AEK v0.2 — 7B full evaluation script.

Changes from v0.1 (evaluate_7b.py):
  - compress_utils_v2.full_compress_v2() with layer-specific γ
  - K dampening: threshold = (1-K·α)·ε  where α = 1/log2(hidden_dim/256)
  - --gamma_mode spectral/fixed/nuclear
  - --k_alpha_override (optional, overrides auto)
  - --block_budget / --no_block_budget
  - Adaptive Fisher sampling: n_samples = hidden_dim // 256

Usage (AWS g5.xlarge):
  python src/evaluate_7b_v2.py \\
      --model Qwen/Qwen2.5-7B-Instruct \\
      --eps 0.35 \\
      --gamma_mode spectral \\
      --output results/aek_v2_7b_eps035.json
"""

import argparse
import json
import os
import sys
import time
import glob

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="AEK v0.2 — 7B evaluation")
    p.add_argument("--model",             default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--eps",               type=float, default=0.35)
    p.add_argument("--gamma_mode",        default="spectral",
                   choices=["spectral", "fixed", "nuclear"])
    p.add_argument("--k_alpha_override",  type=float, default=None,
                   help="Manual k_alpha (None = auto from hidden_dim)")
    p.add_argument("--block_budget",      action="store_true", default=True)
    p.add_argument("--no_block_budget",   dest="block_budget", action="store_false")
    p.add_argument("--fisher_adaptive",   action="store_true", default=True)
    p.add_argument("--fisher_samples",    type=int, default=None)
    p.add_argument("--dtype",             default="bfloat16",
                   choices=["bfloat16", "float32"])
    p.add_argument("--save_model_dir",    default=None)
    p.add_argument("--output",            default="results/aek_v2_7b_eps035.json")
    p.add_argument("--skip_lmeval",       action="store_true", default=False)
    p.add_argument("--ppl_stride",        type=int, default=512)
    p.add_argument("--ppl_samples",       type=int, default=50)
    return p.parse_args()


def sliding_ppl(model, tokenizer, n_samples=50, stride=512, max_length=1024, device="cpu"):
    """Sliding-window perplexity on wikitext-2-raw-v1 test set."""
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


def run_lmeval(model_dir, out_prefix, tasks="hellaswag,arc_easy,arc_challenge"):
    cmd = (
        f"lm_eval --model hf "
        f"--model_args pretrained={model_dir},dtype=bfloat16 "
        f"--tasks {tasks} "
        f"--batch_size 4 "
        f"--output_path {out_prefix}"
    )
    print(f"\n[lm-eval] {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print("[lm-eval] Error code:", ret)
        return None

    for pat in [f"{out_prefix}/**/*.json", f"{out_prefix}/*.json", f"{out_prefix}.json"]:
        files = glob.glob(pat, recursive=True)
        if files:
            with open(files[0]) as f:
                return json.load(f)
    return None


def extract_lmeval_scores(data):
    if data is None:
        return {}
    results = data.get("results", {})
    scores = {}
    for task in ["hellaswag", "arc_easy", "arc_challenge"]:
        if task in results:
            val = results[task].get("acc_norm,none", results[task].get("acc_norm"))
            if val is not None:
                scores[task] = round(float(val) * 100, 2)
    return scores


def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, dtype: {dtype}")

    # Load model
    print(f"\n[1/5] Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    hidden_dim = model.config.hidden_size
    print(f"  hidden_dim={hidden_dim}")

    # Original PPL
    print(f"\n[2/5] Computing original PPL...")
    t0 = time.time()
    ppl_orig = sliding_ppl(model, tokenizer, n_samples=args.ppl_samples,
                           stride=args.ppl_stride, device=device)
    print(f"  PPL original = {ppl_orig:.4f}  ({time.time()-t0:.1f}s)")

    # Original lm-eval (GPU-free pattern: save → del → eval → reload)
    lmeval_orig = {}
    if not args.skip_lmeval:
        print(f"\n[3a/5] Original lm-eval (saving model, freeing GPU)...")
        orig_tmp = "/opt/dlami/nvme/tmp_orig_eval" if os.path.isdir("/opt/dlami/nvme") else "models/tmp_orig_eval"
        os.makedirs(orig_tmp, exist_ok=True)
        model.save_pretrained(orig_tmp)
        tokenizer.save_pretrained(orig_tmp)
        del model
        import gc; gc.collect()
        torch.cuda.empty_cache()

        lmeval_orig = extract_lmeval_scores(run_lmeval(orig_tmp, "results/lmeval_orig_v2"))
        print(f"  lm-eval original: {lmeval_orig}")

        print(f"  Reloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            orig_tmp, dtype=dtype, device_map="auto", trust_remote_code=True)
        model.eval()

    # AEK v0.2 compress
    print(f"\n[3/5] AEK v0.2 compression...")
    print(f"  eps={args.eps}, gamma_mode={args.gamma_mode}")

    sys.path.insert(0, os.path.dirname(__file__))
    from compress_utils_v2 import full_compress_v2, compute_k_alpha

    if args.k_alpha_override is not None:
        k_alpha = args.k_alpha_override
    else:
        k_alpha = compute_k_alpha(hidden_dim)
    print(f"  k_alpha = {k_alpha:.4f}")

    t0 = time.time()
    compress_results = full_compress_v2(
        model=model,
        eps=args.eps,
        gamma_mode=args.gamma_mode,
        k_alpha=k_alpha,
        use_block_budget=args.block_budget,
        fisher_adaptive=args.fisher_adaptive,
        output_path="results/aek_v2_7b_compress_decisions.json",
    )
    compress_time = time.time() - t0
    print(f"  Compress time: {compress_time:.1f}s")
    print(f"  n_eliminated: {compress_results['n_eliminated']}")
    print(f"  E_total/E_bound: {compress_results['E_total']:.4f} / {compress_results['E_bound']:.4f}")

    # AEK PPL
    print(f"\n[4/5] Computing AEK PPL...")
    t0 = time.time()
    ppl_aek = sliding_ppl(model, tokenizer, n_samples=args.ppl_samples,
                          stride=args.ppl_stride, device=device)
    print(f"  PPL AEK = {ppl_aek:.4f}  ({time.time()-t0:.1f}s)")
    ppl_delta = (ppl_aek - ppl_orig) / ppl_orig * 100

    # Save model (optional)
    if args.save_model_dir:
        model.save_pretrained(args.save_model_dir)
        tokenizer.save_pretrained(args.save_model_dir)

    # AEK lm-eval
    lmeval_aek = {}
    if not args.skip_lmeval:
        print(f"\n[5/5] AEK lm-eval (saving model, freeing GPU)...")
        aek_dir = args.save_model_dir or (
            "/opt/dlami/nvme/tmp_aek_eval" if os.path.isdir("/opt/dlami/nvme") else "models/tmp_aek_eval")
        os.makedirs(aek_dir, exist_ok=True)
        model.save_pretrained(aek_dir)
        tokenizer.save_pretrained(aek_dir)
        del model
        import gc; gc.collect()
        torch.cuda.empty_cache()

        lmeval_aek = extract_lmeval_scores(run_lmeval(aek_dir, "results/lmeval_aek_v2"))
        print(f"  lm-eval AEK: {lmeval_aek}")

    # Elim by type
    elim_by_type = {}
    for d in compress_results.get("decisions", []):
        if d.get("eliminated"):
            wname = d.get("weight_name", "unknown")
            elim_by_type[wname] = elim_by_type.get(wname, 0) + 1

    output = {
        "version":    "v0.2",
        "model":      args.model,
        "eps":        args.eps,
        "gamma_mode": args.gamma_mode,
        "k_alpha":    k_alpha,
        "compress": {
            "n_eliminated":    compress_results["n_eliminated"],
            "E_total":         compress_results["E_total"],
            "E_bound":         compress_results["E_bound"],
            "theorem_sat":     compress_results["theorem_sat"],
            "elim_by_type":    elim_by_type,
            "compress_time_s": compress_time,
        },
        "ppl_sliding": {
            "original":  ppl_orig,
            "aek":       ppl_aek,
            "delta_pct": round(ppl_delta, 4),
        },
        "lmeval": {
            "original": lmeval_orig,
            "aek":      lmeval_aek,
        },
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  AEK v0.2 — SUMMARY")
    print(f"{'='*60}")
    print(f"  Model       : {args.model}")
    print(f"  ε           : {args.eps}")
    print(f"  gamma_mode  : {args.gamma_mode}")
    print(f"  k_alpha     : {k_alpha:.4f}")
    print(f"  Eliminations: {compress_results['n_eliminated']}")
    print(f"  Elim types  : {elim_by_type}")
    print(f"  E_total     : {compress_results['E_total']:.4f} / {compress_results['E_bound']:.4f}")
    print(f"  Theorem     : {'SATISFIED' if compress_results['theorem_sat'] else 'VIOLATED'}")
    print(f"  PPL orig    : {ppl_orig:.4f}")
    print(f"  PPL AEK     : {ppl_aek:.4f}")
    print(f"  PPL Δ       : {ppl_delta:+.4f}%")
    if lmeval_orig and lmeval_aek:
        print(f"  HellaSwag   : {lmeval_orig.get('hellaswag','?')} → {lmeval_aek.get('hellaswag','?')}")
        print(f"  ARC-Easy    : {lmeval_orig.get('arc_easy','?')} → {lmeval_aek.get('arc_easy','?')}")
        print(f"  ARC-Chall.  : {lmeval_orig.get('arc_challenge','?')} → {lmeval_aek.get('arc_challenge','?')}")
    print(f"{'='*60}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
