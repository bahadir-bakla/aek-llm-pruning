"""
compress_utils_v4.py
====================
AEK v4 — Structured Pruning + Layer-Adaptive ε

Two advances over v0.2:
  1. Physical head removal (not just zeroing) → real disk savings
  2. Layer-adaptive ε via activation statistics → smarter budget allocation

Pipeline:
  calibrate_layer_eps(model, dataloader) → eps_per_layer dict
  structured_compress(model, eps_per_layer) → physically smaller model
  verify_theorem(results) → E_total < E_bound check
"""

import math
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple


# ── Layer-Adaptive ε ─────────────────────────────────────────────────────────

def collect_activation_stats(model, dataloader, n_samples: int = 32, device: str = "cuda") -> Dict[str, float]:
    """
    Collect per-layer activation L2 norms as sensitivity proxy.
    Layers with high activation variance → low ε (sensitive).
    Layers with low activation variance → high ε (safe to prune).
    """
    stats = {}
    hooks = []
    layer_outputs = {}

    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            norm = out.detach().float().norm(dim=-1).mean().item()
            if name not in layer_outputs:
                layer_outputs[name] = []
            layer_outputs[name].append(norm)
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            if count >= n_samples:
                break
            input_ids = batch["input_ids"].to(device) if isinstance(batch, dict) else batch[0].to(device)
            model(input_ids)
            count += 1

    for h in hooks:
        h.remove()

    for name, vals in layer_outputs.items():
        stats[name] = float(np.std(vals))

    return stats


def calibrate_layer_eps(
    activation_stats: Dict[str, float],
    eps_global: float = 0.35,
    eps_min: float = 0.10,
    eps_max: float = 0.60,
) -> Dict[str, float]:
    """
    Map activation std → per-layer ε.
    High std (sensitive layer) → eps_min.
    Low std (stable layer) → eps_max.
    Linear interpolation, global mean preserved.
    """
    if not activation_stats:
        return {}

    vals = np.array(list(activation_stats.values()))
    v_min, v_max = vals.min(), vals.max()

    eps_per_layer = {}
    for name, std in activation_stats.items():
        if v_max > v_min:
            t = (std - v_min) / (v_max - v_min)  # 0=stable, 1=sensitive
        else:
            t = 0.5
        # Invert: stable → high ε, sensitive → low ε
        eps_layer = eps_max - t * (eps_max - eps_min)
        eps_per_layer[name] = round(float(eps_layer), 4)

    # Rescale so mean ≈ eps_global
    mean_eps = np.mean(list(eps_per_layer.values()))
    scale = eps_global / mean_eps
    for name in eps_per_layer:
        eps_per_layer[name] = float(np.clip(eps_per_layer[name] * scale, eps_min, eps_max))

    return eps_per_layer


# ── Structured Head Removal ──────────────────────────────────────────────────

def get_head_importance(W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor,
                         n_heads: int, head_dim: int) -> torch.Tensor:
    """
    Per-head importance = mean Frobenius norm of Q/K/V slices.
    Shape: (n_heads,)
    """
    scores = []
    for h in range(n_heads):
        s = h * head_dim
        e = s + head_dim
        score = (
            W_q[s:e].norm("fro") +
            W_k[s:e].norm("fro") +
            W_v[s:e].norm("fro")
        ) / 3.0
        scores.append(score.item())
    return torch.tensor(scores)


def physically_remove_heads(
    attn_module,
    heads_to_remove: List[int],
    n_heads: int,
    head_dim: int,
) -> nn.Module:
    """
    Physically remove attention heads from q/k/v/o_proj weight matrices.
    Returns modified module with smaller weight tensors.

    For GQA (grouped query attention), only removes from full heads.
    """
    heads_to_keep = [h for h in range(n_heads) if h not in heads_to_remove]
    if not heads_to_keep:
        return attn_module

    keep_idx = torch.tensor([h * head_dim + d
                              for h in heads_to_keep
                              for d in range(head_dim)], dtype=torch.long)

    with torch.no_grad():
        for proj_name in ["q_proj", "k_proj", "v_proj"]:
            proj = getattr(attn_module, proj_name, None)
            if proj is None:
                continue
            W = proj.weight.data  # (out, in)
            proj.weight = nn.Parameter(W[keep_idx])
            if proj.bias is not None:
                proj.bias = nn.Parameter(proj.bias.data[keep_idx])

        # o_proj: input dim shrinks
        o_proj = getattr(attn_module, "o_proj", None)
        if o_proj is not None:
            W = o_proj.weight.data  # (out, in)
            o_proj.weight = nn.Parameter(W[:, keep_idx])

    # Update config metadata on module
    attn_module.num_heads = len(heads_to_keep)
    if hasattr(attn_module, "num_key_value_heads"):
        # GQA: keep ratio
        kv_ratio = attn_module.num_key_value_heads / n_heads
        attn_module.num_key_value_heads = max(1, round(len(heads_to_keep) * kv_ratio))

    return attn_module


# ── AEK v4 Core ─────────────────────────────────────────────────────────────

def compute_k_alpha(hidden_dim: int) -> float:
    return 1.0 / math.log2(hidden_dim / 256)


def compute_layer_gamma(W: torch.Tensor) -> float:
    rank = min(W.shape)
    return float(W.float().norm("fro") / math.sqrt(rank))


def kalman_threshold(gamma: float, eps: float, k_alpha: float,
                     P_tilde: float = 1.0, H_obs: float = 0.3) -> Tuple[float, float]:
    """Returns (threshold, K)"""
    S = P_tilde * gamma ** 2
    K = S / (S + H_obs)
    threshold = (1.0 - K * k_alpha) * eps
    return threshold, K


def full_compress_v4(
    model,
    eps: float = 0.35,
    activation_stats: Optional[Dict[str, float]] = None,
    eps_min: float = 0.10,
    eps_max: float = 0.60,
    remove_heads: bool = True,
    output_path: Optional[str] = None,
) -> Dict:
    """
    AEK v4: layer-adaptive ε + physical head removal.

    Args:
        model: HuggingFace causal LM
        eps: global ε (used as mean target when adaptive)
        activation_stats: from collect_activation_stats(); if None, uses flat eps
        remove_heads: if True, physically removes heads; if False, just zeros (v0.2 mode)
        output_path: JSON path for decisions log

    Returns dict with elimination counts, E_total, E_bound, disk savings estimate.
    """
    import json

    hidden_dim = model.config.hidden_size
    k_alpha = compute_k_alpha(hidden_dim)

    # Calibrate per-layer ε
    if activation_stats:
        eps_per_layer = calibrate_layer_eps(activation_stats, eps, eps_min, eps_max)
    else:
        eps_per_layer = {}

    # Count blocks and decisions
    n_blocks = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = hidden_dim // n_heads
    n_decisions = n_blocks * 4  # q/k/v/o per block
    E_bound = n_decisions * eps

    E_total = 0.0
    n_eliminated = 0
    decisions = []
    total_heads_removed = 0

    for block_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        heads_to_remove = []

        for proj_name, proj_key in [("q_proj", "q"), ("k_proj", "k"),
                                     ("v_proj", "v"), ("o_proj", "o")]:
            proj = getattr(attn, proj_name, None)
            if proj is None:
                continue

            W = proj.weight.data.float()
            layer_name = f"model.layers.{block_idx}.self_attn.{proj_name}"
            layer_eps = eps_per_layer.get(layer_name, eps)

            gamma = compute_layer_gamma(W)
            threshold, K = kalman_threshold(gamma, layer_eps, k_alpha)

            # Singular values
            try:
                sv = torch.linalg.svdvals(W)
            except Exception:
                sv = torch.linalg.svdvals(W.cpu()).to(W.device)

            sv_norm = sv / (sv[0] + 1e-8)
            elim = float((sv_norm < threshold).float().mean())
            E_total += elim

            eliminated = elim > 0.0
            if eliminated:
                n_eliminated += 1
                if proj_name in ("q_proj", "k_proj", "v_proj"):
                    # Mark heads where norm < threshold for removal
                    head_norms = W.view(n_heads, head_dim, -1).norm(dim=(1, 2))
                    head_norms_norm = head_norms / (head_norms.max() + 1e-8)
                    bad_heads = (head_norms_norm < threshold).nonzero(as_tuple=True)[0].tolist()
                    heads_to_remove.extend(bad_heads)

            decisions.append({
                "block": block_idx,
                "proj": proj_name,
                "gamma": round(gamma, 4),
                "threshold": round(threshold, 4),
                "K": round(K, 4),
                "eps_layer": round(layer_eps, 4),
                "elim": round(elim, 4),
                "eliminated": eliminated,
            })

            # Update Kalman P̃
            P_new = (1 - K) * 1.0
            K_new = P_new * gamma ** 2 / (P_new * gamma ** 2 + 0.3)

        # Physical head removal
        if remove_heads and heads_to_remove:
            heads_to_remove = sorted(set(heads_to_remove))
            physically_remove_heads(attn, heads_to_remove, n_heads, head_dim)
            total_heads_removed += len(heads_to_remove)

        if (block_idx + 1) % 4 == 0:
            print(f"  Block {block_idx+1}/{n_blocks} — elim: {n_eliminated}, "
                  f"E={E_total:.4f}, heads_removed={total_heads_removed}")

    theorem_sat = E_total <= E_bound

    result = {
        "n_eliminated": n_eliminated,
        "E_total": round(E_total, 4),
        "E_bound": round(E_bound, 4),
        "theorem_sat": theorem_sat,
        "heads_removed": total_heads_removed,
        "n_blocks": n_blocks,
        "k_alpha": round(k_alpha, 4),
        "adaptive_eps": bool(activation_stats),
    }

    if output_path:
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"summary": result, "decisions": decisions}, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  AEK v4 SONUÇ")
    print(f"  n_eliminated  : {n_eliminated}")
    print(f"  heads_removed : {total_heads_removed}")
    print(f"  E_total       : {E_total:.4f} / {E_bound:.4f}")
    print(f"  Teorem        : {'✓ SAĞLANDI' if theorem_sat else '✗ İHLAL'}")
    print(f"  Adaptive ε    : {'✓' if activation_stats else '✗ (flat)'}")
    print(f"{'='*55}")

    return result
