"""
faz2_ablation.py
================
5 konfigürasyon ablation çalışması — ε=0.20, Qwen 2.5 0.5B

Config A: Kalman + full SVD        (diag proxy kutuplar, trivial gruplar)
Config B: Kalman + rand SVD        (diag proxy kutuplar, trivial gruplar)
Config C: Kalman + rand SVD + Laplace kutupları + α(n) grupları
Config D: Kalman + rand SVD + Fisher P₀  (Laplace/α yok)
Config E: Tam sistem (rand SVD + Laplace + α(n) + Fisher P₀)

Her konfigürasyon için:
  - n_eliminated
  - E_total
  - PPL Δ%
  - elapsed_s

Sonuç: results/ablation.json
"""

import json, copy, time
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.utils.extmath import randomized_svd as _rsvd

from compress_utils import randomized_eckart_young, grouped_rand_svd, evaluate_perplexity
from faz2_laplace import laplace_poles_real, collect_all_layer_activations
from faz2_alpha import alpha_union_find_adaptive
from faz2_fisher import compute_diagonal_fisher, initialize_kalman_p0

MODEL_ID    = "Qwen/Qwen2.5-0.5B-Instruct"
EPS         = 0.20
H_OBS       = 0.3
BITS        = 4
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

SAMPLE_TEXT = "The capital of France is Paris. " * 20
WEIGHT_NAMES = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]


# ─── helpers ─────────────────────────────────────────────────────────────────

def _diagonal_proxy(W: np.ndarray) -> np.ndarray:
    n = min(W.shape)
    if W.shape[0] == W.shape[1]:
        return np.real(np.diag(W[:n, :n]))
    U, _, _ = np.linalg.svd(W, full_matrices=False)
    return np.real(np.diag(U[:n, :n]))


def _get_weight(layer, wname: str):
    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, wname):
        return getattr(layer.self_attn, wname).weight
    if hasattr(layer, "mlp") and hasattr(layer.mlp, wname):
        return getattr(layer.mlp, wname).weight
    return None


def _svd_truncate(W: np.ndarray, eps: float,
                  use_full_svd: bool,
                  groups=None) -> tuple:
    """
    Returns (sigma_r1, r, U, s, Vt)

    use_full_svd=True  → np.linalg.svd (Config A)
    groups is not None → grouped_rand_svd (Config C, E)
    else               → randomized_eckart_young (Config B, D)
    """
    if use_full_svd:
        U, s, Vt = np.linalg.svd(W, full_matrices=False)
        r = len(s)
        for i, sv in enumerate(s):
            if sv < eps:
                r = i
                break
        sigma_r1 = float(s[r]) if r < len(s) else 0.0
        return sigma_r1, r, U, s, Vt
    elif groups is not None:
        return grouped_rand_svd(W, groups, eps)
    else:
        return randomized_eckart_young(W, eps)


# ─── tek config çalıştır ─────────────────────────────────────────────────────

def run_config(
    model_orig,
    tokenizer,
    eps: float,
    use_full_svd: bool,
    use_laplace: bool,
    use_alpha: bool,
    use_fisher: bool,
    all_acts=None,
    fisher_p0=None,
) -> tuple:
    """
    Modeli kopyala, sıkıştır ve metrikleri döndür.

    Returns: (model_compressed, metrics_dict)
    """
    model_c = copy.deepcopy(model_orig)
    model_c.eval()
    layers   = model_c.model.layers
    n_layers = len(layers)

    # P₀ listesi
    if use_fisher and fisher_p0 is not None:
        p0_list = fisher_p0
    else:
        p0_list = [1.0] * n_layers

    Q_quant = (1.0 / (2 ** BITS)) ** 2
    n_eliminated = 0
    E_total = 0.0

    t0 = time.time()

    for k, layer in enumerate(layers):
        P_tilde = float(p0_list[k])
        acts    = all_acts[k] if all_acts is not None else None

        # Katman spektral norm → Riccati predict
        norms = []
        for wname in WEIGHT_NAMES:
            wp = _get_weight(layer, wname)
            if wp is None:
                continue
            W_np = wp.detach().float().cpu().numpy()
            norms.append(float(np.linalg.norm(W_np, ord=2)))
        a_norm = float(np.mean(norms)) if norms else 1.0
        P_tilde = P_tilde + Q_quant / max(a_norm ** 2, 1.0)

        for wname in WEIGHT_NAMES:
            wp = _get_weight(layer, wname)
            if wp is None:
                continue

            W_np = wp.detach().float().cpu().numpy()

            # Kutuplar
            if use_laplace and acts is not None and acts.shape[0] >= 2:
                poles = laplace_poles_real(W_np, acts)
            else:
                poles = _diagonal_proxy(W_np)

            # Gruplar
            if use_alpha:
                groups = alpha_union_find_adaptive(poles, max_group_size=4)
            else:
                groups = None  # randomized_eckart_young kullanılacak

            # SVD
            sigma_r1, r, U, s, Vt = _svd_truncate(
                W_np, eps, use_full_svd=use_full_svd, groups=groups
            )

            # Kalman gain + adaptif eşik
            K         = (H_OBS ** 2 * P_tilde) / (H_OBS ** 2 * P_tilde + eps)
            threshold = (1.0 - K) * eps

            # Elim kararı
            eliminated = (sigma_r1 > 0.0) and (sigma_r1 < threshold)

            # Kalman güncelle
            Q_svd = sigma_r1 ** 2
            P_new = (1.0 - K) * P_tilde + Q_svd / max(a_norm ** 2, 1e-10)

            if eliminated:
                r_use = max(1, r)
                W_new = (U[:, :r_use] * s[:r_use]) @ Vt[:r_use, :]
                E_total += float(sigma_r1 * (1.0 - K))
                n_eliminated += 1
                with torch.no_grad():
                    wp.data = torch.tensor(
                        W_new.astype(np.float32), dtype=wp.dtype
                    ).to(wp.device)

            P_tilde = float(P_new)

    elapsed = time.time() - t0

    metrics = {
        "n_eliminated": n_eliminated,
        "E_total":      float(E_total),
        "elapsed_s":    float(elapsed),
    }
    return model_c, metrics


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    print(f"Model yükleniyor: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model_orig = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, trust_remote_code=True
    )
    model_orig.eval()

    # Baseline PPL
    print("Baseline PPL hesaplanıyor...")
    ppl_base = evaluate_perplexity(model_orig, tokenizer, SAMPLE_TEXT)
    print(f"  PPL baseline = {ppl_base:.4f}")

    # Shared: aktivasyonlar + Fisher (Configs C/D/E için)
    print("\nAktivasyonlar toplanıyor...")
    all_acts = collect_all_layer_activations(model_orig, tokenizer)
    print(f"  Layer 0 shape: {all_acts[0].shape}")

    print("Fisher P₀ hesaplanıyor...")
    fisher_diag = compute_diagonal_fisher(model_orig, tokenizer, n_samples=8)
    fisher_p0   = initialize_kalman_p0(fisher_diag, len(model_orig.model.layers))
    print(f"  P₀ min={min(fisher_p0):.4f}  max={max(fisher_p0):.4f}")

    # Config tanımları
    configs = [
        dict(name="A", label="Kalman + full SVD",
             use_full_svd=True,  use_laplace=False, use_alpha=False, use_fisher=False,
             all_acts=None,      fisher_p0=None),
        dict(name="B", label="Kalman + rand SVD",
             use_full_svd=False, use_laplace=False, use_alpha=False, use_fisher=False,
             all_acts=None,      fisher_p0=None),
        dict(name="C", label="Kalman + rand SVD + Laplace + α(n)",
             use_full_svd=False, use_laplace=True,  use_alpha=True,  use_fisher=False,
             all_acts=all_acts,  fisher_p0=None),
        dict(name="D", label="Kalman + rand SVD + Fisher P₀",
             use_full_svd=False, use_laplace=False, use_alpha=False, use_fisher=True,
             all_acts=None,      fisher_p0=fisher_p0),
        dict(name="E", label="Tam sistem (rand SVD + Laplace + α(n) + Fisher P₀)",
             use_full_svd=False, use_laplace=True,  use_alpha=True,  use_fisher=True,
             all_acts=all_acts,  fisher_p0=fisher_p0),
    ]

    print(f"\n{'─'*72}")
    print(f"{'Config':<6} {'n_elim':>7} {'E_total':>9} {'PPL_base':>10} "
          f"{'PPL_c':>10} {'ΔPPL%':>8} {'t(s)':>7}")
    print(f"{'─'*72}")

    results = []
    for cfg in configs:
        name  = cfg.pop("name")
        label = cfg.pop("label")

        model_c, metrics = run_config(model_orig, tokenizer, EPS, **cfg)

        ppl_c     = evaluate_perplexity(model_c, tokenizer, SAMPLE_TEXT)
        delta_pct = (ppl_c - ppl_base) / ppl_base * 100

        metrics["ppl_baseline"] = float(ppl_base)
        metrics["ppl_compressed"] = float(ppl_c)
        metrics["ppl_delta_pct"]  = float(delta_pct)

        print(f"  {name:<4} {metrics['n_eliminated']:>7} "
              f"{metrics['E_total']:>9.4f} "
              f"{ppl_base:>10.4f} {ppl_c:>10.4f} "
              f"{delta_pct:>+8.2f}% "
              f"{metrics['elapsed_s']:>6.1f}s")

        results.append({"config": name, "label": label, **metrics})

    print(f"{'─'*72}")

    output = {
        "model":   MODEL_ID,
        "eps":     EPS,
        "results": results,
    }
    out_path = RESULTS_DIR / "ablation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nKaydedildi: {out_path}")


if __name__ == "__main__":
    main()
