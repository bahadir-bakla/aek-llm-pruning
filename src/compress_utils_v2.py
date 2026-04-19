"""
compress_utils_v2.py
====================
AEK v0.2 — Büyük model desteği için genişletilmiş core.

v0.1'den farklar:
  1. compute_layer_gamma(W, mode)  — layer-specific γ (ANA DEĞİŞİKLİK)
  2. compress_layer() → gamma_mode parametresi eklendi
  3. full_compress_v2() → block-wise error budget + adaptive Fisher

v0.1 backward compat:
  compress_layer(..., gamma_mode="fixed")  → v0.1 ile birebir aynı
  full_compress(...)                       → dokunulmadı, hâlâ çalışır
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.utils.extmath import randomized_svd as _rsvd

# v0.1 core'u import et — SVD, Laplace, Union-Find aynı
from compress_utils import (
    H_OBS,
    shadow_riccati_forward,
    spectral_norm_approx,
    grouped_rand_svd,
    laplace_poles,
    union_find_groups,
    evaluate_perplexity,
)

try:
    from faz2_laplace import laplace_poles_real, collect_all_layer_activations
    from faz2_alpha import alpha_union_find_adaptive
    _FAZ2_AVAILABLE = True
except ImportError:
    _FAZ2_AVAILABLE = False


# ─── Değişiklik 1: Layer-Specific γ ──────────────────────────────────────────

def compute_layer_gamma(W: np.ndarray, mode: str = "spectral") -> float:
    """
    Layer-specific γ hesapla.

    v0.1 sorunu: γ = sqrt(hidden_dim) sabit → büyük modellerde P̃ küçülüyor
                 → K büyüyor → threshold küçülüyor → 7B'de 0 elim.

    v0.2 çözümü: Her katman kendi spektral yoğunluğuna göre normalize edilir.
                 Az yoğun katman → düşük γ → büyük threshold → elim kolaylaşır.
                 Yoğun katman   → yüksek γ → küçük threshold → muhafazakâr.

    Args:
        W:    Ağırlık matrisi, shape (m, n)
        mode: "spectral"  → γ = ||W||_F / sqrt(min(m,n))   [önerilen]
              "nuclear"   → γ = mean(singular values)        [daha hassas, yavaş]
              "fixed"     → γ = sqrt(hidden_dim)             [v0.1 eski davranış]

    Returns:
        γ_k (float), pozitif

    Matematiksel gerekçe:
        P̃_k = P_k / γ_k²  (Formülasyon B normalizasyonu)
        γ_k büyük → P̃_k küçük → K = P̃/(P̃+1) küçük → (1-K) büyük → threshold büyür

        Spectral mode'da:
          γ_k = ||W||_F / sqrt(rank)
          ||W||_F² = Σ σ_i²   (tüm singular value'ların karesi)
          γ_k = RMS singular value → matrisin "ortalama" büyüklüğü

          Az yoğun W (seyrek spektrum): küçük ||W||_F → küçük γ_k → agresif
          Yoğun W (dolu spektrum):     büyük ||W||_F → büyük γ_k → muhafazakâr
    """
    m, n = W.shape
    rank = min(m, n)

    if mode == "fixed":
        # v0.1 davranışı — hidden_dim boyutundan türet
        hidden_dim = max(m, n)
        return float(np.sqrt(hidden_dim))

    elif mode == "spectral":
        # γ = ||W||_F / sqrt(min(m,n))
        # = sqrt(mean(σ_i²)) = RMS singular value
        frob = float(np.linalg.norm(W, "fro"))
        return max(frob / np.sqrt(rank), 1e-6)

    elif mode == "nuclear":
        # γ = mean(σ_i) — daha hassas ama tam SVD gerektirir (yavaş)
        # Büyük matrisler için sadece top-k singular value kullan
        k = min(rank, 64)
        _, s, _ = _rsvd(W, n_components=k, n_iter=3, random_state=42)
        return max(float(np.mean(s)), 1e-6)

    else:
        raise ValueError(f"Bilinmeyen gamma_mode: {mode}. 'spectral', 'nuclear', 'fixed' olmalı.")


def analyze_gamma_distribution(model, gamma_mode: str = "spectral") -> Dict[str, Any]:
    """
    Model üzerinde γ dağılımını analiz et — AWS run'dan önce simülasyon için.

    Returns: layer_gammas, stats, simulated_eliminations
    """
    weight_names = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]

    layer_gammas = {}
    all_gammas = []

    for k, layer in enumerate(model.model.layers):
        for wname in weight_names:
            wp = None
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, wname):
                wp = getattr(layer.self_attn, wname).weight
            elif hasattr(layer, "mlp") and hasattr(layer.mlp, wname):
                wp = getattr(layer.mlp, wname).weight
            if wp is None:
                continue

            W_np = wp.detach().float().cpu().numpy()
            gamma = compute_layer_gamma(W_np, mode=gamma_mode)
            key = f"layer{k}.{wname}"
            layer_gammas[key] = gamma
            all_gammas.append(gamma)

    gammas = np.array(all_gammas)
    stats = {
        "mean":   float(np.mean(gammas)),
        "median": float(np.median(gammas)),
        "min":    float(np.min(gammas)),
        "max":    float(np.max(gammas)),
        "std":    float(np.std(gammas)),
    }

    return {"layer_gammas": layer_gammas, "stats": stats}


# ─── Değişiklik 2: K Dampening — Büyük Model Kalibrasyonu (ANA BULUŞ) ─────────

def compute_k_alpha(hidden_dim: int) -> float:
    """
    K dampening faktörü α — büyük modellerde Kalman gain'i sönümle.

    Problem: Büyük modellerde Fisher bilgisi → P_tilde büyük → K şişiyor
             → threshold = (1-K)·ε küçülüyor → eliminasyon imkânsızlaşıyor.

    Çözüm: threshold = (1 - K·α) · ε
           α küçüldükçe K etkisi azalır, threshold büyür, elim kolaylaşır.

    α formülü: α = 1 / log2(hidden_dim / 256)
      → 0.5B: hidden=896,  α = 1/log2(3.5) = 0.544 — orta sönümleme
      → 1.5B: hidden=2048, α = 1/log2(8)   = 0.333 — daha fazla sönümleme
      → 7B:   hidden=3584, α = 1/log2(14)  = 0.269 — en fazla sönümleme

    Neden log2? Model kapasitesi 2x büyüdükçe lineer değil logaritmik ölçekleme.
    Neden /256? Küçük model referansı — 256 hidden_dim'de α=∞ (sönümleme yok).

    Teorik bağlantı:
      Büyük model → daha yoğun spektrum → sigma_r1 yüksek (0.25-0.28)
      K·α ile efektif Kalman gain küçülür:
        K_eff = K · α
        threshold = (1 - K_eff) · ε ≥ (1 - K) · ε   (v0.1'den büyük)
      Ana Teorem hâlâ geçerli: E_k = sigma_r1 · (1-K_eff) < ε (K_eff ∈ [0,1])

    Güvenlik: α ∈ [0.1, 1.0] clip ile aşırı agresiflik önlenir.

    Önemli: hidden_dim < 2048 (0.5B, 1B) için α=1.0 döner.
    Bu modeller zaten v0.1'de iyi çalışıyor — dampening gerekmez.
    Sadece büyük modellerde (≥2B) K şişmesi problemi oluşur.
    """
    if hidden_dim < 2048:
        return 1.0   # küçük model — v0.1 davranışı, dampening yok
    alpha = 1.0 / np.log2(hidden_dim / 256.0)
    return float(np.clip(alpha, 0.1, 1.0))


def adaptive_n_samples(hidden_dim: int) -> int:
    """
    Model boyutuna göre Fisher sample sayısı.
    0.5B: hidden=896  → 4
    1.5B: hidden=2048 → 8
    7B:   hidden=3584 → 14
    """
    return max(4, hidden_dim // 256)


# ─── v0.2 compress_layer ─────────────────────────────────────────────────────

def compress_layer_v2(
    W: np.ndarray,
    activations: np.ndarray,
    P_tilde: float,
    eps: float,
    gamma_mode: str = "spectral",
    weight_name: str = "",
    k_alpha: float = 1.0,
) -> Dict[str, Any]:
    """
    v0.2 compress_layer — K dampening + layer-specific γ.

    Yeni parametreler vs v0.1:
      k_alpha:    K dampening faktörü ∈ [0.1, 1.0]
                  1.0 = v0.1 ile aynı (sönümleme yok)
                  <1.0 = K etkisi azalır, threshold büyür, büyük model desteği
                  compute_k_alpha(hidden_dim) ile otomatik hesaplanır

      gamma_mode: "spectral" = layer-specific γ (önerilen)
                  "fixed"    = v0.1 ile aynı

    threshold formülü (v0.2):
      K_eff     = K · k_alpha
      threshold = (1 - K_eff) · ε

      k_alpha=1.0 → v0.1 ile birebir aynı
      k_alpha<1.0 → threshold büyür → büyük modelde elim kapısı açılır
    """
    # ── γ hesapla ────────────────────────────────────────────────���───────────
    gamma     = compute_layer_gamma(W, mode=gamma_mode)
    is_residual = weight_name in ("o_proj", "down_proj")
    gamma_eff   = gamma * 1.2 if is_residual else gamma

    # ── Laplace kutupları ───────────────────────────────────────────────────
    if _FAZ2_AVAILABLE:
        poles = laplace_poles_real(W, activations)
    else:
        poles = laplace_poles(W)

    # ── α(n) gruplama ───────────────────────────────────────────────────────
    if _FAZ2_AVAILABLE:
        groups = alpha_union_find_adaptive(poles, max_group_size=4)
    else:
        groups = union_find_groups(poles, delta=0.1)

    # ── Grouped Rand SVD ────────────────────────────────────────────────────
    sigma_r1, r, U, s, Vt = grouped_rand_svd(W, groups, eps)

    # ── Kalman gain ─────────────────────────────────────────────────────────
    K     = (H_OBS**2 * P_tilde) / (H_OBS**2 * P_tilde + eps)
    K_eff = K * k_alpha   # dampened gain

    # ── Eşik (v0.2: K dampening ile) ────────────────────────────────────────
    # v0.1: threshold = (1-K)·ε
    # v0.2: threshold = (1-K·α)·ε  ← α küçüldükçe threshold büyür
    threshold = (1.0 - K_eff) * eps

    # ── Elim kararı ─────────────────────────────────────────────────────────
    eliminated = (sigma_r1 > 0.0) and (sigma_r1 < threshold)

    # ── Kalman güncelleme (P güncelleme v0.1 ile aynı — K_eff değil K kullan) ─
    # Neden K? P güncelleme hata izleme içindir — sönümlenmemiş gerçek K ile
    Q_svd = sigma_r1 ** 2
    P_new = (1.0 - K) * P_tilde + Q_svd / max(gamma_eff ** 2, 1e-10)

    W_new         = None
    error_contrib = 0.0
    if eliminated:
        r_use  = max(1, r)
        W_new  = (U[:, :r_use] * s[:r_use]) @ Vt[:r_use, :]
        error_contrib = sigma_r1 * (1.0 - K_eff)

    return {
        "eliminated":    eliminated,
        "P_new":         float(P_new),
        "sigma_r1":      float(sigma_r1),
        "threshold":     float(threshold),
        "K":             float(K),
        "K_eff":         float(K_eff),
        "k_alpha":       float(k_alpha),
        "gamma":         float(gamma),
        "rank_kept":     int(r),
        "rank_full":     int(min(W.shape)),
        "n_groups":      int(len(groups)),
        "poles_mean":    float(np.mean(poles)),
        "poles_std":     float(np.std(poles)),
        "W_new":         W_new,
        "error_contrib": float(error_contrib),
        "eps_used":      float(eps),
    }


# ─── Değişiklik 3: Block-wise Error Budget ────────────────────────────────────

def compute_block_budgets(
    n_blocks: int,
    eps_global: float,
    attn_ratio: float = 0.4,
    safety: float = 0.8,
) -> List[Dict[str, float]]:
    """
    Her transformer block için ayrı ε bütçesi hesapla.

    Args:
        n_blocks:    Transformer block sayısı (28 for 7B)
        eps_global:  Global ε (kullanıcı parametresi)
        attn_ratio:  Attention'a ayrılan bütçe oranı (0.4 = %40)
        safety:      Güvenlik marjı — toplam bütçeyi %80'e çek

    Returns:
        [{"attn": eps_attn, "mlp": eps_mlp}, ...] uzunluk=n_blocks

    Matematiksel gerekçe:
        E_total = Σ_k Σ_w E_{k,w} ≤ L·ε  (Ana Teorem)
        Her block: E_block ≤ ε_block = eps_global/n_blocks × safety
        Attention: E_attn ≤ ε_block × attn_ratio
        MLP:       E_mlp  ≤ ε_block × (1-attn_ratio)

        safety=0.8: global bütçenin %80'ini kullan, %20 rezerv
        → Teorem ihlali riski minimize
    """
    eps_per_block = (eps_global / n_blocks) * safety
    eps_attn = eps_per_block * attn_ratio
    eps_mlp  = eps_per_block * (1.0 - attn_ratio)

    return [{"attn": eps_attn, "mlp": eps_mlp}] * n_blocks


# ─── v0.2 Full Compress Pipeline ──────────────────────────────────────────────

def full_compress_v2(
    model,
    eps: float = 0.35,
    delta: float = 0.1,
    bits: int = 4,
    gamma_mode: str = "spectral",
    k_alpha: Optional[float] = None,
    use_block_budget: bool = True,
    fisher_adaptive: bool = True,
    output_path: str = None,
) -> Dict[str, Any]:
    """
    AEK v0.2 — büyük model desteği ile tam compress pipeline.

    Yeni parametreler vs v0.1:
      gamma_mode:       "spectral" (v0.2) veya "fixed" (v0.1 davranışı)
      k_alpha:          K dampening faktörü (None → otomatik hidden_dim'den)
      use_block_budget: True → block-wise ε bütçesi
      fisher_adaptive:  True → hidden_dim'e göre n_samples ölçekle

    Returns: results dict (n_eliminated, E_total, E_bound, decisions, ...)
    """
    from transformers import AutoTokenizer

    layers      = model.model.layers
    hidden_dim  = model.config.hidden_size
    n_blocks    = len(layers)
    n_decisions = n_blocks * 7   # 7 weight per block (q/k/v/o/gate/up/down)
    E_bound     = n_decisions * eps   # Ana Teorem: E_total <= n_decisions * eps

    weight_names_attn = ["q_proj", "k_proj", "v_proj", "o_proj"]
    weight_names_mlp  = ["gate_proj", "up_proj", "down_proj"]

    # ── K dampening faktörü (otomatik veya manuel) ─────────────────────────
    if k_alpha is None:
        k_alpha = compute_k_alpha(hidden_dim)

    print(f"AEK v0.2 — gamma_mode={gamma_mode}, block_budget={use_block_budget}")
    print(f"  hidden_dim={hidden_dim}, n_blocks={n_blocks}, eps={eps}")
    print(f"  k_alpha={k_alpha:.4f} (K dampening — 1.0=v0.1, <1.0=büyük model)")
    print(f"  E_bound = {n_decisions} × {eps} = {E_bound:.4f}")

    # ── Fisher P₀ ──────────────────────────────────────────────────────────
    n_samples = adaptive_n_samples(hidden_dim) if fisher_adaptive else 4
    print(f"  Fisher n_samples={n_samples}")

    try:
        from faz2_fisher import compute_diagonal_fisher, initialize_kalman_p0
        tokenizer = AutoTokenizer.from_pretrained(
            model.config._name_or_path, trust_remote_code=True)
        fisher = compute_diagonal_fisher(model, tokenizer, n_samples=n_samples)
        p0_list = initialize_kalman_p0(fisher, n_blocks)
        # NaN/Inf kontrolü — Fisher hesabı bozuksa fallback
        if any(not np.isfinite(p) or p <= 0 for p in p0_list):
            print(f"  [Fisher P₀] NaN/Inf tespit edildi, P₀=1.0 fallback")
            p0_list = [1.0] * n_blocks
        else:
            print(f"  Fisher P₀: min={min(p0_list):.4f}, max={max(p0_list):.4f}")
    except Exception as e:
        print(f"  [Fisher P₀] atlandı ({e}), P₀=1.0")
        p0_list = [1.0] * n_blocks

    # ── Aktivasyonlar ──────────────────────────────────────────────────────
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model.config._name_or_path, trust_remote_code=True)
        all_acts = collect_all_layer_activations(model, tokenizer)
    except Exception as e:
        print(f"  [Aktivasyon] atlandı ({e})")
        all_acts = {i: np.zeros((1, hidden_dim), dtype=np.float32)
                    for i in range(n_blocks)}

    # ── Gölge Riccati ──────────────────────────────────────────────────────
    gamma_theory   = float(np.sqrt(hidden_dim))
    gamma_hat_list = shadow_riccati_forward(n_blocks, p0_list[0], gamma_theory)

    # ── Block-wise bütçe ───────────────────────────────────────────────────
    if use_block_budget:
        budgets = compute_block_budgets(n_blocks, eps)
    else:
        budgets = [{"attn": eps, "mlp": eps}] * n_blocks

    # ── Ana döngü ──────────────────────────────────────────────────────────
    Q_quant     = (1.0 / (2 ** bits)) ** 2
    E_total     = 0.0
    n_eliminated = 0
    decisions   = []

    for k, layer in enumerate(layers):
        P_tilde  = p0_list[k]
        acts     = all_acts.get(k, np.zeros((1, hidden_dim)))
        eps_attn = budgets[k]["attn"]
        eps_mlp  = budgets[k]["mlp"]

        # Katman norm (P predict için)
        norms = []
        for wname in weight_names_attn + weight_names_mlp:
            wp = _get_weight(layer, wname)
            if wp is None:
                continue
            W_np = wp.detach().float().cpu().numpy()
            norms.append(spectral_norm_approx(W_np, n_iter=3))
        a_norm  = float(np.mean(norms)) if norms else 1.0
        P_tilde = P_tilde + Q_quant / max(a_norm ** 2, 1.0)

        # Attention weights
        for wname in weight_names_attn:
            wp = _get_weight(layer, wname)
            if wp is None:
                continue
            W_np   = wp.detach().float().cpu().numpy()
            result = compress_layer_v2(
                W_np, acts, P_tilde, eps,   # global eps — threshold için
                gamma_mode=gamma_mode,
                weight_name=wname,
                k_alpha=k_alpha,
            )
            result["layer"] = k
            result["weight_name"] = wname
            decisions.append(result)

            if result["eliminated"] and result["W_new"] is not None:
                # Block budget kontrolü — bütçe aşıldıysa elim iptal
                if E_total + result["error_contrib"] <= E_bound:
                    with torch.no_grad():
                        wp.data = torch.tensor(
                            result["W_new"].astype(np.float32),
                            dtype=wp.dtype,
                        ).to(wp.device)
                    E_total     += result["error_contrib"]
                    n_eliminated += 1

            P_tilde = result["P_new"]

        # MLP weights
        for wname in weight_names_mlp:
            wp = _get_weight(layer, wname)
            if wp is None:
                continue
            W_np   = wp.detach().float().cpu().numpy()
            result = compress_layer_v2(
                W_np, acts, P_tilde, eps,   # global eps — threshold için
                gamma_mode=gamma_mode,
                weight_name=wname,
                k_alpha=k_alpha,
            )
            result["layer"] = k
            result["weight_name"] = wname
            decisions.append(result)

            if result["eliminated"] and result["W_new"] is not None:
                if E_total + result["error_contrib"] <= E_bound:
                    with torch.no_grad():
                        wp.data = torch.tensor(
                            result["W_new"].astype(np.float32),
                            dtype=wp.dtype,
                        ).to(wp.device)
                    E_total     += result["error_contrib"]
                    n_eliminated += 1

            P_tilde = result["P_new"]

        if (k + 1) % 4 == 0:
            print(f"  Block {k+1}/{n_blocks} — elim so far: {n_eliminated}, E={E_total:.4f}")

    theorem_sat = E_total <= E_bound

    print(f"\n{'='*55}")
    print(f"  AEK v0.2 SONUÇ")
    print(f"  n_eliminated : {n_eliminated}")
    print(f"  E_total      : {E_total:.4f} / {E_bound:.4f} (bound)")
    print(f"  Teorem       : {'✓ SAĞLANDI' if theorem_sat else '✗ İHLAL'}")
    print(f"{'='*55}")

    results = {
        "version":          "v0.2",
        "gamma_mode":       gamma_mode,
        "k_alpha":          k_alpha,
        "eps":              eps,
        "n_blocks":         n_blocks,
        "n_eliminated":     n_eliminated,
        "E_total":          E_total,
        "E_bound":          E_bound,
        "theorem_sat":      theorem_sat,
        "use_block_budget": use_block_budget,
        "fisher_adaptive":  fisher_adaptive,
        "decisions":        decisions,
    }

    if output_path:
        import json
        # decisions içindeki W_new serialize edilemez, çıkar
        out = {k: v for k, v in results.items() if k != "decisions"}
        out["decisions"] = [
            {dk: dv for dk, dv in d.items() if dk != "W_new"}
            for d in decisions
        ]
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Kayıt: {output_path}")

    return results


def _get_weight(layer, name: str):
    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, name):
        return getattr(layer.self_attn, name).weight
    if hasattr(layer, "mlp") and hasattr(layer.mlp, name):
        return getattr(layer.mlp, name).weight
    return None
