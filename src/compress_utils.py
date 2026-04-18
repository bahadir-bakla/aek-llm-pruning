"""
compress_utils.py
=================
α-EY-Kalman algoritmasının tüm bileşenleri — doğru sırayla entegre.

Pipeline sırası (compress_layer içinde):
  1. Laplace kutupları   → faz2_laplace.laplace_poles_real()
  2. α(n) gruplama       → faz2_alpha.alpha_union_find_adaptive()
  3. Grouped Rand SVD    → grouped_rand_svd()
  4. Kalman gain + eşik  → compress_layer() içinde
  5. Elim kararı         → compress_layer() döndürür

Public API:
  - randomized_eckart_young(W, eps, n_iter)
  - grouped_rand_svd(W, groups, eps, n_iter)
  - compress_layer(W, activations, P_tilde, eps, gamma)
  - full_compress(model, eps, delta, bits, n_iter, use_fisher)   ← backward compat
  - evaluate_perplexity(model, tokenizer, text)
  - optimal_rank_for_eps(sigmas, eps)
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.utils.extmath import randomized_svd as _rsvd

# faz2 modülleri — import hatası durumunda graceful fallback
try:
    from faz2_laplace import laplace_poles_real, _diagonal_proxy
    from faz2_alpha import alpha_union_find_adaptive
    _FAZ2_AVAILABLE = True
except ImportError:
    _FAZ2_AVAILABLE = False


H_OBS = 0.3


# ─── Yardımcı: eski kod uyumluluğu ──────────────────────────────────────────

def laplace_poles(W: np.ndarray) -> np.ndarray:
    """Eski proxy (aktivasyon yok). Geriye dönük uyumluluk için korundu."""
    n = min(W.shape)
    if W.shape[0] == W.shape[1]:
        return np.real(np.diag(W[:n, :n]))
    U, _, _ = np.linalg.svd(W, full_matrices=False)
    return np.real(np.diag(U[:n, :n]))


def union_find_groups(poles: np.ndarray, delta: float) -> List[List[int]]:
    """Sabit delta ile Union-Find. Geriye dönük uyumluluk için korundu."""
    n = len(poles)
    parent = list(range(n))
    rank_uf = [0] * n

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank_uf[rx] < rank_uf[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank_uf[rx] == rank_uf[ry]:
            rank_uf[rx] += 1

    for i in range(n):
        for j in range(i + 1, n):
            if abs(poles[i] - poles[j]) < delta:
                union(i, j)

    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def optimal_rank_for_eps(sigmas: np.ndarray, eps: float) -> int:
    for i, s in enumerate(sigmas):
        if s < eps:
            return max(1, i)
    return len(sigmas)


def compute_shadow_riccati(weight_norms: List[float],
                            residual_flags: List[bool],
                            P0: float) -> List[float]:
    P = P0
    P_vals = [P]
    for norm, res in zip(weight_norms, residual_flags):
        a = norm + 1.0 if res else norm
        P = a**2 * P
        P_vals.append(P)
    P_L = P_vals[-1]
    return [
        float(np.sqrt(P_L / max(P_vals[k], 1e-20)))
        for k in range(len(weight_norms))
    ]


def spectral_norm_approx(W: np.ndarray, n_iter: int = 3) -> float:
    """Power iteration ile spektral norm tahmini. O(mnk) yerine O(mn·n_iter)."""
    v = np.random.randn(W.shape[1]).astype(np.float32)
    v /= np.linalg.norm(v)
    for _ in range(n_iter):
        u = W @ v;  u /= np.linalg.norm(u)
        v = W.T @ u; v /= np.linalg.norm(v)
    return float(u @ W @ v)


def true_gamma(weight_norms: List[float],
               residual_flags: List[bool],
               k: int) -> float:
    g = 1.0
    for j in range(k, len(weight_norms)):
        a = weight_norms[j] + 1.0 if residual_flags[j] else weight_norms[j]
        g *= a
    return g


# ─── Gölge Riccati ───────────────────────────────────────────────────────────

def shadow_riccati_forward(
    n_layers: int,
    P0: float,
    gamma: float,
) -> List[float]:
    """
    Gölge Riccati ileri geçişi (Q=0).

    MATEMATIK_ZINCIRI.md Bölüm 2 Adım P2:
      P_{L|k} = γ_k² · P_k   (scalar yaklaşım, Q=0)
      γ̂_k    = √(P_{L|k} / P_k) = γ_k   (TAM EŞİTLİK — Lemma 1.5 kalbi)

    Sonuç: her k için γ̂_k = gamma (sabit).
    Karar eşiğinde kullanılır: threshold = (1-K)·ε / γ̂_k

    Returns: gamma_hat_list (List[float], uzunluk=n_layers)
    """
    return [float(gamma)] * n_layers


# ─── SVD Bileşenleri ─────────────────────────────────────────────────────────

def randomized_eckart_young(
    W: np.ndarray,
    eps: float,
    n_iter: int = 4,
) -> tuple:
    """
    Randomized SVD ile Eckart-Young truncation.

    Geleneksel SVD:  O(min(m,n)² × max(m,n))
    Randomized SVD:  O(m × n × k)  burada k << min(m,n)

    Adaptif k_init: eps büyüdükçe daha az bileşen yeterli.
      eps < 0.10  → k_init = 80% full_rank
      eps < 0.20  → k_init = 60%
      eps < 0.40  → k_init = 40%
      eps ≥ 0.40  → k_init = 20%

    Returns: (sigma_r1, rank_kept, U_k, s_k, Vt_k)
    """
    m, n = W.shape
    full_rank = min(m, n)

    if eps < 0.10:
        k_init = max(8, int(full_rank * 0.80))
    elif eps < 0.20:
        k_init = max(8, int(full_rank * 0.60))
    elif eps < 0.40:
        k_init = max(8, int(full_rank * 0.40))
    else:
        k_init = max(8, int(full_rank * 0.20))
    k_init = min(k_init, full_rank - 1)

    U, s, Vt = _rsvd(W, n_components=k_init, n_iter=n_iter, random_state=42)

    r = full_rank
    for i, sv in enumerate(s):
        if sv < eps:
            r = i
            break

    sigma_r1 = float(s[r]) if r < len(s) else 0.0
    return sigma_r1, r, U, s, Vt


def grouped_rand_svd(
    W: np.ndarray,
    groups: List[List[int]],
    eps: float,
    n_iter: int = 4,
) -> tuple:
    """
    Grup bilgisi ile Randomized SVD.

    Grup sayısı k_init'i belirler:
      - Az grup (yüksek tekrarlılık) → düşük k_init yeterli
      - Çok grup (düşük tekrarlılık) → yüksek k_init gerekli

    Tek elemanlı gruplar: direkt randomized_eckart_young.
    Çok elemanlı gruplar: grup sayısı × 3 üst sınır olarak kullanılır.

    Returns: (sigma_r1, rank_kept, U_k, s_k, Vt_k)
    """
    m, n = W.shape
    full_rank = min(m, n)
    n_groups = len(groups)

    # Küçük matrisler için full SVD — randomized SVD kuyruk SV'leri kaçırır
    # (k_proj, v_proj full_rank=128: first_idx of SV<eps genellikle 95-127)
    if full_rank <= 256:
        U, s, Vt = np.linalg.svd(W, full_matrices=False)
        r = full_rank
        for i, sv in enumerate(s):
            if sv < eps:
                r = i
                break
        sigma_r1 = float(s[r]) if r < len(s) else 0.0
        return sigma_r1, r, U, s, Vt

    # Büyük matrisler: adaptif k_hint ile randomized SVD
    if eps < 0.10:
        k_eps = max(8, int(full_rank * 0.85))
    elif eps < 0.20:
        k_eps = max(8, int(full_rank * 0.65))
    elif eps < 0.35:
        k_eps = max(8, int(full_rank * 0.45))
    else:
        k_eps = max(8, int(full_rank * 0.25))

    k_groups = max(8, n_groups * 3)
    k_hint = max(k_eps, k_groups)
    k_hint = min(k_hint, full_rank - 1)

    U, s, Vt = _rsvd(W, n_components=k_hint, n_iter=n_iter, random_state=42)

    r = full_rank
    for i, sv in enumerate(s):
        if sv < eps:
            r = i
            break

    sigma_r1 = float(s[r]) if r < len(s) else 0.0
    return sigma_r1, r, U, s, Vt


# ─── Ana Pipeline Fonksiyonu ─────────────────────────────────────────────────

def compress_layer(
    W: np.ndarray,
    activations: np.ndarray,
    P_tilde: float,
    eps: float,
    gamma: float = 1.0,
    gamma_hat: float = 1.0,
    weight_name: str = "",
    use_adaptive_eps: bool = False,
    use_qjl: bool = False,
    adaptive_safety_factor: float = 0.8,
    adaptive_mode: str = "shape",
) -> Dict[str, Any]:
    """
    Tek ağırlık matrisi için tam α-EY-Kalman pipeline (v2).

    MATEMATIK_ZINCIRI.md Bölüm 2 sırasına göre:
      1. Residual kontrol   → gamma_eff (Adım 1)
      2. Laplace kutupları  (Adım 3)
      3. α(n) gruplama      (Adım 4)
      4. [v2] Adaptive ε    (FAZ5: shape-based veya compressibility-based)
      5. Grouped Rand SVD   (Adım 5)
      6. Kalman gain K      (Adım 6)
      7. Eşik = (1-K)·ε     (Adım 7 — Lemma 1.5, Formülasyon B)
      8. Elim kararı
      9. P güncellemesi     (Adım 8)
      10. [v2] QJL sketch    (FAZ5: elinen katmanlar için)

    Args (v2 eklemeler):
      use_adaptive_eps:      True → her layer için adaptive ε hesapla
      adaptive_mode:         "shape" (default, aspect ratio tabanlı — GQA-aware)
                             "compressibility" (spektral decay tabanlı, eski)
      use_qjl:               True → elinen katmanlar için QJL residual sketch kaydet
      adaptive_safety_factor: compressibility modunda safety factor (0.8 default)

    Returns dict:
      eliminated, P_new, sigma_r1, threshold, K,
      rank_kept, rank_full, n_groups, poles_mean, poles_std,
      W_new, error_contrib,
      eps_used (v2: hangi ε kullanıldı),
      qjl_sketch, qjl_omega (v2: sadece use_qjl=True ve eliminated=True ise)
    """
    # Adım 1 — Residual kontrol
    is_residual = weight_name in ('o_proj', 'down_proj')
    gamma_eff   = gamma * 1.2 if is_residual else gamma

    # 2. Laplace kutupları
    if _FAZ2_AVAILABLE:
        poles = laplace_poles_real(W, activations)
    else:
        poles = laplace_poles(W)
    poles_used = poles

    # 3. α(n) gruplama
    if _FAZ2_AVAILABLE:
        groups = alpha_union_find_adaptive(poles_used, max_group_size=4)
    else:
        groups = union_find_groups(poles_used, delta=0.1)

    # 4. [v2] Adaptive ε kalibrasyonu
    if use_adaptive_eps:
        if adaptive_mode == "shape":
            # GQA-aware aspect ratio tabanlı (FAZ5 v2, önerilen)
            try:
                from faz5_adaptive_eps import adaptive_eps_by_shape
                eps_used = adaptive_eps_by_shape(W, global_eps=eps)
            except ImportError:
                eps_used = eps
        else:
            # Compressibility (spektral decay) tabanlı — FAZ5 v1 eski yöntem
            try:
                from faz5_adaptive_eps import layer_optimal_eps
                eps_info = layer_optimal_eps(W, global_eps=eps,
                                             safety_factor=adaptive_safety_factor)
                eps_used = eps_info["eps_optimal"]
            except ImportError:
                eps_used = eps
    else:
        eps_used = eps

    # 5. Grouped Rand SVD (adaptive eps ile)
    sigma_r1, r, U, s, Vt = grouped_rand_svd(W, groups, eps_used)

    # 6. Kalman gain
    K = (H_OBS**2 * P_tilde) / (H_OBS**2 * P_tilde + eps_used)

    # 7. Adaptif eşik (Formülasyon B)
    threshold = (1.0 - K) * eps_used

    # 8. Elim kararı
    eliminated = (sigma_r1 > 0.0) and (sigma_r1 < threshold)

    # 9. Kalman güncelleme
    Q_svd = sigma_r1 ** 2
    P_new = (1.0 - K) * P_tilde + Q_svd / max(gamma_eff ** 2, 1e-10)

    # Sıkıştırılmış ağırlık
    W_new = None
    error_contrib = 0.0
    qjl_sketch = None
    qjl_omega  = None

    if eliminated:
        r_use = max(1, r)
        W_new = (U[:, :r_use] * s[:r_use]) @ Vt[:r_use, :]
        error_contrib = sigma_r1 * (1.0 - K)

        # 10. [v2] QJL residual sketch
        if use_qjl:
            try:
                from faz5_qjl_correction import apply_qjl_correction
                qjl_sketch, qjl_omega, _, _ = apply_qjl_correction(
                    W_new, W, sketch_dim=64)
            except ImportError:
                pass

    return {
        "eliminated":    eliminated,
        "P_new":         float(P_new),
        "sigma_r1":      float(sigma_r1),
        "threshold":     float(threshold),
        "K":             float(K),
        "rank_kept":     int(r),
        "rank_full":     int(min(W.shape)),
        "n_groups":      int(len(groups)),
        "poles_mean":    float(np.mean(poles_used)),
        "poles_std":     float(np.std(poles_used)),
        "W_new":         W_new,
        "error_contrib": float(error_contrib),
        "eps_used":      float(eps_used),
        "qjl_sketch":    qjl_sketch,
        "qjl_omega":     qjl_omega,
    }


# ─── Perplexity ──────────────────────────────────────────────────────────────

def evaluate_perplexity(
    model, tokenizer, text: str, max_len: int = 512
) -> float:
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_len
    )
    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])
    return float(torch.exp(out.loss))


# ─── Ana Compress Fonksiyonu (Backward Compat) ───────────────────────────────

def _get_w(layer, name: str):
    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, name):
        return getattr(layer.self_attn, name).weight
    if hasattr(layer, "mlp") and hasattr(layer.mlp, name):
        return getattr(layer.mlp, name).weight
    return None


def full_compress(
    model,
    eps: float = 0.05,
    delta: float = 0.1,
    bits: int = 4,
    n_iter: int = 4,
    use_fisher: bool = False,
) -> None:
    """
    Modeli in-place sıkıştır.

    Eski imza (model, eps, delta, bits) korundu.
    İçeride yeni pipeline: Fisher P₀ → layer activations → compress_layer().

    use_fisher=False (default): hızlı mod, P₀=1.0 sabit başlar.
    use_fisher=True : Fisher diagonal ile per-layer P₀ başlatır.
    """
    layers = model.model.layers
    weight_names = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]

    # --- P₀ başlatma ---
    if use_fisher:
        try:
            from faz2_fisher import compute_diagonal_fisher, initialize_kalman_p0
            # Basit tokenizer al (model içinden)
            from transformers import AutoTokenizer
            _tok = AutoTokenizer.from_pretrained(
                model.config._name_or_path, trust_remote_code=True)
            fisher = compute_diagonal_fisher(model, _tok, n_samples=8)
            p0_list = initialize_kalman_p0(fisher, len(layers))
        except Exception as e:
            print(f"  [Fisher P₀] atlandı ({e}), P₀=1.0 kullanılıyor")
            p0_list = [1.0] * len(layers)
    else:
        p0_list = [1.0] * len(layers)

    # --- Aktivasyonlar ---
    try:
        from faz2_laplace import collect_all_layer_activations
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(
            model.config._name_or_path, trust_remote_code=True)
        all_acts = collect_all_layer_activations(model, _tok)
    except Exception as e:
        print(f"  [Aktivasyon] atlandı ({e}), sıfır aktivasyon kullanılıyor")
        hidden = model.config.hidden_size
        all_acts = {i: np.zeros((1, hidden), dtype=np.float32)
                    for i in range(len(layers))}

    # --- Gölge Riccati (Adım P2) ---
    gamma_theory = float(np.sqrt(model.config.hidden_size))
    gamma_hat_list = shadow_riccati_forward(len(layers), p0_list[0], gamma_theory)

    # --- Ana döngü ---
    Q_quant = (1.0 / (2 ** bits)) ** 2

    for k, layer in enumerate(layers):
        P_tilde = p0_list[k]
        acts    = all_acts.get(k, np.zeros((1, model.config.hidden_size)))

        # Katman norm (Riccati predict için)
        norms = []
        for wname in weight_names:
            wp = _get_w(layer, wname)
            if wp is None:
                continue
            W_np = wp.detach().float().cpu().numpy()
            norms.append(spectral_norm_approx(W_np, n_iter=3))
        a_norm = float(np.mean(norms)) if norms else 1.0
        P_tilde = P_tilde + Q_quant / max(a_norm ** 2, 1.0)

        for wname in weight_names:
            wp = _get_w(layer, wname)
            if wp is None:
                continue

            W_np = wp.detach().float().cpu().numpy()
            result = compress_layer(
                W_np, acts, P_tilde, eps,
                gamma=a_norm,
                gamma_hat=gamma_hat_list[k],
                weight_name=wname,
            )

            if result["eliminated"] and result["W_new"] is not None:
                with torch.no_grad():
                    wp.data = torch.tensor(
                        result["W_new"].astype(np.float32),
                        dtype=wp.dtype,
                    ).to(wp.device)

            P_tilde = result["P_new"]
