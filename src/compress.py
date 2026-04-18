"""
02b_compress_fixed.py
=====================
α-EY-Kalman — P normalizasyonu ile düzeltilmiş versiyon.

Problem: Qwen'da ‖W‖₂ ~ 35, A = W+I → P 10^175'e patlıyor → threshold = 0

Fix: P̃_k = P_k / γ_k² takip et.
  Riccati basitleşir: P̃_{k+1} = (1/a²) * (a² * P̃_k * a² + Q) / a²
                               = P̃_k + Q / a²

  Threshold: σ_{r+1} < (1 - K̃) · ε   (γ̂ artık gerekmiyor)

Çalıştır:
  python 02b_compress_fixed.py --eps 0.05
"""

import argparse
import json
import copy
import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--eps',   type=float, default=0.05)
parser.add_argument('--delta', type=float, default=0.1)
parser.add_argument('--bits',  type=int,   default=4)
parser.add_argument('--model', type=str,
                    default='Qwen/Qwen2.5-0.5B-Instruct')
parser.add_argument('--output', type=str, default='compression_fixed_log.json')
parser.add_argument('--dtype',  type=str, default='float32',
                    choices=['float32', 'bfloat16'],
                    help='Model yükleme dtype (büyük modeller için bfloat16)')
parser.add_argument('--fisher_samples', type=int, default=8,
                    help='Fisher P₀ için backward pass sayısı (büyük modeller için 2-4)')
# v2 argümanları (FAZ5)
parser.add_argument('--use_adaptive_eps', action='store_true',
                    help='[v2] Layer-adaptive ε kalibrasyonu (faz5_adaptive_eps)')
parser.add_argument('--use_qjl', action='store_true',
                    help='[v2] QJL residual bias correction sketch (faz5_qjl_correction)')
parser.add_argument('--adaptive_safety', type=float, default=0.8,
                    help='[v2] Adaptive ε safety factor (default=0.8)')
parser.add_argument('--adaptive_mode', type=str, default='shape',
                    choices=['shape', 'compressibility'],
                    help='[v2] Adaptive ε modu: shape=aspect-ratio tabanlı (GQA-aware, önerilen), '
                         'compressibility=spektral decay tabanlı (eski)')
parser.add_argument('--save_model_dir', type=str, default=None,
                    help='AEK sonrası modeli bu dizine kaydet (HuggingFace format). '
                         'Benchmark için gerekli.')
parser.add_argument('--no_plot', action='store_true',
                    help='Matplotlib grafiği kaydetme (dosya sistemi sorunlarında kullan)')
args = parser.parse_args()

MODEL_ID         = args.model
EPS              = args.eps
DELTA            = args.delta
BITS             = args.bits
H_OBS            = 0.3
RESULTS_DIR      = Path('./results')
tokenizer_global = None   # main() tarafından set edilir
RESULTS_DIR.mkdir(exist_ok=True)

_v2_tag = " v2" if (args.use_adaptive_eps or args.use_qjl) else ""
print("=" * 60)
print(f"α-EY-Kalman (Normalized P){_v2_tag}")
print(f"Model: {MODEL_ID}")
print(f"ε={EPS}  δ={DELTA}  bits={BITS}")
if args.use_adaptive_eps:
    print(f"[v2] Adaptive ε: ON  (mode={args.adaptive_mode}, safety={args.adaptive_safety})")
if args.use_qjl:
    print(f"[v2] QJL correction: ON  (sketch_dim=64)")
print("=" * 60)


@dataclass
class LayerDecision:
    k:             int
    weight_name:   str
    shape:         list
    P_tilde:       float   # normalized P
    K:             float
    sigma_r1:      float
    threshold:     float
    eliminated:    bool
    error_contrib: float
    rank_kept:     int
    rank_full:     int
    n_groups:      int     # α(n): kaç grup oluştu
    poles_mean:    float   # Laplace kutup ortalaması
    poles_std:     float   # Laplace kutup std
    gamma_hat:     float   # γ̂_k — bilgi amaçlı, threshold'da kullanılmaz


def get_w(layer, name: str):
    if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, name):
        return getattr(layer.self_attn, name).weight
    if hasattr(layer, 'mlp') and hasattr(layer.mlp, name):
        return getattr(layer.mlp, name).weight
    return None


def _model_device(model):
    """device_map='auto' ile yüklenen modelde ilk parametrenin cihazını döndür."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cpu')


def evaluate_perplexity(model, tokenizer, text: str) -> float:
    inputs = tokenizer(text, return_tensors='pt',
                       truncation=True, max_length=512)
    dev = _model_device(model)
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, labels=inputs['input_ids'])
    return float(torch.exp(out.loss))


def evaluate_quality(model, tokenizer) -> List[dict]:
    prompts = [
        "What is the capital of France?",
        "Explain recursion in programming.",
        "Who wrote '1984'?",
        "What is 2 + 2?",
    ]
    dev = _model_device(model)
    results = []
    for p in prompts:
        inputs = tokenizer(p, return_tensors='pt')
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
        response = tokenizer.decode(
            out[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True)
        results.append({'prompt': p, 'response': response})
    return results


def compress_normalized(model, eps=EPS, delta=DELTA, bits=BITS):
    """
    Normalized Riccati ile α-EY-Kalman pipeline.

    Sıra: Fisher P₀ → Aktivasyonlar → compress_layer() per weight
    Threshold: σ_{r+1} < (1 - K̃) · ε
    """
    from compress_utils import compress_layer, shadow_riccati_forward
    from faz2_laplace import collect_all_layer_activations
    from faz2_fisher import compute_diagonal_fisher, initialize_kalman_p0

    layers = model.model.layers
    L = len(layers)
    weight_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj']
    Q_quant = (1.0 / (2**bits))**2

    # --- Fisher P₀ ---
    print("  [Fisher] P₀ hesaplanıyor...")
    try:
        n_fs = args.fisher_samples if args.fisher_samples > 0 else 4
        fisher = compute_diagonal_fisher(model, tokenizer_global, n_samples=n_fs)
        p0_list = initialize_kalman_p0(fisher, L)
        print(f"  [Fisher] P₀ aralığı: [{min(p0_list):.3f}, {max(p0_list):.3f}]")
    except Exception as e:
        print(f"  [Fisher] atlandı ({e}), P₀=1.0")
        p0_list = [1.0] * L

    # --- Aktivasyonlar (tek pass) ---
    print("  [Aktivasyon] Tüm layer'lar toplanıyor...")
    try:
        all_acts = collect_all_layer_activations(model, tokenizer_global)
        print(f"  [Aktivasyon] {len(all_acts)} layer, "
              f"örnek şekil: {next(iter(all_acts.values())).shape}")
    except Exception as e:
        print(f"  [Aktivasyon] atlandı ({e})")
        hidden = model.config.hidden_size
        all_acts = {i: np.zeros((1, hidden), dtype=np.float32)
                    for i in range(L)}

    # --- Gölge Riccati (MATEMATIK_ZINCIRI.md Adım P2) ---
    import math
    gamma_theory   = math.sqrt(model.config.hidden_size)
    gamma_hat_list = shadow_riccati_forward(L, p0_list[0], gamma_theory)
    print(f"  [Shadow Riccati] γ̂_0={gamma_hat_list[0]:.4f}  "
          f"γ̂_23={gamma_hat_list[23]:.4f}  (γ=√hidden={gamma_theory:.4f})")

    # --- Ana döngü ---
    decisions: List[LayerDecision] = []
    E_total = 0.0
    t0 = time.time()

    for k, layer in enumerate(layers):
        P_tilde = p0_list[k]
        acts = all_acts.get(k, np.zeros((1, model.config.hidden_size)))

        # Katman spektral normu
        norms = []
        for wname in weight_names:
            wp = get_w(layer, wname)
            if wp is None: continue
            W_np = wp.detach().float().cpu().numpy()
            from compress_utils import spectral_norm_approx
            norms.append(spectral_norm_approx(W_np, n_iter=3))
        a_norm = float(np.mean(norms)) if norms else 1.0

        # Normalized Riccati predict
        P_tilde_pred = P_tilde + Q_quant / max(a_norm**2, 1.0)

        last_K = 0.0
        for wname in weight_names:
            wp = get_w(layer, wname)
            if wp is None: continue

            W = wp.detach().float().cpu().numpy()

            # ── TAM PIPELINE ──────────────────────────────────────────
            result = compress_layer(
                W, acts, P_tilde_pred, eps,
                gamma=a_norm,
                gamma_hat=gamma_hat_list[k],
                weight_name=wname,
                use_adaptive_eps=args.use_adaptive_eps,
                use_qjl=args.use_qjl,
                adaptive_safety_factor=args.adaptive_safety,
                adaptive_mode=args.adaptive_mode,
            )
            # ─────────────────────────────────────────────────────────

            if result['eliminated'] and result['W_new'] is not None:
                wp.data = torch.tensor(
                    result['W_new'].astype(np.float32),
                    dtype=wp.dtype).to(wp.device)
                E_total += result['error_contrib']

            decisions.append(LayerDecision(
                k=k,
                weight_name=wname,
                shape=list(W.shape),
                P_tilde=float(P_tilde_pred),
                K=float(result['K']),
                sigma_r1=float(result['sigma_r1']),
                threshold=float(result['threshold']),
                eliminated=result['eliminated'],
                error_contrib=float(result['error_contrib']),
                rank_kept=int(result['rank_kept']),
                rank_full=int(result['rank_full']),
                n_groups=int(result['n_groups']),
                poles_mean=float(result['poles_mean']),
                poles_std=float(result['poles_std']),
                gamma_hat=float(gamma_hat_list[k]),
            ))
            last_K = result['K']
            P_tilde_pred = result['P_new']

        # Normalized update
        P_tilde = (1.0 - last_K * H_OBS) * P_tilde_pred

        if k % 4 == 0:
            n_elim = sum(1 for d in decisions if d.eliminated)
            last = decisions[-1] if decisions else None
            grp_info = f"  grp={last.n_groups}" if last else ""
            pole_info = f"  pole_μ={last.poles_mean:.3f}" if last else ""
            print(f"  Katman {k:2d}/{L}  "
                  f"P̃={P_tilde:.4f}  "
                  f"K={last_K:.4f}  "
                  f"E={E_total:.5f}  "
                  f"elim={n_elim}"
                  f"{grp_info}{pole_info}")

    return decisions, E_total, time.time() - t0, gamma_hat_list


def plot_report(decisions, E_total, eps, L):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f'α-EY-Kalman (Normalized P) — Qwen 2.5 0.5B\n'
        f'ε={eps}  E_total={E_total:.5f}  L·ε={L*eps:.3f}',
        fontsize=12, fontweight='bold')

    ks    = sorted(set(d.k for d in decisions))
    wnames = sorted(set(d.weight_name for d in decisions))
    cmap   = {n: plt.cm.tab10(i) for i, n in enumerate(wnames)}

    # 1. P̃_k
    ax = axes[0, 0]
    for wn in ['q_proj', 'v_proj']:
        ds = [d for d in decisions if d.weight_name == wn]
        ax.plot([d.k for d in ds], [d.P_tilde for d in ds],
                'o-', lw=1.5, markersize=3, color=cmap[wn], label=wn)
    ax.set_title('P̃_k (normalized kovaryans)', fontweight='bold')
    ax.set_xlabel('Katman'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 2. K̃_k
    ax = axes[0, 1]
    for wn in ['q_proj', 'o_proj']:
        ds = [d for d in decisions if d.weight_name == wn]
        ax.plot([d.k for d in ds], [d.K for d in ds],
                'o-', lw=1.5, markersize=3, color=cmap[wn], label=wn)
    ax.axhline(1.0, color='red', linestyle='--', lw=1)
    ax.set_title('K̃_k (compression budget)', fontweight='bold')
    ax.set_xlabel('Katman'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 3. σ vs threshold
    ax = axes[0, 2]
    for wn in ['q_proj', 'o_proj']:
        ds = [d for d in decisions if d.weight_name == wn]
        ax.scatter([d.k for d in ds], [d.sigma_r1  for d in ds],
                   s=20, alpha=0.7, color=cmap[wn], label=f'σ {wn}')
        ax.scatter([d.k for d in ds], [d.threshold for d in ds],
                   s=20, alpha=0.5, color=cmap[wn], marker='^',
                   label=f'eşik {wn}')
    ax.set_title('σ_{r+1} vs Eşik', fontweight='bold')
    ax.set_xlabel('Katman'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 4. Eliminasyon
    ax = axes[1, 0]
    elim_by = {k: 0 for k in ks}
    total_by = {k: 0 for k in ks}
    for d in decisions:
        total_by[d.k] += 1
        if d.eliminated: elim_by[d.k] += 1
    rates = [elim_by[k] / max(total_by[k], 1) for k in ks]
    ax.bar(ks, rates,
           color=['#3b6d11' if r > 0 else '#cccccc' for r in rates])
    ax.set_title('Eliminasyon oranı (katman)', fontweight='bold')
    ax.set_xlabel('Katman'); ax.set_ylim(0, 1.1); ax.grid(True, alpha=0.3)

    # 5. Hata birikimi
    ax = axes[1, 1]
    cumE = np.cumsum([d.error_contrib for d in decisions])
    ax.plot(range(len(decisions)), cumE, 'b-', lw=1.5, label='E_total')
    ax.axhline(L * eps, color='red', linestyle='--', lw=1.5,
               label=f'L·ε={L*eps:.3f}')
    ax.set_title('Ana Teorem: E_total ≤ L·ε', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 6. σ dağılımı — elinen vs tutulanlara göre
    ax = axes[1, 2]
    elim_s  = [d.sigma_r1 for d in decisions if d.eliminated and d.sigma_r1 > 0]
    kept_s  = [d.sigma_r1 for d in decisions if not d.eliminated and d.sigma_r1 > 0]
    if elim_s:
        ax.hist(elim_s, bins=30, alpha=0.7, color='#3b6d11', label='Elinen')
    if kept_s:
        ax.hist(kept_s, bins=30, alpha=0.5, color='#993c1d', label='Tutulan')
    ax.axvline(eps, color='blue', linestyle='--', lw=1.5, label=f'ε={eps}')
    ax.set_title('σ_{r+1} Dağılımı', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'compression_fixed_report.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Grafik: {RESULTS_DIR}/compression_fixed_report.png")


def main():
    global tokenizer_global

    print(f"\nModel yükleniyor: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True)
    tokenizer_global = tokenizer   # compress_normalized içinden erişilir

    _dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float32
    print(f"  dtype: {args.dtype}")

    # Büyük modeller (>6GB bfloat16) için GPU + in-place mod
    # 7B bfloat16=14GB > 16GB RAM → deepcopy OOM; A10G 24GB VRAM yeterli
    _use_gpu = torch.cuda.is_available()
    _device_map = "auto" if _use_gpu else None
    if _use_gpu:
        print(f"  GPU modu: device_map=auto (CUDA available)")
    else:
        print(f"  CPU modu")

    model_orig = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=_dtype, trust_remote_code=True,
        device_map=_device_map)
    model_orig.eval()

    sample = ("The capital of France is Paris. Machine learning is "
              "a subset of artificial intelligence. The Eiffel Tower "
              "was built in 1889. Python is a popular programming language. "
              "The theory of evolution was proposed by Charles Darwin. ") * 8

    print("\n[Baseline] PPL...")
    ppl_base = evaluate_perplexity(model_orig, tokenizer, sample)
    print(f"  Baseline PPL: {ppl_base:.3f}")

    print("\n[Sıkıştırma] Normalized P ile...")
    # Büyük model (GPU mod): deepcopy OOM'dan kaçın, in-place compress
    # Küçük model (CPU mod): deepcopy güvenli
    if _use_gpu:
        model_c = model_orig   # in-place
    else:
        model_c = copy.deepcopy(model_orig)
    decisions, E_total, elapsed, gamma_hat_list = compress_normalized(
        model_c, eps=EPS, delta=DELTA, bits=BITS)

    L = len(model_orig.model.layers)

    print("\n[Sıkıştırılmış] PPL...")
    ppl_c = evaluate_perplexity(model_c, tokenizer, sample)
    dppl  = (ppl_c - ppl_base) / ppl_base * 100
    print(f"  Sıkıştırılmış PPL: {ppl_c:.3f}  ({dppl:+.1f}%)")

    print("\n[Kalite] Cevap üretiliyor...")
    quality = evaluate_quality(model_c, tokenizer)
    for q in quality:
        print(f"  Q: {q['prompt'][:40]}")
        print(f"  A: {q['response'][:80]}\n")

    n_elim    = sum(1 for d in decisions if d.eliminated)
    n_total   = len(decisions)
    theorem_ok = E_total <= L * EPS

    print("=" * 60)
    print("ÖZET")
    print("=" * 60)
    print(f"Toplam karar:     {n_total}")
    print(f"Elinen:           {n_elim} ({100*n_elim/n_total:.1f}%)")
    print(f"E_total:          {E_total:.6f}")
    print(f"L·ε (üst sınır): {L*EPS:.6f}")
    print(f"Ana Teorem:       {'✓ SAĞLANDI' if theorem_ok else '✗ İHLAL'}")
    print(f"Süre:             {elapsed:.1f}s")
    print(f"PPL baseline:     {ppl_base:.3f}")
    print(f"PPL compressed:   {ppl_c:.3f}")
    print(f"PPL değişimi:     {dppl:+.1f}%")

    # Hangi weight türleri elendi?
    from collections import Counter
    elim_types = Counter(d.weight_name for d in decisions if d.eliminated)
    print(f"\nEliminasyon türleri: {dict(elim_types)}")
    print(f"\ngamma_hat_list[0]  = {gamma_hat_list[0]:.6f}")
    print(f"gamma_hat_list[23] = {gamma_hat_list[23]:.6f}")

    log = {
        'config': {'eps': EPS, 'delta': DELTA, 'bits': BITS,
                   'normalization': 'P_tilde = P / gamma^2'},
        'n_layers': L, 'n_decisions': n_total, 'n_eliminated': n_elim,
        'E_total': E_total, 'E_bound': L * EPS,
        'theorem_satisfied': theorem_ok,
        'ppl_baseline': ppl_base, 'ppl_compressed': ppl_c,
        'dppl_pct': dppl, 'elapsed_s': elapsed,
        'quality': quality,
        'elim_by_type': dict(elim_types),
        'gamma_hat_list': gamma_hat_list,
        'laplace_poles': {
            'mean_all': float(np.mean([d.poles_mean for d in decisions])),
            'std_all':  float(np.mean([d.poles_std  for d in decisions])),
        },
        'groups': {
            'mean_n_groups': float(np.mean([d.n_groups for d in decisions])),
            'min_n_groups':  int(min(d.n_groups for d in decisions)),
            'max_n_groups':  int(max(d.n_groups for d in decisions)),
        },
        'decisions': [asdict(d) for d in decisions],
    }
    log_path = args.output
    if not log_path.endswith('.json'):
        log_path += '.json'

    # results/ prefix veya mutlak yol → direkt kullan (çift klasör engel)
    _lp = Path(log_path)
    if _lp.is_absolute() or str(log_path).startswith('results/') or str(log_path).startswith('./'):
        out_file = _lp
        out_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_file = RESULTS_DIR / log_path
    with open(out_file, 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\nLog: {out_file}")

    if not args.no_plot:
        plot_report(decisions, E_total, EPS, L)

    # ── Model Kayıt (--save_model_dir verilmişse) ────────────────────
    if args.save_model_dir:
        save_dir = Path(args.save_model_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[Model Kayıt] {save_dir} ...")
        model_c.save_pretrained(str(save_dir), max_shard_size="2GB")
        tokenizer.save_pretrained(str(save_dir))
        # AEK metadata — benchmark scripti okur
        aek_meta = {
            'base_model':       MODEL_ID,
            'aek_eps':          EPS,
            'adaptive_mode':    args.adaptive_mode if args.use_adaptive_eps else 'none',
            'n_eliminated':     n_elim,
            'elim_by_type':     dict(elim_types),
            'E_total':          E_total,
            'E_bound':          L * EPS,
            'theorem_satisfied': theorem_ok,
            'ppl_baseline':     ppl_base,
            'ppl_compressed':   ppl_c,
            'dppl_pct':         dppl,
        }
        (save_dir / 'aek_metadata.json').write_text(
            json.dumps(aek_meta, indent=2))
        # Disk boyutu
        size_bytes = sum(
            f.stat().st_size for f in save_dir.rglob('*') if f.is_file())
        print(f"  Kaydedildi: {save_dir}  ({size_bytes/1e9:.2f} GB)")
        print(f"  AEK metadata: {save_dir}/aek_metadata.json")


if __name__ == '__main__':
    main()