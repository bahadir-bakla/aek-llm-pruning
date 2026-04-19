# AEK-LLM-Pruning

**α-EY-Kalman (AEK): Theoretically-Guaranteed Pruning for Large Language Models**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Models: Qwen 2.5](https://img.shields.io/badge/models-Qwen2.5-orange.svg)](https://huggingface.co/Qwen)

AEK is a novel LLM pruning algorithm that combines **Laplace pole analysis**, **α(n) Union-Find grouping**, **Eckart-Young SVD**, and a **Kalman filter** to perform theoretically-bounded layer elimination. Unlike empirical pruning methods, AEK provides a formal guarantee: total accumulated error never exceeds `L·ε`.

---

## Key Results

### AEK v0.2 — Large Model Extension (Recommended)

| Model | ε | Eliminations | Elim Types | PPL Δ | Theorem |
|-------|---|:---:|---|:---:|:---:|
| Qwen2.5-7B | 0.35 | **7** | gate_proj (3) + up_proj (4) | **+0.36%** | ✓ |

> v0.2 introduces K dampening (`α = 1/log₂(hidden_dim/256)`) and layer-specific γ normalization,  
> increasing 7B eliminations from 3 → 7 (+133%) while maintaining the Main Theorem guarantee.  
> See [MATH_FOUNDATION_V2.md](MATH_FOUNDATION_V2.md) for the theoretical derivation.

### AEK v0.1 — Baseline Results

| Model | ε | Eliminations | PPL Δ | HellaSwag Δ | ARC-Easy Δ | ARC-Challenge Δ |
|-------|---|:---:|:---:|:---:|:---:|:---:|
| Qwen2.5-0.5B | 0.20 | 7 | **-0.43%** | -0.07% | +0.55% | -0.08% |
| Qwen2.5-1.5B | 0.26 | 2 | **-0.27%** | -0.30% | +0.38% | -0.60% |
| Qwen2.5-7B   | 0.35 | 3 | +0.38%  | -0.08% | +0.38% | **+0.51%** |

> Theorem satisfied (E_total ≤ L·ε) across all scales. PPL degradation < 0.5% for 0.5B and 1.5B.

### Hybrid AEK (Pruning + Quantization) — 7B

| Method | Disk | PPL Δ | Compression |
|--------|:----:|:---:|:---:|
| AEK BF16 | 15.2 GB | 0.00% | 1.0x |
| AEK + Naive INT8 | 8.7 GB | +5.26% | 1.75x |
| **AEK + LLM.int8()** | **8.7 GB** | **+0.57%** | **1.75x** |

> Kalman pressure map guides which layers are INT8 vs BF16 — no manual layer selection needed.

---

## Algorithm Overview

```
Input: Pre-trained LLM weight matrix W_k, tolerance ε

1. Laplace Poles     → AR(1) pole estimation per column of W
2. α(n) Union-Find   → Group columns by pole proximity (max group=4, α(n)≤4 guarantee)
3. Eckart-Young SVD  → Optimal rank-r approximation per group (minimum Frobenius error)
4. Kalman Filter     → Track accumulated error P̃_k across layers
5. Decision          → Eliminate layer k if σ_{r+1} < (1-K_k)·ε

Guarantee: E_total = Σ E_k ≤ L·ε   (Main Theorem)
```

See [MATH_FOUNDATION.md](MATH_FOUNDATION.md) for the full theoretical derivation.

---

## Why AEK?

Most pruning methods (magnitude, gradient, activation) are **empirical** — they work in practice but provide no formal error bound. AEK provides a **provable guarantee**:

> For any ε > 0, the total reconstruction error across all eliminated layers satisfies E_total ≤ L·ε, where L is the number of layers.

This is derived from the Eckart-Young theorem (optimal low-rank approximation) combined with a Kalman filter that propagates uncertainty across layers.

---

## Installation

```bash
git clone https://github.com/bahadir-bakla/aek-llm-pruning.git
cd aek-llm-pruning
pip install -r requirements.txt
```

---

## Usage

### Compress a model

```bash
python src/compress.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --eps 0.20 \
    --fisher_samples 4 \
    --save_model_dir ./models/qwen_0b5_aek \
    --output ./results/my_run.json
```

### Full evaluation (compress + PPL + lm-eval)

```bash
# 0.5B
python src/evaluate_0b5.py --eps 0.20 --fisher_samples 4

# 1.5B
python src/evaluate_1b5.py --eps 0.26 --fisher_samples 4

# 7B
python src/evaluate_7b.py --eps 0.35 --fisher_samples 4
```

### Hybrid AEK (7B, pressure-guided quantization)

```bash
python src/hybrid_aek.py \
    --aek_model_dir ./models/qwen_7b_aek \
    --pressure_thresh 0.7
```

### AEK v0.2 — 7B with large model extension

```bash
python src/evaluate_7b_v2.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --eps 0.35 \
    --gamma_mode spectral \
    --output results/aek_v2_7b_eps035.json
```

### Key parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--eps` | Elimination tolerance ε | 0.20 |
| `--gamma_mode` | γ mode: `spectral` (v0.2) or `fixed` (v0.1) | spectral |
| `--k_alpha_override` | Manual K dampening factor (auto if not set) | None |
| `--fisher_samples` | Samples for Fisher P₀ init | adaptive |
| `--dtype` | Model dtype (bfloat16/float32) | bfloat16 |

---

## Project Structure

```
aek-llm-pruning/
├── src/
│   ├── compress_utils.py      # Core AEK v0.1 (Kalman, SVD, Laplace, Union-Find)
│   ├── compress_utils_v2.py   # AEK v0.2 (K dampening, spectral γ, adaptive Fisher)
│   ├── compress.py            # Main compression pipeline
│   ├── evaluate_0b5.py        # 0.5B full eval (compress + PPL + lm-eval)
│   ├── evaluate_1b5.py        # 1.5B full eval
│   ├── evaluate_7b.py         # 7B full eval (v0.1)
│   ├── evaluate_7b_v2.py      # 7B full eval (v0.2 — recommended for large models)
│   └── hybrid_aek.py          # Hybrid AEK (pressure-guided INT8/INT4)
├── experiments/
│   ├── ablation.py            # 5-config ablation study
│   ├── alpha_analysis.py      # α(n) Union-Find analysis
│   ├── fisher_analysis.py     # Fisher P₀ sensitivity
│   └── laplace_analysis.py    # Laplace pole distribution
├── results/
│   ├── faz9_0b5_full.json         # 0.5B v0.1 results
│   ├── faz9_1b5_eps026_full.json  # 1.5B v0.1 results
│   ├── faz9_7b_eps035_full.json   # 7B v0.1 results
│   ├── faz9_7b_hybrid_eps03_full.json  # 7B Hybrid AEK results
│   ├── aek_v2_7b_eps035.json      # 7B v0.2 results (7 elim, PPL +0.36%)
│   ├── ablation.json              # Ablation study results
│   └── paper_tables.md            # All results in table format
├── MATH_FOUNDATION.md         # v0.1 theoretical derivation
├── MATH_FOUNDATION_V2.md      # v0.2 extension: K dampening + spectral γ
├── requirements.txt
└── README.md
```

---

## Theoretical Background

AEK builds on five classical results:

| Component | Source | Role in AEK |
|-----------|--------|-------------|
| Laplace poles | Signal theory | Column grouping criterion |
| α(n) Union-Find | Tarjan 1975 | O(α(n)) grouping with size bound |
| Eckart-Young theorem | Eckart & Young 1936 | Optimal low-rank error bound |
| Kalman filter | Kalman 1960 | Cross-layer error propagation |
| Shadow Riccati | This work | γ̂_k balancing (closes the proof) |

The **Shadow Riccati** forward pass is the novel contribution: it computes per-layer scale factors γ̂_k that ensure the error bound holds across the full network depth.

---

## Limitations and Future Work

- **Disk compression:** AEK pruning removes individual singular values, not full weight matrices — disk savings require combining with quantization (Hybrid AEK) or low-rank factorization (v0.3, planned).
- **lm-eval on 7B:** Running lm-eval on g5.xlarge requires careful GPU memory management (save → del → eval pattern). PPL is the primary metric for large model evaluation.
- **Planned (v0.3):** Low-rank factorization W = U·S·Vᵀ — real disk compression (target: 15GB → <5GB for 7B) with theoretical error bound from Eckart-Young theorem.

---

## Citation

```bibtex
@software{aek_llm_pruning,
  title  = {AEK-LLM-Pruning: α-EY-Kalman Theoretically-Guaranteed Pruning for LLMs},
  author = {Bakla, Bahadir},
  year   = {2025},
  url    = {https://github.com/bahadir-bakla/aek-llm-pruning}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
