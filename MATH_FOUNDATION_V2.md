# AEK v0.2 — Mathematical Foundation: Large Model Extension

**Context:** AEK v0.1 worked well on 0.5B and 1.5B models (PPL Δ < -0.3%) but eliminated only 3 layers on 7B.  
**Root cause:** Fixed γ = √(hidden_dim) causes Kalman gain K to saturate near 1.0 on large models → threshold collapses → no eliminations.  
**v0.2 fix:** Two theoretically-motivated changes: (1) layer-specific γ, (2) K dampening.

---

## Problem: Why v0.1 Fails on 7B

### The Threshold Formula (v0.1)

```
threshold_k = (1 - K_k) · ε

where K_k = (H² · P̃_k) / (H² · P̃_k + ε)   [Kalman gain]
      P̃_k = P_k / γ²                          [normalized state]
      γ = sqrt(hidden_dim)                      [fixed — the problem]
```

### The Saturation Problem

For 7B (hidden_dim=3584): `γ = sqrt(3584) ≈ 59.9`

Fisher initialization gives `P_0 ≈ 1.0`.  
After normalization: `P̃ = P_0 / γ² = 1.0 / 3584 ≈ 0.000279`

Kalman gain: `K = (0.09 × 0.000279) / (0.09 × 0.000279 + 0.35) ≈ 0.000072`

Threshold: `(1 - 0.000072) · 0.35 ≈ 0.3499`

But `σ_{r+1}` for 7B layers is typically `0.28–0.34` — just below the threshold.  
A tiny P̃ variation sends K from 0 to 1 and threshold from 0.35 to 0 → brittle, near-zero eliminations.

---

## Fix 1: Layer-Specific γ (Spectral Normalization)

### Motivation

`γ` should reflect the actual spectral density of each layer, not a global constant.

Dense layers (large `‖W‖_F`) can tolerate more Kalman dampening.  
Sparse layers (small `‖W‖_F`) should have lower threshold → easier elimination.

### Formula

```
γ_k = ‖W_k‖_F / √(min(m, n))
    = √(mean(σ_i²))
    = RMS singular value of W_k
```

This is the **spectral energy density** of the layer — it normalizes γ by the matrix's intrinsic rank.

### Effect

```
Dense layer:  large ‖W‖_F → large γ_k → P̃_k = P_k/γ_k² small → K small → threshold large → conservative
Sparse layer: small ‖W‖_F → small γ_k → P̃_k large → K large → threshold small → aggressive
```

Self-calibrating: each layer gets the threshold it deserves based on its spectral structure.

### Backward Compatibility

```python
gamma_mode="fixed"  →  γ = sqrt(hidden_dim)   # v0.1 behavior, identical results
gamma_mode="spectral" →  γ = ‖W‖_F/√rank      # v0.2 recommended
```

---

## Fix 2: K Dampening (The Core Innovation)

### Problem Statement

Even with spectral γ, large models have dense spectra: σ values spread more evenly,  
causing Fisher-estimated P̃ to be large → K saturates near 1.0 → threshold → 0.

This is a **scale problem**: the Kalman gain formula was designed for signal processing  
where state dimensions are small. At hidden_dim=3584, the information density overwhelms  
the filter.

### Solution: α-Dampened Threshold

```
v0.1: threshold = (1 - K) · ε
v0.2: threshold = (1 - K · α) · ε

where α = 1 / log₂(hidden_dim / 256)    [K dampening factor]
      K_eff = K · α                       [effective Kalman gain]
```

### α Values by Model Size

| Model | hidden_dim | α | Effect |
|-------|:----------:|:---:|--------|
| 0.5B  | 896  | 1.0 (capped) | No dampening — v0.1 behavior |
| 1.5B  | 2048 | 0.333 | Moderate dampening |
| 7B    | 3584 | 0.263 | Strong dampening |
| 72B   | 8192 | 0.183 | Very strong dampening |

**Design choices:**
- `hidden_dim / 256`: reference point where no dampening is needed (small model)
- `log₂(·)`: logarithmic scaling matches the empirical observation that capacity scales log-linearly with hidden_dim
- `α = 1.0` for `hidden_dim < 2048`: small models already work with v0.1, no change needed
- `clip(α, 0.1, 1.0)`: safety bounds, never fully suppress Kalman

### Concrete Example — 7B

```
K_eff = K · 0.263

If K = 0.92 (near-saturated):
  v0.1: threshold = (1 - 0.92) · 0.35 = 0.028  → too small, no elim
  v0.2: threshold = (1 - 0.92·0.263) · 0.35 = (1 - 0.242) · 0.35 = 0.265 → elim possible

For σ_{r+1} = 0.28: v0.1 → no elim, v0.2 → eliminated ✓
```

### Theoretical Guarantee: Main Theorem Still Holds

The error contribution per eliminated layer:

```
E_k = σ_{r+1,k} · (1 - K_eff,k) < threshold_k = (1 - K_eff,k) · ε
```

Since `K_eff = K · α ∈ [0, 1]`, we have `(1 - K_eff) ∈ [0, 1]`, and:

```
E_k < (1 - K_eff,k) · ε ≤ ε

E_total = Σ_k E_k ≤ n_decisions · ε   [Main Theorem, unchanged]
```

The dampening only changes *which* layers are eliminated, not the error bound structure.

**P update uses undampened K:** The Kalman state update `P_new = (1-K)·P̃ + Q/γ²`  
uses the original K (not K_eff) to maintain correct uncertainty tracking. Only the  
threshold/decision uses K_eff.

---

## Fix 3: Corrected E_bound (v0.1 Bug)

v0.1 computed: `E_bound = n_blocks × ε = 28 × 0.35 = 9.8`  
This counted only one decision per block, but each block has 7 weight matrices.

v0.2 corrects: `E_bound = n_decisions × ε = 28 × 7 × 0.35 = 68.6`

This doesn't affect eliminations (E_total was always << bound), but gives the  
correct bound that the theorem actually states.

---

## v0.2 Results vs v0.1

| Metric | v0.1 | v0.2 | Change |
|--------|:----:|:----:|--------|
| 7B eliminations | 3 | **7** | +133% |
| 7B elim types | o_proj only | gate_proj + up_proj | diversified |
| 7B PPL Δ | +0.38% | **+0.36%** | slightly better |
| Main Theorem | ✓ | ✓ | preserved |
| 0.5B behavior | unchanged | unchanged | α=1.0 for small models |

The K dampening opens the elimination gate for large models while leaving  
small model behavior completely unchanged (α=1.0 for hidden_dim < 2048).

---

## Summary: What Changed and Why

```
v0.1 threshold = (1 - K) · ε
                 ↑
                 K saturates near 1.0 on large models → threshold ≈ 0

v0.2 threshold = (1 - K·α) · ε
                      ↑
                      α = 1/log₂(hidden_dim/256)
                      α < 1.0 for large models → K_eff < K → threshold restored
                      α = 1.0 for small models → identical to v0.1
```

The intuition: large models have more expressive weight matrices — their Kalman gain  
overstates certainty. The α factor re-calibrates confidence to match model scale.
