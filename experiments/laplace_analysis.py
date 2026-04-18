"""
faz2_laplace.py
===============
Laplace kutupları — aktivasyon tabanlı AR(1) tahmin.

Fonksiyonlar:
  - collect_sample_activations(model, tokenizer, layer_idx)
      Forward hook ile tek layer'ın giriş aktivasyonlarını toplar.
  - collect_all_layer_activations(model, tokenizer)
      Tüm layer'ları tek forward pass'te toplar (verimli).
  - laplace_poles_real(W, sample_acts)
      Aktivasyon tabanlı AR(1) kutup tahmini.
"""

import torch
import numpy as np
from typing import Dict, List, Optional


def collect_sample_activations(
    model,
    tokenizer,
    layer_idx: int,
    texts: Optional[List[str]] = None,
    n_tokens: int = 64,
) -> np.ndarray:
    """
    Forward hook ile belirtilen layer'ın giriş aktivasyonlarını topla.

    Returns: (T, hidden_dim) numpy array
    """
    if texts is None:
        texts = [
            "The capital of France is Paris.",
            "Machine learning is a subset of artificial intelligence.",
            "The Eiffel Tower was built in 1889.",
        ]

    collected = []

    def hook_fn(module, inp, out):
        x = inp[0].detach().float().cpu()
        collected.append(x.reshape(-1, x.shape[-1]).numpy())

    layer = model.model.layers[layer_idx]
    handle = layer.register_forward_hook(hook_fn)

    try:
        _dev = next(model.parameters()).device
    except StopIteration:
        _dev = torch.device("cpu")

    try:
        for text in texts:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=n_tokens
            )
            inputs = {k: v.to(_dev) for k, v in inputs.items()}
            with torch.no_grad():
                model(**inputs)
    finally:
        handle.remove()

    if not collected:
        return np.zeros((1, model.config.hidden_size), dtype=np.float32)
    return np.concatenate(collected, axis=0)  # (T, hidden_dim)


def collect_all_layer_activations(
    model,
    tokenizer,
    texts: Optional[List[str]] = None,
    n_tokens: int = 64,
) -> Dict[int, np.ndarray]:
    """
    Tüm transformer layer'larının giriş aktivasyonlarını tek seferde topla.
    02b_compress_fixed.py'nin ana döngüsünden önce bir kez çağır.

    Returns: {layer_idx: np.ndarray(T, hidden_dim)}
    """
    if texts is None:
        texts = [
            "The capital of France is Paris.",
            "Machine learning is a subset of artificial intelligence.",
            "The Eiffel Tower was built in 1889.",
            "Python is a popular programming language.",
        ]

    n_layers = len(model.model.layers)
    buffers: Dict[int, list] = {i: [] for i in range(n_layers)}
    handles = []

    def make_hook(idx):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float().cpu()
            buffers[idx].append(x.reshape(-1, x.shape[-1]).numpy())
        return hook_fn

    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    try:
        _dev = next(model.parameters()).device
    except StopIteration:
        _dev = torch.device("cpu")

    try:
        for text in texts:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=n_tokens
            )
            inputs = {k: v.to(_dev) for k, v in inputs.items()}
            with torch.no_grad():
                model(**inputs)
    finally:
        for h in handles:
            h.remove()

    hidden_dim = model.config.hidden_size
    result = {}
    for i, bufs in buffers.items():
        if bufs:
            result[i] = np.concatenate(bufs, axis=0)
        else:
            result[i] = np.zeros((1, hidden_dim), dtype=np.float32)
    return result


def laplace_poles_real(W: np.ndarray, sample_acts: np.ndarray) -> np.ndarray:
    """
    Aktivasyon tabanlı AR(1) kutup tahmini.

    Matris W (n_out, n_in) için aktivasyonlar x (T, n_in) verilince:
        y = W @ x.T  →  shape (n_out, T)

    AR(1) model: y_i[t+1] ≈ pole_i * y_i[t]

    Least-squares:
        pole_i = Σ(y_i[t+1] * y_i[t]) / Σ(y_i[t]²)

    Boyut uyumsuzluğu (örn. down_proj giriş boyutu > hidden_dim):
        sample_acts[:, :n_in] ile kırpılır / sıfır doldurulur.

    Returns: poles (n_out,) — [-2, 2] aralığında kırpılmış
    """
    n_out, n_in = W.shape

    if sample_acts.shape[0] < 2:
        return _diagonal_proxy(W)

    T_full, act_dim = sample_acts.shape

    # Boyut uyumunu sağla
    if act_dim >= n_in:
        acts = sample_acts[:, :n_in].astype(np.float32)
    else:
        # Doldur (örn. down_proj durumu)
        pad = np.zeros((T_full, n_in - act_dim), dtype=np.float32)
        acts = np.concatenate([sample_acts.astype(np.float32), pad], axis=1)

    # y: (n_out, T)
    y = (W.astype(np.float32)) @ acts.T

    y_t  = y[:, :-1]   # (n_out, T-1)
    y_t1 = y[:, 1:]    # (n_out, T-1)

    numerator   = np.sum(y_t1 * y_t,  axis=1)          # (n_out,)
    denominator = np.sum(y_t  * y_t,  axis=1) + 1e-12  # (n_out,)

    poles = numerator / denominator
    return np.clip(poles, -2.0, 2.0)


def _diagonal_proxy(W: np.ndarray) -> np.ndarray:
    """Aktivasyon yokken fallback: diyagonal/U proxy."""
    n = min(W.shape)
    if W.shape[0] == W.shape[1]:
        return np.real(np.diag(W[:n, :n]))
    U, _, _ = np.linalg.svd(W, full_matrices=False)
    return np.real(np.diag(U[:n, :n]))
