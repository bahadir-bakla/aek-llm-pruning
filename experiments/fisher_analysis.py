"""
faz2_fisher.py
==============
Fisher bilgi matrisi diagonal'ı — Kalman P₀ başlatma.

Fonksiyonlar:
  - compute_diagonal_fisher(model, tokenizer, n_samples=16)
      gradient² ortalaması → her Linear ağırlık için skaler Fisher değeri.
  - initialize_kalman_p0(fisher_p0, n_layers)
      Fisher'dan katman başına P₀ listesi üretir.

Mantık:
  Fisher büyük → parametre kritik → P₀ küçük (Kalman dikkatli eleme yapar)
  Fisher küçük → parametre önemsiz → P₀ büyük (Kalman rahat eliyor)
"""

import torch
import numpy as np
from typing import Dict, List, Optional


def compute_diagonal_fisher(
    model,
    tokenizer,
    n_samples: int = 4,
    texts: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Diagonal Fisher tahmini: E[(∂ log p(x|θ) / ∂θ)²]

    Her Linear parametre için gradient² ortalamasını hesaplar.
    Causal LM kaybı üzerinden backward pass yapılır.

    Returns: {param_name: mean_fisher_value}
    """
    if texts is None:
        texts = [
            "The capital of France is Paris.",
            "Machine learning is a subset of artificial intelligence.",
            "The Eiffel Tower was built in 1889.",
            "Python is a popular programming language.",
            "The theory of evolution was proposed by Charles Darwin.",
            "Water boils at 100 degrees Celsius at sea level.",
            "The human brain has approximately 86 billion neurons.",
            "Isaac Newton formulated the laws of motion.",
        ]

    # İki parametreli Linear'lar için akümülatör
    accum: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if p.requires_grad and p.ndim == 2:
            accum[name] = torch.zeros_like(p.data, dtype=torch.float32)

    model.eval()
    count = 0

    for text in texts[:n_samples]:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=64
        )
        # device_map="auto" ile GPU'ya yüklenmiş modeller için
        try:
            _dev = next(model.parameters()).device
        except StopIteration:
            _dev = torch.device("cpu")
        inputs = {k: v.to(_dev) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()

        model.zero_grad()
        try:
            # bfloat16 modellerinde backward yavaş — loss float32'ye cast et
            out = model(**inputs)
            loss = out.loss.float()
            loss.backward()
        except Exception as e:
            print(f"  [Fisher] backward atlandı: {e}")
            continue

        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None and name in accum:
                accum[name] += p.grad.data.float() ** 2

        count += 1
        if count >= n_samples:
            break

    model.zero_grad()

    if count == 0:
        return {name: 1.0 for name in accum}

    return {name: float(v.mean()) / count for name, v in accum.items()}


def initialize_kalman_p0(
    fisher_p0: Dict[str, float],
    n_layers: int,
    layer_prefix: str = "model.layers",
) -> List[float]:
    """
    Fisher diagonal'dan her transformer katmanı için P₀ skaler listesi.

    Dönüşüm: P₀ = 1 / (1 + mean_fisher × scale)
      - scale=1e4 ile Fisher ~1e-4 → P₀ ~0.5 (orta güven)
      - Fisher büyük (>1e-3) → P₀ küçük (dikkatli)
      - Fisher küçük (<1e-5) → P₀ ~1.0 (serbest)

    Katmana ait parametre yoksa P₀=1.0 (muhafazakâr başlangıç).

    Returns: [P0_0, P0_1, ..., P0_{n_layers-1}]
    """
    SCALE = 1e4
    p0_list = []

    for k in range(n_layers):
        prefix = f"{layer_prefix}.{k}."
        vals = [v for name, v in fisher_p0.items() if name.startswith(prefix)]

        if vals:
            mean_f = float(np.mean(vals))
            p0 = 1.0 / (1.0 + mean_f * SCALE)
            p0 = float(np.clip(p0, 0.01, 1.0))
        else:
            p0 = 1.0

        p0_list.append(p0)

    return p0_list
