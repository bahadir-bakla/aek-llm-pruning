"""
faz2_alpha.py
=============
α(n) — Adaptif pole gruplama.

Fonksiyonlar:
  - adaptive_delta(poles, target_group_size=4)
      Pairwise mesafe persentilinden otomatik delta.
  - alpha_union_find_adaptive(poles, max_group_size=4)
      Union-Find, grup boyutu ≤ max_group_size garantisi.
"""

import numpy as np
from typing import List


def adaptive_delta(poles: np.ndarray, target_group_size: int = 4) -> float:
    """
    Pairwise uzaklık dağılımından otomatik delta hesapla.

    Hedef: ortalama grup büyüklüğü ~target_group_size olsun.
    n elemanı eşit gruplara bölmek için birleşmesi gereken çift oranı:
        p = (target_group_size - 1) / target_group_size

    O orana karşılık gelen pairwise mesafe persentilini döndürür.

    Not: Büyük n için ilk 200 eleman sample alınır (O(n²) → O(1)).
    """
    n = len(poles)
    if n < 2:
        return 0.1

    sample = poles[:min(n, 200)]
    diffs = np.abs(sample[:, None] - sample[None, :])
    # Üst üçgen (i < j)
    upper = diffs[np.triu_indices(len(sample), k=1)]

    if len(upper) == 0:
        return 0.1

    # target_group_size kadar birleşme → persentil
    frac = (target_group_size - 1) / target_group_size
    pct = float(np.clip(frac * 100, 5.0, 80.0))
    return float(np.percentile(upper, pct))


def alpha_union_find_adaptive(
    poles: np.ndarray,
    max_group_size: int = 4,
) -> List[List[int]]:
    """
    Union-Find with adaptive delta and hard max_group_size cap.

    Algoritma:
      1. adaptive_delta() → delta
      2. Yakın çiftleri (|p_i - p_j| < delta) mesafe sırasına göre birleştir
      3. Birleştirme öncesi boyut kontrolü: size[rx] + size[ry] > max_group_size ise atla

    Returns: List[List[int]]  — her liste bir grubun indekslerini içerir
    """
    n = len(poles)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    # Büyük n için O(n²) çift taraması çok yavaş.
    # İlk MAX_POOL kutbu grupla, geri kalanları tekil grup olarak ekle.
    MAX_POOL = 2000
    if n > MAX_POOL:
        grouped_part  = alpha_union_find_adaptive(poles[:MAX_POOL], max_group_size)
        singleton_part = [[i] for i in range(MAX_POOL, n)]
        return grouped_part + singleton_part

    delta = adaptive_delta(poles, target_group_size=max_group_size)

    parent = list(range(n))
    size   = [1] * n

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        # Path compression
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(x: int, y: int) -> bool:
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if size[rx] + size[ry] > max_group_size:
            return False
        # Union by size
        if size[rx] < size[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        size[rx] += size[ry]
        return True

    # Yakın çiftleri mesafe sırasına göre işle (greedy — closest first)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(poles[i] - poles[j])
            if d < delta:
                pairs.append((d, i, j))
    pairs.sort()

    for _, i, j in pairs:
        union(i, j)

    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    result = list(groups.values())

    # α(n) garantisi: hiçbir grup max_group_size'ı aşmamalı
    actual_sizes = [len(g) for g in result]
    assert max(actual_sizes) <= max_group_size, (
        f"α(n) ihlali: max_grup_boyutu={max(actual_sizes)} > {max_group_size}. "
        f"delta={delta:.4f}, n_poles={n}"
    )

    return result
