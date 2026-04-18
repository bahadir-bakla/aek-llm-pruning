# Paper-Ready Tablolar

*α-EY-Kalman (AEK) Pruning — Qwen 2.5 Model Ailesi*

---

## Table 1: Model Karşılaştırması (Qwen 2.5 0.5B, ε=0.20)

| Method           | Bits | PPL   | PPL Δ%   | Disk (GB) | Compression | Elim |
|------------------|------|-------|----------|-----------|-------------|------|
| Original BF16    | 16   | 1.365 |  0.00%   | 0.988     | 1.0x        | -    |
| AEK only (ours)  | 16   | 1.352 | **-0.97%** | 0.988   | 1.0x        | 7    |
| AEK + F16        | 16   | 1.352 | **-0.97%** | 0.494   | 2.0x        | 7    |
| AEK + INT8       | 8    | 1.351 | **-1.00%** | 0.630   | **3.14x**   | 7    |
| MLX-INT4         | 4    | 1.380 | +1.12%   | 0.28      | 3.5x        | -    |
| AWQ INT4†        | 4    | —     | +18.53%† | 0.458     | 2.16x       | -    |

> **Not:** AEK rank-reduction (SVD), MLX/AWQ weight quantization yapar — orthogonal teknikler.
> †AWQ PPL farklı evaluation corpus (wikitext-2 raw stride=512) ile ölçüldü; delta yüzdesi iç tutarlı.
> AEK+INT8 teorema koşulunu sağlar: E_total=1.12 < E_bound=4.80 (Eckart-Young bound).

---

## Table 2: Ablation Study (Qwen 2.5 0.5B, ε=0.20)

| Config | Bileşenler                                         | Elinen | PPL Δ%   |
|--------|----------------------------------------------------|--------|----------|
| A      | Kalman + full SVD                                  | 3      | -0.84%   |
| B      | Kalman + rand SVD                                  | 0      |  0.00%   |
| C      | Kalman + rand SVD + Laplace + α(n)                 | 1      | -0.86%   |
| D      | Kalman + rand SVD + Fisher P₀                      | 0      |  0.00%   |
| **E**  | **Tam sistem: rand SVD + Laplace + α(n) + Fisher P₀** | **10** | **-2.47%** |

Tam sistem (Config E) en yüksek eliminasyon ve en iyi PPL iyileşmesini sağlar.

---

## Table 3: Multi-Scale AEK Sonuçları

| Model  | Params | ε    | PPL Orig | PPL AEK | PPL Δ%   | Elim | Teorem |
|--------|--------|------|----------|---------|----------|------|--------|
| 0.5B   | 494M   | 0.20 | 1.365    | 1.352   | **-0.97%** | 7  | ✓      |
| 1.5B   | 1.54B  | 0.26 | 1.314    | 1.312   | **-0.15%** | 2  | ✓      |
| 7B     | 7.62B  | 0.35 | 1.297    | 1.301   | **+0.38%** | 3  | ✓      |

Her üç ölçekte de teorem koşulu sağlanarak parametre eliminasyonu gerçekleştirildi.
7B'de ε=0.35 ile gate_proj + up_proj katmanlarında 3 eliminasyon elde edildi.

> **FAZ9 Final Ölçümler** (aws g5.xlarge, bfloat16, sliding window PPL):
> | 0.5B  | ε=0.20 | 1.3359→1.3302 | **-0.43%** | 7 elim | ✓ |
> | 1.5B  | ε=0.26 | 1.3139→1.3104 | **-0.27%** | 2 elim | ✓ |
> | 7B    | ε=0.35 | 8.3434→8.3747 | **+0.38%** | 3 elim | ✓ (gate_proj×1, up_proj×2) |

---

## Table 4: lm-eval Benchmark (0-shot, HellaSwag / ARC)

| Model  | Method       | HellaSwag  | ARC-Easy   | ARC-Challenge | HellaSwag Δ | ARC-Easy Δ | ARC-Chall Δ |
|--------|--------------|------------|------------|---------------|-------------|------------|-------------|
| 0.5B   | Original     | 52.46%     | 59.34%     | 33.53%        | —           | —          | —           |
| 0.5B   | AEK ε=0.20   | 52.39%     | **59.89%** | 33.45%        | -0.07%      | **+0.55%** | -0.08%      |
| 1.5B   | Original     | 68.38%     | 76.01%     | 46.76%        | —           | —          | —           |
| 1.5B   | AEK ε=0.26   | 68.02%     | **76.39%** | 46.16%        | -0.36%      | **+0.38%** | -0.60%      |
| 7B     | Original     | **80.44%** | **81.06%** | **55.29%**    | —           | —          | —           |
| 7B     | AEK ε=0.30   | 80.44%     | 81.06%     | 55.29%        | 0.00%       | 0.00%      | 0.00%       |
| 7B     | AEK ε=0.35   | 80.36%     | **81.44%** | **55.80%**    | -0.08%      | **+0.38%** | **+0.51%** |

> 0.5B AEK: ε=0.20, 7 layer eliminated. AWS g5.xlarge GPU, 0-shot. Δ ≤ 0.10% — istatistiksel olarak anlamlı degradasyon yok.
> 7B AEK ε=0.30: 0 elimination (tüm katmanlar threshold=0.271 üzerinde). PPL ve benchmark farkı sıfır — beklenen sonuç.
> 7B AEK ε=0.35: 3 elimination (gate_proj×1, up_proj×2). PPL Δ=+0.38%, HellaSwag -0.08%, ARC-Easy +0.38%, ARC-Chall +0.51%. Teorem ✓ (E=0.79 < E_bound=9.80).
> 1.5B AEK: ε=0.26, 2 layer eliminated. PPL Δ=-0.27%, ARC-Easy +0.38%. Tüm Δ ≤ 0.60%.
> Referans: Qwen 2.5 1.5B HellaSwag 68.4% literatür değeriyle uyuşuyor.

---

## Table 5: Faktüel Kalite Karşılaştırması (0.5B)

| Soru                                  | Original                            | AEK ε=0.20                          | MLX-INT4                            |
|---------------------------------------|-------------------------------------|--------------------------------------|-------------------------------------|
| Who wrote 1984?                       | George Orwell                       | George Orwell                        | The book "1984" by George Orwell wa |
| What is the capital of France?        | The capital of France is Paris.     | The capital of France is Paris. It   | The capital of France is Paris.     |
| Who developed the theory of relativity? | Albert Einstein, a German-born phys | Albert Einstein, a German physicist  | The theory of relativity was develo |
| What is 2+2?                          | The answer to the question "What is | The answer is 4.                     | 2+2 equals 4.                       |
| Who painted the Mona Lisa?            | Leonardo da Vinci.                  | Leonardo da Vinci.                   | The Mona Lisa was painted by Leonar |

AEK compressed model tüm faktüel sorularda orijinal ile eşdeğer yanıt üretiyor.

---

## Table 6: Hybrid AEK — Kalman Baskı Haritası Tabanlı Adaptif Kuantizasyon (7B, ε=0.30)

| Katman Grubu          | Karar         | Adet | Sebebi                                      |
|-----------------------|---------------|------|---------------------------------------------|
| Attention (q/k/v/o)  | INT8          | 65   | σ_r1 > 0 (SVD aktif, pressure=1.07-1.30)    |
| MLP (gate/up/down)   | INT8          | 131  | σ_r1 = 0 (full-rank, konservatif)           |
| Embed + LM head      | BF16          | 143  | Decisions dışı                              |

| Metrik                  | AEK BF16      | AEK+Naive absmax INT8 | AEK+LLM.int8()†   |
|-------------------------|--------------|-----------------------|-------------------|
| Disk (GB)               | 15.231       | 8.706 (1.75x)         | ~7.5 (2.0x)       |
| PPL sliding (wikitext)  | 8.3463       | 8.785 (**+5.3%**)     | 8.393 (**+0.57%**)|
| PPL Δ vs BF16           | 0.00%        | +5.3%                 | **+0.57%**        |
| Compression             | 1.0x         | 1.75x                 | ~2.0x             |

> †LLM.int8() (bitsandbytes 0.49.2) with mixed-precision decomposition — outlier feature sorununu çözer.
> Naive absmax INT8 outlier'lardan etkileniyor (tüm weight scale = max(|w|)/127 → küçük değerler 0'a yuvarlanıyor).
> PPL referansı sliding window (max_len=512, stride=256, 100 sample wikitext-2-raw-v1).
> 0.5B+AEK+INT8 PPL=-1.00% (Table 1): küçük modellerde outlier etkisi zayıf.

---

*Son güncelleme: 2026-04-18 — 7B ε=0.35 final run tamamlandı (3 elim, PPL +0.38%, ARC-Chall +0.51%). Tüm modeller paper-ready.*
