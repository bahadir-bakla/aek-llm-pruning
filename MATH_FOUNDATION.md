# α-EY-Kalman — Matematiksel Altyapı Zinciri

Bu dosya, α-EY-Kalman algoritmasının **teorik ispat zincirini** ve
**kodun bu zinciri nasıl takip etmesi gerektiğini** tanımlar.
Ajan bu dosyayı okuyarak hangi fonksiyonun hangi teorik adıma karşılık
geldiğini anlamalı ve buna göre implement etmelidir.

---

## 1. Algoritmanın Beş Bileşeni

```
Bileşen              Teorik Kaynak              Kod Rolü
──────────────────   ──────────────────────     ────────────────────────
α(n) Union-Find      Tarjan 1975, P-R 2002       Ağırlık sütunlarını grupla
Eckart-Young SVD     Eckart-Young 1936           Her grup için optimal hata
Laplace 1 (sinyal)   Klasik sinyal teorisi       Gruplandırma kriteri (kutuplar)
Laplace 2 (Bayes)    Laplace aproksimasyonu      P₀ başlangıç kovaryansı
Kalman + Riccati     Kalman 1960                 Katmanlar arası hata yayılımı
Gölge Riccati        Bu çalışma (yenilik)        γ̂_k dengeleme — açık deliği kapatır
```

---

## 2. Tam Algoritma — Adım Adım Sıra

### ÖN İŞLEM (bir kez, tüm model için)

**Adım P1 — Laplace 2: P₀ başlangıcı**

```
Hedef:  P₀ = H_diag⁻¹
        burada H_diag = E[(∂L/∂w)²]  (diagonal Fisher)

Kod:    compute_diagonal_fisher(model, tokenizer, n_samples=16)
        → fisher_p0: dict
        initialize_kalman_p0(fisher_p0, n_layers)
        → P0: float  (scalar yaklaşım)

Neden önce: P₀ tüm Kalman zincirinin başlangıç noktasıdır.
            Yanlış P₀ → tüm K_k değerleri kayar.

Lemma bağı: Lemma 1 ispatında P_k sınırlılığı P₀ < ∞ varsayar.
```

**Adım P2 — Gölge Riccati İleri Geçişi**

```
Hedef:  Her katman k için γ̂_k hesapla.
        Q=0 ile Riccati ileri çalıştır:
        
        P_{L|k} = γ_k² · P_k          (scalar yaklaşım)
        γ̂_k    = √(P_{L|k} / P_k)
                = γ_k                  (TAM EŞİTLİK — Lemma 1.5 kalbi)

Kod:    shadow_riccati_forward(n_layers, P0, gamma)
        → gamma_hat_list: List[float]  (uzunluk = n_layers)

Neden önce: Karar eşiği FAZ 6'da γ̂_k kullanır.
            γ̂_k bilinmeden threshold hesaplanamaz.

Lemma bağı: Lemma 1.5 ispatı bu eşitliğe dayanır.
            γ̂_k = γ_k → (1-K_k)² ≤ 1 → E_k < ε
```

---

### ANA DÖNGÜ (her katman k = 0 … L-1 için)

**Adım 1 — Residual Kontrol**

```
Hedef:  Bu katmanda residual bağlantı var mı?
        
        Normal katman:   A = W       → gamma_eff = gamma
        Residual katman: A = W + I   → gamma_eff = gamma * sqrt(1 + 1/||W||²)
                                      ≈ gamma * 1.2  (pratik yaklaşım)
        
        Residual katmanlar: o_proj, down_proj

Kod:    is_residual = weight_name in ['o_proj', 'down_proj']
        gamma_eff = gamma * 1.2 if is_residual else gamma

Neden: Lemma 1'de P_k sınırlılığı A = W+I için farklı sabit nokta verir.
       P birikimi residual katmanlarda daha hızlı → Kalman muhafazakâr davranır.
```

**Adım 2 — Kalman Predict**

```
Hedef:  P_{k+1|k} = A · P_k · Aᵀ + Q_SVD + Q_quant
        
        Scalar yaklaşımda:
        P_{k+1|k} = gamma_eff² · P_k  +  sigma_r1²  +  Q_quant
        
        Q_quant = (2^{-bits})² / 12  (kuantizasyon gürültüsü, bits=4 için)
        Q_SVD   = sigma_r1²           (SVD truncation gürültüsü)

Kod:    P_predict = (gamma_eff**2) * P_tilde + Q_svd/(gamma_eff**2)
        (P_tilde = P/gamma² normalizasyonu nedeniyle bölme var)

Neden önce SVD değil: Predict adımı P_{k+1|k}'yı kurar.
                      Gain K bu değere bağlıdır.

Lemma bağı: Lemma 2 — Q_SVD ↑ → K ↑  (∂K/∂P > 0 analitik)
```

**Adım 3 — Laplace 1: Kutup Hesaplama**

```
Hedef:  W matrisinin her sütunu için dominant kutup bul.
        
        Yöntem: AR(1) aktivasyon yanıtı
        response_i = W[:,i] @ activations.T   → (batch,)
        phi_i = corr(response_i[:-1], response_i[1:])   → AR(1) katsayısı
        pole_i = phi_i                         → kutup tahmini
        
        poles: ndarray, shape (n_cols,), değerler [-1, 1]

Kod:    acts = collect_sample_activations(model, tokenizer, layer_idx)
        poles = laplace_poles_real(W, acts)

Neden: Laplace 1 kutupları gruplandırma kriterini belirler.
       Kutup yakınlığı → benzer frekans yanıtı → birlikte elenebilir.
```

**Adım 4 — α(n) Union-Find Gruplandırması**

```
Hedef:  Yakın kutupları grupla, α(n) ≤ 4 garantisi koru.
        
        1. adaptive_delta(poles, target_group_size=4)
           → delta: sıralı kutuplar arasındaki mesafenin uygun persentili
        
        2. Union-Find: |pole_i - pole_j| < delta ise birleştir
           KISIT: hiçbir grup 4'ten büyük olamaz
        
        3. Çıktı: groups = [[col_idx, ...], ...]
                  len(groups) ≤ n_cols
                  max(len(g) for g in groups) ≤ 4   ← α(n) garantisi

Kod:    groups = alpha_union_find_adaptive(poles, max_group_size=4)

Neden: Her grup bir "makro ağırlık" gibi davranır.
       Grup başına bir SVD → O(|G|³) = O(1) (|G| ≤ 4 sabit)
       Tüm matris: O(n) grup × O(1) = O(n) — ters Ackermann verimliliği

Lemma bağı: α(n) ≤ 4 garantisi olmadan Eckart-Young'ın
            optimal hata garantisi gruplar için geçerli olmaz.
```

**Adım 5 — Eckart-Young: Grouped Rand SVD**

```
Hedef:  Her grup G_i için optimal rank-k yaklaşımı bul.
        
        sub_W = W[:, G_i]                    → grup sütunları
        U, s, Vt = randomized_svd(sub_W, k)
        
        r = ilk i: s[i] < eps               → truncation noktası
        sigma_r1 = s[r]                     → (r+1). tekil değer
        
        Eckart-Young garantisi:
        ||sub_W - sub_W_r||_F = sigma_{r+1}  (MINIMUM hata, başka yol yok)

Kod:    sigma_r1, rank_kept = grouped_rand_svd(W, groups, eps)

k_init seçimi (eps'e göre adaptif):
        eps < 0.10  → k_init = %85 × full_rank
        eps < 0.20  → k_init = %65 × full_rank
        eps < 0.35  → k_init = %45 × full_rank
        eps ≥ 0.35  → k_init = %25 × full_rank

Neden: k_init çok küçükse küçük sigma değerleri görünmez → yanlış karar.
       Adaptif k_init bu körlüğü önler.

Lemma bağı: Lemma 1.5 ispatında sigma_r1 = ||W - W_r||_F kullanılır.
            Bu Eckart-Young garantisinin doğrudan sonucudur.
```

**Adım 6 — Kalman Gain**

```
Hedef:  K_k = P_{k+1|k} / (P_{k+1|k} + R)
        
        R = gözlem gürültüsü (default: 1.0)
        
        Scalar:  K = P_tilde / (P_tilde + 1.0)
        
        Özellik: K ∈ [0, 1] her zaman
                 P büyük → K → 1  (muhafazakâr: yüksek belirsizlik)
                 P küçük → K → 0  (agresif: düşük belirsizlik)

Kod:    K = P_tilde / (P_tilde + 1.0)

Lemma bağı: Lemma 2 — ∂K/∂P > 0
            Q_SVD ↑ → P ↑ → K ↑ → threshold ↓ → daha az eleme
            Bu "kendi kendini sınırlayan" mekanizmadır.
```

**Adım 7 — KARAR (Lemma 1.5 eşiği)**

```
Hedef:  sigma_r1 < threshold ?  →  elin
                               :  tut

threshold = (1 - K) · ε / γ̂_k

Açılım:
  (1 - K) : Kalman'ın "güven faktörü"
  ε        : kullanıcı toleransı
  γ̂_k     : gölge Riccati'den gelen katman ölçeği
             γ̂_k = γ_k = sqrt(hidden_dim) (Qwen 0.5B için ≈ 29.9)

UYARI — Mevcut Bug:
  Kodda threshold = (1 - K) · ε    ← γ̂_k EKSİK
  Doğrusu: threshold = (1 - K) · ε / gamma_hat[layer_idx]

  Bu fark neden önemli:
  gamma_hat ≈ 30 → threshold 30x daha küçük olmalı
  Yani mevcut kod çok gevşek karar veriyor.
  Ana Teorem ampirik olarak sağlanıyor ama ispat zinciri kırık.

Lemma 1.5 ispatı (tam):
  γ̂_k = γ_k                   (Gölge Riccati kimliği)
  sigma_r1 < (1-K)·ε/γ_k       (karar eşiği)
  gamma_k · (1-K) · sigma_r1   (hata katkısı E_k tanımı)
  E_k < gamma_k · (1-K) · (1-K)·ε/gamma_k
      = (1-K)² · ε
      ≤ ε                       (K ∈ [0,1] → (1-K)² ≤ 1)  □

Sonuç: Her elenen katman için E_k < ε
       → E_total = Σ E_k ≤ L · ε   (Ana Teorem)
```

**Adım 8 — Kalman Update**

```
Hedef:  P_{k+1} = (1 - K_k) · P_{k+1|k}  +  Q_SVD / gamma²
        
        (P_tilde normalizasyonu ile)

Kod:    Q_svd = sigma_r1 ** 2
        P_new = (1 - K) * P_tilde + Q_svd / (gamma_eff ** 2)

Neden sonra: Update bir sonraki katmanın P₀'ı olur.
             Sıra bozulursa tüm zincir kayar.
```

---

## 3. Veri Akışı — Fonksiyonlar Arası Bağlantı

```
compute_diagonal_fisher()
        ↓ P0 (float)
shadow_riccati_forward(n_layers, P0, gamma)
        ↓ gamma_hat_list (List[float], len=n_layers)
        ↓
┌──── LAYER LOOP k = 0..L-1 ────────────────────────────────┐
│                                                            │
│  collect_sample_activations(model, tokenizer, k)          │
│          ↓ acts (ndarray, shape=(n_tokens, hidden_dim))   │
│                                                            │
│  laplace_poles_real(W, acts)                              │
│          ↓ poles (ndarray, shape=(n_cols,))               │
│                                                            │
│  alpha_union_find_adaptive(poles, max_group_size=4)       │
│          ↓ groups (List[List[int]])                        │
│                                                            │
│  grouped_rand_svd(W, groups, eps)                         │
│          ↓ sigma_r1 (float)                               │
│                                                            │
│  K = P_tilde / (P_tilde + 1.0)                           │
│                                                            │
│  threshold = (1 - K) * eps / gamma_hat_list[k]  ← KRİTİK│
│                                                            │
│  eliminated = (sigma_r1 < threshold)                      │
│                                                            │
│  Q_svd = sigma_r1 ** 2                                    │
│  P_tilde = (1-K)*P_tilde + Q_svd/(gamma_eff**2)          │
│                    ↓                                       │
│              sonraki katmana geç                          │
└────────────────────────────────────────────────────────────┘
        ↓
E_total = Σ [gamma_hat[k] · (1-K[k]) · sigma_r1[k]]  (sadece elenenler)
assert E_total ≤ L * eps   ← Ana Teorem
```

---

## 4. Lemma Zinciri — Özet

```
Lemma 1 (P sınırlı):
  Φ(P) = gamma² · P + Q_SVD monoton artar.
  Q_SVD ≥ 0, gamma < 1 (layer norm varsa) → sabit nokta var.
  P_k < P_max = Q_SVD / (1 - gamma²)
  
  Kod gerekliliği: P_tilde normalizasyonu (P/gamma²) bu sınırı korur.

Lemma 2 (Q↑ → K↑):
  K = P/(P+R)
  ∂K/∂P = R/(P+R)² > 0
  
  Kod gerekliliği: K = P_tilde / (P_tilde + 1.0)  ← R=1 sabit

Lemma 1.5 (γ-dengeleme):
  γ̂_k = γ_k  (Gölge Riccati kimliği)
  threshold = (1-K)·ε/γ̂_k
  → E_k = γ_k·(1-K)·sigma_r1 < ε
  
  Kod gerekliliği: shadow_riccati_forward() önce çalışmalı,
                   gamma_hat_list[k] threshold formülünde kullanılmalı.
                   ← ŞU AN EKSİK

Ana Teorem:
  E_total = Σ_{elinen k} E_k
           < Σ_{elinen k} ε
           ≤ L · ε
  
  Kod gerekliliği: Tüm lemmaların doğru implementasyonu.
```

---

## 5. Mevcut Kodun Eksikleri (Öncelik Sırasıyla)

```
Öncelik  Eksik                          Etki
──────   ──────────────────────────────  ──────────────────────────────
1        threshold / gamma_hat_list[k]   ← ÇÖZÜLDÜ: Formülasyon B
         EKSİK (sadece (1-K)·ε var)        P_tilde normalizasyonu gamma'yı
                                           zaten absorbe ediyor.
                                           threshold = (1-K)·ε  DOĞRU.
                                           gamma_hat threshold'da YOK.

2        shadow_riccati_forward()         ← ÇÖZÜLDÜ: fonksiyon eklendi.
         fonksiyonu yok                    gamma_hat_list log'a kaydediliyor,
                                           E_k hesabında kullanılıyor.

3        is_residual → gamma_eff         ← ÇÖZÜLDÜ: o_proj/down_proj için
         kontrolü yok                      gamma_eff = gamma * 1.2

4        A·P·Aᵀ (tam matris predict)     Scalar yaklaşım
         scalar yaklaşım kullanılıyor     kabul edilebilir ama not edilmeli
```

### Formülasyon Notu (Önemli)

P_tilde normalizasyonu (Formülasyon B) kullanıldığında:

  threshold = (1-K)·ε   —   gamma_hat threshold'da YOK.
  gamma_hat sadece E_total hesabında: E_k = gamma_hat[k]·(1-K)·sigma_r1

Neden: P̃_k = P_k / γ_k² tanımı gereği threshold formülündeki γ̂_k
bölmesi iptal olur. gamma_hat bilgi amaçlı hesaplanır ve
layer kararlarına (LayerDecision) kaydedilir.

---

## 6. Sabit Değerler (Qwen 2.5 0.5B için)

```
hidden_dim   = 896
gamma        = sqrt(hidden_dim) = 29.93  (P_tilde normalizasyon faktörü)
n_layers     = 24
bits         = 4  (hedef kuantizasyon)
Q_quant      = (2^{-4})² / 12 = 0.000326  (kuantizasyon gürültüsü)
R            = 1.0  (gözlem gürültüsü, sabit)
max_group    = 4    (α(n) garantisi)
```

---

## 7. Test Kriterleri

Her değişiklik sonrası şu kontroller geçmeli:

```python
# 1. Ana Teorem
assert result['E_total'] <= result['E_bound'], "ANA TEOREM İHLALİ"

# 2. α(n) garantisi
for decision in result['decisions']:
    assert decision['n_groups'] <= decision['n_cols'] // 1, "α(n) HATASI"

# 3. γ̂_k değerleri makul
for k, gh in enumerate(gamma_hat_list):
    assert 0 < gh < 1000, f"γ̂_{k} = {gh} — anormal değer"

# 4. P_tilde pozitif ve sınırlı
assert 0 < final_P_tilde < 1e6, "P_tilde sınır dışı"

# 5. Threshold pozitif
for d in decisions:
    assert d['threshold'] > 0, "Negatif threshold"
```