# UltraGen V2 — Matematiksel Referans Tanımı (KİLİTLİ)

Bu doküman, UltraGen V2 aşamasında kullanılan **feature → DSP parametre uzayı projeksiyonu**nun
matematiksel olarak **tanımlanmış ve kilitlenmiş** referansını içerir.

Kapsam bilinçli olarak sınırlıdır:
- End-to-end audio generation YOK
- Deep learning ile ses üretimi YOK
- Frame-level / embedding analizi YOK
- Sadece **global + yavaş varyanslı feature → parametre uzayı** eşlemesi

Referanslar:
- V2 Feature Set (kilitli)
- V2 DSP Parametre Seti
- Rule-Based Initial Mapping
- musicdsp prensipleri (noise, filter, RMS, envelope)

---

## 1) Feature Normalization Stratejisi

### 1.1 Genel Notasyon

- Feature vektörü:  
  \[
  \mathbf{f} = (f_1, f_2, \dots, f_d)
  \]

- Normalize edilmiş feature:  
  \[
  \mathbf{z} = (z_1, z_2, \dots, z_d), \quad z_i \in [-1, 1]
  \]

Normalize edilmiş uzay **tüm V2 boyunca sabittir**.

---

### 1.2 Robust Merkez ve Ölçek

Her feature için veri kümesi üzerinde:

- Merkez:
  \[
  m_i = \mathrm{median}(f_i)
  \]

- Ölçek:
  \[
  \sigma_i = 1.4826 \cdot \mathrm{MAD}(f_i)
  \]

- Aykırı değer kırpma:
  \[
  f_i^{clip} = \mathrm{clip}(f_i, m_i - 3\sigma_i, m_i + 3\sigma_i)
  \]

**Gerekçe:**
- Gerçek dünya steady-state sesleri heavy-tail davranış gösterir
- Ortalama + standart sapma kararsızdır
- Median + MAD deterministik ve stabildir

---

### 1.3 Feature Türüne Göre Normalizasyon

#### (A) Enerji ve Dinamik Feature’lar
- RMS_mean
- RMS_std_slow
- Crest_factor

**Yöntem:**
\[
u_i = \frac{f_i^{clip} - m_i}{\sigma_i + \varepsilon}
\]
\[
z_i = \tanh(\alpha u_i)
\]

Sabitler:
- \(\alpha = 0.5\)
- \(\varepsilon = 10^{-12}\)

**Gerekçe:**
- RMS ve crest factor pozitif ve long-tail
- \(\tanh\) uç değerleri yumuşak biçimde sınırlar
- ML ve rule-based sistemler için güvenli aralık

---

#### (B) Spektral Feature’lar (frekans ölçekli)
- Spectral_centroid_mean
- Spectral_rolloff_85_mean

**Önce log dönüşüm:**
\[
g_i = \log(f_i + \varepsilon)
\]

Sonra (A) ile normalize edilir.

**Gerekçe:**
- Frekans algısı logaritmiktir
- Lineer normalize perceptual karşılığı bozardı

---

#### (C) Spektral Oran ve Doku Feature'ları
- Spectral_flatness_mean
- Low_band_ratio
- Mid_band_ratio
- Zero_crossing_rate_mean

**Robust min–max:**
\[
z_i = 2 \cdot \frac{\mathrm{clip}(f_i, Q_{1\%}, Q_{99\%}) - Q_{1\%}}
{Q_{99\%} - Q_{1\%}} - 1
\]

**Gerekçe:**
- Doğal olarak sınırlı oranlar
- Log dönüşüm gerekli değil
- Oran ilişkisi korunur

---

#### (D) Spektral Eğilim
- Spectral_tilt_estimate (dB/oktav)

**Doğrudan (A) tipi normalize edilir.**

**Gerekçe:**
- Zaten lineer, çift yönlü bir ölçüdür
- White / pink / brown ayrımı korunur

---

#### (E) Yavaş Modülasyon Göstergesi
- Amplitude_modulation_index_slow

(A) tipi normalize edilir, **çok düşük varyanslı** kabul edilir.

---

## 2) Yakın Parametre Uzayı Tanımı

### 2.1 Parametre Notasyonu

DSP parametre vektörü:
\[
\mathbf{p} = (p_1, p_2, \dots, p_M)
\]

Her parametre için kilitli değerler:
- Alt sınır \(L_j\)
- Üst sınır \(U_j\)
- Merkez (nominal) \(\mu_j\)

---

### 2.2 Standartlaştırılmış Parametre Uzayı

\[
s_j = \frac{p_j - \mu_j}{r_j}, \quad
r_j = \frac{U_j - L_j}{2}
\]

Hedef:
\[
s_j \in [-1, 1]
\]

---

### 2.3 Feature → Parametre Projeksiyonu

Normalize feature vektörü \(\mathbf{z}\) için:

\[
\mathbf{s} = \mathrm{clip}(\mathbf{A}\mathbf{z} + \mathbf{b},
- \mathbf{s}_{max}, \mathbf{s}_{max})
\]

\[
\hat{\mathbf{p}} = \mu + r \odot \mathbf{s}
\]

Burada:
- \(\mathbf{A}\): rule-based initial mapping’den gelen duyarlılık matrisi
- \(\mathbf{b}\): bias (genelde 0)
- \(\mathbf{s}_{max}\): parametreye özel **yakınlık limiti**

---

### 2.4 Feature Varyansı → Parametre Varyansı

Normalize feature varyansı:
\[
v_i = \mathrm{Var}(z_i)
\]

Parametreye bağlı izinli varyans:
\[
\Delta_j = r_j \cdot \beta_j
\cdot \sqrt{\sum_i A_{j,i}^2 v_i}
\]

- \(\beta_j \in (0,1]\): parametreye özel yavaş varyans katsayısı
- Bu tanım **hangi feature → hangi parametre** sorusunu açıklanabilir kılar

Yakın parametre uzayı:
\[
p_j \in [\hat{p}_j - \Delta_j,\; \hat{p}_j + \Delta_j]
\]

---

## 3) Sanity Gate Matematiği

Sanity gate fonksiyonu:
\[
G(\mathbf{p}) \in \{0,1\}
\]

### 3.1 Temel Aralık Kısıtları

\[
L_j \le p_j \le U_j
\]

---

### 3.2 Gain / Mix Bütçesi

Katman kazançları:
\[
\sum_{\ell} g_\ell \le 1.0
\]

Dry/Wet, send, mix parametreleri:
\[
0 \le m \le 0.35
\]

---

### 3.3 Filtre İlişkileri

\[
0 < f_{hp} < f_{lp} < \frac{f_s}{2}
\]

\[
f_{lp} - f_{hp} \ge 50\;\text{Hz}
\]

---

### 3.4 Rezonans / Q Sınırı

\[
Q \le 1.2
\]

---

### 3.5 EQ ve Tilt Bütçesi

Tilt:
\[
|t| \le 6\;\text{dB}
\]

Üç bant EQ:
\[
|e_L|, |e_M|, |e_H| \le 4\;\text{dB}
\]
\[
|e_L| + |e_M| + |e_H| \le 8\;\text{dB}
\]

---

### 3.6 Modülasyon Sınırları

\[
0 < r_{lfo} \le 1.0\;\text{Hz}
\]
\[
0 \le d_{lfo} \le 0.25
\]

Bağlı kısıt:
\[
d_{lfo} \le \frac{0.08}{r_{lfo} + 10^{-3}}
\]

---

### 3.7 Stereo Genişlik

\[
0 \le w \le 0.35
\]

---

### 3.8 RMS Güvenliği

\[
R_{target} \le 0.2
\]

---

## Sonuç (Kilitli Tanım)

- Normalizasyon: **robust + log (gerektiğinde) + tanh**
- Parametre uzayı: **merkez + yakın varyans**
- Sanity gate: **bütçe + ilişkisel eşitsizlikler**

Bu doküman, UltraGen V2’nin
**değiştirilemez matematiksel referansı**dır.


# UltraGen V2 — Matematiksel Tanım  
## Özet Denetim Raporu

Bu belge, UltraGen V2 için tanımlanmış matematiksel referansın
teknik, matematiksel ve DSP uyumluluğu açısından denetim özetidir.

---

## Genel Değerlendirme

| Alan                          | Durum                |
|-------------------------------|----------------------|
| Kapsam Uyumu                 | Uygun               |
| Feature Normalization        | Uygun               |
| Parametre Projeksiyonu       | Uygun               |
| Parametre Varyansı Tanımı    | Düşük Risk (kontrollü) |
| Sanity Gate Kuralları        | Uygun               |
| DSP Prensipleriyle Uyum      | Tam uyum            |
| V2 Kapsam Aşımı              | Yok                 |

---

## Belirlenen Risk (Tek Nokta)

**Parametre varyansı tanımı** (2.4) matematiksel olarak geçerli  
ancak feature korelasyonları göz önüne alınmadığında bazı  
parametrelerde varyans kontrolü zayıflayabilir.

Açıklama:  
Bu durum “aşırı oynak” davranış üretmez ancak  
yüksek-dereceden kombinasyonlar dikkatle izlenmelidir.

---

## Geçerlilik Kararı

Bu doküman:

- V2 sisteminin matematiksel çekirdeğini başarıyla tanımlar  
- DSP disiplini ve ses sentezi prensiplerine uygundur  
- V2 kapsamını aşan hiçbir yapı içermez

Geçerli — Kilitlenebilir Referans olarak kullanılabilir.


Bu varyans tanımı, feature’ların bağımsızlığı varsayımı altında tanımlanmıştır; yüksek korelasyonlu feature setlerinde varyans üst sınırları sanity gate tarafından ek olarak sınırlandırılır.