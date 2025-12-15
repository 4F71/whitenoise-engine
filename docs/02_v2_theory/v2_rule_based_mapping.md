# UltraGen V2 — Kural Tabanlı Başlangıç Parametre Eşlemesi

**Durum:** KİLİTLİ (V2)  
**Kapsam:** ML projeksiyonu öncesi pseudo-label (başlangıç etiketleri) üretimi  
**Amaç:** Gerçek dünya seslerinden çıkarılan feature’ların, UltraGen DSP
parametre uzayına deterministik ve açıklanabilir biçimde eşlenmesi.

> Not:
> Bu doküman, V2 aşamasında kullanılan **kural tabanlı referans haritayı**
> tanımlar.  
> V2’de kullanılan ML modellerinin amacı bu haritayı **öğrenerek
> yumuşatmak ve yakın parametre uzayını tahmin etmek**tir;  
> bu kuralları tamamen değiştirmek veya çelişmek değildir.

---

## 1. Tasarım İlkeleri

- **Açıklanabilirlik:**  
  Her DSP parametresi, ölçülebilir bir feature’a dayanır.
- **Deterministik başlangıç:**  
  ML “icat etmez”, mevcut DSP bilgisinin etrafında öğrenir.
- **Yakın parametre uzayı:**  
  Tek bir “en iyi değer” yerine merkez + aralık (range/std) hedeflenir.
- **V2 odaklılık:**  
  Broadband / steady-state noise ve ambience dışına çıkılmaz.

---

## 2. Feature → DSP Parametre Eşleme Kuralları

### 2.1 Noise Türü (`noise_type`)
**Girdi Feature’lar:**  
- `spectral_tilt_estimate`  
- `spectral_flatness_mean`  
- `zero_crossing_rate_mean`

**Kural Mantığı:**
- Spektral eğim ≈ −3 dB/oktav ve flatness yüksek → **pink**
- Spektral eğim ≤ −6 dB/oktav ve low-band baskın → **brown**
- Spektral eğim ≈ 0 dB/oktav ve flatness çok yüksek → **white**
- Pozitif eğim ve yüksek ZCR → **blue / violet**

**Çıktı:**  
Tek bir seçim değil, **öncelik sıralaması**  
(örn. `pink > brown`)

---

### 2.2 Alçak Geçiren Filtre Kesimi (`lp_cutoff_hz`)
**Girdi Feature’lar:**  
- `spectral_centroid_mean`  
- `spectral_rolloff_85_mean`

**Kural:**
- Ortalama cutoff ≈ rolloff_85 değerinin %60–80’i
- Aralık: ±%25

---

### 2.3 Yüksek Geçiren Filtre Kesimi (`hp_cutoff_hz`)
**Girdi Feature’lar:**  
- `low_band_ratio`  
- `RMS_mean`

**Kural:**
- Low-band baskınlığı yüksek → 30–50 Hz
- Düşük → 20–30 Hz
- Aralık dar tutulur (±10 Hz)

---

### 2.4 Spektral Eğilim (`spectral_tilt_db`)
**Girdi Feature:**  
- `spectral_tilt_estimate`

**Kural:**
- Ortalama = ölçülen eğim
- Aralık: ±1–2 dB

---

### 2.5 Kazanç (`gain_db`)
**Girdi Feature:**  
- `RMS_mean`

**Kural:**
- Ortalama kazanç = hedef seviye − RMS_mean
- Aralık: ±3 dB

---

### 2.6 Yumuşak Doygunluk (`soft_saturation_amount`)
**Girdi Feature:**  
- `crest_factor`

**Kural:**
- Crest factor yüksek → düşük–orta doygunluk
- Crest factor düşük → çok düşük doygunluk
- Aralık dar tutulur

---

### 2.7 LFO Hızı (`lfo_rate_hz`)
**Girdi Feature’lar:**  
- `spectral_centroid_std_slow`  
- `amplitude_modulation_index_slow`

**Kural:**
- İkisi de düşük → 0.03–0.07 Hz
- Biri orta → 0.07–0.15 Hz
- Aralık: ±%30

---

### 2.8 LFO Derinliği (`lfo_depth`)
**Girdi Feature’lar:**  
- `RMS_std_slow`  
- `spectral_centroid_std_slow`

**Kural:**
- İki varyans da düşük → çok düşük derinlik
- Biri orta → düşük–orta
- Aralık: ±%40 (üst sınır kısıtlı)

---

### 2.9 LFO Hedefi (`lfo_target`)
**Girdi:**  
- Enerji varyansı / parlaklık varyansı oranı

**Kural:**
- Enerji varyansı baskın → `gain`
- Parlaklık varyansı baskın → `filter`
- Yakınsa → `both` (öncelik filtre)

---

### 2.10 Stereo Genişlik (`stereo_width`)
**Girdi Feature’lar:**  
- `spectral_flatness_mean`  
- `mid_band_ratio`

**Kural:**
- Flatness yüksek & mid dengeli → orta genişlik
- Tonal eğilim artıyorsa → düşük genişlik
- Aralık dar tutulur

---

### 2.11 Reverb Gönderimi (`reverb_send`)
**Girdi Feature’lar:**  
- `spectral_rolloff_85_mean`  
- `RMS_mean`

**Kural:**
- Yüksek frekanslar zayıf & RMS düşük → çok hafif reverb
- Diğer durumlar → sıfıra yakın
- Aralık çok dar

---

### 2.12 Normalizasyon Hedefi (`normalize_target`) [Opsiyonel]
**Girdi Feature:**  
- `RMS_mean`

**Kural:**
- Sabit hedef seviye (örn. −16 dBFS)
- Küçük tolerans payı

---

## 3. Sanity / Güvenlik Kuralları

- `lp_cutoff_hz` düşüldükçe `spectral_tilt_db` aşırılaşamaz
- `lfo_depth` yükseldikçe `lfo_rate_hz` üst sınırı düşürülür
- `hp_cutoff_hz` artarsa `gain_db` otomatik telafi edilir

---

## 4. Çıktı Formatı (Pseudo-Label)

Her DSP parametresi için:

- Ortalama değer (`mean`)
- Yakın uzay aralığı (`range` veya `std`)
- Kategorik parametreler için öncelik sırası

Bu çıktılar, V2 ML projeksiyon modeline **başlangıç etiketi** olarak verilir.

---

## 5. Kapsam Dışı (Bilinçli Olarak)

- Frame-level analiz
- End-to-end ses üretimi
- Deep learning tabanlı embedding’ler
- Event / transient modelleme

Bu sınırlamalar, V2’nin deterministik ve açıklanabilir kalması için **bilinçli** olarak konmuştur.
