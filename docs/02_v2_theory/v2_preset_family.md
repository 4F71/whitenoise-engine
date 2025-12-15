# ultragen v2 — preset family tanımı

Status: LOCKED (V2)  
Scope: Preset family generation (mean + controlled variants)  
Depends on:
- v2_mathematical_lock.md
- v2_dsp_parameter_set.md
- preset_to_dsp_adapter.py

Bu doküman, UltraGen V2 aşamasında
tek bir referans sesten türetilen
**preset ailesi** yapısını tanımlar.

Amaç:
V2’yi tekil preset üreten bir sistem olmaktan çıkarıp,
**deterministik ve karakter-korumalı preset aileleri**
üreten bir üretim aracına dönüştürmek.

---

## 1. Preset Ailesi Kavramı

Bir referans ses için:

- **P0** → merkez (mean) preset
- **P1–P5** → kontrollü varyantlar

Toplam:
- **6 preset / referans ses**

Her preset:
- Aynı karakter ailesine aittir
- Aynı matematiksel kilide tabidir
- Aynı DSP çekirdeğini kullanır

---

## 2. Deterministiklik İlkesi

V2’de preset üretimi:

- Rastgele değildir
- Aynı giriş → aynı çıktı üretir

Deterministiklik:
- Referans ID
- Varyant ID

üzerinden sağlanır.

Amaç:
- Reproducibility
- İçerik üretiminde tutarlılık
- ML çıktılarının izlenebilirliği

---

## 3. Preset Rolleri ve Davranış Profilleri

### P0 — reference
**Tanım:**  
ML veya kural tabanlı sistemden gelen
**mean parametre seti**.

**Davranış:**
- En nötr
- En temsil edici
- Diğer varyantların referans noktası

---

### P1 — soft
**Amaç:**  
Uzun süreli dinleme için güvenli ve yumuşak karakter.

**Davranış:**
- LP cutoff ↓
- LFO depth ↓
- Saturation ↓

---

### P2 — dark
**Amaç:**  
Daha koyu, daha bastırılmış spektral yapı.

**Davranış:**
- LP cutoff ↓↓
- Spectral tilt ↓
- Stereo width hafif ↓

---

### P3 — calm
**Amaç:**  
Daha statik, daha az hareketli ambience.

**Davranış:**
- LFO rate ↓
- LFO depth ↓↓
- Gain dengeli ↓

---

### P4 — airy
**Amaç:**  
Daha açık, daha nefes alan karakter.

**Davranış:**
- LP cutoff ↑
- Stereo width ↑ (sanity gate içinde)
- Spectral tilt ↑

---

### P5 — dense
**Amaç:**  
Daha dolgun ve algısal olarak yoğun ses.

**Davranış:**
- Gain dengeli ↑
- Saturation çok hafif ↑
- HP cutoff hafif ↓

---

## 4. Varyasyon Kuralları

- Varyasyonlar **yalnızca spread içinde**
- Sanity gate sınırları asla aşılmaz
- Karakter bozulmaz
- Parametreler bağımsız değil, **profil bazlı** değişir

---

## 5. Kapsam Notları

- Preset family mantığı **V2’ye özeldir**
- V3’te node/graph mimarisiyle yeniden yorumlanabilir
- V4’te AI evaluator preset aileleri arasında seçim yapabilir

Bu doküman, UltraGen V2 preset üretim davranışının
resmi ve bağlayıcı tanımıdır.
