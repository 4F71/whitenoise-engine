# UltraGen V2 — DSP Parametre Seti (Kilitli)

Bu doküman, UltraGen V2 aşamasında
ML projeksiyonunun hedeflediği
**DSP parametre uzayını** tanımlar.

Amaç:
Gerçek dünya broadband / steady-state seslerinden
çıkarılan feature’ların,
**yakın parametre uzayı (mean + range/std)**
mantığıyla UltraGen DSP motoruna
projeksiyonunu mümkün kılmak.

---

## Genel İlkeler

- Parametre sayısı **sabit ve sınırlıdır**
- Her parametre:
  - bir **merkez değer (mean)**
  - bir **yakın uzay aralığı (range / std)**
  ile temsil edilir
- ML modelleri bu uzayın **etrafında öğrenir**,
  uzayın dışına taşmaz
- V2 kapsamı dışına çıkan parametreler eklenmez

Toplam parametre sayısı: **12**

---

## A) Noise / Spektral Çekirdek Parametreleri (4)

### 1) `noise_type` *(kategorik)*
**Seçenekler:**
- `white`
- `pink`
- `brown`
- `blue`
- `violet`

**Açıklama:**
Temel noise rengi ve spektral karakter.

**Not:**
Tek bir seçim yerine,
öncelik sıralaması (örn. `pink > brown`)
üretilebilir.

---

### 2) `lp_cutoff_hz` *(sürekli)*
**Açıklama:**
Alçak geçiren filtrenin kesim frekansı.

**Algısal Etki:**
- Parlaklık
- Koyuluk / yumuşaklık

**Yakın Uzay:**
- Geniş ama sınırlı
- Ortalama etrafında kontrollü varyasyon

---

### 3) `hp_cutoff_hz` *(sürekli, dar aralık)*
**Açıklama:**
Yüksek geçiren filtrenin kesim frekansı.

**Algısal Etki:**
- Rumble ve DC offset temizliği

**Yakın Uzay:**
- Dar aralık
- Karakteri bozmayacak şekilde sınırlandırılmış

---

### 4) `spectral_tilt_db` *(sürekli)*
**Açıklama:**
Global spektral eğim (noise rengi ince ayarı).

**Algısal Etki:**
- White ↔ pink ↔ brown benzeri davranışlar

**Yakın Uzay:**
- ±1–2 dB ile sınırlı

---

## B) Enerji / Dinamik Parametreleri (3)

### 5) `gain_db` *(sürekli)*
**Açıklama:**
Genel kazanç seviyesi.

**Algısal Etki:**
- Loudness dengesi
- Preset’ler arası seviye tutarlılığı

**Yakın Uzay:**
- Ortalama etrafında birkaç dB

---

### 6) `soft_saturation_amount` *(sürekli, çok küçük)*
**Açıklama:**
Yumuşak doygunluk miktarı.

**Algısal Etki:**
- Crest factor yumuşatma
- Spike bastırma

**Yakın Uzay:**
- Çok dar
- Çoğu durumda sıfıra yakın

---

### 7) `normalize_target_db` *(opsiyonel)*
**Açıklama:**
Preset’ler arası normalize hedef seviyesi.

**Örnek:**
- −16 dBFS

**Not:**
İsteğe bağlıdır ancak
tutarlılık için önerilir.

---

## C) Modülasyon (Hareket) Parametreleri — LFO (3)

### 8) `lfo_rate_hz` *(sürekli, çok düşük frekans)*
**Açıklama:**
LFO hız değeri.

**Algısal Etki:**
- Akış
- Rüzgar benzeri hareket hissi

**Yakın Uzay:**
- Çok düşük Hz bandı
- Ortalama etrafında kontrollü varyasyon

---

### 9) `lfo_depth` *(sürekli)*
**Açıklama:**
Modülasyon derinliği.

**Algısal Etki:**
- Hareketin şiddeti

**Yakın Uzay:**
- Orta genişlik
- Üst sınırı kısıtlı

---

### 10) `lfo_target` *(kategorik)*
**Seçenekler:**
- `gain`
- `filter`
- `both`

**Açıklama:**
LFO’nun hangi parametreyi modüle ettiği.

---

## D) Uzamsal / Sunum Parametreleri (2)

### 11) `stereo_width` *(sürekli)*
**Açıklama:**
Stereo genişlik kontrolü.

**Algısal Etki:**
- Ambience hissi
- Çevresellik

**Yakın Uzay:**
- Dar
- Abartılı genişlikten kaçınılır

---

### 12) `reverb_send` *(sürekli, çok sınırlı)*
**Açıklama:**
Reverb gönderim miktarı.

**Algısal Etki:**
- Algısal derinlik

**Yakın Uzay:**
- Çok dar
- Çoğu preset’te sıfıra yakın

---

## Özet

| Parametre Grubu | Adet |
|-----------------|------|
| Noise / Spektral | 4 |
| Enerji / Dinamik | 3 |
| Modülasyon | 3 |
| Uzamsal | 2 |
| **Toplam** | **12** |

---

## Kapsam Notları

- Bu parametre seti **V2 için kilitlidir**
- Yeni parametre eklenmez
- Parametre sayısı artırılmaz
- ML modelleri bu uzayı öğrenir,
  yeniden tanımlamaz

Bu doküman, UltraGen V2’nin
ML ve DSP entegrasyonunda
hedeflenen parametre uzayını
resmi olarak tanımlar.
