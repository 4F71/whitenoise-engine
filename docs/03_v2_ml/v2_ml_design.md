# ultragen v2 — ml tasarımı

Status: ACTIVE (C1)  
Scope: Feature → DSP parametre projeksiyonu  
Out of scope: Audio synthesis, generative DL, real-time inference

Bu doküman, UltraGen V2 aşamasında
makine öğrenmesinin **nasıl, neden ve hangi sınırlar içinde**
kullanıldığını tanımlar.

Amaç, ML’i ses üretimi yapan bir bileşen olarak değil,
**parametre tahmin eden yardımcı bir katman** olarak konumlandırmaktır.

---

## 1. ML’in Rolü (Net Tanım)

UltraGen V2’de ML:

- Ses üretmez
- Waveform üretmez
- DSP algoritmalarının yerine geçmez

ML’in tek görevi:

> Gerçek dünya seslerinden çıkarılan özellikleri,
> UltraGen’in tanımlı DSP parametre uzayına
> kontrollü ve açıklanabilir şekilde projekte etmek

Bu yaklaşım, sistemin:

- Deterministik
- Denetlenebilir
- Akademik olarak gerekçelendirilebilir

kalmasını sağlar.

---

## 2. Girdi Verisi

### 2.1 Dataset Yapısı

ML, yalnızca aşağıdaki şartları sağlayan verilerle çalışır:

- 60 saniye
- Mono
- 48 kHz
- Telifli ama yalnızca **analiz amaçlı**
- Raw ve processed ayrımı yapılmış

Dataset içeriği:

- room_tone
- mechanical_hum
- wind_ambience
- distant_wind
- urban_background

Toplam örnek sayısı sınırlıdır (V2 için ~17).

Bu bilinçli bir tercihtir.

---

### 2.2 Feature Extraction

Her ses için çıkarılan temel feature grupları:

- Enerji ve dinamik özellikler (RMS, crest factor)
- Spektral özellikler (centroid, flatness, rolloff)
- Spektral eğim (tilt)
- Amplitude modulation göstergeleri

Feature’lar:

- Zaman boyunca özetlenir (mean, std, quantile)
- Normalize edilir
- Aykırı değerler bastırılır

Detaylar için:
`docs/02_v2_theory/v2_mathematical_lock.md`

---

## 3. Model Seçimi

V2’de kullanılan ML yaklaşımı:

- Basit
- Açıklanabilir
- Küçük veriyle çalışabilir

Tercih edilen model türleri:

- Linear regression
- Ridge regression
- Basit MLP (opsiyonel)

Derin, generative veya black-box modeller
**bilinçli olarak kullanılmaz**.

Gerekçe:

- Veri miktarı
- Kontrol ihtiyacı
- Mimari sürdürülebilirlik

---

## 4. Feature → Parametre Projeksiyonu

ML çıktısı:

- Doğrudan DSP parametresi değildir
- Önce normalize edilmiş parametre uzayına düşer

Akış:

```
Audio → Features → Normalization → ML → Parametre Uzayı → Sanity Gate → DSP
```

Bu aşamada:

- Tüm parametreler [-1, 1] aralığındadır
- Fiziksel ve algısal sınırlar korunur
- Sanity gate her zaman son sözü söyler

---

## 5. Sanity Gate ve Güvenlik

ML çıktıları, tek başına uygulanmaz.

Her parametre:

- Alt / üst sınır
- Karşılıklı ilişki
- Toplam bütçe

kurallarına tabidir.

Bu sayede:

- Clip
- Aşırı rezonans
- Duyulabilir artefaktlar

engellenir.

ML, **önerir**.  
DSP ve kurallar **karar verir**.

---

## 6. Çıktı Tanımı

V2 ML katmanının çıktısı:

- Tek bir preset değildir
- Bir ses üretimi değildir

Çıktı:

- Bir referans preset (P0)
- Bu preset etrafında tanımlı varyasyon alanı

Preset aileleri, ML çıktısı üzerine
kural tabanlı olarak inşa edilir.

Detaylar için:
`docs/02_v2_theory/v2_preset_family.md`

---

## 7. V2 Sınırları

Aşağıdakiler **V2 kapsamı dışındadır**:

- End-to-end audio generation
- Diffusion / GAN / transformer tabanlı modeller
- Real-time inference
- Adaptive online learning

Bu konular V3+ aşamalarında ele alınacaktır.

---

## 8. Özet

V2 ML tasarımı:

- Küçük ama sağlam
- Açıklanabilir
- Mimariyle uyumlu

Bu katman, UltraGen’in
ileride daha karmaşık sistemlere evrilmesini
mümkün kılan **kontrollü bir köprü** görevi görür.
