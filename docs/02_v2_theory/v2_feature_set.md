# ultragen v2 — feature set

Status: LOCKED (V2)  
Scope: Broadband / steady-state audio feature set for parameter projection  
Out of scope: Transient/event features, embeddings, time-series ML

Bu doküman, UltraGen V2 aşamasında
ML projeksiyon katmanına girecek
feature setini ve hesaplama/özetleme kurallarını tanımlar.

---

## 1. Kilitli Feature Listesi

Aşağıdaki feature seti V2 için kilitlenmiştir:

- RMS_mean
- RMS_std_slow
- Crest_factor
- Spectral_centroid_mean
- Spectral_rolloff_85_mean
- Spectral_tilt_estimate
- Spectral_flatness_mean
- Low_band_ratio
- Amplitude_modulation_index_slow

---

## 2. Genel Hesaplama İlkeleri

- Girdi ses: 60 sn, mono, 48 kHz
- Feature hesaplamaları kısa pencereler üzerinden yapılır ve tüm klip için özetlenir
- Özetleme hedefi: tek bir sabit boyutlu vektör üretmek
- Normalization ve clipping kuralları `v2_mathematical_lock.md` ile uyumludur

---

## 3. Feature Tanımları ve Özetleme

### 3.1 RMS_mean
Tanım: Klip genelinde ortalama enerji düzeyi.  
Özetleme: kısa pencere RMS değerlerinin ortalaması.  
Amaç: genel loudness/enerji haritalaması.

### 3.2 RMS_std_slow
Tanım: Enerjinin zaman içinde yavaş değişkenliği.  
Özetleme: RMS zaman serisinin (yavaş ölçekli) standart sapması.  
Amaç: “statik” vs “hafif dalgalı” karakter ayrımı.

### 3.3 Crest_factor
Tanım: Tepe enerji / ortalama enerji oranı (dinamik vurgu göstergesi).  
Özetleme: klip düzeyinde tek değer (veya pencere bazlı hesaplanıp ortalama).  
Amaç: “pürüzsüz noise” vs “impulsif yapı” ayrımı.

### 3.4 Spectral_centroid_mean
Tanım: Spektral ağırlık merkezinin ortalaması.  
Özetleme: kısa pencere centroid değerlerinin ortalaması.  
Amaç: parlaklık/açıklık göstergesi.

### 3.5 Spectral_rolloff_85_mean
Tanım: Enerjinin %85’inin altında kaldığı frekansın ortalaması.  
Özetleme: kısa pencere rolloff değerlerinin ortalaması.  
Amaç: yüksek frekans yayılımı ve bant genişliği.

### 3.6 Spectral_tilt_estimate
Tanım: dB/oktav eğim tahmini (broadband spektral renk).  
Özetleme: klip düzeyinde tek tilt (veya pencere bazlı, robust özet).  
Amaç: white/pink/brown benzeri karakter sürekliliği.

### 3.7 Spectral_flatness_mean
Tanım: Tonal yapı vs noise-benzeri doku ölçüsü.  
Özetleme: kısa pencere flatness değerlerinin ortalaması.  
Amaç: “düz noise” ile “tonal/harmonik sızıntı” ayrımı.

### 3.8 Low_band_ratio
Tanım: düşük bant enerjisinin toplam enerjiye oranı (ör. low / full).  
Özetleme: klip düzeyinde oran (robust ortalama).  
Amaç: hum/rumble gibi düşük frekans baskınlığını yakalamak.

### 3.9 Amplitude_modulation_index_slow
Tanım: yavaş genlik modülasyonu göstergesi (algısal “hareket”).  
Özetleme: düşük frekans zarf modülasyonu üzerinden tek değer.  
Amaç: “düz statik” vs “yavaş dalgalı” ambience ayrımı.

---

## 4. Bilinçli Olarak Dışarıda Bırakılanlar

V2 kapsamında aşağıdakiler kullanılmaz:

- transient/event yoğunluk ölçümleri
- onset oranı, zero-crossing yoğunluğu gibi event ağırlıklı metrikler
- embedding tabanlı feature’lar (CLAP vb.)
- zaman serisi modelleme (sequence learning)

Gerekçe:
V2’nin hedefi broadband/steady-state projeksiyonudur ve veri sayısı küçüktür.

---

## 5. Benchmark Notları (C2 için)

Aşağıdaki testler bu feature setinin sağlığını doğrulamak için kullanılır:

- Leave-one-out stabilite: küçük veri altında projeksiyon kararlılığı
- Feature perturbation: küçük değişim → kontrollü parametre değişimi
- Sanity gate çarpma oranı: clamp/limit kullanımının düşük olması beklenir
- Kategori içi kümelenme: aynı kategori örnekleri parametre uzayında yakın olmalıdır

Bu benchmarklar model yarışması değildir; sistem davranışını doğrulamak içindir.


