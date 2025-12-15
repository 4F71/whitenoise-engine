# UltraGen V2 — Feature Set (Kilitli)

Bu doküman, UltraGen V2 aşamasında kullanılan
feature (özellik) kümesini tanımlar.

Amaç:
Gerçek dünya broadband / steady-state seslerinin,
DSP parametre uzayına açıklanabilir ve stabil biçimde
projeksiyonunu mümkün kılmak.

---

## 1. Global Enerji ve Dinamik Feature’lar

1) **RMS_mean**  
Genel enerji seviyesi.

2) **RMS_std_slow**  
Enerji dalgalanmasının yavaş zaman varyansı.

3) **Crest_factor**  
Tepe / ortalama oranı; spike ve stabilite göstergesi.

---

## 2. Global Spektral Feature’lar

4) **Spectral_centroid_mean**  
Genel parlaklık merkezi.

5) **Spectral_centroid_std_slow**  
Parlaklığın zaman içindeki yavaş dalgalanması.

6) **Spectral_rolloff_85_mean**  
Enerjinin %85’inin altında kaldığı frekans.

7) **Spectral_flatness_mean**  
Noise-benzerlik / tonalite göstergesi.

8) **Spectral_tilt_estimate**  
Uzun vadeli spektral eğim (noise rengi).

---

## 3. Bant ve Doku Feature’ları

9) **Zero_crossing_rate_mean**  
Yüksek frekans içeriği için kaba gösterge.

10) **Low_band_ratio**  
Düşük frekans enerji oranı (örn. 20–200 Hz).

11) **Mid_band_ratio**  
Orta bant enerji oranı (örn. 200–2000 Hz).

---

## 4. Yavaş Modülasyon Göstergesi

12) **Amplitude_modulation_index_slow**  
Çok düşük frekanslı genlik dalgalanması göstergesi.

---

## Notlar

- Frame-level feature yoktur.
- Feature’ların tamamı global veya yavaş zaman varyanslıdır.
- Bu set, V2 için **kilitlidir**.
- Event, transient ve tonal modelleme kapsam dışıdır.
