# ultragen v2 — c2 benchmark tasarımı

Status: ACTIVE (C2)  
Scope: Feature → parametre projeksiyonunun davranışsal doğrulaması  
Out of scope: Model karşılaştırma yarışı, accuracy/score optimizasyonu

Bu doküman, UltraGen V2’de kullanılan ML projeksiyonunun
doğruluğunu değil, davranışsal tutarlılığını ölçmeyi amaçlar.

Amaç:
- Küçük veri altında stabilite
- Fiziksel sezgiye uygunluk
- Sanity gate bağımlılığının düşük olması

---

## 1. Benchmark Felsefesi

C2 benchmark’ı bir model yarışması değildir.

Ölçülenler:
- Çıkışların tutarlı olup olmadığı
- Küçük giriş değişimlerine orantılı tepki
- Sanity gate’in olağanüstü durumlar dışında devreye girip girmemesi

Ölçülmeyenler:
- Accuracy / R² / F1
- Cross-validation skorları
- Büyük veri genellemesi

---

## 2. Benchmark 1 — Leave-One-Out Stabilite Testi

### Amaç
Modelin ezber mi yaptığını,
yoksa feature → parametre haritalamasını mı öğrendiğini test etmek.

### Yöntem
- Dataset’ten 1 ses çıkarılır
- Kalan N−1 ses ile model eğitilir
- Çıkarılan ses için parametre tahmini yapılır

Bu işlem tüm örnekler için tekrarlanır.

### Ölçümler
- Tahmin edilen parametre vektörü ile
  tüm-dataset eğitimi sonucu arasındaki fark
- Parametre bazında mutlak sapma (L1)

### Kabul Kriteri
- Parametrelerin büyük çoğunluğu
  tanımlı spread sınırları içinde kalmalı
- Sistematik uç değerler oluşmamalı

---

## 3. Benchmark 2 — Feature Perturbation Testi

### Amaç
Modelin fiziksel sezgiye uygun davranıp davranmadığını ölçmek.

### Yöntem
Tek bir referans ses için:
- Feature vektörü kontrollü şekilde bozulur

Örnek perturbation’lar:
- rms_mean +%10
- spectral_centroid_mean küçük artış
- spectral_tilt_estimate −1 dB/oktav
- amplitude_modulation_index_slow küçük artış

### Beklenen Davranış
- İlgili parametreler orantılı değişmeli
- Alakasız parametrelerde sıçrama olmamalı

### Kabul Kriteri
- Parametre değişimleri monoton ve sınırlı
- Sanity gate’e çarpma nadir

---

## 4. Benchmark 3 — Sanity Gate Çarpma Oranı

### Amaç
ML çıktılarının doğal olarak güvenli aralıkta kalıp kalmadığını ölçmek.

### Yöntem
- Tüm inference çıktıları izlenir
- Sanity gate tarafından clamp edilen parametreler sayılır

### Ölçümler
- Parametre başına clamp oranı
- Toplam clamp oranı

### Kabul Kriteri
- Clamp istisna olmalı, kural değil
- Sürekli clamp eden parametreler varsa:
  feature seti veya normalization gözden geçirilir

---

## 5. Benchmark 4 — Kategori İçi Tutarlılık

### Amaç
Benzer seslerin benzer parametre uzayına düşüp düşmediğini görmek.

### Yöntem
- Aynı kategoriye ait sesler seçilir
- Parametre vektörleri karşılaştırılır

### Ölçümler
- Parametre uzayında mesafe (L2 veya cosine)
- Görsel inceleme (PCA / 2D projeksiyon)

### Kabul Kriteri
- Aynı kategori örnekleri birbirine yakın konumlanmalı
- Kategoriler arası ayrım korunmalı

---

## 6. Başarısızlık Durumları ve Tepki

- LOO testinde büyük sapmalar:
  regularization artırılır

- Perturbation testinde sıçramalar:
  ilgili feature yeniden ölçeklenir veya çıkarılır

- Sanity gate sürekli çalışıyorsa:
  sorun ML’de değil, feature veya normalization’dadır

---

## 7. C2 Başarı Tanımı

C2 başarılıdır eğer:

- Tüm benchmark’lar kabul kriterlerini sağlıyorsa
- Üretilen P0 preset’ler:
  - clip etmiyorsa
  - aşırı parlak veya aşırı koyu değilse
  - kategori karakterini koruyorsa

Bu noktada:
- Feature seti kilitli kalır
- Model davranışı anlaşılmış kabul edilir
- Preset üretimine güvenle geçilir

---

## 8. Sonraki Adım

C2 benchmark tamamlandıktan sonra:

- C2 kodu stabilize edilir
- P0 preset’ler üretilir
- Preset family kuralları uygulanır

Bir sonraki aşama:
C2 implementasyonu (features → model → inference)
