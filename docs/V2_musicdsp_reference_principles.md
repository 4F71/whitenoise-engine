# UltraGen V2 — musicdsp Referans Prensipleri (Seçilmiş)

Bu doküman, UltraGen V2 aşamasında kullanılan
DSP prensiplerinin kavramsal referansını tanımlar.

Amaç:
Gerçek dünya broadband / steady-state seslerinin
DSP parametre uzayına projeksiyonunda kullanılan
kuralların, akademik ve mühendislik temeline
dayandığını açıkça ortaya koymak.

Kaynak:
musicdsp.org (bdejong/musicdsp)  
Bu prensipler birebir kopyalanmaz; **ilke düzeyinde uyarlanır**.

---

## 1. Gaussian White Noise

**Tanım:**  
Gaussian white noise, zaman alanında örneklerin
normal (Gaussian) dağılımdan geldiği ve
frekans alanında düz (flat) enerji dağılımına
sahip gürültü türüdür.

**V2 Bağlamındaki Rolü:**
- En nötr broadband referans noktasıdır.
- Diğer noise türleri için spektral karşılaştırma temelidir.

**DSP Anlamı:**
- Spektral tilt ≈ 0 dB/oktav
- Yüksek spectral flatness
- Yüksek zero-crossing oranı

---

## 2. Pink Noise (Trammell / Correlated)

**Tanım:**  
Pink noise, frekans başına enerji yoğunluğu
yaklaşık olarak 1/f oranında azalan gürültüdür.
Trammell ve benzeri yöntemler, bu spektral eğimi
korumak için korelasyonlu üretim kullanır.

**V2 Bağlamındaki Rolü:**
- Doğal çevresel seslere en yakın broadband noise tipi
- Room tone, distant ambience gibi kaynakların
algısal temelini oluşturur.

**DSP Anlamı:**
- Spektral tilt ≈ −3 dB/oktav
- White noise’a göre daha “yumuşak” algı
- Orta seviye spectral flatness

---

## 3. One-Pole Low-Pass / High-Pass Filtreler

**Tanım:**  
One-pole filtreler, birinci dereceden,
hesaplama olarak ucuz ve stabil
frekans filtreleridir.

**V2 Bağlamındaki Rolü:**
- Genel parlaklık (LP)
- Düşük frekans temizliği / rumble kontrolü (HP)

**DSP Anlamı:**
- Spektral şekillendirmenin ana araçlarıdır
- Parametre uzayında “karakter” belirler
- Yumuşak eğimli, doğal sonuçlar üretir

---

## 4. DC Filter

**Tanım:**  
DC filter, sinyaldeki sıfır frekans (DC offset)
bileşenlerini bastırmak için kullanılan
özel bir yüksek geçiren filtredir.

**V2 Bağlamındaki Rolü:**
- Gerçek dünya kayıtlarından gelen
offset ve çok düşük frekans artefaktlarını temizler.

**DSP Anlamı:**
- RMS ölçümlerini stabilize eder
- Dinamik analizde yalancı enerji artışını engeller

---

## 5. RMS ve Envelope Follower

**Tanım:**  
RMS (Root Mean Square), sinyalin algısal
enerjisini temsil eden ölçüdür.
Envelope follower, bu enerjinin zaman içindeki
değişimini izler.

**V2 Bağlamındaki Rolü:**
- Genel seviye (RMS_mean)
- Enerji dalgalanması (RMS_std_slow)

**DSP Anlamı:**
- Gain ayarlarının temel girdisi
- LFO depth ve hareket hissi için referans

---

## 6. Crest Factor ve Peak Follower

**Tanım:**  
Crest factor, sinyalin tepe değeri ile
ortalama (RMS) seviyesi arasındaki orandır.
Peak follower, ani tepe davranışlarını izler.

**V2 Bağlamındaki Rolü:**
- Stabilite göstergesi
- Spike / transient varlığı için alarm

**DSP Anlamı:**
- Soft saturation miktarının belirlenmesi
- Uzun dinleme güvenliği (ear fatigue kontrolü)

---

## 7. Spectral Tilt ve Noise Shaping

**Tanım:**  
Spectral tilt, frekans arttıkça enerjinin
nasıl değiştiğini tanımlar.
Noise shaping, bu eğimin kontrollü biçimde
değiştirilmesidir.

**V2 Bağlamındaki Rolü:**
- Noise renginin ana belirleyicisi
- White / pink / brown benzeri davranışların
parametrik ifadesi

**DSP Anlamı:**
- Filtrelerden bağımsız, global spektral karakter
- İnce ayar (fine-tuning) aracı

---

## 8. Soft Saturation

**Tanım:**  
Soft saturation, sinyal tepe değerlerini
sert kırpma (clipping) yerine
yumuşak biçimde bastıran
non-lineer bir işlemdir.

**V2 Bağlamındaki Rolü:**
- Crest factor kontrolü
- Algısal stabilite

**DSP Anlamı:**
- Spike’ları bastırır
- Noise karakterini bozmaz
- Parametre uzayında çok dar aralıkta kullanılır

---

## Kapsam Notları

- Bu prensipler **V2 için yeterlidir**.
- Event, tonal sentez, karmaşık reverb yapıları kapsam dışıdır.
- Prensipler **kavramsal referans** olarak kullanılır,
  birebir implementasyon hedeflenmez.

Bu doküman, UltraGen V2’nin
DSP ve ML tasarım kararlarının
dayandığı teknik zemini tanımlar.
