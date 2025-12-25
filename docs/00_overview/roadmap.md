# UltraGen Yol Haritası

Bu doküman, UltraGen projesinin teknik ve ürün odaklı gelişim planını özetler.

Amaç:
- Mimari karmaşaya girmeden
- Her sürümü sağlam temeller üzerine kurarak
- Uzun vadeli ve sürdürülebilir bir sistem geliştirmektir.

---

##  V1 — Core DSP Engine (TAMAMLANDI)

**Durum:**  Kilitlendi

### Kapsam
- Procedural noise synthesis
- White / Pink / Brown / Blue / Violet noise
- Filtreleme, LFO, FX ve render pipeline
- Preset tabanlı yapı
- CLI ve Streamlit UI
- Akademik doğrulama

### Hedef
- Uzun süreli dinlemeye uygun ses motoru
- YouTube ve uygulama kullanımına hazır altyapı
- ML için temiz referans veri seti

---

## v2 — ml parameter projection engine (mevcut odak)

**amaç:**  
Gerçek dünya broadband / steady-state seslerinden,
telifsiz procedural preset’ler üretmek.

### yapılacaklar
- [x] v2 dataset hazırlanması (60 sn, mono, 48 kHz)
- [x] feature setinin kilitlenmesi
- [x] dsp parametre uzayının tanımlanması
- [x] kural tabanlı başlangıç eşlemesi
- [x] matematiksel kilit dokümanı
- [x] core_dsp uyum denetimi
- [x] preset_to_dsp_adapter denetimi
- [x] preset family mantığının tanımlanması
- [x] varyant üretim kuralları (mean + spread)
- [x] ilk regression tabanlı ml modeli
- [x] 17 referans ses → preset aileleri
- [x] v2 çıktılarının içerik üretiminde kullanılması

### kapsam dışı
- deep learning
- event / transient synthesis
- node tabanlı patch sistemi

---

## v3 — graph / node based synthesis engine

**amaç:**  
Pipeline mimarisinden çıkıp,
node (dag) tabanlı bir sentez yapısına geçmek.

### yapılacaklar
- [ ] node tanımı (noise, filter, lfo, gain, pan, fx)
- [ ] json tabanlı patch formatı
- [ ] topological execution engine
- [ ] adapter → node graph dönüşümü
- [ ] bird ambience, simple tonal pad node’ları
- [ ] preset → patch migrasyonu

### kapsam dışı
- ml tabanlı node üretimi
- gerçek zamanlı engine

---

## v4 — ai generative & evaluation system

**amaç:**  
Üretilen preset / patch’lerin
kalitesini otomatik değerlendiren ve
iyileştiren bir ai katmanı eklemek.

### yapılacaklar
- [ ] embedding / evaluator modeli (clap benzeri)
- [ ] text → audio quality scoring
- [ ] genetic algorithm ile preset evrimi
- [ ] auto-mastering kuralları
- [ ] “iyi değilse üretme” quality gate
- [ ] batch üretim otomasyonu

### kapsam dışı
- gerçek zamanlı inference
- doğrudan ses üreten dl modeller

---

## v5 — real-time engine (rust / c++)

**amaç:**  
UltraGen çekirdeğini gerçek zamanlı
çalışabilir hale getirmek.

### yapılacaklar
- [ ] rust / c++ core engine
- [ ] real-time safe dsp
- [ ] v3 graph engine portu
- [ ] vst3 / clap plugin prototipi
- [ ] düşük gecikme optimizasyonları

### kapsam dışı
- ui önceliği
- mobil destek

---

## v6 — desktop studio application

**amaç:**  
UltraGen’i bir masaüstü stüdyo uygulaması
haline getirmek.

### yapılacaklar
- [ ] cross-platform desktop ui
- [ ] node graph editor
- [ ] preset / patch browser
- [ ] offline render pipeline
- [ ] export / batch render
- [ ] kullanıcı preset yönetimi

---

## v7 — cloud api & automation

**amaç:**  
UltraGen’i bir servis olarak sunmak.

### yapılacaklar
- [ ] rest / graphql api
- [ ] cloud render service
- [ ] preset generation endpoints
- [ ] batch job scheduling
- [ ] subscription / quota sistemi
- [ ] içerik üreticiler için otomasyon

---

## v8 — mobile & consumer layer

**amaç:**  
UltraGen teknolojisini
tüketici seviyesine indirmek.

### yapılacaklar
- [ ] mobile sdk (ios / android)
- [ ] basitleştirilmiş preset kontrolü
- [ ] offline / low-power mode
- [ ] sleep / focus / meditation ürünleri
- [ ] consumer ui / ux

---

## genel notlar

- her sürüm bir öncekini **bozmaz**
- matematiksel kilitler geriye dönük korunur
- ml ve dl katmanları **dsp’nin yerini almaz**
- core_dsp prensipleri uzun vadede sabit kalır

Bu yol haritası,
UltraGen projesinin teknik ve ürün
evriminde referans alınacak ana dokümandır.
