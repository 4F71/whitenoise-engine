# UltraGen

**UltraGen**, uzun süreli dinlemeye uygun, deterministik ve akademik olarak doğrulanmış  
**procedural noise & ambience synthesis engine**’dir.

Bu proje, sentetik his oluşturmadan; odaklanma, uyku, meditasyon ve ambiyans
kullanımları için **telifsiz, sınırsız ses üretimi** hedefler.

UltraGen müzik üretmez.  
Tonal yapı veya sample tabanlı içerik içermez.  
V1 sürümü tamamen DSP tabanlıdır.

---

## Projenin Amacı

- Uzun süreli dinlemelerde kulak yorgunluğu oluşturmayan sesler üretmek
- Akademik olarak tanımlı noise türlerini doğru şekilde modellemek
- YouTube, uygulama ve araştırma kullanımları için güvenilir bir ses motoru sunmak
- ML ve ileri seviye sistemler için sağlam bir referans taban oluşturmak

---

## Mimari Genel Bakış (V1)

UltraGen V1 aşağıdaki katmanlardan oluşur:

- **core_dsp/**
  - Noise üretimi (white, pink, brown, blue, violet)
  - Filtreleme, LFO, FX ve render pipeline
- **preset_system/**
  - Preset şeması
  - Manuel preset kütüphanesi
- **ui_app/**
  - CLI ve Streamlit tabanlı kontrol arayüzleri
- **Adapter katmanı**
  - Preset yapılarını DSP render motoruna bağlayan saf orkestrasyon katmanı

Tüm sistem:
- Stateless
- Deterministik
- Uzun süreli render’a uygundur

---

## Akademik Doğrulama

UltraGen V1 preset’leri aşağıdaki kriterlerle doğrulanmıştır:

- Spektral eğim doğruluğu (FFT)
- RMS stabilitesi (zaman içinde)
- DC offset kontrolü
- Uzun süreli kulak testi

Detaylı rapor için:
- [`docs/V1_ACADEMIC_VALIDATION.md`](docs/V1_ACADEMIC_VALIDATION.md)

---

## Preset Kataloğu

V1 sürümü; focus, sleep, meditation, nature ve ambient kategorilerinde
akademik olarak doğrulanmış preset’ler içerir.

Detaylı açıklamalar için:
- [`docs/V1_PRESET_CATALOG.md`](docs/V1_PRESET_CATALOG.md)

---

## Sürüm Durumu

- **V1.0** → Tamamlandı ve kilitlendi  
- **V2** → ML tabanlı timbre modelleme (tasarım aşamasında)  
- **V3** → Modüler graph engine (planlama)  
- **V4** → AI destekli generative audio system (vizyon)

Detaylı yol haritası:
- [`docs/ROADMAP.md`](docs/ROADMAP.md)

---

## Kapsam Dışı

UltraGen V1:

- Müzik üretmez
- Melodik veya tonal yapı üretmez
- Gerçek doğa sample’ları kullanmaz
- Gerçek zamanlı (real-time) synthesis yapmaz
- ML tabanlı karar mekanizması içermez

---

## Lisans & Kullanım

Bu proje, araştırma, içerik üretimi ve ürün geliştirme amaçlı tasarlanmıştır.  
Üretilen sesler telifsizdir.

---

## Not

UltraGen, hızlı sonuç üretmek yerine **doğru temel** kurmayı hedefler.  
Bu nedenle geliştirme süreci bilinçli olarak aşamalıdır.
