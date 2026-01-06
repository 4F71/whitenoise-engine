# WhiteNoise Engine — proje genel bakış

WhiteNoise Engine, modüler ve procedural DSP tabanlı bir
**ambient / noise synthesis engine**’dir.

Projenin temel amacı:

- Telifsiz
- Deterministik
- Akademik olarak gerekçelendirilebilir
- Uzun vadede genişletilebilir

bir ses üretim altyapısı oluşturmaktır.

WhiteNoise Engine, “tek seferlik ses üretimi” yerine
**sistematik preset üretimi ve kontrol edilebilir varyasyon**
üzerine kuruludur.

---

## Temel Tasarım İlkeleri

- **DSP önce gelir**
- ML, yalnızca **parametre tahmini** yapar
- Ses üretimi her zaman procedural’dır
- Rastgelelik sınırlıdır ve deterministiktir
- Tüm sınırlar matematiksel olarak tanımlıdır

---

## Versiyon Felsefesi

WhiteNoise Engine, versiyonlar arasında net sorumluluklar ayırır.

### V1 — Core DSP Framework
- Noise üretimi
- Filtreler
- FX
- LFO
- Preset sistemi

> V1 tamamlanmış ve kilitlenmiştir.

---

### V2 — Parametre Projeksiyonu (ML Destekli)
- Gerçek dünya seslerinden feature çıkarımı
- Feature → DSP parametre uzayı projeksiyonu
- Preset family üretimi
- **ML ses üretmez**, yalnızca parametre tahmin eder

> V2’nin odağı: açıklanabilirlik ve kontrol.

---

### V3 ve Sonrası (Özet)
- V3: Graph / node tabanlı mimari
- V4: AI evaluator + generative preset seçimi
- V5+: Gerçek zamanlı motor, ürünleşme

Detaylar için: `docs/00_overview/roadmap.md`

---

## Doküman Yapısı

Bu repository’deki dokümantasyon şu şekilde organize edilmiştir:

- `00_overview/`  
  Proje geneli, yol haritası

- `01_v1_core/`  
  V1 DSP mimarisi ve doğrulamalar

- `02_v2_theory/`  
  V2 matematiksel, teorik ve kural tabanlı tanımlar

- `03_v2_ml/`  
  V2 ML tasarımı ve deneyler (C aşaması)

---

## Proje Durumu

Şu anki durum:

- V1: Tamamlandı
- V2 (teori): Büyük ölçüde tamamlandı
- V2 (ML): Aktif geliştirme
- V3+: ⏸Planlama aşamasında

WhiteNoise Engine, hızlı sonuç üretmekten çok
**doğru temeller üzerine inşa edilmeyi**
önceliklendiren bir projedir.
