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

##  V2 — ML Timbre Engine (TASARIM AŞAMASI)

**Durum:** Planlama

### Amaç
- Bir ses örneğinden (rain, ambience, noise) spektral özellikler çıkarmak
- Bu özellikleri DSP parametrelerine haritalamak
- Telifsiz ama benzer karakterde yeni sesler üretmek

### Özellikler
- Feature extraction (spectral centroid, band energy, vb.)
- Parametre tahmin eden basit regresyon / MLP modelleri
- DSP motoru değişmeden kalır

> V2, V1 preset’lerini **öğretici referans** olarak kullanır.

---

##  V3 — Graph Engine (Modüler Ses Mimarisi)

**Durum:**  Konsept

### Amaç
- Sabit pipeline’dan çıkmak
- Node-based, DAG tabanlı ses mimarisi kurmak

### Özellikler
- NoiseNode, FilterNode, LfoNode, MixNode, OutputNode
- JSON patch dosyaları ile routing
- Preset’lerin “ses makinesi” haline gelmesi

---

##  V4 — AI Generative Audio System

**Durum:** Vizyon

### Amaç
- Metin → ses hedefleme
- Otomatik preset evrimi
- Kendi çıktısını değerlendiren sistem

### Bileşenler
- Text-audio embedding (CLAP benzeri)
- Genetik algoritma ile preset optimizasyonu
- Otomatik mastering ve kalite kontrol

---

## Uzun Vadeli Genişleme

- Real-time engine (C++ / Rust)
- VST / CLAP plugin
- Desktop studio app
- Cloud API
- Mobile meditation & sleep app

---

## Geliştirme İlkeleri

- Önce doğruluk
- Önce stabilite
- Önce mimari netlik
- Sonra ölçekleme

UltraGen hızlı büyümek için değil,  
**doğru büyümek için** tasarlanmıştır.
