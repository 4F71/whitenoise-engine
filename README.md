# xxxDSP

**Yapay zeka destekli prosedürel ambient ses motoru**  
Gerçek dünya seslerini telifsiz ambient soundscape'lere dönüştürür.

---

##  Mevcut Durum

-  **V1**: Core DSP Engine (5 noise tipi, filtre, LFO, FX)
-  **V2**: ML Parametre Projeksiyonu (R²=0.88, 102 preset üretildi)
-  **V3**: Graph/Node Mimarisi (geliştiriliyor)

**Aşama:** Araştırma & Geliştirme  
**Hedef:** V5 Real-time VST Plugin | V6 Desktop Uygulaması

---

##  Teknoloji

- **DSP:** Python, NumPy, SciPy
- **ML:** scikit-learn (Ridge Regression)
- **Audio:** 48kHz prosedürel sentez
- **Pipeline:** Feature extraction → ML projeksiyonu → Preset üretimi

---

## Proje Yapısı
```
core_dsp/       # DSP motoru (noise, filtre, LFO, FX)
preset_system/  # Preset şeması & kütüphane
v2_ml/          # ML pipeline (extraction → model → preset)
docs/           # Teknik dokümantasyon
```

---

## İletişim

İş birliği, yatırım veya ticari talepler için:  

**[GitHub Issues](https://github.com/4F71/whitenoise-engine/issues)** | **[mehmetonurt@gmail.com]**

---

**Lisans:** [MIT/Proprietary]  
**Durum:** Aktif Geliştirme | Production kullanımı için hazır değil