#  xxxDSP

**Yapay zeka destekli prosedürel ambient ses motoru**  
Gerçek dünya seslerini telifsiz ambient soundscape'lere dönüştürür.

---

##  Proje Hedefleri

**Kısa Vade (2025 Q1-Q2):**
- 300+ telifsiz preset üretimi (ocean, rain, forest, café kategorileri)
- YouTube kanal lansmanı (ambient/meditation içerik, 50+ video)
- Binaural beats entegrasyonu (focus, meditation, sleep kategorileri)

**Orta Vade (2025 Q3-Q4):**
- V3 Graph/Node mimarisi (karmaşık soundscape'ler, bird chirps, tonal pads)
- Real-time VST plugin (müzisyen pazarı için commercial release)
- Desktop uygulaması MVP (cross-platform ambient synthesizer)

**Uzun Vade (2026+):**
- Cloud API servisi (B2B ambient audio generation)
- Mobil uygulama (iOS/Android, subscription model)
- AI-powered preset generation (prompt-based synthesis)

---

##  Mevcut Durum

###  V1: Core DSP Engine (Aralık 2025)
- 5 noise tipi (white, pink, brown, blue, violet)
- Filter + LFO + FX pipeline
- 14 el yapımı preset library
- CLI + Streamlit UI
- Akademik doğrulama tamamlandı (MusicDSP prensipleri)

###  V2: ML Parametre Projeksiyonu (Aralık 2025)
- Ridge model: R²=0.88 (LOO validation)
- 17 kaynak ses → 102 ML-generated preset (P0-P5 profiller)
- Feature extraction: 11 audio özelliği (RMS, spectral, LFO)
- C2 benchmark: 3/4 test başarılı
- Render test: 9/9 başarılı, profil karakterleri doğrulandı
- **Production-ready**

---

##  Roadmap

**Tamamlanan:**
- **V1:** Core DSP Engine (5 noise type, filters, LFO, FX)
- **V2:** ML Parameter Projection (R²=0.88, 102 preset)

**Geliştirme Aşamasında:**
-  **V3:** Graph/Node Architecture (karmaşık soundscape sentezi)

**Planlanan:**
-  **V4:** AI Evaluation System (otomatik kalite değerlendirme)
-  **V5:** Real-time Engine (VST Plugin, Rust/C++)
-  **V6:** Desktop Application (cross-platform synthesizer)
-  **V7:** Cloud API (batch render hizmeti)
-  **V8:** Mobile Apps (iOS/Android, offline generator)

**Ticari Hedefler:**
- YouTube içerik üretimi ve monetization
- VST plugin commercial release (müzisyen pazarı)
- Mobil uygulama subscription modeli
- B2B cloud API servisi

---

##  Teknoloji

- **DSP:** Python, NumPy, SciPy (48kHz prosedürel sentez)
- **ML:** scikit-learn (Ridge Regression, LOO validation)
- **Audio:** Prosedürel noise generation (white/pink/brown/blue/violet)
- **Pipeline:** Feature extraction → ML projection → Preset generation
- **UI:** Streamlit (preset browser, real-time render)

---

##  Proje Yapısı

```
whitenoise-engine/
│
├── core_dsp/                   # V1 DSP Primitives (Production-Ready)
│   ├── dsp_noise.py           # 5 noise generator (white/pink/brown/blue/violet)
│   ├── dsp_filters.py         # One-pole LP/HP filters (20-16000 Hz)
│   ├── dsp_lfo.py             # Sine LFO modulation (amplitude/pan/filter)
│   ├── dsp_fx.py              # FX chain (saturation, stereo width, reverb)
│   ├── dsp_render.py          # Layer mixing & render pipeline
│   ├── dsp_filters_v2.py      # V2 extensions (future use)
│   ├── dsp_lfo_v2.py          # V2 extensions (future use)
│   └── dsp_fx_v2.py           # V2 extensions (future use)
│
├── preset_system/              # Preset Management System
│   ├── preset_schema.py       # PresetConfig dataclass (V1+V2 compatible)
│   ├── preset_library.py      # V1 hardcoded + V2 ML loader functions
│   ├── preset_autogen.py      # V1 variant generator
│   └── presets/               # 14 V1 handcrafted JSON presets
│       ├── deep_focus.json
│       ├── ambient_rain.json
│       ├── rain_texture.json
│       └── ... (11 more)
│
├── v2_ml/                      # V2 ML Pipeline (Production-Ready)
│   ├── scripts/               # 8 sequential scripts (run in order)
│   │   ├── 1_feature_extraction.py      # 17 WAV → features_raw.csv
│   │   ├── 2_normalize_features.py      # z-score, outlier handling
│   │   ├── 3_generate_pseudo_labels.py  # Rule-based feature→param mapping
│   │   ├── 4_train_ridge_model.py       # Ridge model training (R²=0.88)
│   │   ├── 5_generate_p0_presets.py     # 17 baseline presets
│   │   ├── 6_generate_p1_p5_presets.py  # 85 profile variants (P1-P5)
│   │   ├── 7_run_c2_benchmarks.py       # C2 test suite execution
│   │   └── 8_render_test.py             # Render test & spectral analysis
│   │
│   ├── models/                # ML models & encodings (gitignored)
│   │   ├── ridge_baseline.pkl          # Trained Ridge model
│   │   ├── encoding_maps.pkl           # Categorical encodings
│   │   └── model_metrics.json          # Performance metrics
│   │
│   ├── presets/               # 102 ML-generated presets (gitignored)
│   │   ├── p0/  # 17 baseline (ML direct output)
│   │   ├── p1/  # 17 soft variants
│   │   ├── p2/  # 17 dark variants
│   │   ├── p3/  # 17 calm variants
│   │   ├── p4/  # 17 airy variants
│   │   └── p5/  # 17 dense variants
│   │
│   ├── data/                  # Feature CSVs (gitignored)
│   │   ├── features_raw.csv
│   │   ├── features_normalized.csv
│   │   └── pseudo_labels.csv
│   │
│   ├── benchmarks/            # Test reports (gitignored)
│   │   └── c2_report.json
│   │
│   └── render_test/           # Render outputs (gitignored)
│       ├── *.wav              # 9 test WAV files
│       └── render_test_report.json
│
├── v2_dataset/                 # Training audio (gitignored)
│   └── *.wav                  # 17 source sounds (60s, mono, 48kHz)
│
├── ui_app/                     # Streamlit UI
│   ├── ui_streamlit.py        # Main UI (V1+V2 preset browser)
│   └── preset_to_dsp_adapter.py  # Preset → DSP generator bridge
│
├── ai_gen/                     # AI generation experiments (gitignored)
├── benchmarks/                 # Benchmark outputs (gitignored)
├── graph_engine/               # V3 node system (planned, empty)
├── timbre_engine/              # Timbre analysis tools (experimental)
│
├── .gitignore                 # Git ignore rules
├── README.md                  # Bu dosya
├── requirements.txt           # Python dependencies
└── app.py                     # Main application entry point
```



---

##  İletişim


İş birliği, yatırım, ticari talepler veya proje detayları hakkında görüşmek için:

**[mehmetonurt@gmail.com]** | **[https://www.linkedin.com/in/onurtilki/]** | **[GitHub Issues](https://github.com/4F71/whitenoise-engine/issues)**

*Teknik detaylar, demo kayıtları ve performans metrikleri talep üzerine paylaşılabilir.*

---

**Lisans:** MIT  
**Durum:** Aktif Geliştirme | V2 Production-Ready  
**Son Güncelleme:** Aralık 2025