# WhiteNoise Engine Yol Haritası

Bu doküman, WhiteNoise Engine projesinin teknik ve ürün odaklı gelişim planını özetler.

Amaç:
- Mimari karmaşaya girmeden
- Her sürümü sağlam temeller üzerine kurarak
- Uzun vadeli ve sürdürülebilir bir sistem geliştirmektir.

---

##  Hızlı Durum Özeti

| Versiyon | Durum | Tamamlanma | Çıktı |
|----------|-------|------------|-------|
| V1 Core DSP |  Locked | Aralık 2025 | 14 preset, 5 noise, Core DSP |
| V2 ML Projection |  Locked | Aralık 2025 | 102 preset (P0-P5), Ridge R²=0.88 |
| V2.5 Binaural |  Production | Ocak 2026 | 6 Solfeggio, Stereo beats |
| V2.6 Organic |  Production | Ocak 2026 | P6 profile, Ultra-slow LFO |
| Performans |  Production | Ocak 2026 | Multiprocessing, scipy optimize |
| V3 Graph/Node |  Planned | 2026 Q1 | DAG synthesis |
| V4 AI Eval |  Planned | 2026 Q2 | CLAP, GA optimizer |
| V5 Real-time |  Planned | 2026 Q3+ | Rust/C++, VST plugin |
| V6-V8 | Planned | 2026+ | Desktop, Cloud, Mobile |

**Şu Anki Durum (2026-01-18):**
-  102 ML preset production-ready
-  6 Solfeggio binaural preset
-  3 P6 organic test preset
-  Hedef: YouTube 10 saatlik ambient içerik

---

##  V1 — Core DSP Engine (TAMAMLANDI )

**Durum:** Kilitlendi  
**Tamamlanma:** Aralık 2025

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

## V2 — ML Parameter Projection Engine (TAMAMLANDI)

**Durum:** Kilitlendi  
**Tamamlanma:** Aralık 2025

**Amaç:**  
Gerçek dünya broadband / steady-state seslerinden,
telifsiz procedural preset'ler üretmek.

### Research & Theory
- [x] Feature extraction analysis (11 acoustic features)
- [x] DSP parameter space mapping (12 parameters)
- [x] Robust normalization (median + MAD, tanh scaling)
- [x] Ridge regression model (R²=0.88, LOO validation)
- [x] Mathematical lock documentation (feature → param projection)
- [x] Sanity gate rules (gain budget, filter constraints)

### Theory Documentation (Locked)
- v2_feature_set.md (11 features: RMS, spectral, modulation)
- v2_param_set.md (12 DSP params: noise type, filter, LFO, gain)
- v2_mathematical_lock.md (normalization, projection formulas)
- v2_rule_based_mapping.md (category logic: A-E)
- v2_preset_family.md (P0-P5 profiles: mean + spread)
- binaural_beats_theory.md (stereo carrier + beat frequency)

### Yapılacaklar
- [x] v2 dataset hazırlanması (60 sn, mono, 48 kHz)
- [x] feature setinin kilitlenmesi (11 audio feature)
- [x] dsp parametre uzayının tanımlanması (12 parameter)
- [x] kural tabanlı başlangıç eşlemesi
- [x] matematiksel kilit dokümanı
- [x] core_dsp uyum denetimi
- [x] preset_to_dsp_adapter denetimi
- [x] preset family mantığının tanımlanması (P0-P5)
- [x] varyant üretim kuralları (mean + spread)
- [x] ridge regression modeli (R²=0.88 LOO validation)
- [x] 17 referans ses → 102 preset aileleri (17×6)
- [x] C2 benchmark suite (3/4 test başarılı)
- [x] v2 çıktılarının içerik üretiminde kullanılması

### Teslimatlar
- 19 unique categories (distant_wind, soft_pink, etc.)
- 102 ML-generated preset (P0-P5 profiles)
- Ridge model (ridge_baseline.pkl)
- 6 kilitli doküman (v2_theory/)
- 8 sıralı script pipeline

### Kapsam Dışı
- Deep learning ile ses üretimi
- Event / transient synthesis
- Node tabanlı patch sistemi (V3'e kadar)


---

## V2.5 — Binaural Beats (TAMAMLANDI)

**Durum:** Production-Ready  
**Tamamlanma:** Ocak 2026

**Amaç:**  
YouTube meditation/focus content için binaural beats desteği.

### Research & Theory
- [x] Binaural beats psychoacoustic theory (Oster 1973)
- [x] Solfeggio frequency research (432-963 Hz)
- [x] Stereo carrier + beat frequency DSP
- [x] Theory documentation (binaural_beats_theory.md)

### Yapılacaklar
- [x] Stereo binaural beats DSP motoru (dsp_binaural.py)
- [x] Carrier: 100-1200 Hz, Beat: 0.5-50 Hz
- [x] Chunked generation (phase glitch önleme)
- [x] Fade in/out (click önleme)
- [x] 6 Solfeggio preset (432-963 Hz)
- [x] Streamlit V2.5 Pure Binaural Beats modu
- [x] Optional pink noise background
- [x] Peak normalization

### Teslimatlar
- core_dsp/dsp_binaural.py
- 6 Solfeggio JSON preset (v2_presets/binaural/)
- V2.5 kategori (Theta, Alpha, Gamma, Delta, Custom)

---

## V2.6 — Organic Preset Profile (TAMAMLANDI)

**Durum:** Production-Ready  
**Tamamlanma:** Ocak 2026

**Amaç:**  
YouTube ambient için monotonluk giderme ve 10 saat dinlenebilir preset'ler.

### Research & Theory
- [x] Problem tespiti (pure noise monotonluk sorunu)
- [x] YouTube kanal analizi (breathing efekti gereksinimi)
- [x] Ultra-slow LFO theory (0.005-0.01 Hz, 100-200s cycle)
- [x] Multi-layer stacking theory (brown + pink + white)

### Yapılacaklar
- [x] Ultra-slow LFO implementasyonu (0.005-0.01 Hz)
- [x] Multi-layer stacking (brown + pink + white)
- [x] Filter breathing (LFO on cutoff)
- [x] P6 Organic generator script
- [x] 3 dakika test render (3 preset)
- [x] Kulaklık validasyonu

### Teslimatlar
- preset_system/generate_p6_organic.py
- test_organic_presets.py
- 3 test preset (16.5 MB each)
- LFO min rate 0.005 Hz (200s breathing cycle)

### Test Sonuçları
- Layer yapısı: Brown (250Hz) + Pink (~2000Hz) + White (2500Hz)
- LFO cycle: 100-200 saniye
- Peak: 0.211-0.219 (clipping yok)
- Breathing efekti: Doğrulandı

---

## V2.7 — Organic Texture Enhancement (TAMAMLANDI)

**Durum:** Production-Ready  
**Tamamlanma:** Ocak 2026

**Amaç:**  
Sub-bass rumble + air presence ile YouTube-grade ambient texture.

### Research & Theory
- [x] Perlin noise modulation (irregular breathing)
- [x] Sub-bass layer theory (20-80 Hz, subtle rumble)
- [x] Air layer theory (4-8 kHz, presence)
- [x] Binaural carrier breathing
- [x] Theory documentation (organic_texture_theory.md)

### Yapılacaklar
- [x] Perlin-modulated LFO (irregular breathing)
- [x] Sub-bass layer (20-80 Hz, -12 dB)
- [x] Air layer (4-8 kHz, -18 dB)
- [x] Binaural carrier breathing integration

### Teslimatlar
- Enhanced organic texture DSP
- Sub-bass + air layers
- Perlin LFO implementation
- Updated organic_texture_theory.md

---

---

## V3 — Graph / Node Based Synthesis Engine (PLANLI)

**Durum:**  Theory Complete  
**Başlangıç:** Ocak 2026

**Amaç:**
Pipeline mimarisinden çıkıp,
node (dag) tabanlı bir sentez yapısına geçmek.

### Research & Theory
- [x] Graph theory research (Kahn 1962, Web Audio API, VCV Rack)
- [x] Node-based architecture analysis
- [x] Topological sort algorithm (Kahn 1962)
- [x] Pull vs Push execution models
- [x] Patch format analysis (VCV Rack, Max/MSP)

### Theory Documentation (Locked)
- v3_research_notes.md (Web Audio API, VCV Rack, Kahn 1962 summary)
- v3_node_definition.md (node types, interface, ports, parameters)
- v3_graph_theory.md (DAG, cycle detection, execution order)
- v3_patch_format.md (JSON schema, validation, serialization)
- v3_implementation.md (architecture plan, class hierarchy, testing)

### Yapılacaklar
- [ ] Core engine (Graph, topological executor)
- [ ] Node tanımı (noise, filter, lfo, gain, pan, fx)
- [ ] JSON tabanlı patch formatı
- [ ] Topological execution engine
- [ ] Adapter → node graph dönüşümü
- [ ] Bird ambience, simple tonal pad node'ları
- [ ] Preset → patch migrasyonu
- [ ] Gradio UI integration

### Kapsam Dışı
- ML tabanlı node üretimi (V4'e kadar)
- Gerçek zamanlı engine (V5'e kadar)

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
WhiteNoise Engine çekirdeğini gerçek zamanlı
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
WhiteNoise Engine’i bir masaüstü stüdyo uygulaması
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
WhiteNoise Engine’i bir servis olarak sunmak.

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
WhiteNoise Engine teknolojisini
tüketici seviyesine indirmek.

### yapılacaklar
- [ ] mobile sdk (ios / android)
- [ ] basitleştirilmiş preset kontrolü
- [ ] offline / low-power mode
- [ ] sleep / focus / meditation ürünleri
- [ ] consumer ui / ux

---

## Genel Notlar

- Her sürüm bir öncekini **bozmaz**
- Matematiksel kilitler geriye dönük korunur
- ML ve DL katmanları **DSP'nin yerini almaz**
- core_dsp prensipleri uzun vadede sabit kalır

---

##  Başarılar ve Metrikler (2026-01-18)

### Preset Koleksiyonu
-  V1: 14 el yapımı preset
-  V2: 102 ML-generated preset (17 karakter × 6 profil)
-  V2.5: 6 Solfeggio binaural preset
-  V2.6: 3 organic test preset
- **Toplam: 125 preset** 

### ML Pipeline Başarısı
- Ridge Regression: R²=0.88 (LOO validation)
- Feature extraction: 11 audio özelliği
- Parameter projection: 12 DSP parametresi
- C2 Benchmark: 3/4 test başarılı (%75)

### Performans
- Multiprocessing: 2x speedup (600s render)
- Brown noise: %70 optimize edildi
- Filter processing: scipy.signal.lfilter vectorized

### Teknik Altyapı
- 8 sıralı ML script pipeline
- 6 kilitli doküman (v2_theory/)
- Binaural beats DSP motoru (stereo)
- Ultra-slow LFO (0.005-0.01 Hz breathing)

### YouTube Hazırlık Durumu
-  10 saatlik ambient render kapasitesi
-  Monotonluk sorunu çözüldü (P6 organic)
-  Breathing efekti implementasyonu
-  Multi-layer depth (brown + pink + white)
-  Hedef: 50-100 YouTube videosu

---

Bu yol haritası,
WhiteNoise Engine projesinin teknik ve ürün
evriminde referans alınacak ana dokümandır.

**Son Güncelleme:** 2026-01-18  
**Mevcut Versiyon:** V2.6 (Production-Ready)
