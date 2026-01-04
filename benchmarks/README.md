# scripts/

Render pipeline analiz ve utility script'leri.

## profile_render.py

**Amaç:** Render pipeline'ındaki bottleneck fonksiyonları tespit et.

**Kullanım:**
```bash
python scripts/profile_render.py
```

**Ne Yapar:**
1. 1 V2 preset yükler (distant_wind P0)
2. 60 saniye (1 dakika) render eder
3. cProfile ile tüm fonksiyon çağrılarını kaydeder
4. En yavaş 25 fonksiyonu raporlar

**Rapor Kolonları:**
- `ncalls`: Fonksiyon kaç kez çağrıldı
- `tottime`: Bu fonksiyonun kendi içinde geçirdiği süre
- `percall`: Her çağrı için ortalama süre (tottime/ncalls)
- `cumtime`: Bu fonksiyon + alt fonksiyonlar toplam süre (ÖNEMLİ!)
- `percall`: Her çağrı için ortalama toplam süre
- `filename:lineno(function)`: Fonksiyon adı ve konumu

**Örnek Çıktı:**
```
TOP 25 FONKSIYON (Cumulative Time - Toplam Süre)
======================================================================
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.001    0.001   45.230   45.230 dsp_render.py:75(render_sound)
        3    0.005    0.002   30.450   10.150 dsp_noise.py:78(generate_pink_noise)
        3   18.234    6.078   18.234    6.078 dsp_noise.py:26(_dc_block)
        3    0.002    0.001   10.567    3.522 dsp_filters.py:48(one_pole_lowpass)
        3    8.903    2.968    8.903    2.968 dsp_filters.py:68(one_pole_lowpass loop)
```

**Bottleneck Tespiti:**
- `cumtime` kolonuna bakın (toplam süre)
- core_dsp/ modüllerindeki fonksiyonları not edin
- Loop içeren fonksiyonlar genelde yavaştır → Numba JIT kandidatı
- numpy vektörel işlemler hızlıdır → optimize etmeyin

**Optimizasyon Stratejisi:**
1. En yüksek `cumtime` değerine sahip 3-5 fonksiyonu bulun
2. Bu fonksiyonlar loop içeriyor mu? → Numba JIT ekleyin
3. numpy vektörel mi? → Zaten hızlı, dokunmayın
4. Test edin: Numba ekledikten sonra tekrar profile edin

