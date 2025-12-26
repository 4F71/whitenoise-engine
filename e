[33mtag v1.0.0[m
Tagger: Yedi Sarman <mehmetonurt@gmail.com>
Date:   Fri Dec 26 17:23:30 2025 +0300

V1.0.0 - Core DSP Engine

Procedural Noise Synthesis:
- 5 noise type: white, pink, brown, blue, violet
- Filter (LP/HP), LFO (sine), FX (saturation, stereo, reverb)
- Render pipeline: layer mixing, normalization, WAV export

Preset System:
- 14 el yapımı preset (deep_focus, rain_texture, etc.)
- JSON schema + validation
- preset_to_dsp_adapter

Deliverables:
- core_dsp/ (8 modül)
- preset_system/ (schema, library)
- CLI + Streamlit UI
- Akademik doğrulama tamamlandı

[33mcommit f9d0650d33e71c7d38c92cdfcf4401e014b6a28a[m[33m ([m[1;33mtag: [m[1;33mv1.0.0[m[33m)[m
Author: Yedi Sarman <mehmetonurt@gmail.com>
Date:   Sat Dec 13 00:40:58 2025 +0300

    V1 CORE DSP ve Preset System tamamlandı
    
    - core_dsp modülleri finalize edildi (noise, filters, fx, lfo, render)
    - V1 mimari sınırları netleştirildi ve kilitlendi
    - Preset System (schema, library, autogen) eklendi
    - v2 denemeleri ayrı dosyalara alındı (gelecek geliştirmeler için)

[1mdiff --git a/.gitignore b/.gitignore[m
[1mindex 97d91f9..c2d15e3 100644[m
[1m--- a/.gitignore[m
[1m+++ b/.gitignore[m
[36m@@ -4,4 +4,5 @@[m [m__pycache__/[m
 output/[m
 logs/[m
 [m
[31m-user/[m
\ No newline at end of file[m
[32m+[m[32muser/[m
[32m+[m[32mreferans/[m
\ No newline at end of file[m
[1mdiff --git a/core_dsp/dsp_filters.py b/core_dsp/dsp_filters.py[m
[1mindex 5092c39..1d6684a 100644[m
[1m--- a/core_dsp/dsp_filters.py[m
[1m+++ b/core_dsp/dsp_filters.py[m
[36m@@ -1,147 +1,141 @@[m
[32m+[m[32mimport math[m
[32m+[m[32mfrom typing import Tuple[m
[32m+[m
 import numpy as np[m
 [m
[32m+[m[32mFloatArray = np.ndarray[m
[32m+[m[32m_FT = np.float32[m
[32m+[m[32m_DENORM_GUARD = _FT(1e-20)[m
 [m
[31m-def lpf_one_pole(x: np.ndarray, cutoff_hz: float, sample_rate: float) -> np.ndarray:[m
[31m-    """Tek kutuplu alçak geçiren filtre uygular.[m
 [m
[31m-    Parametreler:[m
[31m-        x: Giriş sinyali (1B numpy dizisi).[m
[31m-        cutoff_hz: Kesim frekansı (Hz).[m
[31m-        sample_rate: Örnekleme hızı (Hz).[m
[32m+[m[32mdef _to_float32(signal: FloatArray) -> FloatArray:[m
[32m+[m[32m    """Girişi float32 kopyasına dönüştürür."""[m
[32m+[m[32m    if signal.dtype != _FT:[m
[32m+[m[32m        return signal.astype(_FT, copy=False)[m
[32m+[m[32m    return signal[m
[32m+[m
[32m+[m
[32m+[m[32mdef _safe_cutoff(cutoff_hz: float, sample_rate: int) -> float:[m
[32m+[m[32m    """Kesim frekansını Nyquist sınırında güvenli aralığa sıkıştırır."""[m
[32m+[m[32m    nyquist = sample_rate * 0.5[m
[32m+[m[32m    return float(np.clip(cutoff_hz, 1.0, nyquist * 0.95))[m
 [m
[31m-    Döndürür:[m
[31m-        Filtrelenmiş sinyal (numpy dizisi).[m
[32m+[m
[32m+[m[32mdef dc_block(signal: FloatArray, pole: float = 0.995) -> FloatArray:[m
     """[m
[31m-    x = np.asarray(x, dtype=np.float64)[m
[31m-    if x.ndim != 1:[m
[31m-        raise ValueError("x 1 boyutlu olmalıdır")[m
[31m-    if cutoff_hz <= 0 or sample_rate <= 0:[m
[31m-        raise ValueError("cutoff_hz ve sample_rate pozitif olmalıdır")[m
[32m+[m[32m    DC bileşenini bastıran birinci dereceden yüksek geçiren filtre uygular.[m
[32m+[m
[32m+[m[32m    Parametreler:[m
[32m+[m[32m        signal: Mono giriş sinyali (float32).[m
[32m+[m[32m        pole: Kutup katsayısı; 1.0'a yaklaştıkça daha düşük kesim sağlar.[m
 [m
[31m-    alpha = np.exp(-2.0 * np.pi * cutoff_hz / sample_rate)[m
[32m+[m[32m    Dönüş:[m
[32m+[m[32m        DC'si bastırılmış float32 sinyal.[m
[32m+[m[32m    """[m
[32m+[m[32m    x = _to_float32(signal)[m
     y = np.empty_like(x)[m
[31m-    y[0] = (1 - alpha) * x[0][m
[31m-    for n in range(1, x.size):[m
[31m-        y[n] = (1 - alpha) * x[n] + alpha * y[n - 1][m
[32m+[m[32m    pole_f = _FT(pole)[m
[32m+[m[32m    x_prev = _FT(0.0)[m
[32m+[m[32m    y_prev = _FT(0.0)[m
[32m+[m[32m    for i, sample in enumerate(x):[m
[32m+[m[32m        y_curr = sample - x_prev + pole_f * y_prev + _DENORM_GUARD[m
[32m+[m[32m        y[i] = y_curr[m
[32m+[m[32m        x_prev = sample[m
[32m+[m[32m        y_prev = y_curr[m
     return y[m
 [m
 [m
[31m-def hpf_one_pole(x: np.ndarray, cutoff_hz: float, sample_rate: float) -> np.ndarray:[m
[31m-    """Tek kutuplu yüksek geçiren filtre uygular.[m
[32m+[m[32mdef one_pole_lowpass([m
[32m+[m[32m    signal: FloatArray, cutoff_hz: float, sample_rate: int[m
[32m+[m[32m) -> FloatArray:[m
[32m+[m[32m    """[m
[32m+[m[32m    Birinci dereceden düşük geçiren filtre uygular.[m
 [m
     Parametreler:[m
[31m-        x: Giriş sinyali (1B numpy dizisi).[m
[32m+[m[32m        signal: Mono giriş sinyali (float32).[m
         cutoff_hz: Kesim frekansı (Hz).[m
         sample_rate: Örnekleme hızı (Hz).[m
 [m
[31m-    Döndürür:[m
[31m-        Filtrelenmiş sinyal (numpy dizisi).[m
[32m+[m[32m    Dönüş:[m
[32m+[m[32m        Low-pass uygulanmış float32 sinyal.[m
     """[m
[31m-    x = np.asarray(x, dtype=np.float64)[m
[31m-    if x.ndim != 1:[m
[31m-        raise ValueError("x 1 boyutlu olmalıdır")[m
[31m-    if cutoff_hz <= 0 or sample_rate <= 0:[m
[31m-        raise ValueError("cutoff_hz ve sample_rate pozitif olmalıdır")[m
[31m-[m
[31m-    alpha = np.exp(-2.0 * np.pi * cutoff_hz / sample_rate)[m
[32m+[m[32m    x = _to_float32(signal)[m
     y = np.empty_like(x)[m
[31m-    y[0] = 0.0[m
[31m-    for n in range(1, x.size):[m
[31m-        y[n] = alpha * (y[n - 1] + x[n] - x[n - 1])[m
[32m+[m[32m    cutoff = _safe_cutoff(cutoff_hz, sample_rate)[m
[32m+[m[32m    alpha = _FT(math.exp(-2.0 * math.pi * cutoff / sample_rate))[m
[32m+[m[32m    b0 = _FT(1.0) - alpha[m
[32m+[m[32m    y_prev = _FT(0.0)[m
[32m+[m[32m    for i, sample in enumerate(x):[m
[32m+[m[32m        y_curr = b0 * sample + alpha * y_prev + _DENORM_GUARD[m
[32m+[m[32m        y[i] = y_curr[m
[32m+[m[32m        y_prev = y_curr[m
     return y[m
 [m
 [m
[31m-def bandpass_simple(x: np.ndarray, low_cut_hz: float, high_cut_hz: float, sample_rate: float) -> np.ndarray:[m
[31m-    """Basit bant geçiren filtre (HPF + LPF) uygular.[m
[31m-[m
[31m-    Parametreler:[m
[31m-        x: Giriş sinyali (1B numpy dizisi).[m
[31m-        low_cut_hz: Alt kesim (Hz) için yüksek geçiren filtre.[m
[31m-        high_cut_hz: Üst kesim (Hz) için alçak geçiren filtre.[m
[31m-        sample_rate: Örnekleme hızı (Hz).[m
[31m-[m
[31m-    Döndürür:[m
[31m-        Filtrelenmiş sinyal (numpy dizisi).[m
[32m+[m[32mdef one_pole_highpass([m
[32m+[m[32m    signal: FloatArray, cutoff_hz: float, sample_rate: int[m
[32m+[m[32m) -> FloatArray:[m
     """[m
[31m-    if low_cut_hz >= high_cut_hz:[m
[31m-        raise ValueError("low_cut_hz high_cut_hz değerinden küçük olmalıdır")[m
[31m-    high_passed = hpf_one_pole(x, low_cut_hz, sample_rate)[m
[31m-    return lpf_one_pole(high_passed, high_cut_hz, sample_rate)[m
[31m-[m
[31m-[m
[31m-def eq_3band([m
[31m-    x: np.ndarray,[m
[31m-    low_gain: float,[m
[31m-    mid_gain: float,[m
[31m-    high_gain: float,[m
[31m-    low_cut_hz: float,[m
[31m-    high_cut_hz: float,[m
[31m-    sample_rate: float,[m
[31m-) -> np.ndarray:[m
[31m-    """3 bantlı basit EQ uygular (low/mid/high kazanç).[m
[32m+[m[32m    Birinci dereceden yüksek geçiren filtre uygular.[m
 [m
     Parametreler:[m
[31m-        x: Giriş sinyali (1B numpy dizisi).[m
[31m-        low_gain: Alçak frekans kazancı (linear).[m
[31m-        mid_gain: Orta frekans kazancı (linear).[m
[31m-        high_gain: Yüksek frekans kazancı (linear).[m
[31m-        low_cut_hz: Low geçişi ayırmak için kesim (Hz).[m
[31m-        high_cut_hz: High geçişi ayırmak için kesim (Hz).[m
[32m+[m[32m        signal: Mono giriş sinyali (float32).[m
[32m+[m[32m        cutoff_hz: Kesim frekansı (Hz).[m
         sample_rate: Örnekleme hızı (Hz).[m
 [m
[31m-    Döndürür:[m
[31m-        EQ uygulanmış sinyal (numpy dizisi).[m
[32m+[m[32m    Dönüş:[m
[32m+[m[32m        High-pass uygulanmış float32 sinyal.[m
     """[m
[31m-    if not (0 < low_cut_hz < high_cut_hz < sample_rate * 0.5):[m
[31m-        raise ValueError("Kesim frekansları 0 < low < high < Nyquist olmalıdır")[m
[31m-[m
[31m-    low = lpf_one_pole(x, low_cut_hz, sample_rate)[m
[31m-    high = hpf_one_pole(x, high_cut_hz, sample_rate)[m
[31m-    mid = x - low - high[m
[32m+[m[32m    x = _to_float32(signal)[m
[32m+[m[32m    y = np.empty_like(x)[m
[32m+[m[32m    cutoff = _safe_cutoff(cutoff_hz, sample_rate)[m
[32m+[m[32m    alpha = _FT(math.exp(-2.0 * math.pi * cutoff / sample_rate))[m
[32m+[m[32m    x_prev = _FT(0.0)[m
[32m+[m[32m    y_prev = _FT(0.0)[m
[32m+[m[32m    for i, sample in enumerate(x):[m
[32m+[m[32m        y_curr = alpha * (y_prev + sample - x_prev) + _DENORM_GUARD[m
[32m+[m[32m        y[i] = y_curr[m
[32m+[m[32m        y_prev = y_curr[m
[32m+[m[32m        x_prev = sample[m
[32m+[m[32m    return y[m
 [m
[31m-    return low * low_gain + mid * mid_gain + high * high_gain[m
 [m
[32m+[m[32mdef tilt_filter([m
[32m+[m[32m    signal: FloatArray, tilt_db: float, sample_rate: int[m
[32m+[m[32m) -> FloatArray:[m
[32m+[m[32m    """[m
[32m+[m[32m    Spektral eğimi lineer olarak eğen tek kutuplu tilt filtresi uygular.[m
 [m
[31m-def normalize(x: np.ndarray, target_peak: float = 1.0, eps: float = 1e-12) -> np.ndarray:[m
[31m-    """Sinyali verilen tepe değere ölçekler.[m
[32m+[m[32m    Pozitif değerler üst frekansları yükseltirken düşük frekansları bastırır.[m
 [m
     Parametreler:[m
[31m-        x: Giriş sinyali (1B numpy dizisi).[m
[31m-        target_peak: İstenen tepe değeri (varsayılan 1.0).[m
[31m-        eps: Sıfıra bölünmeyi önlemek için küçük değer.[m
[32m+[m[32m        signal: Mono giriş sinyali (float32).[m
[32m+[m[32m        tilt_db: Her decade başına dB eğim.[m
[32m+[m[32m        sample_rate: Örnekleme hızı (Hz).[m
 [m
[31m-    Döndürür:[m
[31m-        Ölçeklenmiş sinyal (numpy dizisi).[m
[32m+[m[32m    Dönüş:[m
[32m+[m[32m        Spektral eğimli float32 sinyal.[m
     """[m
[31m-    x = np.asarray(x, dtype=np.float64)[m
[31m-    peak = np.max(np.abs(x))[m
[31m-    if peak < eps:[m
[31m-        return np.zeros_like(x)[m
[31m-    return x * (target_peak / peak)[m
[31m-[m
[31m-[m
[31m-if __name__ == "__main__":[m
[31m-    # Basit testler[m
[31m-    sr = 48000[m
[31m-    t = np.linspace(0, 1.0, sr, endpoint=False)[m
[31m-    x = np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 4000 * t)[m
[31m-[m
[31m-    lpf_out = lpf_one_pole(x, 500, sr)[m
[31m-    hpf_out = hpf_one_pole(x, 1000, sr)[m
[31m-    bp_out = bandpass_simple(x, 300, 2000, sr)[m
[31m-    eq_out = eq_3band([m
[31m-        x,[m
[31m-        low_gain=1.2,[m
[31m-        mid_gain=0.8,[m
[31m-        high_gain=1.5,[m
[31m-        low_cut_hz=300,[m
[31m-        high_cut_hz=3000,[m
[31m-        sample_rate=sr,[m
[31m-    )[m
[31m-    norm_out = normalize(eq_out)[m
[31m-[m
[31m-    print("LPF ortalama", float(np.mean(lpf_out)))[m
[31m-    print("HPF ortalama", float(np.mean(hpf_out)))[m
[31m-    print("Bandpass max", float(np.max(np.abs(bp_out))))[m
[31m-    print("EQ max", float(np.max(np.abs(eq_out))))[m
[31m-    print("Normalize hedef", float(np.max(np.abs(norm_out))))[m
[32m+[m[32m    x = _to_float32(signal)[m
[32m+[m[32m    y = np.empty_like(x)[m
[32m+[m[32m    # Tilt katsayısı yüksek geçiren kısmın ağırlığını belirler.[m
[32m+[m[32m    hp_gain = _FT(10 ** (tilt_db / 20.0))[m
[32m+[m[32m    lp_gain = _FT(1.0)[m
[32m+[m[32m    # Orta bandı referanslamak için toplam kazancı normalize et.[m
[32m+[m[32m    norm = _FT(1.0 / (hp_gain + lp_gain))[m
[32m+[m[32m    hp_gain *= norm[m
[32m+[m[32m    lp_gain *= norm[m
[32m+[m[32m    # 200 Hz merkezli hafif geçiş için sabit kesim seçimi.[m
[32m+[m[32m    cutoff = _safe_cutoff(200.0, sample_rate)[m
[32m+[m[32m    alpha = _FT(math.exp(-2.0 * math.pi * cutoff / sample_rate))[m
[32m+[m[32m    lp_prev = _FT(0.0)[m
[32m+[m[32m    hp_prev = _FT(0.0)[m
[32m+[m[32m    x_prev = _FT(0.0)[m
[32m+[m[32m    for i, sample in enumerate(x):[m
[32m+[m[32m        lp_curr = (_FT(1.0) - alpha) * sample + alpha * lp_prev + _DENORM_GUARD[m
[32m+[m[32m        hp_curr = alpha * (hp_prev + sample - x_prev) + _DENORM_GUARD[m
[32m+[m[32m        y[i] = lp_gain * lp_curr + hp_gain * hp_curr[m
[32m+[m[32m        lp_prev = lp_curr[m
[32m+[m[32m        hp_prev = hp_curr[m
[32m+[m[32m        x_prev = sample[m
[32m+[m[32m    return y[m
[1mdiff --git a/core_dsp/dsp_filters_v2.py b/core_dsp/dsp_filters_v2.py[m
[1mnew file mode 100644[m
[1mindex 0000000..918c2aa[m
[1m--- /dev/null[m
[1m+++ b/core_dsp/dsp_filters_v2.py[m
[36m@@ -0,0 +1,402 @@[m
[32m+[m[32mimport math[m
[32m+[m[32mfrom typing import Tuple[m
[32m+[m
[32m+[m[32mimport numpy as np[m
[32m+[m
[32m+[m[32mFloatArray = np.ndarray[m
[32m+[m[32m_FT = np.float32[m
[32m+[m[32m_DENORM_GUARD = _FT(1e-20)[m
[32m+[m
[32m+[m
[32m+[m[32mdef _to_float32(signal: FloatArray) -> FloatArray:[m
[32m+[m[32m    """Girişi float32 kopyasına dönüştürür."""[m
[32m+[m[32m    if signal.dtype != _FT:[m
[32m+[m[32m        return signal.astype(_FT, copy=False)[m
[32m+[m[32m    return signal[m
[32m+[m
[32m+[m
[32m+[m[32mdef _safe_cutoff(cutoff_hz: float, sample_rate: int) -> float:[m
[32m+[m[32m    """Kesim frekansını Nyquist sınırında güvenli aralığa sıkıştırır."""[m
[32m+[m[32m    nyquist = sample_rate * 0.5[m
[32m+[m[32m    return float(np.clip(cutoff_hz, 1.0, nyquist * 0.95))[m
[32m+[m
[32m+[m
[32m+[m[32mdef _biquad_filter([m
[32m+[m[32m    signal: FloatArray,[m
[32m+[m[32m    b0: _FT,[m
[32m+[m[32m    b1: _FT,[m
[32m+[m[32m    b2: _FT,[m
[32m+[m[32m    a1: _FT,[m
[32m+[m[32m    a2: _FT,[m
[32m+[m[32m) -> FloatArray:[m
[32m+[m[32m    """Biquad fark denklemini uygular."""[m
[32m+[m[32m    x = _to_float32(signal)[m
[32m+[m[32m    y = np.empty_like(x)[m
[32m+[m[32m    x1 = _FT(0.0)[m
[32m+[m[32m    x2 = _FT(0.0)[m
[32m+[m[32m    y1 = _FT(0.0)[m
[32m+[m[32m    y2 = _FT(0.0)[m
[32m+[m[32m    for i, sample in enumerate(x):[m
[32m+[m[32m        y0 = ([m
[32m+[m[32m            b0 * sample[m
[32m+[m[32m            + b1 * x1[m
[32m+[m[32m            + b2 * x2[m
[32m+[m[32m            - a1 * y1[m
[32m+[m[32m            - a2 * y2[m
[32m+[m[32m            + _DENORM_GUARD[m
[32m+[m[32m        )[m
[32m+[m[32m        y[i] = y0[m
[32m+[m[32m        x2 = x1[m
[32m+[m[32m        x1 = sample[m
[32m+[m[32m        y2 = y1[m
[32m+[m[32m        y1 = y0[m
[32m+[m[32m    return y[m
[32m+[m
[32m+[m
[32m+[m[32mdef _biquad_coeffs_peaking([m
[32m+[m[32m    freq_hz: float, q: float, gain_db: float, sample_rate: int[m
[32m+[m[32m) -> Tuple[_FT, _FT, _FT, _FT, _FT]:[m
[32m+[m[32m    """RBJ peaking biquad katsayılarını döndürür."""[m
[32m+[m[32m    a = math.pow(10.0, gain_db / 40.0)[m
[32m+[m[32m    w0 = 2.0 * math.pi * freq_hz / sample_rate[m
[32m+[m[32m    alpha = math.sin(w0) / (2.0 * max(q, 1e-6))[m
[32m+[m[32m    cos_w0 = math.cos(w0)[m
[32m+[m
[32m+[m[32m    b0 = 1.0 + alpha * a[m
[32m+[m[32m    b1 = -2.0 * cos_w0[m
[32m+[m[32m    b2 = 1.0 - alpha * a[m
[32m+[m[32m    a0 = 1.0 + alpha / a[m
[32m+[m[32m    a1 = -2.0 * cos_w0[m
[32m+[m[32m    a2 = 1.0 - alpha / a[m
[32m+[m
[32m+[m[32m    return ([m
[32m+[m[32m        _FT(b0 / a0),[m
[32m+[m[32m        _FT(b1 / a0),[m
[32m+[m[32m        _FT(b2 / a0),[m
[32m+[m[32m        _FT(a1 / a0),[m
[32m+[m[32m        _FT(a2 / a0),[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m
[32m+[m[32mdef _biquad_coeffs_shelf([m
[32m+[m[32m    freq_hz: float,[m
[32m+[m[32m    slope: float,[m
[32m+[m[32m    gain_db: float,[m
[32m+[m[32m    sample_rate: int,[m
[32m+[m[32m    shelf_type: str,[m
[32m+[m[32m) -> Tuple[_FT, _FT, _FT, _FT, _FT]:[m
[32m+[m[32m    """RBJ low/high shelf biquad katsayılarını döndürür."""[m
[32m+[m[32m    a = math.pow(10.0, gain_db / 40.0)[m
[32m+[m[32m    w0 = 2.0 * math.pi * freq_hz / sample_rate[m
[32m+[m[32m    cos_w0 = math.cos(w0)[m
[32m+[m[32m    sin_w0 = math.sin(w0)[m
[32m+[m[32m    alpha = sin_w0 / 2.0 * math.sqrt([m
[32m+[m[32m        max((a + 1.0 / a) * (1.0 / max(slope, 1e-6) - 1.0) + 2.0, 0.0)[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m[32m    if shelf_type == "low":[m
[32m+[m[32m        b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * math.sqrt(a) * alpha)[m
[32m+[m[32m        b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0)[m
[32m+[m[32m        b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * math.sqrt(a) * alpha)[m
[32m+[m[32m        a0 = (a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * math.sqrt(a) * alpha[m
[32m+[m[32m        a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0)[m
[32m+[m[32m        a2 = (a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * math.sqrt(a) * alpha[m
[32m+[m[32m    else:[m
[32m+[m[32m        b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * math.sqrt(a) * alpha)[m
[32m+[m[32m        b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0)[m
[32m+[m[32m        b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * math.sqrt(a) * alpha)[m
[32m+[m[32m        a0 = (a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * math.sqrt(a) * alpha[m
[32m+[m[32m        a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0)[m
[32m+[m[32m        a2 = (a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * math.sqrt(a) * alpha[m
[32m+[m
[32m+[m[32m    return ([m
[32m+[m[32m        _FT(b0 / a0),[m
[32m+[m[32m        _FT(b1 / a0),[m
[32m+[m[32m        _FT(b2 / a0),[m
[32m+[m[32m        _FT(a1 / a0),[m
[32m+[m[32m        _FT(a2 / a0),[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m
[32m+[m[32mdef dc_block(signal: FloatArray, pole: float = 0.995) -> FloatArray:[m
[32m+[m[32m    """[m
[32m+[m[32m    DC bileşenini bastıran birinci dereceden yüksek geçiren filtre uygular.[m
[32m+[m
[32m+[m[32m    Parametreler:[m
[32m+[m[32m        signal: Mono giriş sinyali (float32).[m
[32m+[m[32m        pole: 1.0'a yaklaştıkça kesim düşer, tipik 0.995.[m
[32m+[m
[32m+[m[32m    Dönüş:[m
[32m+[m[32m        DC'si bastırılmış float32 sinyal.[m
[32m+[m[32m    """[m
[32m+[m[32m    x = _to_float32(signal)[m
[32m+[m[32m    y = np.empty_like(x)[m
[32m+[m[32m    pole_f = _FT(pole)[m
[32m+[m[32m    x_prev = _FT(0.0)[m
[32m+[m[32m    y_prev = _FT(0.0)[m
[32m+[m[32m    for i, sample in enumerate(x):[m
[32m+[m[32m        y_curr = sample - x_prev + pole_f * y_prev + _DENORM_GUARD[m
[32m+[m[32m        y[i] = y_curr[m
[32m+[m[32m        x_prev = sample[m
[32m+[m[32m        y_prev = y_curr[m
[32m+[m[32m    return y[m
[32m+[m
[32m+[m
[32m+[m[32mdef one_pole_lowpass([m
[32m+[m[32m    signal: FloatArray, cutoff_hz: float, sample_rate: int[m
[32m+[m[32m) -> FloatArray:[m
[32m+[m[32m    """[m
[32m+[m[32m    Birinci dereceden düşük geçiren filtre uygular.[m
[32m+[m
[32m+[m[32m    Parametreler:[m
[32m+[m[32m        signal: Mono giriş sinyali (float32).[m
[32m+[m[32m        cutoff_hz: Kesim frekansı (Hz).[m
[32m+[m[32m        sample_rate: Örnekleme hızı (Hz).[m
[32m+[m
[32m+[m[32m    Dönüş:[m
[32m+[m[32m        Low-pass uygulanmış float32 sinyal.[m
[32m+[m[32m    """[m
[32m+[m[32m    x = _to_float32(signal)[m
[32m+[m[32m    y = np.empty_like(x)[m
[32m+[m[32m    cutoff = _safe_cutoff(cutoff_hz, sample_rate)[m
[32m+[m[32m    alpha = _FT(math.exp(-2.0 * math.pi * cutoff / sample_rate))[m
[32m+[m[32m    b0 = _FT(1.0) - alpha[m
[32m+[m[32m    y_prev = _FT(0.0)[m
[32m+[m[32m    for i, sample in enumerate(x):[m
[32m+[m[32m        y_curr = b0 * sample + alpha * y_prev + _DENORM_GUARD[m
[32m+[m[32m        y[i] = y_curr[m
[32m+[m[32m        y_prev = y_curr[m
[32m+[m[32m    return y[m
[32m+[m
[32m+[m
[32m+[m[32mdef one_pole_highpass([m
[32m+[m[32m    signal: FloatArray, cutoff_hz: float, sample_rate: int[m
[32m+[m[32m) -> FloatArray:[m
[32m+[m[32m    """[m
[32m+[m[32m    Birinci dereceden yüksek geçiren filtre uygular.[m
[32m+[m
[32m+[m[32m    Parametreler:[m
[32m+[m[32m        signal: Mono giriş sinyali (float32).[m
[32m+[m[32m        cutoff_hz: Kesim frekansı (Hz).[m
[32m+[m[32m        sample_rate: Örnekleme hızı (Hz).[m
[32m+[m
[32m+[m[32m    Dönüş:[m
[32m+[m[32m        High-pass uygulanmış float32 sinyal.[m
[32m+[m[32m    """[m
[32m+[m[32m    x = _to_float32(signal)[m
[32m+[m[32m    y = np.empty_like(x)[m
[32m+[m[32m    cutoff = _safe_cutoff(cutoff_hz, sample_rate)[m
[32m+[m[32m    alpha = _FT(math.exp(-2.0 * math.pi * cutoff / sample_rate))[m
[32m+[m[32m    x_prev = _FT(0.0)[m
[32m+[m[32m    y_prev = _FT(0.0)[m
[32m+[m[32m    for i, sample in enumerate(x):[m
[32m+[m[32m        y_curr = alpha * (y_prev + sample - x_prev) + _DENORM_GUARD[m
[32m+[m[32m        y[i] = y_curr[m
[32m+[m[32m        y_prev = y_curr[m
[32m+[m[32m        x_prev = sample[m
[32m+[m[32m    return y[m
[32m+[m
[32m+[m
[32m+[m[32mdef peaking_eq([m
[32m+[m[32m    signal: FloatArray,[m
[32m+[m[32m    freq_hz: float,[m
[32m+[m[32m    q: float,[m
[32m+[m[32m    gain_db: float,[m
[32m+[m[32m    sample_rate: int,[m
[32m+[m[32m) -> FloatArray:[m
[32m+[m[32m    """[m
[32m+[m[32m    RBJ peaking eşitleyici uygular.[m
[32m+[m
[32m+[m[32m    Parametreler:[m
[32m+[m[32m        signal: Mono giriş sinyali (float32).[m
[32m+[m[32m        freq_hz: Merkez frekans (Hz).[m
[32m+[m[32m        q: Q değeri (boyutsuz).[m
[32m+[m[32m        gain_db: Kazanç (dB).[m
[32m+[m[32m        sample_rate: Örnekleme hızı (Hz).[m
[32m+[m
[32m+[m[32m    Dönüş:[m
[32m+[m[32m        Peaking EQ uygulanmış float32 sinyal.[m
[32m+[m[32m    """[m
[32m+[m[32m    freq = _safe_cutoff(freq_hz, sample_rate)[m
[32m+[m[32m    b0, b1, b2, a1, a2 = _biquad_coeffs_peaking(freq, q, gain_db, sample_rate)[m
[32m+[m[32m    return _biquad_filter(signal, b0, b1, b2, a1, a2)[m
[32m+[m
[32m+[m
[32m+[m[32mdef low_shelf([m
[32m+[m[32m    signal: FloatArray,[m
[32m+[m[32m    freq_hz: float,[m
[32m+[m[32m    slope: float,[m
[32m+[m[32m    gain_db: float,[m
[32m+[m[32m    sample_rate: int,[m
[32m+[m[32m) -> FloatArray:[m
[32m+[m[32m    """[m
[32m+[m[32m    RBJ düşük raf (low-shelf) eşitleyici uygular.[m
[32m+[m
[32m+[m[32m    Parametreler:[m
[32m+[m[32m        signal: Mono giriş sinyali (float32).[m
[32m+[m[32m        freq_hz: Raf geçiş frekansı (Hz).[m
[32m+[m[32m        slope: Raf eğimi (boyutsuz, 0-1 arası tipik).[m
[32m+[m[32m        gain_db: Kazanç (dB).[m
[32m+[m[32m        sample_rate: Örnekleme hızı (Hz).[m
[32m+[m
[32m+[m[32m    Dönüş:[m
[32m+[m[32m        Low-shelf uygulanmış float32 sinyal.[m
[32m+[m[32m    """[m
[32m+[m[32m    freq = _safe_cutoff(freq_hz, sample_rate)[m
[32m+[m[32m    b0, b1, b2, a1, a2 = _biquad_coeffs_shelf([m
[32m+[m[32m        freq, max(slope, 1e-4), gain_db, sample_rate, shelf_type="low"[m
[32m+[m[32m    )[m
[32m+[m[32m    return _biquad_filter(signal, b0, b1, b2, a1, a2)[m
[32m+[m
[32m+[m
[32m+[m[32mdef high_shelf([m
[32m+[m[32m    signal: FloatArray,[m
[32m+[m[32m    freq_hz: float,[m
[32m+[m[32m    slope: float,[m
[32m+[m[32m    gain_db: float,[m
[32m+[m[32m    sample_rate: int,[m
[32m+[m[32m) -> FloatArray:[m
[32m+[m[32m    """[m
[32m+[m[32m    RBJ yüksek raf (high-shelf) eşitleyici uygular.[m
[32m+[m
[32m+[m[32m    Parametreler:[m
[32m+[m[32m        signal: Mono giriş sinyali (float32).[m
[32m+[m[32m        freq_hz: Raf geçiş frekansı (Hz).[m
[32m+[m[32m        slope: Raf eğimi (boyutsuz, 0-1 arası tipik).[m
[32m+[m[32m        gain_db: Kazanç (dB).[m
[32m+[m[32m        sample_rate: Örnekleme hızı (Hz).[m
[32m+[m
[32m+[m[32m    Dönüş:[m
[32m+[m[32m        High-shelf uygulanmış float32 sinyal.[m
[32m+[m[32m    """[m
[32m+[m[32m    freq = _safe_cutoff(freq_hz, sample_rate)[m
[32m+[m[32m    b0, b1, b2, a1, a2 = _biquad_coeffs_shelf([m
[32m+[m[32m        freq, max(slope, 1e-4), gain_db, sample_rate, shelf_type="high"[m
[32m+[m[32m    )[m
[32m+[m[32m    return _biquad_filter(signal, b0, b1, b2, a1, a2)[m
[32m+[m
[32m+[m
[32m+[m[32mdef tilt_filter([m
[32m+[m[32m    signal: FloatArray, pivot_hz: float, tilt_db: float, sample_rate: int[m
[32m+[m[32m) -> FloatArray:[m
[32m+[m[32m    """[m
[32m+[m[32m    Basit tilt eşitleyici uygular; pivot çevresinde eğim verir.[m
[32m+[m
[32m+[m[32m    Parametreler:[m
[32m+[m[32m        signal: Mono giriş sinyali (float32).[m
[32m+[m[32m        pivot_hz: Eğimin dönüm noktası (Hz).[m
[32m+[m[32m        tilt_db: Üst frekanslar için dB artışı (negatifse azaltma).[m
[32m+[m[32m        sample_rate: Örnekleme hızı (Hz).[m
[32m+[m
[32m+[m[32m    Dönüş:[m
[32m+[m[32m        Tilt uygulanmış float32 sinyal.[m
[32m+[m[32m    """[m
[32m+[m[32m    freq = _safe_cutoff(pivot_hz, sample_rate)[m
[32m+[m[32m    x = _to_float32(signal)[m
[32m+[m[32m    low = one_pole_lowpass(x, freq, sample_rate)[m
[32m+[m[32m    high = x - low[m
[32m+[m[32m    g = math.pow(10.0, tilt_db / 20.0)[m
[32m+[m[32m    low_gain = _FT(1.0 / math.sqrt(g))[m
[32m+[m[32m    high_gain = _FT(math.sqrt(g))[m
[32m+[m[32m    norm = _FT(1.0 / (abs(low_gain) + abs(high_gain)))[m
[32m+[m[32m    low_gain *= norm[m
[32m+[m[32m    high_gain *= norm[m
[32m+[m[32m    return low_gain * low + high_gain * high[m
[32m+[m
[32m+[m
[32m+[m[32mdef butterworth_lowpass([m
[32m+[m[32m    signal: FloatArray, cutoff_hz: float, sample_rate: int[m
[32m+[m[32m) -> FloatArray:[m
[32m+[m[32m    """[m
[32m+[m[32m    İkinci dereceden Butterworth düşük geçiren filtre uygular.[m
[32m+[m
[32m+[m[32m    Parametreler:[m
[32m+[m[32m        signal: Mono giriş sinyali (float32).[m
[32m+[m[32m        cutoff_hz: Kesim frekansı (Hz).[m
[32m+[m[32m        sample_rate: Örnekleme hızı (Hz).[m
[32m+[m
[32m+[m[32m    Dönüş:[m
[32m+[m[32m        Butterworth low-pass uygulanmış float32 sinyal.[m
[32m+[m[32m    """[m
[32m+[m[32m    freq = _safe_cutoff(cutoff_hz, sample_rate)[m
[32m+[m[32m    w0 = 2.0 * math.pi * freq / sample_rate[m
[32m+[m[32m    cos_w0 = math.cos(w0)[m
[32m+[m[32m    sin_w0 = math.sin(w0)[m
[32m+[m[32m    alpha = sin_w0 / math.sqrt(2.0)[m
[32m+[m[32m    b0 = (1.0 - cos_w0) * 0.5[m
[32m+[m[32m    b1 = 1.0 - cos_w0[m
[32m+[m[32m    b2 = (1.0 - cos_w0) * 0.5[m
[32m+[m[32m    a0 = 1.0 + alpha[m
[32m+[m[32m    a1 = -2.0 * cos_w0[m
[32m+[m[32m    a2 = 1.0 - alpha[m
[32m+[m[32m    return _biquad_filter([m
[32m+[m[32m        signal,[m
[32m+[m[32m        _FT(b0 / a0),[m
[32m+[m[32m        _FT(b1 / a0),[m
[32m+[m[32m        _FT(b2 / a0),[m
[32m+[m[32m        _FT(a1 / a0),[m
[32m+[m[32m        _FT(a2 / a0),[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m
[32m+[m[32mdef butterworth_highpass([m
[32m+[m[32m    signal: FloatArray, cutoff_hz: float, sample_rate: int[m
[32m+[m[32m) -> FloatArray:[m
[32m+[m[32m    """[m
[32m+[m[32m    İkinci dereceden Butterworth yüksek geçiren filtre uygular.[m
[32m+[m
[32m+[m[32m    Parametreler:[m
[32m+[m[32m        signal: Mono giriş sinyali (float32).[m
[32m+[m[32m        cutoff_hz: Kesim frekansı (Hz).[m
[32m+[m[32m        sample_rate: Örnekleme hızı (Hz).[m
[32m+[m
[32m+[m[32m    Dönüş:[m
[32m+[m[32m        Butterworth high-pass uygulanmış float32 sinyal.[m
[32m+[m[32m    """[m
[32m+[m[32m    freq = _safe_cutoff(cutoff_hz, sample_rate)[m
[32m+[m[32m    w0 = 2.0 * math.pi * freq / sample_rate[m
[32m+[m[32m    cos_w0 = math.cos(w0)[m
[32m+[m[32m    sin_w0 = math.sin(w0)[m
[32m+[m[32m    alpha = sin_w0 / math.sqrt(2.0)[m
[32m+[m[32m    b0 = (1.0 + cos_w0) * 0.5[m
[32m+[m[32m    b1 = -(1.0 + cos_w0)[m
[32m+[m[32m    b2 = (1.0 + cos_w0) * 0.5[m
[32m+[m[32m    a0 = 1.0 + alpha[m
[32m+[m[32m    a1 = -2.0 * cos_w0[m
[32m+[m[32m    a2 = 1.0 - alpha[m
[32m+[m[32m    return _biquad_filter([m
[32m+[m[32m        signal,[m
[32m+[m[32m        _FT(b0 / a0),[m
[32m+[m[32m        _FT(b1 / a0),[m
[32m+[m[32m        _FT(b2 / a0),[m
[32m+[m[32m        _FT(a1 / a0),[m
[32m+[m[32m        _FT(a2 / a0),[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m
[32m+[m[32mdef _rms(signal: FloatArray) -> float:[m
[32m+[m[32m    """RMS değerini float olarak döndürür."""[m
[32m+[m[32m    x = _to_float32(signal)[m
[32m+[m[32m    return float(np.sqrt(np.mean(x * x, dtype=_FT)))[m
[32m+[m
[32m+[m
[32m+[m[32mif __name__ == "__main__":[m
[32m+[m[32m    fs = 48000[m
[32m+[m[32m    t = np.arange(fs, dtype=_FT) / _FT(fs)[m
[32m+[m[32m    sine = np.sin(2.0 * math.pi * 440.0 * t, dtype=_FT) * _FT(0.5)[m
[32m+[m
[32m+[m[32m    lp = one_pole_lowpass(sine, cutoff_hz=2000.0, sample_rate=fs)[m
[32m+[m[32m    hp = one_pole_highpass(sine, cutoff_hz=200.0, sample_rate=fs)[m
[32m+[m[32m    peak = peaking_eq(sine, freq_hz=1000.0, q=1.0, gain_db=3.0, sample_rate=fs)[m
[32m+[m[32m    tilt = tilt_filter(sine, pivot_hz=800.0, tilt_db=4.0, sample_rate=fs)[m
[32m+[m[32m    bw_lp = butterworth_lowpass(sine, cutoff_hz=1500.0, sample_rate=fs)[m
[32m+[m[32m    bw_hp = butterworth_highpass(sine, cutoff_hz=300.0, sample_rate=fs)[m
[32m+[m
[32m+[m[32m    print("Test sinyalleri RMS:")[m
[32m+[m[32m    print(f"Orijinal: {_rms(sine):.6f}")[m
[32m+[m[32m    print(f"LP: {_rms(lp):.6f}")[m
[32m+[m[32m    print(f"HP: {_rms(hp):.6f}")[m
[32m+[m[32m    print(f"Peaking: {_rms(peak):.6f}")[m
[32m+[m[32m    print(f"Tilt: {_rms(tilt):.6f}")[m
[32m+[m[32m    print(f"BW LP: {_rms(bw_lp):.6f}")[m
[32m+[m[32m    print(f"BW HP: {_rms(bw_hp):.6f}")[m
[1mdiff --git a/core_dsp/dsp_fx.py b/core_dsp/dsp_fx.py[m
[1mindex a1dc5d2..ce2e774 100644[m
[1m--- a/core_dsp/dsp_fx.py[m
[1m+++ b/core_dsp/dsp_fx.py[m
[36m@@ -1,165 +1,185 @@[m
[31m-"""Lightweight DSP effects for simple audio shaping.[m
[31m-[m
[31m-This module includes:[m
[31m-- Soft saturation using a hyperbolic tangent curve[m
[31m-- Warmth (tilt EQ plus saturation)[m
[31m-- Simple fake reverb (multi-tap delay with a low-pass tail)[m
[31m-- Stereo widening via mid/side processing[m
[31m-- Gain normalization[m
[31m-"""[m
[31m-[m
[31m-from __future__ import annotations[m
[31m-[m
 import math[m
[31m-from typing import Iterable, Optional[m
[32m+[m[32mfrom typing import Tuple[m
 [m
 import numpy as np[m
 [m
[31m-[m
[31m-def _as_2d(signal: np.ndarray) -> np.ndarray:[m
[31m-    """Ensure signal has shape (n_samples, n_channels)."""[m
[31m-    arr = np.asarray(signal, dtype=np.float64)[m
[31m-    if arr.ndim == 1:[m
[31m-        return arr[:, None][m
[31m-    if arr.ndim != 2:[m
[31m-        raise ValueError("Signal must be 1D (mono) or 2D (n_samples, n_channels).")[m
[31m-    return arr[m
[31m-[m
[31m-[m
[31m-def _restore_shape(original: np.ndarray, processed: np.ndarray) -> np.ndarray:[m
[31m-    """Return processed array with the same dimensionality as original."""[m
[31m-    if original.ndim == 1:[m
[31m-        return processed[:, 0][m
[31m-    return processed[m
[31m-[m
[31m-[m
[31m-def _one_pole_lowpass([m
[31m-    signal: np.ndarray,[m
[31m-    sample_rate: float,[m
[31m-    cutoff_hz: float,[m
[31m-) -> np.ndarray:[m
[31m-    """One-pole low-pass filter; simple and stable for tonal shaping."""[m
[31m-    if cutoff_hz <= 0:[m
[31m-        raise ValueError("cutoff_hz must be positive.")[m
[31m-    arr = _as_2d(signal)[m
[31m-    alpha = math.exp(-2.0 * math.pi * cutoff_hz / sample_rate)[m
[31m-    one_minus_alpha = 1.0 - alpha[m
[31m-[m
[31m-    y = np.empty_like(arr)[m
[31m-    state = arr[0].copy()[m
[31m-    y[0] = state[m
[31m-    for i in range(1, len(arr)):[m
[31m-        state = alpha * state + one_minus_alpha * arr[i][m
[31m-        y[i] = state[m
[31m-    return _restore_shape(signal, y)[m
[31m-[m
[31m-[m
[31m-def soft_saturation([m
[31m-    signal: np.ndarray,[m
[31m-    drive: float = 1.25,[m
[31m-    makeup_gain: bool = True,[m
[31m-) -> np.ndarray:[m
[31m-    """Soft tanh saturation with optional makeup gain."""[m
[31m-    if drive <= 0:[m
[31m-        raise ValueError("drive must be positive.")[m
[31m-    saturated = np.tanh(np.asarray(signal, dtype=np.float64) * drive)[m
[31m-    if makeup_gain:[m
[31m-        gain = 1.0 / np.tanh(drive)[m
[31m-        saturated *= gain[m
[31m-    return saturated[m
[31m-[m
[31m-[m
[31m-def tilt_eq([m
[31m-    signal: np.ndarray,[m
[31m-    sample_rate: float,[m
[31m-    tilt_db: float = 3.0,[m
[31m-    pivot_hz: float = 1200.0,[m
[31m-) -> np.ndarray:[m
[31m-    """Tilt EQ around a pivot frequency; positive tilt warms (more lows)."""[m
[31m-    if sample_rate <= 0:[m
[31m-        raise ValueError("sample_rate must be positive.")[m
[31m-    if pivot_hz <= 0:[m
[31m-        raise ValueError("pivot_hz must be positive.")[m
[31m-[m
[31m-    low = _one_pole_lowpass(signal, sample_rate, pivot_hz)[m
[31m-    high = np.asarray(signal, dtype=np.float64) - low[m
[31m-[m
[31m-    low_gain = 10 ** (tilt_db / 20.0)[m
[31m-    high_gain = 1.0 / low_gain[m
[31m-    shaped = low * low_gain + high * high_gain[m
[31m-[m
[31m-    max_gain = max(low_gain, high_gain)[m
[31m-    shaped /= max_gain[m
[31m-    return shaped[m
[31m-[m
[31m-[m
[31m-def warmth([m
[31m-    signal: np.ndarray,[m
[31m-    sample_rate: float,[m
[31m-    tilt_db: float = 3.0,[m
[31m-    saturation_drive: float = 1.1,[m
[31m-) -> np.ndarray:[m
[31m-    """Combine tilt EQ with gentle saturation for a warmer tone."""[m
[31m-    tilted = tilt_eq(signal, sample_rate=sample_rate, tilt_db=tilt_db)[m
[31m-    return soft_saturation(tilted, drive=saturation_drive)[m
[31m-[m
[31m-[m
[31m-def fake_reverb([m
[31m-    signal: np.ndarray,[m
[31m-    sample_rate: float,[m
[31m-    delays_ms: Optional[Iterable[float]] = None,[m
[31m-    decay: float = 0.55,[m
[31m-    lpf_hz: float = 6000.0,[m
[31m-    wet: float = 0.3,[m
[31m-) -> np.ndarray:[m
[31m-    """Simple faux reverb using multi-tap delays and a low-pass tail."""[m
[31m-    if wet < 0 or wet > 1:[m
[31m-        raise ValueError("wet must be between 0 and 1.")[m
[31m-    if decay <= 0:[m
[31m-        raise ValueError("decay must be positive.")[m
[31m-    delays = list(delays_ms) if delays_ms is not None else [25.0, 55.0, 85.0, 120.0][m
[31m-    if not delays:[m
[31m-        raise ValueError("delays_ms must contain at least one value.")[m
[31m-[m
[31m-    dry = _as_2d(signal)[m
[31m-    n_samples, n_channels = dry.shape[m
[31m-    delay_samples = [int(sample_rate * d / 1000.0) for d in delays][m
[31m-    max_delay = max(delay_samples)[m
[31m-    out_len = n_samples + max_delay[m
[31m-[m
[31m-    wet_buf = np.zeros((out_len, n_channels), dtype=np.float64)[m
[31m-    for tap, offset in enumerate(delay_samples):[m
[31m-        attenuation = decay ** tap[m
[31m-        wet_buf[offset: offset + n_samples] += dry * attenuation[m
[31m-[m
[31m-    wet_buf = _as_2d(_one_pole_lowpass(wet_buf, sample_rate, lpf_hz))[m
[31m-[m
[31m-    dry_padded = np.zeros_like(wet_buf)[m
[31m-    dry_padded[:n_samples] = dry[m
[31m-[m
[31m-    mixed = dry_padded * (1.0 - wet) + wet_buf * wet[m
[31m-    return _restore_shape(signal, mixed)[m
[31m-[m
[31m-[m
[31m-def stereo_widen(signal: np.ndarray, width: float = 1.2) -> np.ndarray:[m
[31m-    """Mid/side stereo widening; width > 1 spreads, < 1 narrows."""[m
[31m-    arr = _as_2d(signal)[m
[31m-    if arr.shape[1] != 2:[m
[31m-        raise ValueError("stereo_widen expects a stereo (n_samples, 2) signal.")[m
[31m-    mid = 0.5 * (arr[:, 0] + arr[:, 1])[m
[31m-    side = 0.5 * (arr[:, 0] - arr[:, 1]) * width[m
[31m-    left = mid + side[m
[31m-    right = mid - side[m
[31m-    widened = np.stack([left, right], axis=1)[m
[31m-    return _restore_shape(signal, widened)[m
[31m-[m
[31m-[m
[31m-def normalize_gain(signal: np.ndarray, target_peak: float = 0.99) -> np.ndarray:[m
[31m-    """Normalize signal to a target peak level."""[m
[31m-    if target_peak <= 0:[m
[31m-        raise ValueError("target_peak must be positive.")[m
[31m-    arr = np.asarray(signal, dtype=np.float64)[m
[31m-    peak = np.max(np.abs(arr))[m
[31m-    if peak == 0:[m
[31m-        return arr[m
[31m-    return arr * (target_peak / peak)[m
[32m+[m[32mFloatArray = np.ndarray[m
[32m+[m[32m_FT = np.float32[m
[32m+[m[32m_DENORM_GUARD = _FT(1e-20)[m
[32m+[m
[32m+[m
[32m+[m[32mdef _to_float32(signal: FloatArray) -> FloatArray:[m
[32m+[m[32m    """Girişi float32 kopyasına dönüştürür."""[m
[32m+[m[32m    if signal.dtype != _FT:[m
[32m+[m[32m        return signal.astype(_FT, copy=False)[m
[32m+[m[32m    return signal[m
[32m+[m
[32m+[m
[32m+[m[32mdef _clamp(value: float, min_value: float, max_value: float) -> float:[m
[32m+[m[32m    """Değeri verilen aralıkta sınırlar."""[m
[32m+[m[32m    return float(np.clip(value, min_value, max_value))[m
[32m+[m
[32m+[m
[32m+[m[32mdef _rms(signal: FloatArray) -> float:[m
[32m+[m[32m    """RMS değerini float olarak döndürür."""[m
[32m+[m[32m    x = _to_float32(signal)[m
[32m+[m[32m    return float(np.sqrt(np.mean(x * x, dtype=_FT)))[m
[32m+[m
[32m+[m
[32m+[m[32mdef soft_saturation(signal: FloatArray, drive: float) -> FloatArray:[m
[32m+[m[32m    """[m
[32m+[m[32m    Arctan tabanlı yumuşak saturasyon uygular.[m
[32m+[m
[32m+[m[32m    Parametreler:[m
[32m+[m[32m        signal: Mono giriş sinyali (float32).[m
[32m+[m[32m        drive: Sürüş miktarı (0 ve üzeri).[m
[32m+[m
[32m+[m[32m    Dönüş:[m
[32m+[m[32m        Saturasyon uygulanmış float32 sinyal.[m
[32m+[m[32m    """[m
[32m+[m[32m    x = _to_float32(signal)[m
[32m+[m[32m    drv = _clamp(drive, 0.0, 10.0)[m
[32m+[m[32m    gain = _FT(1.0 + drv * 2.0)[m
[32m+[m[32m    out = (_FT(2.0 / math.pi)) * np.arctan(gain * x + _DENORM_GUARD, dtype=_FT)[m
[32m+[m[32m    comp = _FT(1.0 / (1.0 + 0.5 * drv))[m
[32m+[m[32m    return (out * comp).astype(_FT, copy=False)[m
[32m+[m
[32m+[m
[32m+[m[32mdef warmth(signal: FloatArray, amount: float) -> FloatArray:[m
[32m+[m[32m    """[m
[32m+[m[32m    Hafif yumuşatma ve düşük seviye harmonik ekler.[m
[32m+[m
[32m+[m[32m    Parametreler:[m
[32m+[m[32m        signal: Mono giriş sinyali (float32).[m
[32m+[m[32m        amount: 0-1 arası ısınma miktarı.[m
[32m+[m
[32m+[m[32m    Dönüş:[m
[32m+[m[32m        Isıtılmış float32 sinyal.[m
[32m+[m[32m    """[m
[32m+[m[32m    x = _to_float32(signal)[m
[32m+[m[32m    amt = _clamp(amount, 0.0, 1.0)[m
[32m+[m[32m    alpha = _FT(0.15 + 0.7 * amt)  # düşük geçiren yumuşatma katsayısı[m
[32m+[m[32m    y = np.empty_like(x)[m
[32m+[m[32m    lp_prev = _FT(0.0)[m
[32m+[m[32m    for i, sample in enumerate(x):[m
[32m+[m[32m        lp = (1.0 - alpha) * sample + alpha * lp_prev + _DENORM_GUARD[m
[32m+[m[32m        lp_prev = lp[m
[32m+[m[32m        harm = sample * sample * sample[m
[32m+[m[32m        warmed = (1.0 - amt) * sample + amt * (_FT(0.75) * sample + _FT(0.2) * lp + _FT(0.05) * harm)[m
[32m+[m[32m        y[i] = warmed[m
[32m+[m[32m    return y.astype(_FT, copy=False)[m
[32m+[m
[32m+[m
[32m+[m[32mdef simple_reverb(signal: FloatArray, sample_rate: int, mix: float) -> FloatArray:[m
[32m+[m[32m    """[m
[32m+[m[32m    Hafif geri beslemeli gecikmeye dayalı basit reverb uygular.[m
[32m+[m
[32m+[m[32m    Parametreler:[m
[32m+[m[32m        signal: Mono giriş sinyali (float32).[m
[32m+[m[32m        sample_rate: Örnekleme hızı (Hz).[m
[32m+[m[32m        mix: 0-1 arası ıslak/kuru karışım.[m
[32m+[m
[32m+[m[32m    Dönüş:[m
[32m+[m[32m        Reverb uygulanmış float32 sinyal.[m
[32m+[m[32m    """[m
[32m+[m[32m    x = _to_float32(signal)[m
[32m+[m[32m    wet = _clamp(mix, 0.0, 1.0)[m
[32m+[m[32m    dry = _FT(1.0 - wet)[m
[32m+[m[32m    delay_main = max(1, int(0.055 * sample_rate))[m
[32m+[m[32m    delay_tap = max(1, int(0.019 * sample_rate))[m
[32m+[m[32m    fb = _FT(0.45)[m
[32m+[m[32m    damp = _FT(0.25)[m
[32m+[m[32m    buf_len = delay_main + 1[m
[32m+[m[32m    buf = np.zeros(buf_len, dtype=_FT)[m
[32m+[m[32m    y = np.empty_like(x)[m
[32m+[m[32m    idx = 0[m
[32m+[m[32m    lp_prev = _FT(0.0)[m
[32m+[m[32m    for i, sample in enumerate(x):[m
[32m+[m[32m        delayed = buf[idx][m
[32m+[m[32m        # basit damping[m
[32m+[m[32m        lp = (1.0 - damp) * delayed + damp * lp_prev + _DENORM_GUARD[m
[32m+[m[32m        lp_prev = lp[m
[32m+[m[32m        # erken yansıma tap'i[m
[32m+[m[32m        tap_idx = (idx - delay_tap) % buf_len[m
[32m+[m[32m        tap = buf[tap_idx][m
[32m+[m[32m        new_val = sample + fb * lp[m
[32m+[m[32m        buf[idx] = new_val[m
[32m+[m[32m        idx = (idx + 1) % buf_len[m
[32m+[m[32m        wet_sample = _FT(0.6) * lp + _FT(0.4) * tap[m
[32m+[m[32m        y[i] = dry * sample + wet * wet_sample[m
[32m+[m[32m    return y.astype(_FT, copy=False)[m
[32m+[m
[32m+[m
[32m+[m[32mdef stereo_widen(signal: FloatArray, amount: float) -> FloatArray:[m
[32m+[m[32m    """[m
[32m+[m[32m    Dekorele küçük gecikmelerle stereo genişletme uygular.[m
[32m+[m
[32m+[m[32m    Parametreler:[m
[32m+[m[32m        signal: Mono giriş sinyali (float32).[m
[32m+[m[32m        amount: 0-1 arası genişlik miktarı.[m
[32m+[m
[32m+[m[32m    Dönüş:[m
[32m+[m[32m        İki kanallı (N, 2) float32 sinyal.[m
[32m+[m[32m    """[m
[32m+[m[32m    x = _to_float32(signal)[m
[32m+[m[32m    amt = _clamp(amount, 0.0, 1.0)[m
[32m+[m[32m    g = _FT(0.6 * amt)[m
[32m+[m[32m    d1 = 11[m
[32m+[m[32m    d2 = 7[m
[32m+[m[32m    n = x.shape[0][m
[32m+[m[32m    l = np.empty((n,), dtype=_FT)[m
[32m+[m[32m    r = np.empty((n,), dtype=_FT)[m
[32m+[m[32m    buf1 = np.zeros(d1, dtype=_FT)[m
[32m+[m[32m    buf2 = np.zeros(d2, dtype=_FT)[m
[32m+[m[32m    i1 = 0[m
[32m+[m[32m    i2 = 0[m
[32m+[m[32m    for i, sample in enumerate(x):[m
[32m+[m[32m        ap1 = -g * sample + buf1[i1] + g * buf1[i1 - 1] if d1 > 1 else sample[m
[32m+[m[32m        ap2 = -g * sample + buf2[i2] + g * buf2[i2 - 1] if d2 > 1 else sample[m
[32m+[m[32m        buf1[i1] = sample + g * ap1 + _DENORM_GUARD[m
[32m+[m[32m        buf2[i2] = sample + g * ap2 + _DENORM_GUARD[m
[32m+[m[32m        i1 = (i1 + 1) % d1[m
[32m+[m[32m        i2 = (i2 + 1) % d2[m
[32m+[m[32m        side = _FT(0.5) * (ap1 - ap2)[m
[32m+[m[32m        l[i] = sample + side[m
[32m+[m[32m        r[i] = sample - side[m
[32m+[m[32m    return np.stack((l, r), axis=-1)[m
[32m+[m
[32m+[m
[32m+[m[32mdef normalize_gain(signal: FloatArray, target_rms: float = 0.1) -> FloatArray:[m
[32m+[m[32m    """[m
[32m+[m[32m    Sinyali hedef RMS değerine ölçekler.[m
[32m+[m
[32m+[m[32m    Parametreler:[m
[32m+[m[32m        signal: Mono veya stereo sinyal (float32).[m
[32m+[m[32m        target_rms: İstenen RMS değeri.[m
[32m+[m
[32m+[m[32m    Dönüş:[m
[32m+[m[32m        Ölçeklenmiş float32 sinyal.[m
[32m+[m[32m    """[m
[32m+[m[32m    x = _to_float32(signal)[m
[32m+[m[32m    rms_val = _rms(x)[m
[32m+[m[32m    if rms_val <= 0.0:[m
[32m+[m[32m        return x.copy()[m
[32m+[m[32m    gain = _FT(target_rms / rms_val)[m
[32m+[m[32m    return (x * gain).astype(_FT, copy=False)[m
[32m+[m
[32m+[m
[32m+[m[32mif __name__ == "__main__":[m
[32m+[m[32m    fs = 48000[m
[32m+[m[32m    t = np.arange(fs, dtype=_FT) / _FT(fs)[m
[32m+[m[32m    test = _FT(0.3) * np.sin(2.0 * math.pi * 440.0 * t, dtype=_FT)[m
[32m+[m
[32m+[m[32m    sat = soft_saturation(test, drive=1.5)[m
[32m+[m[32m    warm = warmth(test, amount=0.7)[m
[32m+[m[32m    rev = simple_reverb(test, sample_rate=fs, mix=0.25)[m
[32m+[m[32m    wide = stereo_widen(test, amount=0.6)[m
[32m+[m[32m    norm = normalize_gain(test, target_rms=0.1)[m
[32m+[m
[32m+[m[32m    print("Örnek RMS değerleri:")[m
[32m+[m[32m    print(f"Giriş: {_rms(test):.6f}")[m
[32m+[m[32m    print(f"Saturasyon: {_rms(sat):.6f}")[m
[32m+[m[32m    print(f"Warmth: {_rms(warm):.6f}")[m
[32m+[m[32m    print(f"Reverb: {_rms(rev):.6f}")[m
[32m+[m[32m    print(f"Widen L: {_rms(wide[:,0]):.6f} R: {_rms(wide[:,1]):.6f}")[m
[32m+[m[32m    print(f"Normalize: {_rms(norm):.6f}")[m
[1mdiff --git a/core_dsp/dsp_fx_v2.py b/core_dsp/dsp_fx_v2.py[m
[1mnew file mode 100644[m
[1mindex 0000000..918c2aa[m
[1m--- /dev/null[m
[1m+++ b/core_dsp/dsp_fx_v2.py[m
[36m@@ -0,0 +1,402 @@[m
[32m+[m[32mimport math[m
[32m+[m[32mfrom typing import Tuple[m
[32m+[m
[32m+[m[32mimport numpy as np[m
[32m+[m
[32m+[m[32mFloatArray = np.ndarray[m
[32m+[m[32m_FT = np.float32[m
[32m+[m[32m_DENORM_GUARD = _FT(1e-20)[m
[32m+[m
[32m+[m
[32m+[m[32mdef _to_float32(signal: FloatArray) -> FloatArray:[m
[32m+[m[32m    """Girişi float32 kopyasına dönüştürür."""[m
[32m+[m[32m    if signal.dtype != _FT:[m
[32m+[m[32m        return signal.astype(_FT, copy=False)[m
[32m+[m[32m    return signal[m
[32m+[m
[32m+[m
[32m+[m[32mdef _safe_cutoff(cutoff_hz: float, sample_rate: int) -> float:[m
[32m+[m[32m    """Kesim frekansını Nyquist sınırında güvenli aralığa sıkıştırır."""[m
[32m+[m[32m    nyquist = sample_rate * 0.5[m
[32m+[m[32m    return float(np.clip(cutoff_hz, 1.0, nyquist * 0.95))[m
[32m+[m
[32m+[m
[32m+[m[32mdef _biquad_filter([m
[32m+[m[32m    signal: FloatArray,[m
[32m+[m[32m    b0: _FT,[m
[32m+[m[32m    b1: _FT,[m
[32m+[m[32m    b2: _FT,[m
[32m+[m[32m    a1: _FT,[m
[32m+[m[32m    a2: _FT,[m
[32m+[m[32m) -> FloatArray:[m
[32m+[m[32m    """Biquad fark denklemini uygular."""[m
[32m+[m[32m    x = _to_float32(signal)[m
[32m+[m[32m    y = np.empty_like(x)[m
[32m+[m[32m    x1 = _FT(0.0)[m
[32m+[m[32m    x2 = _FT(0.0)[m
[32m+[m[32m    y1 = _FT(0.0)[m
[32m+[m[32m    y2 = _FT(0.0)[m
[32m+[m[32m    for i, sample in enumerate(x):[m
[32m+[m[32m        y0 = ([m
[32m+[m[32m            b0 * sample[m
[32m+[m[32m            + b1 * x1[m
[32m+[m[32m            + b2 * x2[m
[32m+[m[32m            - a1 * y1[m
[32m+[m[32m            - a2 * y2[m
[32m+[m[32m            + _DENORM_GUARD[m
[32m+[m[32m        )[m
[32m+[m[32m        y[i] = y0[m
[32m+[m[32m        x2 = x1[m
[32m+[m[32m        x1 = sample[m
[32m+[m[32m        y2 = y1[m
[32m+[m[32m        y1 = y0[m
[32m+[m[32m    return y[m
[32m+[m
[32m+[m
[32m+[m[32mdef _biquad_coeffs_peaking([m
[32m+[m[32m    freq_hz: float, q: float, gain_db: float, sample_rate: int[m
[32m+[m[32m) -> Tuple[_FT, _FT, _FT, _FT, _FT]:[m
[32m+[m[32m    """RBJ peaking biquad katsayılarını döndürür."""[m
[32m+[m[32m    a = math.pow(10.0, gain_db / 40.0)[m
[32m+[m[32m    w0 = 2.0 * math.pi * freq_hz / sample_rate[m
[32m+[m[32m    alpha = math.sin(w0) / (2.0 * max(q, 1e-6))[m
[32m+[m[32m    cos_w0 = math.cos(w0)[m
[32m+[m
[32m+[m[32m    b0 = 1.0 + alpha * a[m
[32m+[m[32m    b1 = -2.0 * cos_w0[m
[32m+[m[32m    b2 = 1.0 - alpha * a[m
[32m+[m[32m    a0 = 1.0 + alpha / a[m
[32m+[m[32m    a1 = -2.0 * cos_w0[m
[32m+[m[32m    a2 = 1.0 - alpha / a[m
[32m+[m
[32m+[m[32m    return ([m
[32m+[m[32m        _FT(b0 / a0),[m
[32m+[m[32m        _FT(b1 / a0),[m
[32m+[m[32m        _FT(b2 / a0),[m
[32m+[m[32m        _FT(a1 / a0),[m
[32m+[m[32m        _FT(a2 / a0),[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m
[32m+[m[32mdef _biquad_coeffs_shelf([m
[32m+[m[32m    freq_hz: float,[m
[32m+[m[32m    slope: float,[m
[32m+[m[32m    gain_db: float,[m
[32m+[m[32m    sample_rate: int,[m
[32m+[m[32m    shelf_type: str,[m
[32m+[m[32m) -> Tuple[_FT, _FT, _FT, _FT, _FT]:[m
[32m+[m[32m    """RBJ low/high shelf biquad katsayılarını döndürür."""[m
[32m+[m[32m    a = math.pow(10.0, gain_db / 40.0)[m
[32m+[m[32m    w0 = 2.0 * math.pi * freq_hz / sample_rate[m
[32m+[m[32m    cos_w0 = math.cos(w0)[m
[32m+[m[32m    sin_w0 = math.sin(w0)[m
[32m+[m[32m    alpha = sin_w0 / 2.0 * math.sqrt([m
[32m+[m[32m        max((a + 1.0 / a) * (1.0 / max(slope, 1e-6) - 1.0) + 2.0, 0.0)[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m[32m    if shelf_type == "low":[m
[32m+[m[32m        b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * math.sqrt(a) * alpha)[m
[32m+[m[32m        b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0)[m
[32m+[m[32m        b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * math.sqrt(a) * alpha)[m
[32m+[m[32m        a0 = (a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * math.sqrt(a) * alpha[m
[32m+[m[32m        a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0)[m
[32m+[m[32m        a2 = (a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * math.sqrt(a) * alpha[m
[32m+[m[32m    else:[m
[32m+[m[32m        b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * math.sqrt(a) * alpha)[m
[32m+[m[32m        b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0)[m
[32m+[m[32m        b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * math.sqrt(a) * alpha)[m
[32m+[m[32m        a0 = (a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * math.sqrt(a) * alpha[m
[32m+[m[32m        a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0)[m
[32m+[m[32m        a2 = (a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * math.sqrt(a) * alpha[m
[32m+[m
[32m+[m[32m    return ([m
[32m+[m[32m        _FT(b0 / a0),[m
[32m+[m[32m        _FT(b1 / a0),[m
[32m+[m[32m        _FT(b2 / a0),[m
[32m+[m[32m        _FT(a1 / a0),[m
[32m+[m[32m        _FT(a2 / a0),[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m
[32m+[m[32mdef dc_block(signal: FloatArray, pole: float = 0.995) -> FloatArray:[m
[32m+[m[32m    """[m
[32m+[m[32m    DC bileşenini bastıran birinci dereceden yüksek geçiren filtre uygular.[m
[32m+[m
[32m+[m[32m    Parametreler:[m
[32m+[m[32m        signal: Mono giriş sinyali (float32).[m
[32m+[m[32m        pole: 1.0'a yaklaştıkça kesim düşer, tipik 0.995.[m
[32m+[m
[32m+[m[32m    Dönüş:[m
[32m+[m[32m        DC'si bastırılmış float32 sinyal.[m
[32m+[m[32m    """[m
[32m+[m[32m    x = _to_float32(signal)[m
[32m+[m[32m    y = np.empty_like(x)[m
[32m+[m[32m    pole_f = _FT(pole)[m
[32m+[m[32m    x_prev = _FT(0.0)[m
[32m+[m[32m    y_prev = _FT(0.0)[m
[32m+[m[32m    for i, sample in enumerate(x):[m
[32m+[m[32m        y_curr = sample - x_prev + pole_f * y_prev + _DENORM_GUARD[m
[32m+[m[32m        y[i] = y_curr[m
[32m+[m[32m        x_prev = sample[m
[32m+[m[32m        y_prev = y_curr[m
[32m+[m[32m    return y[m
[32m+[m
[32m+[m
[32m+[m[32mdef one_pole_lowpass([m
[32m+[m[32m    signal: FloatArray, cutoff_hz: float, sample_rate: int[m
[32m+[m[32m) -> FloatArray:[m
[32m+[m[32m    """[m
[32m+[m[32m    Birinci dereceden düşük geçiren filtre uygular.[m
[32m+[m
[32m+[m[32m    Parametreler:[m
[32m+[m[32m        signal: Mono giriş sinyali (float32).[m
[32m+[m[32m        cutoff_hz: Kesim frekansı (Hz).[m
[32m+[m[32m        sample_rate: Örnekleme hızı (Hz).[m
[32m+[m
[32m+[m[32m    Dönüş:[m
[32m+[m[32m        Low-pass uygulanmış float32 sinyal.[m
[32m+[m[32m    """[m
[32m+[m[32m    x = _to_float32(signal)[m
[32m+[m[32m    y = np.empty_like(x)[m
[32m+[m[32m    cutoff = _safe_cutoff(cutoff_hz, sample_rate)[m
[32m+[m[32m    alpha = _FT(math.exp(-2.0 * math.pi * cutoff / sample_rate))[m
[32m+[m[32m    b0 = _FT(1.0) - alpha[m
[32m+[m[32m    y_prev = _FT(0.0)[m
[32m+[m[32m    for i, sample in enumerate(x):[m
[32m+[m[32m        y_curr = b0 * sample + alpha * y_prev + _DENORM_GUARD[m
[32m+[m[32m        y[i] = y_curr[m
[32m+[m[32m        y_prev = y_curr[m
[32m+[m[32m    return y[m
[32m+[m
[32m+[m
[32m+[m[32mdef one_pole_highpass([m
[32m+[m[32m    signal: FloatArray, cutoff_hz: float, sample_rate: int[m
[32m+[m[32m) -> FloatArray:[m
[32m+[m[32m    """[m
[32m+[m[32m    Birinci dereceden yüksek geçiren filtre uygular.[m
[32m+[m
[32m+[m[32m    Parametreler:[m
[32m+[m[32m        signal: Mono giriş sinyali (float32).[m
[32m+[m[32m        cutoff_hz: Kesim frekansı (Hz).[m
[32m+[m[32m        sample_rate: Örnekleme hızı (Hz).[m
[32m+[m
[32m+[m[32m    Dönüş:[m
[32m+[m[32m        High-pass uygulanmış float32 sinyal.[m
[32m+[m[32m    """[m
[32m+[m[32m    x = _to_float32(signal)[m
[32m+[m[32m    y = np.empty_like(x)[m
[32m+[m[32m    cutoff = _safe_cutoff(cutoff_hz, sample_rate)[m
[32m+[m[32m    alpha = _FT(math.exp(-2.0 * math.pi * cutoff / sample_rate))[m
[32m+[m[32m    x_prev = _FT(0.0)[m
[32m+[m[32m    y_prev = _FT(0.0)[m
[32m+[m[32m    for i, sample in enumerate(x):[m
[32m+[m[32m        y_curr = alpha * (y_prev + sample - x_prev) + _DENORM_GUARD[m
[32m+[m[32m        y[i] = y_curr[m
[32m+[m[32m        y_prev = y_curr[m
[32m+[m[32m        x_prev = sample[m
[32m+[m[32m    return y[m
[32m+[m
[32m+[m
[32m+[m[32mdef peaking_eq([m
[32m+[m[32m    signal: FloatArray,[m
[32m+[m[32m    freq_hz: float,[m
[32m+[m[32m    q: float,[m
[32m+[m[32m    gain_db: float,[m
[32m+[m[32m    sample_rate: int,[m
[32m+[m[32m) -> FloatArray:[m
[32m+[m[32m    """[m
[32m+[m[32m    RBJ peaking eşitleyici uygular.[m
[32m+[m
[32m+[m[32m    Parametreler:[m
[32m+[m[32m        signal: Mono giriş sinyali (float32).[m
[32m+[m[32m        freq_hz: Merkez frekans (Hz).[m
[32m+[m[32m        q: Q değeri (boyutsuz).[m
[32m+[m[32m        gain_db: Kazanç (dB).[m
[32m+[m[32m        sample_rate: Örnekleme hızı (Hz).[m
[32m+[m
[32m+[m[32m    Dönüş:[m
[32m+[m[32m        Peaking EQ uygulanmış float32 sinyal.[m
[32m+[m[32m    """[m
[32m+[m[32m    freq = _safe_cutoff(freq_hz, sample_rate)[m
[32m+[m[32m    b0, b1, b2, a1, a2 = _biquad_coeffs_peaking(freq, q, gain_db, sample_rate)[m
[32m+[m[32m    return _biquad_filter(signal, b0, b1, b2, a1, a2)[m
[32m+[m
[32m+[m
[32m+[m[32mdef low_shelf([m
[32m+[m[32m    signal: FloatArray,[m
[32m+[m[32m    freq_hz: float,[m
[32m+[m[32m    slope: float,[m
[32m+[m[32m    gain_db: float,[m
[32m+[m[32m    sample_rate: int,[m
[32m+[m[32m) -> FloatArray:[m
[32m+[m[32m    """[m
[32m+[m[32m    RBJ düşük raf (low-shelf) eşitleyici uygular.[m
[32m+[m
[32m+[m[32m    Parametreler:[m
[32m+[m[32m        signal: Mono giriş sinyali (float32).[m
[32m+[m[32m        freq_hz: Raf geçiş frekansı (Hz).[m
[32m+[m[32m        slope: Raf eğimi (boyutsuz, 0-1 arası tipik).[m
[32m+[m[32m        gain_db: Kazanç (dB).[m
[32m+[m[32m        sample_rate: Örnekleme hızı (Hz).[m
[32m+[m
[32m+[m[32m    Dönüş:[m
[32m+[m[32m        Low-shelf uygulanmış float32 sinyal.[m
[32m+[m[32m    """[m
[32m+[m[32m    freq = _safe_cutoff(freq_hz, sample_rate)[m
[32m+[m[32m    b0, b1, b2, a1, a2 = _biquad_coeffs_shelf([m
[32m+[m[32m        freq, max(slope, 1e-4), gain_db, sample_rate, shelf_type="low"[m
[32m+[m[32m    )[m
[32m+[m[32m    return _biquad_filter(signal, b0, b1, b2, a1, a2)[m
[32m+[m
[32m+[m
[32m+[m[32mdef high_shelf([m
[32m+[m[32m    signal: FloatArray,[m
[32m+[m[32m    freq_hz: float,[m
[32m+[m[32m    slope: float,[m
[32m+[m[32m    gain_db: float,[m
[32m+[m[32m    sample_rate: int,[m
[32m+[m[32m) -> FloatArray:[m
[32m+[m[32m    """[m
[32m+[m[32m    RBJ yüksek raf (high-shelf) eşitleyici uygular.[m
[32m+[m
[32m+[m[32m    Parametreler:[m
[32m+[m[32m        signal: Mono giriş sinyali (float32).[m
[32m+[m[32m        freq_hz: Raf geçiş frekansı (Hz).[m
[32m+[m[32m        slope: Raf eğimi (boyutsuz, 0-1 arası tipik).[m
[32m+[m[32m        gain_db: Kazanç (dB).[m
[32m+[m[32m        sample_rate: Örnekleme hızı (Hz).[m
[32m+[m
[32m+[m[32m    Dönüş:[m
[32m+[m[32m        High-shelf uygulanmış float32 sinyal.[m
[32m+[m[32m    """[m
[32m+[m[32m    freq = _safe_cutoff(freq_hz, sample_rate)[m
[32m+[m[32m    b0, b1, b2, a1, a2 = _biquad_coeffs_shelf([m
[32m+[m[32m        freq, max(slope, 1e-4), gain_db, sample_rate, shelf_type="high"[m
[32m+[m[32m    )[m
[32m+[m[32m    return _biquad_filter(signal, b0, b1, b2, a1, a2)[m
[32m+[m
[32m+[m
[32m+[m[32mdef tilt_filter([m
[32m+[m[32m    signal: FloatArray, pivot_hz: float, tilt_db: float, sample_rate: int[m
[32m+[m[32m) -> FloatArray:[m
[32m+[m[32m    """[m
[32m+[m[32m    Basit tilt eşitleyici uygular; pivot çevresinde eğim verir.[m
[32m+[m
[32m+[m[32m    Parametreler:[m
[32m+[m[32m        signal: Mono giriş sinyali (float32).[m
[32m+[m[32m        pivot_hz: Eğimin dönüm noktası (Hz).[m
[32m+[m[32m        tilt_db: Üst frekanslar için dB artışı (negatifse azaltma).[m
[32m+[m[32m        sample_rate: Örnekleme hızı (Hz).[m
[32m+[m
[32m+[m[32m    Dönüş:[m
[32m+[m[32m        Tilt uygulanmış float32 sinyal.[m
[32m+[m[32m    """[m
[32m+[m[32m    freq = _safe_cutoff(pivot_hz, sample_rate)[m
[32m+[m[32m    x = _to_float32(signal)[m
[32m+[m[32m    low = one_pole_lowpass(x, freq, sample_rate)[m
[32m+[m[32m    high = x - low[m
[32m+[m[32m    g = math.pow(10.0, tilt_db / 20.0)[m
[32m+[m[32m    low_gain = _FT(1.0 / math.sqrt(g))[m
[32m+[m[32m    high_gain = _FT(math.sqrt(g))[m
[32m+[m[32m    norm = _FT(1.0 / (abs(low_gain) + abs(high_gain)))[m
[32m+[m[32m    low_gain *= norm[m
[32m+[m[32m    high_gain *= norm[m
[32m+[m[32m    return low_gain * low + high_gain * high[m
[32m+[m
[32m+[m
[32m+[m[32mdef butterworth_lowpass([m
[32m+[m[32m    signal: FloatArray, cutoff_hz: float, sample_rate: int[m
[32m+[m[32m) -> FloatArray:[m
[32m+[m[32m    """[m
[32m+[m[32m    İkinci dereceden Butterworth düşük geçiren filtre uygular.[m
[32m+[m
[32m+[m[32m    Parametreler:[m
[32m+[m[32m        signal: Mono giriş sinyali (float32).[m
[32m+[m[32m        cutoff_hz: Kesim frekansı (Hz).[m
[32m+[m[32m        sample_rate: Örnekleme hızı (Hz).[m
[32m+[m
[32m+[m[32m    Dönüş:[m
[32m+[m[32m        Butterworth low-pass uygulanmış float32 sinyal.[m
[32m+[m[32m    """[m
[32m+[m[32m    freq = _safe_cutoff(cutoff_hz, sample_rate)[m
[32m+[m[32m    w0 = 2.0 * math.pi * freq / sample_rate[m
[32m+[m[32m    cos_w0 = math.cos(w0)[m
[32m+[m[32m    sin_w0 = math.sin(w0)[m
[32m+[m[32m    alpha = sin_w0 / math.sqrt(2.0)[m
[32m+[m[32m    b0 = (1.0 - cos_w0) * 0.5[m
[32m+[m[32m    b1 = 1.0 - cos_w0[m
[32m+[m[32m    b2 = (1.0 - cos_w0) * 0.5[m
[32m+[m[32m    a0 = 1.0 + alpha[m
[32m+[m[32m    a1 = -2.0 * cos_w0[m
[32m+[m[32m    a2 = 1.0 - alpha[m
[32m+[m[32m    return _biquad_filter([m
[32m+[m[32m        signal,[m
[32m+[m[32m        _FT(b0 / a0),[m
[32m+[m[32m        _FT(b1 / a0),[m
[32m+[m[32m        _FT(b2 / a0),[m
[32m+[m[32m        _FT(a1 / a0),[m
[32m+[m[32m        _FT(a2 / a0),[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m
[32m+[m[32mdef butterworth_highpass([m
[32m+[m[32m    signal: FloatArray, cutoff_hz: float, sample_rate: int[m
[32m+[m[32m) -> FloatArray:[m
[32m+[m[32m    """[m
[32m+[m[32m    İkinci dereceden Butterworth yüksek geçiren fil