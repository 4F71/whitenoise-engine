import math
from typing import Tuple

import numpy as np

FloatArray = np.ndarray
_FT = np.float32
_DENORM_GUARD = _FT(1e-20)


def _to_float32(signal: FloatArray) -> FloatArray:
    """Girişi float32 kopyasına dönüştürür."""
    if signal.dtype != _FT:
        return signal.astype(_FT, copy=False)
    return signal


def _safe_cutoff(cutoff_hz: float, sample_rate: int) -> float:
    """Kesim frekansını Nyquist sınırında güvenli aralığa sıkıştırır."""
    nyquist = sample_rate * 0.5
    return float(np.clip(cutoff_hz, 1.0, nyquist * 0.95))


def _biquad_filter(
    signal: FloatArray,
    b0: _FT,
    b1: _FT,
    b2: _FT,
    a1: _FT,
    a2: _FT,
) -> FloatArray:
    """Biquad fark denklemini uygular."""
    x = _to_float32(signal)
    y = np.empty_like(x)
    x1 = _FT(0.0)
    x2 = _FT(0.0)
    y1 = _FT(0.0)
    y2 = _FT(0.0)
    for i, sample in enumerate(x):
        y0 = (
            b0 * sample
            + b1 * x1
            + b2 * x2
            - a1 * y1
            - a2 * y2
            + _DENORM_GUARD
        )
        y[i] = y0
        x2 = x1
        x1 = sample
        y2 = y1
        y1 = y0
    return y


def _biquad_coeffs_peaking(
    freq_hz: float, q: float, gain_db: float, sample_rate: int
) -> Tuple[_FT, _FT, _FT, _FT, _FT]:
    """RBJ peaking biquad katsayılarını döndürür."""
    a = math.pow(10.0, gain_db / 40.0)
    w0 = 2.0 * math.pi * freq_hz / sample_rate
    alpha = math.sin(w0) / (2.0 * max(q, 1e-6))
    cos_w0 = math.cos(w0)

    b0 = 1.0 + alpha * a
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / a

    return (
        _FT(b0 / a0),
        _FT(b1 / a0),
        _FT(b2 / a0),
        _FT(a1 / a0),
        _FT(a2 / a0),
    )


def _biquad_coeffs_shelf(
    freq_hz: float,
    slope: float,
    gain_db: float,
    sample_rate: int,
    shelf_type: str,
) -> Tuple[_FT, _FT, _FT, _FT, _FT]:
    """RBJ low/high shelf biquad katsayılarını döndürür."""
    a = math.pow(10.0, gain_db / 40.0)
    w0 = 2.0 * math.pi * freq_hz / sample_rate
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / 2.0 * math.sqrt(
        max((a + 1.0 / a) * (1.0 / max(slope, 1e-6) - 1.0) + 2.0, 0.0)
    )

    if shelf_type == "low":
        b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * math.sqrt(a) * alpha)
        b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0)
        b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * math.sqrt(a) * alpha)
        a0 = (a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * math.sqrt(a) * alpha
        a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0)
        a2 = (a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * math.sqrt(a) * alpha
    else:
        b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * math.sqrt(a) * alpha)
        b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0)
        b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * math.sqrt(a) * alpha)
        a0 = (a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * math.sqrt(a) * alpha
        a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0)
        a2 = (a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * math.sqrt(a) * alpha

    return (
        _FT(b0 / a0),
        _FT(b1 / a0),
        _FT(b2 / a0),
        _FT(a1 / a0),
        _FT(a2 / a0),
    )


def dc_block(signal: FloatArray, pole: float = 0.995) -> FloatArray:
    """
    DC bileşenini bastıran birinci dereceden yüksek geçiren filtre uygular.

    Parametreler:
        signal: Mono giriş sinyali (float32).
        pole: 1.0'a yaklaştıkça kesim düşer, tipik 0.995.

    Dönüş:
        DC'si bastırılmış float32 sinyal.
    """
    x = _to_float32(signal)
    y = np.empty_like(x)
    pole_f = _FT(pole)
    x_prev = _FT(0.0)
    y_prev = _FT(0.0)
    for i, sample in enumerate(x):
        y_curr = sample - x_prev + pole_f * y_prev + _DENORM_GUARD
        y[i] = y_curr
        x_prev = sample
        y_prev = y_curr
    return y


def one_pole_lowpass(
    signal: FloatArray, cutoff_hz: float, sample_rate: int
) -> FloatArray:
    """
    Birinci dereceden düşük geçiren filtre uygular.

    Parametreler:
        signal: Mono giriş sinyali (float32).
        cutoff_hz: Kesim frekansı (Hz).
        sample_rate: Örnekleme hızı (Hz).

    Dönüş:
        Low-pass uygulanmış float32 sinyal.
    """
    x = _to_float32(signal)
    y = np.empty_like(x)
    cutoff = _safe_cutoff(cutoff_hz, sample_rate)
    alpha = _FT(math.exp(-2.0 * math.pi * cutoff / sample_rate))
    b0 = _FT(1.0) - alpha
    y_prev = _FT(0.0)
    for i, sample in enumerate(x):
        y_curr = b0 * sample + alpha * y_prev + _DENORM_GUARD
        y[i] = y_curr
        y_prev = y_curr
    return y


def one_pole_highpass(
    signal: FloatArray, cutoff_hz: float, sample_rate: int
) -> FloatArray:
    """
    Birinci dereceden yüksek geçiren filtre uygular.

    Parametreler:
        signal: Mono giriş sinyali (float32).
        cutoff_hz: Kesim frekansı (Hz).
        sample_rate: Örnekleme hızı (Hz).

    Dönüş:
        High-pass uygulanmış float32 sinyal.
    """
    x = _to_float32(signal)
    y = np.empty_like(x)
    cutoff = _safe_cutoff(cutoff_hz, sample_rate)
    alpha = _FT(math.exp(-2.0 * math.pi * cutoff / sample_rate))
    x_prev = _FT(0.0)
    y_prev = _FT(0.0)
    for i, sample in enumerate(x):
        y_curr = alpha * (y_prev + sample - x_prev) + _DENORM_GUARD
        y[i] = y_curr
        y_prev = y_curr
        x_prev = sample
    return y


def peaking_eq(
    signal: FloatArray,
    freq_hz: float,
    q: float,
    gain_db: float,
    sample_rate: int,
) -> FloatArray:
    """
    RBJ peaking eşitleyici uygular.

    Parametreler:
        signal: Mono giriş sinyali (float32).
        freq_hz: Merkez frekans (Hz).
        q: Q değeri (boyutsuz).
        gain_db: Kazanç (dB).
        sample_rate: Örnekleme hızı (Hz).

    Dönüş:
        Peaking EQ uygulanmış float32 sinyal.
    """
    freq = _safe_cutoff(freq_hz, sample_rate)
    b0, b1, b2, a1, a2 = _biquad_coeffs_peaking(freq, q, gain_db, sample_rate)
    return _biquad_filter(signal, b0, b1, b2, a1, a2)


def low_shelf(
    signal: FloatArray,
    freq_hz: float,
    slope: float,
    gain_db: float,
    sample_rate: int,
) -> FloatArray:
    """
    RBJ düşük raf (low-shelf) eşitleyici uygular.

    Parametreler:
        signal: Mono giriş sinyali (float32).
        freq_hz: Raf geçiş frekansı (Hz).
        slope: Raf eğimi (boyutsuz, 0-1 arası tipik).
        gain_db: Kazanç (dB).
        sample_rate: Örnekleme hızı (Hz).

    Dönüş:
        Low-shelf uygulanmış float32 sinyal.
    """
    freq = _safe_cutoff(freq_hz, sample_rate)
    b0, b1, b2, a1, a2 = _biquad_coeffs_shelf(
        freq, max(slope, 1e-4), gain_db, sample_rate, shelf_type="low"
    )
    return _biquad_filter(signal, b0, b1, b2, a1, a2)


def high_shelf(
    signal: FloatArray,
    freq_hz: float,
    slope: float,
    gain_db: float,
    sample_rate: int,
) -> FloatArray:
    """
    RBJ yüksek raf (high-shelf) eşitleyici uygular.

    Parametreler:
        signal: Mono giriş sinyali (float32).
        freq_hz: Raf geçiş frekansı (Hz).
        slope: Raf eğimi (boyutsuz, 0-1 arası tipik).
        gain_db: Kazanç (dB).
        sample_rate: Örnekleme hızı (Hz).

    Dönüş:
        High-shelf uygulanmış float32 sinyal.
    """
    freq = _safe_cutoff(freq_hz, sample_rate)
    b0, b1, b2, a1, a2 = _biquad_coeffs_shelf(
        freq, max(slope, 1e-4), gain_db, sample_rate, shelf_type="high"
    )
    return _biquad_filter(signal, b0, b1, b2, a1, a2)


def tilt_filter(
    signal: FloatArray, pivot_hz: float, tilt_db: float, sample_rate: int
) -> FloatArray:
    """
    Basit tilt eşitleyici uygular; pivot çevresinde eğim verir.

    Parametreler:
        signal: Mono giriş sinyali (float32).
        pivot_hz: Eğimin dönüm noktası (Hz).
        tilt_db: Üst frekanslar için dB artışı (negatifse azaltma).
        sample_rate: Örnekleme hızı (Hz).

    Dönüş:
        Tilt uygulanmış float32 sinyal.
    """
    freq = _safe_cutoff(pivot_hz, sample_rate)
    x = _to_float32(signal)
    low = one_pole_lowpass(x, freq, sample_rate)
    high = x - low
    g = math.pow(10.0, tilt_db / 20.0)
    low_gain = _FT(1.0 / math.sqrt(g))
    high_gain = _FT(math.sqrt(g))
    norm = _FT(1.0 / (abs(low_gain) + abs(high_gain)))
    low_gain *= norm
    high_gain *= norm
    return low_gain * low + high_gain * high


def butterworth_lowpass(
    signal: FloatArray, cutoff_hz: float, sample_rate: int
) -> FloatArray:
    """
    İkinci dereceden Butterworth düşük geçiren filtre uygular.

    Parametreler:
        signal: Mono giriş sinyali (float32).
        cutoff_hz: Kesim frekansı (Hz).
        sample_rate: Örnekleme hızı (Hz).

    Dönüş:
        Butterworth low-pass uygulanmış float32 sinyal.
    """
    freq = _safe_cutoff(cutoff_hz, sample_rate)
    w0 = 2.0 * math.pi * freq / sample_rate
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / math.sqrt(2.0)
    b0 = (1.0 - cos_w0) * 0.5
    b1 = 1.0 - cos_w0
    b2 = (1.0 - cos_w0) * 0.5
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return _biquad_filter(
        signal,
        _FT(b0 / a0),
        _FT(b1 / a0),
        _FT(b2 / a0),
        _FT(a1 / a0),
        _FT(a2 / a0),
    )


def butterworth_highpass(
    signal: FloatArray, cutoff_hz: float, sample_rate: int
) -> FloatArray:
    """
    İkinci dereceden Butterworth yüksek geçiren filtre uygular.

    Parametreler:
        signal: Mono giriş sinyali (float32).
        cutoff_hz: Kesim frekansı (Hz).
        sample_rate: Örnekleme hızı (Hz).

    Dönüş:
        Butterworth high-pass uygulanmış float32 sinyal.
    """
    freq = _safe_cutoff(cutoff_hz, sample_rate)
    w0 = 2.0 * math.pi * freq / sample_rate
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / math.sqrt(2.0)
    b0 = (1.0 + cos_w0) * 0.5
    b1 = -(1.0 + cos_w0)
    b2 = (1.0 + cos_w0) * 0.5
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return _biquad_filter(
        signal,
        _FT(b0 / a0),
        _FT(b1 / a0),
        _FT(b2 / a0),
        _FT(a1 / a0),
        _FT(a2 / a0),
    )


def _rms(signal: FloatArray) -> float:
    """RMS değerini float olarak döndürür."""
    x = _to_float32(signal)
    return float(np.sqrt(np.mean(x * x, dtype=_FT)))


if __name__ == "__main__":
    fs = 48000
    t = np.arange(fs, dtype=_FT) / _FT(fs)
    sine = np.sin(2.0 * math.pi * 440.0 * t, dtype=_FT) * _FT(0.5)

    lp = one_pole_lowpass(sine, cutoff_hz=2000.0, sample_rate=fs)
    hp = one_pole_highpass(sine, cutoff_hz=200.0, sample_rate=fs)
    peak = peaking_eq(sine, freq_hz=1000.0, q=1.0, gain_db=3.0, sample_rate=fs)
    tilt = tilt_filter(sine, pivot_hz=800.0, tilt_db=4.0, sample_rate=fs)
    bw_lp = butterworth_lowpass(sine, cutoff_hz=1500.0, sample_rate=fs)
    bw_hp = butterworth_highpass(sine, cutoff_hz=300.0, sample_rate=fs)

    print("Test sinyalleri RMS:")
    print(f"Orijinal: {_rms(sine):.6f}")
    print(f"LP: {_rms(lp):.6f}")
    print(f"HP: {_rms(hp):.6f}")
    print(f"Peaking: {_rms(peak):.6f}")
    print(f"Tilt: {_rms(tilt):.6f}")
    print(f"BW LP: {_rms(bw_lp):.6f}")
    print(f"BW HP: {_rms(bw_hp):.6f}")
