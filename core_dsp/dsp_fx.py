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


def _clamp(value: float, min_value: float, max_value: float) -> float:
    """Değeri verilen aralıkta sınırlar."""
    return float(np.clip(value, min_value, max_value))


def _rms(signal: FloatArray) -> float:
    """RMS değerini float olarak döndürür."""
    x = _to_float32(signal)
    return float(np.sqrt(np.mean(x * x, dtype=_FT)))


def soft_saturation(signal: FloatArray, drive: float) -> FloatArray:
    """
    Arctan tabanlı yumuşak saturasyon uygular.

    Parametreler:
        signal: Mono giriş sinyali (float32).
        drive: Sürüş miktarı (0 ve üzeri).

    Dönüş:
        Saturasyon uygulanmış float32 sinyal.
    """
    x = _to_float32(signal)
    drv = _clamp(drive, 0.0, 10.0)
    gain = _FT(1.0 + drv * 2.0)
    out = (_FT(2.0 / math.pi)) * np.arctan(gain * x + _DENORM_GUARD, dtype=_FT)
    comp = _FT(1.0 / (1.0 + 0.5 * drv))
    return (out * comp).astype(_FT, copy=False)


def warmth(signal: FloatArray, amount: float) -> FloatArray:
    """
    Hafif yumuşatma ve düşük seviye harmonik ekler.

    Parametreler:
        signal: Mono giriş sinyali (float32).
        amount: 0-1 arası ısınma miktarı.

    Dönüş:
        Isıtılmış float32 sinyal.
    """
    x = _to_float32(signal)
    amt = _clamp(amount, 0.0, 1.0)
    alpha = _FT(0.15 + 0.7 * amt)  # düşük geçiren yumuşatma katsayısı
    y = np.empty_like(x)
    lp_prev = _FT(0.0)
    for i, sample in enumerate(x):
        lp = (1.0 - alpha) * sample + alpha * lp_prev + _DENORM_GUARD
        lp_prev = lp
        harm = sample * sample * sample
        warmed = (1.0 - amt) * sample + amt * (_FT(0.75) * sample + _FT(0.2) * lp + _FT(0.05) * harm)
        y[i] = warmed
    return y.astype(_FT, copy=False)


def simple_reverb(signal: FloatArray, sample_rate: int, mix: float) -> FloatArray:
    """
    Hafif geri beslemeli gecikmeye dayalı basit reverb uygular.

    Parametreler:
        signal: Mono giriş sinyali (float32).
        sample_rate: Örnekleme hızı (Hz).
        mix: 0-1 arası ıslak/kuru karışım.

    Dönüş:
        Reverb uygulanmış float32 sinyal.
    """
    x = _to_float32(signal)
    wet = _clamp(mix, 0.0, 1.0)
    dry = _FT(1.0 - wet)
    delay_main = max(1, int(0.055 * sample_rate))
    delay_tap = max(1, int(0.019 * sample_rate))
    fb = _FT(0.45)
    damp = _FT(0.25)
    buf_len = delay_main + 1
    buf = np.zeros(buf_len, dtype=_FT)
    y = np.empty_like(x)
    idx = 0
    lp_prev = _FT(0.0)
    for i, sample in enumerate(x):
        delayed = buf[idx]
        # basit damping
        lp = (1.0 - damp) * delayed + damp * lp_prev + _DENORM_GUARD
        lp_prev = lp
        # erken yansıma tap'i
        tap_idx = (idx - delay_tap) % buf_len
        tap = buf[tap_idx]
        new_val = sample + fb * lp
        buf[idx] = new_val
        idx = (idx + 1) % buf_len
        wet_sample = _FT(0.6) * lp + _FT(0.4) * tap
        y[i] = dry * sample + wet * wet_sample
    return y.astype(_FT, copy=False)


def stereo_widen(signal: FloatArray, amount: float) -> FloatArray:
    """
    Dekorele küçük gecikmelerle stereo genişletme uygular.

    Parametreler:
        signal: Mono giriş sinyali (float32).
        amount: 0-1 arası genişlik miktarı.

    Dönüş:
        İki kanallı (N, 2) float32 sinyal.
    """
    x = _to_float32(signal)
    amt = _clamp(amount, 0.0, 1.0)
    g = _FT(0.6 * amt)
    d1 = 11
    d2 = 7
    n = x.shape[0]
    l = np.empty((n,), dtype=_FT)
    r = np.empty((n,), dtype=_FT)
    buf1 = np.zeros(d1, dtype=_FT)
    buf2 = np.zeros(d2, dtype=_FT)
    i1 = 0
    i2 = 0
    for i, sample in enumerate(x):
        ap1 = -g * sample + buf1[i1] + g * buf1[i1 - 1] if d1 > 1 else sample
        ap2 = -g * sample + buf2[i2] + g * buf2[i2 - 1] if d2 > 1 else sample
        buf1[i1] = sample + g * ap1 + _DENORM_GUARD
        buf2[i2] = sample + g * ap2 + _DENORM_GUARD
        i1 = (i1 + 1) % d1
        i2 = (i2 + 1) % d2
        side = _FT(0.5) * (ap1 - ap2)
        l[i] = sample + side
        r[i] = sample - side
    return np.stack((l, r), axis=-1)


def normalize_gain(signal: FloatArray, target_rms: float = 0.1) -> FloatArray:
    """
    Sinyali hedef RMS değerine ölçekler.

    Parametreler:
        signal: Mono veya stereo sinyal (float32).
        target_rms: İstenen RMS değeri.

    Dönüş:
        Ölçeklenmiş float32 sinyal.
    """
    x = _to_float32(signal)
    rms_val = _rms(x)
    if rms_val <= 0.0:
        return x.copy()
    gain = _FT(target_rms / rms_val)
    return (x * gain).astype(_FT, copy=False)


if __name__ == "__main__":
    fs = 48000
    t = np.arange(fs, dtype=_FT) / _FT(fs)
    test = _FT(0.3) * np.sin(2.0 * math.pi * 440.0 * t, dtype=_FT)

    sat = soft_saturation(test, drive=1.5)
    warm = warmth(test, amount=0.7)
    rev = simple_reverb(test, sample_rate=fs, mix=0.25)
    wide = stereo_widen(test, amount=0.6)
    norm = normalize_gain(test, target_rms=0.1)

    print("Örnek RMS değerleri:")
    print(f"Giriş: {_rms(test):.6f}")
    print(f"Saturasyon: {_rms(sat):.6f}")
    print(f"Warmth: {_rms(warm):.6f}")
    print(f"Reverb: {_rms(rev):.6f}")
    print(f"Widen L: {_rms(wide[:,0]):.6f} R: {_rms(wide[:,1]):.6f}")
    print(f"Normalize: {_rms(norm):.6f}")
