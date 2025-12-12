import math
from typing import Tuple

import numpy as np

FloatArray = np.ndarray
_FT = np.float32
_DENORM_GUARD = _FT(1e-20)


def _to_float32(arr: FloatArray) -> FloatArray:
    """Girişi float32 kopyasına dönüştürür."""
    if arr.dtype != _FT:
        return arr.astype(_FT, copy=False)
    return arr


def _phase_increment(rate_hz: float, sample_rate: int) -> float:
    """LFO faz artışını hesaplar."""
    return 2.0 * math.pi * rate_hz / sample_rate


def sine_lfo(rate_hz: float, duration_sec: float, sample_rate: int) -> FloatArray:
    """
    Sinüs tabanlı LFO üretir.

    Parametreler:
        rate_hz: LFO frekansı (Hz).
        duration_sec: Süre (saniye).
        sample_rate: Örnekleme hızı (Hz).

    Dönüş:
        -1..1 arası float32 LFO vektörü.
    """
    samples = int(round(duration_sec * sample_rate))
    if samples <= 0:
        return np.zeros(0, dtype=_FT)
    phase_inc = _phase_increment(rate_hz, sample_rate)
    phase = np.arange(samples, dtype=np.float64) * phase_inc
    lfo = np.sin(phase, dtype=np.float64).astype(_FT)
    return lfo + _DENORM_GUARD


def triangle_lfo(rate_hz: float, duration_sec: float, sample_rate: int) -> FloatArray:
    """
    Üçgen dalga LFO üretir.

    Parametreler:
        rate_hz: LFO frekansı (Hz).
        duration_sec: Süre (saniye).
        sample_rate: Örnekleme hızı (Hz).

    Dönüş:
        -1..1 arası float32 LFO vektörü.
    """
    samples = int(round(duration_sec * sample_rate))
    if samples <= 0:
        return np.zeros(0, dtype=_FT)
    phase_inc = rate_hz / sample_rate
    phase = (np.arange(samples, dtype=np.float64) * phase_inc) % 1.0
    tri = 2.0 * np.abs(2.0 * phase - 1.0) - 1.0
    return tri.astype(_FT) + _DENORM_GUARD


def apply_volume_lfo(signal: FloatArray, lfo: FloatArray, depth: float) -> FloatArray:
    """
    LFO'yu genlik modülasyonu olarak uygular.

    Parametreler:
        signal: Mono giriş sinyali (float32).
        lfo: -1..1 arası LFO vektörü.
        depth: 0-1 arası modülasyon derinliği.

    Dönüş:
        Modüle edilmiş float32 sinyal.
    """
    x = _to_float32(signal)
    m = _to_float32(lfo)
    dep = _FT(np.clip(depth, 0.0, 1.0))
    if x.shape[0] != m.shape[0]:
        raise ValueError("signal ve lfo uzunlukları eşleşmelidir.")
    gain = _FT(1.0) + dep * m
    gain = np.maximum(gain, _FT(0.0))
    return (x * gain + _DENORM_GUARD).astype(_FT, copy=False)


def apply_filter_lfo(base_cutoff: float, lfo: FloatArray, depth: float) -> FloatArray:
    """
    LFO ile kesim frekansı yörüngesi üretir.

    Parametreler:
        base_cutoff: Temel kesim frekansı (Hz).
        lfo: -1..1 arası LFO vektörü.
        depth: 0-1 arası modülasyon derinliği (oransal).

    Dönüş:
        Her örnek için kesim frekansı (float32).
    """
    m = _to_float32(lfo)
    dep = float(np.clip(depth, 0.0, 1.0))
    base = float(max(1.0, base_cutoff))
    freq = base * (1.0 + dep * m.astype(np.float64))
    freq = np.clip(freq, 1.0, None)
    return freq.astype(_FT)


def apply_pan_lfo(signal: FloatArray, lfo: FloatArray, depth: float) -> FloatArray:
    """
    LFO ile panorama modülasyonu uygular (eşit güç yasası).

    Parametreler:
        signal: Mono giriş sinyali (float32).
        lfo: -1..1 arası LFO vektörü.
        depth: 0-1 arası pan genişliği.

    Dönüş:
        (N, 2) stereo float32 sinyal.
    """
    x = _to_float32(signal)
    m = _to_float32(lfo)
    dep = _FT(np.clip(depth, 0.0, 1.0))
    if x.shape[0] != m.shape[0]:
        raise ValueError("signal ve lfo uzunlukları eşleşmelidir.")
    pan = dep * m  # -1..1
    angle = (_FT(0.5) * (pan + _FT(1.0))) * _FT(math.pi / 2.0)
    left_gain = np.cos(angle, dtype=_FT)
    right_gain = np.sin(angle, dtype=_FT)
    left = x * left_gain + _DENORM_GUARD
    right = x * right_gain + _DENORM_GUARD
    return np.stack((left, right), axis=-1).astype(_FT, copy=False)


def _rms(signal: FloatArray) -> float:
    """RMS değerini float olarak döndürür."""
    x = _to_float32(signal)
    return float(np.sqrt(np.mean(x * x, dtype=_FT)))


if __name__ == "__main__":
    fs = 48000
    duration = 1.0
    t = np.arange(int(fs * duration), dtype=_FT) / _FT(fs)
    tone = _FT(0.2) * np.sin(2.0 * math.pi * 440.0 * t, dtype=_FT)

    lfo = sine_lfo(rate_hz=0.3, duration_sec=duration, sample_rate=fs)
    tri_lfo = triangle_lfo(rate_hz=0.2, duration_sec=duration, sample_rate=fs)

    vol_mod = apply_volume_lfo(tone, lfo, depth=0.5)
    pan_mod = apply_pan_lfo(tone, tri_lfo, depth=0.8)
    filt_mod = apply_filter_lfo(1200.0, lfo, depth=0.5)

    print("Volume LFO: min {:.6f}, max {:.6f}".format(float(vol_mod.min()), float(vol_mod.max())))
    print("Pan LFO L: min {:.6f}, max {:.6f}".format(float(pan_mod[:, 0].min()), float(pan_mod[:, 0].max())))
    print("Pan LFO R: min {:.6f}, max {:.6f}".format(float(pan_mod[:, 1].min()), float(pan_mod[:, 1].max())))
    print("Filter cutoff: min {:.2f} Hz, max {:.2f} Hz".format(float(filt_mod.min()), float(filt_mod.max())))
    print("RMS tone: {:.6f}, RMS vol_mod: {:.6f}".format(_rms(tone), _rms(vol_mod)))
