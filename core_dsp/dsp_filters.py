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


def dc_block(signal: FloatArray, pole: float = 0.995) -> FloatArray:
    """
    DC bileşenini bastıran birinci dereceden yüksek geçiren filtre uygular.

    Parametreler:
        signal: Mono giriş sinyali (float32).
        pole: Kutup katsayısı; 1.0'a yaklaştıkça daha düşük kesim sağlar.

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


def tilt_filter(
    signal: FloatArray, tilt_db: float, sample_rate: int
) -> FloatArray:
    """
    Spektral eğimi lineer olarak eğen tek kutuplu tilt filtresi uygular.

    Pozitif değerler üst frekansları yükseltirken düşük frekansları bastırır.

    Parametreler:
        signal: Mono giriş sinyali (float32).
        tilt_db: Her decade başına dB eğim.
        sample_rate: Örnekleme hızı (Hz).

    Dönüş:
        Spektral eğimli float32 sinyal.
    """
    x = _to_float32(signal)
    y = np.empty_like(x)
    # Tilt katsayısı yüksek geçiren kısmın ağırlığını belirler.
    hp_gain = _FT(10 ** (tilt_db / 20.0))
    lp_gain = _FT(1.0)
    # Orta bandı referanslamak için toplam kazancı normalize et.
    norm = _FT(1.0 / (hp_gain + lp_gain))
    hp_gain *= norm
    lp_gain *= norm
    # 200 Hz merkezli hafif geçiş için sabit kesim seçimi.
    cutoff = _safe_cutoff(200.0, sample_rate)
    alpha = _FT(math.exp(-2.0 * math.pi * cutoff / sample_rate))
    lp_prev = _FT(0.0)
    hp_prev = _FT(0.0)
    x_prev = _FT(0.0)
    for i, sample in enumerate(x):
        lp_curr = (_FT(1.0) - alpha) * sample + alpha * lp_prev + _DENORM_GUARD
        hp_curr = alpha * (hp_prev + sample - x_prev) + _DENORM_GUARD
        y[i] = lp_gain * lp_curr + hp_gain * hp_curr
        lp_prev = lp_curr
        hp_prev = hp_curr
        x_prev = sample
    return y
