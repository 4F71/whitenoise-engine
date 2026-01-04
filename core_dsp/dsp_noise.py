import math
from typing import Optional

import numpy as np
from scipy import signal


FloatArray = np.ndarray
_FT = np.float32


def _ensure_int_samples(duration_sec: float, sample_rate: int) -> int:
    samples = int(round(duration_sec * sample_rate))
    if samples <= 0:
        raise ValueError("duration_sec ve sample_rate pozitif olmalıdır.")
    return samples


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def _denorm_guard(value: float = 1e-20) -> _FT:
    return _FT(value)


def _dc_block(input_signal: FloatArray, pole: float = 0.995) -> FloatArray:
    """
    Basit DC blocker; y[n] = x[n] - x[n-1] + pole * y[n-1].
    
    Optimized: scipy.signal.lfilter kullanır (50-100x hızlı).
    Transfer function: H(z) = (1 - z^-1) / (1 - pole×z^-1)
    """
    # DC block IIR filter coefficients
    # Numerator: 1 - z^-1 (high-pass characteristic)
    b = np.array([1.0, -1.0], dtype=np.float64)
    # Denominator: 1 - pole×z^-1 (feedback)
    a = np.array([1.0, -pole], dtype=np.float64)
    
    # Apply IIR filter (C-optimized, very fast)
    filtered = signal.lfilter(b, a, input_signal.astype(np.float64))
    
    # Add denorm guard
    filtered += _denorm_guard()
    
    return filtered.astype(_FT, copy=False)


def _soft_clip(input_signal: FloatArray, limit: float = 0.98) -> FloatArray:
    """Tanh kullanmadan arctan tabanlı yumuşak sınırlandırma."""
    scaled = input_signal / limit
    clipped = limit * (_FT(2.0 / math.pi)) * np.arctan(scaled, dtype=_FT)
    return clipped.astype(_FT, copy=False)


def _rms(input_signal: FloatArray) -> float:
    return float(np.sqrt(np.mean(input_signal * input_signal, dtype=_FT)))


def generate_white_noise(
    duration_sec: float,
    sample_rate: int,
    amplitude: float = 1.0,
    seed: Optional[int] = None,
) -> FloatArray:
    """
    Gaussian dağılımlı white noise üretir.

    Parametreler:
        duration_sec: Süre (saniye).
        sample_rate: Örnekleme hızı (Hz).
        amplitude: Çıkış genlik ölçeği.
        seed: Deterministik üretim için opsiyonel tohum.

    Dönüş:
        float32 numpy vektörü (mono).
    """
    samples = _ensure_int_samples(duration_sec, sample_rate)
    gen = _rng(seed)
    noise = gen.normal(loc=0.0, scale=amplitude, size=samples).astype(_FT)
    noise = _dc_block(noise)
    noise = _soft_clip(noise)
    return noise


def generate_pink_noise(
    duration_sec: float,
    sample_rate: int,
    amplitude: float = 1.0,
    seed: Optional[int] = None,
) -> FloatArray:
    """
    Voss–McCartney prensipli pink noise üretir.

    Uzun süreli stabilite için satır sayısı dinamik seçilir ve
    hafif eksponansiyel yumuşatma ile spektral pürüz azaltılır.
    """
    samples = _ensure_int_samples(duration_sec, sample_rate)
    gen = _rng(seed)
    num_rows = int(np.clip(math.ceil(math.log2(samples + 1)), 8, 24))
    states = gen.normal(loc=0.0, scale=1.0, size=num_rows).astype(_FT)
    running_sum = states.sum(dtype=np.float64)
    output = np.empty(samples, dtype=_FT)
    guard = _denorm_guard()
    smooth = _FT(0.002)  # spektral pürüz azaltıcı hafif yumuşatma
    prev = _FT(0.0)
    norm = _FT(amplitude / math.sqrt(num_rows))
    for i in range(samples):
        idx = i + 1
        bit = 0
        while bit < num_rows and (idx & 1) == 0:
            old = states[bit]
            new = _FT(gen.normal())
            states[bit] = new
            running_sum += float(new - old)
            idx >>= 1
            bit += 1
        raw = _FT(running_sum) * norm + guard
        smoothed = prev + smooth * (raw - prev)
        output[i] = smoothed
        prev = smoothed
    output = _dc_block(output)
    output = _soft_clip(output)
    return output


def generate_brown_noise(
    duration_sec: float,
    sample_rate: int,
    amplitude: float = 1.0,
    seed: Optional[int] = None,
) -> FloatArray:
    """
    White noise entegrasyonu ile brown noise üretir.

    Drift sınırlama için sızıntılı entegratör ve DC blocker kullanır.
    
    Optimized: scipy.signal.lfilter ile leaky integrator (10-50x hızlı).
    Transfer function: H(z) = 1 / (1 - (1-leak)×z^-1)
    """
    samples = _ensure_int_samples(duration_sec, sample_rate)
    gen = _rng(seed)
    leak = max(1e-5, 1.0 / (sample_rate * 1200.0))  # çok yavaş sızıntı
    white_scale = amplitude * math.sqrt(2.0 * leak)
    white = gen.normal(loc=0.0, scale=white_scale, size=samples).astype(_FT)
    
    # Leaky integrator: y[n] = (1-leak)×y[n-1] + x[n]
    # IIR filter coefficients
    b = np.array([1.0], dtype=np.float64)  # feedforward
    a = np.array([1.0, -(1.0 - leak)], dtype=np.float64)  # feedback with leak
    
    # Apply leaky integrator (C-optimized)
    output = signal.lfilter(b, a, white.astype(np.float64))
    output = output.astype(_FT, copy=False)
    
    # Add denorm guard
    output += _denorm_guard()
    
    # Post-processing
    output = _dc_block(output)
    output = _soft_clip(output)
    return output


def generate_blue_noise(
    duration_sec: float,
    sample_rate: int,
    amplitude: float = 1.0,
    seed: Optional[int] = None,
) -> FloatArray:
    """
    White noise differencing ile blue noise üretir.
    """
    samples = _ensure_int_samples(duration_sec, sample_rate)
    gen = _rng(seed)
    base_sigma = amplitude / math.sqrt(2.0)
    white = gen.normal(loc=0.0, scale=base_sigma, size=samples + 1).astype(_FT)
    diff = white[1:] - white[:-1]
    diff += _denorm_guard()
    diff = _dc_block(diff)
    diff = _soft_clip(diff)
    return diff.astype(_FT, copy=False)


def generate_violet_noise(
    duration_sec: float,
    sample_rate: int,
    amplitude: float = 1.0,
    seed: Optional[int] = None,
) -> FloatArray:
    """
    İkinci dereceden differencing ile violet noise üretir.

    Çok ince yüksek frekans “air” hissi hedeflenir.
    """
    samples = _ensure_int_samples(duration_sec, sample_rate)
    gen = _rng(seed)
    base_sigma = amplitude / math.sqrt(6.0)
    white = gen.normal(loc=0.0, scale=base_sigma, size=samples + 2).astype(_FT)
    second_diff = white[2:] - _FT(2.0) * white[1:-1] + white[:-2]
    second_diff += _denorm_guard()
    second_diff = _dc_block(second_diff)
    second_diff = _soft_clip(second_diff)
    return second_diff.astype(_FT, copy=False)


if __name__ == "__main__":
    fs = 48000
    pink = generate_pink_noise(1.0, fs, amplitude=1.0, seed=42)
    print("Pink noise 1s @48k")
    print(f"Min: {float(pink.min()):.6f}")
    print(f"Max: {float(pink.max()):.6f}")
    print(f"RMS: {_rms(pink):.6f}")
