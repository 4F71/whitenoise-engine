import math
import numpy as np

FloatArray = np.ndarray
_FT = np.float32


def _to_float32(arr: FloatArray) -> FloatArray:
    """Girişi float32 kopyasına dönüştürür."""
    if arr.dtype != _FT:
        return arr.astype(_FT, copy=False)
    return arr


def _require_same_length(a: FloatArray, b: FloatArray) -> None:
    """İki vektörün uzunluk eşitliğini doğrular."""
    if a.shape[0] != b.shape[0]:
        raise ValueError("signal ve lfo uzunlukları eşleşmelidir.")


def sine_lfo(rate_hz: float, duration_sec: float, sample_rate: int) -> FloatArray:
    """
    Sinüs tabanlı, DC içermeyen LFO üretir.

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
    phase_inc = _FT(2.0 * math.pi * rate_hz / sample_rate)
    phase = np.arange(samples, dtype=np.float64) * phase_inc
    return np.sin(phase, dtype=np.float64).astype(_FT)


def triangle_lfo(rate_hz: float, duration_sec: float, sample_rate: int) -> FloatArray:
    """
    Üçgen dalga formunda, DC içermeyen LFO üretir.

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
    return tri.astype(_FT)


def apply_volume_lfo(signal: FloatArray, lfo: FloatArray, depth: float) -> FloatArray:
    """
    LFO ile genlik modülasyonu uygular; klik oluşumunu engellemek için
    eşitlenmiş uzunlukta yumuşak ölçekleme kullanır.

    Parametreler:
        signal: Mono giriş sinyali (float32).
        lfo: -1..1 arası LFO vektörü.
        depth: 0-1 arası modülasyon derinliği.

    Dönüş:
        Modüle edilmiş float32 sinyal.
    """
    x = _to_float32(signal)
    m = _to_float32(lfo)
    _require_same_length(x, m)
    dep = _FT(np.clip(depth, 0.0, 1.0))
    gain = _FT(1.0) + dep * _FT(0.5) * m  # 0.5..1.5 aralığı
    return (x * gain).astype(_FT, copy=False)


def apply_filter_lfo(base_cutoff: float, lfo: FloatArray, depth: float) -> FloatArray:
    """
    Kesim frekansı için LFO yörüngesi üretir.

    Parametreler:
        base_cutoff: Temel kesim frekansı (Hz).
        lfo: -1..1 arası LFO vektörü.
        depth: 0-1 arası modülasyon derinliği (oransal).

    Dönüş:
        Örnek bazlı kesim frekansları (float32).
    """
    m = _to_float32(lfo)
    dep = float(np.clip(depth, 0.0, 1.0))
    base = float(max(1.0, base_cutoff))
    freq = base * (1.0 + dep * 0.5 * m.astype(np.float64))
    freq = np.clip(freq, 1.0, None)
    return freq.astype(_FT)


def apply_pan_lfo(signal: FloatArray, lfo: FloatArray, depth: float) -> FloatArray:
    """
    Eşit güç yasası ile panorama modülasyonu uygular.

    Parametreler:
        signal: Mono giriş sinyali (float32).
        lfo: -1..1 arası LFO vektörü.
        depth: 0-1 arası pan genişliği.

    Dönüş:
        (N, 2) stereo float32 sinyal.
    """
    x = _to_float32(signal)
    m = _to_float32(lfo)
    _require_same_length(x, m)
    dep = _FT(np.clip(depth, 0.0, 1.0))
    pan = dep * m  # -1..1
    angle = (_FT(0.5) * (pan + _FT(1.0))) * _FT(math.pi / 2.0)
    left_gain = np.cos(angle, dtype=_FT)
    right_gain = np.sin(angle, dtype=_FT)
    left = x * left_gain
    right = x * right_gain
    return np.stack((left, right), axis=-1).astype(_FT, copy=False)


def _rms(signal: FloatArray) -> float:
    """RMS değerini float olarak döndürür."""
    x = _to_float32(signal)
    return float(np.sqrt(np.mean(x * x, dtype=_FT)))


def perlin_modulated_lfo(
    rate_hz: float,
    duration_sec: float,
    sample_rate: int,
    mod_amount: float = 0.1,
    seed: int = None
) -> FloatArray:
    """
    Perlin noise modulated sine LFO - organic breathing effect.
    
    Theory: organic_texture_theory.md Section 2.2
    
    Irregular breathing cycle: LFO frequency modulated by smoothed random walk.
    This creates "living, breathing" effect - each cycle slightly different.
    
    Args:
        rate_hz: Base LFO frequency (Hz), e.g. 0.005-0.01 Hz
        duration_sec: Duration (seconds)
        sample_rate: Sample rate (Hz)
        mod_amount: Modulation depth [0.0-1.0], 0.1 = ±10% frequency variation
        seed: Random seed for reproducibility (None = random)
        
    Returns:
        Float32 LFO vector [-1..1]
        
    Example:
        >>> # 100s breathing cycle with ±10% variation
        >>> lfo = perlin_modulated_lfo(0.01, 180, 48000, mod_amount=0.1, seed=42)
        >>> # Each cycle will be 90-110s (slightly different)
    
    Reference:
        Perlin, Ken. "Improving noise." SIGGRAPH 2002
        organic_texture_theory.md Section 2.2.1-2.2.3
    """
    samples = int(round(duration_sec * sample_rate))
    if samples <= 0:
        return np.zeros(0, dtype=_FT)
    
    # Parameter validation
    rate_hz = max(0.001, min(20.0, rate_hz))
    mod_amount = float(np.clip(mod_amount, 0.0, 1.0))
    
    # Seed RNG
    rng = np.random.RandomState(seed)
    
    # === Simplified Perlin Noise (Smoothed Random Walk) ===
    # Theory: Section 1.3 - Wavelet noise approximation
    # We use smoothed random walk instead of full Perlin for efficiency
    
    # Perlin update rate: ~10x slower than LFO base freq (for smooth modulation)
    perlin_update_samples = max(1, int(sample_rate / (rate_hz * 10)))
    num_perlin_points = (samples // perlin_update_samples) + 2
    
    # Generate random walk
    random_walk = rng.randn(num_perlin_points).astype(np.float64)
    
    # Cumulative sum for smooth evolution
    random_walk = np.cumsum(random_walk)
    
    # Normalize to [-1, 1]
    if random_walk.std() > 0:
        random_walk = (random_walk - random_walk.mean()) / (random_walk.std() * 3.0)
        random_walk = np.clip(random_walk, -1.0, 1.0)
    
    # Interpolate to audio rate (cubic interpolation for smoothness)
    perlin_time = np.arange(num_perlin_points, dtype=np.float64)
    audio_time = np.linspace(0, num_perlin_points - 1, samples, dtype=np.float64)
    perlin_signal = np.interp(audio_time, perlin_time, random_walk).astype(_FT)
    
    # === Frequency Modulation ===
    # f_inst(t) = f_base * (1 + mod_amount * perlin(t))
    # Theory: Section 2.2.1
    
    instantaneous_freq = rate_hz * (1.0 + mod_amount * perlin_signal)
    
    # === Phase Accumulation ===
    # Integrate frequency to get phase
    phase_increment = instantaneous_freq / float(sample_rate)
    phase = np.cumsum(phase_increment, dtype=np.float64)
    
    # === Sine LFO Output ===
    output = np.sin(2.0 * np.pi * phase, dtype=np.float64).astype(_FT)
    
    return output


if __name__ == "__main__":
    fs = 48000
    duration = 1.0
    t = np.arange(int(fs * duration), dtype=_FT) / _FT(fs)
    tone = _FT(0.2) * np.sin(_FT(2.0 * math.pi * 440.0) * t, dtype=_FT)

    lfo_sine = sine_lfo(rate_hz=0.3, duration_sec=duration, sample_rate=fs)
    lfo_tri = triangle_lfo(rate_hz=0.2, duration_sec=duration, sample_rate=fs)

    vol_mod = apply_volume_lfo(tone, lfo_sine, depth=0.5)
    pan_mod = apply_pan_lfo(tone, lfo_tri, depth=0.7)
    cutoff_mod = apply_filter_lfo(1200.0, lfo_sine, depth=0.6)

    print(f"Volume mod min/max: {float(vol_mod.min()):.6f} / {float(vol_mod.max()):.6f}")
    print(f"Pan L min/max: {float(pan_mod[:, 0].min()):.6f} / {float(pan_mod[:, 0].max()):.6f}")
    print(f"Pan R min/max: {float(pan_mod[:, 1].min()):.6f} / {float(pan_mod[:, 1].max()):.6f}")
    print(f"Cutoff min/max: {float(cutoff_mod.min()):.2f} / {float(cutoff_mod.max()):.2f} Hz")
    print(f"RMS original: {_rms(tone):.6f}  RMS vol_mod: {_rms(vol_mod):.6f}")
