"""
core_dsp/dsp_binaural.py

Binaural beats DSP fonksiyonları.

Bu modül:
- Stereo binaural beat sinyalleri üretir
- Sol/sağ kanal faz farkı ile beat algısı sağlar
- Akademik araştırmalara dayalı parametreler kullanır

Referans:
- Oster, G. (1973). "Auditory Beats in the Brain"
- Ingendoh et al. (2023). "Binaural beats to entrain the brain? A systematic review"
- docs/02_v2_theory/binaural_beats_theory.md

Sorumluluk sınırı:
- SADECE sinyal üretimi
- Preset yönetimi İÇERMEZ
- Render pipeline İÇERMEZ
"""

import math
import numpy as np
from typing import Tuple

FloatArray = np.ndarray
_FT = np.float32


# =============================================================================
# SAFE PARAMETER RANGES (binaural_beats_theory.md'den)
# =============================================================================

SAFE_BEAT_RANGES = {
    'delta': (3.0, 4.0),     # Hz - Alt limit: 3 Hz (rotating tone'dan kaçın)
    'theta': (4.0, 8.0),     # Hz - Priority #1 (en çok araştırılmış)
    'alpha': (8.0, 13.0),    # Hz - Priority #2
    'gamma': (38.0, 42.0),   # Hz - Experimental (40 Hz optimal)
}

DANGER_ZONES = {
    'rotating': (0.0, 3.0),   # Hz - Dönen ses, rahatsız edici
    'rough': (20.0, 30.0),    # Hz - Pürüzlü ses, algı azalır
    'separate': (50.0, 100.0), # Hz - Beat algısı kaybolur
}

# Carrier frequency bounds
CARRIER_MIN = 100.0   # Hz
CARRIER_MAX = 1200.0  # Hz (Extended for Solfeggio frequencies - original Licklider: 1000Hz)
CARRIER_OPTIMAL = 400.0  # Hz (Goodin 2012 - en çok araştırılmış)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def _validate_carrier_freq(carrier_freq: float) -> None:
    """
    Carrier frekansını doğrular.
    
    Args:
        carrier_freq: Taşıyıcı frekans (Hz)
        
    Raises:
        ValueError: Geçersiz carrier frequency için
    """
    if carrier_freq < CARRIER_MIN:
        raise ValueError(
            f"Carrier frequency {carrier_freq} Hz < {CARRIER_MIN} Hz (minimum)"
        )
    if carrier_freq > CARRIER_MAX:
        raise ValueError(
            f"Carrier frequency {carrier_freq} Hz > {CARRIER_MAX} Hz "
            f"(Licklider threshold - faz kodlaması kaybolur)"
        )


def _validate_beat_freq(beat_freq: float) -> None:
    """
    Beat frekansını doğrular ve tehlikeli bölgeleri kontrol eder.
    
    Args:
        beat_freq: Beat frekansı (Hz)
        
    Raises:
        ValueError: Tehlikeli bölgedeki beat frequency için
    """
    # Danger zone kontrolü
    for zone_name, (low, high) in DANGER_ZONES.items():
        if low <= beat_freq <= high:
            raise ValueError(
                f"Beat frequency {beat_freq} Hz in danger zone '{zone_name}' "
                f"({low}-{high} Hz). "
                f"Recommended: Use theta (4-8 Hz) or alpha (8-13 Hz) instead."
            )
    
    # Beta band uyarısı (13-30 Hz - hiçbir entrainment kanıtı yok)
    if 13.0 <= beat_freq <= 30.0 and not (20.0 <= beat_freq <= 30.0):
        # 20-30 zaten "rough" danger zone'da
        raise ValueError(
            f"Beat frequency {beat_freq} Hz in beta band (13-30 Hz). "
            f"No entrainment evidence found in research. "
            f"Recommended: Use alpha (8-13 Hz) or gamma (40 Hz) instead."
        )


def _validate_amplitude(amplitude: float) -> None:
    """
    Amplitude değerini doğrular.
    
    Args:
        amplitude: Genlik (0.0 - 1.0)
        
    Raises:
        ValueError: Geçersiz amplitude için
    """
    if amplitude < 0.0 or amplitude > 1.0:
        raise ValueError(
            f"Amplitude {amplitude} must be in range [0.0, 1.0]"
        )


# =============================================================================
# CORE GENERATION FUNCTIONS
# =============================================================================

def generate_binaural_beats(
    carrier_freq: float,
    beat_freq: float,
    amplitude: float,
    duration_sec: float,
    sample_rate: int,
    breathing_lfo: np.ndarray = None
) -> FloatArray:
    """
    Stereo binaural beat sinyali üretir.
    
    Matematiksel Model:
    -------------------
    Left channel:  A × sin(2π × f_carrier × t)
    Right channel: A × sin(2π × (f_carrier + f_beat) × t)
    
    Perceived beat: f_beat = |f_right - f_left|
    
    Mekanizma:
    ----------
    1. İki kulağa farklı frekanslar gönderilir
    2. Beyin sapı (Superior Olivary Complex) faz farkını algılar
    3. Faz farkı sürekli değişir (beat_freq rate'inde)
    4. Beyin bu değişimi "beat" olarak yorumlar
    
    Parametreler:
    ------------
    carrier_freq : float
        Taşıyıcı frekans (Hz).
        - Optimal: 400 Hz (Goodin 2012 - en çok araştırılmış)
        - Alternatif: 440 Hz (Oster 1973)
        - Gamma için: 250 Hz
        - Range: 100-1000 Hz
        
    beat_freq : float
        Beat frekansı (Hz). EEG frekans bantlarına karşılık gelir:
        - Delta: 3-4 Hz (Alt limit: 3 Hz - rotating tone riski)
        - Theta: 4-8 Hz (Priority #1 - en çok araştırılmış)
        - Alpha: 8-13 Hz (Priority #2)
        - Gamma: 38-42 Hz (40 Hz optimal - "consciousness frequency")
        
        Danger Zones (KULLANMA):
        - <3 Hz: Rotating tone (dönen ses, rahatsız edici)
        - 13-30 Hz: Beta band (hiçbir entrainment kanıtı yok)
        - 20-30 Hz: Rough sound (pürüzlü ses)
        - >50 Hz: Separate tones (beat algısı kaybolur)
        
    amplitude : float
        Genlik (0.0 - 1.0).
        - Optimal: 0.5 (-6 dB, comfortable listening)
        - Minimum: 0.1 (-20 dB, hala çalışır - sub-threshold perception)
        
    duration_sec : float
        Süre (saniye).
        Band-specific minimums (binaural_beats_theory.md):
        - Delta: 5 dakika minimum
        - Theta: 3 dakika minimum, 10 dakika optimal
        - Alpha: 5 dakika minimum
        - Gamma: 15 dakika minimum, 20-30 dakika optimal
        
    sample_rate : int
        Örnekleme hızı (Hz). Tipik: 48000 Hz
    
    Dönüş:
    ------
    FloatArray
        Stereo audio buffer (shape: N × 2, dtype: float32)
        - [:, 0]: Sol kanal (carrier_freq)
        - [:, 1]: Sağ kanal (carrier_freq + beat_freq)
    
    Raises:
    -------
    ValueError
        Geçersiz parametreler için (danger zones, range dışı değerler)
    
    Notlar:
    -------
    - STEREO output ZORUNLU (binaural mekanizma iki kulak gerektirir)
    - Kulaklık kullanımı şart (hoparlör çalışmaz)
    - Pure tones kullan (pink noise embedding başarısız)
    - Phase alignment: İki kanal t=0'da aynı fazda başlar
    
    Referanslar:
    -----------
    - Oster (1973): Binaural beat algısı ve psikoakustik
    - Goodin et al. (2012): 400 Hz carrier optimal (ASSR ölçümleri)
    - Ingendoh et al. (2023): Sistematik derleme (14 çalışma)
    
    Örnekler:
    --------
    >>> # Theta meditation (7 Hz - en çok araştırılmış)
    >>> stereo = generate_binaural_beats(
    ...     carrier_freq=400.0,
    ...     beat_freq=7.0,
    ...     amplitude=0.5,
    ...     duration_sec=600.0,  # 10 dakika
    ...     sample_rate=48000
    ... )
    >>> stereo.shape
    (28800000, 2)
    
    >>> # Alpha relaxation (10 Hz - kısa süre yeterli)
    >>> stereo = generate_binaural_beats(
    ...     carrier_freq=400.0,
    ...     beat_freq=10.0,
    ...     amplitude=0.5,
    ...     duration_sec=300.0,  # 5 dakika
    ...     sample_rate=48000
    ... )
    
    >>> # Gamma focus (40 Hz - uzun süre gerekli)
    >>> stereo = generate_binaural_beats(
    ...     carrier_freq=250.0,  # Gamma için özel
    ...     beat_freq=40.0,
    ...     amplitude=0.5,
    ...     duration_sec=1200.0,  # 20 dakika
    ...     sample_rate=48000
    ... )
    """
    # Parametreleri doğrula
    _validate_carrier_freq(carrier_freq)
    _validate_beat_freq(beat_freq)
    _validate_amplitude(amplitude)
    
    # Sample sayısını hesapla
    samples = int(round(duration_sec * sample_rate))
    if samples <= 0:
        # Boş stereo array döndür
        return np.zeros((0, 2), dtype=_FT)
    
    # Zaman vektörü oluştur (float64 - hassasiyet için)
    t = np.arange(samples, dtype=np.float64) / sample_rate
    
    # Breathing LFO (V2.7+): Carrier frequency modulation
    # LFO range: [-1, 1], modulate carrier_freq by ±3% (natural breathing feel)
    if breathing_lfo is not None and len(breathing_lfo) == samples:
        # Modulation depth: ±3% of carrier frequency (subtle but perceptible)
        modulated_carrier_left = carrier_freq * (1.0 + 0.03 * breathing_lfo)
        modulated_carrier_right = (carrier_freq + beat_freq) * (1.0 + 0.03 * breathing_lfo)
    else:
        # No breathing: constant carrier frequency
        modulated_carrier_left = carrier_freq
        modulated_carrier_right = carrier_freq + beat_freq
    
    # Sol kanal: carrier frekansı (with optional breathing)
    # Phase başlangıcı 0 (alignment için kritik)
    left_channel = amplitude * np.sin(
        2.0 * math.pi * modulated_carrier_left * t,
        dtype=np.float64
    )
    
    # Sağ kanal: carrier + beat frekansı (with optional breathing)
    # Phase başlangıcı 0 (alignment için kritik)
    right_channel = amplitude * np.sin(
        2.0 * math.pi * modulated_carrier_right * t,
        dtype=np.float64
    )
    
    # Stereo array oluştur (N × 2)
    # float32'ye çevir (memory efficiency)
    stereo = np.stack([left_channel, right_channel], axis=1).astype(_FT)
    
    return stereo


def apply_fade(
    stereo_signal: FloatArray,
    fade_duration: float = 2.0,
    sample_rate: int = 48000
) -> FloatArray:
    """
    Stereo sinyale cosine fade in/out uygular (click önleme).
    
    Args:
        stereo_signal: Stereo audio buffer (N × 2)
        fade_duration: Fade süresi (saniye). Default: 2.0
        sample_rate: Örnekleme hızı (Hz). Default: 48000
        
    Returns:
        Fade uygulanmış stereo audio buffer (N × 2)
        
    Notes:
        - Cosine fade kullanır (smooth, click-free)
        - Her iki kanala da aynı fade uygulanır
        - Original signal değiştirilmez (copy oluşturulur)
    """
    if stereo_signal.shape[0] == 0:
        return stereo_signal
    
    signal = stereo_signal.copy()
    fade_samples = int(fade_duration * sample_rate)
    
    # Fade sample sayısı sinyal uzunluğundan fazla olamaz
    fade_samples = min(fade_samples, signal.shape[0] // 2)
    
    if fade_samples <= 0:
        return signal
    
    # Fade in (cosine)
    fade_in = (1.0 - np.cos(np.linspace(0, math.pi, fade_samples))) / 2.0
    fade_in = fade_in.astype(_FT)
    
    # Fade out (cosine)
    fade_out = (1.0 + np.cos(np.linspace(0, math.pi, fade_samples))) / 2.0
    fade_out = fade_out.astype(_FT)
    
    # Her iki kanala da uygula
    signal[:fade_samples, 0] *= fade_in
    signal[:fade_samples, 1] *= fade_in
    signal[-fade_samples:, 0] *= fade_out
    signal[-fade_samples:, 1] *= fade_out
    
    return signal


def get_recommended_duration(beat_freq: float) -> Tuple[float, float]:
    """
    Beat frekansına göre önerilen süre aralığını döndürür.
    
    Args:
        beat_freq: Beat frekansı (Hz)
        
    Returns:
        (minimum_duration, optimal_duration) tuple (saniye cinsinden)
        
    Notes:
        Akademik araştırmalara dayalı öneriler:
        - Delta: 5-10 dakika
        - Theta: 3-10 dakika (en çok araştırılmış)
        - Alpha: 5-10 dakika
        - Gamma: 15-30 dakika (en uzun süre gerektirir)
    """
    if beat_freq < 4.0:  # Delta
        return (300.0, 600.0)  # 5-10 dakika
    elif beat_freq < 8.0:  # Theta
        return (180.0, 600.0)  # 3-10 dakika
    elif beat_freq < 13.0:  # Alpha
        return (300.0, 600.0)  # 5-10 dakika
    elif beat_freq >= 30.0:  # Gamma
        return (900.0, 1800.0)  # 15-30 dakika
    else:
        # Beta band (not recommended)
        return (300.0, 600.0)  # Default


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("dsp_binaural.py test başlıyor...")
    print("=" * 60)
    
    # Test 1: Theta meditation (priority #1)
    print("\n[TEST 1] Theta Meditation (7 Hz)")
    theta = generate_binaural_beats(
        carrier_freq=400.0,
        beat_freq=7.0,
        amplitude=0.5,
        duration_sec=1.0,  # 1 saniye test için
        sample_rate=48000
    )
    
    assert theta.shape == (48000, 2), f"Shape yanlış: {theta.shape}"
    assert theta.dtype == _FT, f"Dtype yanlış: {theta.dtype}"
    assert np.isfinite(theta).all(), "NaN/Inf var"
    
    # Sol ve sağ kanal farklı olmalı (beat için)
    assert not np.allclose(theta[:, 0], theta[:, 1]), "Kanallar aynı!"
    
    left_peak = np.max(np.abs(theta[:, 0]))
    right_peak = np.max(np.abs(theta[:, 1]))
    print(f"  Shape: {theta.shape}")
    print(f"  Dtype: {theta.dtype}")
    print(f"  Left peak: {left_peak:.4f}")
    print(f"  Right peak: {right_peak:.4f}")
    print("  [OK]")
    
    # Test 2: Alpha relaxation (priority #2)
    print("\n[TEST 2] Alpha Relaxation (10 Hz)")
    alpha = generate_binaural_beats(
        carrier_freq=400.0,
        beat_freq=10.0,
        amplitude=0.5,
        duration_sec=1.0,
        sample_rate=48000
    )
    
    assert alpha.shape == (48000, 2)
    assert alpha.dtype == _FT
    print(f"  Shape: {alpha.shape}")
    print("  [OK]")
    
    # Test 3: Gamma focus (experimental)
    print("\n[TEST 3] Gamma Focus (40 Hz)")
    gamma = generate_binaural_beats(
        carrier_freq=250.0,  # Gamma için özel carrier
        beat_freq=40.0,
        amplitude=0.5,
        duration_sec=1.0,
        sample_rate=48000
    )
    
    assert gamma.shape == (48000, 2)
    print(f"  Shape: {gamma.shape}")
    print("  [OK]")
    
    # Test 4: Fade in/out
    print("\n[TEST 4] Fade In/Out")
    faded = apply_fade(theta, fade_duration=0.1, sample_rate=48000)
    
    assert faded.shape == theta.shape
    
    # Fade bölgelerinde amplitude azalmalı
    fade_samples = int(0.1 * 48000)
    
    # Fade ortasından bir sample al (başlangıç/bitiş 0'a yakın olabilir)
    fade_mid_start = fade_samples // 2
    fade_mid_end = -fade_samples // 2
    
    assert np.abs(faded[fade_mid_start, 0]) < np.abs(theta[fade_mid_start, 0]), "Fade in çalışmıyor"
    assert np.abs(faded[fade_mid_end, 0]) < np.abs(theta[fade_mid_end, 0]), "Fade out çalışmıyor"
    
    print(f"  Fade samples: {fade_samples}")
    print(f"  Mid start ratio: {np.abs(faded[fade_mid_start, 0]) / np.abs(theta[fade_mid_start, 0]):.4f}")
    print(f"  Mid end ratio: {np.abs(faded[fade_mid_end, 0]) / np.abs(theta[fade_mid_end, 0]):.4f}")
    print("  [OK]")
    
    # Test 5: Danger zone validation (rotating tone)
    print("\n[TEST 5] Danger Zone Validation (<3 Hz)")
    try:
        dangerous = generate_binaural_beats(
            carrier_freq=400.0,
            beat_freq=2.0,  # Rotating tone zone!
            amplitude=0.5,
            duration_sec=1.0,
            sample_rate=48000
        )
        print("  HATA: Exception beklendi!")
        assert False
    except ValueError as e:
        print(f"  ValueError yakalandı: {str(e)[:80]}...")
        print("  [OK]")
    
    # Test 6: Beta band validation (no evidence)
    print("\n[TEST 6] Beta Band Validation (13-30 Hz)")
    try:
        beta = generate_binaural_beats(
            carrier_freq=400.0,
            beat_freq=16.0,  # Beta band - no entrainment evidence
            amplitude=0.5,
            duration_sec=1.0,
            sample_rate=48000
        )
        print("  HATA: Exception beklendi!")
        assert False
    except ValueError as e:
        print(f"  ValueError yakalandı: {str(e)[:80]}...")
        print("  [OK]")
    
    # Test 7: Carrier frequency validation
    print("\n[TEST 7] Carrier Frequency Validation")
    try:
        high_carrier = generate_binaural_beats(
            carrier_freq=1500.0,  # >1000 Hz - Licklider threshold
            beat_freq=7.0,
            amplitude=0.5,
            duration_sec=1.0,
            sample_rate=48000
        )
        print("  HATA: Exception beklendi!")
        assert False
    except ValueError as e:
        print(f"  ValueError yakalandı: {str(e)[:80]}...")
        print("  [OK]")
    
    # Test 8: Recommended duration
    print("\n[TEST 8] Recommended Duration")
    
    durations = {
        'Theta (7 Hz)': get_recommended_duration(7.0),
        'Alpha (10 Hz)': get_recommended_duration(10.0),
        'Gamma (40 Hz)': get_recommended_duration(40.0),
    }
    
    for name, (min_dur, opt_dur) in durations.items():
        print(f"  {name}: {min_dur/60:.0f}-{opt_dur/60:.0f} dakika")
    print("  [OK]")
    
    # Test 9: Amplitude range
    print("\n[TEST 9] Amplitude Range")
    for amp in [0.1, 0.3, 0.5, 0.8, 1.0]:
        sig = generate_binaural_beats(
            carrier_freq=400.0,
            beat_freq=7.0,
            amplitude=amp,
            duration_sec=0.1,
            sample_rate=48000
        )
        peak = np.max(np.abs(sig))
        print(f"  Amplitude {amp:.1f} -> Peak {peak:.4f}")
    print("  [OK]")
    
    # Test 10: Zero duration
    print("\n[TEST 10] Zero Duration")
    zero = generate_binaural_beats(
        carrier_freq=400.0,
        beat_freq=7.0,
        amplitude=0.5,
        duration_sec=0.0,
        sample_rate=48000
    )
    
    assert zero.shape == (0, 2), "Boş stereo array olmalı"
    print(f"  Shape: {zero.shape}")
    print("  [OK]")
    
    print("\n" + "=" * 60)
    print("Tüm testler başarılı!")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("SAFE PARAMETER RECOMMENDATIONS")
    print("=" * 60)
    print("\nPriority #1: THETA (7 Hz)")
    print("  - Carrier: 400 Hz")
    print("  - Beat: 6-7 Hz")
    print("  - Duration: 10 minutes")
    print("  - Evidence: Medium-Weak (most researched)")
    
    print("\nPriority #2: ALPHA (10 Hz)")
    print("  - Carrier: 400 Hz")
    print("  - Beat: 10 Hz")
    print("  - Duration: 5 minutes")
    print("  - Evidence: Medium")
    
    print("\nExperimental: GAMMA (40 Hz)")
    print("  - Carrier: 250 Hz")
    print("  - Beat: 40 Hz")
    print("  - Duration: 20-30 minutes")
    print("  - Evidence: Weak-Medium")
    
    print("\nDANGER ZONES (DO NOT USE):")
    print("  - <3 Hz: Rotating tone")
    print("  - 13-30 Hz: Beta band (no evidence)")
    print("  - >50 Hz: Beat perception lost")
    print("=" * 60)
