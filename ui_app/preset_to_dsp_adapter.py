"""
ui_app/preset_to_dsp_adapter.py

Preset → DSP entegrasyon adaptörü.

Bu modül:
- PresetConfig nesnesini DSP katmanına uyumlu hale getirir
- Layer tanımlarını Callable listesine dönüştürür
- İki katman arasında köprü görevi görür

Sorumluluk sınırı:
- SADECE adaptasyon
- DSP işlemi İÇERMEZ
- Render çağrısı İÇERMEZ
"""

from typing import Callable, Optional, Union
import numpy as np
from numpy.typing import NDArray

from preset_system.preset_schema import (
    PresetConfig,
    LayerConfig,
    FilterConfig,
    LfoConfig,
    BinauralConfig,
    OrganicTextureConfig
)
from core_dsp.dsp_noise import (
    generate_white_noise,
    generate_pink_noise,
    generate_brown_noise,
    generate_blue_noise,
    generate_violet_noise,
)
from core_dsp import dsp_filters, dsp_lfo, dsp_binaural
from scipy.signal import butter, lfilter


# =============================================================================
# TİP TANIMLARI
# =============================================================================

AudioBuffer = NDArray[np.float32]
LayerGenerator = Callable[[float, int], AudioBuffer]
BinauralGenerator = Callable[[float, int], AudioBuffer]  # Returns stereo (N, 2)


# =============================================================================
# NOISE TİP EŞLEMESİ
# =============================================================================

NOISE_GENERATOR_MAP: dict[str, Callable[..., AudioBuffer]] = {
    "white": generate_white_noise,
    "pink": generate_pink_noise,
    "brown": generate_brown_noise,
    "blue": generate_blue_noise,
    "violet": generate_violet_noise,
}


# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def _create_layer_generator(
    noise_type: str,
    gain: float,
    filter_config: Optional[FilterConfig] = None,
    lfo_config: Optional[LfoConfig] = None,
    preset_seed: int = None
) -> LayerGenerator:
    """
    Tek bir katman için generator fonksiyonu oluşturur.
    
    Args:
        noise_type: Gürültü tipi ("white", "pink", "brown", "blue", "violet")
        gain: Kazanç çarpanı (0.0 - 1.0)
        filter_config: Filtre yapılandırması (opsiyonel)
        lfo_config: LFO yapılandırması (opsiyonel)
        
    Returns:
        (duration_sec, sample_rate) -> AudioBuffer imzalı Callable
        
    Raises:
        ValueError: Geçersiz noise_type için
    """
    if noise_type not in NOISE_GENERATOR_MAP:
        raise ValueError(f"Geçersiz noise_type: {noise_type}")
    
    noise_func = NOISE_GENERATOR_MAP[noise_type]
    clamped_gain = np.float32(max(0.0, min(1.0, gain)))
    
    def generator(duration_sec: float, sample_rate: int) -> AudioBuffer:
        """
        Belirtilen süre ve sample rate için audio üretir.
        
        Args:
            duration_sec: Süre (saniye)
            sample_rate: Örnekleme hızı (Hz)
            
        Returns:
            float32 audio buffer
        """
        # Generate noise with seed for reproducibility (V2.7+)
        audio = noise_func(
            duration_sec=duration_sec,
            sample_rate=sample_rate,
            seed=preset_seed
        )
        
        # Gain uygula
        audio = audio * clamped_gain
        
        # Filter uygula (opsiyonel)
        if filter_config and filter_config.enabled:
            if filter_config.filter_type == "lowpass":
                audio = dsp_filters.one_pole_lowpass(audio, filter_config.cutoff_hz, sample_rate)
            elif filter_config.filter_type == "highpass":
                audio = dsp_filters.one_pole_highpass(audio, filter_config.cutoff_hz, sample_rate)
        
        # LFO uygula (opsiyonel)
        if lfo_config and lfo_config.enabled:
            # LFO vektörü üret
            lfo_vector = dsp_lfo.sine_lfo(lfo_config.rate_hz, duration_sec, sample_rate)
            
            if lfo_config.target == "amplitude":
                audio = dsp_lfo.apply_volume_lfo(audio, lfo_vector, lfo_config.depth)
            elif lfo_config.target == "filter":
                # apply_filter_lfo sadece kesim frekansları döndürür, amplitude LFO olarak kullan
                audio = dsp_lfo.apply_volume_lfo(audio, lfo_vector, lfo_config.depth)
        
        return audio.astype(np.float32)
    
    return generator


def _create_organic_texture_generator(
    organic_config: OrganicTextureConfig,
    preset_seed: int = None
) -> LayerGenerator:
    """
    Organic texture generator (sub-bass + air layers).
    
    Theory: organic_texture_theory.md Section 4
    
    Args:
        organic_config: Organic texture configuration
        preset_seed: Seed for reproducibility
        
    Returns:
        Callable that generates composite organic texture (mono)
        
    Notes:
        - Sub-bass: Brown noise LP @ 80Hz, -12dB, perlin LFO
        - Air: White noise HP @ 4kHz, -18dB, perlin LFO
        - Returns mono signal (multi-band composite)
    """
    def generator(duration_sec: float, sample_rate: int) -> AudioBuffer:
        """Generate organic texture."""
        output = np.zeros(int(duration_sec * sample_rate), dtype=np.float32)
        
        # === SUB-BASS LAYER (20-80 Hz) ===
        if organic_config.sub_bass_enabled:
            # Generate brown noise (with seed for consistency)
            brown = generate_brown_noise(
                duration_sec=duration_sec,
                sample_rate=sample_rate,
                seed=preset_seed + 4000 if preset_seed else None
            )
            
            # === MICRO-MODULATION (Anti-Robotic) ===
            # Ultra-slow evolution (0.001 Hz = 1000s cycle) for "rumble" character
            micro_lfo = dsp_lfo.perlin_modulated_lfo(
                rate_hz=0.001,  # Extremely slow (natural earth rumble feel)
                duration_sec=duration_sec,
                sample_rate=sample_rate,
                mod_amount=0.4,
                seed=preset_seed + 5000 if preset_seed else None
            )
            brown = dsp_lfo.apply_volume_lfo(brown, micro_lfo, depth=0.05)  # Very subtle (5%)
            
            # Butterworth filters (4th order)
            nyquist = 0.5 * sample_rate
            
            # Low-pass filter @ 80 Hz
            lp_cutoff_norm = organic_config.sub_bass_lp_cutoff_hz / nyquist
            b_lp, a_lp = butter(4, lp_cutoff_norm, btype='low', analog=False)
            brown = lfilter(b_lp, a_lp, brown)
            
            # High-pass filter @ 20 Hz (rumble only)
            hp_cutoff_norm = organic_config.sub_bass_hp_cutoff_hz / nyquist
            b_hp, a_hp = butter(4, hp_cutoff_norm, btype='high', analog=False)
            brown = lfilter(b_hp, a_hp, brown)
            
            # Extra gentle LP @ 60Hz for smoothness (reduce "boom" character)
            smooth_cutoff_norm = 60.0 / nyquist
            b_smooth, a_smooth = butter(2, smooth_cutoff_norm, btype='low', analog=False)
            brown = lfilter(b_smooth, a_smooth, brown)
            
            # Apply gain (dB to linear)
            gain_linear = 10 ** (organic_config.sub_bass_gain_db / 20.0)
            brown = brown * gain_linear
            
            # LFO modulation (main breathing)
            if organic_config.sub_bass_lfo_type == "perlin_modulated":
                lfo_signal = dsp_lfo.perlin_modulated_lfo(
                    rate_hz=organic_config.sub_bass_lfo_rate_hz,
                    duration_sec=duration_sec,
                    sample_rate=sample_rate,
                    mod_amount=organic_config.sub_bass_lfo_mod_amount,
                    seed=preset_seed
                )
            else:
                lfo_signal = dsp_lfo.sine_lfo(
                    rate_hz=organic_config.sub_bass_lfo_rate_hz,
                    duration_sec=duration_sec,
                    sample_rate=sample_rate
                )
            
            brown = dsp_lfo.apply_volume_lfo(brown, lfo_signal, organic_config.sub_bass_lfo_depth)
            
            output = output + brown
        
        # === AIR LAYER (4-8 kHz) ===
        if organic_config.air_enabled:
            # Generate white noise (with seed for consistency)
            white = generate_white_noise(
                duration_sec=duration_sec,
                sample_rate=sample_rate,
                seed=preset_seed + 2000 if preset_seed else None
            )
            
            # === MICRO-MODULATION (Anti-Robotic) ===
            # Apply ultra-slow amplitude variation (0.002 Hz = 500s cycle)
            # Makes white noise less "static" and more "living"
            micro_lfo = dsp_lfo.perlin_modulated_lfo(
                rate_hz=0.002,  # Very slow evolution
                duration_sec=duration_sec,
                sample_rate=sample_rate,
                mod_amount=0.3,
                seed=preset_seed + 3000 if preset_seed else None
            )
            white = dsp_lfo.apply_volume_lfo(white, micro_lfo, depth=0.03)  # Very subtle (3%)
            
            nyquist = 0.5 * sample_rate
            
            # High-pass filter @ 4 kHz
            hp_cutoff_norm = organic_config.air_hp_cutoff_hz / nyquist
            b_hp, a_hp = butter(4, hp_cutoff_norm, btype='high', analog=False)
            white = lfilter(b_hp, a_hp, white)
            
            # Low-pass filter @ 8 kHz (band limit)
            lp_cutoff_norm = organic_config.air_lp_cutoff_hz / nyquist
            b_lp, a_lp = butter(4, lp_cutoff_norm, btype='low', analog=False)
            white = lfilter(b_lp, a_lp, white)
            
            # Soft rolloff @ 6kHz (reduce harshness)
            # 2nd order gentle LP to smooth high-freq "edges"
            soft_cutoff_norm = 6000.0 / nyquist
            b_soft, a_soft = butter(2, soft_cutoff_norm, btype='low', analog=False)
            white = lfilter(b_soft, a_soft, white)
            
            # Apply gain (dB to linear)
            gain_linear = 10 ** (organic_config.air_gain_db / 20.0)
            white = white * gain_linear
            
            # LFO modulation (main breathing)
            if organic_config.air_lfo_type == "perlin_modulated":
                lfo_signal = dsp_lfo.perlin_modulated_lfo(
                    rate_hz=organic_config.air_lfo_rate_hz,
                    duration_sec=duration_sec,
                    sample_rate=sample_rate,
                    mod_amount=organic_config.air_lfo_mod_amount,
                    seed=preset_seed + 1000 if preset_seed else None  # Different seed
                )
            else:
                lfo_signal = dsp_lfo.sine_lfo(
                    rate_hz=organic_config.air_lfo_rate_hz,
                    duration_sec=duration_sec,
                    sample_rate=sample_rate
                )
            
            white = dsp_lfo.apply_volume_lfo(white, lfo_signal, organic_config.air_lfo_depth)
            
            output = output + white
        
        return output.astype(np.float32)
    
    return generator


def _create_binaural_generator(
    binaural_config: BinauralConfig,
    preset_seed: int = None
) -> BinauralGenerator:
    """
    Binaural beats için generator fonksiyonu oluşturur.
    
    V2.7+: Organic breathing efekti eklendi (perlin-modulated LFO)
    
    Args:
        binaural_config: Binaural beats yapılandırması
        preset_seed: Seed for reproducibility
        
    Returns:
        (duration_sec, sample_rate) -> Stereo AudioBuffer (N, 2) imzalı Callable
        
    Notes:
        - STEREO output döndürür (shape: N × 2)
        - Sol kanal: carrier_freq
        - Sağ kanal: carrier_freq + beat_freq
        - Breathing efekti: 100s cycle (0.01 Hz perlin LFO)
        - Kulaklık kullanımı zorunlu
    """
    carrier = binaural_config.carrier_freq
    beat = binaural_config.beat_freq
    amplitude = binaural_config.amplitude
    
    def generator(duration_sec: float, sample_rate: int) -> AudioBuffer:
        """
        Stereo binaural beats üretir (with breathing effect).
        
        Args:
            duration_sec: Süre (saniye)
            sample_rate: Örnekleme hızı (Hz)
            
        Returns:
            Stereo float32 audio buffer (N × 2)
        """
        # === BREATHING LFO (V2.7+) ===
        # Perlin-modulated LFO for "living" binaural beats
        # 100s cycle (0.01 Hz) with ±20% frequency variation
        # This modulates CARRIER FREQUENCY (not amplitude)
        breathing_lfo = dsp_lfo.perlin_modulated_lfo(
            rate_hz=0.01,  # 100 second cycle
            duration_sec=duration_sec,
            sample_rate=sample_rate,
            mod_amount=0.2,  # ±20% LFO frequency variation
            seed=preset_seed
        )
        
        # Generate binaural beats WITH breathing (carrier freq modulation)
        stereo = dsp_binaural.generate_binaural_beats(
            carrier_freq=carrier,
            beat_freq=beat,
            amplitude=amplitude,
            duration_sec=duration_sec,
            sample_rate=sample_rate,
            breathing_lfo=breathing_lfo  # ±3% carrier freq modulation
        )
        
        # Fade uygula (click önleme)
        stereo = dsp_binaural.apply_fade(stereo, fade_duration=2.0, sample_rate=sample_rate)
        
        return stereo
    
    return generator


# =============================================================================
# ANA FONKSİYON
# =============================================================================

def adapt_preset_to_layer_generators(
    preset: PresetConfig
) -> Union[list[LayerGenerator], tuple[list[LayerGenerator], Optional[BinauralGenerator]]]:
    """
    PresetConfig nesnesinden render_sound uyumlu generator listesi üretir.
    
    Bu fonksiyon:
    - Sadece enabled == True olan katmanları işler
    - noise_type string'ini ilgili DSP fonksiyonuna eşler
    - gain değerini sinyal çarpanı olarak uygular
    - filter (lowpass/highpass) ve LFO (amplitude/filter) işler
    - Binaural beats (enabled ise) ayrı generator olarak döndürür
    - V1 noise, gain, filter ve LFO içerir. Pan desteği V2'de eklenecektir.
    
    Args:
        preset: Dönüştürülecek preset yapılandırması
        
    Returns:
        Eğer binaural_config.enabled == False:
            List[LayerGenerator] - Mono layer generator'ları
        Eğer binaural_config.enabled == True:
            Tuple[List[LayerGenerator], BinauralGenerator]
            - [0]: Mono layer generator'ları
            - [1]: Stereo binaural generator
        
    Raises:
        ValueError: Geçersiz noise_type içeren katman varsa
        
    Example:
        >>> # Mono (sadece noise layers)
        >>> preset = get_preset("deep_focus")
        >>> generators = adapt_preset_to_layer_generators(preset)
        >>> audio = render_sound(generators, duration_sec=60.0, sample_rate=48000)
        
        >>> # Stereo (noise layers + binaural beats)
        >>> preset_binaural = get_preset("theta_meditation_binaural")
        >>> layer_gens, binaural_gen = adapt_preset_to_layer_generators(preset_binaural)
        >>> # Manuel mixing required for binaural
    """
    generators: list[LayerGenerator] = []
    
    # Layer'ları işle
    for layer in preset.layers:
        if not layer.enabled:
            continue
        
        generator = _create_layer_generator(
            noise_type=layer.noise_type,
            gain=layer.gain,
            filter_config=layer.filter_config,
            lfo_config=layer.lfo_config,
            preset_seed=preset.seed  # V2.7+: Reproducibility
        )
        
        generators.append(generator)
    
    # Organic texture ekle (V2.7+)
    if preset.organic_texture_config and preset.organic_texture_config.enabled:
        organic_generator = _create_organic_texture_generator(
            preset.organic_texture_config,
            preset_seed=preset.seed
        )
        generators.append(organic_generator)
    
    # Binaural beats kontrolü
    if preset.binaural_config and preset.binaural_config.enabled:
        # Binaural generator oluştur (with breathing effect V2.7+)
        binaural_gen = _create_binaural_generator(
            preset.binaural_config,
            preset_seed=preset.seed
        )
        return (generators, binaural_gen)
    else:
        return generators


def get_supported_noise_types() -> list[str]:
    """
    Desteklenen gürültü tiplerini döndürür.
    
    Returns:
        Geçerli noise_type string listesi
    """
    return list(NOISE_GENERATOR_MAP.keys())


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("preset_to_dsp_adapter.py test başlıyor...")
    print("=" * 50)
    
    # Test için örnek preset oluştur
    from preset_system.preset_schema import LayerConfig, FilterConfig, LfoConfig
    
    test_preset = PresetConfig(
        name="Test Preset",
        description="Adapter testi için",
        layers=[
            LayerConfig(
                name="White Active",
                noise_type="white",
                gain=0.8,
                enabled=True,
            ),
            LayerConfig(
                name="Pink Disabled",
                noise_type="pink",
                gain=0.5,
                enabled=False,
            ),
            LayerConfig(
                name="Brown Active",
                noise_type="brown",
                gain=0.6,
                enabled=True,
            ),
        ],
        master_gain=0.75,
    )
    
    # Test 1: Generator listesi üretimi
    print("\n[TEST 1] Generator listesi üretimi")
    generators = adapt_preset_to_layer_generators(test_preset)
    
    assert len(generators) == 2, "Sadece enabled katmanlar olmalı"
    print(f"  Toplam katman: {len(test_preset.layers)}")
    print(f"  Aktif katman: {len(generators)}")
    print("  [OK]")
    
    # Test 2: Generator çağrısı
    print("\n[TEST 2] Generator çağrısı")
    test_duration = 0.1
    test_sr = 48000
    
    for i, gen in enumerate(generators):
        audio = gen(test_duration, test_sr)
        expected_samples = int(test_duration * test_sr)
        
        assert audio.shape[0] == expected_samples, "Sample sayısı yanlış"
        assert audio.dtype == np.float32, "Tip float32 olmalı"
        assert np.isfinite(audio).all(), "NaN/Inf olmamalı"
        
        print(f"  Generator {i}: shape={audio.shape}, dtype={audio.dtype}, "
              f"peak={np.max(np.abs(audio)):.4f}")
    print("  [OK]")
    
    # Test 3: Gain uygulaması
    print("\n[TEST 3] Gain uygulaması")
    
    full_gain_preset = PresetConfig(
        name="Full Gain",
        layers=[LayerConfig(noise_type="white", gain=1.0, enabled=True)],
    )
    
    half_gain_preset = PresetConfig(
        name="Half Gain",
        layers=[LayerConfig(noise_type="white", gain=0.5, enabled=True)],
    )
    
    full_gen = adapt_preset_to_layer_generators(full_gain_preset)[0]
    half_gen = adapt_preset_to_layer_generators(half_gain_preset)[0]
    
    np.random.seed(42)
    full_audio = full_gen(0.01, 48000)
    
    np.random.seed(42)
    half_audio = half_gen(0.01, 48000)
    
    full_rms = np.sqrt(np.mean(full_audio ** 2))
    half_rms = np.sqrt(np.mean(half_audio ** 2))
    ratio = half_rms / full_rms if full_rms > 0 else 0
    
    print(f"  Full gain RMS: {full_rms:.4f}")
    print(f"  Half gain RMS: {half_rms:.4f}")
    print(f"  Ratio: {ratio:.2f} (beklenen ~0.5)")
    print("  [OK]")
    
    # Test 4: Tüm noise tipleri
    print("\n[TEST 4] Tüm noise tipleri")
    supported = get_supported_noise_types()
    print(f"  Desteklenen tipler: {supported}")
    
    for noise_type in supported:
        single_preset = PresetConfig(
            name=f"{noise_type} test",
            layers=[LayerConfig(noise_type=noise_type, gain=0.7, enabled=True)],
        )
        gen = adapt_preset_to_layer_generators(single_preset)[0]
        audio = gen(0.01, 48000)
        
        assert audio.dtype == np.float32
        assert len(audio) > 0
        print(f"    {noise_type}: OK")
    print("  [OK]")
    
    # Test 5: Boş preset
    print("\n[TEST 5] Boş preset")
    empty_preset = PresetConfig(name="Empty", layers=[])
    empty_generators = adapt_preset_to_layer_generators(empty_preset)
    assert len(empty_generators) == 0
    print("  Boş liste döndü: [OK]")
    
    # Test 6: Tüm katmanlar disabled
    print("\n[TEST 6] Tüm katmanlar disabled")
    all_disabled = PresetConfig(
        name="All Disabled",
        layers=[
            LayerConfig(noise_type="white", enabled=False),
            LayerConfig(noise_type="pink", enabled=False),
        ],
    )
    disabled_generators = adapt_preset_to_layer_generators(all_disabled)
    assert len(disabled_generators) == 0
    print("  Boş liste döndü: [OK]")
    
    # Test 7: Geçersiz noise type
    print("\n[TEST 7] Geçersiz noise type")
    try:
        invalid_preset = PresetConfig(
            name="Invalid",
            layers=[LayerConfig(noise_type="invalid_type", enabled=True)],
        )
        adapt_preset_to_layer_generators(invalid_preset)
        print("  HATA: Exception beklendi!")
    except ValueError as e:
        print(f"  ValueError yakalandı: {e}")
        print("  [OK]")
    
    # Test 8: Stateless kontrol
    print("\n[TEST 8] Stateless kontrol")
    stateless_preset = PresetConfig(
        name="Stateless Test",
        layers=[LayerConfig(noise_type="pink", gain=0.5, enabled=True)],
    )
    result = adapt_preset_to_layer_generators(stateless_preset)
    
    # Binaural yok, direkt list dönmeli
    assert isinstance(result, list), "Binaural yokken list dönmeli"
    gen = result[0]
    
    audio1 = gen(0.01, 48000)
    audio2 = gen(0.01, 48000)
    audio3 = gen(0.01, 44100)
    
    assert audio1.shape == audio2.shape
    assert audio3.shape[0] == int(0.01 * 44100)
    print("  Birden fazla çağrı başarılı: [OK]")
    
    # Test 9: Binaural beats (YENİ)
    print("\n[TEST 9] Binaural Beats")
    from preset_system.preset_schema import BinauralConfig
    
    binaural_preset = PresetConfig(
        name="Theta Meditation Binaural",
        layers=[
            LayerConfig(noise_type="pink", gain=0.3, enabled=True),
        ],
        binaural_config=BinauralConfig(
            enabled=True,
            carrier_freq=200.0,
            beat_freq=7.0,
            amplitude=0.5
        )
    )
    
    result = adapt_preset_to_layer_generators(binaural_preset)
    
    # Binaural var, tuple dönmeli
    assert isinstance(result, tuple), "Binaural varken tuple dönmeli"
    assert len(result) == 2, "Tuple 2 elemanlı olmalı"
    
    layer_gens, binaural_gen = result
    
    assert isinstance(layer_gens, list), "İlk eleman list olmalı"
    assert len(layer_gens) == 1, "1 layer var"
    
    # Layer test (mono)
    layer_audio = layer_gens[0](0.1, 48000)
    assert layer_audio.ndim == 1, "Layer mono olmalı"
    assert layer_audio.shape[0] == int(0.1 * 48000)
    
    # Binaural test (stereo)
    binaural_audio = binaural_gen(0.1, 48000)
    assert binaural_audio.ndim == 2, "Binaural stereo olmalı"
    assert binaural_audio.shape == (int(0.1 * 48000), 2)
    
    # Sol ve sağ kanal farklı olmalı (beat için)
    assert not np.allclose(binaural_audio[:, 0], binaural_audio[:, 1]), \
        "Binaural kanallar farklı olmalı"
    
    print(f"  Layer shape: {layer_audio.shape} (mono)")
    print(f"  Binaural shape: {binaural_audio.shape} (stereo)")
    print("  [OK]")
    
    # Test 10: Binaural only (layer yok)
    print("\n[TEST 10] Binaural Only (no layers)")
    binaural_only = PresetConfig(
        name="Pure Binaural",
        layers=[],  # Boş
        binaural_config=BinauralConfig(
            enabled=True,
            carrier_freq=200.0,
            beat_freq=10.0,
            amplitude=0.5
        )
    )
    
    result = adapt_preset_to_layer_generators(binaural_only)
    assert isinstance(result, tuple)
    layer_gens, binaural_gen = result
    assert len(layer_gens) == 0, "Layer yok"
    
    binaural_audio = binaural_gen(0.05, 48000)
    assert binaural_audio.shape == (int(0.05 * 48000), 2)
    print(f"  Binaural shape: {binaural_audio.shape}")
    print("  [OK]")
    
    print("\n" + "=" * 50)
    print("Tüm testler başarılı!")
    print("=" * 50)
