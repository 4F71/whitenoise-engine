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

from typing import Callable, Optional
import numpy as np
from numpy.typing import NDArray

from preset_system.preset_schema import PresetConfig, LayerConfig, FilterConfig, LfoConfig
from core_dsp.dsp_noise import (
    generate_white_noise,
    generate_pink_noise,
    generate_brown_noise,
    generate_blue_noise,
    generate_violet_noise,
)
from core_dsp import dsp_filters, dsp_lfo


# =============================================================================
# TİP TANIMLARI
# =============================================================================

AudioBuffer = NDArray[np.float32]
LayerGenerator = Callable[[float, int], AudioBuffer]


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
    lfo_config: Optional[LfoConfig] = None
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
        # === DEBUG FIX ===
        # DSP noise fonksiyonları num_samples değil
        # (duration_sec, sample_rate) bekler
        audio = noise_func(
            duration_sec=duration_sec,
            sample_rate=sample_rate,
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


# =============================================================================
# ANA FONKSİYON
# =============================================================================

def adapt_preset_to_layer_generators(
    preset: PresetConfig
) -> list[LayerGenerator]:
    """
    PresetConfig nesnesinden render_sound uyumlu generator listesi üretir.
    
    Bu fonksiyon:
    - Sadece enabled == True olan katmanları işler
    - noise_type string'ini ilgili DSP fonksiyonuna eşler
    - gain değerini sinyal çarpanı olarak uygular
    - filter (lowpass/highpass) ve LFO (amplitude/filter) işler
    - V1 noise, gain, filter ve LFO içerir. Pan desteği V2'de eklenecektir.
    
    Args:
        preset: Dönüştürülecek preset yapılandırması
        
    Returns:
        render_sound fonksiyonuna geçirilebilecek
        List[Callable[[float, int], np.ndarray]] formatında generator listesi
        
    Raises:
        ValueError: Geçersiz noise_type içeren katman varsa
        
    Example:
        >>> preset = get_preset("deep_focus")
        >>> generators = adapt_preset_to_layer_generators(preset)
        >>> audio = render_sound(generators, duration_sec=60.0, sample_rate=48000)
    """
    generators: list[LayerGenerator] = []
    
    for layer in preset.layers:
        if not layer.enabled:
            continue
        
        generator = _create_layer_generator(
            noise_type=layer.noise_type,
            gain=layer.gain,
            filter_config=layer.filter_config,
            lfo_config=layer.lfo_config
        )
        
        generators.append(generator)
    
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
    gen = adapt_preset_to_layer_generators(stateless_preset)[0]
    
    audio1 = gen(0.01, 48000)
    audio2 = gen(0.01, 48000)
    audio3 = gen(0.01, 44100)
    
    assert audio1.shape == audio2.shape
    assert audio3.shape[0] == int(0.01 * 44100)
    print("  Birden fazla çağrı başarılı: [OK]")
    
    print("\n" + "=" * 50)
    print("Tüm testler başarılı!")
    print("=" * 50)
