"""
preset_system/preset_library.py

Hazır preset koleksiyonu ve erişim fonksiyonları.

Bu modül:
- Kullanıma hazır preset tanımları içerir
- Preset arama ve filtreleme sağlar
- Kategorize edilmiş erişim sunar

Sorumluluk sınırı:
- SADECE preset tanımları ve erişim
- DSP işlemi İÇERMEZ
- Ses üretimi İÇERMEZ
"""

from typing import Optional
from preset_system.preset_schema import (
    PresetConfig,
    LayerConfig,
    FilterConfig,
    LfoConfig,
    FxConfig,
)


# =============================================================================
# KATEGORİ SABİTLERİ
# =============================================================================

CATEGORY_FOCUS: str = "focus"
CATEGORY_SLEEP: str = "sleep"
CATEGORY_MEDITATION: str = "meditation"
CATEGORY_NATURE: str = "nature"
CATEGORY_AMBIENT: str = "ambient"
CATEGORY_PRODUCTIVITY: str = "productivity"


# =============================================================================
# PRESET TANIMLARI
# =============================================================================

def _create_pure_white_noise() -> PresetConfig:
    """Saf beyaz gürültü preset'i."""
    return PresetConfig(
        name="Pure White Noise",
        description="Klasik beyaz gürültü. Tüm frekanslar eşit güçte. "
                    "Keskin ve net bir ses karakteri.",
        author="UltraGen",
        version="1.0",
        tags=[CATEGORY_FOCUS, CATEGORY_PRODUCTIVITY, "classic", "white"],
        layers=[
            LayerConfig(
                name="White Core",
                noise_type="white",
                gain=0.85,
                pan=0.0,
            ),
        ],
        master_gain=0.75,
        duration_sec=3600.0,
        seed=1001,
    )


def _create_soft_pink_noise() -> PresetConfig:
    """Yumuşak pembe gürültü preset'i."""
    return PresetConfig(
        name="Soft Pink Noise",
        description="Doğal ve yumuşak pembe gürültü. Düşük frekanslara doğru "
                    "azalan enerji. İnsan kulağına en doğal gelen gürültü tipi.",
        author="UltraGen",
        version="1.0",
        tags=[CATEGORY_SLEEP, CATEGORY_MEDITATION, "soft", "pink"],
        layers=[
            LayerConfig(
                name="Pink Foundation",
                noise_type="pink",
                gain=0.9,
                pan=0.0,
                filter_config=FilterConfig(
                    enabled=True,
                    filter_type="lowpass",
                    cutoff_hz=8000.0,
                    resonance_q=0.5,
                ),
            ),
        ],
        master_gain=0.7,
        fx_config=FxConfig(
            stereo_width=1.1,
        ),
        duration_sec=3600.0,
        seed=1002,
    )


def _create_deep_brown_noise() -> PresetConfig:
    """Derin kahverengi gürültü preset'i."""
    return PresetConfig(
        name="Deep Brown Noise",
        description="Derin ve zengin kahverengi gürültü. Bas frekanslar baskın. "
                    "Uzak gök gürültüsü veya şelale hissi verir.",
        author="UltraGen",
        version="1.0",
        tags=[CATEGORY_SLEEP, CATEGORY_AMBIENT, "deep", "brown", "bass"],
        layers=[
            LayerConfig(
                name="Brown Deep",
                noise_type="brown",
                gain=0.95,
                pan=0.0,
                filter_config=FilterConfig(
                    enabled=True,
                    filter_type="lowpass",
                    cutoff_hz=2000.0,
                    resonance_q=0.6,
                ),
            ),
        ],
        master_gain=0.8,
        fx_config=FxConfig(
            stereo_width=1.2,
            saturation_amount=0.02,
        ),
        duration_sec=3600.0,
        seed=1003,
    )


def _create_bright_blue_noise() -> PresetConfig:
    """Parlak mavi gürültü preset'i."""
    return PresetConfig(
        name="Bright Blue Noise",
        description="Yüksek frekanslara vurgulu mavi gürültü. "
                    "Tiz ve parlak karakter. Tinnitus maskeleme için uygun.",
        author="UltraGen",
        version="1.0",
        tags=[CATEGORY_FOCUS, "bright", "blue", "tinnitus"],
        layers=[
            LayerConfig(
                name="Blue Bright",
                noise_type="blue",
                gain=0.7,
                pan=0.0,
                filter_config=FilterConfig(
                    enabled=True,
                    filter_type="highpass",
                    cutoff_hz=500.0,
                    resonance_q=0.5,
                ),
            ),
        ],
        master_gain=0.6,
        duration_sec=3600.0,
        seed=1004,
    )


def _create_sharp_violet_noise() -> PresetConfig:
    """Keskin violet gürültü preset'i."""
    return PresetConfig(
        name="Sharp Violet Noise",
        description="En yüksek frekanslara vurgulu violet gürültü. "
                    "Çok parlak ve keskin. Dikkat gerektiren işler için.",
        author="UltraGen",
        version="1.0",
        tags=[CATEGORY_FOCUS, CATEGORY_PRODUCTIVITY, "sharp", "violet", "bright"],
        layers=[
            LayerConfig(
                name="Violet Sharp",
                noise_type="violet",
                gain=0.6,
                pan=0.0,
                filter_config=FilterConfig(
                    enabled=True,
                    filter_type="highpass",
                    cutoff_hz=1000.0,
                    resonance_q=0.4,
                ),
            ),
        ],
        master_gain=0.55,
        duration_sec=3600.0,
        seed=1005,
    )


def _create_deep_focus() -> PresetConfig:
    """Derin odaklanma preset'i."""
    return PresetConfig(
        name="Deep Focus",
        description="Derin konsantrasyon için optimize edilmiş çok katmanlı gürültü. "
                    "Brown temel üzerine hafif pink doku. Uzun çalışma seansları için ideal.",
        author="UltraGen",
        version="1.0",
        tags=[CATEGORY_FOCUS, CATEGORY_PRODUCTIVITY, "layered", "work", "study"],
        layers=[
            LayerConfig(
                name="Brown Foundation",
                noise_type="brown",
                gain=0.65,
                pan=-0.1,
                filter_config=FilterConfig(
                    enabled=True,
                    filter_type="lowpass",
                    cutoff_hz=1500.0,
                    resonance_q=0.5,
                ),
                lfo_config=LfoConfig(
                    enabled=True,
                    waveform="sine",
                    target="amplitude",
                    rate_hz=0.02,
                    depth=0.08,
                ),
            ),
            LayerConfig(
                name="Pink Texture",
                noise_type="pink",
                gain=0.25,
                pan=0.1,
                filter_config=FilterConfig(
                    enabled=True,
                    filter_type="bandpass",
                    cutoff_hz=2000.0,
                    resonance_q=0.8,
                ),
            ),
        ],
        master_gain=0.75,
        fx_config=FxConfig(
            stereo_width=1.15,
            saturation_amount=0.01,
        ),
        duration_sec=7200.0,
        seed=2001,
    )


def _create_sleep_cocoon() -> PresetConfig:
    """Uyku koza preset'i."""
    return PresetConfig(
        name="Sleep Cocoon",
        description="Derin uyku için sarmalayıcı ses ortamı. "
                    "Düşük frekans ağırlıklı, yavaş modülasyonlu. "
                    "Beyin dalgalarını yavaşlatmaya yardımcı.",
        author="UltraGen",
        version="1.0",
        tags=[CATEGORY_SLEEP, "relaxing", "night", "rest", "layered"],
        layers=[
            LayerConfig(
                name="Brown Blanket",
                noise_type="brown",
                gain=0.7,
                pan=0.0,
                filter_config=FilterConfig(
                    enabled=True,
                    filter_type="lowpass",
                    cutoff_hz=800.0,
                    resonance_q=0.4,
                ),
                lfo_config=LfoConfig(
                    enabled=True,
                    waveform="sine",
                    target="amplitude",
                    rate_hz=0.01,
                    depth=0.12,
                ),
            ),
            LayerConfig(
                name="Pink Whisper",
                noise_type="pink",
                gain=0.15,
                pan=0.0,
                filter_config=FilterConfig(
                    enabled=True,
                    filter_type="lowpass",
                    cutoff_hz=1200.0,
                    resonance_q=0.5,
                ),
                lfo_config=LfoConfig(
                    enabled=True,
                    waveform="sine",
                    target="amplitude",
                    rate_hz=0.015,
                    depth=0.1,
                    phase_offset=0.5,
                ),
            ),
        ],
        master_gain=0.65,
        fx_config=FxConfig(
            stereo_width=1.3,
            saturation_amount=0.02,
        ),
        duration_sec=28800.0,  # 8 saat
        seed=2002,
    )


def _create_meditation_space() -> PresetConfig:
    """Meditasyon alanı preset'i."""
    return PresetConfig(
        name="Meditation Space",
        description="Meditasyon ve mindfulness için geniş ses alanı. "
                    "Dengeli frekans dağılımı, nefes ritmiyle uyumlu modülasyon. "
                    "Zihinsel berraklık ve huzur için tasarlandı.",
        author="UltraGen",
        version="1.0",
        tags=[CATEGORY_MEDITATION, "mindfulness", "calm", "breath", "layered"],
        layers=[
            LayerConfig(
                name="Pink Ground",
                noise_type="pink",
                gain=0.5,
                pan=0.0,
                filter_config=FilterConfig(
                    enabled=True,
                    filter_type="bandpass",
                    cutoff_hz=800.0,
                    resonance_q=0.6,
                ),
                lfo_config=LfoConfig(
                    enabled=True,
                    waveform="sine",
                    target="amplitude",
                    rate_hz=0.067,  # ~4 nefes/dakika
                    depth=0.15,
                ),
            ),
            LayerConfig(
                name="Brown Depth",
                noise_type="brown",
                gain=0.35,
                pan=0.0,
                filter_config=FilterConfig(
                    enabled=True,
                    filter_type="lowpass",
                    cutoff_hz=400.0,
                    resonance_q=0.5,
                ),
            ),
            LayerConfig(
                name="White Air",
                noise_type="white",
                gain=0.08,
                pan=0.0,
                filter_config=FilterConfig(
                    enabled=True,
                    filter_type="highpass",
                    cutoff_hz=4000.0,
                    resonance_q=0.3,
                ),
            ),
        ],
        master_gain=0.7,
        fx_config=FxConfig(
            stereo_width=1.4,
        ),
        duration_sec=3600.0,
        seed=2003,
    )


def _create_rain_simulation() -> PresetConfig:
    """Yağmur simülasyonu preset'i."""
    return PresetConfig(
        name="Rain Simulation",
        description="Gerçekçi yağmur ses simülasyonu. "
                    "Pink ve white noise karışımı ile damla hissi. "
                    "Pencere camına vuran yağmur etkisi.",
        author="UltraGen",
        version="1.0",
        tags=[CATEGORY_NATURE, CATEGORY_SLEEP, "rain", "weather", "layered"],
        layers=[
            LayerConfig(
                name="Rain Body",
                noise_type="pink",
                gain=0.55,
                pan=0.0,
                filter_config=FilterConfig(
                    enabled=True,
                    filter_type="bandpass",
                    cutoff_hz=1500.0,
                    resonance_q=1.2,
                ),
                lfo_config=LfoConfig(
                    enabled=True,
                    waveform="random",
                    target="amplitude",
                    rate_hz=0.3,
                    depth=0.2,
                ),
            ),
            LayerConfig(
                name="Rain Drops",
                noise_type="white",
                gain=0.2,
                pan=0.0,
                filter_config=FilterConfig(
                    enabled=True,
                    filter_type="highpass",
                    cutoff_hz=3000.0,
                    resonance_q=0.8,
                ),
                lfo_config=LfoConfig(
                    enabled=True,
                    waveform="random",
                    target="amplitude",
                    rate_hz=0.8,
                    depth=0.35,
                ),
            ),
            LayerConfig(
                name="Distant Thunder",
                noise_type="brown",
                gain=0.25,
                pan=0.0,
                filter_config=FilterConfig(
                    enabled=True,
                    filter_type="lowpass",
                    cutoff_hz=300.0,
                    resonance_q=0.6,
                ),
                lfo_config=LfoConfig(
                    enabled=True,
                    waveform="random",
                    target="amplitude",
                    rate_hz=0.05,
                    depth=0.4,
                ),
            ),
        ],
        master_gain=0.75,
        fx_config=FxConfig(
            stereo_width=1.3,
            saturation_amount=0.01,
        ),
        duration_sec=7200.0,
        seed=3001,
    )


def _create_ocean_waves() -> PresetConfig:
    """Okyanus dalgaları preset'i."""
    return PresetConfig(
        name="Ocean Waves",
        description="Sakin okyanus dalgaları simülasyonu. "
                    "Ritmik dalga hareketi ile huzur verici atmosfer. "
                    "Kıyıda oturma hissi.",
        author="UltraGen",
        version="1.0",
        tags=[CATEGORY_NATURE, CATEGORY_MEDITATION, "ocean", "waves", "beach"],
        layers=[
            LayerConfig(
                name="Wave Body",
                noise_type="brown",
                gain=0.6,
                pan=0.0,
                filter_config=FilterConfig(
                    enabled=True,
                    filter_type="lowpass",
                    cutoff_hz=600.0,
                    resonance_q=0.7,
                ),
                lfo_config=LfoConfig(
                    enabled=True,
                    waveform="sine",
                    target="amplitude",
                    rate_hz=0.08,  # ~5 dalga/dakika
                    depth=0.5,
                ),
            ),
            LayerConfig(
                name="Wave Foam",
                noise_type="pink",
                gain=0.3,
                pan=0.0,
                filter_config=FilterConfig(
                    enabled=True,
                    filter_type="bandpass",
                    cutoff_hz=2500.0,
                    resonance_q=1.0,
                ),
                lfo_config=LfoConfig(
                    enabled=True,
                    waveform="sine",
                    target="amplitude",
                    rate_hz=0.08,
                    depth=0.6,
                    phase_offset=0.3,
                ),
            ),
            LayerConfig(
                name="Shore Hiss",
                noise_type="white",
                gain=0.1,
                pan=0.0,
                filter_config=FilterConfig(
                    enabled=True,
                    filter_type="highpass",
                    cutoff_hz=5000.0,
                    resonance_q=0.4,
                ),
                lfo_config=LfoConfig(
                    enabled=True,
                    waveform="sine",
                    target="amplitude",
                    rate_hz=0.08,
                    depth=0.7,
                    phase_offset=0.6,
                ),
            ),
        ],
        master_gain=0.7,
        fx_config=FxConfig(
            stereo_width=1.5,
            saturation_amount=0.015,
        ),
        duration_sec=7200.0,
        seed=3002,
    )


def _create_forest_wind() -> PresetConfig:
    """Orman rüzgarı preset'i."""
    return PresetConfig(
        name="Forest Wind",
        description="Ormanda esen rüzgar simülasyonu. "
                    "Yaprakların hışırtısı ve dal sallanması. "
                    "Doğayla bağlantı hissi.",
        author="UltraGen",
        version="1.0",
        tags=[CATEGORY_NATURE, CATEGORY_AMBIENT, "forest", "wind", "leaves"],
        layers=[
            LayerConfig(
                name="Wind Base",
                noise_type="pink",
                gain=0.45,
                pan=0.0,
                filter_config=FilterConfig(
                    enabled=True,
                    filter_type="bandpass",
                    cutoff_hz=500.0,
                    resonance_q=0.8,
                ),
                lfo_config=LfoConfig(
                    enabled=True,
                    waveform="sine",
                    target="amplitude",
                    rate_hz=0.04,
                    depth=0.3,
                ),
            ),
            LayerConfig(
                name="Leaves Rustle",
                noise_type="white",
                gain=0.25,
                pan=0.15,
                filter_config=FilterConfig(
                    enabled=True,
                    filter_type="bandpass",
                    cutoff_hz=3500.0,
                    resonance_q=1.5,
                ),
                lfo_config=LfoConfig(
                    enabled=True,
                    waveform="random",
                    target="amplitude",
                    rate_hz=0.5,
                    depth=0.4,
                ),
            ),
            LayerConfig(
                name="Branch Creaks",
                noise_type="brown",
                gain=0.2,
                pan=-0.15,
                filter_config=FilterConfig(
                    enabled=True,
                    filter_type="lowpass",
                    cutoff_hz=250.0,
                    resonance_q=0.9,
                ),
                lfo_config=LfoConfig(
                    enabled=True,
                    waveform="random",
                    target="amplitude",
                    rate_hz=0.1,
                    depth=0.35,
                ),
            ),
        ],
        master_gain=0.7,
        fx_config=FxConfig(
            stereo_width=1.4,
        ),
        duration_sec=7200.0,
        seed=3003,
    )


def _create_space_ambient() -> PresetConfig:
    """Uzay ambiyansı preset'i."""
    return PresetConfig(
        name="Space Ambient",
        description="Derin uzay ambiyansı. Yıldızlararası boşluk hissi. "
                    "Geniş, kaotik olmayan, meditatif ses manzarası.",
        author="UltraGen",
        version="1.0",
        tags=[CATEGORY_AMBIENT, CATEGORY_MEDITATION, "space", "cosmic", "ethereal"],
        layers=[
            LayerConfig(
                name="Cosmic Hum",
                noise_type="brown",
                gain=0.5,
                pan=0.0,
                filter_config=FilterConfig(
                    enabled=True,
                    filter_type="lowpass",
                    cutoff_hz=200.0,
                    resonance_q=1.2,
                ),
                lfo_config=LfoConfig(
                    enabled=True,
                    waveform="sine",
                    target="amplitude",
                    rate_hz=0.005,
                    depth=0.2,
                ),
            ),
            LayerConfig(
                name="Star Field",
                noise_type="pink",
                gain=0.2,
                pan=0.0,
                filter_config=FilterConfig(
                    enabled=True,
                    filter_type="bandpass",
                    cutoff_hz=1000.0,
                    resonance_q=0.5,
                ),
                lfo_config=LfoConfig(
                    enabled=True,
                    waveform="sine",
                    target="pan",
                    rate_hz=0.02,
                    depth=0.6,
                ),
            ),
            LayerConfig(
                name="Solar Wind",
                noise_type="blue",
                gain=0.1,
                pan=0.0,
                filter_config=FilterConfig(
                    enabled=True,
                    filter_type="highpass",
                    cutoff_hz=6000.0,
                    resonance_q=0.3,
                ),
                lfo_config=LfoConfig(
                    enabled=True,
                    waveform="random",
                    target="amplitude",
                    rate_hz=0.15,
                    depth=0.25,
                ),
            ),
        ],
        master_gain=0.65,
        fx_config=FxConfig(
            stereo_width=1.8,
            saturation_amount=0.01,
        ),
        duration_sec=7200.0,
        seed=4001,
    )


def _create_cafe_ambience() -> PresetConfig:
    """Kafe ambiyansı preset'i."""
    return PresetConfig(
        name="Cafe Ambience",
        description="Sakin bir kafe ortamı simülasyonu. "
                    "Arka plan uğultusu ve hafif aktivite sesleri. "
                    "Evden çalışanlar için sosyal ortam hissi.",
        author="UltraGen",
        version="1.0",
        tags=[CATEGORY_PRODUCTIVITY, CATEGORY_AMBIENT, "cafe", "social", "work"],
        layers=[
            LayerConfig(
                name="Crowd Murmur",
                noise_type="pink",
                gain=0.4,
                pan=0.0,
                filter_config=FilterConfig(
                    enabled=True,
                    filter_type="bandpass",
                    cutoff_hz=700.0,
                    resonance_q=1.0,
                ),
                lfo_config=LfoConfig(
                    enabled=True,
                    waveform="random",
                    target="amplitude",
                    rate_hz=0.2,
                    depth=0.15,
                ),
            ),
            LayerConfig(
                name="Activity",
                noise_type="white",
                gain=0.15,
                pan=0.2,
                filter_config=FilterConfig(
                    enabled=True,
                    filter_type="bandpass",
                    cutoff_hz=2000.0,
                    resonance_q=1.2,
                ),
                lfo_config=LfoConfig(
                    enabled=True,
                    waveform="random",
                    target="amplitude",
                    rate_hz=0.4,
                    depth=0.3,
                ),
            ),
            LayerConfig(
                name="Room Tone",
                noise_type="brown",
                gain=0.25,
                pan=-0.1,
                filter_config=FilterConfig(
                    enabled=True,
                    filter_type="lowpass",
                    cutoff_hz=400.0,
                    resonance_q=0.5,
                ),
            ),
        ],
        master_gain=0.6,
        fx_config=FxConfig(
            stereo_width=1.2,
        ),
        duration_sec=7200.0,
        seed=4002,
    )


# =============================================================================
# PRESET KÜTÜPHANESİ
# =============================================================================

_PRESET_LIBRARY: dict[str, PresetConfig] = {
    # Temel gürültüler
    "pure_white_noise": _create_pure_white_noise(),
    "soft_pink_noise": _create_soft_pink_noise(),
    "deep_brown_noise": _create_deep_brown_noise(),
    "bright_blue_noise": _create_bright_blue_noise(),
    "sharp_violet_noise": _create_sharp_violet_noise(),
    # Odaklanma ve üretkenlik
    "deep_focus": _create_deep_focus(),
    "cafe_ambience": _create_cafe_ambience(),
    # Uyku
    "sleep_cocoon": _create_sleep_cocoon(),
    # Meditasyon
    "meditation_space": _create_meditation_space(),
    # Doğa
    "rain_simulation": _create_rain_simulation(),
    "ocean_waves": _create_ocean_waves(),
    "forest_wind": _create_forest_wind(),
    # Ambiyans
    "space_ambient": _create_space_ambient(),
}


# =============================================================================
# ERİŞİM FONKSİYONLARI
# =============================================================================

def get_preset(preset_id: str) -> Optional[PresetConfig]:
    """
    ID ile preset getirir.
    
    Args:
        preset_id: Preset tanımlayıcısı
        
    Returns:
        PresetConfig veya bulunamazsa None
    """
    return _PRESET_LIBRARY.get(preset_id)


def get_preset_by_name(name: str) -> Optional[PresetConfig]:
    """
    İsim ile preset getirir.
    
    Args:
        name: Preset adı (tam eşleşme)
        
    Returns:
        PresetConfig veya bulunamazsa None
    """
    for preset in _PRESET_LIBRARY.values():
        if preset.name == name:
            return preset
    return None


def list_all_presets() -> list[str]:
    """
    Tüm preset ID'lerini listeler.
    
    Returns:
        Preset ID listesi
    """
    return list(_PRESET_LIBRARY.keys())


def list_presets_by_tag(tag: str) -> list[str]:
    """
    Belirli bir etikete sahip preset ID'lerini listeler.
    
    Args:
        tag: Aranacak etiket
        
    Returns:
        Eşleşen preset ID listesi
    """
    results: list[str] = []
    for preset_id, preset in _PRESET_LIBRARY.items():
        if tag in preset.tags:
            results.append(preset_id)
    return results


def list_presets_by_category(category: str) -> list[str]:
    """
    Kategoriye göre preset ID'lerini listeler.
    
    Ana kategoriler:
    - focus: Odaklanma
    - sleep: Uyku
    - meditation: Meditasyon
    - nature: Doğa
    - ambient: Ambiyans
    - productivity: Üretkenlik
    
    Args:
        category: Kategori adı
        
    Returns:
        Eşleşen preset ID listesi
    """
    return list_presets_by_tag(category)


def search_presets(query: str) -> list[str]:
    """
    İsim veya açıklamada arama yapar.
    
    Args:
        query: Arama sorgusu (büyük/küçük harf duyarsız)
        
    Returns:
        Eşleşen preset ID listesi
    """
    query_lower = query.lower()
    results: list[str] = []
    
    for preset_id, preset in _PRESET_LIBRARY.items():
        if (query_lower in preset.name.lower() or 
            query_lower in preset.description.lower()):
            results.append(preset_id)
    
    return results


def get_all_tags() -> list[str]:
    """
    Tüm benzersiz etiketleri listeler.
    
    Returns:
        Alfabetik sıralı etiket listesi
    """
    tags: set[str] = set()
    for preset in _PRESET_LIBRARY.values():
        tags.update(preset.tags)
    return sorted(tags)


def get_preset_info(preset_id: str) -> Optional[dict[str, str]]:
    """
    Preset hakkında özet bilgi döndürür.
    
    Args:
        preset_id: Preset tanımlayıcısı
        
    Returns:
        Özet bilgi dict'i veya None
    """
    preset = get_preset(preset_id)
    if preset is None:
        return None
    
    return {
        "id": preset_id,
        "name": preset.name,
        "description": preset.description,
        "author": preset.author,
        "tags": ", ".join(preset.tags),
        "layer_count": str(len(preset.layers)),
        "duration_minutes": str(int(preset.duration_sec / 60)),
    }


def get_preset_count() -> int:
    """
    Toplam preset sayısını döndürür.
    
    Returns:
        Preset sayısı
    """
    return len(_PRESET_LIBRARY)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("preset_library.py test başlıyor...")
    print("=" * 60)
    
    # Test 1: Toplam preset sayısı
    print(f"\n[TEST 1] Toplam preset sayısı: {get_preset_count()}")
    print("  [OK]")
    
    # Test 2: Tüm preset'leri listele
    print("\n[TEST 2] Tüm preset ID'leri:")
    all_presets = list_all_presets()
    for pid in all_presets:
        preset = get_preset(pid)
        if preset:
            print(f"  - {pid}: {preset.name}")
    print("  [OK]")
    
    # Test 3: ID ile preset getir
    print("\n[TEST 3] ID ile preset getir")
    test_preset = get_preset("deep_focus")
    assert test_preset is not None
    assert test_preset.name == "Deep Focus"
    print(f"  Preset: {test_preset.name}")
    print(f"  Açıklama: {test_preset.description[:50]}...")
    print("  [OK]")
    
    # Test 4: İsim ile preset getir
    print("\n[TEST 4] İsim ile preset getir")
    named_preset = get_preset_by_name("Sleep Cocoon")
    assert named_preset is not None
    assert named_preset.duration_sec == 28800.0
    print(f"  Süre: {named_preset.duration_sec / 3600:.0f} saat")
    print("  [OK]")
    
    # Test 5: Kategoriye göre listele
    print("\n[TEST 5] Kategoriye göre listele")
    sleep_presets = list_presets_by_category(CATEGORY_SLEEP)
    print(f"  Uyku preset'leri ({len(sleep_presets)}):")
    for pid in sleep_presets:
        p = get_preset(pid)
        if p:
            print(f"    - {p.name}")
    print("  [OK]")
    
    # Test 6: Arama
    print("\n[TEST 6] Arama: 'noise'")
    noise_results = search_presets("noise")
    print(f"  Sonuç sayısı: {len(noise_results)}")
    for pid in noise_results:
        p = get_preset(pid)
        if p:
            print(f"    - {p.name}")
    print("  [OK]")
    
    # Test 7: Tüm etiketler
    print("\n[TEST 7] Tüm etiketler:")
    all_tags = get_all_tags()
    print(f"  {all_tags}")
    print("  [OK]")
    
    # Test 8: Preset bilgisi
    print("\n[TEST 8] Preset bilgisi")
    info = get_preset_info("ocean_waves")
    assert info is not None
    for key, value in info.items():
        print(f"  {key}: {value}")
    print("  [OK]")
    
    # Test 9: Doğa preset'leri detay
    print("\n[TEST 9] Doğa preset'leri detay")
    nature_presets = list_presets_by_category(CATEGORY_NATURE)
    for pid in nature_presets:
        preset = get_preset(pid)
        if preset:
            print(f"\n  {preset.name}:")
            print(f"    Katman sayısı: {len(preset.layers)}")
            for layer in preset.layers:
                print(f"      - {layer.name} ({layer.noise_type}, gain={layer.gain})")
    print("  [OK]")
    
    # Test 10: Olmayan preset
    print("\n[TEST 10] Olmayan preset kontrolü")
    missing = get_preset("nonexistent_preset")
    assert missing is None
    print("  None döndü: [OK]")
    
    print("\n" + "=" * 60)
    print("Tüm testler başarılı!")
    print("=" * 60)
    
    # Özet tablo
    print("\n" + "=" * 60)
    print("PRESET KÜTÜPHANESİ ÖZETİ")
    print("=" * 60)
    print(f"{'Kategori':<20} {'Preset Sayısı':<15}")
    print("-" * 35)
    
    categories = [
        CATEGORY_FOCUS,
        CATEGORY_SLEEP,
        CATEGORY_MEDITATION,
        CATEGORY_NATURE,
        CATEGORY_AMBIENT,
        CATEGORY_PRODUCTIVITY,
    ]
    
    for cat in categories:
        count = len(list_presets_by_category(cat))
        print(f"{cat:<20} {count:<15}")
    
    print("-" * 35)
    print(f"{'TOPLAM':<20} {get_preset_count():<15}")
    print("=" * 60)