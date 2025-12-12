"""
preset_system/preset_schema.py

Preset veri modeli tanımları.

Bu modül:
- Preset yapılandırma şemasını tanımlar
- JSON serileştirme / deserileştirme sağlar
- Tip güvenli veri modelleri sunar

Sorumluluk sınırı:
- SADECE veri modeli
- DSP işlemi İÇERMEZ
- Ses üretimi İÇERMEZ
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from enum import Enum
import json


# =============================================================================
# ENUM TANIMLARI
# =============================================================================

class NoiseType(Enum):
    """Desteklenen gürültü tipleri."""
    WHITE = "white"
    PINK = "pink"
    BROWN = "brown"
    BLUE = "blue"
    VIOLET = "violet"


class FilterType(Enum):
    """Desteklenen filtre tipleri."""
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    NOTCH = "notch"
    LOWSHELF = "lowshelf"
    HIGHSHELF = "highshelf"


class LfoWaveform(Enum):
    """LFO dalga formları."""
    SINE = "sine"
    TRIANGLE = "triangle"
    SQUARE = "square"
    SAW = "saw"
    RANDOM = "random"


class LfoTarget(Enum):
    """LFO modülasyon hedefleri."""
    AMPLITUDE = "amplitude"
    PAN = "pan"
    FILTER_CUTOFF = "filter_cutoff"


# =============================================================================
# YAPILANDIRMA SINIFLARI
# =============================================================================

@dataclass
class FilterConfig:
    """
    Filtre yapılandırması.
    
    Attributes:
        enabled: Filtre aktif mi
        filter_type: Filtre tipi
        cutoff_hz: Kesim frekansı (Hz)
        resonance_q: Rezonans değeri (Q faktörü)
        gain_db: Shelf filtreler için kazanç (dB)
    """
    enabled: bool = False
    filter_type: str = "lowpass"
    cutoff_hz: float = 1000.0
    resonance_q: float = 0.707
    gain_db: float = 0.0
    
    def __post_init__(self) -> None:
        """Alan değerlerini doğrula ve sınırla."""
        self.cutoff_hz = max(20.0, min(20000.0, self.cutoff_hz))
        self.resonance_q = max(0.1, min(20.0, self.resonance_q))
        self.gain_db = max(-24.0, min(24.0, self.gain_db))


@dataclass
class LfoConfig:
    """
    LFO (Düşük Frekanslı Osilatör) yapılandırması.
    
    Attributes:
        enabled: LFO aktif mi
        waveform: Dalga formu
        target: Modülasyon hedefi
        rate_hz: LFO hızı (Hz)
        depth: Modülasyon derinliği (0.0 - 1.0)
        phase_offset: Başlangıç fazı (0.0 - 1.0)
    """
    enabled: bool = False
    waveform: str = "sine"
    target: str = "amplitude"
    rate_hz: float = 0.1
    depth: float = 0.0
    phase_offset: float = 0.0
    
    def __post_init__(self) -> None:
        """Alan değerlerini doğrula ve sınırla."""
        self.rate_hz = max(0.001, min(20.0, self.rate_hz))
        self.depth = max(0.0, min(1.0, self.depth))
        self.phase_offset = max(0.0, min(1.0, self.phase_offset))


@dataclass
class FxConfig:
    """
    Efekt zinciri yapılandırması.
    
    Attributes:
        saturation_amount: Saturasyon miktarı (0.0 - 1.0)
        stereo_width: Stereo genişlik (0.0 mono, 1.0 normal, 2.0 geniş)
        reverb_mix: Reverb karışım oranı (0.0 - 1.0)
        reverb_decay: Reverb decay süresi (saniye)
    """
    saturation_amount: float = 0.0
    stereo_width: float = 1.0
    reverb_mix: float = 0.0
    reverb_decay: float = 1.0
    
    def __post_init__(self) -> None:
        """Alan değerlerini doğrula ve sınırla."""
        self.saturation_amount = max(0.0, min(1.0, self.saturation_amount))
        self.stereo_width = max(0.0, min(2.0, self.stereo_width))
        self.reverb_mix = max(0.0, min(1.0, self.reverb_mix))
        self.reverb_decay = max(0.1, min(10.0, self.reverb_decay))


@dataclass
class LayerConfig:
    """
    Tek bir ses katmanı yapılandırması.
    
    Attributes:
        name: Katman adı (tanımlayıcı)
        enabled: Katman aktif mi
        noise_type: Gürültü tipi
        gain: Katman kazancı (0.0 - 1.0)
        pan: Stereo konum (-1.0 sol, 0.0 merkez, +1.0 sağ)
        filter_config: Filtre ayarları
        lfo_config: LFO ayarları
    """
    name: str = "Layer"
    enabled: bool = True
    noise_type: str = "white"
    gain: float = 1.0
    pan: float = 0.0
    filter_config: FilterConfig = field(default_factory=FilterConfig)
    lfo_config: LfoConfig = field(default_factory=LfoConfig)
    
    def __post_init__(self) -> None:
        """Alan değerlerini doğrula ve sınırla."""
        self.gain = max(0.0, min(1.0, self.gain))
        self.pan = max(-1.0, min(1.0, self.pan))
        
        # Dict olarak geldiyse dönüştür
        if isinstance(self.filter_config, dict):
            self.filter_config = FilterConfig(**self.filter_config)
        if isinstance(self.lfo_config, dict):
            self.lfo_config = LfoConfig(**self.lfo_config)


@dataclass
class PresetConfig:
    """
    Tam preset yapılandırması.
    
    Bir preset, birden fazla ses katmanı ve
    genel ayarları içerir.
    
    Attributes:
        name: Preset adı
        description: Preset açıklaması
        author: Oluşturan kişi
        version: Preset versiyonu
        tags: Kategorileme etiketleri
        layers: Ses katmanları listesi
        master_gain: Ana çıkış kazancı (0.0 - 1.0)
        fx_config: Efekt zinciri ayarları
        duration_sec: Varsayılan süre (saniye)
        sample_rate: Örnekleme hızı (Hz)
        seed: Rastgelelik için seed (None ise rastgele)
    """
    name: str = "Untitled Preset"
    description: str = ""
    author: str = ""
    version: str = "1.0"
    tags: list[str] = field(default_factory=list)
    layers: list[LayerConfig] = field(default_factory=list)
    master_gain: float = 0.8
    fx_config: FxConfig = field(default_factory=FxConfig)
    duration_sec: float = 60.0
    sample_rate: int = 48000
    seed: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Alan değerlerini doğrula ve sınırla."""
        self.master_gain = max(0.0, min(1.0, self.master_gain))
        self.duration_sec = max(1.0, min(43200.0, self.duration_sec))  # Max 12 saat
        self.sample_rate = max(22050, min(192000, self.sample_rate))
        
        # Dict olarak geldiyse dönüştür
        if isinstance(self.fx_config, dict):
            self.fx_config = FxConfig(**self.fx_config)
        
        # Katmanları dönüştür
        converted_layers: list[LayerConfig] = []
        for layer in self.layers:
            if isinstance(layer, dict):
                converted_layers.append(LayerConfig(**layer))
            else:
                converted_layers.append(layer)
        self.layers = converted_layers


# =============================================================================
# SERİLEŞTİRME FONKSİYONLARI
# =============================================================================

def preset_to_dict(preset: PresetConfig) -> dict[str, Any]:
    """
    PresetConfig nesnesini dictionary'e dönüştürür.
    
    Args:
        preset: Dönüştürülecek preset
        
    Returns:
        JSON serileştirmeye uygun dictionary
    """
    return asdict(preset)


def dict_to_preset(data: dict[str, Any]) -> PresetConfig:
    """
    Dictionary'den PresetConfig nesnesi oluşturur.
    
    Args:
        data: Preset verisi içeren dictionary
        
    Returns:
        PresetConfig nesnesi
        
    Raises:
        TypeError: Geçersiz veri yapısı
    """
    return PresetConfig(**data)


def preset_to_json(preset: PresetConfig, indent: int = 2) -> str:
    """
    PresetConfig nesnesini JSON string'e dönüştürür.
    
    Args:
        preset: Dönüştürülecek preset
        indent: JSON girinti miktarı
        
    Returns:
        JSON formatında string
    """
    data = preset_to_dict(preset)
    return json.dumps(data, indent=indent, ensure_ascii=False)


def json_to_preset(json_str: str) -> PresetConfig:
    """
    JSON string'den PresetConfig nesnesi oluşturur.
    
    Args:
        json_str: JSON formatında preset verisi
        
    Returns:
        PresetConfig nesnesi
        
    Raises:
        json.JSONDecodeError: Geçersiz JSON
        TypeError: Geçersiz veri yapısı
    """
    data = json.loads(json_str)
    return dict_to_preset(data)


def save_preset_to_file(preset: PresetConfig, filepath: str) -> None:
    """
    Preset'i JSON dosyasına kaydeder.
    
    Args:
        preset: Kaydedilecek preset
        filepath: Dosya yolu
    """
    json_str = preset_to_json(preset)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(json_str)


def load_preset_from_file(filepath: str) -> PresetConfig:
    """
    JSON dosyasından preset yükler.
    
    Args:
        filepath: Dosya yolu
        
    Returns:
        PresetConfig nesnesi
        
    Raises:
        FileNotFoundError: Dosya bulunamazsa
        json.JSONDecodeError: Geçersiz JSON
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        json_str = f.read()
    return json_to_preset(json_str)


# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def create_default_layer(
    name: str = "Layer",
    noise_type: str = "white",
    gain: float = 1.0
) -> LayerConfig:
    """
    Varsayılan ayarlarla yeni katman oluşturur.
    
    Args:
        name: Katman adı
        noise_type: Gürültü tipi
        gain: Kazanç değeri
        
    Returns:
        LayerConfig nesnesi
    """
    return LayerConfig(
        name=name,
        enabled=True,
        noise_type=noise_type,
        gain=gain,
        pan=0.0,
        filter_config=FilterConfig(),
        lfo_config=LfoConfig()
    )


def create_empty_preset(name: str = "Untitled Preset") -> PresetConfig:
    """
    Boş preset oluşturur.
    
    Args:
        name: Preset adı
        
    Returns:
        Katmansız PresetConfig nesnesi
    """
    return PresetConfig(name=name, layers=[])


def validate_preset(preset: PresetConfig) -> list[str]:
    """
    Preset'i doğrular ve hataları listeler.
    
    Args:
        preset: Doğrulanacak preset
        
    Returns:
        Hata mesajları listesi (boşsa geçerli)
    """
    errors: list[str] = []
    
    # İsim kontrolü
    if not preset.name or not preset.name.strip():
        errors.append("Preset adı boş olamaz")
    
    # Katman kontrolü
    if not preset.layers:
        errors.append("En az bir katman gerekli")
    
    # Katman isimleri tekrar kontrolü
    layer_names = [layer.name for layer in preset.layers]
    if len(layer_names) != len(set(layer_names)):
        errors.append("Katman isimleri benzersiz olmalı")
    
    # Noise type kontrolü
    valid_noise_types = {nt.value for nt in NoiseType}
    for i, layer in enumerate(preset.layers):
        if layer.noise_type not in valid_noise_types:
            errors.append(f"Katman {i}: Geçersiz noise_type '{layer.noise_type}'")
    
    # Filter type kontrolü
    valid_filter_types = {ft.value for ft in FilterType}
    for i, layer in enumerate(preset.layers):
        if layer.filter_config.enabled:
            if layer.filter_config.filter_type not in valid_filter_types:
                errors.append(
                    f"Katman {i}: Geçersiz filter_type "
                    f"'{layer.filter_config.filter_type}'"
                )
    
    # LFO kontrolü
    valid_waveforms = {wf.value for wf in LfoWaveform}
    valid_targets = {tg.value for tg in LfoTarget}
    for i, layer in enumerate(preset.layers):
        if layer.lfo_config.enabled:
            if layer.lfo_config.waveform not in valid_waveforms:
                errors.append(
                    f"Katman {i}: Geçersiz LFO waveform "
                    f"'{layer.lfo_config.waveform}'"
                )
            if layer.lfo_config.target not in valid_targets:
                errors.append(
                    f"Katman {i}: Geçersiz LFO target "
                    f"'{layer.lfo_config.target}'"
                )
    
    return errors


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("preset_schema.py test başlıyor...")
    print("=" * 50)
    
    # Test 1: Basit preset oluşturma
    print("\n[TEST 1] Basit preset oluşturma")
    
    simple_preset = PresetConfig(
        name="Test Preset",
        description="Test amaçlı preset",
        author="Developer",
        layers=[
            LayerConfig(name="White Base", noise_type="white", gain=0.5),
            LayerConfig(name="Pink Layer", noise_type="pink", gain=0.3),
        ]
    )
    
    print(f"  Preset adı: {simple_preset.name}")
    print(f"  Katman sayısı: {len(simple_preset.layers)}")
    print("  [OK]")
    
    # Test 2: JSON serileştirme
    print("\n[TEST 2] JSON serileştirme")
    
    json_str = preset_to_json(simple_preset)
    print(f"  JSON uzunluğu: {len(json_str)} karakter")
    assert '"name": "Test Preset"' in json_str
    print("  [OK]")
    
    # Test 3: JSON deserileştirme
    print("\n[TEST 3] JSON deserileştirme")
    
    restored_preset = json_to_preset(json_str)
    assert restored_preset.name == simple_preset.name
    assert len(restored_preset.layers) == len(simple_preset.layers)
    assert restored_preset.layers[0].noise_type == "white"
    print(f"  Geri yüklenen preset: {restored_preset.name}")
    print("  [OK]")
    
    # Test 4: Karmaşık preset
    print("\n[TEST 4] Karmaşık preset")
    
    complex_preset = PresetConfig(
        name="Deep Focus",
        description="Derin konsantrasyon için ambient gürültü",
        author="UltraGen",
        version="1.0",
        tags=["focus", "ambient", "work"],
        layers=[
            LayerConfig(
                name="Brown Foundation",
                noise_type="brown",
                gain=0.6,
                pan=-0.2,
                filter_config=FilterConfig(
                    enabled=True,
                    filter_type="lowpass",
                    cutoff_hz=500.0,
                    resonance_q=0.5
                ),
                lfo_config=LfoConfig(
                    enabled=True,
                    waveform="sine",
                    target="amplitude",
                    rate_hz=0.05,
                    depth=0.1
                )
            ),
            LayerConfig(
                name="Pink Texture",
                noise_type="pink",
                gain=0.3,
                pan=0.3,
                filter_config=FilterConfig(
                    enabled=True,
                    filter_type="bandpass",
                    cutoff_hz=1200.0,
                    resonance_q=1.0
                )
            ),
        ],
        master_gain=0.75,
        fx_config=FxConfig(
            stereo_width=1.2,
            saturation_amount=0.05
        ),
        duration_sec=3600.0,
        seed=42
    )
    
    print(f"  Preset: {complex_preset.name}")
    print(f"  Etiketler: {complex_preset.tags}")
    print(f"  Süre: {complex_preset.duration_sec / 60:.0f} dakika")
    
    json_complex = preset_to_json(complex_preset)
    restored_complex = json_to_preset(json_complex)
    
    assert restored_complex.layers[0].filter_config.cutoff_hz == 500.0
    assert restored_complex.layers[0].lfo_config.rate_hz == 0.05
    assert restored_complex.fx_config.stereo_width == 1.2
    print("  [OK]")
    
    # Test 5: Doğrulama
    print("\n[TEST 5] Doğrulama")
    
    valid_errors = validate_preset(complex_preset)
    print(f"  Geçerli preset hataları: {len(valid_errors)}")
    assert len(valid_errors) == 0
    
    invalid_preset = PresetConfig(
        name="",
        layers=[]
    )
    invalid_errors = validate_preset(invalid_preset)
    print(f"  Geçersiz preset hataları: {len(invalid_errors)}")
    assert len(invalid_errors) >= 2
    for err in invalid_errors:
        print(f"    - {err}")
    print("  [OK]")
    
    # Test 6: Yardımcı fonksiyonlar
    print("\n[TEST 6] Yardımcı fonksiyonlar")
    
    new_layer = create_default_layer("Test Layer", "violet", 0.7)
    assert new_layer.name == "Test Layer"
    assert new_layer.noise_type == "violet"
    
    empty_preset = create_empty_preset("Empty")
    assert len(empty_preset.layers) == 0
    
    print("  create_default_layer: [OK]")
    print("  create_empty_preset: [OK]")
    
    # Test 7: Değer sınırlamaları
    print("\n[TEST 7] Değer sınırlamaları")
    
    clamped_layer = LayerConfig(
        gain=5.0,  # 1.0'a düşmeli
        pan=-10.0  # -1.0'a düşmeli
    )
    assert clamped_layer.gain == 1.0
    assert clamped_layer.pan == -1.0
    
    clamped_filter = FilterConfig(
        cutoff_hz=50000.0,  # 20000'e düşmeli
        resonance_q=100.0   # 20.0'ye düşmeli
    )
    assert clamped_filter.cutoff_hz == 20000.0
    assert clamped_filter.resonance_q == 20.0
    
    print("  Gain sınırlaması: [OK]")
    print("  Pan sınırlaması: [OK]")
    print("  Filter sınırlamaları: [OK]")
    
    # Test 8: Enum değerleri
    print("\n[TEST 8] Enum değerleri")
    
    print(f"  NoiseType: {[nt.value for nt in NoiseType]}")
    print(f"  FilterType: {[ft.value for ft in FilterType]}")
    print(f"  LfoWaveform: {[wf.value for wf in LfoWaveform]}")
    print(f"  LfoTarget: {[tg.value for tg in LfoTarget]}")
    print("  [OK]")
    
    print("\n" + "=" * 50)
    print("Tüm testler başarılı!")
    print("=" * 50)
    
    # Örnek JSON çıktısı
    print("\n" + "=" * 50)
    print("Örnek preset JSON:")
    print("=" * 50)
    print(preset_to_json(complex_preset))