"""
preset_system/preset_autogen.py

xxxDSP / xxxDSP Engine - Preset Varyasyon Motoru
=====================================================
Bu modül:
- Mevcut bir preset'ten kontrollü, deterministik varyasyonlar üretir
- Yapısal tutarlılığı korur
- DSP veya render işlemi yapmaz

Yalnızca PresetConfig tabanlı, statik veri çıktısı sağlar.
"""

import copy
import random
from typing import List, Optional
from preset_system.preset_schema import PresetConfig, LayerConfig


# =============================================================================
# Yardımcı Fonksiyonlar
# =============================================================================

def _jitter_value(rng: random.Random, value: float, ratio: float, min_val: float, max_val: float) -> float:
    """Değeri belirtilen oranda sapma ile rastgele değiştirir."""
    delta = value * ratio
    jittered = value + rng.uniform(-delta, delta)
    return max(min_val, min(max_val, jittered))

def _vary_layer(rng: random.Random, layer: LayerConfig, intensity: float) -> LayerConfig:
    """Tek bir katmanı varye eder."""
    varied = copy.deepcopy(layer)
    varied.gain = _jitter_value(rng, varied.gain, intensity, 0.0, 1.0)
    varied.pan = _jitter_value(rng, varied.pan, intensity, -1.0, 1.0)

    if varied.filter_config.enabled:
        varied.filter_config.cutoff_hz = _jitter_value(rng, varied.filter_config.cutoff_hz, intensity, 20.0, 20000.0)
        varied.filter_config.resonance_q = _jitter_value(rng, varied.filter_config.resonance_q, intensity, 0.1, 20.0)

    if varied.lfo_config.enabled:
        varied.lfo_config.rate_hz = _jitter_value(rng, varied.lfo_config.rate_hz, intensity, 0.001, 20.0)
        varied.lfo_config.depth = _jitter_value(rng, varied.lfo_config.depth, intensity, 0.0, 1.0)

    return varied

def _generate_variant_id(base_id: str, suffix: str) -> str:
    return f"{base_id}_{suffix}"

def _generate_variant_name(base_name: str, suffix: str) -> str:
    return f"{base_name} ({suffix})"


# =============================================================================
# Ana Varyasyon Fonksiyonları
# =============================================================================

def generate_variant(
    base_preset: PresetConfig,
    seed: Optional[int] = None,
    suffix: str = "var",
    intensity: float = 0.1,
) -> PresetConfig:
    """
    Tek bir preset'ten varyant üretir.
    - Yapı korunur (katman sayısı, noise tipleri)
    - Sayısal değerler kontrollü şekilde varye edilir
    - ID ve isim farklılaştırılır
    - Etiketlere "variant" eklenir
    """
    rng = random.Random(seed)
    variant = copy.deepcopy(base_preset)
    #variant.id = _generate_variant_id(base_preset.id, suffix)
    variant.name = _generate_variant_name(base_preset.name, suffix)
    if "variant" not in variant.tags:
        variant.tags.append("variant")

    variant.layers = [_vary_layer(rng, layer, intensity) for layer in base_preset.layers]
    variant.master_gain = _jitter_value(rng, base_preset.master_gain, intensity / 2.0, 0.0, 1.0)
    variant.seed = seed
    return variant

def generate_variants(
    base_preset: PresetConfig,
    count: int,
    seed: Optional[int] = None,
    intensity: float = 0.1,
) -> List[PresetConfig]:
    """
    Verilen preset'ten birden fazla varyasyon üretir.
    """
    variants: List[PresetConfig] = []
    for i in range(count):
        suffix = f"v{i+1}"
        variant_seed = (seed + i * 1000) if seed is not None else None
        variant = generate_variant(base_preset, seed=variant_seed, suffix=suffix, intensity=intensity)
        variants.append(variant)
    return variants


def get_variant_summary(preset: PresetConfig) -> dict:
    return {
        "id": preset.id,
        "name": preset.name,
        "tags": preset.tags,
        "layers": len(preset.layers),
        "master_gain": round(preset.master_gain, 3),
        "seed": preset.seed,
    }


# =============================================================================
# Test Bloğu
# =============================================================================

if __name__ == "__main__":
    # Test preset
    test = PresetConfig(
        id="test_base",
        name="Test Base",
        description="Varyasyon testi",
        tags=["test"],
        target_use="focus",
        duration_sec=3600.0,
        sample_rate=48000,
        layers=[
            LayerConfig(name="Layer A", noise_type="pink", gain=0.5),
            LayerConfig(name="Layer B", noise_type="brown", gain=0.3),
        ],
        master_gain=0.8,
    )

    variants = generate_variants(test, count=3, seed=123, intensity=0.2)

    for v in variants:
        print(get_variant_summary(v))
