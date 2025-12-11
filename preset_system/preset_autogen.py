from __future__ import annotations

"""
Random/varyasyon tabanli otomatik preset ureticisi.

Amac:
- Graph patch semasini kullanarak yeni PresetConfig objeleri uretmek
- Mevcut presetlerden kucuk oynamalarla varyantlar cikarmak

Tum cikti `preset_schema.PresetConfig` dataclass'ini kullanir; graph_patch yapisi
`graph_loader` ile uyumludur.
"""

import copy
import random
import string
from typing import Any, Callable, Dict, List, Mapping, Sequence

from preset_system.preset_library import PRESETS
from preset_system.preset_schema import PresetConfig

GraphPatch = Dict[str, Any]

# Temel havuzlar
TARGET_USE_CHOICES = ["sleep", "focus", "meditation", "asmr", "ambient"]
NOISE_TYPES = ["white", "pink", "brown"]
WAVEFORMS = ["sine", "triangle", "saw"]
BASE_TAGS = {
    "sleep": ["sleep", "calm", "soft"],
    "focus": ["focus", "clarity", "steady"],
    "meditation": ["meditation", "breath", "drone"],
    "asmr": ["asmr", "soft", "sparkle"],
    "ambient": ["ambient", "space", "wide"],
}


# ---------------------------------------------------------------------------
# Yardimcilar
# ---------------------------------------------------------------------------

def _make_rng(seed: int | None) -> random.Random:
    return random.Random(seed)


def _random_id(prefix: str, rng: random.Random) -> str:
    suffix = "".join(rng.choices(string.ascii_lowercase + string.digits, k=5))
    return f"{prefix}_{suffix}"


def _pick(seq: Sequence[Any], rng: random.Random) -> Any:
    if not seq:
        raise ValueError("Secilecek oge yok.")
    return seq[rng.randrange(len(seq))]


def _jitter_number(value: int | float, amount: float, rng: random.Random) -> int | float:
    """Sayisal degere +/-amount oraninda jitter ekler."""
    if amount <= 0:
        return value
    scale = abs(float(value)) if value != 0 else 1.0
    delta = scale * amount * rng.uniform(-1.0, 1.0)
    updated = float(value) + delta
    if isinstance(value, int):
        return int(max(1, round(updated)))
    return updated


def _jitter_patch(patch: Mapping[str, Any], amount: float, rng: random.Random) -> GraphPatch:
    """Graph patch icindeki sayisal degerleri hafifce oynatir."""
    skip_numeric_keys = {"sample_rate"}

    def _walk(obj: Any, key_hint: str | None = None) -> Any:
        if isinstance(obj, Mapping):
            return {k: _walk(v, k) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(v, key_hint) for v in obj]
        if isinstance(obj, (int, float)):
            if key_hint in skip_numeric_keys:
                return obj
            if key_hint == "duration":
                return max(1.0, _jitter_number(float(obj), amount * 0.5, rng))
            return _jitter_number(obj, amount, rng)
        return obj

    return _walk(copy.deepcopy(patch))


def _duration_hint_for_use(target_use: str, rng: random.Random) -> float:
    base = {
        "sleep": rng.uniform(8.0, 12.0),
        "meditation": rng.uniform(6.0, 10.0),
        "focus": rng.uniform(5.0, 8.0),
        "asmr": rng.uniform(4.5, 7.0),
        "ambient": rng.uniform(6.0, 9.0),
    }
    return base.get(target_use, rng.uniform(5.0, 9.0))


def _base_tags(target_use: str) -> List[str]:
    return list(BASE_TAGS.get(target_use, ["ambient"]))


# ---------------------------------------------------------------------------
# Patch sablonlari
# ---------------------------------------------------------------------------

def _patch_noise_bed(rng: random.Random, duration: float) -> GraphPatch:
    noise_type = _pick(NOISE_TYPES, rng)
    low = rng.uniform(60.0, 250.0)
    high = rng.uniform(2_500.0, 9_000.0)
    return {
        "sample_rate": 44_100,
        "global_params": {"duration": duration},
        "nodes": [
            {"name": "bed", "type": "noise", "params": {"type": noise_type, "amplitude": rng.uniform(0.28, 0.55)}},
            {"name": "filter", "type": "filter", "params": {"type": "bandpass", "low_cutoff": low, "high_cutoff": high}},
            {
                "name": "env",
                "type": "envelope",
                "params": {
                    "attack": rng.uniform(0.1, 0.4),
                    "decay": rng.uniform(0.3, 0.8),
                    "sustain": rng.uniform(0.7, 0.95),
                    "release": rng.uniform(0.8, 1.5),
                },
            },
            {"name": "pan", "type": "pan", "params": {"pan": rng.uniform(-0.25, 0.25)}},
            {"name": "out", "type": "output"},
        ],
        "edges": [
            {"source": "bed", "target": "filter"},
            {"source": "filter", "target": "env"},
            {"source": "env", "target": "pan"},
            {"source": "pan", "target": "out"},
        ],
    }


def _patch_pad_with_noise(rng: random.Random, duration: float) -> GraphPatch:
    base_freq = rng.uniform(110.0, 320.0)
    pad_wave = _pick(WAVEFORMS, rng)
    noise_type = _pick(NOISE_TYPES, rng)
    return {
        "sample_rate": 44_100,
        "global_params": {"duration": duration},
        "nodes": [
            {"name": "osc_main", "type": "oscillator", "params": {"waveform": pad_wave, "frequency": base_freq, "amplitude": rng.uniform(0.22, 0.38)}},
            {"name": "osc_air", "type": "oscillator", "params": {"waveform": "sine", "frequency": base_freq * rng.uniform(2.1, 2.6), "amplitude": rng.uniform(0.08, 0.16)}},
            {"name": "bed", "type": "noise", "params": {"type": noise_type, "amplitude": rng.uniform(0.14, 0.24)}},
            {"name": "mix", "type": "mix"},
            {"name": "filter", "type": "filter", "params": {"type": "lowpass", "cutoff": rng.uniform(1_200.0, 3_600.0)}},
            {
                "name": "env",
                "type": "envelope",
                "params": {
                    "attack": rng.uniform(0.4, 0.9),
                    "decay": rng.uniform(0.6, 1.2),
                    "sustain": rng.uniform(0.75, 0.92),
                    "release": rng.uniform(1.2, 2.2),
                },
            },
            {"name": "pan", "type": "pan", "params": {"pan": rng.uniform(-0.15, 0.15)}},
            {"name": "out", "type": "output"},
        ],
        "edges": [
            {"source": "osc_main", "target": "mix"},
            {"source": "osc_air", "target": "mix"},
            {"source": "bed", "target": "mix"},
            {"source": "mix", "target": "env"},
            {"source": "env", "target": "filter"},
            {"source": "filter", "target": "pan"},
            {"source": "pan", "target": "out"},
        ],
    }


def _patch_binaural(rng: random.Random, duration: float) -> GraphPatch:
    base = rng.uniform(120.0, 180.0)
    beat = rng.uniform(4.0, 9.0)
    return {
        "sample_rate": 44_100,
        "global_params": {"duration": duration},
        "nodes": [
            {"name": "osc_l", "type": "oscillator", "params": {"waveform": "sine", "frequency": base, "amplitude": rng.uniform(0.16, 0.24)}},
            {"name": "osc_r", "type": "oscillator", "params": {"waveform": "sine", "frequency": base + beat, "amplitude": rng.uniform(0.16, 0.24)}},
            {"name": "env_l", "type": "envelope", "params": {"attack": 0.2, "decay": 0.5, "sustain": 0.8, "release": 0.8}},
            {"name": "env_r", "type": "envelope", "params": {"attack": 0.2, "decay": 0.5, "sustain": 0.8, "release": 0.8}},
            {"name": "pan_l", "type": "pan", "params": {"pan": -0.5}},
            {"name": "pan_r", "type": "pan", "params": {"pan": 0.5}},
            {"name": "mix", "type": "mix"},
            {"name": "out", "type": "output"},
        ],
        "edges": [
            {"source": "osc_l", "target": "env_l"},
            {"source": "env_l", "target": "pan_l"},
            {"source": "osc_r", "target": "env_r"},
            {"source": "env_r", "target": "pan_r"},
            {"source": "pan_l", "target": "mix"},
            {"source": "pan_r", "target": "mix"},
            {"source": "mix", "target": "out"},
        ],
    }


PATCH_BUILDERS: Sequence[Callable[[random.Random, float], GraphPatch]] = [
    _patch_noise_bed,
    _patch_pad_with_noise,
    _patch_binaural,
]


# ---------------------------------------------------------------------------
# Ana API
# ---------------------------------------------------------------------------

def generate_random_preset(target_use: str | None = None, seed: int | None = None) -> PresetConfig:
    """
    Tamamen rastgele parametrelerle yeni bir preset uretir.

    Args:
        target_use: Hedef kategori; None ise rastgele secilir.
        seed: Deterministik sonuc icin rastgelelik tohumu.
    """
    rng = _make_rng(seed)
    chosen_use = target_use or _pick(TARGET_USE_CHOICES, rng)
    duration = _duration_hint_for_use(chosen_use, rng)
    builder = _pick(PATCH_BUILDERS, rng)
    patch = builder(rng, duration)

    name_token = _pick(["Drift", "Calm", "Flow", "Glimmer", "Pulse", "Mist"], rng)
    preset_id = _random_id(chosen_use, rng)
    tags = sorted(set(_base_tags(chosen_use) + [patch["nodes"][0]["type"], name_token.lower(), "auto"]))

    return PresetConfig(
        id=preset_id,
        name=f"{name_token} ({chosen_use})",
        tags=tags,
        target_use=chosen_use,
        duration_hint=duration,
        graph_patch=patch,
    )


def mutate_preset(
    base: PresetConfig,
    variant_index: int = 1,
    jitter: float = 0.1,
    seed: int | None = None,
) -> PresetConfig:
    """
    Mevcut bir preset'i kucuk jitter ile varyanta donusturur.

    Args:
        base: Kaynak preset.
        variant_index: Isme/ID'ye eklenecek varyant numarasi.
        jitter: Sayisal parametrelerdeki degisim orani (0-1 arasi onerilir).
        seed: Deterministik varyant icin tohum.
    """
    rng = _make_rng(seed if seed is not None else variant_index)
    mutated_patch = _jitter_patch(base.graph_patch, jitter, rng)
    new_tags = sorted(set(base.tags + ["variant", "auto"]))
    new_id = f"{base.id}_v{variant_index}"
    new_name = f"{base.name} - Var {variant_index}"
    duration_hint = mutated_patch.get("global_params", {}).get("duration", base.duration_hint)

    return PresetConfig(
        id=new_id,
        name=new_name,
        tags=new_tags,
        target_use=base.target_use,
        duration_hint=duration_hint,
        graph_patch=mutated_patch,
    )


def generate_variants(
    base: PresetConfig,
    count: int = 3,
    jitter: float = 0.08,
    seed: int | None = None,
) -> List[PresetConfig]:
    """Ayni preset'ten art arda varyant listesi uretir."""
    rng = _make_rng(seed)
    variants: List[PresetConfig] = []
    for idx in range(1, count + 1):
        variants.append(mutate_preset(base, idx, jitter=jitter, seed=rng.randrange(10_000_000)))
    return variants


def generate_from_library(
    count: int = 5,
    jitter: float = 0.06,
    seed: int | None = None,
) -> List[PresetConfig]:
    """
    preset_library icindeki mevcut tanimlardan rastgele secim yapar,
    her biri icin varyant uretir.
    """
    rng = _make_rng(seed)
    if not PRESETS:
        raise ValueError("Preset kutuphanesi bos.")

    results: List[PresetConfig] = []
    for _ in range(count):
        base = _pick(PRESETS, rng)
        variant = mutate_preset(
            base,
            variant_index=rng.randint(1, 9_999),
            jitter=jitter,
            seed=rng.randrange(10_000_000),
        )
        results.append(variant)
    return results


__all__ = [
    "generate_random_preset",
    "mutate_preset",
    "generate_variants",
    "generate_from_library",
    "GraphPatch",
]
