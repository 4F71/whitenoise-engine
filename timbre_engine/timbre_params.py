"""Timbre parametrelerini ve yardımcı fonksiyonları barındıran modül.

Bu modül, tını sentezinde kullanılacak parametrelerin aralıklarını,
rastgele üretimini, sınırlandırılmasını ve JSON serileştirme işlemlerini içerir.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, replace
from typing import Dict, Iterable, Mapping, Tuple

NOISE_COLORS: Tuple[str, ...] = ("white", "pink", "brown", "blue", "violet")
DEFAULT_NOISE_COLOR = "white"


class RangeMap:
    """Parametre adlarını min-maks aralıklarıyla eşleyen yardımcı sınıf."""

    def __init__(self, ranges: Mapping[str, Tuple[float, float]]) -> None:
        self._ranges: Dict[str, Tuple[float, float]] = dict(ranges)

    def keys(self) -> Iterable[str]:
        """Tanımlı parametre anahtarlarını döndürür."""
        return self._ranges.keys()

    def clamp(self, key: str, value: float) -> float:
        """Verilen değeri ilgili parametre aralığına kırpar."""
        if key not in self._ranges:
            raise KeyError(f"Parametre aralığı bulunamadı: {key}")
        low, high = self._ranges[key]
        return min(max(value, low), high)

    def random(self, key: str) -> float:
        """İlgili parametre için aralıkta rastgele bir değer üretir."""
        if key not in self._ranges:
            raise KeyError(f"Parametre aralığı bulunamadı: {key}")
        low, high = self._ranges[key]
        return random.uniform(low, high)


@dataclass
class TimbreParams:
    """Tını sentezinde kullanılan temel parametre seti."""

    noise_color: str = DEFAULT_NOISE_COLOR
    noise_level: float = 0.5
    filter_cutoff: float = 8000.0
    filter_resonance: float = 0.2
    brightness: float = 0.5
    warmth: float = 0.3
    band_emphasis: float = 0.4
    lfo_rate: float = 1.5
    lfo_depth: float = 0.2
    stereo_width: float = 0.5
    saturation: float = 0.1
    reverb_mix: float = 0.25
    reverb_time: float = 1.2

    def to_dict(self) -> Dict[str, float | str]:
        """Parametreleri sözlük olarak döndürür."""
        return asdict(self)

    def to_json(self, indent: int | None = None) -> str:
        """Parametreleri JSON string olarak döndürür."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray | Mapping[str, object]) -> "TimbreParams":
        """JSON veya sözlükten TimbreParams örneği oluşturur."""
        data = payload if isinstance(payload, Mapping) else json.loads(payload)
        return cls(**data)  # type: ignore[arg-type]


DEFAULT_RANGE_MAP = RangeMap(
    {
        "noise_level": (0.0, 1.0),
        "filter_cutoff": (120.0, 18000.0),
        "filter_resonance": (0.0, 1.0),
        "brightness": (0.0, 1.0),
        "warmth": (0.0, 1.0),
        "band_emphasis": (0.0, 1.5),
        "lfo_rate": (0.1, 8.0),
        "lfo_depth": (0.0, 1.0),
        "stereo_width": (0.0, 1.0),
        "saturation": (0.0, 1.5),
        "reverb_mix": (0.0, 1.0),
        "reverb_time": (0.1, 4.0),
    }
)


def _normalize_noise_color(value: str | None) -> str:
    """Geçersiz veya boş noise_color değerlerini güvenli bir seçeneğe çevirir."""
    if not value:
        return DEFAULT_NOISE_COLOR
    lowered = value.lower()
    return lowered if lowered in NOISE_COLORS else DEFAULT_NOISE_COLOR


def random_params(range_map: RangeMap = DEFAULT_RANGE_MAP, seed: int | None = None) -> TimbreParams:
    """Aralık haritasına göre rastgele parametre seti üretir."""
    if seed is not None:
        random.seed(seed)

    numeric_values = {key: range_map.random(key) for key in range_map.keys()}
    return TimbreParams(
        noise_color=random.choice(NOISE_COLORS),
        **numeric_values,
    )


def clamp_params(params: TimbreParams, range_map: RangeMap = DEFAULT_RANGE_MAP) -> TimbreParams:
    """Parametreleri tanımlı aralıklara kırpar ve geçersiz noise_color'u düzeltir."""
    current = asdict(params)
    current["noise_color"] = _normalize_noise_color(current.get("noise_color"))

    for key, value in current.items():
        if isinstance(value, (int, float)) and key in range_map.keys():
            current[key] = range_map.clamp(key, float(value))

    return replace(params, **current)
