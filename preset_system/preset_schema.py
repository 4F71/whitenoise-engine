from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

"""Preset schema definitions."""


@dataclass(slots=True)
class PresetConfig:
    """Simple container for preset metadata and its graph patch payload."""

    id: str
    name: str
    tags: list[str] = field(default_factory=list)
    target_use: str | None = None
    duration_hint: float | None = None
    graph_patch: dict[str, Any] = field(default_factory=dict)
