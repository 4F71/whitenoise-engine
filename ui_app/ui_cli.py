from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import soundfile as sf

from graph_engine.graph_core import Graph
from graph_engine.graph_loader import NODE_CLASS_MAP
from graph_engine.graph_nodes import BaseNode
from preset_system.preset_library import PRESET_MAP
from preset_system.preset_schema import PresetConfig


def _build_graph_from_patch(
    patch: Mapping[str, Any],
    node_classes: Mapping[str, type[BaseNode]] | None = None,
) -> Graph:
    """Patch sozlugunden Graph nesnesi kurar."""
    node_classes = node_classes or NODE_CLASS_MAP
    sample_rate = int(patch.get("sample_rate", 44_100))
    graph = Graph(sample_rate=sample_rate)

    global_params = patch.get("global_params") or {}
    if global_params:
        graph.set_global_params(global_params)

    nodes_cfg = patch.get("nodes") or []
    if not nodes_cfg:
        raise ValueError("Patch icinde hic node yok.")

    for cfg in nodes_cfg:
        name = cfg.get("name")
        type_name = cfg.get("type")
        if not name or not type_name:
            raise ValueError(f"Node tanimi eksik: {cfg}")
        cls = node_classes.get(str(type_name).lower())
        if cls is None:
            raise ValueError(f"Bilinmeyen node tipi: {type_name}")
        params = dict(cfg.get("params") or {})
        graph.add_node(name, cls(name=name, sample_rate=sample_rate), params=params)

    for edge in patch.get("edges") or []:
        src = edge.get("source")
        tgt = edge.get("target")
        if not src or not tgt:
            raise ValueError(f"Gecersiz edge: {edge}")
        graph.add_edge(src, tgt, target_input=edge.get("target_input", "input"))

    return graph


def _resolve_duration(arg_duration: float | None, preset: PresetConfig) -> float:
    """CLI'dan verilen veya preset icinden gelen sureyi sec."""
    if arg_duration is not None:
        return max(0.05, float(arg_duration))
    if preset.duration_hint:
        return float(preset.duration_hint)
    default = (preset.graph_patch.get("global_params") or {}).get("duration")
    return max(0.05, float(default or 8.0))


def _default_output_path(preset_id: str, duration: float, variant: int) -> Path:
    dur_tag = f"{duration:.2f}".rstrip("0").rstrip(".")
    fname = f"{preset_id}_{dur_tag}s_v{variant}.wav"
    return Path("renders") / fname


def render_preset_to_wav(preset: PresetConfig, duration: float, variant: int, output_path: Path) -> tuple[Path, int]:
    """Preset'i calistirip stereo WAV olarak yazar."""
    graph = _build_graph_from_patch(preset.graph_patch)
    duration = float(duration)
    sample_rate = graph.sample_rate
    num_frames = max(1, int(duration * sample_rate))

    # Rastgelelik ve global paramlari yay
    graph.set_global_params({"duration": duration, "seed": int(variant)})
    np.random.seed(int(variant))

    audio = graph.run(
        {
            "duration": duration,
            "num_frames": num_frames,
            "sample_rate": sample_rate,
        }
    )
    stereo = audio.T  # soundfile (num_frames, channels) bekler

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, stereo, sample_rate)
    return output_path, sample_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UltraGen patch tabanli CLI renderer")
    parser.add_argument("--preset", required=True, help="Preset ID (preset_system/preset_library icinden)")
    parser.add_argument("--duration", type=float, help="Saniye cinsinden sure; verilmezse preset sure ipucu kullanilir")
    parser.add_argument("--variant", type=int, default=0, help="Rastgelelik tohumu / varyant numarasi")
    parser.add_argument("--output", type=str, help="Cikti WAV yolu; verilmezse renders/<id>_<sure>_v<variant>.wav")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = PRESET_MAP.get(args.preset)
    if preset is None:
        available = ", ".join(sorted(PRESET_MAP))
        sys.exit(f"Bilinmeyen preset '{args.preset}'. Mevcut ID'ler: {available}")

    duration = _resolve_duration(args.duration, preset)
    output_path = Path(args.output) if args.output else _default_output_path(preset.id, duration, args.variant)

    try:
        out_path, sr = render_preset_to_wav(preset, duration, args.variant, output_path)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Render hatasi: {exc}")

    print(f"Yazildi: {out_path} (preset={preset.id}, duration={duration:.2f}s, variant={args.variant}, sr={sr} Hz)")


if __name__ == "__main__":
    main()
