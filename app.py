from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

from preset_system.preset_library import PRESETS, PRESET_MAP


# ----------------------------------------------------------------------------
# Capability checks
# ----------------------------------------------------------------------------


def _has_streamlit() -> bool:
    """Return True if Streamlit is importable."""

    try:
        import importlib.util

        return importlib.util.find_spec("streamlit") is not None
    except Exception:
        return False


def _is_streamlit_runtime() -> bool:
    """Detect if the script is already running under Streamlit."""

    env_flags = ("STREAMLIT_SERVER_RUNNING", "STREAMLIT_RUNTIME")
    return any(os.environ.get(flag) for flag in env_flags)


# ----------------------------------------------------------------------------
# CLI helpers
# ----------------------------------------------------------------------------


def _list_presets(output: Iterable[str] | None = None) -> None:
    lines = []
    for preset in PRESETS:
        tags = ", ".join(preset.tags) if preset.tags else "-"
        lines.append(f"{preset.id}: {preset.name} [use={preset.target_use or '-'}] tags={tags}")

    target = sys.stdout if output is None else output
    for line in lines:
        print(line, file=target)


def _show_preset(preset_id: str) -> int:
    preset = PRESET_MAP.get(preset_id)
    if preset is None:
        print(f"Preset bulunamadi: {preset_id}", file=sys.stderr)
        return 1

    print(f"ID: {preset.id}")
    print(f"Ad: {preset.name}")
    print(f"Hedef kullanim: {preset.target_use or '-'}")
    print(f"Etiketler: {', '.join(preset.tags) if preset.tags else '-'}")
    print(f"Sure ipucu: {preset.duration_hint or '-'}")
    print("Graph patch:")
    print(preset.graph_patch)
    return 0


def _resolve_duration(arg_duration: float | None, preset) -> float:
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


def _render_from_cli(args: argparse.Namespace) -> int:
    from ui_app import ui_cli

    if not args.preset:
        print("--preset parametresi gerekli.", file=sys.stderr)
        return 2

    preset = PRESET_MAP.get(args.preset)
    if preset is None:
        available = ", ".join(sorted(PRESET_MAP))
        print(f"Bilinmeyen preset '{args.preset}'. Mevcut: {available}", file=sys.stderr)
        return 1

    duration = _resolve_duration(args.duration, preset)
    output_path = Path(args.output) if args.output else _default_output_path(preset.id, duration, args.variant)

    try:
        out_path, sample_rate = ui_cli.render_preset_to_wav(preset, duration, args.variant, output_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Render hatasi: {exc}", file=sys.stderr)
        return 1

    print(f"Yazildi: {out_path} (preset={preset.id}, duration={duration:.2f}s, variant={args.variant}, sr={sample_rate} Hz)")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UltraGen ana uygulamasi")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--ui", action="store_true", help="Streamlit arayuzunu baslat")
    mode.add_argument("--cli", action="store_true", help="CLI modunu zorla")

    parser.add_argument("--preset", help="Render edilecek preset ID'si")
    parser.add_argument("--duration", type=float, help="Saniye cinsinden sure (verilmezse preset ipucu)")
    parser.add_argument("--variant", type=int, default=0, help="Rastgelelik tohumu / varyant numarasi")
    parser.add_argument("--output", help="Cikti WAV yolu")
    parser.add_argument("--list-presets", action="store_true", help="Preset listesini yaz")
    parser.add_argument("--show-preset", help="Belirli bir presetin detaylarini goster")
    return parser


def _decide_mode(args: argparse.Namespace, streamlit_available: bool) -> str:
    if _is_streamlit_runtime():
        return "ui"
    if args.cli:
        return "cli"
    if args.ui:
        return "ui" if streamlit_available else "cli"
    if args.preset or args.list_presets or args.show_preset:
        return "cli"
    return "ui" if streamlit_available else "cli"


def _run_streamlit_ui() -> int:
    try:
        from ui_app import ui_streamlit

        ui_streamlit.main()
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Streamlit baslatilamadi: {exc}", file=sys.stderr)
        return 1


def _run_cli(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.list_presets:
        _list_presets()
        return 0
    if args.show_preset:
        return _show_preset(args.show_preset)
    if not args.preset:
        parser.print_help()
        return 1
    return _render_from_cli(args)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    streamlit_available = _has_streamlit()

    mode = _decide_mode(args, streamlit_available)
    if mode == "ui":
        if not streamlit_available and not _is_streamlit_runtime():
            print("Streamlit yuklu degil, CLI moduna donuluyor.")
            sys.exit(_run_cli(args, parser))
        sys.exit(_run_streamlit_ui())

    sys.exit(_run_cli(args, parser))


if __name__ == "__main__":
    main()
