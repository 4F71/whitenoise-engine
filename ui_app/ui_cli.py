import argparse
import sys
import time
import wave
from typing import Optional

import numpy as np

from preset_system.preset_library import get_preset, list_all_presets
from preset_system.preset_autogen import generate_variant
from core_dsp.dsp_render import render_sound


_DEFAULT_SAMPLE_RATE = 44100
_DEFAULT_DURATION_SEC = 60.0
_DEFAULT_OUTPUT_FILE = "output.wav"


def _write_wav(path: str, signal: np.ndarray, sample_rate: int) -> None:
    """
    Float32 mono sinyali 16-bit PCM WAV olarak yazar.
    """
    signal = np.clip(signal, -1.0, 1.0)
    pcm16 = (signal * 32767.0).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def _cmd_list_presets() -> None:
    """
    Mevcut preset listesini yazdırır.
    """
    presets = list_presets()
    if not presets:
        print("Preset bulunamadı.")
        return

    print("\n--- Mevcut Presetler ---")
    for preset in presets:
        print(f"{preset.id} - {preset.name}")


def _cmd_render(
    preset_id: str,
    output: str,
    duration: float,
    sample_rate: int,
    use_variant: bool = False,
    seed: Optional[int] = None,
) -> None:
    """
    Seçilen preset’i render edip WAV dosyası üretir.
    """
    preset = get_preset(preset_id)
    if preset is None:
        raise ValueError(f"Preset bulunamadı: {preset_id}")

    config = preset
    if use_variant:
        print("-> Varyasyon üretiliyor...")
        config = generate_variant(config, intensity=0.2, seed=seed)
        print(f"-> Varyasyon Hazır: {config.name}")

    print(f"\nRender Başlıyor... ({duration} saniye, {sample_rate} Hz)")
    start = time.time()
    signal = render_sound(
        preset_params=config.to_dict(),
        duration_sec=duration,
        sample_rate=sample_rate,
        seed=seed if seed is not None else config.seed,
    )
    elapsed = time.time() - start
    print(f"Render Tamamlandı ({elapsed:.2f}s)")

    _write_wav(output, signal, sample_rate)
    print(f"\nDosya kaydedildi: {output}")


def main(argv: Optional[list[str]] = None) -> None:
    """
    UltraGen CLI ana giriş noktası.
    """
    parser = argparse.ArgumentParser(
        prog="ultragen",
        description="UltraGen WhiteNoise Engine CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="Presetleri listeler")

    render_parser = subparsers.add_parser("render", help="Preset render eder")
    render_parser.add_argument("--preset", required=True, help="Preset ID")
    render_parser.add_argument("--output", default=_DEFAULT_OUTPUT_FILE, help="Çıktı dosyası")
    render_parser.add_argument("--duration", type=float, default=_DEFAULT_DURATION_SEC, help="Süre (sn)")
    render_parser.add_argument("--sample-rate", type=int, default=_DEFAULT_SAMPLE_RATE, help="Sample rate (Hz)")
    render_parser.add_argument("--variant", action="store_true", help="Varyasyon üret")
    render_parser.add_argument("--seed", type=int, default=None, help="Deterministik seed")

    args = parser.parse_args(argv)

    try:
        if args.command == "list":
            _cmd_list_presets()
        elif args.command == "render":
            _cmd_render(
                preset_id=args.preset,
                output=args.output,
                duration=args.duration,
                sample_rate=args.sample_rate,
                use_variant=args.variant,
                seed=args.seed,
            )
    except Exception as e:
        print(f"Hata: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
