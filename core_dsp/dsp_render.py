"""Basit katman bazlı render yardımcıları.

dsp_noise, dsp_filters, dsp_lfo ve dsp_fx modüllerini kullanarak katman
üretimi, miksaj ve son stereo WAV oluşturma işlemlerini soyutlar.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence, Tuple

from graph_engine.graph_loader import load_graph_from_json
from graph_engine.graph_core import Graph

import numpy as np

try:
    import dsp_noise
except ModuleNotFoundError:  # Proje içinde core_dsp adında kopya bulunabilir
    from core_dsp import dsp_noise  # type: ignore

try:
    import dsp_filters
except ModuleNotFoundError:
    from core_dsp import dsp_filters  # type: ignore

try:
    import dsp_lfo
except ModuleNotFoundError:
    from core_dsp import dsp_lfo  # type: ignore

try:
    import dsp_fx
except ModuleNotFoundError:
    from core_dsp import dsp_fx  # type: ignore


DEFAULT_SAMPLE_RATE = 48_000


def render_with_graph(preset_params: dict, duration: float, sample_rate: int = 44100):
    """
    Graph tabanlı presetleri çalıştırır.
    UI'nin gönderdiği JSON içeriğini alır, Graph Engine'i koşturur,
    stereo (2, N) array döndürür.
    """
    graph_cfg = preset_params.get("graph")
    if graph_cfg is None:
        raise ValueError("Bu preset bir Graph preset, ancak 'graph' alanı bulunamadı.")

    # Graph nesnesini bellekten oluştur
    graph = Graph(sample_rate=sample_rate)

    # Global paramlar (ui_streamlit üzerinden override edilebilir)
    global_params = graph_cfg.get("global_params", {})
    if global_params:
        graph.set_global_params(global_params)

    # Node'ları ekle
    for node_cfg in graph_cfg.get("nodes", []):
        name = node_cfg.get("name")
        type_name = node_cfg.get("type")
        params = node_cfg.get("params", {})

        if not name or not type_name:
            raise ValueError(f"Node tanımı eksik: {node_cfg}")

        # Node sınıfı
        from graph_engine.graph_loader import NODE_CLASS_MAP
        cls = NODE_CLASS_MAP.get(type_name.lower())
        if cls is None:
            raise ValueError(f"Bilinmeyen node tipi: {type_name}")

        instance = cls(name=name, sample_rate=sample_rate)
        graph.add_node(name, instance, params=params)

    # Edge'leri ekle
    for e in graph_cfg.get("edges", []):
        graph.add_edge(
            e["source"],
            e["target"],
            e.get("target_input", "input")
        )

    # Çalışma parametreleri
    num_frames = int(duration * sample_rate)
    run_params = {
        "sample_rate": sample_rate,
        "duration": duration,
        "num_frames": num_frames,
        "node_params": preset_params.get("node_params", {}),
    }


    stereo = graph.run(run_params)

    # Çıkan sinyali (N, 2) formatına zorla
    stereo = _ensure_stereo(stereo)

    return stereo

def _db_to_linear(db: float) -> float:
    """dB değerini lineer katsayıya çevirir."""

    return float(10 ** (db / 20.0))


def _ensure_stereo(signal: np.ndarray) -> np.ndarray:
    """
    Mono veya stereo sinyali (N, 2) stereo şekline getirir.

    Kabul edilen formatlar:
    - Mono: (N,)
    - Stereo: (N, 2)
    - Stereo (kanal-önce): (2, N)  → (N, 2)'ye çevrilir
    """

    arr = np.asarray(signal, dtype=np.float64)

    # Mono: (N,)
    if arr.ndim == 1:
        return np.stack([arr, arr], axis=1)

    # Stereo: (N, 2)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr.astype(np.float64)

    # Stereo: (2, N) → (N, 2)'ye çevir
    if arr.ndim == 2 and arr.shape[0] == 2:
        return arr.T.astype(np.float64)

    raise ValueError(
        f"Beklenmeyen sinyal boyutu {arr.shape}, mono (N,) veya stereo (N, 2)/(2, N) bekleniyordu."
    )


def _pad_to_length(signal: np.ndarray, length: int) -> np.ndarray:
    """Sinyali hedef uzunluğa sıfırlarla pad eder."""

    if signal.shape[0] >= length:
        return signal
    pad_len = length - signal.shape[0]
    pad = np.zeros((pad_len, signal.shape[1]), dtype=signal.dtype)
    return np.vstack([signal, pad])


def generate_layer(params: Mapping[str, Any]) -> np.ndarray:
    """
    Verilen parametrelerle tek bir katman üretir.

    Parametreler:
        params: Katman sözlüğü. Desteklenen anahtarlar:
            - type: Gürültü tipi (white, white_uniform, pink, brown, blue, violet).
            - duration: Saniye cinsinden süre.
            - sample_rate: Örnekleme hızı.
            - level_db: Katman çıkış seviyesi (dBFS).
            - filter: {"mode": "lpf/hpf/band", "cutoff_hz": float, "low_hz": float,
                       "high_hz": float}
            - lfo: {"target": "amp/filter/pan", "shape": "sine/triangle",
                    "freq_hz": float, "depth": float, "depth_hz": float, "width": float}
            - fx: {"warmth_db": float, "saturation_drive": float, "reverb": float,
                   "stereo_width": float, "normalize_peak": float, "pivot_hz": float}

    Döndürür:
        Stereo sinyal (N, 2) float64 ndarray.
    """

    sample_rate = int(params.get("sample_rate", DEFAULT_SAMPLE_RATE))
    duration = float(params.get("duration", 1.0))
    level_db = float(params.get("level_db", -12.0))
    n_samples = max(1, int(sample_rate * duration))

    noise_type = str(params.get("type", "pink")).lower()
    noise_map = {
        "white": dsp_noise.white_noise_gaussian,
        "white_uniform": dsp_noise.white_noise_uniform,
        "gaussian": dsp_noise.white_noise_gaussian,
        "uniform": dsp_noise.white_noise_uniform,
        "pink": dsp_noise.pink_noise_voss_mccartney,
        "brown": dsp_noise.brown_noise,
        "blue": dsp_noise.blue_noise,
        "violet": dsp_noise.violet_noise,
    }
    noise_fn = noise_map.get(noise_type, dsp_noise.white_noise_gaussian)
    mono = np.asarray(noise_fn(n_samples), dtype=np.float64)

    filter_cfg = params.get("filter") or {}
    mode = str(filter_cfg.get("mode", "")).lower()
    if mode == "lpf" and "cutoff_hz" in filter_cfg:
        mono = dsp_filters.lpf_one_pole(mono, float(filter_cfg["cutoff_hz"]), sample_rate)
    elif mode == "hpf" and "cutoff_hz" in filter_cfg:
        mono = dsp_filters.hpf_one_pole(mono, float(filter_cfg["cutoff_hz"]), sample_rate)
    elif mode == "band":
        low = float(filter_cfg.get("low_hz", 200.0))
        high = float(filter_cfg.get("high_hz", 4_000.0))
        mono = dsp_filters.bandpass_simple(mono, low, high, sample_rate)

    lfo_cfg = params.get("lfo") or {}
    lfo_freq = float(lfo_cfg.get("freq_hz", 0.0))
    lfo_shape = str(lfo_cfg.get("shape", "sine")).lower()
    lfo_target = str(lfo_cfg.get("target", "amp")).lower()
    lfo_signal: np.ndarray | None = None
    if lfo_freq > 0:
        if lfo_shape == "triangle":
            lfo_signal = dsp_lfo.triangle_lfo(n_samples, lfo_freq, sample_rate)
        else:
            lfo_signal = dsp_lfo.sine_lfo(n_samples, lfo_freq, sample_rate)

    stereo: np.ndarray
    if lfo_signal is not None and lfo_target == "pan":
        stereo = dsp_lfo.apply_stereo_pan_lfo(
            mono, lfo_signal, width=float(lfo_cfg.get("width", 1.0))
        )
    else:
        stereo = _ensure_stereo(mono)

    if lfo_signal is not None and lfo_target == "amp":
        depth = float(lfo_cfg.get("depth", 1.0))
        stereo = np.column_stack(
            [
                dsp_lfo.apply_volume_lfo(stereo[:, ch], lfo_signal, depth=depth)
                for ch in range(stereo.shape[1])
            ]
        )
    elif lfo_signal is not None and lfo_target == "filter":
        base_cutoff = float(filter_cfg.get("cutoff_hz", filter_cfg.get("low_hz", 2_000.0)))
        depth_hz = float(lfo_cfg.get("depth_hz", lfo_cfg.get("depth", 1_000.0)))
        stereo = np.column_stack(
            [
                dsp_lfo.apply_filter_cutoff_lfo(
                    stereo[:, ch],
                    base_cutoff_hz=base_cutoff,
                    lfo=lfo_signal,
                    depth_hz=depth_hz,
                    sample_rate=sample_rate,
                )
                for ch in range(stereo.shape[1])
            ]
        )

    stereo *= _db_to_linear(level_db)

    fx_cfg = params.get("fx") or {}
    if fx_cfg.get("warmth_db") is not None:
        stereo = dsp_fx.warmth(
            stereo,
            sample_rate=sample_rate,
            tilt_db=float(fx_cfg.get("warmth_db", 3.0)),
            saturation_drive=float(fx_cfg.get("saturation_drive", 1.05)),
        )
    elif fx_cfg.get("saturation_drive") is not None:
        stereo = dsp_fx.soft_saturation(stereo, drive=float(fx_cfg["saturation_drive"]))

    if fx_cfg.get("pivot_hz") or fx_cfg.get("tilt_db"):
        stereo = dsp_fx.tilt_eq(
            stereo,
            sample_rate=sample_rate,
            tilt_db=float(fx_cfg.get("tilt_db", fx_cfg.get("warmth_db", 0.0))),
            pivot_hz=float(fx_cfg.get("pivot_hz", 1_200.0)),
        )

    if fx_cfg.get("reverb") is not None:
        stereo = dsp_fx.fake_reverb(
            stereo,
            sample_rate=sample_rate,
            wet=float(fx_cfg.get("reverb", 0.25)),
            decay=float(fx_cfg.get("decay", 0.55)),
            lpf_hz=float(fx_cfg.get("reverb_lpf_hz", 6_000.0)),
        )

    if fx_cfg.get("stereo_width") is not None:
        stereo = dsp_fx.stereo_widen(stereo, width=float(fx_cfg["stereo_width"]))

    target_peak = float(fx_cfg.get("normalize_peak", 0.98))
    stereo = dsp_fx.normalize_gain(stereo, target_peak=target_peak)
    return stereo.astype(np.float64)


def mix_layers(layer_list: Iterable[np.ndarray]) -> np.ndarray:
    """
    Katman listesini miksleyip ortak uzunlukta stereo sinyal döndürür.

    Parametreler:
        layer_list: Stereo katman dizileri.

    Döndürür:
        Toplam miks (N, 2) float64 ndarray.
    """

    layers = [_ensure_stereo(np.asarray(layer)) for layer in layer_list]
    if not layers:
        raise ValueError("Mikslemek için en az bir katman gerekir.")

    max_len = max(layer.shape[0] for layer in layers)
    mix = np.zeros((max_len, 2), dtype=np.float64)
    for layer in layers:
        padded = _pad_to_length(layer, max_len)
        mix += padded

    return dsp_fx.normalize_gain(mix, target_peak=0.99)

def render_sound(preset_params: Mapping[str, Any], duration: float) -> Tuple[np.ndarray, int]:
    """
    Graph tabanlı presetleri veya klasik layer tabanlı presetleri render eder.
    """

    sample_rate = int(preset_params.get("sample_rate", DEFAULT_SAMPLE_RATE))

    # --- 1) Eğer Graph preset ise yeni motoru çalıştır ---
    if "graph" in preset_params:
        stereo = render_with_graph(preset_params, duration, sample_rate)
        return stereo, sample_rate

    # --- 2) Aksi halde eski layer-based motor çalışsın ---

    master_fx: MutableMapping[str, Any] = {
        **{"target_peak": 0.98},
        **(preset_params.get("master") or {}),
    }

    layers_cfg: Sequence[Mapping[str, Any]] = preset_params.get("layers", [])
    layers = []

    for cfg in layers_cfg:
        cfg_with_defaults: MutableMapping[str, Any] = {
            "duration": duration,
            "sample_rate": sample_rate,
            **cfg,
        }
        layers.append(generate_layer(cfg_with_defaults))

    stereo = mix_layers(layers)

    if master_fx.get("warmth_db") is not None:
        stereo = dsp_fx.warmth(
            stereo,
            sample_rate=sample_rate,
            tilt_db=float(master_fx.get("warmth_db", 2.0)),
            saturation_drive=float(master_fx.get("saturation_drive", 1.05)),
        )

    if master_fx.get("stereo_width") is not None:
        stereo = dsp_fx.stereo_widen(stereo, width=float(master_fx["stereo_width"]))

    if master_fx.get("reverb") is not None:
        stereo = dsp_fx.fake_reverb(
            stereo,
            sample_rate=sample_rate,
            wet=float(master_fx.get("reverb", 0.2)),
            decay=float(master_fx.get("decay", 0.5)),
            lpf_hz=float(master_fx.get("reverb_lpf_hz", 6_000.0)),
        )

    stereo = dsp_fx.normalize_gain(stereo, target_peak=float(master_fx["target_peak"]))

    output_path = preset_params.get("output_path")
    if output_path:
        _write_wav(Path(output_path), stereo, sample_rate)

    return stereo, sample_rate


def _write_wav(path: Path, stereo: np.ndarray, sample_rate: int) -> None:
    """Verilen stereo sinyali 16-bit WAV olarak diske yazar."""

    path.parent.mkdir(parents=True, exist_ok=True)
    audio16 = np.int16(np.clip(stereo, -1.0, 1.0) * 32767)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio16.tobytes())