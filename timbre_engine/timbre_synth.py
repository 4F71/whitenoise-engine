from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from core_dsp import dsp_filters, dsp_fx, dsp_lfo, dsp_noise
from timbre_engine.timbre_params import TimbreParams, clamp_params, DEFAULT_NOISE_COLOR

NoiseFunc = Callable[[int], np.ndarray]


def _pick_noise_generator(color: str) -> NoiseFunc:
    """Map noise color to the corresponding generator."""
    table: Dict[str, NoiseFunc] = {
        "white": lambda n: dsp_noise.white_noise_gaussian(n, std=1.0),
        "pink": dsp_noise.pink_noise_voss_mccartney,
        "brown": dsp_noise.brown_noise,
        "blue": dsp_noise.blue_noise,
        "violet": dsp_noise.violet_noise,
    }
    return table.get(color.lower(), table[DEFAULT_NOISE_COLOR])


def _apply_static_tone(signal: np.ndarray, params: TimbreParams, sample_rate: float) -> np.ndarray:
    """Apply simple filter, band emphasis, and brightness shaping."""
    # One-pole low-pass for base tone plus a resonant-ish band component blended in.
    cutoff = max(20.0, min(params.filter_cutoff, sample_rate * 0.45))
    lpf = dsp_filters.lpf_one_pole(signal, cutoff_hz=cutoff, sample_rate=sample_rate)
    band_low = max(20.0, cutoff * 0.55)
    band_high = min(sample_rate * 0.49, cutoff * 1.6)
    band = dsp_filters.bandpass_simple(signal, low_cut_hz=band_low, high_cut_hz=band_high, sample_rate=sample_rate)
    toned = lpf * (1.0 - params.filter_resonance) + band * params.filter_resonance

    # 3-band EQ: brightness lifts highs, band_emphasis lifts mids.
    high_gain = 0.7 + 0.8 * float(np.clip(params.brightness, 0.0, 1.0))  # 0.7..1.5
    low_gain = 1.2 - 0.5 * float(np.clip(params.brightness, 0.0, 1.0))    # 1.2..0.7
    mid_gain = 1.0 + 0.5 * float(np.clip(params.band_emphasis, 0.0, 1.5))
    return dsp_filters.eq_3band(
        toned,
        low_gain=low_gain,
        mid_gain=mid_gain,
        high_gain=high_gain,
        low_cut_hz=band_low,
        high_cut_hz=band_high,
        sample_rate=sample_rate,
    )


def generate_sound(params: TimbreParams, duration: float, sr: int) -> np.ndarray:
    """
    Procedural timbre synthesis pipeline.
    Steps: noise gen -> tone shaping -> LFO movement -> warmth/saturation -> stereo/pan -> reverb -> normalize.
    Returns a stereo numpy array (float64).
    """
    params = clamp_params(params)
    num_samples = max(1, int(duration * sr))

    # Noise generation
    noise = _pick_noise_generator(params.noise_color)(num_samples)
    signal = noise * float(params.noise_level)

    # Static tone shaping
    signal = _apply_static_tone(signal, params, sample_rate=sr)

    # LFO movement: volume + dynamic low-pass sweep + pan
    lfo = dsp_lfo.sine_lfo(num_samples, freq_hz=params.lfo_rate, sample_rate=sr)
    if params.lfo_depth > 0:
        signal = dsp_lfo.apply_volume_lfo(signal, lfo, depth=params.lfo_depth)
        cutoff_depth = params.filter_cutoff * 0.6 * params.lfo_depth
        signal = dsp_lfo.apply_filter_cutoff_lfo(
            signal,
            base_cutoff_hz=params.filter_cutoff,
            lfo=lfo,
            depth_hz=cutoff_depth,
            sample_rate=sr,
            min_cutoff_hz=40.0,
            max_cutoff_hz=sr * 0.48,
        )
    stereo = dsp_lfo.apply_stereo_pan_lfo(signal, lfo=lfo, width=params.stereo_width)

    # Warmth + saturation + widening
    warmth_db = 1.0 + 5.0 * float(np.clip(params.warmth, 0.0, 1.0))
    stereo = dsp_fx.warmth(stereo, sample_rate=sr, tilt_db=warmth_db, saturation_drive=1.0 + params.warmth * 0.6)
    stereo = dsp_fx.soft_saturation(stereo, drive=1.0 + params.saturation, makeup_gain=True)
    stereo = dsp_fx.stereo_widen(stereo, width=1.0 + params.stereo_width * 0.6)

    # Reverb
    decay = np.clip(0.35 + 0.18 * params.reverb_time, 0.25, 0.95)
    stereo = dsp_fx.fake_reverb(
        stereo,
        sample_rate=sr,
        decay=decay,
        wet=float(np.clip(params.reverb_mix, 0.0, 1.0)),
        lpf_hz=min(sr * 0.45, 8000.0),
    )

    # Final gain trim
    return dsp_fx.normalize_gain(stereo, target_peak=0.98)
