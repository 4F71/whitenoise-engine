"""Lightweight DSP effects for simple audio shaping.

This module includes:
- Soft saturation using a hyperbolic tangent curve
- Warmth (tilt EQ plus saturation)
- Simple fake reverb (multi-tap delay with a low-pass tail)
- Stereo widening via mid/side processing
- Gain normalization
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np


def _as_2d(signal: np.ndarray) -> np.ndarray:
    """Ensure signal has shape (n_samples, n_channels)."""
    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError("Signal must be 1D (mono) or 2D (n_samples, n_channels).")
    return arr


def _restore_shape(original: np.ndarray, processed: np.ndarray) -> np.ndarray:
    """Return processed array with the same dimensionality as original."""
    if original.ndim == 1:
        return processed[:, 0]
    return processed


def _one_pole_lowpass(
    signal: np.ndarray,
    sample_rate: float,
    cutoff_hz: float,
) -> np.ndarray:
    """One-pole low-pass filter; simple and stable for tonal shaping."""
    if cutoff_hz <= 0:
        raise ValueError("cutoff_hz must be positive.")
    arr = _as_2d(signal)
    alpha = math.exp(-2.0 * math.pi * cutoff_hz / sample_rate)
    one_minus_alpha = 1.0 - alpha

    y = np.empty_like(arr)
    state = arr[0].copy()
    y[0] = state
    for i in range(1, len(arr)):
        state = alpha * state + one_minus_alpha * arr[i]
        y[i] = state
    return _restore_shape(signal, y)


def soft_saturation(
    signal: np.ndarray,
    drive: float = 1.25,
    makeup_gain: bool = True,
) -> np.ndarray:
    """Soft tanh saturation with optional makeup gain."""
    if drive <= 0:
        raise ValueError("drive must be positive.")
    saturated = np.tanh(np.asarray(signal, dtype=np.float64) * drive)
    if makeup_gain:
        gain = 1.0 / np.tanh(drive)
        saturated *= gain
    return saturated


def tilt_eq(
    signal: np.ndarray,
    sample_rate: float,
    tilt_db: float = 3.0,
    pivot_hz: float = 1200.0,
) -> np.ndarray:
    """Tilt EQ around a pivot frequency; positive tilt warms (more lows)."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if pivot_hz <= 0:
        raise ValueError("pivot_hz must be positive.")

    low = _one_pole_lowpass(signal, sample_rate, pivot_hz)
    high = np.asarray(signal, dtype=np.float64) - low

    low_gain = 10 ** (tilt_db / 20.0)
    high_gain = 1.0 / low_gain
    shaped = low * low_gain + high * high_gain

    max_gain = max(low_gain, high_gain)
    shaped /= max_gain
    return shaped


def warmth(
    signal: np.ndarray,
    sample_rate: float,
    tilt_db: float = 3.0,
    saturation_drive: float = 1.1,
) -> np.ndarray:
    """Combine tilt EQ with gentle saturation for a warmer tone."""
    tilted = tilt_eq(signal, sample_rate=sample_rate, tilt_db=tilt_db)
    return soft_saturation(tilted, drive=saturation_drive)


def fake_reverb(
    signal: np.ndarray,
    sample_rate: float,
    delays_ms: Optional[Iterable[float]] = None,
    decay: float = 0.55,
    lpf_hz: float = 6000.0,
    wet: float = 0.3,
) -> np.ndarray:
    """Simple faux reverb using multi-tap delays and a low-pass tail."""
    if wet < 0 or wet > 1:
        raise ValueError("wet must be between 0 and 1.")
    if decay <= 0:
        raise ValueError("decay must be positive.")
    delays = list(delays_ms) if delays_ms is not None else [25.0, 55.0, 85.0, 120.0]
    if not delays:
        raise ValueError("delays_ms must contain at least one value.")

    dry = _as_2d(signal)
    n_samples, n_channels = dry.shape
    delay_samples = [int(sample_rate * d / 1000.0) for d in delays]
    max_delay = max(delay_samples)
    out_len = n_samples + max_delay

    wet_buf = np.zeros((out_len, n_channels), dtype=np.float64)
    for tap, offset in enumerate(delay_samples):
        attenuation = decay ** tap
        wet_buf[offset: offset + n_samples] += dry * attenuation

    wet_buf = _as_2d(_one_pole_lowpass(wet_buf, sample_rate, lpf_hz))

    dry_padded = np.zeros_like(wet_buf)
    dry_padded[:n_samples] = dry

    mixed = dry_padded * (1.0 - wet) + wet_buf * wet
    return _restore_shape(signal, mixed)


def stereo_widen(signal: np.ndarray, width: float = 1.2) -> np.ndarray:
    """Mid/side stereo widening; width > 1 spreads, < 1 narrows."""
    arr = _as_2d(signal)
    if arr.shape[1] != 2:
        raise ValueError("stereo_widen expects a stereo (n_samples, 2) signal.")
    mid = 0.5 * (arr[:, 0] + arr[:, 1])
    side = 0.5 * (arr[:, 0] - arr[:, 1]) * width
    left = mid + side
    right = mid - side
    widened = np.stack([left, right], axis=1)
    return _restore_shape(signal, widened)


def normalize_gain(signal: np.ndarray, target_peak: float = 0.99) -> np.ndarray:
    """Normalize signal to a target peak level."""
    if target_peak <= 0:
        raise ValueError("target_peak must be positive.")
    arr = np.asarray(signal, dtype=np.float64)
    peak = np.max(np.abs(arr))
    if peak == 0:
        return arr
    return arr * (target_peak / peak)
