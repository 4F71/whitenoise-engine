"""Node tabanli moduler ses motoru icin temel node siniflari."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence

import numpy as np


def _get_primary_input(inputs: Dict[str, np.ndarray]) -> np.ndarray:
    """Gelen sözlükteki ilk sinyali döndürür."""
    if not inputs:
        raise ValueError("En az bir girdi sinyali bekleniyor.")
    return next(iter(inputs.values()))


def _ensure_mono(signal: np.ndarray) -> np.ndarray:
    """
    Stereo veya garip şekilli (N,1)/(1,N) veriyi tek kanala indirger,
    mono ise aynen döndürür.
    """
    arr = np.asarray(signal, dtype=np.float64)

    # (N,) → zaten mono
    if arr.ndim == 1:
        return arr

    # 2D ise: hem 2-kanallı stereo’yu hem de 1-kanallı durumları tolere et
    if arr.ndim == 2:
        h, w = arr.shape

        # Tek kanallı 2D (N,1) veya (1,N) → düz mono vektöre çevir
        if 1 in arr.shape and (h == 1 or w == 1):
            return arr.reshape(-1)

        # Stereo: kanallar eksen-0’da (2, N)
        if h == 2:
            return arr.mean(axis=0)

        # Stereo: kanallar eksen-1’de (N, 2)
        if w == 2:
            return arr.mean(axis=1)

    raise ValueError(f"Beklenmeyen sinyal boyutu (mono bekleniyordu): shape={arr.shape}")


def _ensure_stereo(signal: np.ndarray) -> np.ndarray:
    """
    Mono sinyali stereo’ya çevirir, stereo ise şekli normalize eder.
    Çıkış daima (2, N) şeklindedir.
    """
    arr = np.asarray(signal, dtype=np.float64)

    # (N,) → [2, N]
    if arr.ndim == 1:
        return np.stack([arr, arr], axis=0)

    if arr.ndim == 2:
        h, w = arr.shape

        # (2, N) → kanallar zaten ilk eksende
        if h == 2:
            return arr

        # (N, 2) → transpoze edip (2, N) yap
        if w == 2:
            return arr.T

        # Tek kanallı 2D (N,1) veya (1,N) → önce mono yap, sonra stereo
        if 1 in arr.shape and (h == 1 or w == 1):
            mono = arr.reshape(-1)
            return np.stack([mono, mono], axis=0)

    raise ValueError(f"Beklenmeyen sinyal boyutu (stereo bekleniyordu): shape={arr.shape}")


def _to_channels_first(signal: np.ndarray) -> np.ndarray:
    """
    Sinyali (kanal, num_frames) formatına çevirir.
    1D → (1, N)
    2D → en kısa ekseni kanal kabul eder.
    3D tek-kanallı durumlarda da sıkıştırmayı dener.
    """
    arr = np.asarray(signal, dtype=np.float64)

    if arr.ndim == 1:
        return arr[np.newaxis, :]

    if arr.ndim == 2:
        h, w = arr.shape
        # Kanal sayısını küçük olan eksen kabul et (1,2 vs.)
        if h <= w:
            return arr
        return arr.T

    if arr.ndim == 3 and 1 in arr.shape:
        squeezed = arr.squeeze()
        return _to_channels_first(squeezed)

    raise ValueError(f"Beklenmeyen sinyal boyutu (kanal-first): shape={arr.shape}")



class BaseNode(ABC):
    """Tum node turetmeleri icin soyut taban sinif."""

    def __init__(self, name: str, sample_rate: int = 44_100) -> None:
        self.name = name
        self.sample_rate = sample_rate

    @abstractmethod
    def process(
        self, inputs: Dict[str, np.ndarray], params: Dict[str, Any]
    ) -> np.ndarray:
        """Girdileri isleyip cikti sinyali dondurur."""


class OscillatorNode(BaseNode):
    """Basit salinim dalga ureteci."""

    def process(
        self, inputs: Dict[str, np.ndarray], params: Dict[str, Any]
    ) -> np.ndarray:
        sr = int(params.get("sample_rate", self.sample_rate))
        freq = float(params.get("frequency", 440.0))
        amplitude = float(params.get("amplitude", 1.0))
        phase = float(params.get("phase", 0.0))
        duration = params.get("duration")
        num_frames = params.get("num_frames")
        if num_frames is None:
            if duration is None:
                raise ValueError("OscillatorNode icin duration veya num_frames verilmeli.")
            num_frames = int(duration * sr)
        t = np.arange(num_frames) / sr
        waveform = str(params.get("waveform", "sine")).lower()

        if waveform == "sine":
            signal = np.sin(2 * math.pi * freq * t + phase)
        elif waveform == "square":
            signal = np.sign(np.sin(2 * math.pi * freq * t + phase))
        elif waveform in {"saw", "sawtooth"}:
            signal = 2 * (t * freq - np.floor(0.5 + t * freq))
        elif waveform == "triangle":
            signal = 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1
        else:
            raise ValueError(f"Bilinmeyen waveform: {waveform}")

        return amplitude * signal


class NoiseNode(BaseNode):
    """Farkli tipte gurultu uretir."""

    def _pink_noise(self, num_frames: int, rng: np.random.Generator) -> np.ndarray:
        pink = np.zeros(num_frames)
        b0 = b1 = b2 = 0.0
        for i in range(num_frames):
            white = rng.standard_normal()
            b0 = 0.99765 * b0 + 0.0990460 * white
            b1 = 0.96300 * b1 + 0.2965164 * white
            b2 = 0.57000 * b2 + 1.0526913 * white
            pink[i] = b0 + b1 + b2 + 0.1848 * white
        return pink

    def _brown_noise(self, num_frames: int, rng: np.random.Generator) -> np.ndarray:
        white = rng.standard_normal(num_frames)
        brown = np.cumsum(white)
        brown -= brown.mean()
        brown /= np.max(np.abs(brown)) + 1e-9
        return brown

    def process(
        self, inputs: Dict[str, np.ndarray], params: Dict[str, Any]
    ) -> np.ndarray:
        sr = int(params.get("sample_rate", self.sample_rate))
        duration = params.get("duration")
        num_frames = params.get("num_frames")
        if num_frames is None:
            if duration is None:
                raise ValueError("NoiseNode icin duration veya num_frames verilmeli.")
            num_frames = int(duration * sr)

        noise_type = str(params.get("type", "white")).lower()
        seed = params.get("seed")
        rng = np.random.default_rng(seed)

        if noise_type == "white":
            signal = rng.standard_normal(num_frames)
        elif noise_type == "pink":
            signal = self._pink_noise(num_frames, rng)
        elif noise_type == "brown":
            signal = self._brown_noise(num_frames, rng)
        else:
            raise ValueError(f"Bilinmeyen gurultu tipi: {noise_type}")

        amplitude = float(params.get("amplitude", 1.0))
        return amplitude * signal


class FilterNode(BaseNode):
    """Basit bir kutuplu filtre uygular."""

    def _lowpass(self, signal: np.ndarray, cutoff: float, sr: int) -> np.ndarray:
        alpha = math.exp(-2.0 * math.pi * cutoff / sr)
        b0 = 1 - alpha
        y = np.zeros_like(signal)
        if signal.size == 0:
            return y
        y[0] = b0 * signal[0]
        for i in range(1, signal.size):
            y[i] = b0 * signal[i] + alpha * y[i - 1]
        return y

    def _highpass(self, signal: np.ndarray, cutoff: float, sr: int) -> np.ndarray:
        low = self._lowpass(signal, cutoff, sr)
        return signal - low

    def _apply_per_channel(
        self, signal: np.ndarray, func: Any, *args: Any
    ) -> np.ndarray:
        if signal.ndim == 1:
            return func(signal, *args)
        if signal.ndim == 2:
            return np.vstack([func(ch, *args) for ch in signal])
        raise ValueError("Beklenmeyen sinyal boyutu.")

    def process(
        self, inputs: Dict[str, np.ndarray], params: Dict[str, Any]
    ) -> np.ndarray:
        signal = _get_primary_input(inputs)
        sr = int(params.get("sample_rate", self.sample_rate))
        filter_type = str(params.get("type", "lowpass")).lower()
        cutoff = float(params.get("cutoff", 1_000.0))

        if filter_type == "lowpass":
            return self._apply_per_channel(signal, self._lowpass, cutoff, sr)
        if filter_type == "highpass":
            return self._apply_per_channel(signal, self._highpass, cutoff, sr)
        if filter_type == "bandpass":
            low_cutoff = float(params.get("low_cutoff", 200.0))
            high_cutoff = float(params.get("high_cutoff", 4_000.0))
            hp = self._apply_per_channel(signal, self._highpass, low_cutoff, sr)
            return self._apply_per_channel(hp, self._lowpass, high_cutoff, sr)

        raise ValueError(f"Bilinmeyen filtre tipi: {filter_type}")


class MixNode(BaseNode):
    """Birden fazla sinyali miksler."""

    def process(
        self, inputs: Dict[str, np.ndarray], params: Dict[str, Any]
    ) -> np.ndarray:
        if not inputs:
            raise ValueError("MixNode icin girdi sinyali yok.")

        signals = list(inputs.values())
        gains: Sequence[float] | None = params.get("gains")
        normalized = [_to_channels_first(sig) for sig in signals]
        max_len = max(sig.shape[1] for sig in normalized)

        mixed: list[np.ndarray] = []
        for idx, sig in enumerate(normalized):
            gain = gains[idx] if gains and idx < len(gains) else 1.0
            if sig.shape[1] < max_len:
                pad_width = ((0, 0), (0, max_len - sig.shape[1]))
                sig = np.pad(sig, pad_width)
            mixed.append(sig * gain)

        mix = sum(mixed) / len(mixed)
        return mix if mix.shape[0] > 1 else mix.squeeze(0)


class GainNode(BaseNode):
    """Sinyali lineer veya dB cinsinden kazandir."""

    def process(
        self, inputs: Dict[str, np.ndarray], params: Dict[str, Any]
    ) -> np.ndarray:
        signal = _get_primary_input(inputs)
        gain = float(params.get("gain", 1.0))
        gain_db = params.get("gain_db")
        if gain_db is not None:
            gain *= 10 ** (float(gain_db) / 20)
        return signal * gain


class EnvelopeNode(BaseNode):
    """ADSR zarfi uygular."""

    def process(
        self, inputs: Dict[str, np.ndarray], params: Dict[str, Any]
    ) -> np.ndarray:
        signal = _get_primary_input(inputs)
        mono = _ensure_mono(signal)
        sr = int(params.get("sample_rate", self.sample_rate))

        attack = float(params.get("attack", 0.01))
        decay = float(params.get("decay", 0.05))
        sustain = float(params.get("sustain", 0.8))
        release = float(params.get("release", 0.1))

        n = mono.size
        a_samp = int(attack * sr)
        d_samp = int(decay * sr)
        r_samp = int(release * sr)
        s_samp = max(n - a_samp - d_samp - r_samp, 0)

        envelope_parts = []
        if a_samp > 0:
            envelope_parts.append(np.linspace(0.0, 1.0, a_samp, endpoint=False))
        if d_samp > 0:
            envelope_parts.append(
                np.linspace(1.0, sustain, d_samp, endpoint=False)
            )
        if s_samp > 0:
            envelope_parts.append(np.full(s_samp, sustain))
        if r_samp > 0:
            envelope_parts.append(np.linspace(sustain, 0.0, r_samp))

        envelope = np.concatenate(envelope_parts) if envelope_parts else np.ones(n)
        if envelope.size < n:
            envelope = np.pad(envelope, (0, n - envelope.size), mode="edge")
        envelope = envelope[:n]

        shaped = mono * envelope
        if signal.ndim == 2:
            return _ensure_stereo(shaped)
        return shaped


class PanNode(BaseNode):
    """Mono sinyali stereo panorama alanina yerlestirir."""

    def process(
        self, inputs: Dict[str, np.ndarray], params: Dict[str, Any]
    ) -> np.ndarray:
        signal = _ensure_mono(_get_primary_input(inputs))
        pan = float(params.get("pan", 0.0))
        pan = float(np.clip(pan, -1.0, 1.0))
        angle = (pan + 1) * math.pi / 4
        left = math.cos(angle) * signal
        right = math.sin(angle) * signal
        return np.vstack([left, right])


class OutputNode(BaseNode):
    """Son ciktiyi duzenler ve klipler."""

    def process(
        self, inputs: Dict[str, np.ndarray], params: Dict[str, Any]
    ) -> np.ndarray:
        signal = _get_primary_input(inputs)
        stereo = _ensure_stereo(signal)
        return np.clip(stereo, -1.0, 1.0)
