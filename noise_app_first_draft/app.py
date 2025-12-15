#!/usr/bin/env python3
import argparse
import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Literal, Dict

import numpy as np

try:
    import streamlit as st
except Exception:  # Streamlit yoksa CLI çalışsın
    st = None

# ───────────────────────────────────
# SECTION 1 — CONFIG
# ───────────────────────────────────

SAMPLE_RATE = 44_100
DEFAULT_DURATIONS = ["1h", "2h", "3h", "8h", "10h"]
BIT_DEPTH = 16
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Şablon ve etiket sözlüğü (standartlaştırma)
STANDARD_TONE_TAGS = {"warm", "deep", "airy", "bright", "shimmer"}
STANDARD_AMBIENCE_TAGS = {"rain", "ocean", "wind", "fire", "forest", "urban", "space", "water"}
TEMPLATE_DEFAULTS = {
    "sleep": {"tone": "warm", "ambience": "rain", "noise": "brown"},
    "focus": {"tone": "bright", "ambience": "urban", "noise": "white"},
    "meditation": {"tone": "airy", "ambience": "ocean", "noise": "pink"},
    "asmr": {"tone": "soft", "ambience": "rain", "noise": "pink"},
    "ambient": {"tone": "warm", "ambience": "forest", "noise": "pink"},
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ambience_app")

# ───────────────────────────────────
# SECTION 2 — DATA MODELS
# ───────────────────────────────────


@dataclass
class NoiseLayerConfig:
    """Gürültü katman ayarları."""
    type: str
    level: float = -18.0
    color: Optional[str] = None
    lp_hz: Optional[float] = None
    hp_hz: Optional[float] = None
    lfo_rate: float = 0.0
    drift: float = 0.0
    width: float = 0.5


@dataclass
class AmbienceLayerConfig:
    """Ambiyans katman ayarları."""
    type: str
    level: float = -18.0
    variant: Optional[str] = None
    width: float = 0.7
    movement: float = 0.3
    lfo_rate: float = 0.1


@dataclass
class ToneLayerConfig:
    """Ton/pad katman ayarları."""
    type: str
    frequency: float
    level: float = -24.0
    wave: str = "sine"
    pad: bool = False
    pattern: Optional[str] = None
    style: str = "light"


@dataclass
class BrainwaveLayerConfig:
    """Beyin dalgası katman ayarları."""
    mode: str  # binaural / isochronic / solfeggio / chakra / om
    base_freq: float
    beat_freq: float = 0.0
    pulse_rate: float = 0.0
    level: float = -24.0
    chakra: Optional[str] = None
    breath_pattern: Optional[str] = None


@dataclass
class SleepMixConfig:
    """Uyku miks ana ayarları."""
    piano: bool = True
    piano_intensity: str = "light"
    ambience_type: str = "rain"
    noise_type: str = "pink"
    brainwave: Optional[str] = None
    pad: bool = True
    width: float = 0.6
    volume_db: float = -14.0


@dataclass
class PresetConfig:
    """Preset tanımı."""
    id: str
    name: str
    description: str
    tags: List[str]
    target_use: Literal["sleep", "focus", "meditation", "asmr", "ambient"]
    base_duration: str
    noise_layers: List[NoiseLayerConfig] = field(default_factory=list)
    ambience_layers: List[AmbienceLayerConfig] = field(default_factory=list)
    tone_layers: List[ToneLayerConfig] = field(default_factory=list)
    brainwave_layers: List[BrainwaveLayerConfig] = field(default_factory=list)
    sleepmix: Optional[SleepMixConfig] = None


@dataclass
class MasterRenderConfig:
    """Genel render ayarları."""
    master_gain_db: float = -1.0
    fade_in: float = 15.0
    fade_out: float = 20.0


MASTER_RENDER = MasterRenderConfig()

# ───────────────────────────────────
# SECTION 3 — UTILITY FUNCTIONS
# ───────────────────────────────────


def parse_duration(text: str) -> int:
    """Süreyi saniyeye çevirir, ör: 2h30m, 45m, 600s."""
    t = text.lower().strip()
    total = 0
    num = ""
    for ch in t:
        if ch.isdigit():
            num += ch
        else:
            if not num:
                continue
            val = int(num)
            if ch == "h":
                total += val * 3600
            elif ch == "m":
                total += val * 60
            elif ch == "s":
                total += val
            num = ""
    if num:
        total += int(num)
    return max(total, 1)


def db_to_linear(db: float) -> float:
    """dB değeri lineere çevirir."""
    return 10 ** (db / 20.0)


def normalize_audio(audio: np.ndarray, target_level: float = -1.0) -> np.ndarray:
    """Sinyali hedef dBFS seviyesine normalizer."""
    peak = np.max(np.abs(audio)) + 1e-9
    target = db_to_linear(target_level)
    return audio * (target / peak)


def fade_in_out(audio: np.ndarray, sample_rate: int, fade_in_seconds: float, fade_out_seconds: float) -> np.ndarray:
    """Başa/sona fade uygular (mono veya stereo)."""
    if audio.ndim == 1:
        n = len(audio)
        fi = min(int(sample_rate * fade_in_seconds), n)
        fo = min(int(sample_rate * fade_out_seconds), n)
        env = np.ones(n)
        if fi > 0:
            env[:fi] = np.linspace(0, 1, fi)
        if fo > 0:
            env[-fo:] = np.linspace(1, 0, fo)
        return audio * env
    elif audio.ndim == 2:
        n = audio.shape[1]
        fi = min(int(sample_rate * fade_in_seconds), n)
        fo = min(int(sample_rate * fade_out_seconds), n)
        env = np.ones(n)
        if fi > 0:
            env[:fi] = np.linspace(0, 1, fi)
        if fo > 0:
            env[-fo:] = np.linspace(1, 0, fo)
        return audio * env
    else:
        return audio


def stereo_widen(left_right: np.ndarray, amount: float) -> np.ndarray:
    """Stereo genişliği arttırır."""
    mid = (left_right[0] + left_right[1]) / 2
    side = (left_right[0] - left_right[1]) / 2
    widened_side = side * (1 + amount)
    return np.vstack([mid + widened_side, mid - widened_side])


def pan(mono: np.ndarray, pan_value: float) -> np.ndarray:
    """Mono sinyali stereo panorama taşır."""
    left = mono * math.sqrt(0.5 * (1 - pan_value))
    right = mono * math.sqrt(0.5 * (1 + pan_value))
    return np.vstack([left, right])


def apply_jitter(audio: np.ndarray, depth: float) -> np.ndarray:
    """Hızlı küçük jitter ekler."""
    noise = (np.random.rand(*audio.shape) - 0.5) * depth
    return np.clip(audio + noise, -1, 1)


def apply_drift(audio: np.ndarray, drift_rate: float) -> np.ndarray:
    """Yavaş drift modülasyonu ekler."""
    n = len(audio)
    t = np.linspace(0, n / SAMPLE_RATE, n)
    drift = np.sin(2 * np.pi * drift_rate * t) * 0.01
    return audio * (1 + drift)


def apply_warm_coloration(audio: np.ndarray) -> np.ndarray:
    """Hafif sıcaklık için EQ eğimi ve yumuşak saturasyon."""
    tilt = np.linspace(1.1, 0.9, len(audio))
    colored = audio * tilt
    return soft_saturation(colored, 0.6)


def natural_dither(audio: np.ndarray, level: float = 1e-4) -> np.ndarray:
    """Çok düşük seviyeli dither ekler, yapay hissi azaltır."""
    return np.clip(audio + (np.random.rand(*audio.shape) - 0.5) * level, -1, 1)


def micro_room_reverb(stereo: np.ndarray, sample_rate: int, mix: float = 0.12) -> np.ndarray:
    """Basit comb+allpass mikro oda hissi ekler."""
    dry = stereo.copy()
    delays_ms = [23, 47, 59]
    fb = 0.2
    wet = np.zeros_like(stereo)
    for d in delays_ms:
        delay = int(sample_rate * d / 1000)
        if delay <= 0 or delay >= stereo.shape[1]:
            continue
        padded = np.pad(stereo, ((0, 0), (delay, 0)), mode="constant")[:, :stereo.shape[1]]
        wet += padded * fb
    # basit allpass dokunuşu
    ap_delay = int(sample_rate * 0.011)
    if ap_delay > 0 and ap_delay < stereo.shape[1]:
        ap = np.pad(stereo, ((0, 0), (ap_delay, 0)), mode="constant")[:, :stereo.shape[1]]
        wet = wet + (ap - wet) * 0.3
    out = dry * (1 - mix) + wet * mix
    return out


def generate_filename(preset_id: str, duration_str: str, variant_index: int) -> str:
    """Çıkış dosya adını üretir."""
    return f"{preset_id}_{duration_str}_v{variant_index}.wav"


# ───────────────────────────────────
# SECTION 4 — CORE DSP NOISE ENGINE
# ───────────────────────────────────


def white_noise_gaussian(length: int) -> np.ndarray:
    """Gaussian beyaz gürültü üretir."""
    return np.random.normal(0, 0.3, length)


def white_noise_uniform(length: int) -> np.ndarray:
    """Uniform beyaz gürültü üretir."""
    return (np.random.rand(length) - 0.5) * 0.6


def pink_noise(length: int) -> np.ndarray:
    """Basit pink gürültü üretir."""
    # Voss-McCartney benzeri
    rows = 16
    array = np.zeros(length)
    values = np.random.rand(rows)
    counter = np.zeros(rows)
    for i in range(length):
        idx = np.where(counter == 0)[0]
        if len(idx) > 0:
            # Üst sınırın alt sınırdan büyük olması için +1 ekliyoruz
            counter[idx] = np.random.randint(1, 2 ** (np.arange(len(idx)) + 1))
            values[idx] = np.random.rand(len(idx))
        counter -= 1
        array[i] = values.sum()
    array = (array - np.mean(array)) / (np.std(array) + 1e-9)
    return array * 0.3


def brown_noise(length: int) -> np.ndarray:
    """Brownian gürültü üretir."""
    steps = np.random.normal(0, 0.02, length)
    out = np.cumsum(steps)
    out = out - np.mean(out)
    out /= (np.max(np.abs(out)) + 1e-9)
    return out * 0.6


def blue_noise(length: int) -> np.ndarray:
    """Blue noise üretir."""
    white = white_noise_gaussian(length)
    diff = np.diff(white, prepend=white[0])
    diff /= (np.max(np.abs(diff)) + 1e-9)
    return diff * 0.5


def violet_noise(length: int) -> np.ndarray:
    """Violet noise üretir."""
    blue = blue_noise(length)
    diff = np.diff(blue, prepend=blue[0])
    diff /= (np.max(np.abs(diff)) + 1e-9)
    return diff * 0.4


def low_pass_filter(audio: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
    """Basit IIR low-pass."""
    rc = 1.0 / (2 * math.pi * cutoff_hz)
    dt = 1.0 / sample_rate
    alpha = dt / (rc + dt)
    out = np.zeros_like(audio)
    out[0] = audio[0]
    for i in range(1, len(audio)):
        out[i] = out[i - 1] + alpha * (audio[i] - out[i - 1])
    return out


def high_pass_filter(audio: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
    """Basit IIR high-pass."""
    rc = 1.0 / (2 * math.pi * cutoff_hz)
    dt = 1.0 / sample_rate
    alpha = rc / (rc + dt)
    out = np.zeros_like(audio)
    out[0] = audio[0]
    for i in range(1, len(audio)):
        out[i] = alpha * (out[i - 1] + audio[i] - audio[i - 1])
    return out


def band_pass_filter(audio: np.ndarray, low_cut: float, high_cut: float, sample_rate: int) -> np.ndarray:
    """Basit band-pass."""
    return high_pass_filter(low_pass_filter(audio, high_cut, sample_rate), low_cut, sample_rate)


def lfo_sine(length_samples: int, rate_hz: float, sample_rate: int) -> np.ndarray:
    """Sine LFO üretir."""
    t = np.arange(length_samples) / sample_rate
    return np.sin(2 * np.pi * rate_hz * t)


def apply_lfo_volume(audio: np.ndarray, lfo_signal: np.ndarray) -> np.ndarray:
    """LFO ile volume modülasyonu."""
    mod = 0.5 + 0.5 * lfo_signal
    return audio * mod


def apply_lfo_filter_cutoff(audio: np.ndarray, base_cutoff: float, depth: float, lfo_signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """LFO ile cutoff modülasyonu."""
    out = np.zeros_like(audio)
    for i in range(len(audio)):
        cutoff = max(50.0, base_cutoff + depth * lfo_signal[i])
        out[i] = low_pass_filter(audio[i:i+1], cutoff, sample_rate)[0]
    return out


def split_into_bands(audio: np.ndarray, sample_rate: int):
    """Düşük/orta/yüksek bantlara ayırır."""
    low = low_pass_filter(audio, 200.0, sample_rate)
    high = high_pass_filter(audio, 4000.0, sample_rate)
    mid = audio - (low + high)
    return low, mid, high


def multiband_shape(low: np.ndarray, mid: np.ndarray, high: np.ndarray, lg: float, mg: float, hg: float) -> np.ndarray:
    """Bant kazançlarını uygular."""
    return low * lg + mid * mg + high * hg


def soft_saturation(audio: np.ndarray, drive: float) -> np.ndarray:
    """Yumuşak tanh saturasyon."""
    driven = audio * (1 + drive)
    return np.tanh(driven)


def generate_subharmonic(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Subharmonik üretir."""
    rect = np.abs(audio)
    lp = low_pass_filter(rect, 120, sample_rate)
    sub = lp * 0.5
    return sub


def blend_subbass(audio: np.ndarray, sub: np.ndarray, amount: float) -> np.ndarray:
    """Sub'u miksler."""
    return audio + sub * amount


def add_random_events(audio: np.ndarray, sample_rate: int, events_config: Dict) -> np.ndarray:
    """Rastgele mikro olaylar ekler."""
    out = audio.copy()
    n = len(audio)
    for _ in range(events_config.get("swells", 8)):
        pos = random.randint(0, n - 1)
        length = min(int(sample_rate * random.uniform(0.2, 1.5)), n - pos - 1)
        window = np.hanning(length * 2)[length:]
        out[pos:pos+length] += window * 0.05
    for _ in range(events_config.get("wiggles", 8)):
        pos = random.randint(0, n - 1)
        out[pos] *= random.uniform(0.9, 1.1)
    return np.clip(out, -1, 1)


# ───────────────────────────────────
# SECTION 5 — AMBIENCE ENGINE
# ───────────────────────────────────

def _make_noise_bed(length_samples: int, noise_type: str) -> np.ndarray:
    if noise_type == "white":
        base = white_noise_gaussian(length_samples)
    elif noise_type == "white_u":
        base = white_noise_uniform(length_samples)
    elif noise_type == "pink":
        base = pink_noise(length_samples)
    elif noise_type == "brown":
        base = brown_noise(length_samples)
    elif noise_type == "blue":
        base = blue_noise(length_samples)
    elif noise_type == "violet":
        base = violet_noise(length_samples)
    else:
        base = white_noise_gaussian(length_samples)
    # Doğallık için hafif karışım: biraz pink/brown serpiştir
    mix = base
    mix += 0.08 * pink_noise(length_samples)
    mix += 0.05 * brown_noise(length_samples)
    mix /= np.max(np.abs(mix) + 1e-9)
    return mix * 0.9


def ambience_template(length_seconds: int, noise_type: str, lp: float, hp: float, lfo_rate: float, drift: float,
                      width: float, movement: float, events=True) -> np.ndarray:
    """Ambiyans üretim şablonu."""
    length = length_seconds * SAMPLE_RATE
    bed = _make_noise_bed(length, noise_type)
    if lp:
        bed = low_pass_filter(bed, lp, SAMPLE_RATE)
    if hp:
        bed = high_pass_filter(bed, hp, SAMPLE_RATE)
    lfo = lfo_sine(length, lfo_rate, SAMPLE_RATE) if lfo_rate > 0 else np.zeros(length)
    bed = apply_lfo_volume(bed, lfo * 0.5 + 0.5)
    bed = apply_drift(bed, drift)
    if events:
        bed = add_random_events(bed, SAMPLE_RATE, {"swells": 6, "wiggles": 12})
    # Yağmur/su türlerinde damla efektleri
    if "rain" in noise_type or "water" in noise_type or "ocean" in noise_type:
        drops = np.zeros_like(bed)
        drop_count = max(20, length_seconds // 2)
        for _ in range(drop_count):
            pos = random.randint(0, max(1, length - 2000))
            dlen = random.randint(800, 2000)
            if pos + dlen >= len(drops):
                continue
            env = np.hanning(dlen * 2)[dlen:]
            clicks = high_pass_filter(white_noise_uniform(dlen), 3000, SAMPLE_RATE) * env * 0.15
            drops[pos:pos + dlen] += clicks
        bed += drops * 0.4
    stereo = pan(bed, movement)

    # Hafif hava/air katmanı
    air = high_pass_filter(white_noise_uniform(length), 6000, SAMPLE_RATE) * 0.03
    stereo += pan(air, -movement * 0.5)

    # Stereo genişlik ve hareket LFO
    stereo = stereo_widen(stereo, width)
    t = np.arange(length) / SAMPLE_RATE
    pan_lfo = 0.15 * np.sin(2 * np.pi * random.uniform(0.015, 0.05) * t)
    stereo[0] *= 1 - pan_lfo * 0.5
    stereo[1] *= 1 + pan_lfo * 0.5
    return stereo


def rain_light(length_seconds: int) -> np.ndarray:
    return ambience_template(length_seconds, "pink", lp=8000, hp=300, lfo_rate=0.2, drift=0.1, width=0.5, movement=0.1)


def rain_medium(length_seconds: int) -> np.ndarray:
    return ambience_template(length_seconds, "white", lp=9000, hp=200, lfo_rate=0.25, drift=0.12, width=0.6, movement=0.15)


def rain_heavy(length_seconds: int) -> np.ndarray:
    return ambience_template(length_seconds, "white", lp=10000, hp=120, lfo_rate=0.3, drift=0.15, width=0.7, movement=0.2)


def rain_window(length_seconds: int) -> np.ndarray:
    amb = ambience_template(length_seconds, "pink", lp=6000, hp=180, lfo_rate=0.18, drift=0.12, width=0.65, movement=0.18)
    amb = blend_subbass(amb, generate_subharmonic(amb.mean(axis=0), SAMPLE_RATE), 0.05)
    return amb


def rain_tent(length_seconds: int) -> np.ndarray:
    return ambience_template(length_seconds, "brown", lp=5000, hp=80, lfo_rate=0.15, drift=0.14, width=0.55, movement=0.25)


def rainforest_rain(length_seconds: int) -> np.ndarray:
    amb = ambience_template(length_seconds, "pink", lp=7000, hp=150, lfo_rate=0.22, drift=0.13, width=0.6, movement=0.22)
    chirp = ambience_template(length_seconds, "blue", lp=6000, hp=2000, lfo_rate=0.4, drift=0.05, width=0.8, movement=-0.2, events=False)
    return normalize_audio(amb + 0.3 * chirp, -6)


def ocean_soft(length_seconds: int) -> np.ndarray:
    return ambience_template(length_seconds, "pink", lp=6000, hp=60, lfo_rate=0.1, drift=0.1, width=0.7, movement=0.2)


def ocean_deep(length_seconds: int) -> np.ndarray:
    amb = ambience_template(length_seconds, "brown", lp=4000, hp=40, lfo_rate=0.08, drift=0.08, width=0.7, movement=0.15)
    sub = generate_subharmonic(amb.mean(axis=0), SAMPLE_RATE)
    amb = blend_subbass(amb, sub, 0.2)
    return amb


def river_stream(length_seconds: int) -> np.ndarray:
    return ambience_template(length_seconds, "white", lp=7000, hp=300, lfo_rate=0.35, drift=0.09, width=0.6, movement=-0.2)


def waterfall_soft(length_seconds: int) -> np.ndarray:
    return ambience_template(length_seconds, "white", lp=10000, hp=500, lfo_rate=0.4, drift=0.12, width=0.55, movement=0.1)


def lake_waves(length_seconds: int) -> np.ndarray:
    return ambience_template(length_seconds, "pink", lp=5000, hp=80, lfo_rate=0.15, drift=0.1, width=0.5, movement=0.05)


def fireplace_soft(length_seconds: int) -> np.ndarray:
    amb = ambience_template(length_seconds, "brown", lp=4000, hp=150, lfo_rate=0.2, drift=0.1, width=0.45, movement=0.1)
    crackle = ambience_template(length_seconds, "white_u", lp=8000, hp=2000, lfo_rate=0.5, drift=0.02, width=0.7, movement=-0.1)
    return normalize_audio(amb + 0.2 * crackle, -6)


def campfire_crackle(length_seconds: int) -> np.ndarray:
    return fireplace_soft(length_seconds)


def soft_wind(length_seconds: int) -> np.ndarray:
    return ambience_template(length_seconds, "pink", lp=6000, hp=80, lfo_rate=0.1, drift=0.08, width=0.6, movement=-0.1)


def mountain_wind(length_seconds: int) -> np.ndarray:
    return ambience_template(length_seconds, "blue", lp=9000, hp=150, lfo_rate=0.15, drift=0.12, width=0.7, movement=0.2)


def storm_wind(length_seconds: int) -> np.ndarray:
    amb = ambience_template(length_seconds, "white", lp=12000, hp=60, lfo_rate=0.22, drift=0.18, width=0.8, movement=0.25)
    return blend_subbass(amb, generate_subharmonic(amb.mean(axis=0), SAMPLE_RATE), 0.15)


def desert_wind(length_seconds: int) -> np.ndarray:
    return ambience_template(length_seconds, "pink", lp=8000, hp=120, lfo_rate=0.12, drift=0.09, width=0.65, movement=-0.15)


def forest_morning(length_seconds: int) -> np.ndarray:
    amb = ambience_template(length_seconds, "pink", lp=7000, hp=150, lfo_rate=0.18, drift=0.12, width=0.55, movement=0.12)
    birds = ambience_template(length_seconds, "blue", lp=7000, hp=2500, lfo_rate=0.5, drift=0.03, width=0.8, movement=-0.2, events=False)
    return normalize_audio(amb + 0.25 * birds, -7)


def forest_night(length_seconds: int) -> np.ndarray:
    amb = ambience_template(length_seconds, "brown", lp=6000, hp=70, lfo_rate=0.1, drift=0.1, width=0.5, movement=-0.05)
    insects = ambience_template(length_seconds, "blue", lp=9000, hp=3000, lfo_rate=0.6, drift=0.02, width=0.75, movement=0.2, events=False)
    return normalize_audio(amb + 0.2 * insects, -7)


def jungle_air(length_seconds: int) -> np.ndarray:
    amb = ambience_template(length_seconds, "pink", lp=8000, hp=120, lfo_rate=0.22, drift=0.13, width=0.65, movement=0.2)
    life = ambience_template(length_seconds, "blue", lp=8000, hp=2500, lfo_rate=0.5, drift=0.04, width=0.8, movement=-0.2, events=False)
    return normalize_audio(amb + 0.3 * life, -6)


def cold_mountain(length_seconds: int) -> np.ndarray:
    return ambience_template(length_seconds, "brown", lp=5000, hp=50, lfo_rate=0.12, drift=0.1, width=0.6, movement=0.15)


def fan_noise(length_seconds: int) -> np.ndarray:
    return ambience_template(length_seconds, "white_u", lp=6000, hp=120, lfo_rate=0.08, drift=0.05, width=0.4, movement=0.05)


def air_conditioner(length_seconds: int) -> np.ndarray:
    return ambience_template(length_seconds, "white", lp=7000, hp=150, lfo_rate=0.06, drift=0.03, width=0.35, movement=-0.05)


def server_room(length_seconds: int) -> np.ndarray:
    return ambience_template(length_seconds, "blue", lp=10000, hp=80, lfo_rate=0.12, drift=0.08, width=0.5, movement=0.1)


def city_low_rumble(length_seconds: int) -> np.ndarray:
    amb = ambience_template(length_seconds, "brown", lp=4000, hp=40, lfo_rate=0.05, drift=0.09, width=0.5, movement=-0.1)
    sub = generate_subharmonic(amb.mean(axis=0), SAMPLE_RATE)
    return blend_subbass(amb, sub, 0.2)


def train_rumble(length_seconds: int) -> np.ndarray:
    amb = ambience_template(length_seconds, "brown", lp=3000, hp=50, lfo_rate=0.2, drift=0.1, width=0.45, movement=0.15)
    ratt = ambience_template(length_seconds, "white", lp=9000, hp=500, lfo_rate=0.3, drift=0.05, width=0.7, movement=-0.2, events=False)
    return normalize_audio(amb + 0.25 * ratt, -6)


def space_station(length_seconds: int) -> np.ndarray:
    return ambience_template(length_seconds, "white", lp=8000, hp=60, lfo_rate=0.15, drift=0.12, width=0.7, movement=0.2)


def interstellar_drone(length_seconds: int) -> np.ndarray:
    amb = ambience_template(length_seconds, "pink", lp=7000, hp=40, lfo_rate=0.1, drift=0.1, width=0.75, movement=0.1)
    sub = generate_subharmonic(amb.mean(axis=0), SAMPLE_RATE)
    return blend_subbass(amb, sub, 0.3)


def nebula_hum(length_seconds: int) -> np.ndarray:
    amb = ambience_template(length_seconds, "violet", lp=12000, hp=200, lfo_rate=0.2, drift=0.08, width=0.8, movement=-0.15)
    return amb

AMBIENCE_MAP = {
    "rain_light": rain_light,
    "rain_medium": rain_medium,
    "rain_heavy": rain_heavy,
    "rain_window": rain_window,
    "rain_tent": rain_tent,
    "rainforest_rain": rainforest_rain,
    "ocean_soft": ocean_soft,
    "ocean_deep": ocean_deep,
    "river_stream": river_stream,
    "waterfall_soft": waterfall_soft,
    "lake_waves": lake_waves,
    "fireplace_soft": fireplace_soft,
    "campfire_crackle": campfire_crackle,
    "soft_wind": soft_wind,
    "mountain_wind": mountain_wind,
    "storm_wind": storm_wind,
    "desert_wind": desert_wind,
    "forest_morning": forest_morning,
    "forest_night": forest_night,
    "jungle_air": jungle_air,
    "cold_mountain": cold_mountain,
    "fan_noise": fan_noise,
    "air_conditioner": air_conditioner,
    "server_room": server_room,
    "city_low_rumble": city_low_rumble,
    "train_rumble": train_rumble,
    "space_station": space_station,
    "interstellar_drone": interstellar_drone,
    "nebula_hum": nebula_hum,
}

# ───────────────────────────────────
# SECTION 6 — TONE / PAD / PIANO ENGINE
# ───────────────────────────────────

def sine_wave(freq: float, duration: float, sample_rate: int) -> np.ndarray:
    """Sine dalgası üretir."""
    t = np.arange(int(duration * sample_rate)) / sample_rate
    return np.sin(2 * np.pi * freq * t)


def triangle_wave(freq: float, duration: float, sample_rate: int) -> np.ndarray:
    """Üçgen dalga üretir."""
    t = np.arange(int(duration * sample_rate)) / sample_rate
    return 2 * np.abs(2 * (t * freq - np.floor(0.5 + t * freq))) - 1


def square_wave(freq: float, duration: float, sample_rate: int) -> np.ndarray:
    """Kare dalga üretir."""
    t = np.arange(int(duration * sample_rate)) / sample_rate
    return np.sign(np.sin(2 * np.pi * freq * t))


def adsr_envelope(length_samples: int, attack: float, decay: float, sustain_level: float, release: float, sample_rate: int) -> np.ndarray:
    """ADSR zarf üretir."""
    a = int(sample_rate * attack)
    d = int(sample_rate * decay)
    r = int(sample_rate * release)
    s = max(0, length_samples - a - d - r)
    env = np.concatenate([
        np.linspace(0, 1, a, endpoint=False) if a > 0 else np.array([]),
        np.linspace(1, sustain_level, d, endpoint=False) if d > 0 else np.array([]),
        np.full(s, sustain_level),
        np.linspace(sustain_level, 0, r) if r > 0 else np.array([]),
    ])
    if len(env) < length_samples:
        env = np.pad(env, (0, length_samples - len(env)), mode="edge")
    return env[:length_samples]


def generate_ambient_pad(config: ToneLayerConfig, duration: int, sample_rate: int) -> np.ndarray:
    """Yumuşak pad üretir, stil bazlı tonlama uygular."""
    base = sine_wave(config.frequency, duration, sample_rate)
    noise = low_pass_filter(white_noise_gaussian(len(base)), 1800, sample_rate) * 0.25
    shimmer = sine_wave(config.frequency * 2, duration, sample_rate) * 0.1
    # Formant benzeri hafif band-pass ekle
    formant = band_pass_filter(white_noise_gaussian(len(base)), 500, 1800, sample_rate) * 0.08
    tone = base + noise + shimmer + formant

    style = (config.style or "warm").lower()
    if style == "airy":
        tone = high_pass_filter(tone, 500, sample_rate)
        tone += sine_wave(config.frequency * 1.5, duration, sample_rate) * 0.15
    elif style == "bowed":
        tone = band_pass_filter(tone, 300, 2500, sample_rate)
        trem = lfo_sine(len(tone), 0.35, sample_rate) * 0.15 + 0.85
        tone *= trem
    elif style == "pluck":
        tone = high_pass_filter(tone, 800, sample_rate)
    elif style == "shimmer":
        upper = sine_wave(config.frequency * 4, duration, sample_rate) * 0.08
        tone += upper

    mix = normalize_audio(tone, -6)
    env = adsr_envelope(len(mix), 1.2 if style == "pluck" else 2.5, 2.5, 0.65, 2.8, sample_rate)
    lfo = lfo_sine(len(mix), 0.08, sample_rate) * 0.2 + 0.8
    pad = mix * env * lfo
    pad = soft_saturation(pad, 0.4 if style != "airy" else 0.3)
    return pad


def generate_piano_note(frequency: float, intensity: float, sample_rate: int) -> np.ndarray:
    """Ambient piyano notası üretir, harmonik ve karakter ekler."""
    dur = 4.0 + random.random() * 2
    t = np.arange(int(dur * sample_rate)) / sample_rate

    # Hafif wow/flutter ile zaman modülasyonu
    wow = 0.0015 * np.sin(2 * np.pi * 0.2 * t)
    flutter = 0.0005 * np.sin(2 * np.pi * 5.0 * t)
    t_mod = t * (1 + wow + flutter)

    # Detune edilmiş harmonikler
    partials = [
        (frequency, 0.55, 0.0),
        (frequency * 2.01, 0.28, -0.3),
        (frequency * 3.02, 0.12, -0.6),
        (frequency * 5.0, 0.05, -0.9),  # bell partial
    ]
    base = np.zeros_like(t_mod)
    for freq, amp, decay_shape in partials:
        env_decay = np.exp(-t * (1.2 - decay_shape))
        base += amp * np.sin(2 * np.pi * freq * t_mod) * env_decay

    # Hafif AM dokusu
    am = 0.02 * np.sin(2 * np.pi * random.uniform(0.5, 1.5) * t)
    base *= (1 + am)

    # Dinamik LPF: yoğunluğa göre 2-6 kHz
    cutoff = 2000 + intensity * 4000
    base = low_pass_filter(base, cutoff, sample_rate)

    # Hammer transient (kısa HP noise)
    attack_len = int(0.02 * sample_rate)
    hammer = high_pass_filter(white_noise_gaussian(attack_len), 2000, sample_rate) * 0.15
    base[:attack_len] += np.hanning(attack_len) * hammer

    env = adsr_envelope(len(base), 0.01, 0.35, 0.6, 1.6, sample_rate)
    note = base * env * intensity
    note = soft_saturation(note, 0.25)

    # Key-off / release noise (hafif bandpass)
    release_len = int(0.3 * sample_rate)
    release_noise = band_pass_filter(white_noise_gaussian(release_len), 500, 4000, sample_rate) * 0.05
    tail = np.zeros(len(note) + release_len)
    tail[:len(note)] += note
    tail[-release_len:] += np.hanning(release_len) * release_noise

    # Pseudo reverb delay
    delay = int(0.25 * sample_rate)
    out = np.zeros(len(tail) + delay)
    out[:len(tail)] += tail
    out[delay:] += tail * 0.35
    return out


def generate_piano_pattern(scale: List[float], bpm_like: float, length_seconds: int, style: str, sample_rate: int) -> np.ndarray:
    """Rastgele ambient piyano paterni üretir."""
    beat = 60.0 / bpm_like
    times = np.arange(0, length_seconds, beat * random.uniform(0.8, 1.2))
    acc = np.zeros(length_seconds * sample_rate + sample_rate * 6)

    last_freq = None
    for t in times:
        # Aynı notaya saplanmamak için komşu dışı seçim
        candidates = scale.copy()
        if last_freq in candidates and len(candidates) > 1:
            candidates = [c for c in candidates if c != last_freq]
        freq = random.choice(candidates)

        # Oktav varyasyonu
        if random.random() < 0.35:
            freq *= random.choice([0.5, 2.0])

        # Küçük pitch bend/humanize
        freq *= 1 + random.uniform(-0.01, 0.01)

        intensity = {"light": 0.25, "medium": 0.4, "deep": 0.55}.get(style, 0.3)
        note = generate_piano_note(freq, intensity, sample_rate)
        pos = int(t * sample_rate)
        end = pos + len(note)
        if end > len(acc):
            break
        acc[pos:end] += note
        last_freq = freq

    # Tonu yumuşatmak için LPF ve hafif shimmer ekle
    acc = low_pass_filter(acc, 5200, sample_rate)
    if style == "deep":
        acc += low_pass_filter(white_noise_gaussian(len(acc)), 1500, sample_rate) * 0.01
    return normalize_audio(acc, -8)


def ambient_guitar_pad(freq: float, duration: int, sample_rate: int) -> np.ndarray:
    """Basit gitar-pad benzeri doku üretir."""
    tri = triangle_wave(freq, duration, sample_rate)
    noise = low_pass_filter(white_noise_gaussian(len(tri)), 2000, sample_rate) * 0.2
    env = adsr_envelope(len(tri), 0.5, 2.0, 0.6, 2.0, sample_rate)
    pad = (tri + noise) * env
    return soft_saturation(pad, 0.4)


# ───────────────────────────────────
# SECTION 7 — BRAINWAVE / HEALING / CHAKRA ENGINE
# ───────────────────────────────────

CHAKRA_FREQS = {
    "root": 194.18,
    "sacral": 210.42,
    "solar": 126.22,
    "heart": 136.10,
    "throat": 141.27,
    "third_eye": 221.23,
    "crown": 172.06,
}
SOLFEGGIO = [174, 285, 396, 417, 432, 528, 639, 741, 852, 963]


def generate_binaural_tone(base_freq: float, beat_freq: float, duration: int, level_db=-18.0) -> np.ndarray:
    """Binaural beat üretir."""
    left = sine_wave(base_freq - beat_freq / 2, duration, SAMPLE_RATE)
    right = sine_wave(base_freq + beat_freq / 2, duration, SAMPLE_RATE)
    stereo = np.vstack([left, right])
    stereo *= db_to_linear(level_db)
    return stereo


def generate_isochronic_tone(base_freq: float, pulse_rate: float, duration: int, level_db=-18.0) -> np.ndarray:
    """Isochronic ton üretir."""
    tone = sine_wave(base_freq, duration, SAMPLE_RATE)
    lfo = (lfo_sine(len(tone), pulse_rate, SAMPLE_RATE) > 0).astype(float)
    tone *= lfo
    stereo = np.vstack([tone, tone]) * db_to_linear(level_db)
    return stereo


def generate_om_like_tone(duration: int, level_db=-18.0) -> np.ndarray:
    """OM benzeri ton üretir."""
    base = sine_wave(136.1, duration, SAMPLE_RATE)
    formant = low_pass_filter(base, 500, SAMPLE_RATE) + low_pass_filter(base, 150, SAMPLE_RATE) * 0.6
    env = adsr_envelope(len(formant), 1.5, 2.0, 0.8, 2.5, SAMPLE_RATE)
    tone = soft_saturation(formant * env, 0.5)
    stereo = np.vstack([tone, tone]) * db_to_linear(level_db)
    return stereo


def apply_breath_pattern(audio: np.ndarray, pattern: str) -> np.ndarray:
    """Nefes ritmine göre modülasyon uygular."""
    if pattern not in ["4-4-6", "4-7-8"]:
        return audio
    inhale, hold, exhale = (4, 4, 6) if pattern == "4-4-6" else (4, 7, 8)
    cycle = inhale + hold + exhale
    t = np.arange(audio.shape[1]) / SAMPLE_RATE
    pos = np.mod(t, cycle)
    env = np.where(pos < inhale, pos / inhale, np.where(pos < inhale + hold, 1.0, 1 - ((pos - inhale - hold) / exhale)))
    return audio * env


def generate_brainwave_layer(cfg: BrainwaveLayerConfig, duration: int) -> np.ndarray:
    """Beyin dalgası katmanı üretir."""
    if cfg.mode == "binaural":
        bw = generate_binaural_tone(cfg.base_freq, cfg.beat_freq, duration, cfg.level)
    elif cfg.mode == "isochronic":
        bw = generate_isochronic_tone(cfg.base_freq, cfg.pulse_rate, duration, cfg.level)
    elif cfg.mode == "solfeggio":
        bw = np.vstack([sine_wave(cfg.base_freq, duration, SAMPLE_RATE)] * 2) * db_to_linear(cfg.level)
    elif cfg.mode == "chakra":
        freq = CHAKRA_FREQS.get(cfg.chakra, 136.1)
        bw = np.vstack([sine_wave(freq, duration, SAMPLE_RATE)] * 2) * db_to_linear(cfg.level)
    elif cfg.mode == "om":
        bw = generate_om_like_tone(duration, cfg.level)
    else:
        bw = np.zeros((2, duration * SAMPLE_RATE))
    if cfg.breath_pattern:
        bw = apply_breath_pattern(bw, cfg.breath_pattern)
    # Tekdüzeliği kırmak için mikro rasgelelik
    bw = sprinkle_brainwave_randomness(bw, SAMPLE_RATE)
    # Beat drift: çok yavaş ±0.1 Hz kayma
    t = np.arange(bw.shape[1]) / SAMPLE_RATE
    drift_lfo = np.sin(2 * np.pi * 0.01 * t) * 0.05
    bw *= 1 + drift_lfo
    return bw


def pulsing_frequency_mode(freq: float, duration: int, rate: float, level_db: float) -> np.ndarray:
    """Vurgulu puls modu üretir."""
    tone = sine_wave(freq, duration, SAMPLE_RATE)
    lfo = (lfo_sine(len(tone), rate, SAMPLE_RATE) * 0.5 + 0.5)
    tone *= lfo
    stereo = np.vstack([tone, tone]) * db_to_linear(level_db)
    return stereo


def sprinkle_brainwave_randomness(stereo: np.ndarray, sample_rate: int) -> np.ndarray:
    """Beyin dalgasına düşük derinlikte rasgelelik ve hareket ekler."""
    length = stereo.shape[1]
    t = np.arange(length) / sample_rate
    # Yavaş genlik LFO
    lfo_a = 0.03 * np.sin(2 * np.pi * random.uniform(0.02, 0.08) * t) + 0.97
    lfo_b = 0.03 * np.sin(2 * np.pi * random.uniform(0.025, 0.09) * t + 1.3) + 0.97
    # Mikro tremolo
    trem = 0.02 * np.sin(2 * np.pi * random.uniform(0.6, 1.2) * t) + 1.0
    stereo[0] *= lfo_a * trem
    stereo[1] *= lfo_b * trem
    # Hafif stereo wiggle
    wiggle = 0.01 * np.sin(2 * np.pi * random.uniform(0.1, 0.3) * t)
    stereo[0] = apply_drift(stereo[0], random.uniform(0.01, 0.05)) + wiggle
    stereo[1] = apply_drift(stereo[1], random.uniform(0.01, 0.05)) - wiggle
    return stereo


# ───────────────────────────────────
# SECTION 8 — SLEEP MIX ENGINE
# ───────────────────────────────────

def generate_sleep_mix(mix_config: SleepMixConfig, duration_seconds: int, sample_rate: int) -> np.ndarray:
    """Uyku miksini üretir."""
    length = duration_seconds
    layers = []

    if mix_config.ambience_type.startswith("rain"):
        amb_fn = rain_medium
    elif mix_config.ambience_type.startswith("ocean"):
        amb_fn = ocean_soft
    elif mix_config.ambience_type.startswith("wind"):
        amb_fn = soft_wind
    elif mix_config.ambience_type.startswith("fire"):
        amb_fn = fireplace_soft
    elif mix_config.ambience_type.startswith("forest"):
        amb_fn = forest_night
    elif mix_config.ambience_type.startswith("urban"):
        amb_fn = fan_noise
    else:
        amb_fn = ocean_soft
    amb = amb_fn(length)
    amb *= db_to_linear(-16)
    layers.append(amb)

    noise = _make_noise_bed(length * sample_rate, mix_config.noise_type)
    noise = apply_warm_coloration(noise)
    noise = np.vstack([noise, noise]) * db_to_linear(-20)
    layers.append(noise)

    if mix_config.pad:
        pad_cfg = ToneLayerConfig(type="pad", frequency=220.0, level=-24.0, pad=True, style="deep")
        pad = generate_ambient_pad(pad_cfg, length, sample_rate)
        layers.append(pan(pad, 0.0))

    if mix_config.piano:
        scale = [261.63, 293.66, 329.63, 392.0, 440.0, 523.25]
        bpm_like = 55 if mix_config.piano_intensity == "light" else 50 if mix_config.piano_intensity == "medium" else 45
        piano = generate_piano_pattern(scale, bpm_like, length, mix_config.piano_intensity, sample_rate)
        layers.append(pan(piano, 0.0))

    if mix_config.brainwave:
        bw_type = mix_config.brainwave
        beat = {"delta": 2.0, "theta": 6.0, "alpha": 10.0}.get(bw_type, 4.0)
        base = 100.0 if bw_type != "alpha" else 200.0
        bw_cfg = BrainwaveLayerConfig(mode="binaural", base_freq=base, beat_freq=beat, level=-28.0)
        bw = generate_brainwave_layer(bw_cfg, length)
        layers.append(bw)

    # Katmanları uzunluklarına göre hizalayarak topla
    mix = mix_layers(layers)
    mix = multiband_shape(*split_into_bands(mix.mean(axis=0), sample_rate), 1.05, 1.0, 0.95)
    mix = np.vstack([mix, mix])
    mix = soft_saturation(mix, 0.3)
    mix = micro_room_reverb(mix, sample_rate, mix=0.12)
    mix = natural_dither(mix, 2e-4)
    mix = normalize_audio(mix, mix_config.volume_db)
    mix = fade_in_out(mix, sample_rate, 15, 20)
    return mix


# ───────────────────────────────────
# SECTION 9 — PRESET CATALOG + GENERATORS
# ───────────────────────────────────

def simple_preset(pid, name, desc, tags, target, base_duration, ambience=None, noise=None, brain=None, sleep=False):
    """Kısa preset oluşturur."""
    noise_layer = [NoiseLayerConfig(type=noise or "pink", level=-20.0)]
    amb_layer = [AmbienceLayerConfig(type=ambience or "rain_medium", level=-18.0)]
    brain_layers = []
    if brain:
        brain_layers.append(BrainwaveLayerConfig(mode=brain[0], base_freq=brain[1], beat_freq=brain[2], level=-24.0))
    sleepmix = SleepMixConfig() if sleep else None
    return PresetConfig(
        id=pid,
        name=name,
        description=desc,
        tags=tags,
        target_use=target,
        base_duration=base_duration,
        noise_layers=noise_layer,
        ambience_layers=amb_layer,
        brainwave_layers=brain_layers,
        sleepmix=sleepmix,
    )

BASE_PRESETS: List[PresetConfig] = []

# Noise focus
noise_entries = [
    ("white_focus", "White Focus", "Crisp beyaz gürültü ile odak", ["white", "focus"], "focus"),
    ("pink_focus", "Pink Focus", "Dengeli pink gürültü", ["pink", "focus"], "focus"),
    ("brown_sleep", "Brown Sleep", "Sıcak brown gürültü uyku", ["brown", "sleep"], "sleep"),
    ("blue_focus", "Blue Focus", "Hafif parlak mavi gürültü", ["blue"], "focus"),
    ("violet_focus", "Violet Focus", "Yüksek frekanslı odak", ["violet"], "focus"),
    ("fan_like", "Fan Like", "Vantilatör benzeri uğultu", ["fan", "ambient"], "ambient"),
    ("ac_like", "AC Like", "Klima uğultusu", ["ac", "ambient"], "ambient"),
]
for pid, name, desc, tags, target in noise_entries:
    BASE_PRESETS.append(simple_preset(pid, name, desc, tags, target, "3h", ambience="fan_noise", noise=pid.split("_")[0]))

# Rain sleep
rain_entries = [
    ("rain_light_8h", "Light Rain 8H", "Yumuşak yağmur, uzun uyku", ["rain", "sleep"], "sleep", "8h", "rain_light"),
    ("rain_window_sleep", "Window Rain", "Camda damlayan yağmur", ["rain", "window"], "sleep", "8h", "rain_window"),
    ("tent_rain_cozy", "Tent Rain Cozy", "Çadır üstünde yağmur", ["rain", "tent"], "sleep", "8h", "rain_tent"),
    ("rainforest_rain", "Rainforest Rain", "Orman yağmuru detaylı", ["rain", "forest"], "sleep", "8h", "rainforest_rain"),
    ("rain_medium_focus", "Rain Focus", "Orta yağmur odak", ["rain", "focus"], "focus", "3h", "rain_medium"),
    ("rain_heavy_wall", "Heavy Rain Wall", "Gür güçlü yağmur duvarı", ["rain", "heavy"], "sleep", "8h", "rain_heavy"),
]
for pid, name, desc, tags, target, dur, amb in rain_entries:
    BASE_PRESETS.append(simple_preset(pid, name, desc, tags, target, dur, ambience=amb, noise="pink"))

# Ocean / water
water_entries = [
    ("ocean_soft_8h", "Ocean Soft", "Yumuşak dalga uğultusu", ["ocean", "sleep"], "sleep", "8h", "ocean_soft"),
    ("ocean_deep_relax", "Ocean Deep", "Derin dalga ve sub", ["ocean", "deep"], "sleep", "8h", "ocean_deep"),
    ("river_stream_focus", "River Stream", "Akan dere odak", ["river", "focus"], "focus", "3h", "river_stream"),
    ("waterfall_soft", "Waterfall Soft", "Uzak şelale", ["waterfall", "sleep"], "sleep", "8h", "waterfall_soft"),
    ("lake_waves_calm", "Lake Waves", "Hafif göl dalgaları", ["lake", "calm"], "sleep", "8h", "lake_waves"),
]
for pid, name, desc, tags, target, dur, amb in water_entries:
    BASE_PRESETS.append(simple_preset(pid, name, desc, tags, target, dur, ambience=amb, noise="pink"))

# Fire
fire_entries = [
    ("fireplace_sleep", "Fireplace Sleep", "Soba çıtırtısı ve sıcaklık", ["fireplace", "sleep"], "sleep", "8h", "fireplace_soft"),
    ("campfire_night", "Campfire Night", "Açık hava kamp ateşi", ["campfire", "night"], "sleep", "8h", "campfire_crackle"),
]
for pid, name, desc, tags, target, dur, amb in fire_entries:
    BASE_PRESETS.append(simple_preset(pid, name, desc, tags, target, dur, ambience=amb, noise="brown"))

# Wind & nature
wind_entries = [
    ("mountain_wind_sleep", "Mountain Wind", "Dağ esintisi", ["wind", "mountain"], "sleep", "8h", "mountain_wind"),
    ("soft_wind_focus", "Soft Wind Focus", "Hafif rüzgar", ["wind", "focus"], "focus", "3h", "soft_wind"),
    ("storm_wind_sleep", "Storm Wind", "Derin fırtına", ["wind", "storm"], "sleep", "8h", "storm_wind"),
    ("desert_wind", "Desert Wind", "Kuru çöl rüzgarı", ["wind", "desert"], "ambient", "3h", "desert_wind"),
    ("forest_night_insects", "Forest Night", "Böcekli gece ormanı", ["forest", "night"], "sleep", "8h", "forest_night"),
    ("forest_morning_birds", "Forest Morning", "Sabah kuş sesli orman", ["forest", "morning"], "ambient", "3h", "forest_morning"),
    ("jungle_breath", "Jungle Air", "Yoğun tropik hava", ["jungle"], "sleep", "8h", "jungle_air"),
    ("cold_mountain_air", "Cold Mountain", "Soğuk dağ sessizliği", ["mountain"], "sleep", "8h", "cold_mountain"),
]
for pid, name, desc, tags, target, dur, amb in wind_entries:
    BASE_PRESETS.append(simple_preset(pid, name, desc, tags, target, dur, ambience=amb, noise="pink"))

# Urban
urban_entries = [
    ("cafe_like_focus_no_voices", "Cafe Focus", "Ses yok, ortam uğultusu", ["urban", "cafe"], "focus", "3h", "fan_noise"),
    ("city_low_rumble_focus", "City Rumble", "Uzak şehir uğultusu", ["city", "rumble"], "focus", "3h", "city_low_rumble"),
    ("train_rumble_focus", "Train Rumble", "Ray uğultusu", ["train"], "focus", "3h", "train_rumble"),
    ("server_room_focus", "Server Room", "Teknolojik uğultu", ["server"], "focus", "3h", "server_room"),
    ("air_conditioner_sleep", "AC Sleep", "Klima uğultusu uyku", ["ac", "sleep"], "sleep", "8h", "air_conditioner"),
]
for pid, name, desc, tags, target, dur, amb in urban_entries:
    BASE_PRESETS.append(simple_preset(pid, name, desc, tags, target, dur, ambience=amb, noise="white"))

# Space
space_entries = [
    ("space_station_sleep", "Space Station", "Uzay istasyonu hum", ["space"], "sleep", "8h", "space_station"),
    ("interstellar_drone_meditation", "Interstellar Drone", "Derin uzay drone", ["space", "drone"], "meditation", "3h", "interstellar_drone"),
    ("nebula_hum_focus", "Nebula Hum", "Nebula uğultusu", ["space", "focus"], "focus", "3h", "nebula_hum"),
]
for pid, name, desc, tags, target, dur, amb in space_entries:
    BASE_PRESETS.append(simple_preset(pid, name, desc, tags, target, dur, ambience=amb, noise="violet"))

# Meditation & brainwave
brain_entries = [
    ("theta_meditation", "Theta Meditation", "6Hz theta binaural", ["theta", "meditation"], "meditation", 100.0, 6.0),
    ("delta_deep_sleep", "Delta Deep Sleep", "2Hz delta uyku", ["delta", "sleep"], "sleep", 100.0, 2.0),
    ("alpha_focus", "Alpha Focus", "10Hz alpha", ["alpha", "focus"], "focus", 200.0, 10.0),
    ("gamma_space", "Gamma Space", "40Hz gamma dokunuşu", ["gamma", "space"], "meditation", 400.0, 40.0),
]
for pid, name, desc, tags, target, base, beat in brain_entries:
    cfg = PresetConfig(
        id=pid,
        name=name,
        description=desc,
        tags=tags,
        target_use=target,
        base_duration="2h",
        noise_layers=[NoiseLayerConfig(type="pink", level=-24.0)],
        ambience_layers=[AmbienceLayerConfig(type="space_station", level=-20.0)],
        brainwave_layers=[BrainwaveLayerConfig(mode="binaural", base_freq=base, beat_freq=beat, level=-22.0)],
    )
    BASE_PRESETS.append(cfg)

# Solfeggio & chakra
solf_entries = [
    ("528_healing_sleep", "528 Healing Sleep", "528Hz solfeggio", ["528", "healing"], "sleep", 528),
    ("432_relaxation", "432 Relax", "432Hz huzur", ["432", "relax"], "sleep", 432),
    ("963_crown_meditation", "963 Crown", "963Hz taç çakra", ["963", "chakra"], "meditation", 963),
    ("639_heart", "639 Heart", "Kalp açılımı", ["639", "heart"], "meditation", 639),
]
for pid, name, desc, tags, target, freq in solf_entries:
    cfg = PresetConfig(
        id=pid,
        name=name,
        description=desc,
        tags=tags,
        target_use=target,
        base_duration="2h",
        noise_layers=[NoiseLayerConfig(type="brown", level=-26.0)],
        ambience_layers=[AmbienceLayerConfig(type="ocean_soft", level=-22.0)],
        brainwave_layers=[BrainwaveLayerConfig(mode="solfeggio", base_freq=freq, level=-20.0)],
    )
    BASE_PRESETS.append(cfg)

# Hybrid sleep mixes
hybrid_entries = [
    ("rain_piano_432", "Rain Piano 432", "Yağmur + piyano + 432Hz", ["rain", "432", "piano"], "sleep", "8h", "rain_window", "pink", "solfeggio", 432),
    ("ocean_theta_pad", "Ocean Theta Pad", "Okyanus + theta + pad", ["ocean", "theta", "pad"], "sleep", "8h", "ocean_deep", "pink", "binaural", 100),
    ("brown_fireplace_pad", "Brown Fireplace Pad", "Brown gürültü + şömine + pad", ["brown", "fire"], "sleep", "8h", "fireplace_soft", "brown", None, None),
    ("space_gamma_drone", "Space Gamma Drone", "Uzay + gamma + drone", ["space", "gamma"], "meditation", "3h", "interstellar_drone", "violet", "binaural", 400),
]
for pid, name, desc, tags, target, dur, amb, noise, brainmode, basefreq in hybrid_entries:
    brain_layers = []
    if brainmode:
        beat = 8.0 if "theta" in tags else 40.0 if "gamma" in tags else 4.0
        brain_layers.append(BrainwaveLayerConfig(mode=brainmode, base_freq=basefreq, beat_freq=beat, level=-24.0))
    cfg = PresetConfig(
        id=pid,
        name=name,
        description=desc,
        tags=tags,
        target_use=target,
        base_duration=dur,
        noise_layers=[NoiseLayerConfig(type=noise, level=-22.0)],
        ambience_layers=[AmbienceLayerConfig(type=amb, level=-20.0)],
        tone_layers=[ToneLayerConfig(type="pad", frequency=220, pad=True)],
        brainwave_layers=brain_layers,
        sleepmix=SleepMixConfig(piano=True, piano_intensity="light", ambience_type=amb, noise_type=noise, brainwave="theta" if "theta" in tags else None),
    )
    BASE_PRESETS.append(cfg)

# Extra fillers to reach 100+
extra_names = [
    ("rain_soft_focus", "Rain Soft Focus", "Hafif yağmur odak", "rain_medium"),
    ("rain_brown_mix", "Rain Brown", "Brown tabanlı yağmur", "rain_light"),
    ("wind_space_hum", "Wind Space Hum", "Rüzgar + uzay uğultu", "soft_wind"),
    ("forest_breath", "Forest Breath", "Orman nefesli", "forest_morning"),
    ("urban_low_focus", "Urban Low", "Şehir düşük uğultu", "city_low_rumble"),
    ("ocean_air", "Ocean Air", "Okyanus ve esinti", "ocean_soft"),
    ("campfire_pad", "Campfire Pad", "Ateş + pad", "campfire_crackle"),
    ("mountain_pad", "Mountain Pad", "Dağ rüzgarı + pad", "mountain_wind"),
    ("space_pad_sleep", "Space Pad Sleep", "Uzay pad uyku", "space_station"),
    ("nebula_pad_focus", "Nebula Pad Focus", "Nebula pad odak", "nebula_hum"),
    ("train_sleep", "Train Sleep", "Tren uğultusu uyku", "train_rumble"),
    ("server_focus_clean", "Server Clean", "Düzenli server uğultu", "server_room"),
    ("desert_night", "Desert Night", "Çöl gece rüzgarı", "desert_wind"),
    ("lake_night", "Lake Night", "Göl gece dalgaları", "lake_waves"),
    ("river_pad", "River Pad", "Dere + pad", "river_stream"),
    ("storm_deep", "Storm Deep", "Derin fırtına sub", "storm_wind"),
    ("forest_zen", "Forest Zen", "Orman zen", "forest_night"),
    ("jungle_focus", "Jungle Focus", "Tropik odak", "jungle_air"),
    ("ac_focus", "AC Focus", "Klima odak", "air_conditioner"),
    ("fan_focus", "Fan Focus", "Vantilatör odak", "fan_noise"),
]
for pid, name, desc, amb in extra_names:
    BASE_PRESETS.append(simple_preset(pid, name, desc, ["extra"], "ambient", "3h", ambience=amb, noise="pink"))

def generate_sleep_mix_presets(count: int = 50) -> List[PresetConfig]:
    """Otomatik uyku presetleri üretir."""
    presets = []
    amb_choices = ["rain_window", "ocean_deep", "soft_wind", "fireplace_soft", "forest_night", "space_station"]
    noise_choices = ["brown", "pink", "white"]
    brain_choices = [None, "delta", "theta", "alpha"]
    for i in range(count):
        amb = random.choice(amb_choices)
        noi = random.choice(noise_choices)
        brain = random.choice(brain_choices)
        pid = f"auto_sleep_{i}"
        presets.append(PresetConfig(
            id=pid,
            name=f"Auto Sleep {i}",
            description=f"Auto uyku miks {amb} + {noi}",
            tags=["auto", "sleep"],
            target_use="sleep",
            base_duration=random.choice(DEFAULT_DURATIONS),
            noise_layers=[NoiseLayerConfig(type=noi)],
            ambience_layers=[AmbienceLayerConfig(type=amb)],
            brainwave_layers=[BrainwaveLayerConfig(mode="binaural", base_freq=100, beat_freq=2.0, level=-26.0)] if brain else [],
            sleepmix=SleepMixConfig(piano=bool(random.getrandbits(1)), ambience_type=amb, noise_type=noi, brainwave=brain),
        ))
    return presets


def generate_ambience_variants(base_id: str, ambience_types: List[str], target: str) -> List[PresetConfig]:
    """Ambiyans varyantları oluşturur."""
    presets = []
    for amb in ambience_types:
        pid = f"{base_id}_{amb}"
        presets.append(simple_preset(pid, f"{base_id} {amb}", f"{amb} tabanlı varyant", ["variant", amb], target, "3h", ambience=amb, noise="pink"))
    return presets


def generate_noise_variants(base_id: str, noises: List[str], target: str) -> List[PresetConfig]:
    """Gürültü varyantları oluşturur."""
    presets = []
    for n in noises:
        pid = f"{base_id}_{n}"
        presets.append(simple_preset(pid, f"{base_id} {n}", f"{n} gürültü varyantı", ["variant", n], target, "3h", ambience="fan_noise", noise=n))
    return presets


AUTO_PRESETS = generate_sleep_mix_presets(40)
VARIANT_PRESETS = generate_ambience_variants("rain_family", list(AMBIENCE_MAP.keys())[:10], "sleep")
NOISE_VARIANTS = generate_noise_variants("noise_family", ["white", "pink", "brown", "blue", "violet"], "focus")

ALL_PRESETS: List[PresetConfig] = BASE_PRESETS + AUTO_PRESETS + VARIANT_PRESETS + NOISE_VARIANTS

PRESET_MAP = {p.id: p for p in ALL_PRESETS}

# ───────────────────────────────────
# PRESET ETİKET STANDARTLAŞTIRMA
# ───────────────────────────────────

def standardize_preset_tags(preset: PresetConfig) -> PresetConfig:
    """Preset etiketlerini standart kümeyle birleştirir."""
    tags = set(preset.tags)
    tags.add(preset.target_use)
    # Ton tahmini
    tone_guess = None
    if any(n.type == "brown" for n in preset.noise_layers):
        tone_guess = "warm"
    if any(n.type == "violet" for n in preset.noise_layers):
        tone_guess = tone_guess or "bright"
    if any("deep" in a.type for a in preset.ambience_layers):
        tone_guess = tone_guess or "deep"
    if preset.sleepmix and preset.sleepmix.piano_intensity == "deep":
        tone_guess = tone_guess or "deep"
    if tone_guess:
        tags.add(tone_guess)
    # Ambiyans etiketi
    for a in preset.ambience_layers:
        for key in STANDARD_AMBIENCE_TAGS:
            if key in a.type:
                tags.add(key)
    preset.tags = list(dict.fromkeys(tags))
    return preset

# Standartlaştırılmış harita
PRESET_MAP = {pid: standardize_preset_tags(p) for pid, p in PRESET_MAP.items()}

# ───────────────────────────────────
# SECTION 10 — RENDER ENGINE
# ───────────────────────────────────

def mix_layers(layers: List[np.ndarray]) -> np.ndarray:
    """Katmanları toplar."""
    max_len = max(layer.shape[1] for layer in layers)
    mix = np.zeros((2, max_len))
    for layer in layers:
        if layer.shape[1] < max_len:
            pad = np.zeros((2, max_len - layer.shape[1]))
            layer = np.hstack([layer, pad])
        mix += layer
    return mix


def render_preset(
    preset_id: str,
    duration_str: str,
    variant: int = 0,
    sample_rate: int = SAMPLE_RATE,
    write_file: bool = True,
    return_audio: bool = False,
) -> Path | tuple:
    """Preset'i render eder; isteğe göre dosyaya yazar veya bellekte döndürür."""
    if preset_id not in PRESET_MAP:
        raise ValueError("Preset bulunamadı")
    preset = PRESET_MAP[preset_id]
    duration = parse_duration(duration_str)
    random.seed(variant)
    np.random.seed(variant)

    layers = []

    # Gürültü katmanları
    for nl in preset.noise_layers:
        noise = _make_noise_bed(duration * sample_rate, nl.type)
        if nl.lp_hz:
            noise = low_pass_filter(noise, nl.lp_hz, sample_rate)
        if nl.hp_hz:
            noise = high_pass_filter(noise, nl.hp_hz, sample_rate)
        if nl.lfo_rate > 0:
            lfo = lfo_sine(len(noise), nl.lfo_rate, sample_rate)
            noise = apply_lfo_volume(noise, lfo * 0.5 + 0.5)
        if nl.drift:
            noise = apply_drift(noise, nl.drift)
        noise = apply_warm_coloration(noise)
        stereo = pan(noise, random.uniform(-0.2, 0.2))
        stereo *= db_to_linear(nl.level)
        layers.append(stereo)

    # Ambiyans katmanları
    for al in preset.ambience_layers:
        fn = AMBIENCE_MAP.get(al.type, rain_medium)
        amb = fn(duration)
        amb = stereo_widen(amb, al.width)
        amb *= db_to_linear(al.level)
        layers.append(amb)

    # Ton/pad katmanları
    for tl in preset.tone_layers:
        if tl.pad:
            pad = generate_ambient_pad(tl, duration, sample_rate)
            stereo = pan(pad, 0.0)
        else:
            wave = sine_wave if tl.wave == "sine" else triangle_wave
            tone = wave(tl.frequency, duration, sample_rate)
            stereo = pan(tone, 0.0)
        stereo *= db_to_linear(tl.level)
        layers.append(stereo)

    # Beyin dalgaları
    for bw in preset.brainwave_layers:
        layers.append(generate_brainwave_layer(bw, duration))

    # Sleep mix override
    if preset.sleepmix:
        layers.append(generate_sleep_mix(preset.sleepmix, duration, sample_rate))

    if not layers:
        raise ValueError("Katman bulunamadı")

    mix = mix_layers(layers)
    mix = multiband_shape(*split_into_bands(mix.mean(axis=0), sample_rate), 1.05, 1.0, 0.95)
    mix = np.vstack([mix, mix])
    mix = soft_saturation(mix, 0.25)
    mix = normalize_audio(mix, MASTER_RENDER.master_gain_db)
    mix = fade_in_out(mix, sample_rate, MASTER_RENDER.fade_in, MASTER_RENDER.fade_out)
    mix = apply_jitter(mix, 0.0005)

    if return_audio:
        return mix, sample_rate

    if not write_file:
        return None

    filename = generate_filename(preset_id, duration_str, variant)
    out_path = OUTPUT_DIR / filename
    write_wav(out_path, mix, sample_rate)
    logger.info("Render tamam: %s", out_path)
    return out_path


def write_wav(path: Path, audio: np.ndarray, sample_rate: int):
    """Stereo WAV yazar."""
    audio16 = np.int16(np.clip(audio, -1, 1) * 32767)
    import wave
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio16.T.tobytes())

# ───────────────────────────────────
# SECTION 11 — STREAMLIT UI
# ───────────────────────────────────

def run_streamlit_app():
    """Streamlit arayüzünü çalıştırır."""
    st.set_page_config(page_title="Ambience & Sleep Mixer", layout="wide", page_icon="🎧", initial_sidebar_state="expanded")
    st.title("🎧 Ambience & Sleep Mixer")
    st.caption("Tamamen koddan sentezlenen uzun form ambient miksler. Türkçe arayüz, YouTube hazır dosyalar.")

    def resolve_amb_option(val: str) -> str:
        """Ambiyans varsayılanını temel seçeneklere indirger."""
        opts = ["rain", "ocean", "wind", "fire", "forest", "urban", "space"]
        if val in opts:
            return val
        for o in opts:
            if val and val.startswith(o):
                return o
        return "rain"

    def safe_index(lst, value, fallback=0):
        """Listede yoksa emniyetli index döndürür."""
        return lst.index(value) if value in lst else fallback

    # Hafif/Lite arayüz: daha az bileşen, daha hızlı etkileşim
    lite_mode = st.sidebar.checkbox("Hızlı/Lite arayüz", value=True, help="Daha az bileşenle hızlı kullanım.")
    if lite_mode:
        @st.cache_data
        def cached_ids():
            return sorted(PRESET_MAP.keys())

        # Basit filtre
        cat_filter = st.sidebar.selectbox("Kullanım", ["Tümü"] + sorted(set(p.target_use for p in PRESET_MAP.values())))
        ids = cached_ids()
        if cat_filter != "Tümü":
            ids = [i for i in ids if PRESET_MAP[i].target_use == cat_filter]
        pid = st.sidebar.selectbox("Preset", ids, format_func=lambda x: PRESET_MAP[x].name)
        preset = PRESET_MAP[pid]

        dur_map = {"15s Test": "15s", "10m": "10m", "1h": "1h", "3h": "3h", "8h": "8h"}
        d_lbl = st.sidebar.select_slider("Süre", list(dur_map.keys()))
        seed = st.sidebar.number_input("Seed", min_value=0, max_value=9999, value=0, step=1)
        fast_sr = st.sidebar.checkbox("Hızlı render (22kHz)", value=True)
        btn = st.sidebar.button("Render", type="primary")

        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader(preset.name)
            st.info(preset.description)
            st.write("Etiketler:", ", ".join(preset.tags))
        with c2:
            st.subheader("Özet")
            st.write(f"Kullanım: **{preset.target_use}**")
            st.write(f"Önerilen süre: **{preset.base_duration}**")
            tmpl = TEMPLATE_DEFAULTS.get(preset.target_use, {})
            if tmpl:
                st.write(f"Şablon: ton={tmpl.get('tone')} | ambiyans={tmpl.get('ambience')} | gürültü={tmpl.get('noise')}")

        if btn:
            bar = st.progress(0)
            dur_str = dur_map[d_lbl]
            sr = 22_050 if fast_sr else SAMPLE_RATE
            with st.spinner("Render ediliyor..."):
                out = render_preset(pid, dur_str, seed, sr)
            bar.progress(100)
            st.success(f"Tamamlandı: {out}")
            with open(out, "rb") as f:
                st.audio(f.read(), format="audio/wav")
        return

    # Filtreler
    st.subheader("Preset filtresi")
    use_filter = st.selectbox("Kullanım", ["Tümü"] + sorted(set(p.target_use for p in PRESET_MAP.values())))
    tone_filter = st.selectbox("Ton etiketi", ["Tümü"] + sorted(STANDARD_TONE_TAGS))
    amb_filter = st.selectbox("Ambiyans etiketi", ["Tümü"] + sorted(STANDARD_AMBIENCE_TAGS))

    def preset_match(p: PresetConfig) -> bool:
        if use_filter != "Tümü" and p.target_use != use_filter:
            return False
        if tone_filter != "Tümü" and tone_filter not in p.tags:
            return False
        if amb_filter != "Tümü" and amb_filter not in p.tags:
            return False
        return True

    filtered_ids = [pid for pid, p in PRESET_MAP.items() if preset_match(p)]
    if not filtered_ids:
        st.warning("Filtreye uyan preset yok, filtreyi genişletin.")
        filtered_ids = list(PRESET_MAP.keys())

    col1, col2 = st.columns([1, 1])
    preset_ids = sorted(filtered_ids)
    with col1:
        preset_id = st.selectbox("Preset seç", preset_ids)
        preset = PRESET_MAP[preset_id]
        if st.button("Preset detayları"):
            st.write(f"**Ad:** {preset.name}")
            st.write(f"**Açıklama:** {preset.description}")
            st.write(f"**Etiketler:** {', '.join(preset.tags)}")
            st.write(f"**Hedef:** {preset.target_use}")
            st.write(f"**Önerilen süre:** {preset.base_duration}")
            tmpl = TEMPLATE_DEFAULTS.get(preset.target_use, {})
            if tmpl:
                st.info(f"Şablon: {preset.target_use} | Ton: {tmpl.get('tone')} | Ambiyans: {tmpl.get('ambience')} | Gürültü: {tmpl.get('noise')}")
        # Preset varsayılanlarını yakala
        default_noise = None
        default_amb = None
        default_brain = None
        default_piano_intensity = "light"
        if preset.sleepmix:
            default_noise = preset.sleepmix.noise_type
            default_amb = preset.sleepmix.ambience_type
            default_brain = preset.sleepmix.brainwave
            default_piano_intensity = preset.sleepmix.piano_intensity
        elif preset.noise_layers:
            default_noise = preset.noise_layers[0].type
        if not default_amb and preset.ambience_layers:
            default_amb = preset.ambience_layers[0].type
        if preset.brainwave_layers:
            bw = preset.brainwave_layers[0]
            default_brain = bw.mode
        st.caption(f"Varsayılanlar → Gürültü: {default_noise or '-'} | Ambiyans: {default_amb or '-'} | Frekans: {default_brain or '-'} | Piyano: {default_piano_intensity}")
        # Buton: preset değerlerini kontrol alanlarına uygula
        if st.button("Preset ayarlarını uygula"):
            st.session_state["noise_type"] = default_noise or "pink"
            st.session_state["amb_type"] = resolve_amb_option(default_amb or "rain")
            st.session_state["brain_type"] = default_brain or "delta"
            st.session_state["piano_intensity"] = default_piano_intensity
    with col2:
        st.subheader("Süre")
        duration_option = st.radio("Süre seç", DEFAULT_DURATIONS + ["Custom"])
        if duration_option == "Custom":
            minutes = st.number_input("Dakika", min_value=1, max_value=600, value=30)
            duration_str = f"{int(minutes)}m"
        else:
            duration_str = duration_option

    st.subheader("Uyku Miks Kontrolleri")
    piano_on = st.checkbox("Piyano aktif", value=True)
    piano_intensity = st.selectbox(
        "Piyano yoğunluğu",
        ["light", "medium", "deep"],
        index=safe_index(["light", "medium", "deep"], st.session_state.get("piano_intensity", "light")),
        key="piano_intensity",
    )
    brain_on = st.checkbox("Frekans katmanı", value=False)
    brain_type = st.selectbox(
        "Frekans modu",
        ["delta", "theta", "alpha", "432", "528", "963"],
        index=safe_index(["delta", "theta", "alpha", "432", "528", "963"], st.session_state.get("brain_type", "delta")),
        key="brain_type",
    )
    amb_on = st.checkbox("Ambiyans aktif", value=True)
    amb_option = resolve_amb_option(st.session_state.get("amb_type", "rain"))
    amb_type = st.selectbox(
        "Ambiyans tipi",
        ["rain", "ocean", "wind", "fire", "forest", "urban", "space"],
        index=safe_index(["rain", "ocean", "wind", "fire", "forest", "urban", "space"], amb_option),
        key="amb_type",
    )
    noise_type = st.selectbox(
        "Gürültü tipi",
        ["white", "pink", "brown", "blue", "violet"],
        index=safe_index(["white", "pink", "brown", "blue", "violet"], st.session_state.get("noise_type", "pink"), fallback=1),
        key="noise_type",
    )
    master_volume = st.slider("Master seviye (dB)", -24, -1, -10)
    stereo_width = st.slider("Stereo genişlik", 0.0, 1.0, 0.6)
    st.caption("Stereo genişliği arttırır, kulaklıkta daha geniş sahne verir.")

    st.subheader("Performans & Rastgeleleştirme")
    fast_mode = st.checkbox("Hızlı mod (önizleme için düşük SR)", value=True)
    variant = st.number_input("Seed / varyant", min_value=0, max_value=9999, value=0, step=1)
    if st.button("Seed rastgeleleştir"):
        variant = random.randint(0, 9999)
        # Yeni Streamlit sürümlerinde experimental_rerun kaldırıldı; varsa kullan, yoksa rerun.
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        else:
            st.rerun()

    st.subheader("Eylemler")
    status = st.empty()
    if st.button("30 sn Önizleme üret"):
        status.info("Önizleme üretiliyor...")
        mix_cfg = SleepMixConfig(
            piano=piano_on,
            piano_intensity=piano_intensity,
            ambience_type=f"{amb_type}_soft" if amb_type == "ocean" else f"{amb_type}_medium" if amb_type == "rain" else amb_type,
            noise_type=noise_type,
            brainwave=brain_type if brain_on else None,
            pad=True,
            width=stereo_width,
            volume_db=master_volume,
        )
        preview_sr = 22_050 if fast_mode else SAMPLE_RATE
        preview_len = 10 if fast_mode else 30
        preview = generate_sleep_mix(mix_cfg, preview_len, preview_sr)
        # Diske yazmadan oynat
        import io, wave
        buffer = io.BytesIO()
        audio16 = np.int16(np.clip(preview, -1, 1) * 32767)
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(preview_sr)
            wf.writeframes(audio16.T.tobytes())
        buffer.seek(0)
        status.success("Önizleme hazır (hafızadan)")
        st.audio(buffer.read(), format="audio/wav")
    if st.button("Tam miks üret"):
        status.info("Render başlatılıyor...")
        full_audio, sr = render_preset(preset_id, duration_str, variant, SAMPLE_RATE, write_file=False, return_audio=True)
        # Bellekten oynat
        import io, wave
        buffer = io.BytesIO()
        audio16 = np.int16(np.clip(full_audio, -1, 1) * 32767)
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio16.T.tobytes())
        buffer.seek(0)
        status.success("Tam miks hazır (bellekten)")
        st.audio(buffer.read(), format="audio/wav")
        # İndir butonu (disk yazmadan)
        st.download_button("WAV indir", data=buffer.getvalue(), file_name=f"{preset_id}_{duration_str}_v{variant}.wav", mime="audio/wav")

    st.subheader("YouTube Önerileri")
    title = generate_youtube_title(preset, duration_str)
    desc = generate_youtube_description(preset, duration_str)
    tags = generate_youtube_tags(preset)
    thumb = suggest_thumbnail_theme(preset)
    st.text_area("Başlık", value=title, height=60)
    st.text_area("Açıklama (EN)", value=desc, height=150)
    st.write("Etiketler:", ", ".join(tags))
    st.write("Thumbnail tema:", thumb)


# ───────────────────────────────────
# SECTION 12 — YOUTUBE AUTOMATION HELPERS
# ───────────────────────────────────

def generate_youtube_title(preset: PresetConfig, duration_str: str) -> str:
    """YouTube başlığı üretir."""
    return f"{duration_str.upper()} {preset.description} | {preset.name}"


def generate_youtube_description(preset: PresetConfig, duration_str: str) -> str:
    """YouTube açıklaması üretir (EN)."""
    text = (
        f"{duration_str} long-form ambient mix: {preset.description}. "
        "Perfect for sleep, study, meditation, and deep focus. "
        "Hand-crafted synthesis with warm noise beds, evolving ambience, and gentle mastering. "
        "Use headphones for best spatial effect. "
        "No ads inside the audio. Relax and enjoy."
    )
    return text[:600]


def generate_youtube_tags(preset: PresetConfig) -> List[str]:
    """YouTube etiketleri üretir."""
    base = ["sleep", "focus", "meditation", "asmr", "ambient", "rain", "ocean", "wind", "fire", "brown noise", "pink noise"]
    return list(dict.fromkeys(preset.tags + base))[:20]


def suggest_thumbnail_theme(preset: PresetConfig) -> Dict[str, str]:
    """Thumbnail renk/ikon önerir."""
    if "rain" in preset.tags:
        return {"main_color": "blue", "secondary_color": "teal", "icon": "rain"}
    if "ocean" in preset.tags:
        return {"main_color": "navy", "secondary_color": "aqua", "icon": "wave"}
    if "fire" in preset.tags:
        return {"main_color": "orange", "secondary_color": "red", "icon": "fire"}
    if "space" in preset.tags:
        return {"main_color": "purple", "secondary_color": "black", "icon": "star"}
    return {"main_color": "green", "secondary_color": "blue", "icon": "leaf"}


# ───────────────────────────────────
# SECTION 13 — CLI
# ───────────────────────────────────

def list_presets():
    """Preset listesini yazdırır."""
    for p in ALL_PRESETS:
        print(f"{p.id}: {p.name} [{p.target_use}] {p.base_duration}")


def show_preset_info(preset_id: str):
    """Preset bilgisi gösterir."""
    p = PRESET_MAP.get(preset_id)
    if not p:
        print("Preset bulunamadı")
        return
    print(f"ID: {p.id}")
    print(f"Ad: {p.name}")
    print(f"Açıklama: {p.description}")
    print(f"Etiketler: {', '.join(p.tags)}")
    print(f"Hedef: {p.target_use}")
    print(f"Süre: {p.base_duration}")


def cli():
    """CLI giriş noktası."""
    parser = argparse.ArgumentParser(description="Ambience & Sleep Mixer")
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("--info", type=str, help="Preset ID")
    parser.add_argument("--preset", type=str, help="Preset ID")
    parser.add_argument("--duration", type=str, default="1h", help="Süre örn: 8h, 2h30m")
    parser.add_argument("--variant", type=int, default=0, help="Seed/varyant")
    parser.add_argument("--batch-from", type=str, help="Virgülle ayrılmış preset ID listesi")
    args = parser.parse_args()

    if args.list_presets:
        list_presets()
        return
    if args.info:
        show_preset_info(args.info)
        return
    if args.batch_from:
        ids = args.batch_from.split(",")
        for pid in ids:
            pid = pid.strip()
            if pid:
                render_preset(pid, args.duration, args.variant)
        return
    if args.preset:
        render_preset(args.preset, args.duration, args.variant)
        return
    # No args -> run streamlit
    print("Streamlit arayüzü için: streamlit run app.py")


# ───────────────────────────────────
# SECTION 14 — MAIN ENTRY
# ───────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli()
    else:
        if st is None:
            print("Streamlit yüklü değil. CLI kullanın veya 'pip install streamlit'.")
        else:
            run_streamlit_app()
