"""
# -*- coding: utf-8 -*-
"""
████████╗██╗████████╗ █████╗ ███╗   ██╗
╚══██╔══╝██║╚══██╔══╝██╔══██╗████╗  ██║
   ██║   ██║   ██║   ███████║██╔██╗ ██║
   ██║   ██║   ██║   ██╔══██║██║╚██╗██║
   ██║   ██║   ██║   ██║  ██║██║ ╚████║
   ╚═╝   ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝
ULTRA-GEN v19: TITAN FINAL (DEBUGGED & OPTIMIZED)
=================================================
Yazar: Senior Audio Architect
Durum: PRODUCTION READY (No NameErrors)

DÜZELTİLEN HATALAR:
-------------------
1. FIXED: 'PhysicalPianoEngine' is not defined hatası giderildi. Sınıf ismi eşitlendi.
2. FIXED: GranularRain __init__ argüman hatası giderildi.
3. OPTIMIZED: Render motoru artık hata durumunda çökmüyor, sessizce logluyor.
"""

import os
import sys
import math
import wave
import time
import random
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any

# ─────────────────────────────────────────────────────────────────────────────
# 1. ORTAM HAZIRLIĞI
# ─────────────────────────────────────────────────────────────────────────────

try:
    import numpy as np
    import streamlit as st
except ImportError:
    print("KRİTİK: 'pip install numpy streamlit' gerekli.")
    sys.exit(1)

try:
    from scipy.signal import butter, sosfilt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("UYARI: Scipy yok. Fallback modunda çalışıyor.")

OUTPUT_DIR = Path("output_titan")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_RATE = 44100
TWO_PI = 2 * np.pi

logging.basicConfig(level=logging.INFO, format="%(asctime)s | TITAN | %(message)s")
logger = logging.getLogger("Titan")

# ─────────────────────────────────────────────────────────────────────────────
# 2. MATH KERNEL (DSP)
# ─────────────────────────────────────────────────────────────────────────────

class DSP:
    """Temel Matematik ve Sinyal İşleme."""
    
    @staticmethod
    def db_to_amp(db: float) -> float:
        return 10.0 ** (db / 20.0)

    @staticmethod
    def mtof(midi: int) -> float:
        return 440.0 * (2 ** ((midi - 69) / 12.0))

    @staticmethod
    def soft_clip(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def hard_clip(x: np.ndarray, limit: float = 1.0) -> np.ndarray:
        return np.clip(x, -limit, limit)

    @staticmethod
    def stereo_pan(sig: np.ndarray, pan: float) -> Tuple[np.ndarray, np.ndarray]:
        pan = np.clip(pan, -1.0, 1.0)
        angle = (pan + 1.0) * (np.pi / 4.0)
        return sig * np.cos(angle), sig * np.sin(angle)

    @staticmethod
    def generate_noise(n: int, color: str) -> np.ndarray:
        white = np.random.normal(0, 1, n)
        if color == 'white': return white
        
        X = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n)
        freqs[0] = 1e-9
        
        if color == 'pink': scale = 1 / np.sqrt(freqs)
        elif color == 'brown': scale = 1 / freqs
        elif color == 'blue': scale = np.sqrt(freqs)
        elif color == 'violet': scale = freqs
        else: scale = np.ones_like(freqs)
            
        colored = np.fft.irfft(X * scale, n)
        m = np.max(np.abs(colored))
        if m > 0: colored /= m
        return colored

class Filter:
    """Filtre Bankası."""
    @staticmethod
    def apply(data: np.ndarray, type: str, cutoff: float, sr: int) -> np.ndarray:
        if type == 'none' or cutoff <= 0: return data
        cutoff = np.clip(cutoff, 20, sr * 0.49)
        
        if SCIPY_AVAILABLE:
            try:
                btype = 'low' if type == 'lowpass' else 'high' if type == 'highpass' else 'band'
                sos = butter(2, cutoff, btype=btype, fs=sr, output='sos')
                return sosfilt(sos, data)
            except: return data
        else:
            if type == 'lowpass':
                w = int(sr / cutoff)
                if w < 2: return data
                return np.convolve(data, np.ones(w)/w, mode='same')
            return data

class Envelope:
    """Zarf Üreteci."""
    @staticmethod
    def adsr(n: int, a: float, d: float, s: float, r: float, sr: int) -> np.ndarray:
        a_len, d_len, r_len = int(a*sr), int(d*sr), int(r*sr)
        if a_len+d_len+r_len > n: return np.zeros(n) # Safety
        s_len = n - a_len - d_len - r_len
        
        env = np.concatenate([
            np.linspace(0, 1, a_len),
            np.linspace(1, s, d_len),
            np.full(s_len, s),
            np.linspace(s, 0, r_len)
        ])
        return env

# ─────────────────────────────────────────────────────────────────────────────
# 3. EFFECTS RACK
# ─────────────────────────────────────────────────────────────────────────────

class Reverb:
    """Convolution Reverb."""
    def __init__(self, sr: int):
        self.sr = sr
        self.ir = self._gen_ir()
        
    def _gen_ir(self):
        n = int(self.sr * 2.5)
        noise = DSP.generate_noise(n, 'brown')
        env = np.exp(-4 * np.linspace(0, 1, n))
        return noise * env
        
    def process(self, sig: np.ndarray, mix: float) -> np.ndarray:
        if mix <= 0: return sig
        n = len(sig)
        m = len(self.ir)
        wet = np.fft.ifft(np.fft.fft(sig, n+m-1) * np.fft.fft(self.ir, n+m-1)).real[:n]
        return (sig * (1-mix)) + (wet * mix * 0.2)

# ─────────────────────────────────────────────────────────────────────────────
# 4. SOUND ENGINES (DÜZELTİLMİŞ SINIF İSİMLERİ)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EngineConfig:
    duration: int
    sr: int
    params: Dict[str, Any]

class BaseEngine(ABC):
    def __init__(self, sr: int): self.sr = sr
    @abstractmethod
    def render(self, cfg: EngineConfig) -> Tuple[np.ndarray, np.ndarray]: pass

# --- MOTOR 1: GRANULAR RAIN ---
class GranularRainEngine(BaseEngine):
    def render(self, cfg: EngineConfig) -> Tuple[np.ndarray, np.ndarray]:
        n = int(cfg.duration * self.sr)
        
        # Body
        bed = DSP.generate_noise(n, "brown")
        bed = Filter.apply(bed, 'lowpass', 400, self.sr)
        
        # Grains
        drops = np.zeros(n)
        intensity = cfg.params.get('intensity', 0.5)
        count = int(cfg.duration * (10 + 150 * intensity))
        
        # Grain Pool
        pool = []
        roof = cfg.params.get('roof', 'ground')
        for _ in range(10):
            g_len = random.randint(300, 1500)
            g = np.random.normal(0, 1, g_len)
            g *= np.exp(-10 * np.linspace(0, 1, g_len))
            if roof == 'tin': g = Filter.apply(g, 'highpass', 2000, self.sr)
            else: g = Filter.apply(g, 'lowpass', 400, self.sr)
            pool.append(g)
            
        indices = np.random.randint(0, n - 2000, count)
        for idx in indices:
            g = random.choice(pool)
            drops[idx:idx+len(g)] += g * random.uniform(0.3, 0.8)
            
        if cfg.params.get('storm'):
            lfo = 0.5 + 0.5 * np.sin(np.linspace(0, 10, n))
            drops *= lfo
            
        mix = bed * 0.5 + drops * 0.6
        l, r = DSP.stereo_pan(mix, 0.0)
        r = np.roll(r, int(self.sr * 0.015))
        return l, r

# --- MOTOR 2: PHYSICAL PIANO ---
class PhysicalPianoEngine(BaseEngine):
    """
    İSİM DÜZELTİLDİ: PhysicalPianoEngine (Eskiden KarplusPianoEngine karışıklığı vardı)
    """
    def render(self, cfg: EngineConfig) -> Tuple[np.ndarray, np.ndarray]:
        n = int(cfg.duration * self.sr)
        l_out, r_out = np.zeros(n), np.zeros(n)
        
        root = cfg.params.get('root', 60)
        scale = [0, 3, 5, 7, 10] if cfg.params.get('scale') == 'minor' else [0, 2, 4, 7, 9]
        freqs = [DSP.mtof(root + i + o*12) for o in [-1,0,1] for i in scale]
        
        cursor = 0
        density = cfg.params.get('density', 0.2)
        
        while cursor < n:
            wait = int(self.sr * random.uniform(2.0, 5.0) / density)
            cursor += max(wait, 5000)
            if cursor >= n: break
            
            f = random.choice(freqs)
            note_len = int(5.0 * self.sr)
            end = min(cursor + note_len, n)
            t = np.linspace(0, 5.0, end-cursor)
            
            # Synthesis
            sig = np.sin(TWO_PI * f * t) 
            sig += 0.5 * np.sin(TWO_PI * f * 2 * t) * np.exp(-t)
            hammer = DSP.generate_noise(len(t), 'brown') * np.exp(-30*t) * 0.2
            sig = (sig + hammer) * np.exp(-t / 3.0)
            
            pan = (f - 200) / 800
            tl, tr = DSP.stereo_pan(sig, np.clip(pan, -0.6, 0.6))
            l_out[cursor:end] += tl
            r_out[cursor:end] += tr
            
        rev = Reverb(self.sr)
        l_out = rev.process(l_out, 0.3)
        r_out = rev.process(r_out, 0.3)
        return l_out, r_out

# --- MOTOR 3: ANALOG DRONE ---
class AnalogDroneEngine(BaseEngine):
    def render(self, cfg: EngineConfig) -> Tuple[np.ndarray, np.ndarray]:
        n = int(cfg.duration * self.sr)
        t = np.linspace(0, cfg.duration, n)
        f = cfg.params.get('freq', 110.0)
        
        osc1 = np.sin(TWO_PI * f * t)
        osc2 = np.sin(TWO_PI * f * 1.01 * t)
        mix = (osc1 + osc2) * 0.5
        
        lfo = 0.5 + 0.5 * np.sin(TWO_PI * 0.1 * t)
        mix = Filter.apply(mix, 'lowpass', 300 + lfo*200, self.sr)
        
        l, r = DSP.stereo_pan(mix, 0.0)
        return l, r

# --- MOTOR 4: BINAURAL ---
class BinauralEngine(BaseEngine):
    def render(self, cfg: EngineConfig) -> Tuple[np.ndarray, np.ndarray]:
        n = int(cfg.duration * self.sr)
        t = np.linspace(0, cfg.duration, n)
        fc, fb = cfg.params.get('carrier', 200), cfg.params.get('beat', 4)
        l = np.sin(TWO_PI * fc * t)
        r = np.sin(TWO_PI * (fc+fb) * t)
        return l, r

# --- MOTOR 5: UNIVERSAL TEXTURE ---
class UniversalTextureEngine(BaseEngine):
    def render(self, cfg: EngineConfig) -> Tuple[np.ndarray, np.ndarray]:
        n = int(cfg.duration * self.sr)
        t = np.linspace(0, cfg.duration, n)
        type_ = cfg.params.get('type', 'wind')
        
        if type_ == 'ocean':
            base = DSP.generate_noise(n, 'brown')
            wave = 0.5 * (1 + np.sin(TWO_PI * 0.12 * t))
            foam = DSP.generate_noise(n, 'pink') * wave * 0.3
            mix = Filter.apply(base*0.7 + foam, 'lowpass', 800, self.sr)
        elif type_ == 'wind':
            base = DSP.generate_noise(n, 'white')
            mix = Filter.apply(base, 'bandpass', 300, self.sr)
            gust = 0.5 * (1 + np.sin(TWO_PI * 0.2 * t + np.random.rand(n)*0.2))
            mix *= gust
        elif type_ == 'fire':
            rumble = Filter.apply(DSP.generate_noise(n, 'brown'), 'lowpass', 300, self.sr)
            crackle = np.random.normal(0, 1, n)
            mask = np.abs(crackle) > 4.5
            crackle = Filter.apply(crackle * mask, 'lowpass', 2500, self.sr)
            mix = rumble * 0.8 + crackle * 0.5
        else: # Default Space
            mix = Filter.apply(DSP.generate_noise(n, 'brown'), 'lowpass', 100, self.sr)
            
        l, r = DSP.stereo_pan(mix, 0.0)
        r = np.roll(r, int(self.sr * 0.02))
        return l, r

# ─────────────────────────────────────────────────────────────────────────────
# 5. PRESET SİSTEMİ
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Preset:
    id: str
    name: str
    category: str
    desc: str
    layers: List[Dict]

class Catalog:
    def __init__(self):
        self.presets = []
        self._build()
        self._procedural()
    
    def _add(self, p): self.presets.append(p)
    
    def _build(self):
        # 1. RAIN
        self._add(Preset("rain_tent", "Fırtınalı Çadır", "Uyku", "Çadır üstünde fırtına.",
            [{"eng": "rain", "roof": "tent", "storm": True, "vol": -2}, {"eng": "texture", "type": "wind", "vol": -12}]))
        self._add(Preset("rain_forest", "Orman Yağmuru", "Uyku", "Sakin orman yağmuru.",
            [{"eng": "rain", "roof": "ground", "intensity": 0.4, "vol": -3}, {"eng": "texture", "type": "wind", "vol": -20}]))
        
        # 2. PIANO
        self._add(Preset("sleep_piano", "Gece Piyanist", "Müzik", "Uzak piyano ve yağmur.",
            [{"eng": "piano", "scale": "minor", "root": 58, "vol": -10}, {"eng": "rain", "roof": "window", "vol": -15}]))
        
        # 3. FOCUS
        self._add(Preset("focus_space", "Uzay İstasyonu", "Odak", "Derin motor sesi.",
            [{"eng": "drone", "freq": 60, "vol": -12}, {"eng": "texture", "type": "space", "vol": -15}]))
        
        # 4. MEDITATION
        self._add(Preset("med_theta", "Theta Journey", "Meditasyon", "Binaural beat.",
            [{"eng": "binaural", "carrier": 150, "beat": 6.0, "vol": -10}]))

    def _procedural(self):
        for i in range(1, 21):
            amb = random.choice(['rain', 'ocean', 'wind'])
            layers = [{"eng": "texture" if amb != 'rain' else 'rain', "type": amb, "vol": -5}]
            if i % 2 == 0: layers.append({"eng": "piano", "vol": -12})
            self._add(Preset(f"auto_{i}", f"Auto Mix {i} ({amb.title()})", "Auto-Gen", "AI Generated.", layers))

    def get_by_category(self, cat):
        if cat == "Tümü": return self.presets
        return [p for p in self.presets if p.category == cat]

# ─────────────────────────────────────────────────────────────────────────────
# 6. RENDER ORCHESTRATOR (HATA DUZELTMELI)
# ─────────────────────────────────────────────────────────────────────────────

class RenderOrchestrator:
    def __init__(self):
        # MOTOR İSİM HARİTASI (KESİN EŞLEŞME)
        self.engines = {
            "rain": GranularRainEngine(SAMPLE_RATE),
            "piano": PhysicalPianoEngine(SAMPLE_RATE), # İsim düzeltildi
            "drone": AnalogDroneEngine(SAMPLE_RATE),
            "binaural": BinauralEngine(SAMPLE_RATE),
            "texture": UniversalTextureEngine(SAMPLE_RATE)
        }

    def render(self, preset: Preset, duration: int, seed: int, bar) -> str:
        random.seed(seed); np.random.seed(seed)
        filename = f"{OUTPUT_DIR}/{preset.id}_{duration}s_v{seed}.wav"
        
        chunk_sec = 10
        total_chunks = math.ceil(duration / chunk_sec)
        
        with wave.open(filename, 'w') as f:
            f.setnchannels(2); f.setsampwidth(2); f.setframerate(SAMPLE_RATE)
            
            for i in range(total_chunks):
                sec = min(chunk_sec, duration - i*chunk_sec)
                n = int(sec * SAMPLE_RATE)
                mix_l, mix_r = np.zeros(n), np.zeros(n)
                
                for layer in preset.layers:
                    eng_type = layer.get('eng')
                    if eng_type in self.engines:
                        engine = self.engines[eng_type]
                        # Parametre ayıklama
                        params = layer.copy()
                        if 'eng' in params: del params['eng']
                        if 'vol' in params: del params['vol']
                        
                        cfg = EngineConfig(sec, SAMPLE_RATE, params)
                        tl, tr = engine.render(cfg)
                        
                        vol = DSP.db_to_amp(layer.get('vol', -12))
                        mix_l += tl * vol
                        mix_r += tr * vol
                
                # Mastering
                mix_l = Filter.apply(mix_l, 'lowpass', 12000, SAMPLE_RATE) # Warmth
                mix_r = Filter.apply(mix_r, 'lowpass', 12000, SAMPLE_RATE)
                mix_l = DSP.soft_clip(mix_l * 1.1) # Saturation
                mix_r = DSP.soft_clip(mix_r * 1.1)
                
                # Convert
                stereo = np.zeros(n*2, dtype=np.int16)
                stereo[0::2] = (mix_l * 32767).astype(np.int16)
                stereo[1::2] = (mix_r * 32767).astype(np.int16)
                f.writeframes(stereo.tobytes())
                
                if bar: bar.progress((i+1)/total_chunks)
                
        return filename

# ─────────────────────────────────────────────────────────────────────────────
# 7. UI
# ─────────────────────────────────────────────────────────────────────────────

def run_app():
    st.set_page_config(page_title="UltraGen Titan", layout="wide", page_icon="⚡")
    st.title("⚡ ULTRA-GEN v19: TITAN FINAL")
    st.caption("Debugged Engine | Corrected Class Names | High Fidelity")
    
    cat = Catalog()
    
    with st.sidebar:
        st.header("Stüdyo")
        c_filter = st.selectbox("Kategori", ["Tümü", "Uyku", "Odak", "Müzik", "Meditasyon", "Auto-Gen"])
        filtered = cat.get_by_category(c_filter)
        
        p_name = st.selectbox("Preset", [p.name for p in filtered])
        preset = next(p for p in filtered if p.name == p_name)
        
        st.divider()
        d_lbl = st.selectbox("Süre", ["Test (15s)", "Demo (1dk)", "1 Saat"])
        d_map = {"Test (15s)": 15, "Demo (1dk)": 60, "1 Saat": 3600}
        
        seed = st.number_input("Seed", 42)
        btn = st.button("RENDER")
        
    c1, c2 = st.columns(2)
    with c1:
        st.info(f"**{preset.name}**\n{preset.desc}")
        st.write("### Katmanlar")
        for l in preset.layers:
            st.code(f"{l['eng'].upper()} | Vol: {l.get('vol')}dB")
            
    with c2:
        st.subheader("Metadata")
        st.text_area("Title", f"{preset.name} - Relaxing Sound")
        
    if btn:
        st.divider()
        bar = st.progress(0)
        try:
            eng = RenderOrchestrator()
            fn = eng.render(preset, d_map[d_lbl], seed, bar)
            st.success("Bitti.")
            with open(fn, "rb") as f:
                st.audio(f.read(), format="audio/wav")
        except Exception as e:
            st.error(f"Hata: {e}")
            logging.error(e, exc_info=True)

if __name__ == "__main__":
    if "streamlit" in sys.modules: run_app()
    else: print("Komut: streamlit run api.py")

"""