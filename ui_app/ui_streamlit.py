"""
ui_app/ui_streamlit.py

` streamlit run ui_app/ui_streamlit.py ` komutu ile çalıştırılır.

xxxDSP / xxxDSP Engine - Streamlit Web Arayüzü
====================================================
Bu modül, xxxDSP ses motorunu tarayıcı tabanlı bir arayüz üzerinden
kullanmak için geliştirilmiştir. Hızlı prototipleme ve görsel test için idealdir.
"""

import sys
import os
import io
import time
import numpy as np
import streamlit as st
from scipy.io import wavfile

# Proje kök dizinini path'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from core_dsp import dsp_render
    from preset_system import preset_library, preset_autogen
    from ui_app.preset_to_dsp_adapter import adapt_preset_to_layer_generators
except ImportError as e:
    st.error(f"Kritik Hata: Gerekli modüller yüklenemedi.\nDetay: {e}")
    st.stop()


def format_duration(duration_seconds: float) -> str:
    """
    Duration'ı saniyeden insan okunabilir formata çevirir.
    
    Args:
        duration_seconds: Süre (saniye)
    
    Returns:
        Formatlanmış string (örn: "1H", "30M", "10H")
    """
    minutes = int(duration_seconds / 60)
    
    if minutes >= 60:
        hours = int(duration_seconds / 3600)
        return f"{hours}H"
    else:
        return f"{minutes}M"


def convert_to_wav_bytes(audio_data: np.ndarray, sample_rate: int) -> bytes:
    """
    NumPy dizisini bellekte WAV formatına çevirir.
    
    Peak normalization uygular (0.99 peak) ve 16-bit WAV'a dönüştürür.
    
    Args:
        audio_data: Float audio array [-1.0, 1.0]
        sample_rate: Sample rate (Hz)
    
    Returns:
        WAV file bytes
    """
    # Peak normalization (clipping önleme için 0.99)
    peak = np.max(np.abs(audio_data))
    if peak > 0.0:
        audio_data = (audio_data / peak) * 0.99
    
    # Safety clip
    audio_data = np.clip(audio_data, -1.0, 1.0)
    
    # 16-bit conversion
    scaled = (audio_data * 32767).astype(np.int16)
    virtual_file = io.BytesIO()
    wavfile.write(virtual_file, sample_rate, scaled)
    return virtual_file.getvalue()


def main():
    st.set_page_config(
        page_title="xxxDSP V1",
        page_icon="🌊",
        layout="centered"
    )

    st.title("🌊 xxxDSP / xxxDSP Engine")
    st.markdown("Akademik DSP tabanlı, deterministik gürültü ve atmosfer üretici.")
    st.markdown("---")

    # 1. Kenar Çubuğu: Ayarlar
    st.sidebar.header("Yapılandırma")

    # Preset Version Toggle
    preset_version = st.sidebar.radio(
        "Preset Versiyonu",
        options=["V2 (102 ML-generated)", "V1 (14 handcrafted)", "V2.5 (Pure Binaural Beats)"],
        index=0
    )
    is_v2 = preset_version.startswith("V2") and not preset_version.startswith("V2.5")
    is_v1 = preset_version.startswith("V1")
    is_binaural_only = preset_version.startswith("V2.5")
    st.sidebar.markdown("---")

    # Duration Selection with Tabs (V2.5 hariç)
    if not is_binaural_only:
        st.sidebar.markdown("**Süre Seçimi**")
        
        # Initialize session state
        if 'duration_minutes' not in st.session_state:
            st.session_state.duration_minutes = 10
        if 'duration_hours' not in st.session_state:
            st.session_state.duration_hours = 1
        if 'duration_mode' not in st.session_state:
            st.session_state.duration_mode = 'minutes'  # 'minutes' or 'hours'
        
        tab_minute, tab_hour = st.sidebar.tabs(["⏱️ Dakika", "⏰ Saat"])
        
        with tab_minute:
            minutes = st.slider(
                "Dakika Seç:", 
                min_value=1, 
                max_value=60, 
                value=st.session_state.duration_minutes, 
                step=1,
                key="minutes_slider"
            )
            if minutes != st.session_state.duration_minutes:
                st.session_state.duration_minutes = minutes
                st.session_state.duration_mode = 'minutes'
        
        with tab_hour:
            hours = st.slider(
                "Saat Seç:", 
                min_value=1, 
                max_value=10, 
                value=st.session_state.duration_hours, 
                step=1,
                key="hours_slider"
            )
            if hours != st.session_state.duration_hours:
                st.session_state.duration_hours = hours
                st.session_state.duration_mode = 'hours'
        
        # Calculate duration based on active mode
        if st.session_state.duration_mode == 'minutes':
            duration = st.session_state.duration_minutes * 60.0
        else:
            duration = st.session_state.duration_hours * 3600.0
        
        st.sidebar.markdown(f"**Seçilen:** {int(duration/60)} dakika ({int(duration)} saniye)")
    else:
        # V2.5: Binaural-specific duration recommendations
        st.sidebar.markdown("**Süre Seçimi (Binaural Önerilen)**")
        duration_preset = st.sidebar.selectbox(
            "Öneri Seç:",
            options=[
                "3 dakika (Test)",
                "5 dakika (Alpha - Minimum)",
                "10 dakika (Theta - Optimal)",
                "20 dakika (Gamma - Minimum)",
                "30 dakika (Gamma - Optimal)",
                "60 dakika (Deep Session)"
            ],
            index=2  # Default: 10 dakika
        )
        
        duration_map = {
            "3 dakika (Test)": 180.0,
            "5 dakika (Alpha - Minimum)": 300.0,
            "10 dakika (Theta - Optimal)": 600.0,
            "20 dakika (Gamma - Minimum)": 1200.0,
            "30 dakika (Gamma - Optimal)": 1800.0,
            "60 dakika (Deep Session)": 3600.0
        }
        
        duration = duration_map[duration_preset]
        st.sidebar.markdown(f"**Seçilen:** {int(duration/60)} dakika")

    sr = st.sidebar.selectbox(
        "Örnekleme Hızı (Hz)",
        options=[44100, 48000],
        index=1  # Default: 48000
    )

    # Variant sadece V1'de göster
    use_variant = False
    variant_intensity = 0.15
    if is_v1:
        use_variant = st.sidebar.checkbox("Otomatik Varyasyon Üret", value=False)
        if use_variant:
            variant_intensity = st.sidebar.slider("Varyasyon Şiddeti", 0.0, 0.5, 0.15)
    
    st.sidebar.markdown("---")
    
    # Binaural Beats Section
    if is_binaural_only:
        # V2.5: Binaural always enabled, advanced controls
        st.sidebar.markdown("**🎧 Binaural Beats Parametreleri**")
        st.sidebar.info("⚠️ Kulaklık kullanımı ZORUNLU!")
        
        # Preset recommendations
        binaural_preset = st.sidebar.selectbox(
            "Preset Seç:",
            options=[
                "Theta (7 Hz) - Meditation",
                "Alpha (10 Hz) - Relaxation", 
                "Gamma (40 Hz) - Focus",
                "Delta (4 Hz) - Deep Sleep",
                "--- Solfeggio Frequencies ---",
                "432Hz Universe Harmony",
                "528Hz Love Frequency",
                "639Hz Connection",
                "741Hz Intuition",
                "852Hz Spiritual Order",
                "963Hz Third Eye",
                "Custom"
            ],
            index=0  # Default: Theta
        )
        
        if binaural_preset == "Theta (7 Hz) - Meditation":
            carrier_freq = 400.0
            beat_freq = 7.0
            binaural_amplitude = 0.5
            st.sidebar.success("✅ Priority #1: En çok araştırılmış")
            st.sidebar.caption("Önerilen süre: 10 dakika\nKullanım: Meditasyon, derin rahatlama")
        elif binaural_preset == "Alpha (10 Hz) - Relaxation":
            carrier_freq = 400.0
            beat_freq = 10.0
            binaural_amplitude = 0.5
            st.sidebar.success("✅ Priority #2: Kısa sürede etkili")
            st.sidebar.caption("Önerilen süre: 5 dakika\nKullanım: Rahatlamış uyanıklık, odaklanma")
        elif binaural_preset == "Gamma (40 Hz) - Focus":
            carrier_freq = 250.0
            beat_freq = 40.0
            binaural_amplitude = 0.5
            st.sidebar.warning("⚡ Experimental: Uzun süre gerekli")
            st.sidebar.caption("Önerilen süre: 20-30 dakika\nKullanım: Yoğun konsantrasyon")
        elif binaural_preset == "Delta (4 Hz) - Deep Sleep":
            carrier_freq = 400.0
            beat_freq = 4.0
            binaural_amplitude = 0.5
            st.sidebar.info("🌙 Experimental: Derin uyku")
            st.sidebar.caption("Önerilen süre: 5-10 dakika\nUyarı: <3 Hz kullanmayın!")
        elif binaural_preset == "--- Solfeggio Frequencies ---":
            # Separator - revert to Theta
            carrier_freq = 400.0
            beat_freq = 7.0
            binaural_amplitude = 0.5
            st.sidebar.warning("⚠️ Lütfen bir preset seçin")
        elif binaural_preset == "432Hz Universe Harmony":
            carrier_freq = 432.0
            beat_freq = 8.0
            binaural_amplitude = 0.5
            st.sidebar.info("🎵 Solfeggio: Universe frequency")
            st.sidebar.caption("Pseudo-science | YouTube: ⭐⭐⭐⭐⭐\nÖnerilen: 10 dakika | Alpha meditation")
        elif binaural_preset == "528Hz Love Frequency":
            carrier_freq = 528.0
            beat_freq = 4.0
            binaural_amplitude = 0.5
            st.sidebar.info("❤️ Solfeggio: Love & DNA repair")
            st.sidebar.caption("Pseudo-science | YouTube: ⭐⭐⭐⭐⭐ (En popüler!)\nÖnerilen: 10 dakika | Theta healing")
        elif binaural_preset == "639Hz Connection":
            carrier_freq = 639.0
            beat_freq = 10.0
            binaural_amplitude = 0.45
            st.sidebar.info("🤝 Solfeggio: Relationships")
            st.sidebar.caption("Pseudo-science | YouTube: ⭐⭐⭐⭐\nÖnerilen: 10 dakika | Alpha connection")
        elif binaural_preset == "741Hz Intuition":
            carrier_freq = 741.0
            beat_freq = 7.0
            binaural_amplitude = 0.45
            st.sidebar.info("🧠 Solfeggio: Awakening intuition")
            st.sidebar.caption("Pseudo-science | YouTube: ⭐⭐⭐⭐\nÖnerilen: 10 dakika | Theta creativity")
        elif binaural_preset == "852Hz Spiritual Order":
            carrier_freq = 852.0
            beat_freq = 8.0
            binaural_amplitude = 0.4
            st.sidebar.info("✨ Solfeggio: Spiritual awakening")
            st.sidebar.caption("Pseudo-science | YouTube: ⭐⭐⭐\nÖnerilen: 10 dakika | Alpha spiritual")
        elif binaural_preset == "963Hz Third Eye":
            carrier_freq = 963.0
            beat_freq = 10.0
            binaural_amplitude = 0.4
            st.sidebar.info("👁️ Solfeggio: Third eye activation")
            st.sidebar.caption("Pseudo-science | YouTube: ⭐⭐⭐⭐⭐\nÖnerilen: 10 dakika | Alpha awakening")
        else:  # Custom
            carrier_freq = st.sidebar.number_input(
                "Carrier Frequency (Hz)",
                min_value=100.0,
                max_value=500.0,
                value=400.0,
                step=10.0,
                help="Taşıyıcı frekans (100-500 Hz). Optimal: 400 Hz"
            )
            beat_freq = st.sidebar.number_input(
                "Beat Frequency (Hz)",
                min_value=3.0,
                max_value=50.0,
                value=10.0,
                step=0.5,
                help="Beat frekansı (3-50 Hz). Theta: 4-8, Alpha: 8-13, Gamma: 38-42"
            )
            binaural_amplitude = st.sidebar.slider(
                "Amplitude",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Genlik (0.1-1.0). Optimal: 0.5"
            )
            
            # Danger zone warnings
            if beat_freq < 3.0:
                st.sidebar.error("⚠️ <3 Hz: Rotating tone - Rahatsız edici!")
            elif 13.0 <= beat_freq <= 30.0:
                st.sidebar.warning("⚠️ Beta band (13-30 Hz): Hiçbir kanıt yok!")
            elif beat_freq > 50.0:
                st.sidebar.warning("⚠️ >50 Hz: Beat algısı kaybolur!")
        
        # Optional: Pink noise background
        st.sidebar.markdown("---")
        st.sidebar.info("⚠️ Pink noise render süresi uzatır (Python loop overhead)")
        add_pink_noise = st.sidebar.checkbox("Pink Noise Arka Plan Ekle", value=False)
        pink_noise_gain = 0.0
        if add_pink_noise:
            pink_noise_gain = st.sidebar.slider(
                "Pink Noise Gain",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Arka plan pink noise seviyesi"
            )
            st.sidebar.caption("📝 Not: Render süresi ~2-3x daha uzun olacak (8M+ iterasyon)")
            st.sidebar.caption("💡 Öneri: Pure binaural beats (checkbox kapalı) daha hızlı")
        
        enable_binaural = True  # Always enabled in V2.5
        
    # Initialize organic texture variables (default disabled for V2.5)
    # Theory reference: organic_texture_theory.md Section 4.1-4.2
    enable_organic = False
    organic_preset = "Theory"
    sub_bass_gain_db = -15.0  # More conservative (was -9, too deep)
    air_gain_db = -18.0        # Theory recommended
    sub_bass_lfo_depth = 0.12  # Reduced from 0.2 (less aggressive)
    air_lfo_depth = 0.08       # Reduced from 0.15 (more subtle)
    lfo_mod_amount = 0.25      # Moderate variation (was 0.3)
    
    if not is_binaural_only:
        # === ORGANIC TEXTURE (V2.7+) ===
        st.sidebar.markdown("---")
        st.sidebar.markdown("**🌿 Organic Texture (V2.7) - YouTube Ready**")
        st.sidebar.caption("Sub-bass rumble + Air presence + Irregular breathing")
        
        enable_organic = st.sidebar.checkbox("Organic Texture Ekle", value=True, help="Sub-bass (20-80Hz) ve air (4-8kHz) katmanları + Perlin-modulated breathing")
        
        if enable_organic:
            organic_preset = st.sidebar.radio(
                "Preset",
                options=["Subtle", "Theory (Recommended)", "Enhanced"],
                index=1,
                help="Subtle: Minimal organic texture\nTheory: Academic reference (organic_texture_theory.md)\nEnhanced: More presence, healthy balance"
            )
            
            if organic_preset == "Subtle":
                # Minimal organic presence
                sub_bass_gain_db = -18.0
                air_gain_db = -21.0
                sub_bass_lfo_depth = 0.06
                air_lfo_depth = 0.04
                lfo_mod_amount = 0.15
                st.sidebar.info("🔇 Minimal presence - Ultra subtle")
            elif organic_preset == "Theory (Recommended)":
                # Academic reference: organic_texture_theory.md Section 4
                # Sub-bass: -12dB (Section 4.1.2), Air: -18dB (Section 4.2.2)
                # LFO: Moderate depth for breathing without overpowering
                sub_bass_gain_db = -15.0  # Conservative (theory: -12, but adjusted for health)
                air_gain_db = -18.0       # Theory baseline
                sub_bass_lfo_depth = 0.12
                air_lfo_depth = 0.08
                lfo_mod_amount = 0.25
                st.sidebar.success("✅ Theory-based healthy balance")
            else:  # Enhanced
                # More presence but still healthy
                sub_bass_gain_db = -12.0  # Theory baseline
                air_gain_db = -15.0       # Slightly more air
                sub_bass_lfo_depth = 0.18
                air_lfo_depth = 0.12
                lfo_mod_amount = 0.35
                st.sidebar.info("🌊 Enhanced presence - Still healthy")
            
            # Advanced controls (expandable)
            with st.sidebar.expander("🔧 Advanced Parameters"):
                sub_bass_gain_db = st.slider("Sub-Bass Gain (dB)", -24.0, 0.0, sub_bass_gain_db, 1.0)
                air_gain_db = st.slider("Air Gain (dB)", -24.0, 0.0, air_gain_db, 1.0)
                sub_bass_lfo_depth = st.slider("Sub-Bass LFO Depth", 0.0, 1.0, sub_bass_lfo_depth, 0.05)
                air_lfo_depth = st.slider("Air LFO Depth", 0.0, 1.0, air_lfo_depth, 0.05)
                lfo_mod_amount = st.slider("LFO Mod Amount (±%)", 0.0, 1.0, lfo_mod_amount, 0.05, 
                                          help="Perlin noise frequency variation. 0.3 = ±30%")
        
        # V1/V2: Optional binaural addon
        st.sidebar.markdown("---")
        st.sidebar.markdown("**🎧 Binaural Beats (Opsiyonel)**")
        enable_binaural = st.sidebar.checkbox("Binaural Beats Ekle", value=False)
        
        carrier_freq = 200.0
        beat_freq = 10.0
        binaural_amplitude = 0.5
        add_pink_noise = False
        pink_noise_gain = 0.0
        
        if enable_binaural:
            st.sidebar.info("⚠️ Kulaklık kullanımı ZORUNLU!")
            
            # Preset recommendations (V2.5 ile aynı liste)
            binaural_preset = st.sidebar.selectbox(
                "Preset Seç:",
                options=[
                    "Theta (7 Hz) - Meditation",
                    "Alpha (10 Hz) - Relaxation",
                    "Gamma (40 Hz) - Focus",
                    "Delta (4 Hz) - Deep Sleep",
                    "--- Solfeggio Frequencies ---",
                    "432Hz Universe Harmony",
                    "528Hz Love Frequency",
                    "639Hz Connection",
                    "741Hz Intuition",
                    "852Hz Spiritual Order",
                    "963Hz Third Eye",
                    "Custom"
                ],
                index=0  # Default: Theta
            )
            
            if binaural_preset == "Theta (7 Hz) - Meditation":
                carrier_freq = 400.0
                beat_freq = 7.0
                binaural_amplitude = 0.5
                st.sidebar.caption("Priority #1: En çok araştırılmış (10 dk önerilen)")
            elif binaural_preset == "Alpha (10 Hz) - Relaxation":
                carrier_freq = 400.0
                beat_freq = 10.0
                binaural_amplitude = 0.5
                st.sidebar.caption("Priority #2: Kısa sürede etkili (5 dk yeterli)")
            elif binaural_preset == "Gamma (40 Hz) - Focus":
                carrier_freq = 250.0
                beat_freq = 40.0
                binaural_amplitude = 0.5
                st.sidebar.caption("Experimental: Uzun süre gerekli (20+ dk)")
            elif binaural_preset == "Delta (4 Hz) - Deep Sleep":
                carrier_freq = 400.0
                beat_freq = 4.0
                binaural_amplitude = 0.5
                st.sidebar.caption("Experimental: Derin uyku (5-10 dk)")
            elif binaural_preset == "--- Solfeggio Frequencies ---":
                # Separator - keep defaults
                carrier_freq = 400.0
                beat_freq = 7.0
                binaural_amplitude = 0.5
                st.sidebar.caption("⬇️ Lütfen bir Solfeggio preset seçin")
            elif binaural_preset == "432Hz Universe Harmony":
                carrier_freq = 432.0
                beat_freq = 8.0
                binaural_amplitude = 0.5
                st.sidebar.caption("🎵 Solfeggio | YouTube: ⭐⭐⭐⭐⭐")
            elif binaural_preset == "528Hz Love Frequency":
                carrier_freq = 528.0
                beat_freq = 4.0
                binaural_amplitude = 0.5
                st.sidebar.caption("❤️ Solfeggio | YouTube: ⭐⭐⭐⭐⭐ (En popüler!)")
            elif binaural_preset == "639Hz Connection":
                carrier_freq = 639.0
                beat_freq = 10.0
                binaural_amplitude = 0.45
                st.sidebar.caption("🤝 Solfeggio | YouTube: ⭐⭐⭐⭐")
            elif binaural_preset == "741Hz Intuition":
                carrier_freq = 741.0
                beat_freq = 7.0
                binaural_amplitude = 0.45
                st.sidebar.caption("🧠 Solfeggio | YouTube: ⭐⭐⭐⭐")
            elif binaural_preset == "852Hz Spiritual Order":
                carrier_freq = 852.0
                beat_freq = 8.0
                binaural_amplitude = 0.4
                st.sidebar.caption("✨ Solfeggio | YouTube: ⭐⭐⭐")
            elif binaural_preset == "963Hz Third Eye":
                carrier_freq = 963.0
                beat_freq = 10.0
                binaural_amplitude = 0.4
                st.sidebar.caption("👁️ Solfeggio | YouTube: ⭐⭐⭐⭐⭐")
            else:  # Custom
                carrier_freq = st.sidebar.number_input(
                    "Carrier Frequency (Hz)",
                    min_value=100.0,
                    max_value=1200.0,
                    value=200.0,
                    step=10.0,
                    help="Taşıyıcı frekans (100-1200 Hz). Optimal: 200-400 Hz"
                )
                beat_freq = st.sidebar.number_input(
                    "Beat Frequency (Hz)",
                    min_value=3.0,
                    max_value=50.0,
                    value=10.0,
                    step=0.5,
                    help="Beat frekansı (3-50 Hz). Theta: 4-8, Alpha: 8-13, Gamma: 38-42"
                )
                binaural_amplitude = st.sidebar.slider(
                    "Amplitude",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    help="Genlik (0.1-1.0). Optimal: 0.3-0.6"
                )
                
                # Danger zone warnings
                if beat_freq < 3.0:
                    st.sidebar.error("⚠️ <3 Hz: Rotating tone - Rahatsız edici!")
                elif 13.0 <= beat_freq <= 30.0:
                    st.sidebar.warning("⚠️ Beta band (13-30 Hz): Hiçbir kanıt yok!")
                elif beat_freq > 50.0:
                    st.sidebar.warning("⚠️ >50 Hz: Beat algısı kaybolur!")

    # 2. Preset Seçimi
    st.subheader("1. Preset Seçimi")

    if is_binaural_only:
        # V2.5: Pure Binaural Beats (no preset selection)
        st.info(
            "🎧 **V2.5 - Pure Binaural Beats Mode**\n\n"
            "Bu modda sadece binaural beats üretilir.\n"
            "- Stereo output (2 kanal)\n"
            "- Akademik araştırmalara dayalı parametreler\n"
            "- Kulaklık kullanımı zorunlu\n\n"
            f"**Seçili Preset:** {binaural_preset}\n"
            f"**Carrier:** {carrier_freq} Hz | **Beat:** {beat_freq} Hz | **Amplitude:** {binaural_amplitude}"
        )
        
        if add_pink_noise:
            st.caption(f"🌸 Pink noise arka plan eklendi (gain: {pink_noise_gain})")
        
        # Create dummy preset for rendering
        from preset_system.preset_schema import PresetConfig, LayerConfig, BinauralConfig
        
        if add_pink_noise:
            # Pink noise layer
            selected_preset = PresetConfig(
                name=f"V2.5 Pure Binaural - {binaural_preset}",
                description=f"Pure binaural beats at {beat_freq} Hz with pink noise background",
                author="xxxDSP Engine V2.5",
                version="2.5",
                tags=["binaural", "v2.5", "pure"],
                layers=[
                    LayerConfig(
                        name="Pink Noise Background",
                        enabled=True,
                        noise_type="pink",
                        gain=pink_noise_gain,
                        pan=0.0
                    )
                ],
                binaural_config=BinauralConfig(
                    enabled=True,
                    carrier_freq=carrier_freq,
                    beat_freq=beat_freq,
                    amplitude=binaural_amplitude
                ),
                master_gain=0.8,
                duration_sec=duration,
                sample_rate=sr
            )
        else:
            # Pure binaural only (no layers)
            selected_preset = PresetConfig(
                name=f"V2.5 Pure Binaural - {binaural_preset}",
                description=f"Pure binaural beats at {beat_freq} Hz (no background)",
                author="xxxDSP Engine V2.5",
                version="2.5",
                tags=["binaural", "v2.5", "pure"],
                layers=[],  # No layers
                binaural_config=BinauralConfig(
                    enabled=True,
                    carrier_freq=carrier_freq,
                    beat_freq=beat_freq,
                    amplitude=binaural_amplitude
                ),
                master_gain=0.8,
                duration_sec=duration,
                sample_rate=sr
            )
        
        selected_preset_id = f"v2.5_binaural_{int(beat_freq)}hz"
        
    elif is_v2:
        # V2: ML-generated presets (Nested: Category → Profile)
        from preset_system.preset_library import list_v2_presets, get_v2_preset
        v2_presets = list_v2_presets()
        
        # Extract unique categories (first part before "__")
        categories = {}
        for preset in v2_presets:
            # Split by "__" and get first part as category
            name_parts = preset["name"].split("__")
            category = name_parts[0]
            profile = preset["profile"]
            
            if category not in categories:
                categories[category] = []
            categories[category].append(preset)
        
        # Sort categories
        sorted_categories = sorted(categories.keys())
        
        # Step 1: Category selection
        selected_category = st.selectbox(
            "1. Kategori Seç:",
            sorted_categories,
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        # Step 2: Profile selection (filtered by category)
        available_presets = categories[selected_category]
        available_presets_sorted = sorted(available_presets, key=lambda x: x["profile"])
        
        # Profile labels
        profile_labels = {
            "P0": "Baseline",
            "P1": "Soft",
            "P2": "Dark",
            "P3": "Calm",
            "P4": "Airy",
            "P5": "Dense"
        }
        
        selected_profile_idx = st.selectbox(
            "2. Profil Seç:",
            range(len(available_presets_sorted)),
            format_func=lambda x: f"{available_presets_sorted[x]['profile']} ({profile_labels[available_presets_sorted[x]['profile']]})"
        )
        
        # Get final preset
        selected_preset_dict = available_presets_sorted[selected_profile_idx]
        selected_preset = get_v2_preset(selected_preset_dict["path"])
        selected_preset_id = selected_preset_dict["name"]
    else:
        # V1: Handcrafted presets
        from preset_system.preset_library import list_all_presets, get_preset
        v1_preset_ids = list_all_presets()

        selected_index = st.selectbox(
            "V1 Preset Seç (14 handcrafted):",
            range(len(v1_preset_ids)),
            format_func=lambda x: v1_preset_ids[x]
        )
        selected_preset_id = v1_preset_ids[selected_index]
        selected_preset = get_preset(selected_preset_id)


    # Preset Detayları
    if not is_binaural_only:
        with st.expander("Preset Detayları", expanded=True):
            st.markdown(f"**ID:** `{selected_preset_id}`")
            st.markdown(f"**Açıklama:** {selected_preset.description}")
            st.markdown(f"**Etiketler:** {', '.join(selected_preset.tags)}")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Katman Sayısı", len(selected_preset.layers))
            with col2:
                rev_mix = float(selected_preset.fx_config.reverb_mix)
                st.metric("Reverb", f"%{int(rev_mix * 100)}")
            with col3:
                if enable_organic:
                    st.metric("Organic", f"✅ {organic_preset.split()[0]}")
                else:
                    st.metric("Organic", "❌ Kapalı")
            with col4:
                if enable_binaural:
                    st.metric("Binaural", "✅ Aktif")
                else:
                    st.metric("Binaural", "❌ Kapalı")
    else:
        # V2.5: Binaural info already shown above
        pass

    # 3. Render
    st.markdown("---")
    st.subheader("2. Ses Üretimi")

    if st.button("RENDER BAŞLAT", type="primary", use_container_width=True):
        status_text = st.empty()
        progress_bar = st.progress(0)

        try:
            render_config = selected_preset

            if use_variant and is_v1:
                status_text.text("Varyasyon hesaplanıyor...")
                render_config = preset_autogen.generate_variant(
                    selected_preset,
                    intensity=variant_intensity,
                    suffix="Streamlit"
                )
                time.sleep(0.5)
            
            # Organic texture control (V2.7+)
            if enable_organic and not is_binaural_only:
                from preset_system.preset_schema import OrganicTextureConfig
                render_config.organic_texture_config = OrganicTextureConfig(
                    enabled=True,
                    sub_bass_enabled=True,
                    sub_bass_noise_type="brown",
                    sub_bass_lp_cutoff_hz=80.0,
                    sub_bass_hp_cutoff_hz=20.0,
                    sub_bass_gain_db=sub_bass_gain_db,
                    sub_bass_lfo_rate_hz=0.008,
                    sub_bass_lfo_type="perlin_modulated",
                    sub_bass_lfo_depth=sub_bass_lfo_depth,
                    sub_bass_lfo_mod_amount=lfo_mod_amount,
                    air_enabled=True,
                    air_noise_type="white",
                    air_hp_cutoff_hz=4000.0,
                    air_lp_cutoff_hz=8000.0,
                    air_gain_db=air_gain_db,
                    air_lfo_rate_hz=0.01,
                    air_lfo_type="perlin_modulated",
                    air_lfo_depth=air_lfo_depth,
                    air_lfo_mod_amount=lfo_mod_amount
                )
            else:
                # CRITICAL: Disable organic if checkbox is OFF
                # Presets may have organic_texture_config in JSON
                render_config.organic_texture_config = None
            
            # Binaural beats config ekle (eğer V1/V2'de aktifse)
            if enable_binaural and not is_binaural_only:
                from preset_system.preset_schema import BinauralConfig
                render_config.binaural_config = BinauralConfig(
                    enabled=True,
                    carrier_freq=carrier_freq,
                    beat_freq=beat_freq,
                    amplitude=binaural_amplitude
                )
                status_text.text(f"Binaural beats ekleniyor... ({beat_freq} Hz)")
                time.sleep(0.3)

            status_text.text(f"DSP Motoru Çalışıyor... ({duration} sn)")
            progress_bar.progress(30)

            start_time = time.time()

            # === BINAURAL SUPPORT: Check if binaural is enabled ===
            result = adapt_preset_to_layer_generators(render_config)
            
            if isinstance(result, tuple):
                # Binaural aktif - stereo rendering
                layer_gens, binaural_gen = result
                
                status_text.text("Mono layer'lar render ediliyor...")
                progress_bar.progress(50)
                
                # Render mono layers
                if len(layer_gens) > 0:
                    print(f"[DEBUG] Rendering {len(layer_gens)} mono layers (duration={duration}s)...")
                    
                    # Pink noise warning (çok yavaş)
                    if is_binaural_only and add_pink_noise:
                        status_text.text(f"Pink noise render ediliyor... ({int(duration)}s, ~{int(duration*2)}s sürebilir)")
                    
                    # CRITICAL: Pink noise için multiprocessing kullanma (Windows pickle + Python loop overhead)
                    use_mp = not (is_binaural_only and add_pink_noise)
                    
                    mono_audio = dsp_render.render_sound(
                        layer_gens,
                        duration_sec=duration,
                        sample_rate=sr,
                        use_multiprocessing=use_mp
                    )
                    print(f"[DEBUG] Mono audio rendered: shape={mono_audio.shape}, dtype={mono_audio.dtype}")
                    
                    # Convert to stereo (duplicate channels)
                    stereo_layers = np.stack([mono_audio, mono_audio], axis=1)
                    print(f"[DEBUG] Stereo layers created: shape={stereo_layers.shape}")
                else:
                    print(f"[DEBUG] No layers, creating zero stereo buffer...")
                    stereo_layers = np.zeros((int(duration * sr), 2), dtype=np.float32)
                    print(f"[DEBUG] Zero stereo buffer: shape={stereo_layers.shape}")
                
                status_text.text("Binaural beats render ediliyor...")
                progress_bar.progress(75)
                
                # Render binaural (stereo)
                print(f"[DEBUG] Rendering binaural beats...")
                binaural_audio = binaural_gen(duration, sr)
                print(f"[DEBUG] Binaural audio rendered: shape={binaural_audio.shape}, dtype={binaural_audio.dtype}")
                
                status_text.text("Mixing: Layers + Binaural...")
                progress_bar.progress(90)
                
                # Mix stereo
                print(f"[DEBUG] Mixing stereo_layers + binaural_audio...")
                audio_data = stereo_layers + binaural_audio
                print(f"[DEBUG] Mix complete: shape={audio_data.shape}, dtype={audio_data.dtype}")
                
            else:
                # Binaural yok - normal mono rendering
                layer_gens = result
                audio_data = dsp_render.render_sound(
                    layer_gens,
                    duration_sec=duration,
                    sample_rate=sr
                )

            elapsed = time.time() - start_time
            progress_bar.progress(100)
            
            # Channel info
            if audio_data.ndim == 2:
                channel_info = f"Stereo ({audio_data.shape[1]} kanal)"
            else:
                channel_info = "Mono"
            
            status_text.success(f"Tamamlandı! ({elapsed:.2f}s) - {channel_info}")

            # 4. Sonuç
            st.markdown("---")
            st.subheader("3. Sonuç")

            wav_bytes = convert_to_wav_bytes(audio_data, sr)
            st.audio(wav_bytes, format="audio/wav")

            # Generate filename based on version
            duration_str = format_duration(duration)
            
            if is_binaural_only:
                # V2.5: binaural_<preset>_<beat_freq>hz_<duration>.wav
                preset_name = binaural_preset.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
                if add_pink_noise:
                    file_name = f"binaural_{preset_name}_{int(beat_freq)}hz_{duration_str}_pink.wav"
                else:
                    file_name = f"binaural_{preset_name}_{int(beat_freq)}hz_{duration_str}.wav"
            elif is_v2:
                # V2: <index>_<category>_<profile>_<duration>.wav
                # Index: 1-6 per category (selected_profile_idx + 1)
                index = str(selected_profile_idx + 1).zfill(3)
                category = selected_category.lower()
                profile = available_presets_sorted[selected_profile_idx]['profile'].lower()
                
                # Add binaural suffix if enabled
                if enable_binaural:
                    file_name = f"{index}_{category}_{profile}_{duration_str}_binaural_{int(beat_freq)}hz.wav"
                else:
                    file_name = f"{index}_{category}_{profile}_{duration_str}.wav"
            else:
                # V1: <index>_<preset_name>_<duration>.wav
                # Index: 1-14 global (selected_index + 1)
                index = str(selected_index + 1).zfill(3)
                preset_name = selected_preset_id.lower()
                
                # Add binaural suffix if enabled
                if enable_binaural:
                    file_name = f"{index}_{preset_name}_{duration_str}_binaural_{int(beat_freq)}hz.wav"
                else:
                    file_name = f"{index}_{preset_name}_{duration_str}.wav"
            
            st.download_button(
                label="📥 WAV Olarak İndir",
                data=wav_bytes,
                file_name=file_name,
                mime="audio/wav",
                use_container_width=True
            )
            
            # Binaural info
            if enable_binaural or is_binaural_only:
                band_info = ""
                if 3.0 <= beat_freq < 4.0:
                    band_info = "Delta (Derin uyku)"
                elif 4.0 <= beat_freq < 8.0:
                    band_info = "Theta (Meditasyon)"
                elif 8.0 <= beat_freq < 13.0:
                    band_info = "Alpha (Rahatlama)"
                elif 13.0 <= beat_freq < 30.0:
                    band_info = "Beta (Dikkat - kanıt yok!)"
                elif 30.0 <= beat_freq <= 50.0:
                    band_info = "Gamma (Yoğun odaklanma)"
                
                st.info(
                    f"🎧 **Binaural Beats Aktif:**\n"
                    f"- Band: {band_info}\n"
                    f"- Carrier: {carrier_freq} Hz\n"
                    f"- Beat: {beat_freq} Hz\n"
                    f"- Amplitude: {binaural_amplitude}\n\n"
                    f"⚠️ **KULAKLIK KULLANIN!** Hoparlörde çalışmaz."
                )

        except Exception as e:
            st.error(f"Render sırasında hata oluştu: {e}")
            progress_bar.empty()


if __name__ == "__main__":
    main()
