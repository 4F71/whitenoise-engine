"""
ui_app/ui_gradio.py

` python ui_app/ui_gradio.py ` komutu ile çalıştırılır.

xxxDSP / xxxDSP Engine - Gradio Web Arayüzü
====================================================
V2.8: Simple Dropdown Design (First Version Restored)
"""

import sys
import os
from pathlib import Path
import numpy as np
import gradio as gr
from scipy.io import wavfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core_dsp import dsp_render
from preset_system import preset_library, preset_autogen
from preset_system.preset_schema import PresetConfig, LayerConfig, BinauralConfig, OrganicTextureConfig
from ui_app.preset_to_dsp_adapter import adapt_preset_to_layer_generators


# =============================================================================
# HELPERS
# =============================================================================

def format_duration(seconds: float) -> str:
    """Format duration."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def load_v2_data():
    """Load V2 categories and presets."""
    v2_presets = preset_library.list_v2_presets()
    categories = {}
    
    for p in v2_presets:
        cat = p["name"].split("__")[0]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(p)
    
    return sorted(categories.keys()), categories


def get_v2_profiles(cat_dict, category):
    """Get profiles for category."""
    if category not in cat_dict:
        return []
    
    presets = sorted(cat_dict[category], key=lambda x: x["profile"])
    labels = {"P0": "Baseline", "P1": "Soft", "P2": "Dark", "P3": "Calm", "P4": "Airy", "P5": "Dense"}
    
    return [f"{p['profile']} ({labels[p['profile']]})" for p in presets]


def get_preset(version, v1_id, v2_cat, v2_prof, cat_dict):
    """Get preset config."""
    if version == "V1 (14 handcrafted)":
        return preset_library.get_preset(v1_id)
    elif version == "V2 (102 ML-generated)":
        presets = sorted(cat_dict[v2_cat], key=lambda x: x["profile"])
        prof_key = v2_prof.split()[0]
        selected = next(p for p in presets if p["profile"] == prof_key)
        return preset_library.get_v2_preset(selected["path"])
    return None


def cleanup_temp_files():
    """Clean up old render files to prevent memory buildup."""
    import glob
    try:
        output_dir = Path("gradio_renders")
        if output_dir.exists():
            for wav_file in output_dir.glob("*.wav"):
                try:
                    wav_file.unlink()
                except:
                    pass
    except:
        pass


def get_binaural_params(preset_name):
    """Get carrier, beat, amplitude for V2.5 binaural preset."""
    params_map = {
        "Theta (7 Hz) - Meditation": (400.0, 7.0, 0.5, "Priority #1: En çok araştırılmış\n\nÖnerilen: 10 dakika | Meditasyon"),
        "Alpha (10 Hz) - Relaxation": (400.0, 10.0, 0.5, "Priority #2: Kısa sürede etkili\n\nÖnerilen: 5 dakika | Rahatlamış uyanıklık"),
        "Gamma (40 Hz) - Focus": (250.0, 40.0, 0.5, "Experimental: Uzun süre gerekli\n\nÖnerilen: 20-30 dakika | Konsantrasyon"),
        "Delta (4 Hz) - Deep Sleep": (400.0, 4.0, 0.5, "Experimental: Derin uyku\n\nÖnerilen: 5-10 dakika | Uyarı: <3 Hz kullanmayın!"),
        "432Hz Universe Harmony": (432.0, 8.0, 0.5, "Solfeggio: Universe frequency\n\nYouTube: 5/5 stars | 10 dakika"),
        "528Hz Love Frequency": (528.0, 4.0, 0.5, "Solfeggio: Love & DNA repair\n\nYouTube: 5/5 stars (En popüler!) | 10 dakika"),
        "639Hz Connection": (639.0, 10.0, 0.45, "Solfeggio: Relationships\n\nYouTube: 4/5 stars | 10 dakika"),
        "741Hz Intuition": (741.0, 7.0, 0.45, "Solfeggio: Awakening intuition\n\nYouTube: 4/5 stars | 10 dakika"),
        "852Hz Spiritual Order": (852.0, 8.0, 0.4, "Solfeggio: Spiritual awakening\n\nYouTube: 3/5 stars | 10 dakika"),
        "963Hz Third Eye": (963.0, 10.0, 0.4, "Solfeggio: Third eye activation\n\nYouTube: 5/5 stars | 10 dakika"),
        "--- Solfeggio Frequencies ---": (400.0, 7.0, 0.5, "WARNING: Lütfen bir preset seçin")
    }
    
    return params_map.get(preset_name, (400.0, 10.0, 0.5, ""))


# =============================================================================
# RENDER
# =============================================================================

def render_preset(
    version, v1_id, use_variant, variant_int, v2_cat, v2_prof,
    v25_preset_name, add_pink_v25, pink_gain_v25,
    dur_mode, dur_minutes, dur_hours, sr,
    org_en, sub_gain, air_gain, lfo_mod,
    bin_en, carrier, beat, amp,
    cat_dict,
    progress=gr.Progress()
):
    """Main render function."""
    try:
        # Cleanup old temp files
        cleanup_temp_files()
        
        progress(0.05, desc="Cleaning up...")
        
        # V2.5 special handling
        if version == "V2.5 (Pure Binaural Beats)":
            # Get binaural params from preset
            carrier_v25, beat_v25, amp_v25, _ = get_binaural_params(v25_preset_name)
            
            # Create dummy preset for V2.5
            if add_pink_v25:
                preset = PresetConfig(
                    name=f"V2.5 {v25_preset_name}",
                    description=f"Pure binaural beats at {beat_v25} Hz with pink noise",
                    author="xxxDSP V2.5",
                    version="2.5",
                    tags=["binaural", "v2.5"],
                    layers=[LayerConfig(
                        name="Pink Noise Background",
                        enabled=True,
                        noise_type="pink",
                        gain=pink_gain_v25,
                        pan=0.0
                    )],
                    binaural_config=BinauralConfig(
                        enabled=True,
                        carrier_freq=carrier_v25,
                        beat_freq=beat_v25,
                        amplitude=amp_v25
                    ),
                    master_gain=0.8,
                    duration_sec=600.0,
                    sample_rate=sr
                )
            else:
                preset = PresetConfig(
                    name=f"V2.5 {v25_preset_name}",
                    description=f"Pure binaural beats at {beat_v25} Hz",
                    author="xxxDSP V2.5",
                    version="2.5",
                    tags=["binaural", "v2.5"],
                    layers=[],
                    binaural_config=BinauralConfig(
                        enabled=True,
                        carrier_freq=carrier_v25,
                        beat_freq=beat_v25,
                        amplitude=amp_v25
                    ),
                    master_gain=0.8,
                    duration_sec=600.0,
                    sample_rate=sr
                )
        else:
            preset = get_preset(version, v1_id, v2_cat, v2_prof, cat_dict)
            if not preset:
                return None, "ERROR: Preset seçin!", "", "", "", "", "", ""
        
        # Variant (V1 only)
        if version == "V1 (14 handcrafted)" and use_variant:
            preset = preset_autogen.generate_variant(preset, variant_int, "Gradio")
        
        # Duration
        dur_sec = dur_minutes * 60 if dur_mode == "minutes" else dur_hours * 3600
        
        progress(0.2, desc="Configuring...")
        
        # Organic (V1/V2 only)
        if org_en and version != "V2.5 (Pure Binaural Beats)":
            preset.organic_texture_config = OrganicTextureConfig(
                enabled=True,
                sub_bass_enabled=True, sub_bass_noise_type="brown",
                sub_bass_lp_cutoff_hz=80.0, sub_bass_hp_cutoff_hz=20.0,
                sub_bass_gain_db=sub_gain, sub_bass_lfo_rate_hz=0.008,
                sub_bass_lfo_type="perlin_modulated", sub_bass_lfo_depth=0.12,
                sub_bass_lfo_mod_amount=lfo_mod,
                air_enabled=True, air_noise_type="white",
                air_hp_cutoff_hz=4000.0, air_lp_cutoff_hz=8000.0,
                air_gain_db=air_gain, air_lfo_rate_hz=0.01,
                air_lfo_type="perlin_modulated", air_lfo_depth=0.08,
                air_lfo_mod_amount=lfo_mod
            )
        else:
            preset.organic_texture_config = None
        
        # Binaural (V1/V2 only - V2.5 already has binaural)
        if bin_en and version != "V2.5 (Pure Binaural Beats)":
            preset.binaural_config = BinauralConfig(
                enabled=True, carrier_freq=carrier, beat_freq=beat, amplitude=amp
            )
        
        progress(0.3, desc="Rendering...")
        
        result = adapt_preset_to_layer_generators(preset)
        
        if isinstance(result, tuple):
            layer_gens, binaural_gen = result
            if len(layer_gens) > 0:
                mono = dsp_render.render_sound(layer_gens, dur_sec, sr)
                stereo = np.stack([mono, mono], axis=1)
            else:
                stereo = np.zeros((int(dur_sec * sr), 2), dtype=np.float32)
            
            progress(0.7, desc="Binaural...")
            binaural_audio = binaural_gen(dur_sec, sr)
            audio_data = stereo + binaural_audio
        else:
            layer_gens = result
            audio_data = dsp_render.render_sound(layer_gens, dur_sec, sr)
        
        # Normalize
        peak = np.max(np.abs(audio_data))
        if peak > 0.99:
            audio_data *= (0.99 / peak)
        
        progress(0.95, desc="Saving...")
        
        # Save
        output_dir = Path("gradio_renders")
        output_dir.mkdir(exist_ok=True)
        
        preset_name = preset.name.lower().replace(" ", "_")[:30]
        dur_str = format_duration(dur_sec)
        output_file = output_dir / f"{preset_name}_{dur_str}.wav"
        
        audio_int16 = np.int16(audio_data * 32767)
        wavfile.write(str(output_file), sr, audio_int16)
        
        # Metrics
        rms = np.sqrt(np.mean(audio_data**2))
        peak_dbfs = 20 * np.log10(peak + 1e-12)
        rms_dbfs = 20 * np.log10(rms + 1e-12)
        dr = peak_dbfs - rms_dbfs
        dc = np.mean(audio_data)
        
        filesize_mb = output_file.stat().st_size / (1024**2)
        size_str = f"{filesize_mb:.1f} MB" if filesize_mb < 1024 else f"{filesize_mb/1024:.2f} GB"
        
        return (
            str(output_file),
            f"Render complete! ({format_duration(dur_sec)})",
            format_duration(dur_sec),
            size_str,
            f"{peak_dbfs:.2f} dBFS",
            f"{rms_dbfs:.2f} dBFS",
            f"{dr:.2f} dB",
            f"{dc:.6f}"
        )
        
    except Exception as e:
        return None, f"ERROR: {str(e)}", "", "", "", "", "", ""


# =============================================================================
# UI
# =============================================================================

def create_ui():
    """Create UI."""
    
    v1_ids = preset_library.list_all_presets()
    v2_cats, v2_dict = load_v2_data()
    
    with gr.Blocks(title="xxxDSP Engine") as demo:
        
        cat_state = gr.State(v2_dict)
        
        gr.Markdown("# xxxDSP / xxxDSP Engine")
        gr.Markdown("Akademik DSP tabanlı, deterministik gürültü ve atmosfer üretici.")
        
        # Preset Version
        version = gr.Radio(
            choices=["V1 (14 handcrafted)", "V2 (102 ML-generated)", "V2.5 (Pure Binaural Beats)"],
            value="V1 (14 handcrafted)",
            label="Preset Versiyonu"
        )
        
        with gr.Row():
            # LEFT PANEL
            with gr.Column(scale=1):
                
                # V1 Controls
                with gr.Group(visible=True) as v1_grp:
                    v1_dd = gr.Dropdown(choices=v1_ids, value=v1_ids[0], label="V1 Preset Seç")
                    use_var = gr.Checkbox(label="Otomatik Varyasyon Üret", value=False)
                    var_int = gr.Slider(0.0, 0.5, 0.15, 0.05, label="Varyasyon Şiddeti", visible=False)
                
                # V2 Controls
                with gr.Group(visible=False) as v2_grp:
                    v2_cat = gr.Dropdown(
                        choices=[c.replace("_", " ").title() for c in v2_cats],
                        value=v2_cats[0].replace("_", " ").title(),
                        label="1. Kategori Seç"
                    )
                    v2_prof = gr.Dropdown(
                        choices=get_v2_profiles(v2_dict, v2_cats[0]),
                        value=get_v2_profiles(v2_dict, v2_cats[0])[0],
                        label="2. Profil Seç"
                    )
                
                # V2.5 Controls
                with gr.Group(visible=False) as v25_grp:
                    gr.Markdown("### V2.5 Binaural Beats")
                    gr.Markdown("**WARNING: Kulaklık kullanımı ZORUNLU!**")
                    
                    v25_preset = gr.Dropdown(
                        choices=[
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
                            "963Hz Third Eye"
                        ],
                        value="Theta (7 Hz) - Meditation",
                        label="Binaural Preset"
                    )
                    
                    v25_info = gr.Markdown("Priority #1: En çok araştırılmış\n\nÖnerilen: 10 dakika | Meditasyon")
                    
                    add_pink = gr.Checkbox(
                        label="Pink Noise Arka Plan Ekle",
                        value=False,
                        info="WARNING: Render süresi uzar"
                    )
                    pink_gain = gr.Slider(0.1, 0.5, 0.2, 0.05, label="Pink Noise Gain", visible=False)
                
                gr.Markdown("---")
                
                # Duration
                gr.Markdown("### Süre Seçimi")
                dur_mode = gr.Radio(choices=["minutes", "hours"], value="minutes", label="Birim")
                dur_min = gr.Slider(1, 60, 10, 1, label="Dakika", visible=True)
                dur_hr = gr.Slider(1, 10, 1, 1, label="Saat", visible=False)
                
                # Sample Rate
                sr = gr.Dropdown(choices=[44100, 48000], value=48000, label="Örnekleme Hızı (Hz)")
                
                gr.Markdown("---")
                
                # Organic
                with gr.Accordion("Organic Texture (Opsiyonel)", open=False):
                    org_chk = gr.Checkbox(label="Organic Texture Ekle", value=False)
                    with gr.Group(visible=False) as org_grp:
                        sub_sl = gr.Slider(-24, 0, -15, 1, label="Sub-Bass Gain (dB)")
                        air_sl = gr.Slider(-24, 0, -18, 1, label="Air Gain (dB)")
                        lfo_sl = gr.Slider(0, 1, 0.25, 0.05, label="LFO Mod Amount")
                
                gr.Markdown("---")
                
                # Binaural
                with gr.Accordion("Binaural Beats (Opsiyonel)", open=False):
                    bin_chk = gr.Checkbox(label="Binaural Beats Ekle", value=False, info="WARNING: Kulaklık kullanımı ZORUNLU!")
                    with gr.Group(visible=False) as bin_grp:
                        car_num = gr.Number(value=400, label="Carrier Frequency (Hz)", minimum=100, maximum=1200)
                        beat_num = gr.Number(value=10, label="Beat Frequency (Hz)", minimum=3, maximum=50)
                        amp_sl = gr.Slider(0.1, 1.0, 0.5, 0.1, label="Amplitude")
            
            # RIGHT PANEL
            with gr.Column(scale=2):
                gr.Markdown("## Ses Üretimi")
                
                btn = gr.Button("RENDER BAŞLAT", variant="primary", size="lg")
                status = gr.Textbox(label="Durum", interactive=False, visible=False)
                
                gr.Markdown("---")
                gr.Markdown("## Sonuç")
                
                audio = gr.Audio(label="Audio Preview", type="filepath", visible=False)
                dl = gr.File(label="WAV Olarak İndir", visible=False)
                
                gr.Markdown("---")
                gr.Markdown("## Kalite Metrikleri")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Süre**")
                        m_dur = gr.Textbox(value="", interactive=False, show_label=False)
                    with gr.Column():
                        gr.Markdown("**Dosya Boyutu**")
                        m_size = gr.Textbox(value="", interactive=False, show_label=False)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Peak**")
                        m_peak = gr.Textbox(value="", interactive=False, show_label=False)
                    with gr.Column():
                        gr.Markdown("**RMS**")
                        m_rms = gr.Textbox(value="", interactive=False, show_label=False)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Dynamic Range**")
                        m_dyn = gr.Textbox(value="", interactive=False, show_label=False)
                    with gr.Column():
                        gr.Markdown("**DC Offset**")
                        m_dc = gr.Textbox(value="", interactive=False, show_label=False)
        
        # =====================================================================
        # EVENTS
        # =====================================================================
        
        def update_version_visibility(v):
            is_v1 = v == "V1 (14 handcrafted)"
            is_v2 = v == "V2 (102 ML-generated)"
            is_v25 = v == "V2.5 (Pure Binaural Beats)"
            return {
                v1_grp: gr.update(visible=is_v1),
                v2_grp: gr.update(visible=is_v2),
                v25_grp: gr.update(visible=is_v25)
            }
        
        def update_v2_profiles(cat, d):
            cat_key = cat.lower().replace(" ", "_")
            profs = get_v2_profiles(d, cat_key)
            return gr.update(choices=profs, value=profs[0] if profs else None)
        
        def update_duration_visibility(mode):
            return {
                dur_min: gr.update(visible=mode == "minutes"),
                dur_hr: gr.update(visible=mode == "hours")
            }
        
        def toggle_variant(enabled):
            return gr.update(visible=enabled)
        
        def toggle_organic(enabled):
            return gr.update(visible=enabled)
        
        def toggle_binaural(enabled):
            return gr.update(visible=enabled)
        
        def toggle_pink_noise(enabled):
            return gr.update(visible=enabled)
        
        def clear_audio_on_render_start():
            """Clear audio player and show render started message."""
            return {
                audio: gr.update(visible=False, value=None),
                dl: gr.update(visible=False, value=None),
                status: gr.update(value="Rendering başladı...", visible=True),
                m_dur: gr.update(value=""),
                m_size: gr.update(value=""),
                m_peak: gr.update(value=""),
                m_rms: gr.update(value=""),
                m_dyn: gr.update(value=""),
                m_dc: gr.update(value="")
            }
        
        # Wire up
        version.change(update_version_visibility, version, [v1_grp, v2_grp, v25_grp])
        v2_cat.change(update_v2_profiles, [v2_cat, cat_state], v2_prof)
        dur_mode.change(update_duration_visibility, dur_mode, [dur_min, dur_hr])
        use_var.change(toggle_variant, use_var, var_int)
        org_chk.change(toggle_organic, org_chk, org_grp)
        bin_chk.change(toggle_binaural, bin_chk, bin_grp)
        add_pink.change(toggle_pink_noise, add_pink, pink_gain)
        
        # Render
        btn.click(
            clear_audio_on_render_start,
            None,
            [audio, dl, status, m_dur, m_size, m_peak, m_rms, m_dyn, m_dc]
        ).then(
            render_preset,
            [version, v1_dd, use_var, var_int, v2_cat, v2_prof,
             v25_preset, add_pink, pink_gain,
             dur_mode, dur_min, dur_hr, sr,
             org_chk, sub_sl, air_sl, lfo_sl,
             bin_chk, car_num, beat_num, amp_sl, cat_state],
            [audio, status, m_dur, m_size, m_peak, m_rms, m_dyn, m_dc]
        ).then(
            lambda a, s: (
                gr.update(value=s, visible=True),
                gr.update(value=a, visible=a is not None),
                gr.update(value=a, visible=a is not None)
            ),
            [audio, status],
            [status, audio, dl]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    
    # Print clickable URLs BEFORE launch (blocking call)
    print("\n" + "=" * 60)
    print("xxxDSP Engine Running")
    print("=" * 60)
    print(f"* Local URL:    http://127.0.0.1:7860")
    print(f"* Network URL:  http://localhost:7860")
    print("=" * 60 + "\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
