"""
ui_app/ui_streamlit.py

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


def convert_to_wav_bytes(audio_data: np.ndarray, sample_rate: int) -> bytes:
    """NumPy dizisini bellekte WAV formatına çevirir."""
    audio_data = np.clip(audio_data, -1.0, 1.0)
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
        options=["V2 (102 ML-generated)", "V1 (14 handcrafted)"],
        index=0
    )
    is_v2 = preset_version.startswith("V2")
    st.sidebar.markdown("---")

    duration = st.sidebar.number_input(
        "Süre (Saniye)",
        min_value=1.0,
        max_value=36000.0,
        value=10.0,
        step=1.0
    )

    sr = st.sidebar.selectbox(
        "Örnekleme Hızı (Hz)",
        options=[44100, 48000],
        index=0
    )

    # Variant sadece V1'de göster
    use_variant = False
    variant_intensity = 0.15
    if not is_v2:
        use_variant = st.sidebar.checkbox("Otomatik Varyasyon Üret", value=False)
        if use_variant:
            variant_intensity = st.sidebar.slider("Varyasyon Şiddeti", 0.0, 0.5, 0.15)

    # 2. Preset Seçimi
    st.subheader("1. Preset Seçimi")

    if is_v2:
        # V2: ML-generated presets
        from preset_system.preset_library import list_v2_presets, get_v2_preset
        v2_presets = list_v2_presets()
        preset_names = [p["name"] for p in v2_presets]

        selected_index = st.selectbox(
            "V2 Preset Seç (102 ML-generated):",
            range(len(v2_presets)),
            format_func=lambda x: preset_names[x]
        )
        selected_preset = get_v2_preset(v2_presets[selected_index]["path"])
        selected_preset_id = v2_presets[selected_index]["name"]
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


    with st.expander("Preset Detayları", expanded=True):
        st.markdown(f"**ID:** `{selected_preset_id}`")
        st.markdown(f"**Açıklama:** {selected_preset.description}")
        st.markdown(f"**Etiketler:** {', '.join(selected_preset.tags)}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Katman Sayısı", len(selected_preset.layers))
        with col2:
            rev_mix = float(selected_preset.fx_config.reverb_mix)
            st.metric("Reverb", f"%{int(rev_mix * 100)}")

    # 3. Render
    st.markdown("---")
    st.subheader("2. Ses Üretimi")

    if st.button("RENDER BAŞLAT", type="primary", use_container_width=True):
        status_text = st.empty()
        progress_bar = st.progress(0)

        try:
            render_config = selected_preset

            if use_variant:
                status_text.text("Varyasyon hesaplanıyor...")
                render_config = preset_autogen.generate_variant(
                    selected_preset,
                    intensity=variant_intensity,
                    suffix="Streamlit"
                )
                time.sleep(0.5)

            status_text.text(f"DSP Motoru Çalışıyor... ({duration} sn)")
            progress_bar.progress(30)

            start_time = time.time()

            # === DEBUG FIX: Preset → Adapter → DSP ===
            generators = adapt_preset_to_layer_generators(render_config)

            audio_data = dsp_render.render_sound(
                generators,
                duration_sec=duration,
                sample_rate=sr
            )

            elapsed = time.time() - start_time
            progress_bar.progress(100)
            status_text.success(f"Tamamlandı! ({elapsed:.2f}s)")

            # 4. Sonuç
            st.markdown("---")
            st.subheader("3. Sonuç")

            wav_bytes = convert_to_wav_bytes(audio_data, sr)
            st.audio(wav_bytes, format="audio/wav")

            version_prefix = "v2_" if is_v2 else "v1_"
            file_name = f"xxxdsp_{version_prefix}{selected_preset_id}_{int(time.time())}.wav"
            st.download_button(
                label="📥 WAV Olarak İndir",
                data=wav_bytes,
                file_name=file_name,
                mime="audio/wav",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Render sırasında hata oluştu: {e}")
            progress_bar.empty()


if __name__ == "__main__":
    main()
