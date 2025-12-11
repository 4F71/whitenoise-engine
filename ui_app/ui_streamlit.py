from __future__ import annotations

import io
import json
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import soundfile as sf
import streamlit as st

import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from graph_engine.graph_core import Graph
from graph_engine.graph_loader import NODE_CLASS_MAP
from graph_engine.graph_nodes import BaseNode
from preset_system.preset_library import PRESETS, PRESET_MAP
from preset_system.preset_schema import PresetConfig



# --------------------------------------------------------------------------- #
# Yardımcılar
# --------------------------------------------------------------------------- #

def _build_graph_from_patch(
    patch: Mapping[str, Any],
    node_classes: Mapping[str, type[BaseNode]] | None = None,
) -> Graph:
    """Patch sözlüğünden Graph nesnesi oluşturur."""
    node_classes = node_classes or NODE_CLASS_MAP
    sample_rate = int(patch.get("sample_rate", 44_100))
    graph = Graph(sample_rate=sample_rate)

    global_params = patch.get("global_params") or {}
    if global_params:
        graph.set_global_params(global_params)

    nodes_cfg: Iterable[Mapping[str, Any]] = patch.get("nodes") or []
    if not nodes_cfg:
        raise ValueError("Patch içinde hiç node yok.")

    for cfg in nodes_cfg:
        name = cfg.get("name")
        type_name = cfg.get("type")
        if not name or not type_name:
            raise ValueError(f"Node tanımı eksik: {cfg}")
        cls = node_classes.get(str(type_name).lower())
        if cls is None:
            raise ValueError(f"Bilinmeyen node tipi: {type_name}")
        params = dict(cfg.get("params") or {})
        graph.add_node(name, cls(name=name, sample_rate=sample_rate), params=params)

    for edge in patch.get("edges") or []:
        src = edge.get("source")
        tgt = edge.get("target")
        if not src or not tgt:
            raise ValueError(f"Geçersiz edge: {edge}")
        graph.add_edge(src, tgt, target_input=edge.get("target_input", "input"))

    return graph


def _render_preset_audio(
    preset: PresetConfig,
    duration: float,
    sample_rate: int,
    global_overrides: Mapping[str, Any] | None = None,
    node_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[np.ndarray, int]:
    """Patch'i Graph'a çevirip ses üretir."""
    graph = _build_graph_from_patch(preset.graph_patch)
    if global_overrides:
        graph.set_global_params({**graph.global_params, **dict(global_overrides)})

    num_frames = max(1, int(duration * sample_rate))
    audio = graph.run(
        {
            "duration": duration,
            "num_frames": num_frames,
            "sample_rate": sample_rate,
            "node_params": node_overrides or {},
        }
    )
    return audio, sample_rate


def _to_wav_buffer(audio: np.ndarray, sample_rate: int) -> io.BytesIO:
    """Stereo numpy dizisini WAV'e çevirip hafızada döndürür."""
    if audio.ndim != 2:
        raise ValueError("Ses çıktısı 2 boyutlu olmalı.")
    if audio.shape[0] == 2:
        stereo = audio.T
    elif audio.shape[1] == 2:
        stereo = audio
    else:
        raise ValueError("Ses çıktısı stereo değil.")

    buf = io.BytesIO()
    sf.write(buf, stereo.astype(np.float32), sample_rate, format="WAV")
    buf.seek(0)
    return buf


def _parse_json_field(label: str, raw_value: str) -> tuple[dict[str, Any], bool]:
    """JSON alanını parse eder, hata durumunda Streamlit uyarısı döner."""
    text = raw_value.strip()
    if not text:
        return {}, True
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("Sadece sözlük bekleniyor.")
        return parsed, True
    except Exception as exc:  # noqa: BLE001
        st.error(f"{label} JSON parse hatası: {exc}")
        return {}, False


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #


def _preset_options() -> list[tuple[str, str]]:
    return [(f"{p.name} · {p.id}", p.id) for p in sorted(PRESETS, key=lambda p: p.name.lower())]


def _unique_tags() -> list[str]:
    tags: set[str] = set()
    for p in PRESETS:
        tags.update(p.tags)
    return sorted(tags)


def main() -> None:
    st.set_page_config(page_title="UltraGen Preset Panel", layout="wide")
    st.title("UltraGen Preset Panel")
    st.caption("Preset seçimi · param kontrolü · preview · full render")

    # --- Sidebar: filtreler
    with st.sidebar:
        st.header("Preset Filtre")
        tag_filter = st.multiselect("Etiket ile filtrele", _unique_tags())
        search = st.text_input("İsim/ID ara", "").strip().lower()

    presets = []
    for label, pid in _preset_options():
        preset = PRESET_MAP[pid]
        if tag_filter and not any(t in preset.tags for t in tag_filter):
            continue
        if search and search not in preset.name.lower() and search not in preset.id.lower():
            continue
        presets.append((label, pid))

    if not presets:
        st.warning("Filtreye uyan preset bulunamadı.")
        return

    default_idx = 0
    labels = [lbl for lbl, _ in presets]
    ids = [pid for _, pid in presets]
    selected_label = st.selectbox("Preset seç", labels, index=default_idx)
    preset_id = ids[labels.index(selected_label)]
    preset = PRESET_MAP[preset_id]

    cols = st.columns([2, 1])
    with cols[0]:
        st.subheader(preset.name)
        st.write(f"ID: `{preset.id}`")
        st.write(f"Hedef kullanım: **{preset.target_use or 'belirtilmedi'}**")
        st.write(f"Etiketler: {', '.join(preset.tags) if preset.tags else '—'}")
        if preset.duration_hint:
            st.write(f"Önerilen süre: {preset.duration_hint:.1f} sn")
        st.json(preset.graph_patch, expanded=False)
    with cols[1]:
        st.info("Preview kısa süreli WAV, Full Render seçilen süre kadar üretir. Parametre değişiklikleri yalnızca oturumda geçerlidir.")

    # --- Parametre formu
    patch_sr = int(preset.graph_patch.get("sample_rate", 44_100))
    duration_default = float(preset.duration_hint or 8.0)
    duration_max = max(10.0, duration_default * 3)

    with st.form("render_form"):
        pcol1, pcol2, pcol3 = st.columns(3)
        sample_rate = pcol1.selectbox("Sample rate", [22_050, 44_100, 48_000, 96_000], index=[22_050, 44_100, 48_000, 96_000].index(patch_sr) if patch_sr in [22_050, 44_100, 48_000, 96_000] else 1)
        duration = pcol2.slider("Full render süresi (sn)", min_value=1.0, max_value=duration_max, value=duration_default, step=0.5)
        preview_duration = pcol3.slider("Preview süresi (sn)", min_value=1.0, max_value=min(duration, 12.0), value=min(6.0, duration_default), step=0.5)
        seed = pcol1.number_input("Seed", value=0, min_value=0, step=1)

        st.markdown("**Global param override (JSON, opsiyonel)** — örn: `{ \"duration\": 5.5 }`")
        global_raw = st.text_area("Global parametreler", placeholder="{}", height=100)

        st.markdown("**Node override (JSON, opsiyonel)** — örn: `{ \"osc\": {\"frequency\": 330}, \"gain\": {\"gain_db\": -3} }`")
        node_raw = st.text_area("Node parametreleri", placeholder="{\n  \"osc\": {\"frequency\": 330}\n}", height=140)

        preview_btn = st.form_submit_button("Preview üret")
        render_btn = st.form_submit_button("Full render üret")

    global_overrides, ok_global = _parse_json_field("Global param", global_raw)
    node_overrides, ok_node = _parse_json_field("Node param", node_raw)

    if preview_btn or render_btn:
        if not ok_global or not ok_node:
            st.stop()
        try:
            node_params = node_overrides
            graph_globals = {**global_overrides, "seed": int(seed)}

            target_duration = preview_duration if preview_btn else duration
            with st.spinner("Ses üretiliyor..."):
                audio, sr = _render_preset_audio(
                    preset,
                    duration=target_duration,
                    sample_rate=int(sample_rate),
                    global_overrides=graph_globals,
                    node_overrides=node_params,
                )
                wav_buf = _to_wav_buffer(audio, sr)
            st.success("Hazır!")
            st.audio(wav_buf, format="audio/wav")
            fname = f"{preset.id}_{'preview' if preview_btn else 'full'}_{int(target_duration)}s.wav"
            st.download_button("WAV indir", data=wav_buf, file_name=fname, mime="audio/wav")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Render sırasında hata: {exc}")


if __name__ == "__main__":
    main()
