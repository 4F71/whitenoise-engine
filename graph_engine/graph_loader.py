"""JSON patch dosyasını okuyup Graph nesnesi oluşturan yardımcı modül."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence, Tuple, Type

from graph_engine.graph_core import Graph
from graph_engine.graph_nodes import (
    BaseNode,
    EnvelopeNode,
    FilterNode,
    GainNode,
    MixNode,
    NoiseNode,
    OscillatorNode,
    OutputNode,
    PanNode,
)

NodeFactory = Mapping[str, Type[BaseNode]]

# Node tipi -> sınıf eşlemesi; yeni node eklemek için buraya mapleyin.
NODE_CLASS_MAP: NodeFactory = {
    "oscillator": OscillatorNode,
    "osc": OscillatorNode,
    "noise": NoiseNode,
    "filter": FilterNode,
    "mix": MixNode,
    "gain": GainNode,
    "envelope": EnvelopeNode,
    "env": EnvelopeNode,
    "pan": PanNode,
    "output": OutputNode,
}


def _normalize_type(type_name: str) -> str:
    return str(type_name).strip().lower()


def _create_node(
    cfg: Mapping[str, Any],
    sample_rate: int,
    node_classes: NodeFactory,
) -> Tuple[str, BaseNode, Dict[str, Any]]:
    """Tek bir node tanımını doğrular ve örneğini döndürür."""
    name = cfg.get("name")
    if not name:
        raise ValueError("Her node için 'name' alanı zorunludur.")

    type_name = cfg.get("type")
    if not type_name:
        raise ValueError(f"'{name}' nodu için 'type' alanı zorunludur.")

    cls = node_classes.get(_normalize_type(type_name))
    if cls is None:
        raise ValueError(f"Bilinmeyen node tipi: {type_name}")

    params: Dict[str, Any] = dict(cfg.get("params") or {})
    instance = cls(name=name, sample_rate=sample_rate)
    return name, instance, params


def _add_edges(graph: Graph, edges: Sequence[Mapping[str, Any]]) -> None:
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        target_input = edge.get("target_input", "input")
        if not source or not target:
            raise ValueError(f"Edge tanımı eksik: {edge}")
        graph.add_edge(source, target, target_input=target_input)


def load_graph_from_json(path: str | Path, node_classes: NodeFactory | None = None) -> Graph:
    """
    JSON patch dosyasını okuyup `Graph` nesnesi oluşturur.

    JSON şeması (önerilen):
        {
            "sample_rate": 44100,               # opsiyonel, varsayılan 44100
            "global_params": {...},             # opsiyonel, tüm node'lara yayılır
            "nodes": [
                {"name": "osc1", "type": "oscillator", "params": {"frequency": 440}},
                {"name": "out", "type": "output"}
            ],
            "edges": [
                {"source": "osc1", "target": "out", "target_input": "input"}
            ]
        }

    Parametreler:
        path: JSON patch dosyasının yolu.
        node_classes: Özel node tipi eşlemesi vermek isterseniz dict[str, Type[BaseNode]].

    Döndürür:
        Graph: Dosyada tanımlanan DAG yapısı.

    Hatalar:
        ValueError: Eksik alan, bilinmeyen node tipi veya geçersiz edge tanımı varsa.
    """
    patch_path = Path(path)
    data = json.loads(patch_path.read_text(encoding="utf-8"))

    node_classes = node_classes or NODE_CLASS_MAP
    sample_rate = int(data.get("sample_rate", 44_100))

    graph = Graph(sample_rate=sample_rate)

    global_params = data.get("global_params") or {}
    if global_params:
        graph.set_global_params(global_params)

    nodes_cfg: Sequence[Mapping[str, Any]] = data.get("nodes", [])
    if not nodes_cfg:
        raise ValueError("Patch dosyasında en az bir node tanımı bulunmalı.")

    for node_cfg in nodes_cfg:
        name, instance, params = _create_node(node_cfg, sample_rate, node_classes)
        graph.add_node(name, instance, params=params)

    edges_cfg: Sequence[Mapping[str, Any]] = data.get("edges", [])
    _add_edges(graph, edges_cfg)

    return graph
