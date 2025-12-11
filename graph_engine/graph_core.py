"""Node tabanli moduler ses graf cekirdegi.

Graph sinifi:
- DAG yapisini tutar, topolojik siralama ile isler.
- Node baglantilarini ve parametrelerini yonetir.
- buffer yönetimi yaparak node ciktilarini cache'ler.
- run(graph_params) cagrisi stereo (2, N) ndarray dondurur.

graph_params yapisi:
{
    "sample_rate": int,
    "duration": float,        # saniye
    "num_frames": int,
    "node_params": { "<node>": {...} },
    ...                       # global paramlar
}
Node param onceligi: global -> Graph.add_node sırasında verilen -> graph_params["node_params"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np

from graph_engine.graph_nodes import BaseNode, OutputNode


def _to_channels_first(signal: np.ndarray) -> np.ndarray:
    """Sinyali (kanal, num_frames) formatina getirir."""
    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim == 1:
        return arr[np.newaxis, :]
    if arr.ndim == 2:
        return arr if arr.shape[0] <= arr.shape[1] else arr.T
    raise ValueError("Beklenmeyen sinyal boyutu, 1D veya 2D bekleniyor.")


def _ensure_stereo(signal: np.ndarray) -> np.ndarray:
    """Mono veya tek kanalli sinyali stereo'ya donusturur; stereo ise dokunmaz."""
    ch_first = _to_channels_first(signal)
    if ch_first.shape[0] == 1:
        return np.vstack([ch_first, ch_first])
    if ch_first.shape[0] == 2:
        return ch_first
    raise ValueError("Stereo cikti icin sinyal 1 veya 2 kanalli olmalidir.")


def _mix_signals(signals: Sequence[np.ndarray]) -> np.ndarray:
    """Ayni isimli input'lar birden fazla kaynaktan geliyorsa pad + ortalama ile miksler."""
    if not signals:
        raise ValueError("Mikslenecek sinyal yok.")
    if len(signals) == 1:
        return signals[0]

    max_len = max(sig.shape[1] for sig in signals)
    padded: list[np.ndarray] = []
    for sig in signals:
        if sig.shape[1] < max_len:
            pad_width = ((0, 0), (0, max_len - sig.shape[1]))
            sig = np.pad(sig, pad_width)
        padded.append(sig)
    return sum(padded) / len(padded)


@dataclass(frozen=True)
class Edge:
    """Node'lar arasi yonlu baglanti."""
    source: str
    target: str
    target_input: str = "input"


@dataclass
class NodeEntry:
    """Graph icindeki tek bir node ve varsayilan paramlari."""
    name: str
    node: BaseNode
    params: Dict[str, Any] = field(default_factory=dict)


class Graph:
    """DAG tabanli moduler ses isleme grafi."""

    def __init__(self, sample_rate: int = 44_100) -> None:
        self.sample_rate = int(sample_rate)
        self.nodes: Dict[str, NodeEntry] = {}
        self.edges: list[Edge] = []
        self.global_params: Dict[str, Any] = {}

    # ---- Yapilandirma -------------------------------------------------

    def add_node(self, name: str, node: BaseNode, params: Mapping[str, Any] | None = None) -> None:
        """Graf'a yeni node ekler."""
        if name in self.nodes:
            raise ValueError(f"{name} zaten grafa eklenmis.")
        self.nodes[name] = NodeEntry(name=name, node=node, params=dict(params or {}))

    def add_edge(self, source: str, target: str, target_input: str = "input") -> None:
        """Iki node arasinda yonlu baglanti kurar."""
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("Baglanti icin hem kaynak hem hedef node mevcut olmali.")
        self.edges.append(Edge(source=source, target=target, target_input=target_input))

    def set_global_params(self, params: Mapping[str, Any]) -> None:
        """Tüm node'lara yayilacak global parametreleri ayarlar."""
        self.global_params = {**self.global_params, **dict(params)}

    # ---- Topoloji -----------------------------------------------------

    def topological_sort(self) -> list[str]:
        """DAG icin topolojik siralama dondurur; cycle varsa hata firlatir."""
        indegree: MutableMapping[str, int] = {name: 0 for name in self.nodes}
        adjacency: MutableMapping[str, list[str]] = {name: [] for name in self.nodes}
        for edge in self.edges:
            indegree[edge.target] += 1
            adjacency[edge.source].append(edge.target)

        queue: list[str] = [n for n, deg in indegree.items() if deg == 0]
        order: list[str] = []

        while queue:
            current = queue.pop(0)
            order.append(current)
            for neighbor in adjacency[current]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.nodes):
            raise ValueError("Graf ciklik iceriyor veya tum node'lar siralanamadi.")
        return order

    # ---- Calistirma ---------------------------------------------------

    def _edges_by_target(self) -> MutableMapping[str, list[Edge]]:
        mapping: MutableMapping[str, list[Edge]] = {name: [] for name in self.nodes}
        for edge in self.edges:
            mapping[edge.target].append(edge)
        return mapping

    def _merge_params(self, node_name: str, run_params: Mapping[str, Any]) -> Dict[str, Any]:
        """Param onceligi: global -> node varsayilani -> run.node_params -> run global alanlari."""
        merged: Dict[str, Any] = {}
        merged.update(self.global_params)
        merged.update(self.nodes[node_name].params)
        node_overrides = (run_params.get("node_params") or {}).get(node_name, {})
        merged.update(node_overrides)

        # duration / num_frames gibi global calisma paramlari
        for key in ("sample_rate", "duration", "num_frames"):
            if key in run_params and key not in merged:
                merged[key] = run_params[key]

        merged.setdefault("sample_rate", self.sample_rate)
        return merged

    def _gather_inputs(
        self,
        node_name: str,
        buffers: Mapping[str, np.ndarray],
        edges_by_target: Mapping[str, list[Edge]],
    ) -> Dict[str, np.ndarray]:
        """Hedef node icin girdi sozlugunu hazirlar."""
        collected: Dict[str, list[np.ndarray]] = {}
        for edge in edges_by_target.get(node_name, []):
            if edge.source not in buffers:
                raise RuntimeError(f"{edge.source} nodunun cikti buffer'i yok.")
            collected.setdefault(edge.target_input, []).append(buffers[edge.source])

        return {name: _mix_signals([_to_channels_first(sig) for sig in sigs]) for name, sigs in collected.items()}

    def _select_output_node(self, explicit: str | None) -> str:
        if explicit:
            if explicit not in self.nodes:
                raise ValueError(f"Istenen cikis nodu bulunamadi: {explicit}")
            return explicit

        for name, entry in self.nodes.items():
            if isinstance(entry.node, OutputNode):
                return name
        # yedek: son node
        return self.topological_sort()[-1]

    def run(self, graph_params: Mapping[str, Any] | None = None, output_node: str | None = None) -> np.ndarray:
        """
        Grafi topolojik siralama ile calistirir.

        Parametreler:
            graph_params: Calisma anda verilen global + node parametreleri.
            output_node: Spesifik bir node'un cikisini almak icin isim.

        Doner:
            Stereo sinyal (2, num_frames) numpy.ndarray.
        """
        if not self.nodes:
            raise ValueError("Graf bos, calistirilacak node yok.")

        run_params = graph_params or {}
        edges_by_target = self._edges_by_target()
        order = self.topological_sort()

        buffers: Dict[str, np.ndarray] = {}
        for name in order:
            inputs = self._gather_inputs(name, buffers, edges_by_target)
            params = self._merge_params(name, run_params)
            output = self.nodes[name].node.process(inputs, params)
            buffers[name] = _to_channels_first(np.asarray(output))

        selected = self._select_output_node(output_node)
        if selected not in buffers:
            raise RuntimeError(f"{selected} icin cikti bulunamadi.")
        return _ensure_stereo(buffers[selected])

    # ---- Yardimci -----------------------------------------------------

    def summarize(self) -> str:
        """Graf topolojisini kisaca metin olarak dondurur."""
        lines = []
        for edge in self.edges:
            lines.append(f"{edge.source} -> {edge.target} [{edge.target_input}]")
        return "\n".join(lines)
