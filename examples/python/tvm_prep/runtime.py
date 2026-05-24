from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tvm
from tvm.contrib import graph_executor


def load_metadata(artifact_dir: str | Path) -> dict[str, Any]:
    """Read metadata written by compilation/compile.py."""
    metadata_path = Path(artifact_dir) / "metadata.json"
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def run_graph_executor(artifact_dir: str | Path, input_data: np.ndarray, device=None) -> np.ndarray:
    """Load graph-executor artifacts and run one inference."""
    artifact_path = Path(artifact_dir)
    metadata = load_metadata(artifact_path)
    lib_name = metadata.get("library", "model.so")
    graph_json = (artifact_path / "model.json").read_text(encoding="utf-8")
    params = (artifact_path / "model.params").read_bytes()
    lib = tvm.runtime.load_module(str(artifact_path / lib_name))
    dev = device or tvm.cpu(0)
    module = graph_executor.create(graph_json, lib, dev)
    module.load_params(params)
    module.set_input(metadata["input_name"], input_data)
    module.run()
    return module.get_output(0).numpy()


def topk(logits: np.ndarray, labels: list[str] | None = None, k: int = 5):
    """Return the top-k softmax probabilities from a model output tensor."""
    flat = logits.reshape(-1)
    exp = np.exp(flat - np.max(flat))
    probs = exp / exp.sum()
    indices = np.argsort(probs)[::-1][:k]
    result = []
    for idx in indices:
        label = labels[idx] if labels and idx < len(labels) else "<no-label>"
        result.append((int(idx), float(probs[idx]), label))
    return result
