from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def load_metadata(artifact_dir: str | Path) -> dict[str, Any]:
    """Read metadata written by compilation/compile.py."""
    metadata_path = Path(artifact_dir) / "metadata.json"
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def configure_tvm_runtime(artifact_dir: str | Path) -> None:
    """Use the packaged TVM runtime next to the compiled model artifacts."""
    artifact_path = Path(artifact_dir).resolve()
    metadata = load_metadata(artifact_path)
    runtime_dir = _find_runtime_dir(artifact_path, metadata)

    if runtime_dir:
        os.environ["TVM_LIBRARY_PATH"] = str(runtime_dir)
        os.environ["TVM_USE_RUNTIME_LIB"] = "1"
        python_dir = runtime_dir / "python"
        if python_dir.exists() and str(python_dir) not in sys.path:
            sys.path.insert(0, str(python_dir))

    for python_path in _candidate_tvm_python_paths():
        if python_path.exists() and str(python_path) not in sys.path:
            sys.path.append(str(python_path))


def _find_runtime_dir(artifact_dir: Path, metadata: dict[str, Any]) -> Path | None:
    target = metadata.get("target")
    artifacts_root = artifact_dir.parent.parent
    candidates = []

    if target:
        candidates.append(artifacts_root / "runtime" / target)
    candidates.extend([
        artifact_dir,
        artifact_dir.parent,
    ])

    for candidate in candidates:
        if (candidate / "libtvm_runtime.so").exists():
            return candidate.resolve()
    return None


def _candidate_tvm_python_paths() -> list[Path]:
    examples_dir = Path(__file__).resolve().parents[2]
    repo_root = examples_dir.parent
    home = Path.home()
    return [
        repo_root / "tvm" / "python",
        repo_root.parent / "tvm" / "python",
        home / "tvm" / "python",
        Path("/opt/tvm/python"),
        Path("/usr/local/tvm/python"),
    ]


def _import_tvm(artifact_dir: str | Path):
    configure_tvm_runtime(artifact_dir)
    try:
        import tvm
        from tvm.contrib import graph_executor
    except Exception as exc:
        raise RuntimeError(
            "Could not import TVM Python. Copy the runtime package produced by "
            "compilation/build_runtime.sh, install TVM Python on this device, "
            "or keep TVM source at ~/tvm or /opt/tvm."
        ) from exc

    return tvm, graph_executor


def run_graph_executor(artifact_dir: str | Path, input_data: np.ndarray, device=None) -> np.ndarray:
    """Load graph-executor artifacts and run one inference."""
    artifact_path = Path(artifact_dir)
    tvm, graph_executor = _import_tvm(artifact_path)
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
