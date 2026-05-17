from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import tvm
from tvm import relay

from .model_zoo import LoadedModel
from .targets import TargetProfile


def import_to_relay(loaded: LoadedModel):
    if loaded.frontend == "pytorch":
        import torch

        example_input = torch.randn(*loaded.input_shape)
        scripted = torch.jit.trace(loaded.model, example_input).eval()
        shape_list = [(loaded.input_name, loaded.input_shape)]
        return relay.frontend.from_pytorch(scripted, shape_list)

    if loaded.frontend == "tensorflow":
        return relay.frontend.from_keras(loaded.model, shape={loaded.input_name: loaded.input_shape})

    if loaded.frontend == "onnx":
        return relay.frontend.from_onnx(
            loaded.model,
            shape={loaded.input_name: loaded.input_shape},
            freeze_params=True,
        )

    if loaded.frontend == "tflite":
        try:
            import tflite
        except ImportError as exc:
            raise RuntimeError("TFLite import requires the `tflite` Python package.") from exc

        tflite_model = tflite.Model.GetRootAsModel(loaded.model, 0)
        return relay.frontend.from_tflite(
            tflite_model,
            shape_dict={loaded.input_name: loaded.input_shape},
            dtype_dict={loaded.input_name: loaded.input_dtype},
        )

    raise ValueError(f"Unsupported frontend: {loaded.frontend}")


def build_and_export(
    loaded: LoadedModel,
    profile: TargetProfile,
    output_root: str | Path,
    opt_level: int = 3,
    override_target: Optional[str] = None,
    override_host: Optional[str] = None,
) -> Path:
    mod, params = import_to_relay(loaded)
    target = override_target or profile.target
    host = override_host if override_host is not None else profile.host
    artifact_dir = Path(output_root) / loaded.name / profile.name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with tvm.transform.PassContext(opt_level=opt_level):
        lib = relay.build(mod, target=target, target_host=host, params=params)

    graph_json = lib.get_graph_json()
    params_bytes = relay.save_param_dict(lib.get_params())
    lib_path = artifact_dir / f"model.{profile.output_format}"

    if profile.output_format == "tar":
        lib.export_library(str(lib_path))
    elif profile.cc:
        lib.export_library(str(lib_path), cc=profile.cc)
    else:
        lib.export_library(str(lib_path))

    (artifact_dir / "model.json").write_text(graph_json, encoding="utf-8")
    (artifact_dir / "model.params").write_bytes(params_bytes)
    metadata = {
        "model": loaded.name,
        "frontend": loaded.frontend,
        "input_name": loaded.input_name,
        "input_shape": list(loaded.input_shape),
        "input_dtype": loaded.input_dtype,
        "target_profile": profile.name,
        "target": target,
        "host": host,
        "library": lib_path.name,
    }
    if loaded.labels:
        (artifact_dir / "labels.txt").write_text("\n".join(loaded.labels) + "\n", encoding="utf-8")
        metadata["labels"] = "labels.txt"
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return artifact_dir
