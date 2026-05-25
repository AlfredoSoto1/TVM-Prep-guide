#!/usr/bin/env python3
"""Compile a model with Apache TVM.

This file intentionally stays flat and explicit. It shows the normal TVM flow:
load a model, import it into Relay, compile for a target, and write graph
executor artifacts.

Usage:
    python compilation/compile.py --frontend pytorch --model resnet18 --target x86_64
    python compilation/compile.py --frontend pytorch --model resnet18 --target raspi4_aarch64
    python compilation/compile.py --frontend onnx \\
        --model-path model.onnx --input-name input0 --input-shape 1,3,224,224 \\
        --labels labels.txt --target raspi4_aarch64

Artifacts written to examples/artifacts/<model>/<target>/:
    model.so / model.wasm / model.tar  - compiled target library
    model.json                          - graph executor graph
    model.params                        - serialized parameters
    metadata.json                       - input name, shape, target, library filename
    labels.txt                          - label list, when the model provides one
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from targets import TARGETS


# Relay import helpers.

def import_pytorch(model_name: str, input_name: str, input_shape: tuple):
    """Trace a torchvision model and import it into Relay."""
    import torch
    import torchvision.models as models
    from tvm import relay

    registry = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
        "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT),
        "squeezenet1_1": (models.squeezenet1_1, models.SqueezeNet1_1_Weights.DEFAULT),
        "shufflenet_v2_x1_0": (
            models.shufflenet_v2_x1_0,
            models.ShuffleNet_V2_X1_0_Weights.DEFAULT,
        ),
    }
    if model_name not in registry:
        raise SystemExit(f"Unknown PyTorch model '{model_name}'. Known: {', '.join(registry)}")

    fn, weights = registry[model_name]
    model = fn(weights=weights).eval()
    torch.set_grad_enabled(False)

    example = torch.zeros(input_shape, dtype=torch.float32)
    scripted = torch.jit.trace(model, example).eval()
    mod, params = relay.frontend.from_pytorch(scripted, [(input_name, input_shape)])
    labels = list(weights.meta.get("categories", []))
    return mod, params, labels


def import_tensorflow(model_name: str, input_name: str, input_shape: tuple):
    """Load a Keras application model and import it into Relay."""
    import tensorflow as tf
    from tvm import relay

    if model_name == "mobilenet_v2":
        model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=True)
    elif model_name == "resnet50":
        model = tf.keras.applications.ResNet50(weights="imagenet", include_top=True)
    else:
        raise SystemExit(f"Unknown TF model '{model_name}'. Known: mobilenet_v2, resnet50")

    mod, params = relay.frontend.from_keras(model, shape={input_name: input_shape})
    return mod, params, []


def import_onnx(model_path: str, input_name: str, input_shape: tuple):
    """Load an ONNX file and import it into Relay."""
    import onnx
    from tvm import relay

    model = onnx.load(model_path)
    mod, params = relay.frontend.from_onnx(
        model, shape={input_name: input_shape}, freeze_params=True
    )
    return mod, params, []


def import_tflite(model_path: str, input_name: str, input_shape: tuple, input_dtype: str):
    """Load a TFLite flatbuffer and import it into Relay."""
    from tvm import relay

    try:
        import tflite as tflite_lib
    except ImportError as exc:
        raise SystemExit("Install the 'tflite' package to use the tflite frontend") from exc

    with open(model_path, "rb") as f:
        buf = f.read()
    tflite_model = tflite_lib.Model.GetRootAsModel(buf, 0)
    mod, params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={input_name: input_shape},
        dtype_dict={input_name: input_dtype},
    )
    return mod, params, []


# Build and save.

def build_and_save(
    mod,
    params,
    model_name: str,
    target_name: str,
    output_dir: str | Path,
    opt_level: int = 3,
    labels: list[str] | None = None,
    input_name: str = "input0",
    input_shape: tuple = (1, 3, 224, 224),
    input_dtype: str = "float32",
) -> Path:
    """Compile a Relay module and write graph executor artifacts."""
    import tvm
    from tvm import relay

    if target_name not in TARGETS:
        raise ValueError(f"Unknown target '{target_name}'. Known: {', '.join(TARGETS)}")

    profile = TARGETS[target_name]
    target_str = profile["target"]
    host_str = profile.get("host")
    cc = profile.get("cc")
    ext = profile["ext"]

    if host_str:
        target = tvm.target.Target(target_str, host=host_str)
    else:
        target = tvm.target.Target(target_str)

    with tvm.transform.PassContext(opt_level=opt_level):
        lib = relay.build(mod, target=target, params=params)

    artifact_dir = Path(output_dir) / model_name / target_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    lib_path = artifact_dir / f"model.{ext}"
    if cc:
        lib.export_library(str(lib_path), cc=cc)
    else:
        lib.export_library(str(lib_path))

    (artifact_dir / "model.json").write_text(lib.get_graph_json(), encoding="utf-8")
    (artifact_dir / "model.params").write_bytes(relay.save_param_dict(lib.get_params()))

    metadata = {
        "model": model_name,
        "input_name": input_name,
        "input_shape": list(input_shape),
        "input_dtype": input_dtype,
        "target": target_name,
        "target_str": target_str,
        "library": lib_path.name,
    }
    if labels:
        (artifact_dir / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
        metadata["labels"] = "labels.txt"

    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return artifact_dir


# CLI.


def parse_shape(text: str) -> tuple[int, ...]:
    """Parse a comma-separated shape, for example '1,3,224,224'."""
    try:
        shape = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid shape: {text}") from exc
    if not shape:
        raise argparse.ArgumentTypeError("Input shape cannot be empty")
    return shape


def print_targets() -> None:
    """Print target profiles in the same form TVM receives them."""
    for name, profile in sorted(TARGETS.items()):
        print(name)
        print(f"  target: {profile['target']}")
        if "host" in profile:
            print(f"  host:   {profile['host']}")
        if "cc" in profile:
            print(f"  cc:     {profile['cc']}")
        print(f"  output: model.{profile['ext']}")


def main():
    parser = argparse.ArgumentParser(
        description="Compile a model with Apache TVM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--frontend", choices=["pytorch", "tensorflow", "onnx", "tflite"])
    parser.add_argument("--model", default="resnet18", help="Named model for pytorch/tensorflow.")
    parser.add_argument("--model-path", default=None, help="Path to .onnx or .tflite file.")
    parser.add_argument("--labels", default=None, help="Optional text file with one output label per line.")
    parser.add_argument("--input-name", default="input0")
    parser.add_argument("--input-shape", default=(1, 3, 224, 224), type=parse_shape)
    parser.add_argument("--input-dtype", default="float32")
    parser.add_argument("--target", default="x86_64", choices=list(TARGETS))
    parser.add_argument("--opt-level", default=3, type=int)
    parser.add_argument("--output-dir", default="examples/artifacts")
    parser.add_argument("--list-targets", action="store_true", help="Print target profiles and exit.")
    args = parser.parse_args()

    if args.list_targets:
        print_targets()
        return

    if not args.frontend:
        parser.error("--frontend is required unless --list-targets is used")

    if args.frontend == "pytorch":
        mod, params, labels = import_pytorch(args.model, args.input_name, args.input_shape)
        model_name = args.model
    elif args.frontend == "tensorflow":
        mod, params, labels = import_tensorflow(args.model, args.input_name, args.input_shape)
        model_name = args.model
    elif args.frontend == "onnx":
        if not args.model_path:
            raise SystemExit("--model-path is required for the onnx frontend")
        mod, params, labels = import_onnx(args.model_path, args.input_name, args.input_shape)
        model_name = Path(args.model_path).stem
    elif args.frontend == "tflite":
        if not args.model_path:
            raise SystemExit("--model-path is required for the tflite frontend")
        mod, params, labels = import_tflite(
            args.model_path, args.input_name, args.input_shape, args.input_dtype
        )
        model_name = Path(args.model_path).stem

    if args.labels:
        labels = Path(args.labels).read_text(encoding="utf-8").splitlines()

    artifact_dir = build_and_save(
        mod,
        params,
        model_name,
        args.target,
        args.output_dir,
        opt_level=args.opt_level,
        labels=labels or None,
        input_name=args.input_name,
        input_shape=args.input_shape,
        input_dtype=args.input_dtype,
    )
    print(f"Artifacts written to {artifact_dir}")


if __name__ == "__main__":
    main()
