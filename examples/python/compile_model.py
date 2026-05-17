from __future__ import annotations

import argparse
from pathlib import Path

from tvm_prep.compile import build_and_export
from tvm_prep.model_zoo import (
    load_onnx_model,
    load_pytorch_model,
    load_tensorflow_model,
    load_tflite_model,
)
from tvm_prep.targets import get_target_profile, list_target_profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile a model with Apache TVM.")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--frontend", choices=["pytorch", "tensorflow", "onnx", "tflite"], default="pytorch")
    parser.add_argument("--model-path", help="Required for ONNX and TFLite imports.")
    parser.add_argument("--input-name", default="input0")
    parser.add_argument("--input-shape", default="1,3,224,224")
    parser.add_argument("--target-profile", choices=list_target_profiles(), default="x86_64")
    parser.add_argument("--target", help="Raw TVM target string override.")
    parser.add_argument("--host", help="Raw TVM host target override.")
    parser.add_argument("--output-dir", default="examples/artifacts")
    parser.add_argument("--opt-level", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frontend == "pytorch":
        loaded = load_pytorch_model(args.model)
    elif args.frontend == "tensorflow":
        loaded = load_tensorflow_model(args.model)
    else:
        if not args.model_path:
            raise SystemExit("--model-path is required for ONNX and TFLite frontends")
        shape = tuple(int(part) for part in args.input_shape.split(","))
        if args.frontend == "onnx":
            loaded = load_onnx_model(args.model_path, args.input_name, shape)
        else:
            loaded = load_tflite_model(args.model_path, args.input_name, shape)

    profile = get_target_profile(args.target_profile)
    artifact_dir = build_and_export(
        loaded,
        profile,
        Path(args.output_dir),
        opt_level=args.opt_level,
        override_target=args.target,
        override_host=args.host,
    )
    print(f"Artifacts written to {artifact_dir}")


if __name__ == "__main__":
    main()
