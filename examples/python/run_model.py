from __future__ import annotations

import argparse
from pathlib import Path

from tvm_prep.preprocess import load_imagenet_image
from tvm_prep.runtime import configure_tvm_runtime, load_metadata, run_graph_executor, topk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exported TVM graph artifacts with Python.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--labels")
    parser.add_argument("--layout", choices=["NCHW", "NHWC"], default=None)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--topk", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    configure_tvm_runtime(artifact_dir)
    metadata = load_metadata(artifact_dir)
    shape = tuple(metadata["input_shape"])
    layout = args.layout or ("NCHW" if len(shape) == 4 and shape[1] == 3 else "NHWC")
    image = load_imagenet_image(args.image, shape, layout=layout, normalize=not args.no_normalize)
    output = run_graph_executor(artifact_dir, image)
    labels_path = Path(args.labels) if args.labels else artifact_dir / metadata.get("labels", "labels.txt")
    labels = labels_path.read_text(encoding="utf-8").splitlines() if labels_path.exists() else None

    for rank, (idx, prob, label) in enumerate(topk(output, labels, k=args.topk), start=1):
        print(f"{rank}: id={idx} p={prob:.6f} {label}")


if __name__ == "__main__":
    main()
