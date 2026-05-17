from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Tuple


@dataclass(frozen=True)
class LoadedModel:
    name: str
    frontend: str
    model: Any
    input_name: str
    input_shape: Tuple[int, ...]
    input_dtype: str = "float32"
    labels: Sequence[str] | None = None


def load_pytorch_model(name: str) -> LoadedModel:
    import torch
    import torchvision.models as models

    normalized = name.lower()
    if normalized == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights).eval()
    elif normalized == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights).eval()
    elif normalized == "squeezenet1_1":
        weights = models.SqueezeNet1_1_Weights.DEFAULT
        model = models.squeezenet1_1(weights=weights).eval()
    elif normalized == "shufflenet_v2_x1_0":
        weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT
        model = models.shufflenet_v2_x1_0(weights=weights).eval()
    else:
        raise ValueError(
            "Unsupported PyTorch model. Choose resnet18, mobilenet_v2, "
            "squeezenet1_1, or shufflenet_v2_x1_0."
        )

    torch.set_grad_enabled(False)
    return LoadedModel(
        name=normalized,
        frontend="pytorch",
        model=model,
        input_name="input0",
        input_shape=(1, 3, 224, 224),
        labels=weights.meta.get("categories"),
    )


def load_tensorflow_model(name: str) -> LoadedModel:
    import tensorflow as tf

    normalized = name.lower()
    if normalized == "mobilenet_v2":
        model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=True)
    elif normalized == "resnet50":
        model = tf.keras.applications.ResNet50(weights="imagenet", include_top=True)
    else:
        raise ValueError("Unsupported TensorFlow model. Choose mobilenet_v2 or resnet50.")

    return LoadedModel(
        name=normalized,
        frontend="tensorflow",
        model=model,
        input_name="input_1",
        input_shape=(1, 224, 224, 3),
    )


def load_onnx_model(path: str | Path, input_name: str, input_shape: Tuple[int, ...]) -> LoadedModel:
    import onnx

    model_path = Path(path)
    model = onnx.load(str(model_path))
    return LoadedModel(
        name=model_path.stem,
        frontend="onnx",
        model=model,
        input_name=input_name,
        input_shape=input_shape,
    )


def load_tflite_model(path: str | Path, input_name: str, input_shape: Tuple[int, ...]) -> LoadedModel:
    model_path = Path(path)
    return LoadedModel(
        name=model_path.stem,
        frontend="tflite",
        model=model_path.read_bytes(),
        input_name=input_name,
        input_shape=input_shape,
    )
