from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype="float32")


def load_imagenet_image(
    image_path: str | Path,
    input_shape: Iterable[int] = (1, 3, 224, 224),
    layout: str = "NCHW",
    normalize: bool = True,
) -> np.ndarray:
    """Load an image with the preprocessing expected by ImageNet classifiers."""
    shape = tuple(input_shape)
    if layout not in {"NCHW", "NHWC"}:
        raise ValueError("layout must be NCHW or NHWC")

    height, width = (shape[2], shape[3]) if layout == "NCHW" else (shape[1], shape[2])
    image = Image.open(image_path).convert("RGB")
    image = _resize_shortest_side(image, 256)
    image = _center_crop(image, (width, height))
    data = np.asarray(image).astype("float32") / 255.0

    if normalize:
        data = (data - IMAGENET_MEAN) / IMAGENET_STD

    if layout == "NCHW":
        data = np.transpose(data, (2, 0, 1))[None, ...]
    else:
        data = data[None, ...]
    return data.astype("float32")


def _resize_shortest_side(image: Image.Image, shortest: int) -> Image.Image:
    width, height = image.size
    if width < height:
        new_size = (shortest, int(height * shortest / width))
    else:
        new_size = (int(width * shortest / height), shortest)
    return image.resize(new_size)


def _center_crop(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    width, height = image.size
    crop_width, crop_height = size
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height))
