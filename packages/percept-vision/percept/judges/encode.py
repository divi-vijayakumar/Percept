"""Frame images to the JPEG bytes a provider accepts."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

MAX_SIDE = 640
QUALITY = 80


def to_jpeg(image: np.ndarray, *, max_side: int = MAX_SIDE, quality: int = QUALITY) -> bytes:
    """Encode a ``uint8`` BGR (``HxWx3``) or grayscale (``HxW``) image as JPEG.

    Colour images are converted to RGB. The longest side is scaled down to ``max_side``
    pixels, keeping the aspect ratio; smaller images are not enlarged.
    """
    if image.dtype != np.uint8 or image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError("expected a uint8 HxW or HxWx3 (BGR) image")
    pil = Image.fromarray(np.ascontiguousarray(image[..., ::-1]) if image.ndim == 3 else image)
    longest = max(pil.size)
    if longest > max_side:
        scale = max_side / longest
        size = (max(1, round(pil.width * scale)), max(1, round(pil.height * scale)))
        pil = pil.resize(size, Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()
