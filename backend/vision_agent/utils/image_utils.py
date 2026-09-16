"""
Image loading, resizing, conversion, and Base64 encoding utilities.
"""
from __future__ import annotations

import io
import base64
import uuid
from pathlib import Path
from PIL import Image

from vision_agent.config.settings import settings


def load_image_safely(source: str | Path | bytes | Image.Image) -> Image.Image:
    """
    Loads an image from a path, raw bytes, or returns the Image instance,
    ensuring it is in RGB format.
    """
    if isinstance(source, Image.Image):
        img = source
    elif isinstance(source, bytes):
        img = Image.open(io.BytesIO(source))
    else:
        img = Image.open(str(source))

    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        return background
    elif img.mode != "RGB":
        return img.convert("RGB")
    return img


def resize_image_if_needed(image: Image.Image, max_dim: int | None = None) -> Image.Image:
    """
    Resizes an image proportionally if its width or height exceeds max_dim.
    """
    max_dimension = max_dim or settings.MAX_IMAGE_DIMENSION
    width, height = image.size

    if width <= max_dimension and height <= max_dimension:
        return image

    if width > height:
        new_width = max_dimension
        new_height = int((height / width) * max_dimension)
    else:
        new_height = max_dimension
        new_width = int((width / height) * max_dimension)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def encode_image_to_base64(
    source: str | Path | bytes | Image.Image,
    max_dim: int | None = None,
    quality: int | None = None,
) -> str:
    """
    Converts image to JPEG-compressed Base64 string for Vision LLM consumption.
    """
    img = load_image_safely(source)
    img = resize_image_if_needed(img, max_dim)

    jpeg_quality = quality or settings.IMAGE_JPEG_QUALITY
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded


def save_temp_image(image: Image.Image, prefix: str = "temp") -> Path:
    """
    Saves an in-memory PIL Image to the temporary outputs directory.
    """
    temp_dir = settings.TEMP_DIR
    temp_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    target_path = temp_dir / filename
    image.save(target_path, format="PNG")
    return target_path
