"""
Image preparation, resolution normalization, and format conversion.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union
from PIL import Image

from vision_agent.config.settings import settings
from vision_agent.utils.image_utils import load_image_safely, resize_image_if_needed, encode_image_to_base64
from vision_agent.utils.logging_utils import get_logger

logger = get_logger("image_processor")


class ImageProcessor:
    def __init__(self, max_dimension: int | None = None):
        self.max_dimension = max_dimension or settings.MAX_IMAGE_DIMENSION

    def prepare_image(self, source: Union[str, Path, bytes, Image.Image]) -> tuple[Image.Image, str]:
        """
        Loads image, resizes if larger than maximum dimension, and converts to Base64.
        Returns (PIL.Image, base64_string).
        """
        img = load_image_safely(source)
        orig_size = img.size
        resized_img = resize_image_if_needed(img, self.max_dimension)

        if resized_img.size != orig_size:
            logger.debug(f"Resized image from {orig_size} to {resized_img.size}")

        b64_str = encode_image_to_base64(resized_img, self.max_dimension)
        return resized_img, b64_str
