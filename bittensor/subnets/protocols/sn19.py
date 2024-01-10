import base64
from typing import List, Optional, Tuple, Union
from core import dataclasses as dc, constants as cst
import bittensor as bt
from pydantic import Field, root_validator, validator
import random
from enum import Enum


class IsAlive(bt.Synapse):
    answer: Optional[str] = None

    def deserialize(self) -> Optional[str]:
        return self.answer


class GenerateImagesFromText(bt.Synapse):
    """Generates an image from a text prompt"""

    text_prompts: List[dc.TextPrompt] = Field([], description="Prompts for the image generation", title="text_prompts")
    cfg_scale: int = Field(cst.DEFAULT_CFG_SCALE, description="Scale for the configuration")
    height: int = Field(cst.DEFAULT_HEIGHT, description="Height of the generated image")
    width: int = Field(cst.DEFAULT_WIDTH, description="Width of the generated image")
    samples: int = Field(cst.DEFAULT_SAMPLES, description="Number of sample images to generate")
    steps: int = Field(cst.DEFAULT_STEPS, description="Number of steps in the image generation process")
    style_preset: str = Field(cst.DEFAULT_STYLE_PRESET, description="Preset style for the image")
    seed: int = Field(default=random.randint(1, cst.LARGEST_SEED), description="Random seed for generating the image")

    image_b64s: Optional[List[str]] = Field(None, description="The base64 encoded images to return", title="image_b64s")

    def deserialize(self) -> Optional[List[str]]:
        return self.image_b64s


class GenerateImagesFromImage(bt.Synapse):
    """Generates an image from an image (and text) prompt"""

    init_image: Optional[str] = Field(..., description="The base64 encoded image", title="init_image")
    text_prompts: List[dc.TextPrompt] = Field([], description="Prompts for the image generation", title="text_prompts")
    image_strength: float = Field(0.0, description="The strength of the init image")
    cfg_scale: float = Field(cst.DEFAULT_CFG_SCALE, description="Scale for the configuration")
    samples: int = Field(cst.DEFAULT_SAMPLES, description="Number of sample images to generate")
    steps: int = Field(cst.DEFAULT_STEPS, description="Number of steps in the image generation process")
    sampler: Optional[str] = Field(None, description="The sampler to use for image generation")
    style_preset: str = Field(cst.DEFAULT_STYLE_PRESET, description="Preset style for the image")
    seed: int = Field(default=random.randint(1, cst.LARGEST_SEED), description="Random seed for generating the image")
    init_image_mode: Optional[str] = Field("IMAGE_STRENGTH", description="The mode of the init image")

    image_b64s: Optional[List[str]] = Field(None, description="The base64 encoded images to return", title="image_b64s")

    def deserialize(self) -> Optional[List[str]]:
        return self.image_b64s


class MaskSource(str, Enum):
    MASK_IMAGE_WHITE = "MASK_IMAGE_WHITE"
    MASK_IMAGE_BLACK = "MASK_IMAGE_BLACK"
    INIT_IMAGE_ALPHA = "INIT_IMAGE_ALPHA"


class GenerateImagesFromInpainting(bt.Synapse):
    """Generates an image from an image (and text) prompt"""

    init_image: Optional[str] = Field(..., description="The base64 encoded image", title="init_image")
    text_prompts: List[dc.TextPrompt] = Field([], description="Prompts for the image generation", title="text_prompts")
    mask_source: Optional[MaskSource] = Field(None, description="The base64 encoded mask", title="mask_source")
    mask_image: Optional[str] = Field(None, description="The base64 encoded mask", title="mask_source")
    cfg_scale: int = Field(cst.DEFAULT_CFG_SCALE, description="Scale for the configuration")
    samples: int = Field(cst.DEFAULT_SAMPLES, description="Number of sample images to generate")
    steps: int = Field(cst.DEFAULT_STEPS, description="Number of steps in the image generation process")
    sampler: Optional[str] = Field(None, description="The sampler to use for image generation")
    style_preset: str = Field(cst.DEFAULT_STYLE_PRESET, description="Preset style for the image")
    seed: int = Field(default=random.randint(1, cst.LARGEST_SEED), description="Random seed for generating the image")

    image_b64s: Optional[List[str]] = Field(None, description="The base64 encoded images to return", title="image_b64s")

    def deserialize(self) -> Optional[List[str]]:
        return self.image_b64s


class UpscaleImage(bt.Synapse):
    image: Optional[str] = Field(..., description="The base64 encoded image", title="image")
    height: Optional[int] = Field(None, description="Height of the upscaled image")
    width: Optional[int] = Field(None, description="Width of the upscaled image")

    image_b64s: Optional[List[str]] = Field(None, description="The base64 encoded images to return", title="image_b64s")


class ClipEmbeddingImages(bt.Synapse):
    """Generates a clip embedding for images"""

    image_b64s: Optional[List[str]] = Field(None, description="The base64 encoded images", title="images")
    image_embeddings: Optional[List[List[float]]] = Field(
        default=None, description="The image embeddings", title="image_embeddings"
    )
    error_message: Optional[str] = Field(None, description="The error message", title="error_message")

    @validator("image_b64s", pre=True)
    def check_number_of_images(cls, values):
        if values is not None and len(values) > 10:
            raise ValueError("Number of images should not exceed 10 please")
        return values

    @root_validator(pre=True)
    def check_total_image_size(cls, values):
        if values is not None:
            max_size_mb = 10
            total_size_mb = 0
            image_b64s = values.get("image_b64s", [])
            if image_b64s:
                total_size_mb = sum((len(base64.b64decode(img)) for img in image_b64s)) / (1024 * 1024)
            if total_size_mb > max_size_mb:
                raise ValueError(f"Total image size should not exceed {max_size_mb} MB, we are not made of bandwidth")
        return values

    def deserialize(self) -> Optional[List[List[float]]]:
        return self.image_embeddings


class ClipEmbeddingTexts(bt.Synapse):
    text_prompts: Optional[List[str]] = Field(None, description="The text prompts", title="text_prompts")

    text_embeddings: Optional[List[List[float]]] = Field(
        default=None, description="The text embeddings", title="text_embeddings"
    )
    error_message: Optional[str] = Field(None, description="The error message", title="error_message")

    def deserialize(self) -> Optional[List[List[float]]]:
        return self.text_embeddings


class SegmentingSynapse(bt.Synapse):
    """
    Segment according the given points and boxes for a given image embedding.

    If you dont have the image uuid yet, it will be generated from the base64 you give us :)
    """

    image_uuid: Optional[str] = Field(None, description="The UUID for the image to be segmented", title="embedding")

    image_b64: Optional[str] = Field(None, description="The base64 encoded image", title="image")
    error_message: Optional[str] = Field(
        None,
        description="Details about any error that may have occurred",
        title="success",
    )

    text_prompt: Optional[List[float]] = Field(
        default=None,
        description="The text prompt for the thing you wanna segment in the image",
        title="text_prompt",
    )

    input_points: Optional[List[List[Union[float, int]]]] = Field(
        default=None,
        description="The json encoded points for the image",
        title="points",
    )
    input_labels: Optional[List[int]] = Field(
        default=None,
        description="The labels of the points. 1 for a positive point, 0 for a negative",
        title="labels",
    )
    input_boxes: Optional[Union[List[List[Union[int, float]]], List[Union[int, float]]]] = Field(
        default=None,
        description="The boxes for the image. For now, we only accept one",
        title="boxes",
    )

    image_shape: Optional[List[int]] = Field(
        default=None,
        description="The shape of the image (y_dim, x_dim)",
        title="image_shape",
    )

    masks: Optional[List[List[List[int]]]] = Field(
        default=None,
        description="The json encoded RLE masks",
        title="masks",
    )

    def deserialize(self) -> Tuple[Optional[List[List[List[int]]]], Optional[List[int]]]:
        """
        Deserialize the emebeddings response with masks and image shape
        """
        return self.masks, self.image_shape
