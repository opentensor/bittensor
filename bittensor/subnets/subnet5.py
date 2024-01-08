# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2022-2023 Opentensor Foundation
# Copyright © 2023 Opentensor Technologies Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import typing
import pydantic
import bittensor as bt
from typing import Literal


def validate_synapse(synapse, enforce_batch_size=False) -> (bool, str):
    # check tensor type first
    if not all([isinstance(image, bt.Tensor) for image in synapse.images]):
        return (False, "images are not tensors")
    # check number of images
    if len(synapse.images) != synapse.num_images_per_prompt and enforce_batch_size:
        return (False, "number of images does not match num_images_per_prompt")
    # check image size
    if not all(
        [
            image.shape[1] == synapse.height and image.shape[2] == synapse.width
            for image in synapse.images
        ]
    ):
        return (False, "image size does not match height and width")
    # if all checks pass, return True, ""
    return (True, "")


class TextToImage(bt.Synapse):
    images: list[bt.Tensor] = []
    text: str = pydantic.Field(..., allow_mutation=False)
    negative_prompt: str = pydantic.Field(..., allow_mutation=False)
    height: int = pydantic.Field(512, allow_mutation=False)
    width: int = pydantic.Field(512, allow_mutation=False)
    num_images_per_prompt: int = pydantic.Field(1, allow_mutation=False)
    seed: int = pydantic.Field(-1, allow_mutation=False)
    nsfw_allowed: bool = pydantic.Field(False, allow_mutation=False)
    required_hash_fields: list[str] = pydantic.Field(
        [
            "text",
            "negative_prompt",
            "height",
            "width",
            "num_images_per_prompt",
            "seed",
            "nsfw_allowed",
        ],
        allow_mutation=False,
    )


class ImageToImage(TextToImage):
    # Width x height will get overwritten by image size
    image: bt.Tensor = pydantic.Field(..., allow_mutation=False)

    # Miners must choose how to define similarity themselves based on their model
    # by default, the strength values are 0.3, 0.7, 0.9
    similarity: Literal["low", "medium", "high"] = pydantic.Field(
        "medium", allow_mutation=False
    )

    required_hash_fields: list[str] = pydantic.Field(
        [
            "text",
            "negative_prompt",
            "height",
            "width",
            "num_images_per_prompt",
            "seed",
            "nsfw_allowed",
            "image",
            "similarity",
        ],
        allow_mutation=False,
    )


class ValidatorSettings(bt.Synapse):
    _version: list[int] = pydantic.Field([0, 0, 1], allow_mutation=False)
    nsfw_allowed: bool


class MinerSettings(bt.Synapse):
    _version: list[int] = pydantic.Field([0, 0, 1], allow_mutation=False)
    is_public: bool  # set to true if you want anyone (non validator) to query your miner
    min_validator_stake: int  # minimum stake required to query this miner as a validator
    nsfw_allowed: bool
    max_images: int
    max_pixels: int  # max pixels may be different than (max_images * height * width)
    min_width: int
    max_width: int
    min_height: int
    max_height: int


# TO BE IMPLEMENTED
class Upscale(TextToImage):
    scale: float = pydantic.Field(2.0, allow_mutation=False)
