import typing
import bittensor as bt
import pydantic
import torch

class ImageGenerating(bt.Synapse):
    prompt: str = pydantic.Field(
        default="",
        title="Prompt",
        description="Requested prompt for text to image generating",
    )
    seed: int = pydantic.Field(
        default=0, title="Seed", description="Seed for deterministic generation"
    )
    image: str = pydantic.Field(
        default="", title="Image", description="Output of text to image model in base64 format"
    )
    model_name: str = pydantic.Field(
        default="",
        title="Model Name",
        description="Name of the model used for generation",
    )
    pipeline_params: dict = pydantic.Field(
        default={"num_inference_steps": 30},
        title="Pipeline Parameters",
        description="Additional generating params",
    )
    request_dict: dict = pydantic.Field(
        default={},
        title="Dictionary contains request",
        description="Dict contains arbitary information",
    )

    response_dict: dict = pydantic.Field(
        default={},
        title="Dictionary contains response",
        description="Dict contains arbitary information",
    )

    def deserialize(self) -> typing.List[str]:
        return self.image
