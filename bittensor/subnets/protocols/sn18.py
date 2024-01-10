from enum import Enum
from typing import AsyncIterator, Dict, List, Literal, Optional

import bittensor as bt
import pydantic
from starlette.responses import StreamingResponse

# from ..providers.image import DallE, Stability

# from ..providers.text import Anthropic, GeminiPro, OpenAI


class IsAlive( bt.Synapse ):
    answer: Optional[str] = None
    completion: str = pydantic.Field(
        "",
        title="Completion",
        description="Completion status of the current StreamPrompting object. "
                    "This attribute is mutable and can be updated.",
    )

class ImageResponse(bt.Synapse):
    """ A class to represent the response for an image-related request. """
    # https://platform.stability.ai/docs/api-reference#tag/v1generation/operation/textToImage

    completion: Optional[Dict] = pydantic.Field(
        None,
        title="Completion",
        description="The completion data of the image response."
    )

    messages: str = pydantic.Field(
        ...,
        title="Messages",
        description="Messages related to the image response."
    )

    provider: str = pydantic.Field(
        default="OpenAI",
        title="Provider",
        description="The provider to use when calling for your response."
    )

    seed: int = pydantic.Field(
        default=1234,
        title="Seed",
        description="The seed that which to generate the image with"
    )

    samples: int = pydantic.Field(
        default=1,
        title="Samples",
        description="The number of samples to generate"
    )

    cfg_scale: float = pydantic.Field(
        default=8.0,
        title="cfg_scale",
        description="The cfg_scale to use for image generation"
    )

     # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m, k_dpmpp_sde)
    sampler: str = pydantic.Field(
        default="",
        title="Sampler",
        description="The sampler to use for image generation"
    )

    steps: int = pydantic.Field(
        default=30,
        title="Seed",
        description="The steps to take in generating the image"
    )

    model: str = pydantic.Field(
        default="dall-e-2",
        title="Model",
        description="The model used for generating the image."
    )

    style: str = pydantic.Field(
        default="vivid",
        title="Style",
        description="The style of the image."
    )

    size: str = pydantic.Field(
        default="1024x1024",
        title="The size of the image, used for Openai generation. Options are 1024x1024, 1792x1024, 1024x1792 for dalle3",
        description="The size of the image."
    )

    height: int = pydantic.Field(
        default=1024,
        title="Height used for non Openai images",
        description="height"
    )

    width: int = pydantic.Field(
        default=1024,
        title="Width used for non Openai images",
        description="width"
    )

    quality: str = pydantic.Field(
        default="standard",
        title="Quality",
        description="The quality of the image."
    )

    required_hash_fields: List[str] = pydantic.Field(
        ["messages"],
        title="Required Hash Fields",
        description="A list of fields required for the hash."
    )

    def deserialize(self) -> Optional[Dict]:
        """ Deserialize the completion data of the image response. """
        return self.completion

class Embeddings( bt.Synapse):
    """ A class to represent the embeddings request and response. """

    texts: List[str] = pydantic.Field(
        ...,
        title="Text",
        description="The list of input texts for which embeddings are to be generated."
    )

    model: str = pydantic.Field(
        default="text-embedding-ada-002",
        title="Model",
        description="The model used for generating embeddings."
    )

    embeddings: Optional[List[List[float]]] = pydantic.Field(
        None,
        title="Embeddings",
        description="The resulting list of embeddings, each corresponding to an input text."
    )



class StreamPrompting(bt.StreamingSynapse):

    messages: List[Dict[str, str]] = pydantic.Field(
        ...,
        title="Messages",
        description="A list of messages in the StreamPrompting scenario, "
                    "each containing a role and content. Immutable.",
        allow_mutation=False,
    )

    required_hash_fields: List[str] = pydantic.Field(
        ["messages"],
        title="Required Hash Fields",
        description="A list of required fields for the hash.",
        allow_mutation=False,
    )

    seed: int = pydantic.Field(
        default="1234",
        title="Seed",
        description="Seed for text generation. This attribute is immutable and cannot be updated.",
    )

    temperature: float = pydantic.Field(
        default=0.0001,
        title="Temperature",
        description="Temperature for text generation. "
                    "This attribute is immutable and cannot be updated.",
    )

    max_tokens: int = pydantic.Field(
        default=2048,
        title="Max Tokens",
        description="Max tokens for text generation. "
                    "This attribute is immutable and cannot be updated.",
    )

    top_p: float = pydantic.Field(
        default=0.001,
        title="Top_p",
        description="Top_p for text generation. The sampler will pick one of "
                    "the top p percent tokens in the logit distirbution. "
                    "This attribute is immutable and cannot be updated.",
    )

    top_k: int = pydantic.Field(
        default=1,
        title="Top_k",
        description="Top_k for text generation. Sampler will pick one of  "
                    "the k most probablistic tokens in the logit distribtion. "
                    "This attribute is immutable and cannot be updated.",
    )

    completion: str = pydantic.Field(
        None,
        title="Completion",
        description="Completion status of the current StreamPrompting object. "
                    "This attribute is mutable and can be updated.",
    )

    provider: str = pydantic.Field(
        default="OpenAI",
        title="Provider",
        description="The provider to use when calling for your response."
    )

    model: str = pydantic.Field(
        default="gpt-3.5-turbo",
        title="model",
        description="The model to use when calling provider for your response.",
    )

    async def process_streaming_response(self, response: StreamingResponse) -> AsyncIterator[str]:
        if self.completion is None:
            self.completion = ""
        async for chunk in response.content.iter_any():
            tokens = chunk.decode("utf-8")
            for token in tokens:
                if token:
                    self.completion += token
            yield tokens

    def deserialize(self) -> str:
        return self.completion

    def extract_response_json(self, response: StreamingResponse) -> dict:
        headers = {
            k.decode("utf-8"): v.decode("utf-8")
            for k, v in response.__dict__["_raw_headers"]
        }

        def extract_info(prefix: str) -> dict[str, str]:
            return {
                key.split("_")[-1]: value
                for key, value in headers.items()
                if key.startswith(prefix)
            }

        return {
            "name": headers.get("name", ""),
            "timeout": float(headers.get("timeout", 0)),
            "total_size": int(headers.get("total_size", 0)),
            "header_size": int(headers.get("header_size", 0)),
            "dendrite": extract_info("bt_header_dendrite"),
            "axon": extract_info("bt_header_axon"),
            "messages": self.messages,
            "completion": self.completion,
        }
