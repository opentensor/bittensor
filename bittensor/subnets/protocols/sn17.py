# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Copyright © 2023 Cortex Foundation

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
from PIL import Image

import typing
from typing import List
from pydantic import BaseModel
import bittensor as bt
from starlette.responses import StreamingResponse
# Your existing Dummy class here...

# New classes inheriting from bt.Synapse
class TextInteractive(bt.StreamingSynapse):
    model: str
    prompt: str
    temperature: typing.Optional[float] = 0.8
    repetition_penalty: typing.Optional[float] = 1.1
    top_p: typing.Optional[float] = 0.9
    top_k: typing.Optional[float] = 40
    max_tokens: typing.Optional[int] = 512
    completion: typing.Optional[str] = None

    async def process_streaming_response(self, response: StreamingResponse):
        """
        `process_streaming_response` is an asynchronous method designed to process the incoming streaming response from the
        Bittensor network. It's the heart of the StreamPrompting class, ensuring that streaming tokens, which represent
        prompts or messages, are decoded and appropriately managed.

        As the streaming response is consumed, the tokens are decoded from their 'utf-8' encoded format, split based on
        newline characters, and concatenated into the `completion` attribute. This accumulation of decoded tokens in the
        `completion` attribute allows for a continuous and coherent accumulation of the streaming content.

        Args:
            response: The streaming response object containing the content chunks to be processed. Each chunk in this
                      response is expected to be a set of tokens that can be decoded and split into individual messages or prompts.
        """
        if self.completion is None:
            self.completion = ""
        bt.logging.debug("Processing streaming response (TextCompletion)")
        async for chunk in response.content.iter_any():
            tokens = chunk.decode("utf-8").split("\n")
            for token in tokens:
                if token:
                    self.completion += token
            yield tokens

    def deserialize(self) -> str:
        """
        Deserializes the response by returning the completion attribute.

        Returns:
            str: The completion result.
        """
        return self.completion
    
    def extract_response_json(self, response: StreamingResponse) -> dict:
        headers = {
            k.decode("utf-8"): v.decode("utf-8")
            for k, v in response.__dict__["_raw_headers"]
        }

        def extract_info(prefix):
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
            "prompt": self.prompt,
            "model": self.model,
            "completion": self.completion,
        }
    

class TextCompletion(bt.StreamingSynapse):
    model: str
    messages: typing.List
    temperature: typing.Optional[float] = 0.8
    repetition_penalty: typing.Optional[float] = 1.1
    top_p: typing.Optional[float] = 0.9
    max_tokens: typing.Optional[int] = 512
    completion: typing.Optional[str] = None

    async def process_streaming_response(self, response: StreamingResponse):
        """
        `process_streaming_response` is an asynchronous method designed to process the incoming streaming response from the
        Bittensor network. It's the heart of the StreamPrompting class, ensuring that streaming tokens, which represent
        prompts or messages, are decoded and appropriately managed.

        As the streaming response is consumed, the tokens are decoded from their 'utf-8' encoded format, split based on
        newline characters, and concatenated into the `completion` attribute. This accumulation of decoded tokens in the
        `completion` attribute allows for a continuous and coherent accumulation of the streaming content.

        Args:
            response: The streaming response object containing the content chunks to be processed. Each chunk in this
                      response is expected to be a set of tokens that can be decoded and split into individual messages or prompts.
        """
        if self.completion is None:
            self.completion = ""
        bt.logging.debug("Processing streaming response (TextCompletion)")
        async for chunk in response.content.iter_any():
            tokens = chunk.decode("utf-8").split("\n")
            for token in tokens:
                if token:
                    self.completion += token
            yield tokens

    def deserialize(self) -> str:
        """
        Deserializes the response by returning the completion attribute.

        Returns:
            str: The completion result.
        """
        return self.completion
    
    def extract_response_json(self, response: StreamingResponse) -> dict:
        headers = {
            k.decode("utf-8"): v.decode("utf-8")
            for k, v in response.__dict__["_raw_headers"]
        }

        def extract_info(prefix):
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
            "model": self.model,
            "messages": self.messages,
            "completion": self.completion,
        }
    

class TextToImage(bt.Synapse):
    model: str
    prompt: str
    height: typing.Optional[int] = 1024
    width: typing.Optional[int] = 1024
    num_inference_steps: typing.Optional[int] = 30
    seed: typing.Optional[int] = -1 
    batch_size: typing.Optional[int] = 1
    refiner: typing.Optional[bool] = False
    output: list[ bt.Tensor ] = [] #base64

    def deserialize(self):
        # Implementation of the deserialize method
        pass  # Customize based on your requirements

class ImageToImage(bt.Synapse):
    model: str
    image: bt.Tensor =  None
    prompt: str
    height: typing.Optional[int] = 1024
    width: typing.Optional[int] = 1024
    strength: typing.Optional[int] = 1
    seed: typing.Optional[int] = -1 
    batch_size: typing.Optional[int] = 1
    output: list[ bt.Tensor ] = [] #base64

    def deserialize(self):
        # Implementation of the deserialize method
        pass  # Customize based on your requirements

class isOnline(bt.Synapse):
    active: str = False

    def deserialize(self):
        # Implementation of the deserialize method
        pass  # Customize based on your requirements
# Additional implementation details here...
class Models(bt.Synapse):
    active: str = False

    def deserialize(self):
        # Implementation of the deserialize method
        pass  # Customize based on your requirements
# Additional implementation details here...    
class ServerInfo(bt.Synapse):
    active: str = False

    def deserialize(self):
        # Implementation of the deserialize method
        pass  # Customize based on your requirements

class StartModel(bt.Synapse):
    active: str = False

    def deserialize(self):
        # Implementation of the deserialize method
        pass  # Customize based on your requirements

class StopModel(bt.Synapse):
    active: str = False

    def deserialize(self):
        # Implementation of the deserialize method
        pass  # Customize based on your requirements
# Additional implementation details here...