# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

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

import pydantic
import time
import torch
from typing import List
import bittensor as bt
from starlette.responses import StreamingResponse


class Message(pydantic.BaseModel):
    name: str = pydantic.Field(
        ..., title="Name", description="The name field of the message."
    )
    content: str = pydantic.Field(
        ..., title="Content", description="The content of the message."
    )


class PromptingMixin(pydantic.BaseModel):

    """
    A Pydantic model representing a prompting scenario involving interactions between AI characters and human users.

    This model includes details about the characters, users, and the interaction criteria, as well as the messages exchanged
    and the completion status of the prompt. The fields related to character and user information, as well as the criteria
    and messages, are immutable once set during object initialization. In contrast, the `completion` field is mutable and
    can be updated as the scenario progresses.

    The Config inner class specifies that assignment validation should occur on this class (validate_assignment = True),
    meaning value assignments to the instance fields are checked against their defined types for correctness.

    Attributes:
        character_info (str): Information about the AI character who is responding. Immutable.
        character_name (str): Name of the AI responding character. Immutable.
        user_names (List[str]): Names of the human participants. Immutable.
        char_names (List[str]): Names of all of the AI characters. Immutable.
        criteria (List[str]): Criteria for the AI response. Immutable.
        messages (List[Message]): List of messages containing character name and content. Immutable.
        completion (str): Mutable string that captures the completion status of the prompt.
        required_hash_fields (List[str]): A list of fields that are required for the hash. Immutable.
    """

    class Config:
        """
        Pydantic model configuration class for Prompting. This class sets validation of attribute assignment as True.
        validate_assignment set to True means the pydantic model will validate attribute assignments on the class.
        """

        validate_assignment = True

    character_info: str = pydantic.Field(
        ...,
        title="Character Info",
        description="Information about the AI character who is responding.",
        allow_mutation=False,
    )
    character_name: str = pydantic.Field(
        ...,
        title="Character Name",
        description="Name of the AI responding character.",
        allow_mutation=False,
    )
    user_names: List[str] = pydantic.Field(
        ...,
        title="User Names",
        description="Names of the human participants.",
        allow_mutation=False,
    )
    char_names: List[str] = pydantic.Field(
        ...,
        title="All character names",
        description="Names of all of the AI characters.",
        allow_mutation=False,
    )
    criteria: List[str] = pydantic.Field(
        ...,
        title="Criteria",
        description="Criteria for the AI response.",
        allow_mutation=False,
    )
    messages: List[Message] = pydantic.Field(
        ...,
        title="Messages",
        description="List of messages containing character name, and content.",
        allow_mutation=False,
    )

    completion: str = pydantic.Field(
        "",
        title="Completion",
        description="Completion status of the current Prompting object. This attribute is mutable and can be updated.",
    )

    required_hash_fields: List[str] = pydantic.Field(
        ["messages"],
        title="Required Hash Fields",
        description="A list of required fields for the hash.",
        allow_mutation=False,
    )


class Prompting(PromptingMixin, bt.Synapse):
    """
    The Prompting subclass of the Synapse class encapsulates the functionalities related to prompting scenarios.

    Methods:
        deserialize() -> "Prompting": Returns the instance of the current object.

    Here is an example of how the `Prompting` class can be used:

    Example of Usage:
    ```python
    # Example Messages
    messages = [
        Message(name="Alice", content="How's the weather today?"),
        Message(name="Bob", content="It's sunny and warm today!")
    ]

    # Create a Prompting instance with character info, names, criteria, messages, and completion status
    prompt = Prompting(
        character_info="AI Assistant",
        character_name="Mr.Robot",
        user_names=["Alice", "Bob"],
        char_names=["Mr.Robot"],
        criteria=["You should be polite and friendly."],
        messages=messages
    )

    # Print the character info and the first message
    print("Character Info:", prompt.character_info)
    print("First Message:", prompt.messages[0].content)

    # Update the completion
    model_prompt =... # Use prompt.messages to generate a prompt
    for your LLM as a single string.
    prompt.completion = model(model_prompt)

    # Print the updated completion status
    print("Completion:", prompt.completion)
    ```

    This example demonstrates how to create an instance of the `Prompting` class with detailed character information,
    user names, character names, criteria for the interaction, a list of messages, and a completion text.
    """

    def deserialize(self) -> "Prompting":
        """
        Returns the instance of the current Prompting object.

        This method is intended to be potentially overridden by subclasses for custom deserialization logic.
        In the context of the Prompting class, it simply returns the instance itself. However, for subclasses
        inheriting from this class, it might give a custom implementation for deserialization if need be.

        Returns:
            Prompting: The current instance of the Prompting class.
        """
        return self


class StreamPrompting(PromptingMixin, bt.StreamingSynapse):
    """
    StreamPrompting is a specialized implementation of the `StreamingSynapse` tailored for prompting functionalities within
    the Bittensor network. This class is intended to interact with a streaming response that contains a sequence of tokens,
    which represent prompts or messages in a certain scenario.

    As a developer, when using or extending the `StreamPrompting` class, you should be primarily focused on the structure
    and behavior of the prompts you are working with. The class has been designed to seamlessly handle the streaming,
    decoding, and accumulation of tokens that represent these prompts.

    Methods:
    - `process_streaming_response`: This method asynchronously processes the incoming streaming response by decoding
                                    the tokens and accumulating them in the `completion` attribute.

    - `deserialize`: Converts the `completion` attribute into its desired data format, in this case, a string.

    - `extract_response_json`: Extracts relevant JSON data from the response, useful for gaining insights on the response's
                               metadata or for debugging purposes.

    Note: While you can directly use the `StreamPrompting` class, it's designed to be extensible. Thus, you can create
    subclasses to further customize behavior for specific prompting scenarios or requirements.
    """

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
        """
        `extract_response_json` is a method that performs the crucial task of extracting pertinent JSON data from the given
        response. The method is especially useful when you need a detailed insight into the streaming response's metadata
        or when debugging response-related issues.

        Beyond just extracting the JSON data, the method also processes and structures the data for easier consumption
        and understanding. For instance, it extracts specific headers related to dendrite and axon, offering insights
        about the Bittensor network's internal processes. The method ultimately returns a dictionary with a structured
        view of the extracted data.

        Args:
            response: The response object from which to extract the JSON data. This object typically includes headers and
                      content which can be used to glean insights about the response.

        Returns:
            dict: A structured dictionary containing:
                - Basic response metadata such as name, timeout, total_size, and header_size.
                - Dendrite and Axon related information extracted from headers.
                - Roles and Messages pertaining to the current StreamPrompting instance.
                - The accumulated completion.
        """
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
            "roles": self.roles,
            "messages": self.messages,
            "completion": self.completion,
        }
