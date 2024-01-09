# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2022 Opentensor Foundation

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

import ast
import sys
import torch
import json
import base64
import typing
import hashlib
import pydantic
from pydantic.schema import schema
import bittensor
from typing import Optional, List, Any


def get_size(obj, seen=None) -> int:
    """
    Recursively finds size of objects.

    This function traverses every item of a given object and sums their sizes to compute the total size.

    Args:
        obj (any type): The object to get the size of.
        seen (set): Set of object ids that have been calculated.

    Returns:
        int: The total size of the object.

    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def cast_int(raw: str) -> int:
    """
    Converts a string to an integer, if the string is not None.

    This function attempts to convert a string to an integer. If the string is None,
    it simply returns None.

    Args:
        raw (str): The string to convert.

    Returns:
        int or None: The converted integer, or None if the input was None.

    """
    return int(raw) if raw != None else raw


def cast_float(raw: str) -> float:
    """
    Converts a string to a float, if the string is not None.

    This function attempts to convert a string to a float. If the string is None,
    it simply returns None.

    Args:
        raw (str): The string to convert.

    Returns:
        float or None: The converted float, or None if the input was None.

    """
    return float(raw) if raw != None else raw


class TerminalInfo(pydantic.BaseModel):
    """
    TerminalInfo is a crucial class in the Bittensor framework, encapsulating detailed information about
    a network synapse (node) involved in a communication process. This class serves as a metadata carrier,
    providing essential details about the state and configuration of a terminal during network interactions.

    The TerminalInfo class contains information such as HTTP status codes and messages, processing times,
    IP addresses, ports, Bittensor version numbers, and unique identifiers. These details are vital for
    maintaining network reliability, security, and efficient data flow within the Bittensor network.

    This class includes Pydantic validators and root validators to enforce data integrity and format. It is
    designed to be used natively within Synapses, so that you will not need to call this directly, but rather
    is used as a helper class for Synapses.

    Attributes:
        status_code (int): HTTP status code indicating the result of a network request. Essential for identifying the outcome of network interactions.
        status_message (str): Descriptive message associated with the status code, providing additional context about the request's result.
        process_time (float): Time taken by the terminal to process the call, important for performance monitoring and optimization.
        ip (str): IP address of the terminal, crucial for network routing and data transmission.
        port (int): Network port used by the terminal, key for establishing network connections.
        version (int): Bittensor version running on the terminal, ensuring compatibility between different nodes in the network.
        nonce (int): Unique, monotonically increasing number for each terminal, aiding in identifying and ordering network interactions.
        uuid (str): Unique identifier for the terminal, fundamental for network security and identification.
        hotkey (str): Encoded hotkey string of the terminal wallet, important for transaction and identity verification in the network.
        signature (str): Digital signature verifying the tuple of nonce, axon_hotkey, dendrite_hotkey, and uuid, critical for ensuring data authenticity and security.

    Usage:
        # Creating a TerminalInfo instance
        terminal_info = TerminalInfo(
            status_code=200,
            status_message="Success",
            process_time=0.1,
            ip="198.123.23.1",
            port=9282,
            version=111,
            nonce=111111,
            uuid="5ecbd69c-1cec-11ee-b0dc-e29ce36fec1a",
            hotkey="5EnjDGNqqWnuL2HCAdxeEtN2oqtXZw6BMBe936Kfy2PFz1J1",
            signature="0x0813029319030129u4120u10841824y0182u091u230912u"
        )

        # Accessing TerminalInfo attributes
        ip_address = terminal_info.ip
        processing_duration = terminal_info.process_time

        # TerminalInfo can be used to monitor and verify network interactions, ensuring proper communication and security within the Bittensor network.

    TerminalInfo plays a pivotal role in providing transparency and control over network operations, making it an indispensable tool for developers and users interacting with the Bittensor ecosystem.
    """

    class Config:
        validate_assignment = True

    # The HTTP status code from: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
    status_code: Optional[int] = pydantic.Field(
        title="status_code",
        description="The HTTP status code from: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status",
        examples=200,
        default=None,
        allow_mutation=True,
    )
    _extract_status_code = pydantic.validator(
        "status_code", pre=True, allow_reuse=True
    )(cast_int)

    # The HTTP status code from: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
    status_message: Optional[str] = pydantic.Field(
        title="status_message",
        description="The status_message associated with the status_code",
        examples="Success",
        default=None,
        allow_mutation=True,
    )

    # Process time on this terminal side of call
    process_time: Optional[float] = pydantic.Field(
        title="process_time",
        description="Process time on this terminal side of call",
        examples=0.1,
        default=None,
        allow_mutation=True,
    )
    _extract_process_time = pydantic.validator(
        "process_time", pre=True, allow_reuse=True
    )(cast_float)

    # The terminal ip.
    ip: Optional[str] = pydantic.Field(
        title="ip",
        description="The ip of the axon recieving the request.",
        examples="198.123.23.1",
        default=None,
        allow_mutation=True,
    )

    # The host port of the terminal.
    port: Optional[int] = pydantic.Field(
        title="port",
        description="The port of the terminal.",
        examples="9282",
        default=None,
        allow_mutation=True,
    )
    _extract_port = pydantic.validator("port", pre=True, allow_reuse=True)(cast_int)

    # The bittensor version on the terminal as an int.
    version: Optional[int] = pydantic.Field(
        title="version",
        description="The bittensor version on the axon as str(int)",
        examples=111,
        default=None,
        allow_mutation=True,
    )
    _extract_version = pydantic.validator("version", pre=True, allow_reuse=True)(
        cast_int
    )

    # A unique monotonically increasing integer nonce associate with the terminal
    nonce: Optional[int] = pydantic.Field(
        title="nonce",
        description="A unique monotonically increasing integer nonce associate with the terminal generated from time.monotonic_ns()",
        examples=111111,
        default=None,
        allow_mutation=True,
    )
    _extract_nonce = pydantic.validator("nonce", pre=True, allow_reuse=True)(cast_int)

    # A unique identifier associated with the terminal, set on the axon side.
    uuid: Optional[str] = pydantic.Field(
        title="uuid",
        description="A unique identifier associated with the terminal",
        examples="5ecbd69c-1cec-11ee-b0dc-e29ce36fec1a",
        default=None,
        allow_mutation=True,
    )

    # The bittensor version on the terminal as an int.
    hotkey: Optional[str] = pydantic.Field(
        title="hotkey",
        description="The ss58 encoded hotkey string of the terminal wallet.",
        examples="5EnjDGNqqWnuL2HCAdxeEtN2oqtXZw6BMBe936Kfy2PFz1J1",
        default=None,
        allow_mutation=True,
    )

    # A signature verifying the tuple (axon_nonce, axon_hotkey, dendrite_hotkey, axon_uuid)
    signature: Optional[str] = pydantic.Field(
        title="signature",
        description="A signature verifying the tuple (nonce, axon_hotkey, dendrite_hotkey, uuid)",
        examples="0x0813029319030129u4120u10841824y0182u091u230912u",
        default=None,
        allow_mutation=True,
    )


class Synapse(pydantic.BaseModel):
    """
    Represents a Synapse in the Bittensor network, serving as a communication schema between neurons (nodes).
    Synapses ensure the format and correctness of transmission tensors according to the Bittensor protocol.
    Each Synapse type is tailored for a specific machine learning (ML) task, following unique compression and
    communication processes. This helps maintain sanitized, correct, and useful information flow across the network.

    The Synapse class encompasses essential network properties such as HTTP route names, timeouts, request sizes, and
    terminal information. It also includes methods for serialization, deserialization, attribute setting, and hash
    computation, ensuring secure and efficient data exchange in the network.

    The class includes Pydantic validators and root validators to enforce data integrity and format. Additionally,
    properties like 'is_success', 'is_failure', 'is_timeout', etc., provide convenient status checks based on
    dendrite responses.

    Think of Bittensor Synapses as glorified pydantic wrappers that have been designed to be used in a distributed
    network. They provide a standardized way to communicate between neurons, and are the primary mechanism for
    communication between neurons in Bittensor.

    Key Features:

    1. HTTP Route Name (`name` attribute):
        Enables the identification and proper routing of requests within the network. Essential for users
        defining custom routes for specific machine learning tasks.

    2. Query Timeout (`timeout` attribute):
        Determines the maximum duration allowed for a query, ensuring timely responses and network
        efficiency. Crucial for users to manage network latency and response times, particularly in
        time-sensitive applications.

    3. Request Sizes (`total_size`, `header_size` attributes):
        Keeps track of the size of request bodies and headers, ensuring efficient data transmission without
        overloading the network. Important for users to monitor and optimize the data payload, especially
        in bandwidth-constrained environments.

    4. Terminal Information (`dendrite`, `axon` attributes):
        Stores information about the dendrite (receiving end) and axon (sending end), facilitating communication
        between nodes. Users can access detailed information about the communication endpoints, aiding in
        debugging and network analysis.

    5. Body Hash Computation (`computed_body_hash`, `required_hash_fields`):
        Ensures data integrity and security by computing hashes of transmitted data. Provides users with a
        mechanism to verify data integrity and detect any tampering during transmission.

    6. Serialization and Deserialization Methods:
        Facilitates the conversion of Synapse objects to and from a format suitable for network transmission.
        Essential for users who need to customize data formats for specific machine learning models or tasks.

    7. Status Check Properties (`is_success`, `is_failure`, `is_timeout`, etc.):
        Provides quick and easy methods to check the status of a request, improving error handling and
        response management. Users can efficiently handle different outcomes of network requests, enhancing
        the robustness of their applications.

    Example usage:
        # Creating a Synapse instance with default values
        synapse = Synapse()

        # Setting properties and input
        synapse.timeout = 15.0
        synapse.name = "MySynapse"
        # Not setting fields that are not defined in your synapse class will result in an error, e.g.:
        synapse.dummy_input = 1 # This will raise an error because dummy_input is not defined in the Synapse class

        # Get a dictionary of headers and body from the synapse instance
        synapse_dict = synapse.json()

        # Get a dictionary of headers from the synapse instance
        headers = synapse.to_headers()

        # Reconstruct the synapse from headers using the classmethod 'from_headers'
        synapse = Synapse.from_headers(headers)

        # Deserialize synapse after receiving it over the network, controlled by `deserialize` method
        deserialized_synapse = synapse.deserialize()

        # Checking the status of the request
        if synapse.is_success:
            print("Request succeeded")

        # Checking and setting the status of the request
        print(synapse.axon.status_code)
        synapse.axon.status_code = 408 # Timeout

    Attributes:
        name (str): HTTP route name, set on axon.attach.
        timeout (float): Total query length, set by the dendrite terminal.
        total_size (int): Total size of request body in bytes.
        header_size (int): Size of request header in bytes.
        dendrite (TerminalInfo): Information about the dendrite terminal.
        axon (TerminalInfo): Information about the axon terminal.
        computed_body_hash (str): Computed hash of the request body.
        required_hash_fields (List[str]): Fields required to compute the body hash.

    Methods:
        deserialize: Custom deserialization logic for subclasses.
        __setattr__: Override method to make 'required_hash_fields' read-only.
        get_total_size: Calculates and returns the total size of the object.
        to_headers: Constructs a dictionary of headers from instance properties.
        body_hash: Computes a SHA3-256 hash of the serialized body.
        parse_headers_to_inputs: Parses headers to construct an inputs dictionary.
        from_headers: Creates an instance from a headers dictionary.

    This class is a cornerstone in the Bittensor framework, providing the necessary tools for secure, efficient, and
    standardized communication in a decentralized environment.
    """

    class Config:
        validate_assignment = True

    def deserialize(self) -> "Synapse":
        """
        Deserializes the Synapse object.

        This method is intended to be overridden by subclasses for custom deserialization logic.
        In the context of the Synapse superclass, this method simply returns the instance itself.
        When inheriting from this class, subclasses should provide their own implementation for
        deserialization if specific deserialization behavior is desired.

        By default, if a subclass does not provide its own implementation of this method, the
        Synapse's deserialize method will be used, returning the object instance as-is.

        In its default form, this method simply returns the instance of the Synapse itself without any modifications. Subclasses of Synapse can override this method to add specific deserialization behaviors, such as converting serialized data back into complex object types or performing additional data integrity checks.

        Example:
            class CustomSynapse(Synapse):
                additional_data: str

                def deserialize(self) -> "CustomSynapse":
                    # Custom deserialization logic
                    # For example, decoding a base64 encoded string in 'additional_data'
                    if self.additional_data:
                        self.additional_data = base64.b64decode(self.additional_data).decode('utf-8')
                    return self

            serialized_data = '{"additional_data": "SGVsbG8gV29ybGQ="}'  # Base64 for 'Hello World'
            custom_synapse = CustomSynapse.parse_raw(serialized_data)
            deserialized_synapse = custom_synapse.deserialize()

            # deserialized_synapse.additional_data would now be 'Hello World'

        Returns:
            Synapse: The deserialized Synapse object. In this default implementation, it returns the object itself.
        """
        return self

    @pydantic.root_validator(pre=True)
    def set_name_type(cls, values) -> dict:
        values["name"] = cls.__name__
        return values

    # Defines the http route name which is set on axon.attach( callable( request: RequestName ))
    name: Optional[str] = pydantic.Field(
        title="name",
        description="Defines the http route name which is set on axon.attach( callable( request: RequestName ))",
        examples="Forward",
        allow_mutation=True,
        default=None,
        repr=False,
    )

    # The call timeout, set by the dendrite terminal.
    timeout: Optional[float] = pydantic.Field(
        title="timeout",
        description="Defines the total query length.",
        examples=12.0,
        default=12.0,
        allow_mutation=True,
        repr=False,
    )
    _extract_timeout = pydantic.validator("timeout", pre=True, allow_reuse=True)(
        cast_float
    )

    # The call timeout, set by the dendrite terminal.
    total_size: Optional[int] = pydantic.Field(
        title="total_size",
        description="Total size of request body in bytes.",
        examples=1000,
        default=0,
        allow_mutation=True,
        repr=False,
    )
    _extract_total_size = pydantic.validator("total_size", pre=True, allow_reuse=True)(
        cast_int
    )

    # The call timeout, set by the dendrite terminal.
    header_size: Optional[int] = pydantic.Field(
        title="header_size",
        description="Size of request header in bytes.",
        examples=1000,
        default=0,
        allow_mutation=True,
        repr=False,
    )
    _extract_header_size = pydantic.validator(
        "header_size", pre=True, allow_reuse=True
    )(cast_int)

    # The dendrite Terminal Information.
    dendrite: Optional[TerminalInfo] = pydantic.Field(
        title="dendrite",
        description="Dendrite Terminal Information",
        examples="bittensor.TerminalInfo",
        default=TerminalInfo(),
        allow_mutation=True,
        repr=False,
    )

    # A axon terminal information
    axon: Optional[TerminalInfo] = pydantic.Field(
        title="axon",
        description="Axon Terminal Information",
        examples="bittensor.TerminalInfo",
        default=TerminalInfo(),
        allow_mutation=True,
        repr=False,
    )

    computed_body_hash: Optional[str] = pydantic.Field(
        title="computed_body_hash",
        description="The computed body hash of the request.",
        examples="0x0813029319030129u4120u10841824y0182u091u230912u",
        default="",
        allow_mutation=False,
        repr=False,
    )

    required_hash_fields: Optional[List[str]] = pydantic.Field(
        title="required_hash_fields",
        description="The list of required fields to compute the body hash.",
        examples=["roles", "messages"],
        default=[],
        allow_mutation=False,
        repr=False,
    )

    def __setattr__(self, name: str, value: Any):
        """
        Override the __setattr__ method to make the `required_hash_fields` property read-only.

        This is a security mechanism such that the `required_hash_fields` property cannot be
        overridden by the user or malicious code.
        """
        if name == "body_hash":
            raise AttributeError(
                "body_hash property is read-only and cannot be overridden."
            )
        super().__setattr__(name, value)

    def get_total_size(self) -> int:
        """
        Get the total size of the current object.

        This method first calculates the size of the current object, then assigns it
        to the instance variable self.total_size and finally returns this value.

        Returns:
            int: The total size of the current object.
        """
        self.total_size = get_size(self)
        return self.total_size

    @property
    def is_success(self) -> bool:
        """
        Checks if the dendrite's status code indicates success.

        This method returns True if the status code of the dendrite is 200,
        which typically represents a successful HTTP request.

        Returns:
            bool: True if dendrite's status code is 200, False otherwise.
        """
        return self.dendrite.status_code == 200

    @property
    def is_failure(self) -> bool:
        """
        Checks if the dendrite's status code indicates failure.

        This method returns True if the status code of the dendrite is not 200,
        which would mean the HTTP request was not successful.

        Returns:
            bool: True if dendrite's status code is not 200, False otherwise.
        """
        return self.dendrite.status_code != 200

    @property
    def is_timeout(self) -> bool:
        """
        Checks if the dendrite's status code indicates a timeout.

        This method returns True if the status code of the dendrite is 408,
        which is the HTTP status code for a request timeout.

        Returns:
            bool: True if dendrite's status code is 408, False otherwise.
        """
        return self.dendrite.status_code == 408

    @property
    def is_blacklist(self) -> bool:
        """
        Checks if the dendrite's status code indicates a blacklisted request.

        This method returns True if the status code of the dendrite is 403,
        which is the HTTP status code for a forbidden request.

        Returns:
            bool: True if dendrite's status code is 403, False otherwise.
        """
        return self.dendrite.status_code == 403

    @property
    def failed_verification(self) -> bool:
        """
        Checks if the dendrite's status code indicates failed verification.

        This method returns True if the status code of the dendrite is 401,
        which is the HTTP status code for unauthorized access.

        Returns:
            bool: True if dendrite's status code is 401, False otherwise.
        """
        return self.dendrite.status_code == 401

    def to_headers(self) -> dict:
        """
        Converts the state of a Synapse instance into a dictionary of HTTP headers. This method is essential for
        packaging Synapse data for network transmission in the Bittensor framework, ensuring that each key aspect of
        the Synapse is represented in a format suitable for HTTP communication.

        Process:
        1. Basic Information: It starts by including the 'name' and 'timeout' of the Synapse, which are fundamental
            for identifying the query and managing its lifespan on the network.
        2. Complex Objects: The method serializes the 'axon' and 'dendrite' objects, if present, into strings. This
            serialization is crucial for preserving the state and structure of these objects over the network.
        3. Encoding: Non-optional complex objects are serialized and encoded in base64, making them safe for HTTP
            transport.
        4. Size Metrics: The method calculates and adds the size of headers and the total object size, providing
            valuable information for network bandwidth management.

        Example Usage:
            synapse = Synapse(name="ExampleSynapse", timeout=30)
            headers = synapse.to_headers()
            # headers now contains a dictionary representing the Synapse instance

        Returns:
            dict: A dictionary containing key-value pairs representing the Synapse's properties, suitable for HTTP communication.
        """
        # Initializing headers with 'name' and 'timeout'
        headers = {"name": self.name, "timeout": str(self.timeout)}

        # Adding headers for 'axon' and 'dendrite' if they are not None
        headers.update(
            {
                f"bt_header_axon_{k}": str(v)
                for k, v in self.axon.dict().items()
                if v is not None
            }
        )
        headers.update(
            {
                f"bt_header_dendrite_{k}": str(v)
                for k, v in self.dendrite.dict().items()
                if v is not None
            }
        )

        # Getting the type hints for the properties of the instance
        property_type_hints = typing.get_type_hints(self)

        # Getting the fields of the instance
        instance_fields = self.dict()

        # Iterating over the fields of the instance
        for field, value in instance_fields.items():
            # If the object is not optional, serializing it, encoding it, and adding it to the headers
            required = schema([self.__class__])["definitions"][self.name].get(
                "required"
            )

            # Skipping the field if it's already in the headers or its value is None
            if field in headers or value is None:
                continue

            elif required and field in required:
                try:
                    # create an empty (dummy) instance of type(value) to pass pydantic validation on the axon side
                    serialized_value = json.dumps(value.__class__.__call__())
                    encoded_value = base64.b64encode(serialized_value.encode()).decode(
                        "utf-8"
                    )
                    headers[f"bt_header_input_obj_{field}"] = encoded_value
                except TypeError as e:
                    raise ValueError(
                        f"Error serializing {field} with value {value}. Objects must be json serializable."
                    ) from e

        # Adding the size of the headers and the total size to the headers
        headers["header_size"] = str(sys.getsizeof(headers))
        headers["total_size"] = str(self.get_total_size())
        headers["computed_body_hash"] = self.body_hash

        return headers

    @property
    def body_hash(self) -> str:
        """
        Computes a SHA3-256 hash of the serialized body of the Synapse instance. This hash is used to
        ensure the data integrity and security of the Synapse instance when it's transmitted across the
        network. It's a crucial feature for verifying that the data received is the same as the data sent.

        Process:
        1. Iterates over each required field as specified in `required_fields_hash`.
        2. Concatenates the string representation of these fields.
        3. Applies SHA3-256 hashing to the concatenated string to produce a unique fingerprint of the data.

        Example:
            synapse = Synapse(name="ExampleRoute", timeout=10)
            hash_value = synapse.body_hash
            # hash_value is the SHA3-256 hash of the serialized body of the Synapse instance

        Returns:
            str: The SHA3-256 hash as a hexadecimal string, providing a fingerprint of the Synapse instance's data for integrity checks.
        """
        # Hash the body for verification
        hashes = []

        # Getting the fields of the instance
        instance_fields = self.dict()

        for field, value in instance_fields.items():
            # If the field is required in the subclass schema, hash and add it.
            if field in self.required_hash_fields:
                hashes.append(bittensor.utils.hash(str(value)))

        # Hash and return the hashes that have been concatenated
        return bittensor.utils.hash("".join(hashes))

    @classmethod
    def parse_headers_to_inputs(cls, headers: dict) -> dict:
        """
        Interprets and transforms a given dictionary of headers into a structured dictionary, facilitating
        the reconstruction of Synapse objects. This method is essential for parsing network-transmitted
        data back into a Synapse instance, ensuring data consistency and integrity.

        Process:
        1. Separates headers into categories based on prefixes ('axon', 'dendrite', etc.).
        2. Decodes and deserializes 'input_obj' headers into their original objects.
        3. Assigns simple fields directly from the headers to the input dictionary.

        Example:
            received_headers = {
                'bt_header_axon_address': '127.0.0.1',
                'bt_header_dendrite_port': '8080',
                # Other headers...
            }
            inputs = Synapse.parse_headers_to_inputs(received_headers)
            # inputs now contains a structured representation of Synapse properties based on the headers

        Note: This is handled automatically when calling Synapse.from_headers(headers) and does not
        need to be called directly.

        Args:
            headers (dict): The headers dictionary to parse.

        Returns:
            dict: A structured dictionary representing the inputs for constructing a Synapse instance.
        """

        # Initialize the input dictionary with empty sub-dictionaries for 'axon' and 'dendrite'
        inputs_dict = {"axon": {}, "dendrite": {}}

        # Iterate over each item in the headers
        for key, value in headers.items():
            # Handle 'axon' headers
            if "bt_header_axon_" in key:
                try:
                    new_key = key.split("bt_header_axon_")[1]
                    inputs_dict["axon"][new_key] = value
                except Exception as e:
                    bittensor.logging.error(
                        f"Error while parsing 'axon' header {key}: {e}"
                    )
                    continue
            # Handle 'dendrite' headers
            elif "bt_header_dendrite_" in key:
                try:
                    new_key = key.split("bt_header_dendrite_")[1]
                    inputs_dict["dendrite"][new_key] = value
                except Exception as e:
                    bittensor.logging.error(
                        f"Error while parsing 'dendrite' header {key}: {e}"
                    )
                    continue
            # Handle 'input_obj' headers
            elif "bt_header_input_obj" in key:
                try:
                    new_key = key.split("bt_header_input_obj_")[1]
                    # Skip if the key already exists in the dictionary
                    if new_key in inputs_dict:
                        continue
                    # Decode and load the serialized object
                    inputs_dict[new_key] = json.loads(
                        base64.b64decode(value.encode()).decode("utf-8")
                    )
                except json.JSONDecodeError as e:
                    bittensor.logging.error(
                        f"Error while json decoding 'input_obj' header {key}: {e}"
                    )
                    continue
                except Exception as e:
                    bittensor.logging.error(
                        f"Error while parsing 'input_obj' header {key}: {e}"
                    )
                    continue
            else:
                pass  # TODO: log unexpected keys

        # Assign the remaining known headers directly
        inputs_dict["timeout"] = headers.get("timeout", None)
        inputs_dict["name"] = headers.get("name", None)
        inputs_dict["header_size"] = headers.get("header_size", None)
        inputs_dict["total_size"] = headers.get("total_size", None)
        inputs_dict["computed_body_hash"] = headers.get("computed_body_hash", None)

        return inputs_dict

    @classmethod
    def from_headers(cls, headers: dict) -> "Synapse":
        """
        Constructs a new Synapse instance from a given headers dictionary, enabling the re-creation of the
        Synapse's state as it was prior to network transmission. This method is a key part of the
        deserialization process in the Bittensor network, allowing nodes to accurately reconstruct Synapse
        objects from received data.

        Example:
            received_headers = {
                'bt_header_axon_address': '127.0.0.1',
                'bt_header_dendrite_port': '8080',
                # Other headers...
            }
            synapse = Synapse.from_headers(received_headers)
            # synapse is a new Synapse instance reconstructed from the received headers

        Args:
            headers (dict): The dictionary of headers containing serialized Synapse information.

        Returns:
            Synapse: A new instance of Synapse, reconstructed from the parsed header information, replicating the original instance's state.
        """

        # Get the inputs dictionary from the headers
        input_dict = cls.parse_headers_to_inputs(headers)

        # Use the dictionary unpacking operator to pass the inputs to the class constructor
        synapse = cls(**input_dict)

        return synapse
