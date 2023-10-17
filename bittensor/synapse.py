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
        This function constructs a dictionary of headers from the properties of the instance.

        Headers for 'name' and 'timeout' are directly taken from the instance.
        Further headers are constructed from the properties 'axon' and 'dendrite'.

        If the object is a tensor, its shape and data type are added to the headers.
        For non-optional objects, these are serialized and encoded before adding to the headers.

        Finally, the function adds the sizes of the headers and the total size to the headers.

        Returns:
            dict: A dictionary of headers constructed from the properties of the instance.
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
        instance_fields = self.__dict__

        # Iterating over the fields of the instance
        for field, value in instance_fields.items():
            # If the object is not optional, serializing it, encoding it, and adding it to the headers
            required = schema([self.__class__])["definitions"][self.name].get(
                "required"
            )

            # Skipping the field if it's already in the headers or its value is None
            if field in headers or value is None:
                continue

            # Adding the tensor shape and data type to the headers if the object is a tensor
            if isinstance(value, bittensor.Tensor):
                headers[f"bt_header_tensor_{field}"] = f"{value.shape}-{value.dtype}"

            elif isinstance(value, list) and all(
                isinstance(elem, bittensor.Tensor) for elem in value
            ):
                serialized_list_tensor = []
                for i, tensor in enumerate(value):
                    serialized_list_tensor.append(f"{tensor.shape}-{tensor.dtype}")
                headers[f"bt_header_list_tensor_{field}"] = str(serialized_list_tensor)

            elif isinstance(value, dict) and all(
                isinstance(elem, bittensor.Tensor) for elem in value.values()
            ):
                serialized_dict_tensor = []
                for key, tensor in value.items():
                    serialized_dict_tensor.append(
                        f"{key}-{tensor.shape}-{tensor.dtype}"
                    )
                headers[f"bt_header_dict_tensor_{field}"] = str(serialized_dict_tensor)

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
        Compute a SHA3-256 hash of the serialized body of the Synapse instance.

        The body of the Synapse instance comprises its serialized and encoded
        non-optional fields. This property retrieves these fields using the
        `required_fields_hash` field, then concatenates their string representations,
        and finally computes a SHA3-256 hash of the resulting string.

        Returns:
            str: The hexadecimal representation of the SHA3-256 hash of the instance's body.
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
        This class method parses a given headers dictionary to construct an inputs dictionary.
        Different types of fields ('axon', 'dendrite', 'tensor', and 'input_obj') are identified
        by their prefixes, extracted, and transformed appropriately.
        Remaining fields are directly assigned.

        Args:
            headers (dict): The dictionary of headers to parse

        Returns:
            dict: The parsed inputs dictionary constructed from the headers
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
            # Handle 'tensor' headers
            elif "bt_header_tensor_" in key:
                try:
                    new_key = key.split("bt_header_tensor_")[1]
                    shape, dtype = value.split("-")
                    # TODO: Verify if the shape and dtype values need to be converted before being used
                    inputs_dict[new_key] = bittensor.Tensor(shape=shape, dtype=dtype)
                except Exception as e:
                    bittensor.logging.error(
                        f"Error while parsing 'tensor' header {key}: {e}"
                    )
                    continue
            elif "bt_header_list_tensor_" in key:
                try:
                    new_key = key.split("bt_header_list_tensor_")[1]
                    deserialized_tensors = []
                    stensors = ast.literal_eval(value)
                    for value in stensors:
                        shape, dtype = value.split("-")
                        deserialized_tensors.append(
                            bittensor.Tensor(shape=shape, dtype=dtype)
                        )
                    inputs_dict[new_key] = deserialized_tensors
                except Exception as e:
                    bittensor.logging.error(
                        f"Error while parsing 'tensor' header {key}: {e}"
                    )
                    continue
            elif "bt_header_dict_tensor_" in key:
                try:
                    new_key = key.split("bt_header_dict_tensor_")[1]
                    deserialized_dict_tensors = {}
                    stensors = ast.literal_eval(value)
                    for value in stensors:
                        key, shape, dtype = value.split("-")
                        deserialized_dict_tensors[key] = bittensor.Tensor(
                            shape=shape, dtype=dtype
                        )
                    inputs_dict[new_key] = deserialized_dict_tensors
                except Exception as e:
                    bittensor.logging.error(
                        f"Error while parsing 'tensor' header {key}: {e}"
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
        This class method creates an instance of the class from a given headers dictionary.

        Args:
            headers (dict): The dictionary of headers to parse

        Returns:
            Synapse: A new Synapse instance created from the parsed inputs
        """

        # Get the inputs dictionary from the headers
        input_dict = cls.parse_headers_to_inputs(headers)

        # Use the dictionary unpacking operator to pass the inputs to the class constructor
        synapse = cls(**input_dict)

        return synapse
