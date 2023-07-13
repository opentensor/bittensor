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

import sys
import torch
import pickle
import base64
import typing
import pydantic
import bittensor
from abc import abstractmethod
from fastapi.responses import Response
from fastapi import Request
from typing import Dict, Optional, Tuple, Union, List, Callable

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
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
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def cast_int(raw: str) -> int:
    return int( raw ) if raw != None else raw

def cast_float( raw: str ) -> float:
    return float( raw ) if raw != None else raw

class TerminalInfo( pydantic.BaseModel ):

    class Config:
        validate_assignment = True

    # The HTTP status code from: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
    status_code: Optional[int] = pydantic.Field(
        title = 'status_code',
        description = 'The HTTP status code from: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status',
        examples = 200,
        default = None,
        allow_mutation = True
    )
    _extract_status_code = pydantic.validator('status_code', pre=True, allow_reuse=True)(cast_int)

    # The HTTP status code from: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
    status_message: Optional[str] = pydantic.Field(
        title = 'status_message',
        description = 'The status_message associated with the status_code',
        examples = 'Success',
        default = None,
        allow_mutation = True
    )
        
    # Process time on this terminal side of call
    process_time: Optional[float] = pydantic.Field(
        title = 'process_time',
        description = 'Process time on this terminal side of call',
        examples = 0.1,
        default = None,
        allow_mutation = True
    )
    _extract_process_time = pydantic.validator('process_time', pre=True, allow_reuse=True)(cast_float)

    # The terminal ip.
    ip: Optional[ str ] = pydantic.Field(
        title = 'ip',
        description = 'The ip of the axon recieving the request.',
        examples = '198.123.23.1',
        default = None,
        allow_mutation = True
    )

    # The host port of the terminal.
    port: Optional[ int ] = pydantic.Field(
        title = 'port',
        description = 'The port of the terminal.',
        examples = '9282',
        default = None,
        allow_mutation = True
    )
    _extract_port = pydantic.validator('port', pre=True, allow_reuse=True)(cast_int)

    # The bittensor version on the terminal as an int.
    version: Optional[ int ] = pydantic.Field(
        title = 'version',
        description = 'The bittensor version on the axon as str(int)',
        examples = 111,
        default = None,
        allow_mutation = True
    )
    _extract_version = pydantic.validator('version', pre=True, allow_reuse=True)(cast_int)

    # A unique monotonically increasing integer nonce associate with the terminal
    nonce: Optional[ int ] = pydantic.Field(
        title = 'nonce',
        description = 'A unique monotonically increasing integer nonce associate with the terminal generated from time.monotonic_ns()',
        examples = 111111,
        default = None,
        allow_mutation = True
    )
    _extract_nonce = pydantic.validator('nonce', pre=True, allow_reuse=True)(cast_int)

    # A unique identifier associated with the terminal, set on the axon side.
    uuid: Optional[ str ] = pydantic.Field(
        title = 'uuid',
        description = 'A unique identifier associated with the terminal',
        examples = "5ecbd69c-1cec-11ee-b0dc-e29ce36fec1a",
        default = None,
        allow_mutation = True
    )

    # The bittensor version on the terminal as an int.
    hotkey: Optional[ str ] = pydantic.Field(
        title = 'hotkey',
        description = 'The ss58 encoded hotkey string of the terminal wallet.',
        examples = "5EnjDGNqqWnuL2HCAdxeEtN2oqtXZw6BMBe936Kfy2PFz1J1",
        default = None,
        allow_mutation = True
    )

    # A signature verifying the tuple (axon_nonce, axon_hotkey, dendrite_hotkey, axon_uuid)
    signature: Optional[ str ] = pydantic.Field(
        title = 'signature',
        description = 'A signature verifying the tuple (nonce, axon_hotkey, dendrite_hotkey, uuid)',
        examples = "0x0813029319030129u4120u10841824y0182u091u230912u",
        default = None,
        allow_mutation = True
    )

class Synapse( pydantic.BaseModel ):

    class Config:
        validate_assignment = True

    def deserialize(self) -> 'Synapse':
        return self

    @pydantic.root_validator(pre=True)
    def set_name_type(cls, values):
        values['name'] = cls.__name__
        return values

    # Defines the http route name which is set on axon.attach( callable( request: RequestName ))
    name: Optional[ str ] = pydantic.Field(
        title = 'name',
        description = 'Defines the http route name which is set on axon.attach( callable( request: RequestName ))',
        examples = 'Forward',
        allow_mutation = True,
        default = None,
        repr = False
    )

    # The call timeout, set by the dendrite terminal.
    timeout: Optional[ float ] = pydantic.Field(
        title = 'timeout',
        description = 'Defines the total query length.',
        examples = 12.0,
        default = 12.0,
        allow_mutation = True,
        repr = False
    )
    _extract_timeout = pydantic.validator('timeout', pre=True, allow_reuse=True)(cast_float)

    # The call timeout, set by the dendrite terminal.
    total_size: Optional[ int ] = pydantic.Field(
        title = 'total_size',
        description = 'Total size of request body in bytes.',
        examples = 1000,
        default = 0,
        allow_mutation = True,
        repr = True
    )
    _extract_total_size = pydantic.validator('total_size', pre=True, allow_reuse=True)(cast_int)

    # The call timeout, set by the dendrite terminal.
    header_size: Optional[ int ] = pydantic.Field(
        title = 'header_size',
        description = 'Size of request header in bytes.',
        examples = 1000,
        default = 0,
        allow_mutation = True,
        repr = True
    )
    _extract_header_size = pydantic.validator('header_size', pre=True, allow_reuse=True)(cast_int)

    # The dendrite Terminal Information.
    dendrite: Optional[ TerminalInfo ] = pydantic.Field(
        title = 'dendrite',
        description = 'Dendrite Terminal Information',
        examples = "bt.TerminalInfo",
        default = TerminalInfo(),
        allow_mutation = True,
        repr = False
    )

    # A axon terminal information
    axon: Optional[ TerminalInfo ] = pydantic.Field(
        title = 'axon',
        description = 'Axon Terminal Information',
        examples = "bt.TerminalInfo",
        default = TerminalInfo(),
        allow_mutation = True,
        repr = False
    )

    def get_total_size(self) -> int: 
        self.total_size = get_size( self ); 
        return self.total_size
    
    @property
    def is_success(self) -> bool:
        return self.dendrite.status_code == 200
    
    @property
    def is_failure(self) -> bool:
        return self.dendrite.status_code != 200
    
    @property
    def is_timeout(self) -> bool:
        return self.dendrite.status_code == 408
    
    @property
    def is_blacklist(self) -> bool:
        return self.dendrite.status_code == 403
    
    @property
    def failed_verification(self) -> bool:
        return self.dendrite.status_code == 401

    def to_headers( self ) -> dict:
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
        headers = {
            'name': self.name,
            'timeout': str(self.timeout),
        }

        # Adding headers for 'axon' and 'dendrite' if they are not None
        headers.update({f'bt_header_axon_{k}': str(v) for k, v in self.axon.dict().items() if v is not None})
        headers.update({f'bt_header_dendrite_{k}': str(v) for k, v in self.dendrite.dict().items() if v is not None})

        # Getting the type hints for the properties of the instance
        property_type_hints = typing.get_type_hints(self)

        # Getting the fields of the instance
        instance_fields = self.__dict__

        # Iterating over the fields of the instance
        for field, value in instance_fields.items():
            # Skipping the field if it's already in the headers or its value is None
            if field in headers or value is None: 
                continue 

            # Adding the tensor shape and data type to the headers if the object is a tensor
            if isinstance(value, bittensor.Tensor):
                headers[f'bt_header_tensor_{field}'] = f'{value.shape}-{value.dtype}'

            # If the object is not optional, serializing it, encoding it, and adding it to the headers
            elif field in property_type_hints and 'typing.Optional' not in str(property_type_hints[field]):
                serialized_value = pickle.dumps(value)
                encoded_value = base64.b64encode(serialized_value).decode('utf-8')
                headers[f'bt_header_input_obj_{field}'] = encoded_value

        # Adding the size of the headers and the total size to the headers
        headers['header_size'] = str(sys.getsizeof(headers))
        headers['total_size'] = str(self.get_total_size())

        return headers
    
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
        inputs_dict = {'axon': {}, 'dendrite': {}}

        # Iterate over each item in the headers
        for key, value in headers.items():
            # Handle 'axon' headers
            if 'bt_header_axon_' in key:
                try:
                    new_key = key.split('bt_header_axon_')[1]
                    inputs_dict['axon'][new_key] = value
                except Exception as e:
                    bittensor.logging.error(f"Error while parsing 'axon' header {key}: {e}")
                    continue
            # Handle 'dendrite' headers
            elif 'bt_header_dendrite_' in key:
                try:
                    new_key = key.split('bt_header_dendrite_')[1]
                    inputs_dict['dendrite'][new_key] = value
                except Exception as e:
                    bittensor.logging.error(f"Error while parsing 'dendrite' header {key}: {e}")
                    continue
            # Handle 'tensor' headers
            elif 'bt_header_tensor_' in key:
                try:
                    new_key = key.split('bt_header_tensor_')[1]
                    shape, dtype = value.split('-')
                    # TODO: Verify if the shape and dtype values need to be converted before being used
                    inputs_dict[new_key] = bittensor.Tensor(shape=shape, dtype=dtype)
                except Exception as e:
                    bittensor.logging.error(f"Error while parsing 'tensor' header {key}: {e}")
                    continue
            # Handle 'input_obj' headers
            elif 'bt_header_input_obj' in key:
                try:
                    new_key = key.split('bt_header_input_obj_')[1]
                    # Skip if the key already exists in the dictionary
                    if new_key in inputs_dict:
                        continue
                    # Decode and load the serialized object
                    inputs_dict[new_key] = pickle.loads(base64.b64decode(value.encode('utf-8')))
                except Exception as e:
                    bittensor.logging.error(f"Error while parsing 'input_obj' header {key}: {e}")
                    continue
            else:
                bittensor.logging.trace(f"Unexpected key in headers: {key}")  # log unexpected keys

        # Assign the remaining known headers directly
        inputs_dict['timeout'] = headers.get('timeout', None)
        inputs_dict['name'] = headers.get('name', None)
        inputs_dict['header_size'] = headers.get('header_size', None)
        inputs_dict['total_size'] = headers.get('total_size', None)

        return inputs_dict


    @classmethod
    def from_headers(cls, headers: dict) -> 'Synapse':
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
    

def test_parse_headers_to_inputs():

    class Test(Synapse):
        key1: List[int]
        key2: bittensor.Tensor

    # Define a mock headers dictionary to use for testing
    headers = {
        'bt_header_axon_nonce': '111',
        'bt_header_dendrite_ip': '12.1.1.2',
        'bt_header_input_obj_key1': base64.b64encode(pickle.dumps([1,2,3,4])).decode('utf-8'),
        'bt_header_tensor_key2': '[3]-torch.float32',
        'timeout': '12',
        'name': 'Test',
        'header_size': '111',
        'total_size': '111',
    }

    # Run the function to test
    inputs_dict = Test.parse_headers_to_inputs(headers)

    # Check the resulting dictionary
    assert inputs_dict == {
        'axon': {'nonce': '111'}, 
        'dendrite': {'ip': '12.1.1.2'}, 
        'key1': [1, 2, 3, 4], 
        'key2': bittensor.Tensor(dtype='torch.float32', shape=[3]), 
        'timeout': '12', 
        'name': 'Test', 
        'header_size': '111', 
        'total_size': '111'
    }

def test_from_headers():
    class Test(Synapse):
        key1: List[int]
        key2: bittensor.Tensor

    # Define a mock headers dictionary to use for testing
    headers = {
        'bt_header_axon_nonce': '111',
        'bt_header_dendrite_ip': '12.1.1.2',
        'bt_header_input_obj_key1': base64.b64encode(pickle.dumps([1,2,3,4])).decode('utf-8'),
        'bt_header_tensor_key2': '[3]-torch.float32',
        'timeout': '12',
        'name': 'Test',
        'header_size': '111',
        'total_size': '111',
    }

    # Run the function to test
    synapse = Test.from_headers(headers)

    # Check that the resulting object is an instance of YourClass
    assert isinstance(synapse, Test)

    # Check the properties of the resulting object
    # Replace with actual checks based on the structure of your class
    assert synapse.axon.nonce == 111
    assert synapse.dendrite.ip == '12.1.1.2'
    assert synapse.key1 == [1,2,3,4]
    assert synapse.key2.shape == [3]
    assert synapse.key2.dtype == 'torch.float32'
    assert synapse.timeout == 12
    assert synapse.name == 'Test'
    assert synapse.header_size == 111
    assert synapse.total_size == 111

def test_synapse_create():
    # Create an instance of Synapse
    synapse = Synapse()

    # Ensure the instance created is of type Synapse
    assert isinstance(synapse, Synapse)

    # Check default properties of a newly created Synapse
    assert synapse.name == 'Synapse'
    assert synapse.timeout == 12.0
    assert synapse.header_size == 0
    assert synapse.total_size == 0

    # Convert the Synapse instance to a headers dictionary
    headers = synapse.to_headers()

    # Ensure the headers is a dictionary and contains the expected keys
    assert isinstance(headers, dict)
    assert 'timeout' in headers
    assert 'name' in headers
    assert 'header_size' in headers
    assert 'total_size' in headers

    # Ensure the 'name' and 'timeout' values match the Synapse's properties
    assert headers['name'] == 'Synapse'
    assert headers['timeout'] == '12.0'

    # Create a new Synapse from the headers and check its 'timeout' property
    next_synapse = synapse.from_headers(synapse.to_headers())
    assert next_synapse.timeout == 12.0

def test_custom_synapse():        
    # Define a custom Synapse subclass
    class Test(Synapse):
        a: int  # Carried through because required.
        b: int = None  # Not carried through headers
        c: typing.Optional[int]  # Not carried through headers
        d: typing.Optional[typing.List[int]]  # Not carried through headers
        e: typing.List[int]  # Not carried through headers

    # Create an instance of the custom Synapse subclass
    synapse = Test(a=1, c=3, d=[1,2,3,4], e=[1,2,3,4])

    # Ensure the instance created is of type Test and has the expected properties
    assert isinstance(synapse, Test)
    assert synapse.name == 'Test'
    assert synapse.a == 1
    assert synapse.b == None
    assert synapse.c == 3
    assert synapse.d == [1,2,3,4]
    assert synapse.e == [1,2,3,4]

    # Convert the Test instance to a headers dictionary
    headers = synapse.to_headers()

    # Ensure the headers contains 'a' but not 'b'
    assert 'bt_header_input_obj_a' in headers
    assert 'bt_header_input_obj_b' not in headers

    # Create a new Test from the headers and check its properties
    next_synapse = synapse.from_headers(synapse.to_headers())
    assert next_synapse.a == 1
    assert next_synapse.b == None
    assert next_synapse.c == None
    assert next_synapse.d == None
    assert next_synapse.e == [1,2,3,4]
