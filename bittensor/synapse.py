""" Create and init Axon, whcih services Forward and Backward requests from other neurons.
"""
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

    def to_headers(self) -> dict:
        headers = {
            'name': self.name,
            'timeout': str(self.timeout),
        }
        # Fill axon and dendrite headers.
        headers.update( { str('bt_header_axon_'+k):str(v) for k, v in self.axon.dict().items() if v != None} )
        headers.update( { str('bt_header_dendrite_'+k):str(v) for k, v in self.dendrite.dict().items() if v != None})

        # Iterate over fields, if an object is a tensor
        # add the tensor shape to the headers. 
        metadata = typing.get_type_hints(self)
        fields = self.__dict__
        for field, value in fields.items():
            if field in headers: continue
            if not value: continue 
            if isinstance( value, bittensor.Tensor ):
                headers[ 'bt_header_tensor_' + str(field) ] = str(value.shape) + '-' + str(value.dtype)
            else:
                # If the object is not optional we must add it to the headers.
                if field in metadata:
                    if 'typing.Optional' not in str(metadata[field]):
                        headers[ 'bt_header_input_obj_' + str(field) ] = base64.b64encode( pickle.dumps(value) ).decode('utf-8')

        headers['header_size'] = str( sys.getsizeof( headers ) )
        headers['total_size'] = str( self.get_total_size() )
        return headers
    
    @classmethod
    def _headers_to_inputs_dict( cls, headers: dict ) -> 'Synapse':
        inputs_dict = {}
        inputs_dict['axon'] = {}
        inputs_dict['dendrite'] = {}
        for k, v in headers.items():
            if 'bt_header_axon_' in k:
                try:
                    k = k.split('bt_header_axon_')[1]
                    inputs_dict['axon'][k] = v
                except: continue
            elif 'bt_header_dendrite_' in k:
                try:
                    k = k.split('bt_header_dendrite_')[1]
                    inputs_dict['dendrite'][k] = v
                except: continue
            elif 'bt_header_tensor_' in k:
                try:
                    k = k.split('bt_header_tensor_')[1]
                    shape = v.split('-')[0]
                    dtype = v.split('-')[1]
                    inputs_dict[k] = bittensor.Tensor( shape = shape, dtype = dtype ) 
                except: continue
            elif 'bt_header_input_obj' in k:
                try:
                    k = k.split('bt_header_input_obj_')[1]
                    if k in inputs_dict: continue
                    inputs_dict[k] = pickle.loads( base64.b64decode( v.encode('utf-8')  ) )
                except: continue
            else:
                continue
        inputs_dict['timeout'] = headers.get('timeout', None)
        inputs_dict['name'] = headers.get('name', None)
        inputs_dict['header_size'] = headers.get('header_size', None)
        inputs_dict['total_size'] = headers.get('total_size', None)
        return inputs_dict

    @classmethod
    def from_headers( cls, headers: dict ) -> 'Synapse':
        input_dict = cls._headers_to_inputs_dict( headers )
        synapse = cls( **input_dict )
        return synapse