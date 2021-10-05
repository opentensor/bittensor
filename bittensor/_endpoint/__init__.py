""" Create and init endpoint object, with attr hotkey, coldkey, modality and ip
"""
# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

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

import json
import torch
import bittensor

from . import endpoint_impl

ENDPOINT_BUFFER_SIZE = 250

class endpoint:
    """ Create and init neuron object, with attr hotkey, coldkey, modality and ip
    """

    def __new__( 
        cls, 
        version: int,
        uid:int, 
        hotkey:str, 
        ip:str, 
        ip_type:int, 
        port:int, 
        modality:int, 
        coldkey:str 
    ) -> 'bittensor.Endpoint':
        return endpoint_impl.Endpoint( version, uid, hotkey, ip, ip_type, port, modality, coldkey )

    @staticmethod
    def from_dict(endpoint_dict: dict) -> 'bittensor.Endpoint':
        """ Return an endpoint with spec from dictionary
        """
        return endpoint_impl.Endpoint(
            version = endpoint_dict['version'],
            uid = endpoint_dict['uid'], 
            hotkey = endpoint_dict['hotkey'], 
            port = endpoint_dict['port'],
            ip = endpoint_dict['ip'], 
            ip_type = endpoint_dict['ip_type'], 
            modality = endpoint_dict['modality'], 
            coldkey = endpoint_dict['coldkey']
        )
    
    @staticmethod
    def from_tensor( tensor: torch.LongTensor) -> 'bittensor.Endpoint':
        """ Return an endpoint with spec from tensor
        """
        if len(tensor.shape) == 2:
            if tensor.shape[0] != 1:
                error_msg = 'Endpoints tensor should have a single first dimension or none got {}'.format( tensor.shape[0] )
                raise ValueError(error_msg)
            tensor = tensor[0]

        if tensor.shape[0] != ENDPOINT_BUFFER_SIZE:
            error_msg = 'Endpoints tensor should be length {}, got {}'.format( tensor.shape[0], ENDPOINT_BUFFER_SIZE)
            raise ValueError(error_msg)
            
        endpoint_list = tensor.tolist()
        if -1 in endpoint_list:
            endpoint_list = endpoint_list[ :endpoint_list.index(-1)]
            
        if len(endpoint_list) == 0:
            return endpoint.dummy()
        else:
            endpoint_bytes = bytearray( endpoint_list )
            endpoint_string = endpoint_bytes.decode('utf-8')
            endpoint_dict = json.loads( endpoint_string )
            return endpoint.from_dict(endpoint_dict)

    @staticmethod
    def dummy():
        return endpoint_impl.Endpoint(uid=-1,version=0, hotkey = "", ip_type = 4, ip = '0.0.0.0', port = 0, modality= 0, coldkey = "")





