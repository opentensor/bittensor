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
from typing import Union
import torch
import bittensor

from . import endpoint_impl

MAX_IP_LENGTH = 8*4
MAX_VERSION = 999
SS58_LENGTH = 48
MAXPORT = 65535
MAXUID = 4294967295
ACCEPTABLE_IPTYPES = [4,6,0]
ACCEPTABLE_PROTOCOLS = [0,4] # TODO
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
        coldkey:str,
        protocol:int = 0, # TODO: activate protocol
        modality: int = 0 # TODO: remove modality
    ) -> 'bittensor.Endpoint':
        endpoint.assert_format(
            version=version,
            uid = uid,
            ip = ip,
            ip_type = ip_type,
            port = port,
            coldkey = coldkey,
            hotkey = hotkey,
            protocol=protocol
        )
        return endpoint_impl.Endpoint( version, uid, hotkey, ip, ip_type, port, protocol, coldkey, modality )


    @staticmethod
    def from_neuron( neuron: Union['bittensor.NeuronInfo', 'bittensor.NeuronInfoLite'] ) -> 'bittensor.Endpoint':
        """
        endpoint.assert_format(
            version = neuron.version,
            uid = neuron.uid, 
            hotkey = neuron.hotkey, 
            port = neuron.axon_info.port,
            ip = neuron.axon_info.ip, 
            ip_type = neuron.axon_info.ip_type, 
            protocol = neuron.axon_info.protocol, 
            coldkey = neuron.coldkey
        )
        """
        if neuron.is_null:
            raise ValueError('Cannot create endpoint from null neuron')
        
        if hasattr(neuron, 'axon_info'): #if config.subtensor.network == 'finney'
            return endpoint_impl.Endpoint(
                version = neuron.axon_info.version,
                uid = neuron.uid, 
                hotkey = neuron.hotkey, 
                port = neuron.axon_info.port,
                ip = neuron.axon_info.ip, 
                ip_type = neuron.axon_info.ip_type, 
                protocol = neuron.axon_info.protocol,
                coldkey = neuron.coldkey
            )
        else:
            return endpoint_impl.Endpoint(
                version = neuron.version,
                uid = neuron.uid, 
                hotkey = neuron.hotkey, 
                port = neuron.port,
                ip = neuron.ip, 
                ip_type = neuron.ip_type, 
                modality = neuron.modality, 
                coldkey = neuron.coldkey,
                protocol = None
            )

    @staticmethod
    def from_dict(endpoint_dict: dict) -> 'bittensor.Endpoint':
        """ Return an endpoint with spec from dictionary
        """
        if not endpoint.assert_format(
            version = endpoint_dict['version'],
            uid = endpoint_dict['uid'], 
            hotkey = endpoint_dict['hotkey'], 
            port = endpoint_dict['port'],
            ip = endpoint_dict['ip'], 
            ip_type = endpoint_dict['ip_type'], 
            protocol = endpoint_dict['protocol'], 
            coldkey = endpoint_dict['coldkey']
        ):
            raise ValueError('Invalid endpoint dict')
        return endpoint_impl.Endpoint(
            version = endpoint_dict['version'],
            uid = endpoint_dict['uid'], 
            hotkey = endpoint_dict['hotkey'], 
            port = endpoint_dict['port'],
            ip = endpoint_dict['ip'], 
            ip_type = endpoint_dict['ip_type'], 
            protocol = endpoint_dict['protocol'], 
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
            try:
                return endpoint.from_dict(endpoint_dict)
            except ValueError:
                return endpoint.dummy()

    @staticmethod
    def dummy():
        return endpoint_impl.Endpoint(uid=0, version=0, hotkey = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX", ip_type = 4, ip = '0.0.0.0', port = 0, protocol= 0, coldkey = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

    @staticmethod
    def assert_format(
            version: int,
            uid:int, 
            hotkey:str, 
            ip:str, 
            ip_type:int, 
            port:int, 
            protocol:int, 
            coldkey:str 
        ) -> bool:
        """ Asserts that the endpoint has a valid format
        """
        try:
            assert version >= 0, 'endpoint version must be positive. - got {}'.format(version)
            assert version <= MAX_VERSION, 'endpoint version must be less than 999. - got {}'.format(version)
            assert uid >= 0 and uid <= MAXUID, 'endpoint uid must positive and be less than u32 max: 4294967295. - got {}'.format(uid)
            assert len(ip) < MAX_IP_LENGTH, 'endpoint ip string must have length less than 8*4. - got {}'.format(ip) 
            assert ip_type in ACCEPTABLE_IPTYPES, 'endpoint ip_type must be either 4 or 6.- got {}'.format(ip_type)
            assert port >= 0 and port < MAXPORT , 'port must be positive and less than 65535 - got {}'.format(port)
            assert len(coldkey) == SS58_LENGTH, 'coldkey string must be length 48 - got {}'.format(coldkey)
            assert len(hotkey) == SS58_LENGTH, 'hotkey string must be length 48 - got {}'.format(hotkey)
            # TODO
            assert protocol in ACCEPTABLE_PROTOCOLS, 'protocol must be 0 (for now) - got {}'.format(protocol)

            return True
        except AssertionError:
            return False


