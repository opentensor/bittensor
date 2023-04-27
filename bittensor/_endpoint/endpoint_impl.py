""" Implementation of an endpoint object, with attr hotkey, coldkey, modality and ip
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

import bittensor
import json
import torch
import bittensor.utils.networking as net

MAX_IP_LENGTH = 8*4
MAX_VERSION = 999
SS58_LENGTH = 48
MAXPORT = 65535
MAXUID = 4294967295
ACCEPTABLE_IPTYPES = [4,6]
ACCEPTABLE_PROTOCOLS = [0] # TODO
ENDPOINT_BUFFER_SIZE = 250

class Endpoint:
    """ Implementation of an endpoint object, with attr hotkey, coldkey, modality and ip
    """
    def __init__( self, version: int, uid:int, hotkey:str, ip:str, ip_type:int, port:int , protocol:int, coldkey:str, modality: int = 0 ):
        self.version = version
        self.uid = uid
        self.hotkey = hotkey
        self.ip = net.int_to_ip (ip)
        self.ip_type = ip_type
        self.port = port
        self.coldkey = coldkey
        self.protocol = protocol

        # TODO: remove modality from endpoint.
        self.modality = modality


    def assert_format( self ) -> bool:
        """ Asserts that the endpoint has a valid format
        """
        try:
            assert self.version > 0, 'endpoint version must be positive. - got {}'.format(self.version)
            assert self.version < MAX_VERSION, 'endpoint version must be less than 999. - got {}'.format(self.version)
            assert self.uid >= 0 and self.uid < MAXUID, 'endpoint uid must positive and be less than u32 max: 4294967295. - got {}'.format(self.uid)
            assert len(self.ip) < MAX_IP_LENGTH, 'endpoint ip string must have length less than 8*4. - got {}'.format(self.ip) 
            assert self.ip_type in ACCEPTABLE_IPTYPES, 'endpoint ip_type must be either 4 or 6.- got {}'.format(self.ip_type)
            assert self.port > 0 and self.port < MAXPORT , 'port must be positive and less than 65535 - got {}'.format(self.port)
            assert len(self.coldkey) == SS58_LENGTH, 'coldkey string must be length 48 - got {}'.format(self.coldkey)
            assert len(self.hotkey) == SS58_LENGTH, 'hotkey string must be length 48 - got {}'.format(self.hotkey)
            assert self.protocol in ACCEPTABLE_PROTOCOLS, 'protocol must be 0 (for now) - got {}'.format(self.protocol)

            return True
        except AssertionError as e:
            return False

    @property
    def is_serving(self) -> bool:
        """ True if the endpoint is serving.
        """
        if self.ip == '0.0.0.0':
            return False
        else:
            return True

    def check_format( self ) -> bool:
        """ Checks that the endpoint has a valid format.
            Raises:
                is_valid_format (bool):
                    True if the endpoint has a valid format.
        """
        if self.version < 0:
            # 'endpoint version must be positive.'
            return False
        if self.version > MAX_VERSION:
            # 'endpoint version must be less than 999.'
            return False
        if self.uid < 0 or self.uid > MAXUID: 
            # 'endpoint uid must positive and be less than u32 max: 4294967295.'
            return False
        if len(self.ip) > MAX_IP_LENGTH:
            # 'endpoint ip string must have length less than 8*4.'
            return False
        if self.ip_type != 4 and self.ip_type != 6:
            # 'endpoint ip_type must be either 4 or 6.'
            return False
        if self.port < 0 or self.port > MAXPORT:
            # 'port must be positive and less than 65535'
            return False
        if len(self.coldkey) != SS58_LENGTH:
            # 'coldkey string must be length 48'
            return False
        if len(self.hotkey) != SS58_LENGTH:
            # 'hotkey string must be length 48'
            return False
        if self.protocol not in ACCEPTABLE_PROTOCOLS:
            # 'protocol must be 0 (for now)'
            return False
        return True
    
    def to_tensor( self ) -> torch.LongTensor: 
        """ Return the specification of an endpoint as a tensor
        """ 
        string_json = self.dumps()
        bytes_json = bytes(string_json, 'utf-8')
        ints_json = list(bytes_json)
        if len(ints_json) > ENDPOINT_BUFFER_SIZE:
            raise ValueError('Endpoint {} representation is too large, got size {} should be less than {}'.format(self, len(ints_json), ENDPOINT_BUFFER_SIZE))
        ints_json += [-1] * (ENDPOINT_BUFFER_SIZE - len(ints_json))
        endpoint_tensor = torch.tensor( ints_json, dtype=torch.int64, requires_grad=False)
        return endpoint_tensor

    def dumps(self):
        """ Return json with the endpoints's specification
        """ 
        return json.dumps(
            {
                'version': self.version,
                'uid': self.uid,
                'hotkey': self.hotkey,
                'ip': self.ip,
                'ip_type': self.ip_type,
                'port': self.port,
                'coldkey': self.coldkey,
                'protocol': self.protocol,
                'modality': self.modality,
            })

    def ip_str(self) -> str:
        """ Return the whole ip as string
        """ 
        return net.ip__str__(self.ip_type, self.ip, self.port)

    def __eq__ (self, other: 'Endpoint'):
        if other == None:
            return False
        if self.version == other.version and self.uid == other.uid and self.ip == other.ip and self.port == other.port and self.ip_type == other.ip_type and  self.coldkey == other.coldkey and self.hotkey == other.hotkey and self.protocol == other.protocol:
            return True
        else:
            return False 

    def __str__(self):
        return "Endpoint({}, {}, {}, {})".format(str(self.ip_str()), str(self.uid), str(self.hotkey), str(self.coldkey))
    
    def __repr__(self):
        return self.__str__()
