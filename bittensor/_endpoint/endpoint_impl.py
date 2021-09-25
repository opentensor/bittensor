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


class Endpoint:
    """ Implementation of an endpoint object, with attr hotkey, coldkey, modality and ip
    """
    def __init__( self, version: int, uid:int, hotkey:str, ip:str, ip_type:int, port:int , modality:int, coldkey:str ):
        self.version = version
        self.uid = uid
        self.hotkey = hotkey
        self.ip = net.int_to_ip (ip)
        self.ip_type = ip_type
        self.port = port
        self.coldkey = coldkey
        self.modality = modality
        if self.check_format():
            # possibly throw a warning here.
            pass
            

    def assert_format( self ):
        """ Asserts that the endpoint has a valid format
            Raises:
                Multiple assertion errors.
        """
        assert self.version > 0, 'endpoint version must be positive. - got {}'.format(self.version)
        assert self.version < 999, 'endpoint version must be less than 999. - got {}'.format(self.version)
        assert self.uid >= 0 and self.uid < 4294967295, 'endpoint uid must positive and be less than u32 max: 4294967295. - got {}'.format(self.uid)
        assert len(self.ip) < 8*4, 'endpoint ip string must have length less than 8*4. - got {}'.format(self.ip) 
        assert self.ip_type == 4 or self.ip_type == 6 , 'endpoint ip_type must be either 4 or 6.- got {}'.format(self.ip_type)
        assert self.port > 0 and self.port < 65535 , 'port must be positive and less than 65535 - got {}'.format(self.port)
        assert len(self.coldkey) == 48, 'coldkey string must be length 48 - got {}'.format(self.coldkey)
        assert len(self.hotkey) == 48, 'hotkey string must be length 48 - got {}'.format(self.hotkey)
        assert self.modality == 0, 'modality must be 0 (for now) - got {}'.format(self.modality)

    def check_format( self ) -> bool:
        """ Checks that the endpoint has a valid format.
            Raises:
                is_valid_format (bool):
                    True if the endpoint has a valid format.
        """
        if self.version < 0:
            # 'endpoint version must be positive.'
            return False
        if self.version > 999:
            # 'endpoint version must be less than 999.'
            return False
        if self.uid < 0 or self.uid > 4294967295: 
            # 'endpoint uid must positive and be less than u32 max: 4294967295.'
            return False
        if len(self.ip) > 8*4:
            # 'endpoint ip string must have length less than 8*4.'
            return False
        if self.ip_type != 4 and self.ip_type != 6:
            # 'endpoint ip_type must be either 4 or 6.'
            return False
        if self.port < 0 or self.port > 65535:
            # 'port must be positive and less than 65535'
            return False
        if len(self.coldkey) != 48:
            # 'coldkey string must be length 48'
            return False
        if len(self.hotkey) != 48:
            # 'hotkey string must be length 48'
            return False
        if self.modality != 0:
            # 'modality must be 0 (for now)'
            return False
        return True
    
    def to_tensor( self ) -> torch.LongTensor: 
        """ Return the specification of an endpoint as a tensor
        """ 
        string_json = self.dumps()
        bytes_json = bytes(string_json, 'utf-8')
        ints_json = list(bytes_json)
        if len(ints_json) > 250:
            raise ValueError('Endpoint {} representation is too large, got size {} should be less than 250'.format(self, len(ints_json)))
        ints_json += [-1] * (250 - len(ints_json))
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
                'modality': self.modality,
            })

    def ip_str(self) -> str:
        """ Return the whole ip as string
        """ 
        return net.ip__str__(self.ip_type, self.ip, self.port)

    def __eq__ (self, other: 'Endpoint'):
        if self.version == other.version and self.uid == other.uid and self.ip == other.ip and self.port == other.port and self.ip_type == other.ip_type and  self.coldkey == other.coldkey and self.hotkey == other.hotkey and self.modality == other.modality:
            return True
        else:
            return False 

    def __str__(self):
        return "Endpoint({}, {}, {}, {})".format(str(self.ip_str()), str(self.uid), str(self.hotkey), str(self.coldkey))
    
    def __repr__(self):
        return self.__str__()
