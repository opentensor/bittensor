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

import torch
import json
import bittensor
from loguru import logger
import bittensor.utils.networking as net

class Endpoint:

    def __init__( self, uid:int, hotkey:str, ip:str, ip_type:int, port:int , modality:int, coldkey:str ):
        self.uid = uid
        self.hotkey = hotkey
        self.ip = net.int_to_ip (ip)
        self.ip_type = ip_type
        self.port = port
        self.coldkey = coldkey
        self.modality = modality
    
    def to_tensor( self ) -> torch.LongTensor:  
        string_json = self.dumps()
        bytes_json = bytes(string_json, 'utf-8')
        ints_json = list(bytes_json)
        endpoint_tensor = torch.tensor( ints_json, dtype=torch.int64, requires_grad=False)
        return endpoint_tensor

    def dumps(self):
        return json.dumps(
            {
                'uid': self.uid,
                'hotkey': self.hotkey,
                'ip': self.ip,
                'ip_type': self.ip_type,
                'port': self.port,
                'coldkey': self.coldkey,
                'modality': self.modality,
            })

    def ip_str(self) -> str:
        return net.ip__str__(self.ip_type, self.ip, self.port)

    def __str__(self):
        return "<endpoint uid: %s hotkey: %s ip: %s modality: %s coldkey: %s>" % (self.uid, self.hotkey, self.ip_str(), self.modality, self.coldkey)
    
    def __repr__(self):
        return self.__str__()






