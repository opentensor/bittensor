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

from . import endpoint_impl

class endpoint:

    def __new__( cls, uid:int, hotkey:str, ip:str, ip_type:int, port:int , modality:int, coldkey:str ) -> 'bittensor.Endpoint':
        return endpoint_impl.Endpoint( uid, hotkey, ip, ip_type, port, modality, coldkey )

    @staticmethod
    def from_dict(neuron_dict: dict) -> 'bittensor.Endpoint':
        return endpoint_impl.Endpoint(
            uid = neuron_dict['uid'], 
            hotkey = neuron_dict['hotkey'], 
            port = neuron_dict['port'],
            ip = neuron_dict['ip'], 
            ip_type = neuron_dict['ip_type'], 
            modality = neuron_dict['modality'], 
            coldkey = neuron_dict['coldkey']
        )
    
    @staticmethod
    def from_tensor( tensor: torch.LongTensor) -> 'bittensor.Endpoint':
        neuron_string = bittensor.__tokenizer__.decode( tensor )
        neuron_dict = json.loads( neuron_string )
        return endpoint.from_dict(neuron_dict)





