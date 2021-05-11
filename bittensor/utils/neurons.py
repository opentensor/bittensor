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

class NeuronEndpoint:
    uid: int
    hotkey : str
    ip: str
    ip_type: int
    port: int
    modality: int
    coldkey : str

    def __init__( self, uid:int, hotkey:str, ip:str, ip_type:int, port:int , modality:int, coldkey:str ):
        self.uid = uid
        self.hotkey = hotkey
        self.ip = net.int_to_ip (ip)
        self.ip_type = ip_type
        self.port = port
        self.coldkey = coldkey
        self.modality = modality

    @staticmethod
    def from_dict(neuron_dict: dict) -> 'NeuronEndpoint':
        return NeuronEndpoint(
            uid = neuron_dict['uid'], 
            hotkey = neuron_dict['hotkey'], 
            port = neuron_dict['port'],
            ip = neuron_dict['ip'], 
            ip_type = neuron_dict['ip_type'], 
            modality = neuron_dict['modality'], 
            coldkey = neuron_dict['coldkey']
        )

    @staticmethod
    def from_tensor( tensor: torch.LongTensor) -> 'NeuronEndpoint':
        neuron_string = bittensor.__tokenizer__.decode( tensor )
        neuron_dict = json.loads( neuron_string )
        return NeuronEndpoint.from_dict(neuron_dict)
    
    def to_tensor( self ) -> torch.LongTensor:  
        string_json = self.dumps()
        neuron_tokens = bittensor.__tokenizer__( string_json )['input_ids']
        neuron_tensor = torch.tensor( neuron_tokens, dtype = torch.int64, requires_grad=False)
        return neuron_tensor

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

    def __str__(self):
        return "<neuron uid: %s hotkey: %s ip: %s modality: %s coldkey: %s>" % (self.uid, self.hotkey, net.ip__str__(self.ip_type, self.ip, self.port), self.modality, self.coldkey)


class NeuronEndpoints(list):
    @staticmethod
    def from_list(input : list):
        output = NeuronEndpoints()

        if not input:
            return output

        for row in input:
            data = row[1]  # Attributes of the neuron are stored in the second element of the list
            output.append(NeuronEndpoint.from_dict( data ))

        return output

    def has_uid(self,uid):
        neurons = filter(lambda x: x.uid == uid, self)
        return len(list(neurons)) > 0

    def get_by_uid(self, uid):
        neurons = Neurons(filter(lambda x: x.uid == uid, self))
        return None if len(neurons) == 0 else neurons[0]

    def __str__(self):
        y = map(lambda x : x.__str__(), self)
        return "".join(y)





