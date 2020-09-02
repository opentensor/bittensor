from bittensor import bittensor_pb2_grpc as bittensor_grpc
from bittensor import bittensor_pb2
from bittensor.serializer import PyTorchSerializer
import bittensor

from loguru import logger
from typing import List, Tuple, Dict, Optional

import os
import grpc
import torch
import torch.nn as nn

from torch.autograd.function import once_differentiable

DUMMY = torch.empty(0, requires_grad=True)  # dummy tensor that triggers autograd in RemoteExpert  

class Dendrite:
    def __init__(self, config: bittensor.Config):
        self._config = config
        self._remotes = {}
        
    # TODO (const): connection handling.
    # Cleaning remote connections and updating signatures.
    def run(self):
        pass
        
    def forward(self, synapses: List[bittensor_pb2.Synapse], x: List[torch.Tensor]) -> List[torch.Tensor]:
        """ forward tensor processes """
        results = []
        for idx, synapse in enumerate(synapses):
            forward_inputs = x[ idx ]
                        
            # Get or create remote_synapse.
            remote_synapse = None
            if synapse.synapse_key in self._remotes:
                remote_synapse = self._remotes[synapse.synapse_key]
            else:
                # Create remote connection.
                remote_synapse = RemoteSynapse (synapse, self._config)
                self._remotes[synapse.synapse_key] = remote_synapse
                
            # Call remote synapse.
            results.append(remote_synapse(forward_inputs))
        return results
    
# NOTE: (const) This code has been ported from hivemind thanks to Yozh and Max.
# Credit to them for designing this structure and api around torch. Here being ported to 
# bittensor, and eventually should interact seemlessly with hivemind nodes as well.
# TODO (const): needs to check shapes/ input types/ other.
class RemoteSynapse(nn.Module):
    """ Class which bundles a grpc connection to a remote host as a standard auto-grad torch.nn.Module.
    """
    def __init__(self, synapse: bittensor_pb2.Synapse, config: bittensor.Config):
        super().__init__()
        self.synapse = synapse
        self.local_neuron_key = config.neuron_key       
        # Loop back if the synapse is local.
        if synapse.address == config.remote_ip:
            self.endpoint = 'localhost:' + synapse.port
        else:
            self.endpoint = synapse.address + ':' + synapse.port
        # TODO(const): should accept defaults. config = bittensor.config_or_defaults(config) 
        
        self.channel = grpc.insecure_channel(self.endpoint, options=[
                ('grpc.max_send_message_length', -1),
                ('grpc.max_receive_message_length', -1)])
        self.stub = bittensor_grpc.BittensorStub(self.channel)        
        # TODO(const): setter and getters for signature and nounce.
        self.signature = None
        self.nounce = None

    def __del__(self):
        if self.channel is not None:
            self.channel.close()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # TODO (const) compare schema
        # assert schema shape and size. raise type error.
        # if not check_schema(inputs, self.synapse.forward_schema):
        #    raise TypeError(f"Inputs do not match expert input schema. Did you pass the right number of parameters?")
        # TODO (const): consistend packing.
        # flattened = flatten(inputs)
        # Note: (hivemind) we send DUMMY to prevent torch from excluding expert from backward if no other inputs require grad
        outputs = _RemoteModuleCall.apply(self, DUMMY, inputs)
        # TODO (const) consitent unpacking
        # return unpack_to_schema(outputs, structure = self.synapse.output_schema) 
        return outputs
        
# Adapted from hivemind. Thanks Yozh.
class _RemoteModuleCall(torch.autograd.Function):
    """ Internal autograd-friendly call of a remote module over grpc"""

    # TODO (const) signatures + nounce.
    # TODO (const) check schema.
    # TODO (const) should take multiple input tensors and kwargs.
    @staticmethod
    def forward(ctx, caller: RemoteSynapse, dummy: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        # Save for backward call.
        ctx.caller = caller
        
        # Serialize inputs to bytes.
        serialized_inputs = PyTorchSerializer.serialize(inputs)
        ctx.serialized_inputs = serialized_inputs
        
        # Build request for forward.
        request = bittensor_pb2.TensorMessage( 
                                                version = bittensor.__version__,
                                                neuron_key = ctx.caller.local_neuron_key,
                                                synapse_key = ctx.caller.synapse.synapse_key,
                                                nounce = ctx.caller.nounce,
                                                signature = ctx.caller.signature,
                                                tensors = [serialized_inputs]
                                            )
        
        # Make rpc call.
        response = ctx.caller.stub.Forward(request)
                
        # Deserialize outputs and return.
        outputs = PyTorchSerializer.deserialize(response.tensors[0])
        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, grads: torch.Tensor) -> Optional[torch.Tensor]:
            
        # Serialize inputs to bytes.
        serialized_grads = PyTorchSerializer.serialize(grads)
        serialized_inputs = ctx.serialized_inputs
        
        # Build request for forward.
        request = bittensor_pb2.TensorMessage( 
                                                version = bittensor.__version__,
                                                neuron_key = ctx.caller.local_neuron_key,
                                                synapse_key = ctx.caller.synapse.synapse_key,
                                                nounce = ctx.caller.nounce,
                                                signature = ctx.caller.signature,
                                                tensors = [serialized_inputs, serialized_grads]
                                            )
        
        # Attain backward response
#        print ('dendrite ->', request)
        response = ctx.caller.stub.Backward(request)

        # Deserialize grad responses.
        deserialized_grad_inputs = PyTorchSerializer.deserialize(response.tensors[0])

        # Return grads
        return (None, None, deserialized_grad_inputs)