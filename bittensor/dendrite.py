from bittensor import bittensor_pb2_grpc as bittensor_grpc
from bittensor import bittensor_pb2
from bittensor.serializer import PyTorchSerializer
import bittensor

from loguru import logger
from typing import List, Tuple, Dict, Optional

import os
import grpc
import PIL
import torch
import torch.nn as nn

from torch.autograd.function import once_differentiable

DUMMY = torch.empty(0, requires_grad=True)  # dummy tensor that triggers autograd in RemoteExpert  

class Dendrite(nn.Module):
    def __init__(self, config: bittensor.Config):
        super().__init__()
        self._config = config
        self._remotes = {}
    
    def forward_text(self, synapses: List[bittensor_pb2.Synapse], x: List[ torch.Tensor ]) -> List[torch.Tensor]:
        r""" Forward text inputs to synapses.

            Args:
                synapses (:obj:`List[bittensor_pb2.Synapse]` of shape :obj:`(num_synapses)`, `required`): 
                    List of remote synapses which match length of x. Tensors from x are sent forward to these synapses.

                x (:obj:`List[torch.Tensor]` of shape :obj:`(num_synapses * [batch_size, sequence_len])`, `required`): 
                    List of tensors to send to corresponsing synapses. Tensors are text input_ids encoded using the
                    bittensor tokenizer of shape [batch_size, sequence_len].
            
            Returns:
                forwad_output (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, sequence_len, bittensor.network_size)`, `required`): 
                    Output encodings of inputs produced by remote synapses. Non-responses are zeroes of common shape.
        """
        if len(x[0].shape) != 2:
            raise ValueError('Text inputs should rank 2 with semantic shape: [batch_size, sequence_len]')
        return self.forward(synapses, x, bittensor_pb2.Modality.TEXT)
    
    def forward_image(self, synapses: List[bittensor_pb2.Synapse], x: List[ torch.Tensor ]) -> List[torch.Tensor]:
        r""" Forward image inputs to synapses.

            Args:
                synapses (:obj:`List[bittensor_pb2.Synapse]` of shape :obj:`(num_synapses)`, `required`): 
                    List of remote synapses which match length of x. Tensors from x are sent forward to these synapses.

                x (:obj:`List[torch.Tensor]` of shape :obj:`(num_synapses * [batch_size, channels, rows, cols])`, `required`): 
                    List of image-tensors to send to corresponsing synapses. Tensors are images encoded using the
                    torch.toTensor() or other encoding which produces the shape [batch_size, channels, rows, cols].
            
            Returns:
                forwad_output (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, sequence_len, bittensor.network_size)`, `required`): 
                    Output encodings of images produced by remote synapses. Non-responses are zeroes of common shape.
        """
        # TODO(const): Checks across all tensors and other shape checks.
        # TODO(const): Add sequence length.
        if len(x[0].shape) != 4:
            raise ValueError('Image inputs should be rank 4 with semantic shape: [batch_size, channels, rows, cols]')
        return self.forward(synapses, x, bittensor_pb2.Modality.IMAGE)
    
    def forward_tensor(self, synapses: List[bittensor_pb2.Synapse], x: List[ torch.Tensor ]) -> List[torch.Tensor]:
        r""" Forward tensor inputs to synapses.

            Args:
                synapses (:obj:`List[bittensor_pb2.Synapse]` of shape :obj:`(num_synapses)`, `required`): 
                    List of remote synapses which match length of x. Tensors from x are sent forward to these synapses.

                x (:obj:`List[torch.Tensor]` of shape :obj:`(num_synapses * [batch_size, sequence_len, feature_len])`, `required`): 
                    List of tensors to send to corresponsing synapses. Tensors are of arbitrary type and
                    with shape [batch_size, sequence_len, feature_len].
            
            Returns:
                forwad_output (:obj:`List[torch.FloatTensor]` of shape :obj:`num_synapses * (batch_size, sequence_len, bittensor.network_size)]`, `required`): 
                    Output encodings of tensors produced by remote synapses. Non-responses are zeroes of common shape.
        """
        if len(x[0].shape) != 3:
            raise ValueError('Tensor inputs should be rank 3 with semantic shape: [batch_size, sequence_len, feature_len]')
        return self.forward(synapses, x, bittensor_pb2.Modality.TENSOR)
    
    def forward(self, synapses: List[bittensor_pb2.Synapse], x: List[ torch.Tensor ], mode: bittensor_pb2.Modality) -> List[torch.Tensor]:
        r""" Forward tensor inputs to synapses.

            Args:
                synapses (:obj:`List[bittensor_pb2.Synapse]` of shape :obj:`(num_synapses)`, `required`): 
                    List of remote synapses which match length of x. Tensors from x are sent forward to these synapses.

                x (:obj:`List[torch.Tensor]` of shape :obj:`(num_synapses * [shape])`, `required`): 
                    List of tensors to send to corresponsing synapses. Tensors are of arbitrary type and shape depending on the 
                    modality.

                mode (:obj:`bittensor_pb2.Modality` of shape :obj:`(1)`, `required`): 
                    Bittensor forward modality type. Enum in [TEXT, IMAGE, TENSOR]
            
            Returns:
                forwad_output (:obj:`List[torch.FloatTensor]` of shape :obj:`num_synapses * (batch_size, sequence_len, bittensor.network_size)]`, `required`): 
                    Output encodings of tensors produced by remote synapses. Non-responses are zeroes of common shape.
        """
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
            results.append(remote_synapse(forward_inputs, mode))

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

    def forward(self, inputs: torch.Tensor, mode: bittensor_pb2.Modality) -> torch.Tensor:
        # TODO (const): consistend packing.
        # flattened = flatten(inputs)
        # Note: (hivemind) we send DUMMY to prevent torch from excluding expert from backward if no other inputs require grad
        outputs = _RemoteModuleCall.apply(self, DUMMY, inputs, mode)
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
    def forward(ctx, caller: RemoteSynapse, dummy: torch.Tensor, inputs: torch.Tensor, mode: bittensor_pb2.Modality) -> torch.Tensor:
        # Save for backward call.
        ctx.caller = caller
        ctx.mode = mode
        
        # Serialize inputs to bytes.         
        serialized_inputs = PyTorchSerializer.serialize( inputs, mode )

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
        
        # Make RPC call.
        try:
            response = ctx.caller.stub.Forward(request)                
        except grpc._channel._InactiveRpcError as ire:
            # Error making remote request.
            logger.error("Error making forward call to {} with error {}", RemoteSynapse, ire)
            return torch.zeros((inputs.size(0), bittensor.__network_dim__))

        # Deserialize outputs.
        try:
            output = PyTorchSerializer.deserialize_tensor(response.tensors[0])               
        except Exception as e:
            logger.error("Error deserializing responses with response {} with error {}", response.tensors[0], e)
            return torch.zeros((inputs.size(0), bittensor.__network_dim__))

        # Check batch_size.
        # TODO(const) check sequence_len when images have sequence len.
        if output.size(0) != inputs.size(0):    
            return torch.zeros((inputs.size(0), bittensor.__network_dim__))

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grads: torch.Tensor) -> Optional[torch.Tensor]:
            
        # Serialize inputs to bytes.
        serialized_grads = PyTorchSerializer.serialize_tensor(grads)
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
        
        deserialized_grad_inputs = torch.zeros(1,1)
        
        try:
            # Attain backward response
            response = ctx.caller.stub.Backward(request)
            deserialized_grad_inputs = PyTorchSerializer.deserialize (response.tensors[0])
            return (None, None, deserialized_grad_inputs, None)        
        except grpc._channel._InactiveRpcError as ire:
            return (None, None, deserialized_grad_inputs, None)
