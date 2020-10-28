from loguru import logger
import os
import grpc
import torch
import torch.nn as nn
from torch.autograd.function import once_differentiable
from typing import List, Tuple, Dict, Optional

import bittensor
from bittensor import bittensor_pb2_grpc as bittensor_grpc
from bittensor import bittensor_pb2
from bittensor.serializer import PyTorchSerializer
from bittensor.exceptions.Exceptions import EmptyTensorException, ResponseShapeException, SerializationException

# dummy tensor that triggers autograd in RemoteExpert
DUMMY = torch.empty(0, requires_grad=True)


class Dendrite(nn.Module):
    r"""
    This is the bittensr object used to make calls to the network. It can be used like a normal torch nn.Module
    and is differentiable. Messages passed through this module will be sent to synapse objects, either remote
    or local, and return response torch tensors. Gradients passing through this module on a .backward() call will trigger
    the Backward rpc calls, passing gradients to the remote synapse instances called during the Forward operation.   

    Args:
        config (:obj:`bittensor.Config`, `required`):
            Bittensor config object.
    Examples::

        >>> from bittensor

        >>> # Initialize config defaults.
        >>> config = bittensor.Config()

        >>> # Build metagraph object for network connectivity
        >>> metagraph = bittensor.Metagraph(config)
        >>> metagraph.start() # Begin gossip.

        >>> # Create Dendrite nn.module
        >>> dendrite = bittensor.Dendrite(config)

        >>> if len(metagraph.synapses()) > 0
        >>>     requests = [torch.rand(10, 100) for _ in metagraph.synapses()] # Random query tensors.
        >>>     responses = dendrite (requests, metagraph.synapses()) # Make network queries.
        >>>     responses[0].backward() # Backprop through dendrite.
    """
    def __init__ (
        self, 
        config: bittensor.Config
    ):
        super().__init__()
        self._config = config
        self._remotes = {}

    def forward_text(self, synapses: List[bittensor_pb2.Synapse],
                     x: List[torch.Tensor]) -> List[torch.Tensor]:
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
            error_msg = 'Text inputs should rank 2 with semantic shape: [batch_size, sequence_len]'
            raise ValueError(error_msg)
        return self.forward(synapses, x, bittensor_pb2.Modality.TEXT)

    def forward_image(self, synapses: List[bittensor_pb2.Synapse],
                      x: List[torch.Tensor]) -> List[torch.Tensor]:
        r""" Forward image inputs to synapses.

            Args:
                synapses (:obj:`List[bittensor_pb2.Synapse]` of shape :obj:`(num_synapses)`, `required`): 
                    List of remote synapses which match length of x. Tensors from x are sent forward to these synapses.

                x (:obj:`List[torch.Tensor]` of shape :obj:`(num_synapses * [batch_size, sequence_len, channels, rows, cols])`, `required`): 
                    List of image-tensors to send to corresponsing synapses. Tensors are images encoded using the
                    torch.toTensor() or other encoding which produces the shape [batch_size, channels, rows, cols].
            
            Returns:
                forwad_output (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, sequence_len, bittensor.network_size)`, `required`): 
                    Output encodings of images produced by remote synapses. Non-responses are zeroes of common shape.
        """
        # TODO(const): Checks across all tensors and other shape checks.
        if len(x[0].shape) != 5:
            error_msg = 'Image inputs should be rank 5 with semantic shape: [batch_size, sequence_dim, channels, rows, cols]'
            raise ValueError(error_msg)
        return self.forward(synapses, x, bittensor_pb2.Modality.IMAGE)

    def forward_tensor(self, synapses: List[bittensor_pb2.Synapse],
                       x: List[torch.Tensor]) -> List[torch.Tensor]:
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
            error_msg = 'Tensor inputs should be rank 3 with semantic shape: [batch_size, sequence_len, feature_len]'
            raise ValueError(error_msg)
        return self.forward(synapses, x, bittensor_pb2.Modality.TENSOR)

    def forward(self, synapses: List[bittensor_pb2.Synapse],
                x: List[torch.Tensor],
                mode: bittensor_pb2.Modality) -> List[torch.Tensor]:
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
            forward_inputs = x[idx]

            # Get or create remote_synapse.
            remote_synapse = None
            if synapse.synapse_key in self._remotes:
                remote_synapse = self._remotes[synapse.synapse_key]
            else:
                # Create remote connection.
                remote_synapse = RemoteSynapse(synapse, self._config)
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

    def __init__(self, synapse: bittensor_pb2.Synapse,
                 config: bittensor.Config):
        super().__init__()
        self.synapse = synapse
        self.local_neuron_key = config.neuron_key
        # Loop back if the synapse is local.
        if synapse.address == config.remote_ip:
            ip = "localhost:"
            if config.remote_ip == "host.docker.internal":
                ip = "host.docker.internal:"
            self.endpoint = ip + str(synapse.port)
        else:
            self.endpoint = synapse.address + ':' + synapse.port
        # TODO(const): should accept defaults. config = bittensor.config_or_defaults(config)

        self.channel = grpc.insecure_channel(
            self.endpoint,
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1)])
        self.stub = bittensor_grpc.BittensorStub(self.channel)
        # TODO(const): setter and getters for signature and nounce.
        self.signature = None
        self.nounce = None

    def __del__(self):
        if self.channel is not None:
            self.channel.close()

    def forward(self, inputs: torch.Tensor,
                mode: bittensor_pb2.Modality) -> torch.Tensor:
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
    # TODO (const) should take multiple input tensors and kwargs.
    @staticmethod
    def forward(ctx, caller: RemoteSynapse, dummy: torch.Tensor,
                inputs: torch.Tensor,
                mode: bittensor_pb2.Modality) -> torch.Tensor:
        # Save for backward call.
        ctx.caller = caller
        ctx.mode = mode
        try:

            # Serialize inputs to bytest buffer.
            try:
                serialized_inputs = PyTorchSerializer.serialize(inputs, mode)
            except:
                raise SerializationException

            ctx.serialized_inputs = serialized_inputs

            # Build request for forward.
            request = bittensor_pb2.TensorMessage(
                version=bittensor.__version__,
                neuron_key=ctx.caller.local_neuron_key,
                synapse_key=ctx.caller.synapse.synapse_key,
                nounce=ctx.caller.nounce,
                signature=ctx.caller.signature,
                tensors=[serialized_inputs])

            # Forward tensor.
            response = ctx.caller.stub.Forward(request)

            # Deserialize outputs and return.
            if len(response.tensors) > 0:
                outputs = PyTorchSerializer.deserialize_tensor(
                    response.tensors[0])
            else:
                raise EmptyTensorException(
                    'Forward request returned no tensors.')

            # Check batch_size.
            if     outputs.size(0) != inputs.size(0) \
                or outputs.size(1) != inputs.size(1) \
                or outputs.size(2) != bittensor.__network_dim__:
                raise ResponseShapeException(
                    'Forward request returned tensor with incorrect shape {}'.
                    format(list(outputs.shape)))

            # Safe catch NaNs and replace with 0.0.
            outputs = torch.where(torch.isnan(outputs),
                                  torch.zeros_like(outputs), outputs)

        # Catch Errors and return zeros.
        except (grpc._channel._InactiveRpcError, EmptyTensorException,
                SerializationException, ResponseShapeException) as e:
            outputs = torch.zeros(
                (inputs.size(0), inputs.size(1), bittensor.__network_dim__))

        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, grads: torch.Tensor) -> Optional[torch.Tensor]:

        # Serialize inputs to bytes.
        serialized_grads = PyTorchSerializer.serialize_tensor(grads)
        serialized_inputs = ctx.serialized_inputs

        # Build request for forward.
        request = bittensor_pb2.TensorMessage(
            version=bittensor.__version__,
            neuron_key=ctx.caller.local_neuron_key,
            synapse_key=ctx.caller.synapse.synapse_key,
            nounce=ctx.caller.nounce,
            signature=ctx.caller.signature,
            tensors=[serialized_inputs, serialized_grads])

        deserialized_grad_inputs = torch.zeros(1, 1)

        try:
            # Attain backward response
            response = ctx.caller.stub.Backward(request)
            deserialized_grad_inputs = PyTorchSerializer.deserialize(
                response.tensors[0])
            return (None, None, deserialized_grad_inputs, None)
        except grpc._channel._InactiveRpcError as _:
            return (None, None, deserialized_grad_inputs, None)
