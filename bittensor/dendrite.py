import argparse
import grpc
import torch
import torch.nn as nn
import bittensor
import time

from torch.autograd.function import once_differentiable
from typing import List, Optional
from loguru import logger
from munch import Munch

from bittensor import bittensor_pb2_grpc as bittensor_grpc
from bittensor import bittensor_pb2
from bittensor.tb_logger import TBLogger
from bittensor.serializer import PyTorchSerializer
from bittensor.exceptions.Exceptions import EmptyTensorException, ResponseShapeException, SerializationException
import time

# dummy tensor that triggers autograd in RemoteExpert
DUMMY = torch.empty(0, requires_grad=True)

class Dendrite(nn.Module):
    r"""
    This is the bittensr object used to make calls to the network. It can be used like a normal torch nn.Module
    and is differentiable. Messages passed through this module will be sent to neuron objects, either remote
    or local, and return response torch tensors. Gradients passing through this module on a .backward() call will trigger
    the Backward rpc calls, passing gradients to the remote neuron instances called during the Forward operation.   

    Args:
        config (:obj:`bittensor.Config`, `required`):
            Bittensor config object.
    Examples::

        >>> from bittensor

        >>> # Initialize config defaults.
        >>> config = Config.load()

        >>> # Create Dendrite nn.module
        >>> dendrite = bittensor.Dendrite(config)

        >>> if len(metagraph.neurons()) > 0
        >>>     requests = [torch.rand(10, 100) for _ in metagraph.neurons()] # Random query tensors.
        >>>     responses = dendrite (requests, metagraph.neurons()) # Make network queries.
        >>>     responses[0].backward() # Backprop through dendrite.
    """
    def __init__ (
        self, 
        config,
        keypair,
    ):
        super().__init__()
        self._config = config
        self.__keypair = keypair
        self._remotes = {}

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--dendrite.pass_gradients', default=True, type=bool, 
                            help='Switch to true is the neuron passes gradients to downstream peers.')
        parser.add_argument('--dendrite.timeout', default=0.5, type=float, 
                            help='Per request RPC timeout.')
        return parser

    @staticmethod   
    def check_config(config: Munch) -> Munch:
        assert config.dendrite.timeout >= 0, 'timeout must be positive value, got {}'.format(config.dendrite.timeout)
        return config

    def forward_text(self, neurons: List[bittensor_pb2.Neuron],
                     x: List[torch.Tensor]) -> List[torch.Tensor]:
        r""" Forward text inputs to neurons.

            Args:
                neurons (:obj:`List[bittensor_pb2.Neuron]` of shape :obj:`(num_neurons)`, `required`): 
                    List of remote neurons which match length of x. Tensors from x are sent forward to these neurons.

                x (:obj:`List[torch.Tensor]` of shape :obj:`(num_neurons * [batch_size, sequence_len])`, `required`): 
                    List of tensors to send to corresponsing neurons. Tensors are text input_ids encoded using the
                    bittensor tokenizer of shape [batch_size, sequence_len].
            
            Returns:
                forwad_output (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, sequence_len, bittensor.network_size)`, `required`): 
                    Output encodings of inputs produced by remote neurons. Non-responses are zeroes of common shape.
        """
        if len(x[0].shape) != 2:
            error_msg = 'Text inputs should rank 2 with semantic shape: [batch_size, sequence_len]'
            raise ValueError(error_msg)
        if len(x) != len(neurons):
            error_msg = 'List of text inputs x should have the same length as passed destination neurons, got {} and {}'.format(len(x), len(neurons))
            raise ValueError(error_msg)
        if len(x) < 1:
            error_msg = 'Must pass more than 0 input for argument x, got {}'.format(len(x))
            raise ValueError(error_msg)
        return self.forward(neurons, x, bittensor_pb2.Modality.TEXT)

    def forward_image(self, neurons: List[bittensor_pb2.Neuron],
                      x: List[torch.Tensor]) -> List[torch.Tensor]:
        r""" Forward image inputs to neurons.

            Args:
                neurons (:obj:`List[bittensor_pb2.Neuron]` of shape :obj:`(num_neurons)`, `required`): 
                    List of remote neurons which match length of x. Tensors from x are sent forward to these neurons.
             
                x (:obj:`List[torch.Tensor]` of shape :obj:`(num_neurons * [batch_size, sequence_len, channels, rows, cols])`, `required`): 
                    List of image-tensors to send to corresponsing neurons. Tensors are images encoded using the
                    torch.toTensor() or other encoding which produces the shape [batch_size, channels, rows, cols].
            
            Returns:
                forwad_output (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, sequence_len, bittensor.network_size)`, `required`): 
                    Output encodings of images produced by remote neurons. Non-responses are zeroes of common shape.
        """
        # TODO(const): Checks across all tensors and other shape checks.
        if len(x[0].shape) != 5:
            error_msg = 'Image inputs should be rank 5 with semantic shape: [batch_size, sequence_dim, channels, rows, cols]'
            raise ValueError(error_msg)
        if len(x) != len(neurons):
            error_msg = 'List of image inputs x should have the same length as passed destination neurons, got {} and {}'.format(len(x), len(neurons))
            raise ValueError(error_msg)
        if len(x) < 1:
            error_msg = 'Must pass more than 0 input for argument x, got {}'.format(len(x))
            raise ValueError(error_msg)
        return self.forward(neurons, x, bittensor_pb2.Modality.IMAGE)

    def forward_tensor(self, neurons: List[bittensor_pb2.Neuron],
                       x: List[torch.Tensor]) -> List[torch.Tensor]:
        r""" Forward tensor inputs to neurons.

            Args:
                neurons (:obj:`List[bittensor_pb2.Neuron]` of shape :obj:`(num_neurons)`, `required`): 
                    List of remote neurons which match length of x. Tensors from x are sent forward to these neurons.

                x (:obj:`List[torch.Tensor]` of shape :obj:`(num_neurons * [batch_size, sequence_len, feature_len])`, `required`): 
                    List of tensors to send to corresponsing neurons. Tensors are of arbitrary type and
                    with shape [batch_size, sequence_len, feature_len].
            
            Returns:
                forwad_output (:obj:`List[torch.FloatTensor]` of shape :obj:`num_neurons * (batch_size, sequence_len, bittensor.network_size)]`, `required`): 
                    Output encodings of tensors produced by remote neurons. Non-responses are zeroes of common shape.
        """
        if len(x[0].shape) != 3:
            error_msg = 'Tensor inputs should be rank 3 with semantic shape: [batch_size, sequence_len, feature_len]'
            raise ValueError(error_msg)
        if len(x) != len(neurons):
            error_msg = 'List of tensor inputs x should have the same length as passed destination neurons, got {} and {}'.format(len(x), len(neurons))
            raise ValueError(error_msg)
        if len(x) < 1:
            error_msg = 'Must pass more than 0 input for argument x, got {}'.format(len(x))
            raise ValueError(error_msg)
        return self.forward(neurons, x, bittensor_pb2.Modality.TENSOR)

    def forward(self, neurons: List[bittensor_pb2.Neuron],
                x: List[torch.Tensor],
                mode: bittensor_pb2.Modality) -> List[torch.Tensor]:
        r""" Forward tensor inputs to neurons.

            Args:
                neurons (:obj:`List[bittensor_pb2.Neuron]` of shape :obj:`(num_neurons)`, `required`): 
                    List of remote neurons which match length of x. Tensors from x are sent forward to these neurons.

                x (:obj:`List[torch.Tensor]` of shape :obj:`(num_neurons * [shape])`, `required`): 
                    List of tensors to send to corresponsing neurons. Tensors are of arbitrary type and shape depending on the 
                    modality.

                mode (:obj:`bittensor_pb2.Modality` of shape :obj:`(1)`, `required`): 
                    Bittensor forward modality type. Enum in [TEXT, IMAGE, TENSOR]
            
            Returns:
                forwad_output (:obj:`List[torch.FloatTensor]` of shape :obj:`num_neurons * (batch_size, sequence_len, bittensor.network_size)]`, `required`): 
                    Output encodings of tensors produced by remote neurons. Non-responses are zeroes of common shape.
        """
        if len(x) != len(neurons):
            error_msg = 'List of inputs x should have the same length as passed destination neurons, got {} and {}'.format(len(x), len(neurons))
            raise ValueError(error_msg)
        if len(x) < 1:
            error_msg = 'Must pass more than 0 input for argument x, got {}'.format(len(x))
            raise ValueError(error_msg)
        
        results = []
        for idx, (forward_inputs, neuron) in enumerate(list(zip(x,neurons))):
            # Get or create remote_neuron.
            remote_neuron = None
            if neuron.public_key in self._remotes:
                remote_neuron = self._remotes[neuron.public_key]
            else:
                # Create remote connection.
                remote_neuron = RemoteNeuron(neuron, self._config, self.__keypair)
                self._remotes[neuron.public_key] = remote_neuron

            # Call remote neuron.
            try:
                results.append(remote_neuron(forward_inputs, mode))
            except (SerializationException, EmptyTensorException, ResponseShapeException) as e:
                logger.error("Exception occured: {}".format(e))
        return results


# NOTE: (const) This code has been ported from hivemind thanks to Yozh and Max.
# Credit to them for designing this structure and api around torch. Here being ported to
# bittensor, and eventually should interact seemlessly with hivemind nodes as well.
# TODO (const): needs to check shapes/ input types/ other.
class RemoteNeuron(nn.Module):
    """ Class which bundles a grpc connection to a remote host as a standard auto-grad torch.nn.Module.
    """

    def __init__(self, neuron: bittensor_pb2.Neuron, config, keypair):
        super().__init__()
        self.neuron = neuron
        self.config = config
        self.keypair = keypair
        # Loop back if the neuron is local.
        if neuron.address == config.axon.remote_ip:
            ip = "localhost:"
            if config.axon.remote_ip == "host.docker.internal":
                ip = "host.docker.internal:"
            self.endpoint = ip + str(neuron.port)
        else:
            self.endpoint = neuron.address + ':' + str(neuron.port)
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
        try:
            outputs = _RemoteModuleCall.apply(self, DUMMY, inputs, mode)
        except (SerializationException, EmptyTensorException, ResponseShapeException) as e:
            logger.warning("Exception occured in Remoteneuron forward call: {}".format(e))
            outputs = torch.zeros(
                (inputs.size(0), inputs.size(1), bittensor.__network_dim__))
        return outputs


# Adapted from hivemind. Thanks Yozh.
class _RemoteModuleCall(torch.autograd.Function):
    """ Internal autograd-friendly call of a remote module over grpc"""

    # TODO (const) signatures + nounce.
    # TODO (const) should take multiple input tensors and kwargs.
    @staticmethod
    def forward(ctx, caller: RemoteNeuron, dummy: torch.Tensor,
                inputs: torch.Tensor,
                mode: bittensor_pb2.Modality) -> torch.Tensor:
        # Save for backward call.
        ctx.caller = caller
        ctx.mode = mode
        ctx.inputs = inputs
        try:
            # Serialize inputs to bytest buffer.
            try:
                serialized_inputs = PyTorchSerializer.serialize(inputs, mode)
            except SerializationException:
                raise SerializationException

            ctx.serialized_inputs = serialized_inputs

            # Build request for forward.
            request = bittensor_pb2.TensorMessage(
                version=bittensor.__version__,
                public_key=ctx.caller.keypair.public_key,
                nounce=ctx.caller.nounce,
                signature=ctx.caller.signature,
                tensors=[serialized_inputs])

            # Forward tensor.
            pre_response_time = time.time() # in seconds
            response = ctx.caller.stub.Forward(request, timeout=caller.config.dendrite.timeout)
            # Time (in seconds) response took
            elapsed_time = time.time() - pre_response_time
            bittensor.session.tbwriter.write_dendrite_network_data('Remote Module Forward Call Response Message Size (MB)', response.ByteSize() / 1024)
            bittensor.session.tbwriter.write_dendrite_network_data('Remote Module Forward Call Turnaround latency (seconds)', round(elapsed_time, 2))

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

        deserialized_grad_inputs = torch.zeros_like(ctx.inputs)

        # Swtich for passing gradients.
        if not ctx.caller.config.dendrite.pass_gradients:
            return (None, None, deserialized_grad_inputs, None)

        try:
            # Serialize inputs to bytes.
            serialized_grads = PyTorchSerializer.serialize_tensor(grads)
            serialized_inputs = ctx.serialized_inputs

            # Build request for forward.
            request = bittensor_pb2.TensorMessage(
                version=bittensor.__version__,
                public_key=ctx.caller.keypair.public_key,
                nounce=ctx.caller.nounce,
                signature=ctx.caller.signature,
                tensors=[serialized_inputs, serialized_grads])

            # Attain backward response
            pre_response_time = time.time()
            response = ctx.caller.stub.Backward(request, timeout=ctx.caller.config.dendrite.timeout)
            elapsed_time = time.time() - pre_response_time
            bittensor.session.tbwriter.write_dendrite_network_data('Remote Module Backward Call Response Message Size (MB)', response.ByteSize() / 1024)
            bittensor.session.tbwriter.write_dendrite_network_data('Remote Module Backward Call Turnaround latency (seconds)', round(elapsed_time, 2))
            deserialized_grad_inputs = PyTorchSerializer.deserialize(
                response.tensors[0])
            return (None, None, deserialized_grad_inputs, None)
            
        except grpc._channel._InactiveRpcError as _:
            return (None, None, deserialized_grad_inputs, None)

        except SerializationException as _:
            logger.warning("Serialization of gradients {} failed".format(grads))
            return (None, None, deserialized_grad_inputs, None)

        except Exception as e:
            logger.warning("Uncaught exception {}", e)
            return (None, None, deserialized_grad_inputs, None)
