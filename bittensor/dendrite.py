import argparse
import grpc
import torch
import torch.nn as nn
import bittensor
import time

from torch.autograd.function import once_differentiable
from typing import Tuple, List, Optional
from loguru import logger
from munch import Munch

from bittensor import bittensor_pb2_grpc as bittensor_grpc
from bittensor import bittensor_pb2
from bittensor.serializer import PyTorchSerializer
import time
import asyncio
from bittensor.exceptions.handlers import rollbar

# dummy tensor that triggers autograd in a RemoteExpert
DUMMY = torch.empty(0, requires_grad=True)

def nill_response_for(inputs):
    if torch.numel(inputs) == 0:
        return torch.zeros([], dtype = torch.float32)
    return torch.zeros( (inputs.size(0), inputs.size(1), bittensor.__network_dim__), dtype = torch.float32)

class Dendrite(nn.Module):
    r"""
    Bittensor object used to make calls to the network. It can called like a normal torch nn.Module
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

    def __init__(
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
        parser.add_argument('--dendrite.do_backoff', default=True, type=bool, 
                            help='Neurons who return non successful return codes are periodically not called with a multiplicative backoff.')
        parser.add_argument('--dendrite.max_backoff', default=100, type=int, 
                            help='Backoff saturates at this value.')
        return parser

    @staticmethod   
    def check_config(config: Munch) -> Munch:
        assert config.dendrite.timeout >= 0, 'timeout must be positive value, got {}'.format(config.dendrite.timeout)
        return config

    def forward_text(self, neurons: List[bittensor_pb2.Neuron],
                     x: List[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        r""" Forward text inputs to neurons.

            Args:
                neurons (:obj:`List[bittensor_pb2.Neuron]` of shape :obj:`(num_neurons)`, `required`):
                    List of remote neurons which match length of x. Tensors from x are sent forward to these neurons.

                x (:obj:`List[torch.Tensor]` of shape :obj:`(num_neurons * [batch_size, sequence_len])`, `required`):
                    List of tensors to send to corresponsing neurons. Tensors are text input_ids encoded using the
                    bittensor tokenizer of shape [batch_size, sequence_len].

            Returns:
                forwad_output (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                    Output encodings of inputs produced by remote neurons. Non-responses are zeroes of common shape.

                return_codes (:obj:`List[torch.LongTensor]` of shape :obj:`[num_neurons]`, `required`):
                    dendrite call return ops.
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
                      x: List[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
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

                return_codes (:obj:`List[torch.LongTensor]` of shape :obj:`[num_neurons]`, `required`):
                    dendrite call return ops.
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
                       x: List[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        r""" Forward tensor inputs to neurons.

            Args:
                neurons (:obj:`List[bittensor_pb2.Neuron]` of shape :obj:`(num_neurons)`, `required`):
                    List of remote neurons which match length of x. Tensors from x are sent forward to these neurons.

                x (:obj:`List[torch.Tensor]` of shape :obj:`(num_neurons * [batch_size, sequence_len, bittensor.__network_dim__])`, `required`):
                    List of tensors to send to corresponsing neurons. Tensors are of arbitrary type and
                    with shape [batch_size, sequence_len, bittensor.__network_dim__].

            Returns:
                forwad_output (:obj:`List[torch.FloatTensor]` of shape :obj:`num_neurons * (batch_size, sequence_len, bittensor.__network_dim__)]`, `required`):
                    Output encodings of tensors produced by remote neurons. Non-responses are zeroes of common shape.

                return_codes (:obj:`List[torch.LongTensor]` of shape :obj:`[num_neurons]`, `required`):
                    dendrite call return ops.
        """
        if len(x[0].shape) != 3:
            error_msg = 'Tensor inputs should be rank 3 with semantic shape: [batch_size, sequence_len, feature_len]'
            raise ValueError(error_msg)
        if len(x) != len(neurons):
            error_msg = 'List of tensor inputs x should have the same length as passed destination neurons, got {} and {}'.format(len(x), len(neurons))
            raise ValueError(error_msg)
        if x[0].shape[2] != bittensor.__network_dim__:
            error_msg = 'Passed tensor must have last dimension {} got {}'.format(bittensor.__network_dim__, x[0].shape[2])
            raise ValueError(error_msg)
        if len(x) == 0:
            error_msg = 'Must pass more than 0 input for argument x, got {}'.format(len(x))
            raise ValueError(error_msg)
        return self.forward(neurons, x, bittensor_pb2.Modality.TENSOR)

    def forward(self, neurons: List[bittensor_pb2.Neuron],
                x: List[torch.Tensor],
                mode: bittensor_pb2.Modality) -> Tuple[List[torch.Tensor], torch.LongTensor]:
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
                forward_outputs (:obj:`List[torch.FloatTensor]` of shape :obj:`num_neurons * (batch_size, sequence_len, bittensor.network_size)]`, `required`):
                    Output encodings of tensors produced by remote neurons. Non-responses are zeroes of common shape.

                return_codes (:obj:`List[torch.LongTensor]` of shape :obj:`[num_neurons]`, `required`):
                    dendrite call return ops.
        """
        if len(x) != len(neurons):
            error_msg = 'List of inputs x should have the same length as passed destination neurons, got {} and {}'.format(len(x), len(neurons))
            raise ValueError(error_msg)
        if len(x) < 1:
            error_msg = 'Must pass more than 0 input for argument x, got {}'.format(len(x))
            raise ValueError(error_msg)

        # Run async calls.
        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(self.gather(loop, x, neurons, mode))
        loop.stop()

        tensor_results = [res[0] for res in results]
        return_codes = torch.tensor([res[1] for res in results])
        return tensor_results, return_codes

    async def gather(self, loop: asyncio.base_events.BaseEventLoop, inputs, neurons, mode):
            
        # Fill async calls.
        calls = []
        for i, (inputs_i, neuron_i) in enumerate(list(zip(inputs, neurons))):
            
            # Find remote or create one.
            if neuron_i.public_key not in self._remotes:
                self._remotes[neuron_i.public_key] = RemoteNeuron(neuron_i, self._config, self.__keypair)
            remote = self._remotes[neuron_i.public_key]

            # Append async call.
            calls.append( loop.run_in_executor(None, remote.forward, inputs_i, mode) )
        
        # Gather results and return.
        results = await asyncio.gather(*calls)
        return results

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
        self.channel = grpc.insecure_channel(
            self.endpoint,
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1)])
        self.stub = bittensor_grpc.BittensorStub(self.channel)
        # TODO(const): setter and getters for signature and nounce.
        self.signature = None
        self.nounce = None

        # Current backoff counter.
        self.backoff = 0
        # Next backoff, which doubles until dendrite.max_backoff
        self.next_backoff = 1

    def __del__(self):
        if self.channel is not None:
            self.channel.close()


    def forward(self, inputs: torch.Tensor, mode: bittensor_pb2.Modality) -> Tuple[torch.Tensor, bool, float]:
        
        # Dont make call if we are backing off.
        if self.config.dendrite.do_backoff and self.backoff >= 1:
            self.backoff -= 1
            logger.trace('Still backing off from endpoint {}', self.endpoint)
            return nill_response_for(inputs), torch.tensor(bittensor_pb2.ReturnCode.Backoff)
    
        try:
            outputs, code = _RemoteModuleCall.apply(self, DUMMY, inputs, mode)

            # Update backoff on non-success.
            if code.item() != bittensor_pb2.ReturnCode.Success and self.config.dendrite.do_backoff:
                self.backoff = self.next_backoff
                self.next_backoff = min(self.config.dendrite.max_backoff, self.next_backoff * 2)

            return outputs, code

        except Exception as e:
            error_msg = 'Uncaught error in forward call with error {}'.format( e )
            logger.error(error_msg)
            return nill_response_for(inputs), torch.tensor(bittensor_pb2.ReturnCode.UnknownException)

class _RemoteModuleCall(torch.autograd.Function):
    """ Internal autograd-friendly call of a remote module over grpc"""

    @staticmethod
    def forward(ctx, caller: RemoteNeuron, dummy: torch.Tensor,
                inputs: torch.Tensor,
                mode: bittensor_pb2.Modality) -> Tuple[torch.Tensor, int]:
        # Save for backward call.
        ctx.caller = caller
        ctx.mode = mode
        ctx.inputs = inputs

        zeros = nill_response_for(inputs)
        try:
            # If this is an empty call get nill response.
            if torch.numel(inputs) == 0:
                return zeros, torch.tensor(bittensor_pb2.ReturnCode.EmptyRequest)

            # Serialize inputs.
            try:
                serialized_inputs = PyTorchSerializer.serialize(inputs, mode)
            except:
                logger.warning('Serialization error with inputs {}', inputs)
                return zeros, torch.tensor(bittensor_pb2.ReturnCode.RequestSerializationException)
            ctx.serialized_inputs =  serialized_inputs

            # Build request.
            request = bittensor_pb2.TensorMessage(
                version=bittensor.__version__,
                public_key=ctx.caller.keypair.public_key,
                nounce=ctx.caller.nounce,
                signature=ctx.caller.signature,
                tensors=[serialized_inputs])
        
            try:
                response = ctx.caller.stub.Forward(request, timeout=caller.config.dendrite.timeout)

                bittensor_code = response.return_code
                if bittensor_code == bittensor_pb2.ReturnCode.UnknownException:
                    logger.error('Unknown exception returned from remote host with message {}', response.message)
                    return zeros, torch.tensor(bittensor_code)

                elif bittensor_code != bittensor_pb2.ReturnCode.Success:
                    return zeros, torch.tensor(bittensor_code)

            # Catch GRPC Errors
            except grpc.RpcError as rpc_error_call:
                grpc_code = rpc_error_call.code()

                if grpc_code == grpc.StatusCode.DEADLINE_EXCEEDED:
                    logger.warning('Deadline exceeds on endpoint {}', caller.endpoint)
                    return zeros, torch.tensor(bittensor_pb2.ReturnCode.Timeout)

                elif grpc_code == grpc.StatusCode.UNAVAILABLE:
                    logger.warning('Endpoint unavailable {}', caller.endpoint)
                    return zeros, torch.tensor(bittensor_pb2.ReturnCode.Unavailable)

                else:
                    logger.error('Uncaught GPRC error exception with code {} from endpoint {}', e.code(), caller.endpoint)
                    return zeros, torch.tensor(bittensor_pb2.ReturnCode.Unknown)

            # Catch Unknown
            except Exception as e:
                logger.error('Uncaught error in forward call with error {} and endpoint', e, caller.endpoint)
                return zeros, torch.tensor(bittensor_pb2.ReturnCode.Unknown)

            # Check tensor response.
            if len(response.tensors) == 0:
                logger.error('Empty response from endpoint {}', caller.endpoint)
                return zeros, torch.tensor(bittensor_pb2.ReturnCode.EmptyResponse)

            # Deserialize.
            try:
                outputs = PyTorchSerializer.deserialize_tensor(response.tensors[0])
            except Exception as e:
                logger.error('Failed to serialize responses from forward call with response {} and error {}', response.tensors[0], e)
                return zeros, torch.tensor(bittensor_pb2.ReturnCode.ResponseDeserializationException)
        
            # Check shape
            if  outputs.size(0) != inputs.size(0) \
                or outputs.size(1) != inputs.size(1) \
                or outputs.size(2) != bittensor.__network_dim__:
                    logger.error('Forward request returned tensor with incorrect shape {}', list(outputs.shape))
                    return zeros, torch.tensor(bittensor_pb2.ReturnCode.ResponseShapeException)

            # Safe catch NaNs and replace with 0.0.
            outputs = torch.where(torch.isnan(outputs), torch.zeros_like(outputs), outputs)
        
        except Exception as e:
            logger.error('Forward request returned unknown error {}', e)
            return zeros, torch.tensor(bittensor_pb2.ReturnCode.UnknownException)

        # Return.
        return outputs, torch.tensor(response.return_code)

    @staticmethod
    @once_differentiable
    def backward(ctx, grads: torch.Tensor, code: torch.Tensor) -> Optional[torch.Tensor]:

        # Fill zeros response. Gradient do not pass through peers.
        zeros = nill_response_for(ctx.inputs)

        # Swtich for passing gradients.
        if not ctx.caller.config.dendrite.pass_gradients:
            return (None, None, zeros, None)

        else:
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

                # Non blocking future.
                ctx.caller.stub.Backward.future(request, timeout=ctx.caller.config.dendrite.timeout)
                return (None, None, zeros, None)

            except:
                rollbar.send_exception()
                return (None, None, zeros, None)

