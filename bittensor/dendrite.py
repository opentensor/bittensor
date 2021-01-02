import argparse
import asyncio
import grpc
import math
import sys
import time
import torch
import torch.nn as nn

from termcolor import colored
from torch.autograd.function import once_differentiable
from types import SimpleNamespace
from typing import Tuple, List, Optional
from loguru import logger
from munch import Munch

import bittensor
import bittensor.utils.stats as stat_utils
import bittensor.serialization as serialization
from bittensor import bittensor_pb2_grpc as bittensor_grpc
from bittensor import bittensor_pb2
from bittensor.exceptions.handlers import rollbar

# dummy tensor that triggers autograd in a RemoteExpert
DUMMY = torch.empty(0, requires_grad=True)

# Helper function for filling nill (zero) responses on failures.
def nill_response_for(inputs):
    if torch.numel(inputs) == 0:
        return torch.tensor([])
    return torch.zeros( (inputs.size(0), inputs.size(1), bittensor.__network_dim__), dtype = torch.float32)

class Dendrite(nn.Module):
    r"""
    Bittensor object used to make calls to the network. It can be called like a normal torch nn.Module and is differentiable. 
    Messages passed through this module will be sent to neuron objects, either remote
    or local, and return responses as torch tensors. Gradients passing through this module on a .backward() call will trigger
    the Backward rpc calls, passing gradients to the remote neuron instances called during corresponding Forward operation.

    Args:
        config (:obj:`bittensor.Config`, `required`):
            Bittensor config object.
    """

    def __init__(self, config):
        super().__init__()
        self._config = config
        self.__keypair = config.neuron.keypair
        self._remotes = {}

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument('--dendrite.pass_gradients', default=True, type=bool, 
                            help='Switch to true is the neuron passes gradients to downstream peers.')
        parser.add_argument('--dendrite.timeout', default=0.5, type=float, 
                            help='Per request RPC timeout.')
        parser.add_argument('--dendrite.do_backoff', default=True, type=bool, 
                            help='Neurons who return non successful return codes are periodically not called with a multiplicative backoff.')
        parser.add_argument('--dendrite.max_backoff', default=100, type=int, 
                            help='Backoff saturates at this value.')
        return parser

    def __str__(self):
        total_bytes_out = 0
        total_bytes_in = 0
        for remote in self._remotes.values():
            total_bytes_out += remote.stats.forward_bytes_out.value
            total_bytes_in += remote.stats.forward_bytes_in.value
        total_in_bytes_str = colored('\u290A {:.1f}'.format((total_bytes_out*8)/1000), 'green')
        total_out_bytes_str = colored('\u290B {:.1f}'.format((total_bytes_in*8)/1000), 'red')
        return total_in_bytes_str + "/" + total_out_bytes_str + "kB/s"
    
    def __full_str__(self):
        response = ""
        for remote in self._remotes.values():
            response += str(remote) + "\n"
        
        return response

    @staticmethod   
    def check_config(config: Munch):
        assert config.dendrite.timeout >= 0, 'timeout must be positive value, got {}'.format(config.dendrite.timeout)

    @property
    def remotes(self):
        return self._remotes.values()

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

        # ---- Run async calls ----
        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(self._gather(loop, x, neurons, mode))
        loop.stop()

        # ---- Process results and return ----
        tensor_results = [res[0] for res in results]
        return_codes = torch.tensor([res[1] for res in results])
        return tensor_results, return_codes

    async def _gather(self, loop: asyncio.base_events.BaseEventLoop, inputs, neurons, mode) -> List[Tuple[torch.FloatTensor, torch.LongTensor]]:
        r""" Creates and returns the results from len(neurons) torch forward requests. Uses asyncio for concurrency.

            Args:
                loop (:obj:`asyncio.base_events.BaseEventLoop`, `required`):
                    The asyncio concurrency loop to use while making the n calls.

                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(num_neurons * [shape])`, `required`):
                    List of tensors to send to corresponsing neurons. Tensors are of arbitrary type and shape depending on the
                    modality.

                neurons (:obj:`List[bittensor_pb2.Neuron]` of shape :obj:`(num_neurons)`, `required`):
                    List of remote neurons which match length of x. Tensors from x are sent forward to these neurons.

                mode (:obj:`bittensor_pb2.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality type. Enum in [TEXT, IMAGE, TENSOR]

            Returns:
                results (:obj:`List[Tuple[torch.FloatTensor, torch.LongTensor]]`, `required`):
                    result tuples from the forward call on a RemoteNeuron class.
        """
            
        # ---- Calls to fill ---- 
        calls = []
        for (inputs_i, neuron_i) in list(zip(inputs, neurons)):

            # ---- Find remote or create one ---- 
            if neuron_i.public_key not in self._remotes:
                self._remotes[neuron_i.public_key] = RemoteNeuron(neuron_i, self._config, self.__keypair)
            remote = self._remotes[neuron_i.public_key]

            # ---- Append async calls ---- 
            calls.append( loop.run_in_executor(None, remote.forward, inputs_i, mode) )

        # ---- Gather results and return ---- 
        results = await asyncio.gather(*calls)
        return results

class RemoteNeuron(nn.Module):
    """ Class which bundles a grpc connection to a remote host as a standard auto-grad torch.nn.Module.
    """

    def __init__(self, neuron: bittensor_pb2.Neuron, config, keypair):
        super().__init__()
        self.neuron = neuron # Endpoint information.
        self.config = config # Configuration i.e. rpc timeout.
        self.keypair = keypair # Cryptographic keypair.
        self.signature = None # Call signature.
        self.nounce = None # Call nounce.
        self.backoff = 0 # Number o queries to backoff.
        self.next_backoff = 1 # Next backoff level.
        self.stats = SimpleNamespace(
            n_forward_calls = 0,
            n_backward_calls = 0,
            forward_elapsed_time = stat_utils.timed_rolling_avg(0.0, 0.01),
            forward_bytes_out = stat_utils.timed_rolling_avg(0.0, 0.01),
            forward_bytes_in = stat_utils.timed_rolling_avg(0.0, 0.01),
            backward_bytes_out = stat_utils.timed_rolling_avg(0.0, 0.01),
            backward_bytes_in = stat_utils.timed_rolling_avg(0.0, 0.01),
            codes = {
                bittensor_pb2.ReturnCode.Success: 0,
                bittensor_pb2.ReturnCode.Timeout: 0,
                bittensor_pb2.ReturnCode.Backoff: 0,
                bittensor_pb2.ReturnCode.Unavailable: 0,
                bittensor_pb2.ReturnCode.NotImplemented: 0,
                bittensor_pb2.ReturnCode.EmptyRequest: 0,
                bittensor_pb2.ReturnCode.EmptyResponse: 0,
                bittensor_pb2.ReturnCode.InvalidResponse: 0,
                bittensor_pb2.ReturnCode.InvalidRequest: 0,
                bittensor_pb2.ReturnCode.RequestShapeException: 0,
                bittensor_pb2.ReturnCode.ResponseShapeException: 0,
                bittensor_pb2.ReturnCode.RequestSerializationException: 0,
                bittensor_pb2.ReturnCode.ResponseSerializationException: 0,
                bittensor_pb2.ReturnCode.RequestDeserializationException: 0,
                bittensor_pb2.ReturnCode.ResponseDeserializationException: 0,
                bittensor_pb2.ReturnCode.NotServingSynapse: 0,
                bittensor_pb2.ReturnCode.NucleusTimeout: 0,
                bittensor_pb2.ReturnCode.NucleusFull: 0,
                bittensor_pb2.ReturnCode.UnknownException: 0,
            }
        ) 
        # Loop back if the neuron is local.
        if neuron.address == config.axon.external_ip:
            ip = "localhost:"
            if config.axon.external_ip == "host.docker.internal":
                ip = "host.docker.internal:"
            self.endpoint = ip + str(neuron.port)
        else:
            self.endpoint = neuron.address + ':' + str(neuron.port)
        self.channel = grpc.insecure_channel(
            self.endpoint,
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1)])
        self.stub = bittensor_grpc.BittensorStub(self.channel)

    def __str__(self):
        total_out_bytes = self.stats.forward_bytes_out.value + self.stats.backward_bytes_out.value
        total_in_bytes = self.stats.forward_bytes_in.value + self.stats.backward_bytes_in.value
        total_in_bytes_str = colored('\u290A {:.1f}'.format((total_out_bytes*8)/1000), 'green')
        total_out_bytes_str = colored('\u290B {:.1f}'.format((total_in_bytes*8)/1000), 'red')
        return str(self.neuron.uid) + ":(" + total_in_bytes_str + "/" + total_out_bytes_str + "kB/s)"

    def __del__(self):
        if self.channel is not None:
            self.channel.close()

    def forward(self, inputs: torch.Tensor, mode: bittensor_pb2.Modality) -> Tuple[torch.Tensor, int]:
        r""" Torch.nn.Module forward call: Triggers the grpc call to the remote neuron on the associated endpoint.
            Call returns the output tensor and a bittensor_pb2.ReturnCode.

            Args:
                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    Single torch tensor to be sent to the remote neuron endpoint.

                mode (:obj:`bittensor_pb2.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality type. Enum in [TEXT, IMAGE, TENSOR]

            Returns:
                output (:obj:`Tuple[torch.FloatTensor, torch.LongTensor]`, `required`):
                    Result tuple from the forward call.

        """
        # ---- On Backoff: We dont make an RPC and return zeros instead ----  
        if self.config.dendrite.do_backoff and self.backoff >= 1:
            outputs = nill_response_for(inputs)
            code = torch.tensor(bittensor_pb2.ReturnCode.Backoff)

        # ---- On Not-backoff: We make the Forward RPC ---- 
        else:
            try:
                # Make and time the query.
                start_time = time.time()
                outputs, code = _RemoteModuleCall.apply(self, DUMMY, inputs, mode)
                elapsed_time = time.time() - start_time

            # ---- On unknown failure: we return zeros and unknown code ---- 
            except Exception as e:
                logger.error('Uncaught error in forward call with error {}'.format( e ))
                outputs = nill_response_for(inputs)
                code = torch.tensor(bittensor_pb2.ReturnCode.UnknownException)

        # ---- On Success: set zero backoff and halve the next backoff ---- 
        self.stats.codes[code.item()] += 1
        if code.item() == bittensor_pb2.ReturnCode.Success:
            self.backoff = 0
            self.next_backoff = max(1, self.next_backoff / 2)
            
        # ---- On Backoff: Lower backoff value by 1 ---- 
        elif code.item() == bittensor_pb2.ReturnCode.Backoff:
            # We slowly lower the backoff count until 0.
            self.backoff -= 1

        # ---- On failure: Increase backoff and double next_backoff towards max value ---- 
        # Catch all non-success / non-backoff codes and trigger backoff increase. This catches
        # serialization errors, timeouts, unavailable endpoints etc. Note, it can 
        # be triggered by invalid requests on this side of the query.
        else:
            # ---- Do backoff: incease backoff until max_backoff is reached ---- 
            self.backoff = self.next_backoff
            self.next_backoff = min(self.config.dendrite.max_backoff, self.next_backoff * 2)

        # ---- Finally return ---- 
        return outputs, code

class _RemoteModuleCall(torch.autograd.Function):
    @staticmethod
    def forward(ctx, caller: RemoteNeuron, dummy: torch.Tensor, inputs: torch.Tensor, mode: bittensor_pb2.Modality) -> Tuple[torch.Tensor, int]:

        """ Internal autograd-friendly Forward RPC call to a remote neuron (calls the Forward method on an Axon terminal.)

            Args:
                ctx: (:obj:`torch.autograd.ctx`, `required`):
                    Autograd context, saves state information between forward and backward calls. i.e. inputs for gradient computation.

                caller: (:obj:`RemoteNeuron`, `required`):
                    Caller object the remote neuron containing the endpoint information, RPC channel etc.

                dummy: (:obj:`torch.Tensor`, `required`):
                    Dummy torch tensor used to ensure that torch.backward computation is called on this function 
                    regardless of the input types.
  
                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    Torch tensor to be sent to the caller associated endpoint neurons.

                mode (:obj:`bittensor_pb2.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality type. Enum in [TEXT, IMAGE, TENSOR]

            Returns:
                output (:obj:`Tuple[torch.FloatTensor`, torch.LongTensor]`, `optional`):
                    Result from forward call. May be None in the case of failure.

                code (:obj:`bittensor_pb2.ReturnCode`, `required`):
                    Return code associated with forward call.
        """
        
        # ---- Save for backward call ----
        ctx.caller = caller
        ctx.mode = mode
        ctx.inputs = inputs

        zeros = nill_response_for(inputs)
        try:
            # ---- Check inputs size ----
            if torch.numel(inputs) == 0:
                return zeros, torch.tensor(bittensor_pb2.ReturnCode.EmptyRequest)

            # ---- Inputs Serialization ----
            try:
                serializer = serialization.get_serializer( bittensor_pb2.Serializer.MSGPACK )
                serialized_inputs = serializer.serialize(inputs, modality = mode, from_type = bittensor_pb2.TensorType.TORCH)
            except Exception as e:
                logger.warning('Serialization error with error {}', e)
                return zeros, torch.tensor(bittensor_pb2.ReturnCode.RequestSerializationException)
            ctx.serialized_inputs =  serialized_inputs

            # ---- Build request ----
            request = bittensor_pb2.TensorMessage(
                version = bittensor.__version__,
                public_key = ctx.caller.keypair.public_key,
                nounce = ctx.caller.nounce,
                signature = ctx.caller.signature,
                tensors = [serialized_inputs])
        
            # ---- Make RPC call ----
            try:
                
                start_time = time.time()
                ctx.caller.stats.n_forward_calls += 1
                ctx.caller.stats.forward_bytes_out.update(sys.getsizeof(request))
                response = ctx.caller.stub.Forward(request, timeout=caller.config.dendrite.timeout)
                ctx.caller.stats.forward_bytes_in.update(sys.getsizeof(response))
                ctx.caller.stats.forward_elapsed_time.update((time.time() - start_time))

                # ---- Catch non-code ----
                try:
                    bittensor_code = response.return_code
                except:
                    logger.error('Unknown exception returned from remote host with message {}', response.message)
                    return zeros, torch.tensor(bittensor_code)

                # ---- Catch bittensor errors ----
                if bittensor_code == bittensor_pb2.ReturnCode.UnknownException:
                    logger.error('Unknown exception returned from remote host with message {}', response.message)
                    return zeros, torch.tensor(bittensor_code)

                elif bittensor_code != bittensor_pb2.ReturnCode.Success:
                    return zeros, torch.tensor(bittensor_code)

            # ---- Catch GRPC Errors ----
            except grpc.RpcError as rpc_error_call:
                grpc_code = rpc_error_call.code()

                if grpc_code == grpc.StatusCode.DEADLINE_EXCEEDED:
                    return zeros, torch.tensor(bittensor_pb2.ReturnCode.Timeout)

                elif grpc_code == grpc.StatusCode.UNAVAILABLE:
                    return zeros, torch.tensor(bittensor_pb2.ReturnCode.Unavailable)

                else:
                    logger.error('Uncaught GPRC error exception with code {} from endpoint {}', grpc_code, caller.endpoint)
                    return zeros, torch.tensor(bittensor_pb2.ReturnCode.UnknownException)

            # ---- Catch Unknown Errors ----
            except Exception as e:
                logger.error('Uncaught error in forward call with error {} and endpoint', e, caller.endpoint)
                return zeros, torch.tensor(bittensor_pb2.ReturnCode.Unknown)

            # ---- Check tensor response length ----
            if len(response.tensors) == 0:
                return zeros, torch.tensor(bittensor_pb2.ReturnCode.EmptyResponse)

            # ---- Deserialize response ----
            try:
                outputs = response.tensors[0]
                deserializer = serialization.get_serializer(  outputs.serializer )
                outputs = deserializer.deserialize( outputs, to_type = bittensor_pb2.TensorType.TORCH )

            except Exception as e:
                logger.error('Failed to serialize responses from forward call with error {}', e)
                return zeros, torch.tensor(bittensor_pb2.ReturnCode.ResponseDeserializationException)
        
            # ---- Check response shape ----
            if  outputs.size(0) != inputs.size(0) \
                or outputs.size(1) != inputs.size(1) \
                or outputs.size(2) != bittensor.__network_dim__:
                    logger.error('Forward request returned tensor with incorrect shape {}', list(outputs.shape))
                    return zeros, torch.tensor(bittensor_pb2.ReturnCode.ResponseShapeException)

            # ---- Safe catch NaNs and replace with 0.0 ----
            outputs = torch.where(torch.isnan(outputs), torch.zeros_like(outputs), outputs)
        
        # ---- Catch all ----
        except Exception as e:
            logger.error('Forward request returned unknown error {}', e)
            return zeros, torch.tensor(bittensor_pb2.ReturnCode.UnknownException)

        # ---- Return ----
        return outputs, torch.tensor(response.return_code)

    @staticmethod
    @once_differentiable
    def backward(ctx, grads: torch.FloatTensor, code: torch.FloatTensor) -> Optional[torch.Tensor]:
        """ Internal autograd-friendly Backward RPC call to a remote neuron (calls the Backward method on an remote Axon terminal.)

            Args:
                ctx: (:obj:`torch.autograd.ctx`, `required`):
                    Autograd context, saves state information between forward and backward calls. i.e. inputs for gradient computation.
  
                grads (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    Gradients of this function's outputs computed during the loss.backward() call.

                code (:obj:`bittensor_pb2.Modality` of shape :obj:`(1)`, `required`):
                    Code output from the forward call.

            Returns:
                output (:obj:`Tuple[torch.FloatTensor`, torch.LongTensor]`, `optional`):
                    Gradients of the inputs with respect to the inputs and grads of the outputs.
        """
        # ---- Zeros response in the case of failure ----
        zeros = nill_response_for(ctx.inputs)

        # ---- Check if are passing gradients ----
        if not ctx.caller.config.dendrite.pass_gradients:
            return (None, None, zeros, None)

        # ---- Check that forward query was a success ----
        if code.item() != bittensor_pb2.ReturnCode.Success:
            return (None, None, zeros, None)

        # ---- Try to pass gradients ----
        else:
            try:

                # ---- Get forward call serialzied inputs ----
                try:
                    serialized_inputs = ctx.serialized_inputs
                except:
                    logger.trace('backward failed because forward previously failed.')
                    return (None, None, zeros, None)

                # ---- Serialization ----
                try:
                    # ---- Get serializer ----
                    serializer = serialization.get_serializer( bittensor_pb2.Serializer.MSGPACK )

                    # ---- Serialize grads to bitensor_pb2.Tensors ----
                    serialized_grads = serializer.serialize (grads, modality = bittensor_pb2.Modality.TENSOR, from_type = bittensor_pb2.TensorType.TORCH)

                except Exception as e:
                    logger.trace('backward failed during serialization of gradients.')
                    return (None, None, zeros, None)

    
                # ---- Build request for backward ----
                request = bittensor_pb2.TensorMessage(
                    version = bittensor.__version__,
                    public_key = ctx.caller.keypair.public_key,
                    nounce = ctx.caller.nounce,
                    signature = ctx.caller.signature,
                    tensors = [serialized_inputs, serialized_grads])

                # --- Send non blocking grad request ----
                # NOTE(const): we dont care about the response.
                try:
                    ctx.caller.stats.n_backward_calls += 1
                    ctx.caller.stats.backwar_bytes_out.update(sys.getsizeof(request))
                    ctx.caller.stub.Backward.future(request, timeout=ctx.caller.config.dendrite.timeout)
                    ctx.caller.stats.backwar_bytes_in.update(0.0) # responses are dropped.

                except:
                    logger.trace('backward failed during backward call. Do not care.')
                    return (None, None, zeros, None)

                # ---- Always return zeros ----
                # NOTE(const): We can return non zeros but a remote host could mess with your training
                # without you knowing about it. i.e. by passing you malicious gradients.
                return (None, None, zeros, None)

            except:

                # ---- Catch all exceptions in Backward ----
                rollbar.send_exception()
                return (None, None, zeros, None)
