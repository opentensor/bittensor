
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

import grpc
import sys
import time
import torch
import torch.nn as nn
import traceback

from munch import Munch
from types import SimpleNamespace
from torch.autograd.function import once_differentiable
from typing import Tuple, List, Optional

import bittensor
import bittensor.utils.stats as stat_utils
from bittensor.exceptions.handlers import rollbar

from loguru import logger
logger = logger.opt(colors=True)

# dummy tensor that triggers autograd in a RemoteExpert
DUMMY = torch.empty(0, requires_grad=True)

# Helper function for filling nill (zero) responses on failures.
def nill_response_for(inputs):
    if torch.numel(inputs) == 0:
        return torch.tensor([])
    return torch.zeros( (inputs.size(0), inputs.size(1), bittensor.__network_dim__), dtype = torch.float32)

class Receptor(nn.Module):
    """ Encapsulates a grpc connection to an axon endpoint as a standard auto-grad torch.nn.Module.
    """

    def __init__(
            self, 
            config: Munch, 
            wallet: 'bittensor.wallet',
            endpoint: 'bittensor.Endpoint', 
            channel: 'grpc._Channel',
            stub: 'bittensor.grpc.BittensorStub'
            
        ):
        r""" Initializes a receptor grpc connection.
            Args:
                config (:obj:`Munch`, `required`): 
                    receptor.Receptor.default_config()
                wallet (:obj:`bittensor.Wallet`, `required`):
                    bittensor wallet with hotkey and coldkeypub.
                endpoint (:obj:`bittensor.Endpoint`, `required`):
                    neuron endpoint descriptor proto.
                channel (:obj:`grpc._Channel`, `required`):
                    grpc TCP channel.
                endpoint (:obj:`bittensor.grpc.BittensorStub`, `required`):
                    bittensor protocol stub created from channel.
        """
        super().__init__()
        self.config = config
        self.wallet = wallet # Keypair information
        self.endpoint = endpoint # Endpoint information.
        self.channel = channel
        self.stub = stub
        self.signature = None # Call signature.
        self.nounce = None # Call nounce.
        self.backoff = 0 # Number o queries to backoff.
        self.next_backoff = 1 # Next backoff level.
        self.stats = SimpleNamespace(
            forward_qps = stat_utils.timed_rolling_avg(0.0, 0.01),
            backward_qps = stat_utils.timed_rolling_avg(0.0, 0.01),
            forward_elapsed_time = stat_utils.timed_rolling_avg(0.0, 0.01),
            forward_bytes_out = stat_utils.timed_rolling_avg(0.0, 0.01),
            forward_bytes_in = stat_utils.timed_rolling_avg(0.0, 0.01),
            backward_bytes_out = stat_utils.timed_rolling_avg(0.0, 0.01),
            backward_bytes_in = stat_utils.timed_rolling_avg(0.0, 0.01),

            codes = {
                bittensor.proto.ReturnCode.Success: 0,
                bittensor.proto.ReturnCode.Timeout: 0,
                bittensor.proto.ReturnCode.Backoff: 0,
                bittensor.proto.ReturnCode.Unavailable: 0,
                bittensor.proto.ReturnCode.NotImplemented: 0,
                bittensor.proto.ReturnCode.EmptyRequest: 0,
                bittensor.proto.ReturnCode.EmptyResponse: 0,
                bittensor.proto.ReturnCode.InvalidResponse: 0,
                bittensor.proto.ReturnCode.InvalidRequest: 0,
                bittensor.proto.ReturnCode.RequestShapeException: 0,
                bittensor.proto.ReturnCode.ResponseShapeException: 0,
                bittensor.proto.ReturnCode.RequestSerializationException: 0,
                bittensor.proto.ReturnCode.ResponseSerializationException: 0,
                bittensor.proto.ReturnCode.RequestDeserializationException: 0,
                bittensor.proto.ReturnCode.ResponseDeserializationException: 0,
                bittensor.proto.ReturnCode.NotServingSynapse: 0,
                bittensor.proto.ReturnCode.NucleusTimeout: 0,
                bittensor.proto.ReturnCode.NucleusFull: 0,
                bittensor.proto.ReturnCode.RequestIncompatibleVersion: 0,
                bittensor.proto.ReturnCode.ResponseIncompatibleVersion: 0,
                bittensor.proto.ReturnCode.SenderUnknown: 0,
                bittensor.proto.ReturnCode.UnknownException: 0,
            }
        )

    def __del__(self):
        if self.channel is not None:
            self.channel.close()

    def forward(self, inputs: torch.Tensor, mode: bittensor.proto.Modality) -> Tuple[torch.Tensor, int]:
        r""" Torch.nn.Module forward call: Triggers the grpc call to the remote endpoint.
            Call returns the output tensor and a bittensor.proto.ReturnCode.

            Args:
                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    Single torch tensor to be sent to the remote endpoint.

                mode (:obj:`bittensor.proto.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality type. Enum in [TEXT, IMAGE, TENSOR]

            Returns:
                output (:obj:`Tuple[torch.FloatTensor, torch.LongTensor]`, `required`):
                    Result tuple from the forward call.

        """
        # ---- On Backoff: We dont make an RPC and return zeros instead ----  
        if self.config.receptor.do_backoff and self.backoff >= 1:
            outputs = nill_response_for(inputs)
            code = torch.tensor(bittensor.proto.ReturnCode.Backoff)

        # ---- On Not-backoff: We make the Forward RPC ---- 
        else:
            try:
                # Make and time the query.
                outputs, code = _ReceptorCall.apply(self, DUMMY, inputs, mode)

            # ---- On unknown failure: we return zeros and unknown code ---- 
            except Exception as e:
                logger.error('Uncaught error in forward call with error {}, {}'.format( e, traceback.format_exc()))
                outputs = nill_response_for(inputs)
                code = torch.tensor(bittensor.proto.ReturnCode.UnknownException)

        # ---- On Success: set zero backoff and halve the next backoff ---- 
        try:
            self.stats.codes[code.item()] += 1
        except Exception: 
            pass
        if code.item() == bittensor.proto.ReturnCode.Success:
            self.backoff = 0
            self.next_backoff = max(1, self.next_backoff / 2)

        elif code.item() == bittensor.proto.ReturnCode.EmptyRequest:
            # This was a NO-OP
            pass
            
        # ---- On Backoff: Lower backoff value by 1 ---- 
        elif code.item() == bittensor.proto.ReturnCode.Backoff:
            # We slowly lower the backoff count until 0.
            self.backoff -= 1

        # ---- On failure: Increase backoff and double next_backoff towards max value ---- 
        # Catch all non-success / non-backoff codes and trigger backoff increase. This catches
        # serialization errors, timeouts, unavailable endpoints etc. Note, it can 
        # be triggered by invalid requests on this side of the query.
        else:
            # ---- Do backoff: incease backoff until max_backoff is reached ---- 
            self.backoff = self.next_backoff
            self.next_backoff = min(self.config.receptor.max_backoff, self.next_backoff * 2)

        # ---- Finally return ---- 
        return outputs, code

class _ReceptorCall(torch.autograd.Function):

    @staticmethod
    def forward(ctx, caller: Receptor, dummy: torch.Tensor, inputs: torch.Tensor, mode: bittensor.proto.Modality) -> Tuple[torch.Tensor, int]:  
        r""" Internal autograd-friendly Forward RPC call to a remote endpoint (calls the Forward method on an Axon terminal.)

            Args:
                ctx: (:obj:`torch.autograd.ctx`, `required`):
                    Autograd context, saves state information between forward and backward calls. i.e. inputs for gradient computation.

                caller: (:obj:`Receptor`, `required`):
                    Caller receptor object containing the endpoint information, RPC channel etc.

                dummy: (:obj:`torch.Tensor`, `required`):
                    Dummy torch tensor used to ensure that torch.backward computation is called on this function 
                    regardless of the input types.
  
                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    Torch tensor to be sent to the caller associated endpoint endpoint..

                mode (:obj:`bittensor.proto.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality type. Enum in [TEXT, IMAGE, TENSOR]

            Returns:
                output (:obj:`Tuple[torch.FloatTensor`, torch.LongTensor]`, `optional`):
                    Result from forward call. May be None in the case of failure.

                code (:obj:`bittensor.proto.ReturnCode`, `required`):
                    Return code associated with forward call.
        """
        # ---- Save for backward call ---
        ctx.caller = caller
        ctx.mode = mode
        ctx.inputs = inputs

        zeros = nill_response_for(inputs)
        try:
            # ---- Check inputs size ----
            if torch.numel(inputs) == 0:
                return zeros, torch.tensor(bittensor.proto.ReturnCode.EmptyRequest)

            # ---- Inputs Serialization ----
            try:
                serializer = bittensor.serializer( bittensor.proto.Serializer.MSGPACK )
                serialized_inputs = serializer.serialize(inputs, modality = mode, from_type = bittensor.proto.TensorType.TORCH)
            except Exception as e:
                logger.warning('Serialization error with error {}', e)
                return zeros, torch.tensor(bittensor.proto.ReturnCode.RequestSerializationException)
            ctx.serialized_inputs =  serialized_inputs

            # ---- Build request ----
            request = bittensor.proto.TensorMessage(
                version = bittensor.__version__,
                public_key = ctx.caller.wallet.hotkey.public_key,
                nounce = ctx.caller.nounce,
                signature = ctx.caller.signature,
                tensors = [serialized_inputs])
        
            # ---- Make RPC call ----
            try:
                start_time = time.time()
                ctx.caller.stats.forward_qps.update(1)
                ctx.caller.stats.forward_bytes_out.update(sys.getsizeof(request))
                logger.debug('<white>Dendrite</white> <green>Forward Request</green> ---> <white>to</white>:{}, <white>inputs</white>:{}, <white>mode</white>:{}', caller.endpoint, inputs.shape, mode)
                response = ctx.caller.stub.Forward(request, timeout=caller.config.receptor.timeout)
                ctx.caller.stats.forward_bytes_in.update(sys.getsizeof(response))
                ctx.caller.stats.forward_elapsed_time.update((time.time() - start_time))

                # Get message
                try:
                    response_message = response.message 
                except:
                    response_message = ''

                # ---- Catch non-code ----
                try:
                    bittensor_code = response.return_code
                except:
                    logger.debug('<white>Dendrite</white> <green>Forward Response</> <--- <white>code</white>:<yellow>UnknownException</yellow>, <white>from</white>:{}, message:<red>{}</red>', caller.endpoint, inputs.shape, mode)
                    return zeros, torch.tensor(bittensor.proto.ReturnCode.UnknownException)

                # ---- Catch bittensor errors ----
                if bittensor_code == bittensor.proto.ReturnCode.UnknownException:
                    logger.debug('<white>Dendrite</white> <green>Forward Response</green> <--- <white>code</white>:<yellow>UnknownException</yellow>, <white>from</white>:{}, message:<red>{}</red>', caller.endpoint, response_message)
                    return zeros, torch.tensor(bittensor_code)

                elif bittensor_code != bittensor.proto.ReturnCode.Success:
                    logger.debug('<white>Dendrite</white> <green>Forward Response</green> <--- <white>code</white>:<yellow>{}</yellow>, <white>from</white>:{}, message:<red>{}</red>', bittensor_code, caller.endpoint,response_message)
                    return zeros, torch.tensor(bittensor_code)

            # ---- Catch GRPC Errors ----
            except grpc.RpcError as rpc_error_call:
                grpc_code = rpc_error_call.code()

                if grpc_code == grpc.StatusCode.DEADLINE_EXCEEDED:
                    logger.debug('<white>Dendrite</white> <green>Forward Response</green> <--- <white>code</white>:<yellow>Timeout</yellow>, <white>from</white>:{}', caller.endpoint )
                    return zeros, torch.tensor(bittensor.proto.ReturnCode.Timeout)

                elif grpc_code == grpc.StatusCode.UNAVAILABLE:
                    logger.debug('<white>Dendrite</white> <green>Forward Response</green> <--- <white>code</white>:<yellow>Unavailable</yellow>, <white>from</white>:{}', caller.endpoint )
                    return zeros, torch.tensor(bittensor.proto.ReturnCode.Unavailable)

                else:
                    logger.debug('<white>Dendrite</white> <green>Forward Response</green> <--- <white>code</white>:<red>UnknownException</red>, <white>from</white>:{} ', caller.endpoint )
                    return zeros, torch.tensor(bittensor.proto.ReturnCode.UnknownException)

            # ---- Catch Unknown Errors ----
            except Exception as e:
                logger.debug('<white>Dendrite</white> <green>Forward Response</green> <--- <white>code</white>:<red>UnknownException</red>, <white>from</white>:{}, <white>message</white>:<red>{}</red>', caller.endpoint, e)
                return zeros, torch.tensor(bittensor.proto.ReturnCode.UnknownException)

            # ---- Check tensor response length ----
            if len(response.tensors) == 0:
                logger.debug('<white>Dendrite</white> <green>Forward Response</green> <--- <white>code</white>:<yellow>EmptyResponse</yellow>, <white>from</white>:{}', caller.endpoint )
                return zeros, torch.tensor(bittensor.proto.ReturnCode.EmptyResponse)

            # ---- Deserialize response ----
            try:
                outputs = response.tensors[0]
                deserializer = bittensor.serializer(  outputs.serializer )
                outputs = deserializer.deserialize( outputs, to_type = bittensor.proto.TensorType.TORCH )

            except Exception as e:
                logger.debug('<white>Dendrite</white> <green>Forward Response</green> <--- <white>code</white>:<red>ResponseDeserializationException</red>, <white>from</white>:{}, message:<red>{}</red> ]', caller.endpoint, e)
                return zeros, torch.tensor(bittensor.proto.ReturnCode.ResponseDeserializationException)
        
            # ---- Check response shape ----
            if  outputs.size(0) != inputs.size(0) \
                or outputs.size(1) != inputs.size(1) \
                or outputs.size(2) != bittensor.__network_dim__:
                    logger.debug('<white>Dendrite</white> <green>Forward Response</green> <--- <white>code</white>:<red>ResponseShapeException</red>, <white>from</white>:{}, <white>shape</white>:{}, <white>expected</white>:{}', caller.endpoint, list(outputs.shape), [inputs.size(0), inputs.size(1), bittensor.__network_dim__])
                    return zeros, torch.tensor(bittensor.proto.ReturnCode.ResponseShapeException)

            # ---- Safe catch NaNs and replace with 0.0 ----
            outputs = torch.where(torch.isnan(outputs), torch.zeros_like(outputs), outputs)
            
        # ---- Catch all ----
        except Exception as e:
            logger.error('<white>Dendrite</white> <green>Forward Response</green> <--- <white>code</white>:<red>UnknownException</red>, <white>from</white>:{}, <white>message</white>:<red>{}</red>', caller.endpoint, e)
            return zeros, torch.tensor(bittensor.proto.ReturnCode.UnknownException)

        # ---- Return ----
        logger.debug('<white>Dendrite</white> <green>Forward Response</green> <--- <white>code</white>:<green>Success</green>, <white>from</white>:{}, <white>outputs</white>:{}', caller.endpoint, outputs.shape)
        return outputs, torch.tensor(response.return_code)

    @staticmethod
    @once_differentiable
    def backward(ctx, grads: torch.FloatTensor, code: torch.FloatTensor) -> Optional[torch.Tensor]:
        """ Internal autograd-friendly Backward RPC call to a remote endpoint (calls the Backward method on an remote Axon terminal.)

            Args:
                ctx: (:obj:`torch.autograd.ctx`, `required`):
                    Autograd context, saves state information between forward and backward calls. i.e. inputs for gradient computation.
  
                grads (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    Gradients of this function's outputs computed during the loss.backward() call.

                code (:obj:`bittensor.proto.Modality` of shape :obj:`(1)`, `required`):
                    Code output from the forward call.

            Returns:
                output (:obj:`Tuple[torch.FloatTensor`, torch.LongTensor]`, `optional`):
                    Gradients of the inputs with respect to the inputs and grads of the outputs.
        """
        # ---- Zeros response in the case of failure ----
        zeros = nill_response_for(ctx.inputs)

        # ---- Check if are passing gradients ----
        if not ctx.caller.config.receptor.pass_gradients:
            return (None, None, zeros, None)

        # ---- Check that forward query was a success ----
        if code.item() != bittensor.proto.ReturnCode.Success:
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
                    serializer = bittensor.serializer( bittensor.proto.Serializer.MSGPACK )

                    # ---- Serialize grads to bitensor_pb2.Tensors ----
                    serialized_grads = serializer.serialize (grads, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)

                except Exception as e:
                    logger.trace('backward failed during serialization of gradients.')
                    return (None, None, zeros, None)

    
                # ---- Build request for backward ----
                request = bittensor.proto.TensorMessage(
                    version = bittensor.__version__,
                    public_key = ctx.caller.wallet.hotkey.public_key,
                    nounce = ctx.caller.nounce,
                    signature = ctx.caller.signature,
                    tensors = [serialized_inputs, serialized_grads])

                # --- Send non blocking grad request ----
                # NOTE(const): we dont care about the response.
                try:
                    ctx.caller.stats.backward_qps.update(1)
                    ctx.caller.stats.backwar_bytes_out.update(sys.getsizeof(request))
                    logger.debug('<white>Dendrite</white> <green>Backward Request</green> ---> <white>from</white>:{}, <white>grads</white>:{}', ctx.caller.endpoint, grads.shape)
                    ctx.caller.stub.Backward.future(request, timeout=ctx.caller.config.receptor.timeout)
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
