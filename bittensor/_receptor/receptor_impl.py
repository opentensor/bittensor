
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

from types import SimpleNamespace
from torch.autograd.function import once_differentiable
from typing import Tuple, List, Optional

import bittensor
import bittensor.utils.stats as stat_utils

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
            wallet: 'bittensor.wallet',
            endpoint: 'bittensor.Endpoint', 
            channel: 'grpc._Channel',
            stub: 'bittensor.grpc.BittensorStub',
        ):
        r""" Initializes a receptor grpc connection.
            Args:
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

    def forward (
        self, 
        inputs: torch.Tensor, 
        modality: bittensor.proto.Modality,
        timeout: int,
    ) -> Tuple[torch.Tensor, int]:
        r""" Torch.nn.Module forward call: Triggers the grpc call to the remote endpoint.
            Call returns the output tensor and a bittensor.proto.ReturnCode.

            Args:
                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    Single torch tensor to be sent to the remote endpoint.

                modality (:obj:`bittensor.proto.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality type. Enum in [TEXT, IMAGE, TENSOR]

            Returns:
                output (:obj:`Tuple[torch.FloatTensor, torch.LongTensor]`, `required`):
                    Result tuple from the forward call.

                code (:obj:`bittensor.proto.ReturnCode`, `required`):
                    Return code associated with forward call.
        """
        outputs, code = self._call_forward( 
            inputs = inputs, 
            modality = modality, 
            timeout = timeout 
        )
        try:
            self.stats.codes[code] += 1
        except Exception: 
            pass
        return outputs, code

    def backward(
            self, 
            inputs_x: torch.Tensor, 
            grads_dy: torch.Tensor, 
            modality: bittensor.proto.Modality,
            timeout: int
        ) -> Tuple[ torch.Tensor, int ]:
        r""" Backward call: Triggers the grpc Backward call to the associated endpoint.

            Args:
                inputs_x (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    inputs from previous forward call.
    
                grads_dy (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    gradient outputs.

                modality (:obj:`bittensor.proto.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality type. Enum in [TEXT, IMAGE, TENSOR]

                timeout (int):
                    request timeout.

            Returns:
                output (:obj:`Tuple[torch.FloatTensor, torch.LongTensor]`, `required`):
                    Result tuple from the forward call.

                code (:obj:`bittensor.proto.ReturnCode`, `required`):
                    Return code associated with backward call.
        """
        outputs, code = self._call_backward( 
            inputs_x = inputs_x, 
            grads_dy = grads_dy, 
            modality = modality,
            timeout = timeout
        )
        try:
            self.stats.codes[code] += 1
        except Exception: 
            pass
        return outputs, code

    def _call_forward(
        self, 
        inputs: torch.Tensor, 
        modality: bittensor.proto.Modality,
        timeout: int
    ) -> Tuple[torch.Tensor, int]:  
        r""" Internal autograd-friendly Forward RPC call to a remote endpoint (calls the Forward method on an Axon terminal.)

            Args:  
                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    Torch tensor to be sent to this endpoint.

                modality (:obj:`bittensor.proto.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality of type Enum: [TEXT, IMAGE, TENSOR]

                timeout (int):
                    request timeout.

            Returns:
                output (:obj:`Tuple[torch.FloatTensor`, torch.LongTensor]`, `optional`):
                    Result from forward call. May be None in the case of failure.

                code (:obj:`bittensor.proto.ReturnCode`, `required`):
                    Return code associated with forward call.
        """
        zeros = nill_response_for(inputs)
        try:
            # ---- Check inputs size ----
            if torch.numel(inputs) == 0:
                return zeros, bittensor.proto.ReturnCode.EmptyRequest

            # ---- Inputs Serialization ----
            try:
                serializer = bittensor.serializer( bittensor.proto.Serializer.MSGPACK )
                serialized_inputs = serializer.serialize(inputs, modality = modality, from_type = bittensor.proto.TensorType.TORCH)
            except Exception as e:
                logger.warning('Serialization error with error {}', e)
                return zeros, bittensor.proto.ReturnCode.RequestSerializationException

            # ---- Build request ----
            request = bittensor.proto.TensorMessage (
                version = bittensor.__version__,
                public_key = self.wallet.hotkey.public_key,
                nounce = self.nounce,
                signature = self.signature,
                tensors = [serialized_inputs]
            )
        
            # ---- Make RPC call ----
            try:
                start_time = time.time()
                self.stats.forward_qps.update(1)
                self.stats.forward_bytes_out.update(sys.getsizeof(request))
                logger.debug('<white>Dendrite</white> <green>Forward Request</green> ---> <white>to</white>:{}, <white>inputs</white>:{}, <white>mode</white>:{}', self.endpoint.ip_str(), inputs.shape, modality)
                response = self.stub.Forward(request, timeout = timeout)
                self.stats.forward_bytes_in.update(sys.getsizeof(response))
                self.stats.forward_elapsed_time.update((time.time() - start_time))

                # Get message
                try:
                    response_message = response.message 
                except:
                    response_message = ''

                # ---- Catch non-code ----
                try:
                    bittensor_code = response.return_code
                except:
                    logger.debug('<white>Dendrite</white> <green>Forward Response</> <--- <white>code</white>:<yellow>UnknownException</yellow>, <white>from</white>:{}, message:None', self.endpoint.ip_str())
                    return zeros, bittensor.proto.ReturnCode.UnknownException

                # ---- Catch bittensor errors ----
                if bittensor_code == bittensor.proto.ReturnCode.UnknownException:
                    logger.debug('<white>Dendrite</white> <green>Forward Response</green> <--- <white>code</white>:<yellow>UnknownException</yellow>, <white>from</white>:{}, message:<red>{}</red>', self.endpoint.ip_str(), response_message)
                    return zeros, bittensor_code

                elif bittensor_code != bittensor.proto.ReturnCode.Success:
                    logger.debug('<white>Dendrite</white> <green>Forward Response</green> <--- <white>code</white>:<yellow>{}</yellow>, <white>from</white>:{}, message:<red>{}</red>', bittensor_code, self.endpoint.ip_str(), response_message)
                    return zeros, bittensor_code

            # ---- Catch GRPC Errors ----
            except grpc.RpcError as rpc_error_call:
                grpc_code = rpc_error_call.code()

                if grpc_code == grpc.StatusCode.DEADLINE_EXCEEDED:
                    logger.debug('<white>Dendrite</white> <green>Forward Response</green> <--- <white>code</white>:<yellow>Timeout</yellow>, <white>from</white>:{}', self.endpoint.ip_str() )
                    return zeros, bittensor.proto.ReturnCode.Timeout

                elif grpc_code == grpc.StatusCode.UNAVAILABLE:
                    logger.debug('<white>Dendrite</white> <green>Forward Response</green> <--- <white>code</white>:<yellow>Unavailable</yellow>, <white>from</white>:{}', self.endpoint.ip_str() )
                    return zeros, bittensor.proto.ReturnCode.Unavailable

                else:
                    logger.debug('<white>Dendrite</white> <green>Forward Response</green> <--- <white>code</white>:<red>UnknownException</red>, <white>from</white>:{} ', self.endpoint.ip_str() )
                    return zeros, bittensor.proto.ReturnCode.UnknownException

            # ---- Catch Unknown Errors ----
            except Exception as e:
                logger.debug('<white>Dendrite</white> <green>Forward Response</green> <--- <white>code</white>:<red>UnknownException</red>, <white>from</white>:{}, <white>message</white>:<red>{}</red>', self.endpoint.ip_str(), e)
                return zeros, bittensor.proto.ReturnCode.UnknownException

            # ---- Check tensor response length ----
            if len(response.tensors) == 0:
                logger.debug('<white>Dendrite</white> <green>Forward Response</green> <--- <white>code</white>:<yellow>EmptyResponse</yellow>, <white>from</white>:{}', self.endpoint.ip_str() )
                return zeros, bittensor.proto.ReturnCode.EmptyResponse

            # ---- Deserialize response ----
            try:
                outputs = response.tensors[0]
                deserializer = bittensor.serializer(  outputs.serializer )
                outputs = deserializer.deserialize( outputs, to_type = bittensor.proto.TensorType.TORCH )

            except Exception as e:
                logger.debug('<white>Dendrite</white> <green>Forward Response</green> <--- <white>code</white>:<red>ResponseDeserializationException</red>, <white>from</white>:{}, message:<red>{}</red> ]', self.endpoint.ip_str(), e)
                return zeros, bittensor.proto.ReturnCode.ResponseDeserializationException
        
            # ---- Check response shape ----
            if  outputs.size(0) != inputs.size(0) \
                or outputs.size(1) != inputs.size(1) \
                or outputs.size(2) != bittensor.__network_dim__:
                    logger.debug('<white>Dendrite</white> <green>Forward Response</green> <--- <white>code</white>:<red>ResponseShapeException</red>, <white>from</white>:{}, <white>shape</white>:{}, <white>expected</white>:{}', self.endpoint.ip_str(), list(outputs.shape), [inputs.size(0), inputs.size(1), bittensor.__network_dim__])
                    return zeros, bittensor.proto.ReturnCode.ResponseShapeException

            # ---- Safe catch NaNs and replace with 0.0 ----
            outputs = torch.where(torch.isnan(outputs), torch.zeros_like(outputs), outputs)
            
        # ---- Catch all ----
        except Exception as e:
            logger.error('<white>Dendrite</white> <green>Forward Response</green> <--- <white>code</white>:<red>UnknownException</red>, <white>from</white>:{}, <white>message</white>:<red>{}</red>', self.endpoint.ip_str(), e)
            return zeros, bittensor.proto.ReturnCode.UnknownException

        # ---- Return ----
        logger.debug('<white>Dendrite</white> <green>Forward Response</green> <--- <white>code</white>:<green>Success</green>, <white>from</white>:{}, <white>outputs</white>:{}', self.endpoint.ip_str(), outputs.shape)
        return outputs, response.return_code

    def backward(
            self,
            inputs_x: torch.Tensor, 
            grads_dy: torch.FloatTensor, 
            modality: bittensor.proto.Modality,
            timeout: int
        ) -> Tuple[torch.Tensor, int]:
        """ Checks and makes RPC Forward call to a remote neuron (calls the Forward method on an Axon terminal of the endpoint)

            Args:
                inputs_x (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    Torch tensor to be sent to the caller associated endpoint neurons.
  
                grads_dy (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    Gradients of this function's outputs computed during the loss.backward() call.
                
                timeout (int):
                    request timeout.

            Returns:
                outputs (:obj:`Tuple[torch.FloatTensor`, torch.LongTensor]`, `optional`):
                    Gradients of the inputs with respect to the inputs and grads of the outputs.

                code (:obj:`bittensor.proto.ReturnCode`, `required`):
                    Return code associated with backward call.

        """
        # ---- Zeros response in the case of failure ----
        zeros = nill_response_for( inputs_x )
 
        # ---- Check inputs size ----
        if torch.numel( inputs_x ) == 0:
            return zeros, bittensor.proto.ReturnCode.EmptyRequest

        # ---- Check grads size ----
        if torch.numel( grads_dy ) == 0:
            return zeros, bittensor.proto.ReturnCode.EmptyRequest

        # ---- Serialization ----
        try:
            serializer = bittensor.serializer( bittensor.proto.Serializer.MSGPACK )
            serialized_inputs = serializer.serialize (inputs_x, modality = modality, from_type = bittensor.proto.TensorType.TORCH )
            serialized_grads = serializer.serialize (grads_dy, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH )
        except Exception as e:
            return zeros, bittensor.proto.ReturnCode.RequestSerializationException
        
        # ---- Make RPC call ----
        try:
            request = bittensor.proto.TensorMessage(
                version = bittensor.__version__,
                public_key = self.wallet.hotkey.public_key,
                nounce = self.nounce,
                signature = self.signature,
                tensors = [serialized_inputs, serialized_grads]
            )
            logger.debug('-> Backward rpc to: {}', self.endpoint)
            response = self.stub.Backward(request, timeout = timeout)

        # ---- Catch GRPC Errors ----
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                return zeros, bittensor.proto.ReturnCode.Timeout
            elif e.code() == grpc.StatusCode.UNAVAILABLE:
                return zeros, bittensor.proto.ReturnCode.Unavailable
            else:
                return zeros, bittensor.proto.ReturnCode.UnknownException

        # ---- Catch Unknown RPC Errors ----
        except Exception as e:
            return zeros, bittensor.proto.ReturnCode.UnknownException

        # ---- Catch Code Errors ----
        try:
            bittensor_code = response.return_code
        except:
            if response.message != None:
                return zeros, bittensor.proto.ReturnCode.UnknownException
            else:
                return zeros, bittensor.proto.ReturnCode.UnknownException

        # ---- Catch negative codes ----
        if bittensor_code != bittensor.proto.ReturnCode.Success:
            if response.message != None:
                return zeros, bittensor_code
            else:
                return zeros, bittensor_code

        # ---- Check for empty response ----
        if len(response.tensors) == 0:
            return zeros, bittensor.proto.ReturnCode.EmptyResponse

        # ---- Post-process request ----
        try:
            outputs = response.tensors[0]
            deserializer = bittensor.serializer( outputs.serializer )
            outputs = deserializer.deserialize( outputs, to_type = bittensor.proto.TensorType.TORCH )
        except Exception as e:
            return zeros, bittensor.proto.ReturnCode.ResponseDeserializationException

        # ---- Check response shape is same as inputs ----
        if  outputs.size(0) != inputs_x.size(0) \
            or outputs.size(1) != inputs_x.size(1) \
            or outputs.size(2) != inputs_x.size(2):
                return zeros, bittensor.proto.ReturnCode.ResponseShapeException 

        # ---- Safe catch NaNs and replace with 0.0 ----
        outputs = torch.where(torch.isnan(outputs), torch.zeros_like(outputs), outputs)
   
        # ---- Return ----
        return outputs, bittensor.proto.ReturnCode.Success
