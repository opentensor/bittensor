
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

import argparse
import grpc
import sys
import os
import time
import torch
import torch.nn as nn
import traceback

from termcolor import colored
from loguru import logger
from munch import Munch
from types import SimpleNamespace
from typing import Tuple, List, Optional

import bittensor
import bittensor.utils.networking as net
import bittensor.utils.stats as stat_utils
import bittensor.serialization as serialization
from bittensor.exceptions.handlers import rollbar

# Helper function for filling nill (zero) responses on failures.
def nill_response_for(inputs):
    if torch.numel(inputs) == 0:
        return torch.tensor([])
    return torch.zeros( (inputs.size(0), inputs.size(1), bittensor.__network_dim__), dtype = torch.float32)

class Receptor(nn.Module):
    """ Encapsulates a grpc connection to an axon endpoint.
    """

    def __init__(
            self, 
            neuron: bittensor.proto.Neuron, 
            config: Munch = None, 
            wallet: 'bittensor.Wallet' = None, 
            **kwargs
        ):
        r""" Initializes a receptor grpc connection.
            Args:
                neuron (:obj:`bittensor.proto.Neuron`, `required`):
                    neuron endpoint descriptor proto.
                config (:obj:`Munch`, `optional`): 
                    receptor.Receptor.config()
                wallet (:obj:`bittensor.Wallet`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
                pass_gradients (default=True, type=bool)
                    Switch to true if the neuron passes gradients to downstream peers.
                        By default the backward call i.e. loss.backward() triggers passing gradients on the wire.
                timeout (default=0.5, type=float):
                    The per request RPC timeout. a.k.a the maximum request time.
                do_backoff (default=True, type=bool)
                    Neurons who return non successful return codes are
                        periodically not called with a multiplicative backoff.
                        The backoff doubles until max_backoff and then halves on ever sequential successful request.
                max_backoff (default=100, type=int)
                    The backoff doubles until this saturation point.
        """
        super().__init__()
        if config == None:
            config = Receptor.default_config()
        bittensor.Config.update_with_kwargs(config.receptor, kwargs) 
        Receptor.check_config( config )
        self.config = config # Configuration information.

        if wallet == None:
            wallet = bittensor.Wallet( self.config )
        self.config.wallet = wallet.config.wallet
        self.wallet = wallet # Keypair information
        self.neuron = neuron # Endpoint information.
        self.signature = None # Call signature.
        self.nounce = None # Call nounce.
        self.backoff = 0 # Number o queries to backoff.
        self.next_backoff = 1 # Next backoff level.
        self.stats = SimpleNamespace (
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
        try:
            # Check to see if we have the axon on 
            # the local network. 
            # if local endpoint = localhost:port
            if neuron.address == self.config.axon.external_ip:
                self.endpoint = "localhost:" + str(neuron.port)
                
            else:
                self.endpoint = neuron.address + ':' + str(neuron.port)
        except:
            # Otherwise fall back to the remote: address:port
            self.endpoint = neuron.address + ':' + str(neuron.port)
        self.channel = grpc.insecure_channel(
            self.endpoint,
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1)])
        self.stub = bittensor.grpc.BittensorStub(self.channel)
        bittensor.__logger__.debug( 'New receptor: {}{}{}', self.endpoint, self.channel, self.stub )

    @staticmethod   
    def default_config() -> Munch:
        parser = argparse.ArgumentParser()
        Receptor.add_args(parser) 
        config = bittensor.Config.to_config(parser); 
        return config

    @staticmethod   
    def check_config(config: Munch):
        assert config.receptor.forward_timeout >= 0, 'timeout must be positive value, got {}'.format(config.receptor.forward_timeout)
        assert config.receptor.backward_timeout >= 0, 'timeout must be positive value, got {}'.format(config.receptor.backward_timeout)
        assert config.receptor.max_backoff >= 0, 'max_backoff must be positive value, got {}'.format(config.receptor.max_backoff)

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        bittensor.Wallet.add_args( parser )
        try:
            # Can be called multiple times.
            parser.add_argument('--receptor.pass_gradients', default=True, type=bool, 
                help='''Switch to true if the neuron passes gradients to downstream peers.
                        By default the backward call i.e. loss.backward() triggers passing gradients on the wire.''')
            parser.add_argument('--receptor.forward_timeout', default=0.5, type=float, 
                help='''The per forward request RPC timeout. a.k.a the maximum request time.''')
            parser.add_argument('--receptor.backward_timeout', default=0.5, type=float, 
                help='''The per backward request RPC timeout. a.k.a the maximum request time.''')
            parser.add_argument('--receptor.do_backoff', default=True, type=bool, 
                help='''Neurons who return non successful return codes are
                        periodically not called with a multiplicative backoff.
                        The backoff doubles until max_backoff and then halves on ever sequential successful request.''')
            parser.add_argument('--receptor.max_backoff', default=100, type=int, 
                help='''The backoff doubles until this saturation point.''')
        except:
            pass

    def toString(self):
        return self.__str__()

    def __str__(self):
        total_out_bytes = self.stats.forward_bytes_out.value + self.stats.backward_bytes_out.value
        total_in_bytes = self.stats.forward_bytes_in.value + self.stats.backward_bytes_in.value
        total_in_bytes_str = colored('\u290A {:.1f}'.format((total_out_bytes*8)/1000), 'green')
        total_out_bytes_str = colored('\u290B {:.1f}'.format((total_in_bytes*8)/1000), 'red')
        return str(self.neuron.uid) + ":(" + total_in_bytes_str + "/" + total_out_bytes_str + "kB/s)"

    def __del__(self):
        if self.channel is not None:
            self.channel.close()

    def forward(
            self, 
            inputs: torch.Tensor, 
            modality: bittensor.proto.Modality
        ) -> Tuple[torch.Tensor, int]:
        r""" Forward call: Triggers the grpc Forward call to the associated endpoint.

            Args:
                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    Single torch tensor to be sent to the remote neuron endpoint.

                modality (:obj:`bittensor.proto.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality type. Enum in [TEXT, IMAGE, TENSOR]

            Returns:
                output (:obj:`Tuple[torch.FloatTensor, torch.LongTensor]`, `required`):
                    Result tuple from the forward call.

                code (:obj:`bittensor.proto.ReturnCode`, `required`):
                    Return code associated with backward call.
        """
        # ---- Make the query ----
        try:
            outputs, code, message = self._call_forward( 
                inputs = inputs, 
                modality = modality 
            )
        except Exception as e:
            # ---- Uncaught failure in the forward call ----
            message = 'Uncaught error in Forward call with error {}, {}'.format( e, traceback.format_exc())
            outputs = nill_response_for(inputs)
            code = bittensor.proto.ReturnCode.UnknownException

        # --- Stats ----
        try:
            if code in self.stats.codes:
                self.stats.codes[ code ] += 1

            # ---- On failure: Increase backoff and double next_backoff towards max value ---- 
            # Catch all non-success / non-backoff codes and trigger backoff increase. This catches
            # serialization errors, timeouts, unavailable endpoints etc. Note, it can 
            # be triggered by invalid requests on this side of the query.
            if code == bittensor.proto.ReturnCode.Backoff:
                self.backoff -= 1
            elif code in [ bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.EmptyRequest, bittensor.proto.ReturnCode.RequestSerializationException ]:
                self.backoff = 0
                self.next_backoff = max(1, self.next_backoff / 2) # halve the next backoff.
            # Failure was on server side.
            else:
                self.backoff = self.next_backoff
                self.next_backoff = min(self.config.receptor.max_backoff, self.next_backoff * 2)
        except Exception as e:
            message = 'Uncaught error while updating stats with error {}'.format( e )

        # ---- Return outputs and code ---- 
        return outputs, code, message
    
    def backward(
            self, 
            inputs: torch.Tensor, 
            grads: torch.Tensor, 
            code: int, 
            modality: bittensor.proto.Modality
        ) -> Tuple[torch.Tensor, int, str]:
        r""" Backward call: Triggers the grpc Backward call to the associated endpoint.

            Args:
                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    Single torch tensor to be sent to the remote neuron endpoint.
    
                grads (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    List of grad-tensors to send to corresponsing neurons. 

                code (`bittensor.proto.ReturnCode`, `required`):
                    dendrite call return ops from previous forward call.

                modality (:obj:`bittensor.proto.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality type. Enum in [TEXT, IMAGE, TENSOR]

            Returns:
                output (:obj:`Tuple[torch.FloatTensor, torch.LongTensor]`, `required`):
                    Result tuple from the forward call.

                code (:obj:`bittensor.proto.ReturnCode`, `required`):
                    Return code associated with backward call.

                message (str, `required):
                    Message associated with backward call.

        """
        # ---- Make the query ----
        try:
            outputs, code, message = self._call_backward( 
                inputs = inputs, 
                grads = grads, 
                code = code, 
                modality = modality
            )
        except Exception as e:
            message = 'Uncaught error in backward call with error {}'.format( e )
            outputs = nill_response_for(inputs)
            code = bittensor.proto.ReturnCode.UnknownException

        # NOTE (const): Stats/ backoff are not updated for backward calls ---
        # ---- Return outputs and code ---- 
        return outputs, code, message


    def _call_forward(
            self, 
            inputs: torch.Tensor, 
            modality: bittensor.proto.Modality 
        ) -> Tuple[torch.Tensor, int, str]:
        """ Checks and makes RPC Forward call to a remote neuron (calls the Forward method on an Axon terminal of the endpoint)

            Args:  
                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    Torch tensor to be sent to the caller associated endpoint neurons.

                modality (:obj:`bittensor.proto.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality type. Enum in [TEXT, IMAGE, TENSOR]

            Returns:
                output (:obj:`Tuple[torch.FloatTensor`, torch.LongTensor]`, `optional`):
                    Result from forward call. May be None in the case of failure.

                code (:obj:`bittensor.proto.ReturnCode`, `required`):
                    Return code associated with forward call.

                message (str, `required):
                    Message associated with forward call.
        """
        
        zeros = nill_response_for(inputs)

        # ---- Check backoff ----  
        if self.config.receptor.do_backoff and self.backoff >= 1:
            return zeros, bittensor.proto.ReturnCode.Backoff, "backoff"

        # ---- Check inputs size ----
        if torch.numel(inputs) == 0:
            return zeros, bittensor.proto.ReturnCode.EmptyRequest, "empty request"

        # ---- Inputs Serialization ----
        # TODO(const): move serialization up to the process.
        try:
            serializer = serialization.get_serializer( bittensor.proto.Serializer.MSGPACK )
            serialized_inputs = serializer.serialize(inputs, modality = modality, from_type = bittensor.proto.TensorType.TORCH)
        except Exception as e:
            return zeros, bittensor.proto.ReturnCode.RequestSerializationException, str(e)

        # ---- Make RPC call ----
        try:
            # ---- Build request ----
            request = bittensor.proto.TensorMessage(
                version = bittensor.__version__,
                public_key = self.wallet.hotkey.public_key,
                nounce = self.nounce,
                signature = self.signature,
                tensors = [serialized_inputs]
            )

            start_time = time.time()
            self.stats.forward_qps.update(1)
            self.stats.forward_bytes_out.update(sys.getsizeof(request))
            bittensor.__logger__.debug('-> Forward rpc to: {}', self.endpoint)
            response = self.stub.Forward(request, timeout=self.config.receptor.forward_timeout)
            self.stats.forward_bytes_in.update(sys.getsizeof(response))
            self.stats.forward_elapsed_time.update((time.time() - start_time))

        # ---- Catch GRPC Errors ----
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                return zeros, bittensor.proto.ReturnCode.Timeout, 'grpc timeout'
            elif e.code() == grpc.StatusCode.UNAVAILABLE:
                return zeros, bittensor.proto.ReturnCode.Unavailable, 'grpc unavailable'
            else:
                return zeros, bittensor.proto.ReturnCode.UnknownException, 'grpc error with code {}'.format(e.code())

        # ---- Catch Unknown Errors ----
        except Exception as e:
            return zeros, bittensor.proto.ReturnCode.UnknownException, str(e)

        # ---- Catch bittensor codes ----
        try:
            bittensor_code = response.return_code
        except:
            if response.message != None:
                return zeros, bittensor.proto.ReturnCode.UnknownException, response.message
            else:
                return zeros, bittensor.proto.ReturnCode.UnknownException, 'unknown error with no message'

        # ---- Catch negative codes ----
        if bittensor_code != bittensor.proto.ReturnCode.Success:
            if response.message != None:
                return zeros, bittensor_code, response.message
            else:
                return zeros, bittensor_code, 'unknown message for exception'

        # ---- Check tensor response length ----
        if len(response.tensors) == 0:
            return zeros, bittensor.proto.ReturnCode.EmptyResponse, 'empty response'

        # ---- Deserialize response ----
        try:
            outputs = response.tensors[0]
            deserializer = serialization.get_serializer( outputs.serializer )
            outputs = deserializer.deserialize( outputs, to_type = bittensor.proto.TensorType.TORCH )
        except Exception as e:
            return zeros, bittensor.proto.ReturnCode.ResponseDeserializationException, str(e)
    
        # ---- Check response shape ----
        if  outputs.size(0) != inputs.size(0) \
            or outputs.size(1) != inputs.size(1) \
            or outputs.size(2) != bittensor.__network_dim__:
                message = 'Forward request returned tensor with incorrect shape {}'.format(list(outputs.shape))
                return zeros, bittensor.proto.ReturnCode.ResponseShapeException, message

        # ---- Safe catch NaNs and replace with 0.0 ----
        outputs = torch.where(torch.isnan(outputs), torch.zeros_like(outputs), outputs)
        
        # ---- Return ----
        return outputs, bittensor.proto.ReturnCode.Success, 'success'

    def _call_backward( 
            self,
            inputs: torch.Tensor, 
            grads: torch.FloatTensor, 
            code: int,
            modality: bittensor.proto.Modality
        ) -> Tuple[torch.Tensor, int]:
        """ Checks and makes RPC Forward call to a remote neuron (calls the Forward method on an Axon terminal of the endpoint)

            Args:
                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    Torch tensor to be sent to the caller associated endpoint neurons.
  
                grads (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    Gradients of this function's outputs computed during the loss.backward() call.

                code (int):
                    code from forward call.

            Returns:
                outputs (:obj:`Tuple[torch.FloatTensor`, torch.LongTensor]`, `optional`):
                    Gradients of the inputs with respect to the inputs and grads of the outputs.

                code (:obj:`bittensor.proto.ReturnCode`, `required`):
                    Return code associated with backward call.

        """
        # ---- Zeros response in the case of failure ----
        zeros = nill_response_for( inputs )

        # ---- Check if we are passing gradients ----
        if not self.config.receptor.pass_gradients:
            return zeros, bittensor.proto.ReturnCode.Success, 'not passing gradients'
 
        # ---- Check the code from the previous call. Returns the same code.
        if code != bittensor.proto.ReturnCode.Success:
            return zeros, code, 'failed on forward'

        # ---- Check inputs size ----
        if torch.numel(inputs) == 0:
            return zeros, bittensor.proto.ReturnCode.EmptyRequest, 'empty request'

        # ---- Check grads size ----
        if torch.numel(grads) == 0:
            return zeros, bittensor.proto.ReturnCode.EmptyRequest, 'empty request'

        # ---- Serialization ----
        try:
            serializer = serialization.get_serializer( bittensor.proto.Serializer.MSGPACK )
            serialized_grads = serializer.serialize (grads, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH )
            serialized_inputs = serializer.serialize (inputs, modality = modality, from_type = bittensor.proto.TensorType.TORCH )
        except Exception as e:
            return zeros, bittensor.proto.ReturnCode.RequestSerializationException, str(e) 
        
        # ---- Make RPC call ----
        try:
            request = bittensor.proto.TensorMessage(
                version = bittensor.__version__,
                public_key = self.wallet.hotkey.public_key,
                nounce = self.nounce,
                signature = self.signature,
                tensors = [serialized_inputs, serialized_grads]
            )
            bittensor.__logger__.debug('-> Backward rpc to: {}', self.endpoint)
            response = self.stub.Backward(request, timeout=self.config.receptor.backward_timeout)

        # ---- Catch GRPC Errors ----
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                return zeros, bittensor.proto.ReturnCode.Timeout, 'grpc timeout'
            elif e.code() == grpc.StatusCode.UNAVAILABLE:
                return zeros, bittensor.proto.ReturnCode.Unavailable, 'grpc endpoint unavailable'
            else:
                return zeros, bittensor.proto.ReturnCode.UnknownException, 'grpc error with code {}'.format(grpc_code)

        # ---- Catch Unknown RPC Errors ----
        except Exception as e:
            return zeros, bittensor.proto.ReturnCode.UnknownException, str(e)

        # ---- Catch Code Errors ----
        try:
            bittensor_code = response.return_code
        except:
            if response.message != None:
                return zeros, bittensor.proto.ReturnCode.UnknownException, response.message
            else:
                return zeros, bittensor.proto.ReturnCode.UnknownException, 'unknown error from endpoint with no message'

        # ---- Catch negative codes ----
        if bittensor_code != bittensor.proto.ReturnCode.Success:
            if response.message != None:
                return zeros, bittensor_code, response.message
            else:
                return zeros, bittensor_code, 'unknown message for non-successful response code'

        # ---- Check for empty response ----
        if len(response.tensors) == 0:
            return zeros, bittensor.proto.ReturnCode.EmptyResponse, 'empty request'

        # ---- Post-process request ----
        try:
            outputs = response.tensors[0]
            deserializer = serialization.get_serializer( outputs.serializer )
            outputs = deserializer.deserialize( outputs, to_type = bittensor.proto.TensorType.TORCH )
        except Exception as e:
            return zeros, bittensor.proto.ReturnCode.ResponseDeserializationException, str(e)

        # ---- Check response shape is same as inputs ----
        if  outputs.size(0) != inputs.size(0) \
            or outputs.size(1) != inputs.size(1) \
            or outputs.size(2) != inputs.size(2):
                message = 'Backward request returned tensor with incorrect shape {}'.format(list(outputs.shape))
                return zeros, bittensor.proto.ReturnCode.ResponseShapeException, message

        # ---- Safe catch NaNs and replace with 0.0 ----
        outputs = torch.where(torch.isnan(outputs), torch.zeros_like(outputs), outputs)
   
        # ---- Return ----
        return outputs, bittensor.proto.ReturnCode.Success, 'success'
