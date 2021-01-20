'''
The MIT License (MIT)
Copyright © 2021 Opentensor.ai

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.
'''
import argparse
import grpc
import sys
import os
import time
import torch
import torch.nn as nn

from termcolor import colored
from loguru import logger
from munch import Munch
from types import SimpleNamespace
from torch.autograd.function import once_differentiable
from typing import Tuple, List, Optional

import bittensor
import bittensor.utils.networking as net
import bittensor.utils.stats as stat_utils
import bittensor.serialization as serialization
from bittensor.exceptions.handlers import rollbar

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

    def __init__(self, neuron: bittensor.proto.Neuron, config: Munch = None, wallet: 'bittensor.wallet.Wallet' = None):
        r""" Initializes a receptor grpc connection.
            Args:
                neuron (:obj:`bittensor.proto.Neuron`, `required`):
                    neuron endpoint descriptor proto.
                config (:obj:`Munch`, `optional`): 
                    receptor.Receptor.config()
                wallet (:obj:`bittensor.wallet.Wallet`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
        """
        super().__init__()
        if config == None:
            config = Receptor.build_config()
        self.config = config # Configuration information.
        if wallet == None:
            wallet = bittensor.wallet.Wallet()
        self.wallet = wallet # Keypair information
        self.neuron = neuron # Endpoint information.
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
        # Loop back if the neuron is local.
        try:
            external_ip = self.config.axon.external_ip
        except:
            pass
        try:
            external_ip = self.config.axon.external_ip
        except:
            pass
        finally:
            external_ip = None
        if neuron.address == external_ip:
            ip = "localhost:"
            self.endpoint = ip + str(neuron.port)
        else:
            self.endpoint = neuron.address + ':' + str(neuron.port)
        self.channel = grpc.insecure_channel(
            self.endpoint,
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1)])
        self.stub = bittensor.grpc.BittensorStub(self.channel)

    @staticmethod   
    def build_config() -> Munch:
        parser = argparse.ArgumentParser()
        Receptor.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        Receptor.check_config(config)
        return config

    @staticmethod   
    def check_config(config: Munch):
        bittensor.wallet.Wallet.check_config( config )
        assert config.receptor.timeout >= 0, 'timeout must be positive value, got {}'.format(config.receptor.timeout)

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        bittensor.wallet.Wallet.add_args( parser )
        try:
            # Can be called multiple times.
            parser.add_argument('--receptor.pass_gradients', default=True, type=bool, 
                help='''Switch to true if the neuron passes gradients to downstream peers.
                        By default the backward call i.e. loss.backward() triggers passing gradients on the wire.''')
            parser.add_argument('--receptor.timeout', default=0.5, type=float, 
                help='''The per request RPC timeout. a.k.a the maximum request time.''')
            parser.add_argument('--receptor.do_backoff', default=True, type=bool, 
                help='''Neurons who return non successful return codes are
                        periodically not called with a multiplicative backoff.
                        The backoff doubles until max_backoff and then halves on ever sequential successful request.''')
            parser.add_argument('--receptor.max_backoff', default=100, type=int, 
                help='''The backoff doubles until this saturation point.''')
        except:
            pass

    def __str__(self):
        total_out_bytes = self.stats.forward_bytes_out.value + self.stats.backward_bytes_out.value
        total_in_bytes = self.stats.forward_bytes_in.value + self.stats.backward_bytes_in.value
        total_in_bytes_str = colored('\u290A {:.1f}'.format((total_out_bytes*8)/1000), 'green')
        total_out_bytes_str = colored('\u290B {:.1f}'.format((total_in_bytes*8)/1000), 'red')
        return str(self.neuron.uid) + ":(" + total_in_bytes_str + "/" + total_out_bytes_str + "kB/s)"

    def __del__(self):
        if self.channel is not None:
            self.channel.close()

    def forward(self, inputs: torch.Tensor, mode: bittensor.proto.Modality) -> Tuple[torch.Tensor, int]:
        r""" Torch.nn.Module forward call: Triggers the grpc call to the remote neuron on the associated endpoint.
            Call returns the output tensor and a bittensor.proto.ReturnCode.

            Args:
                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    Single torch tensor to be sent to the remote neuron endpoint.

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

        """ Internal autograd-friendly Forward RPC call to a remote neuron (calls the Forward method on an Axon terminal.)

            Args:
                ctx: (:obj:`torch.autograd.ctx`, `required`):
                    Autograd context, saves state information between forward and backward calls. i.e. inputs for gradient computation.

                caller: (:obj:`Receptor`, `required`):
                    Caller object the remote neuron containing the endpoint information, RPC channel etc.

                dummy: (:obj:`torch.Tensor`, `required`):
                    Dummy torch tensor used to ensure that torch.backward computation is called on this function 
                    regardless of the input types.
  
                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    Torch tensor to be sent to the caller associated endpoint neurons.

                mode (:obj:`bittensor.proto.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality type. Enum in [TEXT, IMAGE, TENSOR]

            Returns:
                output (:obj:`Tuple[torch.FloatTensor`, torch.LongTensor]`, `optional`):
                    Result from forward call. May be None in the case of failure.

                code (:obj:`bittensor.proto.ReturnCode`, `required`):
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
                return zeros, torch.tensor(bittensor.proto.ReturnCode.EmptyRequest)

            # ---- Inputs Serialization ----
            try:
                serializer = serialization.get_serializer( bittensor.proto.Serializer.MSGPACK )
                serialized_inputs = serializer.serialize(inputs, modality = mode, from_type = bittensor.proto.TensorType.TORCH)
            except Exception as e:
                logger.warning('Serialization error with error {}', e)
                return zeros, torch.tensor(bittensor.proto.ReturnCode.RequestSerializationException)
            ctx.serialized_inputs =  serialized_inputs

            # ---- Build request ----
            request = bittensor.proto.TensorMessage(
                version = bittensor.__version__,
                public_key = ctx.caller.wallet.keypair.public_key,
                nounce = ctx.caller.nounce,
                signature = ctx.caller.signature,
                tensors = [serialized_inputs])
        
            # ---- Make RPC call ----
            try:
                
                start_time = time.time()
                ctx.caller.stats.forward_qps.update(1)
                ctx.caller.stats.forward_bytes_out.update(sys.getsizeof(request))
                response = ctx.caller.stub.Forward(request, timeout=caller.config.receptor.timeout)
                ctx.caller.stats.forward_bytes_in.update(sys.getsizeof(response))
                ctx.caller.stats.forward_elapsed_time.update((time.time() - start_time))

                # ---- Catch non-code ----
                try:
                    bittensor_code = response.return_code
                except:
                    logger.error('Unknown exception returned from remote host with message {}, {}', response.message, traceback.format_exc())
                    return zeros, torch.tensor(bittensor_code)

                # ---- Catch bittensor errors ----
                if bittensor_code == bittensor.proto.ReturnCode.UnknownException:
                    logger.error('Unknown exception returned from remote host with message {}, {}', response.message, traceback.format_exc())
                    return zeros, torch.tensor(bittensor_code)

                elif bittensor_code != bittensor.proto.ReturnCode.Success:
                    return zeros, torch.tensor(bittensor_code)

            # ---- Catch GRPC Errors ----
            except grpc.RpcError as rpc_error_call:
                grpc_code = rpc_error_call.code()

                if grpc_code == grpc.StatusCode.DEADLINE_EXCEEDED:
                    return zeros, torch.tensor(bittensor.proto.ReturnCode.Timeout)

                elif grpc_code == grpc.StatusCode.UNAVAILABLE:
                    return zeros, torch.tensor(bittensor.proto.ReturnCode.Unavailable)

                else:
                    logger.error('Uncaught GPRC error exception with code {} from endpoint {}', grpc_code, caller.endpoint)
                    return zeros, torch.tensor(bittensor.proto.ReturnCode.UnknownException)

            # ---- Catch Unknown Errors ----
            except Exception as e:
                logger.error('Uncaught error in forward call with error {} and endpoint', e, caller.endpoint)
                return zeros, torch.tensor(bittensor.proto.ReturnCode.UnknownException)

            # ---- Check tensor response length ----
            if len(response.tensors) == 0:
                return zeros, torch.tensor(bittensor.proto.ReturnCode.EmptyResponse)

            # ---- Deserialize response ----
            try:
                outputs = response.tensors[0]
                deserializer = serialization.get_serializer(  outputs.serializer )
                outputs = deserializer.deserialize( outputs, to_type = bittensor.proto.TensorType.TORCH )

            except Exception as e:
                logger.error('Failed to serialize responses from forward call with error {}', e)
                return zeros, torch.tensor(bittensor.proto.ReturnCode.ResponseDeserializationException)
        
            # ---- Check response shape ----
            if  outputs.size(0) != inputs.size(0) \
                or outputs.size(1) != inputs.size(1) \
                or outputs.size(2) != bittensor.__network_dim__:
                    logger.error('Forward request returned tensor with incorrect shape {}', list(outputs.shape))
                    return zeros, torch.tensor(bittensor.proto.ReturnCode.ResponseShapeException)

            # ---- Safe catch NaNs and replace with 0.0 ----
            outputs = torch.where(torch.isnan(outputs), torch.zeros_like(outputs), outputs)
        
        # ---- Catch all ----
        except Exception as e:
            logger.error('Forward request returned unknown error {}', e)
            return zeros, torch.tensor(bittensor.proto.ReturnCode.UnknownException)

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
                    serializer = serialization.get_serializer( bittensor.proto.Serializer.MSGPACK )

                    # ---- Serialize grads to bitensor_pb2.Tensors ----
                    serialized_grads = serializer.serialize (grads, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)

                except Exception as e:
                    logger.trace('backward failed during serialization of gradients.')
                    return (None, None, zeros, None)

    
                # ---- Build request for backward ----
                request = bittensor.proto.TensorMessage(
                    version = bittensor.__version__,
                    public_key = ctx.caller.wallet.keypair.public_key,
                    nounce = ctx.caller.nounce,
                    signature = ctx.caller.signature,
                    tensors = [serialized_inputs, serialized_grads])

                # --- Send non blocking grad request ----
                # NOTE(const): we dont care about the response.
                try:
                    ctx.caller.stats.backward_qps.update(1)
                    ctx.caller.stats.backwar_bytes_out.update(sys.getsizeof(request))
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
