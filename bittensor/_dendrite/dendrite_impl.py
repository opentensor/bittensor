""" Implementation of class dendrite, which quries endpoints with tensors.
"""
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

from types import SimpleNamespace
from typing import Tuple, List, Union, Optional

import sys
import torch
import pandas
import random
import time
import uuid

from torch.autograd.function import once_differentiable
from loguru import logger
from transformers.utils.logging import enable_explicit_format
from yaml import serialize

import bittensor
from bittensor._endpoint.endpoint_impl import Endpoint
from bittensor._serializer import serializer, serializer_impl
from bittensor._synapse import TextCausalLM, synapse
import bittensor.utils.stats as stat_utils
import bittensor.utils.codes as codes

import wandb


logger = logger.opt(colors=True)

# dummy tensor that triggers autograd 
DUMMY = torch.empty(0, requires_grad=True)

# Global prometheus 
from prometheus_client import Summary, Counter, Histogram, CollectorRegistry
PROM_prometheus_counters = Counter('dendrite_counters', 'dendrite_counters', ['wallet', 'identifier', 'name'])
PROM_prometheus_latency = Histogram('dendrite_latency', 'dendrite_latency', ['wallet', 'identifier'], buckets=list(range(0,bittensor.__blocktime__,1))) 
PROM_prometheus_latency_per_uid = Summary('dendrite_latency_per_uid', 'dendrite_latency_per_uid', ['wallet', 'identifier', 'uid'])
PROM_prometheus_successes_per_uid = Counter('dendrite_successes_per_uid', 'dendrite_successes_per_uid', ['wallet', 'identifier', 'uid'])
PROM_prometheus_failures_per_uid = Counter('dendrite_failures_per_uid', 'dendrite_failures_per_uid', ['wallet', 'identifier', 'uid'])


class Dendrite(torch.autograd.Function):
    r""" This is the implementation class for a bittensor.dendrite(). The dendrite class operates as a normal torch autograd friendly operation
    which accepts a list of bittensor.endpoints and a list of torch tensors. The passed endpoints are queried with the passed inputs and either return
    results or zeros. The operation is fully differentiable with a torch computation graph such that calls to loss.backward() produce Backward calls on
    the passed endpoints.

    Args:
        config (:obj:`bittensor.Config`, `optional`, defaults to bittensor.dendrite.config()):
            config namespace object created by calling bittensor.dendrite.config()
        wallet (:obj:`bittensor.Wallet`, `optional`, defaults to bittensor.wallet( name = 'default', wallet ='default')):
            A bittensor wallet object containing a pair of cryptographic keys, the hot and coldkey, used for signing messages
            on the wire.
        receptor_pool (:obj:`bittensor.ReceptorPool`, `optional`, defaults to bittensor.receptor_pool()):
            A bittensor receptor pool object which maintains a set of connections to other peers in the network and operates as
            a normal torch.nn.Module. By default this object is created with the dendrite config.
    """

    def __init__(
            self,
            config: 'bittensor.Config',
            wallet: 'bittensor.Wallet',
            receptor_pool: 'bittensor.ReceptorPool',
            manager: 'BaseManager' = None,
    ):
        r""" Initializes a new Dendrite entry point.
            Args:
                receptor_pool (:obj:`bittensor.ReceptorPool`, `required`):
                    bittensor receptor pool
        """
        self.config = config
        self.wallet = wallet
        self.receptor_pool = receptor_pool
        self.manager = manager
        # ---- Dendrite stats
        # num of time we have sent request to a peer, received successful respond, and the respond time
        self.stats = self._init_stats()
        self._prometheus_uuid = uuid.uuid1()

    def __str__(self):
        return "Dendrite({}, {})".format(self.wallet.hotkey.ss58_address, self.receptor_pool)

    def __repr__(self):
        return self.__str__()

    def __del__(self):
        if self.manager:
            self.manager.deduct_connection_count()

        if bittensor != None:
            bittensor.logging.success('Dendrite Deleted', sufix = '')


    @staticmethod
    def forward(
            ctx,
            dendrite: 'bittensor.Dendrite',
            dummy: torch.Tensor,
            endpoints: List['bittensor.Endpoint'],
            synapses: List[ 'bittensor.Synapse' ],
            timeout: int,
            requires_grad: bool,
            *inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """ Internal autograd-friendly Forward RPC call to a list of neuron endpoints.

            Args:
                ctx: (:obj:`torch.autograd.ctx`, `required`):
                    Autograd context, saves state information between forward and backward calls. i.e. inputs for gradient computation.

                dendrite: (:obj:`bittensor.Dendrite`, `required`):
                    Pointer to a bittensor dendrite object on which we are creating the forward requests.

                dummy: (:obj:`torch.Tensor`, `required`):
                    Dummy torch tensor used to ensure that torch.backward computation is called on this function 
                    regardless of the input types.

                endpoints (:obj:`List[bittensor.Endpoint']` of shape :obj:`(n_endpoints)`, `required`):
                    List of endpoints which match length of inputs. Inputs are sent forward to these endpoints.

                synapses (:obj:`List[ 'bittensor.Synapse' ]` of shape :obj:`(num_synapses)`, `required`):
                    Bittensor synapse objects with arguments. Each corresponds to a synapse function on the axon.
                    Responses are packed in this ordering. 

                timeout (int):
                    request timeout.

                requires_grad (int, default = dendrite.requires_grad, `optional`):
                    If true, the backward pass triggers passing gradients on the wire.

                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(n_endpoints)`, `required`):
                    List of torch tensors to be sent to the associated endpoints.

            Returns:
                codes (:obj:`torch.LongTensor` of shape :obj:`(n_endpoints)` `required`):
                    Return code associated with forward call.

                times (:obj:`torch.FloatTensor` of shape :obj:`[ num_endpoints ]`, `required`):
                    times per call.
                
                outputs (:obj:`List[torch.FloatTensor` of shape :obj:`num_synapses * n_endpoints * (-1, -1, -1) `, `required`):
                    List of outputs from each synapses and each endpoint unfolded into a single list. Non-responses are zeroes of expected shape.
        """
        ctx.receptor_pool = dendrite.receptor_pool
        ctx.endpoints, ctx.synapses, ctx.inputs, ctx.timeout, ctx.does_requires_grad = endpoints, synapses, inputs, timeout, requires_grad
        inputs:List[torch.Tensor] = [x.cpu().clone().detach() for x in inputs]

        # Ouputs are list of lists where the outer list corresponds to the endpoints and the 
        # inner list corresponds to the synapses.
        forward_outputs, forward_codes, forward_times = ctx.receptor_pool.forward (
            endpoints = endpoints,
            synapses = synapses,
            inputs = inputs,
            timeout = timeout,
        )
        ctx.forward_codes = forward_codes

        # We need to flatten the outputs across the synapse dimension.
        def flatten(t):
            return [item for sublist in t for item in sublist]
        # flattened items now have length num_endpoints * num_synapses
        # where endpoint i's jth outputs is at position (num_synapses * i ) + j
        flattened_forward_codes: List[ bittensor.proto.ReturnCode ] = flatten( forward_codes )
        flattened_forward_times: List[float] = flatten( forward_times )
        flattened_forward_outputs: List[torch.Tensor] = flatten( forward_outputs )

        # We will pack all the codes and times into a single tensor 
        flattened_torch_codes: torch.LongTensor = torch.tensor(flattened_forward_codes, dtype=torch.int64)
        flattened_torch_times: torch.FloatTensor  = torch.tensor(flattened_forward_times, dtype=torch.float32)

        # Return all outputs as a tuple of torch tensors of length 2 + (num_endpoints * num_synapses) 
        return (flattened_torch_codes, flattened_torch_times, *flattened_forward_outputs)

    @staticmethod
    @once_differentiable
    def backward(
            ctx,
            unused_code_grads: torch.FloatTensor,
            unused_time_grads: torch.FloatTensor,
            *output_grads: torch.FloatTensor
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """ Internal autograd-friendly Backward RPC call to a list of neuron endpoints.

            Args:
                ctx: (:obj:`torch.autograd.ctx`, `required`):
                    Autograd context, saves state information between forward and backward calls. i.e. inputs for gradient computation.

                unused_code_grads: (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    Gradients of this function's codes. (Unused)

                unused_time_grads: (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    Gradients of this function's query times. (Unused)

                grads (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    Gradients of this function's outputs computed during the loss.backward() call.
                    This is a list item of size num_endpoints * num_synapses.
            
            Returns:
                DUMMY, None, None, None,
                outputs (:obj:`List[torch.FloatTensor], `optional`):
                    Gradient results for each input.

        """
        # output_grads is a list of gradients per synapse. They need to be packed (unflattened)
        # into a list of lists.
        packed_grads: List[ List [ torch.FloatTensor ] ] = [ output_grads[ s : s + len(ctx.synapses) ] for s in range (0, len(output_grads), len( ctx.synapses )) ]
        if ctx.does_requires_grad:
            input_grads, _, _ = ctx.receptor_pool.backward(
                endpoints = ctx.endpoints,
                inputs = ctx.inputs,
                synapses = ctx.synapses,
                grads = packed_grads,
                timeout = ctx.timeout,
            )
            # Input grads is a list of lists
            # We need to flatten the outputs across the synapse dimension.
            def flatten(t):
                return [item for sublist in t for item in sublist]
            flattened_input_grads: List[torch.FloatTensor]  = flatten( input_grads )
            return (None, None, None, None, None, None, *flattened_input_grads)
        else:
            # Create nill responses for each input and each synapse.
            input_grads = [ syn.nill_backward_response_tensor ( inp ) for inp in ctx.inputs for syn in ctx.synapses ]
            return (None, None, None, None, None, None, *input_grads)

    def _forward(
            self,
            endpoints: List [ 'bittensor.Endpoint' ],
            synapses: List[ 'bittensor.Synapse' ],
            inputs: List [ torch.Tensor ],
            timeout: Optional [ int ]  = None,
            requires_grad: Optional [ bool ] = None,
    ) -> Tuple [ List[ torch.Tensor ], List[ torch.LongTensor ], List [ torch.FloatTensor ]]:
        r""" Internal Forward tensor inputs to a list of neuron endpoints.

            Args:
                endpoints (:obj:`List[bittensor.Endpoint]` of shape :obj:`(num_endpoints)`, `required`):
                    List of remote endpoints which match length of inputs. Tensors from inputs are sent forward to these endpoints.

                synapses (:obj:`List[ 'bittensor.Synapse' ]` of shape :obj:`(num_synapses)`, `required`):
                    Bittensor synapse objects with arguments. Each corresponds to a synapse function on the axon.
                    Responses are packed in this ordering. 

                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(num_endpoints * [shape])`, `required`):
                    List of tensors to send to corresponding endpoints. Tensors are of arbitrary type and shape depending on the
                    synapse.

                timeout (int, default = dendrite.timeout, `optional`):
                    request timeout.

                requires_grad (int, default = dendrite.requires_grad, `optional`):
                    If true, the backward pass triggers passing gradients on the wire.

            Returns:
                outputs (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                    Output encodings of inputs produced by the remote endpoints. Non-responses are zeroes of common shape.

                codes (:obj:`List[torch.LongTensor]` of shape :obj:`[num_endpoints]`, `required`):
                    Return codes per endpoint per synapse.

                times (:obj:`torch.FloatTensor` of shape :obj:`[ num_endpoints ]`, `required`):
                    Call times per endpoint per synapse.

        """
        timeout:int = timeout if timeout is not None else self.config.dendrite.timeout
        requires_grad:bool = requires_grad if requires_grad is not None else self.config.dendrite.requires_grad

        # The forwarnd response is a tuple with shape (flattened_torch_codes, flattened_torch_times, *flattened_forward_outputs)
        # packed with torch tensors of length 2 + (num_endpoints * num_synapses). The first two tensors are codes and times
        # the last (num_endpoints * num_synapses) tensors are per endpoint per synapse output tensors.
        forward_response: List[torch.Tensor] = Dendrite.apply (
            self,
            DUMMY,
            endpoints,
            synapses,
            timeout,
            requires_grad,
            *inputs
        )

        # Split codes into num_synapse lists of codes
        # split_codes is a list of tensors codes each with length num_synapses
        codes: torch.LongTensor = forward_response[0]
        packed_codes: List[torch.LongTensor] = torch.split( codes, len( synapses ) )

        # Split times into num_synapse lists of codes
        # split_times is a list of tensors times each with length num_synapses
        times: torch.FloatTensor  = forward_response[1]
        packed_times: List[torch.FloatTensor] = torch.split( times, len( synapses ) )

        # Output responses is a list with length num_endpoints num_synapses
        # we need to pack the responses into a list of lists corresponding to
        # each endpoint.
        outputs: List[torch.Tensor] = forward_response[2:]
        packed_outputs: List[ List[torch.Tensor] ] = [  outputs[ s : s + len(synapses) ] for s in range (0, len(outputs), len( synapses )) ]

        # === Prometheus counters.
        if self.config.dendrite.prometheus.level != bittensor.prometheus.level.OFF.name:
            PROM_prometheus_counters.labels( wallet = self.wallet.hotkey.ss58_address, identifier = self._prometheus_uuid, name = 'total_requests' ).inc()
            PROM_prometheus_counters.labels( wallet = self.wallet.hotkey.ss58_address, identifier = self._prometheus_uuid, name = 'total_endpoint_requests' ).inc( len(endpoints) )
            PROM_prometheus_counters.labels( wallet = self.wallet.hotkey.ss58_address, identifier = self._prometheus_uuid, name = 'total_request_bytes' ).inc( sum(p.element_size() * p.nelement() for p in inputs) )
            PROM_prometheus_counters.labels( wallet = self.wallet.hotkey.ss58_address, identifier = self._prometheus_uuid, name = 'total_request_params' ).inc( sum(p.numel() for p in inputs) )

            # Capture synapses.
            for synapse in enumerate( synapses ):
                PROM_prometheus_counters.labels( wallet = self.wallet.hotkey.ss58_address, identifier = self._prometheus_uuid, name = str(synapse) ).inc()

            for i in range(len(endpoints)):
                n_success = (codes[i] == 1).sum().item()
                is_success = (n_success > 0) # One is a success.
                response_time = times[i].mean().item()

                # Capture outputs.
                PROM_prometheus_counters.labels( wallet = self.wallet.hotkey.ss58_address, identifier = self._prometheus_uuid, name = 'total_response_bytes' ).inc( sum(p.element_size() * p.nelement() for p in outputs[i]) )
                PROM_prometheus_counters.labels( wallet = self.wallet.hotkey.ss58_address, identifier = self._prometheus_uuid, name = 'total_response_params' ).inc( sum(p.numel() for p in outputs[i]) )

                # Capture global success rates.
                if is_success: 
                    PROM_prometheus_counters.labels( wallet = self.wallet.hotkey.ss58_address, identifier = self._prometheus_uuid, name = 'total_success' ).inc()
                    PROM_prometheus_latency.labels( wallet = self.wallet.hotkey.ss58_address, identifier = self._prometheus_uuid).observe( response_time )
                else: 
                    PROM_prometheus_counters.labels( wallet = self.wallet.hotkey.ss58_address, identifier = self._prometheus_uuid, name = 'total_failure' ).inc()

                # === Prometheus DEBUG (per uid info.)
                if self.config.dendrite.prometheus.level == bittensor.prometheus.level.DEBUG.name:
                    if is_success:
                        PROM_prometheus_latency_per_uid.labels( wallet = self.wallet.hotkey.ss58_address, identifier = self._prometheus_uuid, uid = str(endpoints[i].uid) ).observe( response_time )
                        PROM_prometheus_successes_per_uid.labels( wallet = self.wallet.hotkey.ss58_address, identifier = self._prometheus_uuid, uid = str(endpoints[i].uid) ).inc()
                    else:
                        PROM_prometheus_failures_per_uid.labels( wallet = self.wallet.hotkey.ss58_address, identifier = self._prometheus_uuid, uid = str(endpoints[i].uid) ).inc()

        return packed_outputs, packed_codes, packed_times


    def generate(
        self,
        endpoints: Union[ torch.LongTensor, List[torch.LongTensor], List['bittensor.Endpoint'], 'bittensor.Endpoint' ],
        prompt: Union[str, List[str], List[torch.LongTensor], torch.LongTensor],
        timeout: int = None,
        topk:int = 50, 
        num_to_generate: int = 256,
        num_beams: int = 5,
        no_repeat_ngram_size: int = 2,
        early_stopping: bool = False,
        num_return_sequences: int = 1,
        do_sample: bool = False,
        top_p: float = 0.95, 
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        max_time: float = 150,
        num_beam_groups: int = 1,
    ) -> Tuple[ List[str], List[float], List[str] ]:
        """
        Returns a tuple containing the prompt generations produced by endpoints with corresponding parsed codes and query times.

        Args:
            endpoints (:obj:`Union[torch.LongTensor, List[torch.LongTensor], List[bittensor.Endpoint], bittensor.Endpoint]` of shape :obj:`(num_endpoints)`, `required`):
                        Endpoints to send inputs to. Endpoint can be one of the following types:
                            - a single endpoint tensor shape [250]
                            - a set of endpoint tensors shape [n, 250]
                            - a list of endpoints tensors each of shape [250]
                            - a single endpoint object. Inputs will be sent to this endpoint alone.
                            - a list of endpoint objects. All inputs will be sent to these endpoints.

            prompts (:obj:`Union[str,  List[str], List[torch.LongTensor], torch.LongTensor]` of shape :obj:`(num_endpoints * [batch_size, sequence_len])`, `required`):
                        Tokenized sentences to send on the wire. Inputs can be one of the following types:
                            - a single string: the string will be tokenized using the bittensor tokenizer.
                            - a list of strings: the strings will be tokenized using the bittensor tokenizer.
                            - a tensor with shape [batch_size, sequence_len], assumed to be the output of bittensor tokenizer.
                            - a tensor with shape [n, batch_size, sequence_len], the operation will unbind the tensor and pass inputs to endpoints.
                            - a list of tensors of type long each representing a tokenized sentence to be sent to each endpoint.
                        If inputs are tensors they will be cast to int64 format before sending on the wire.

            timeout (:type:`int`, default = dendrite.timeout `optional`):
                Request timeout. Queries that do not respond will be replaced by zeros.

            Topk (:obj:int, :default: 50):
                The number of highest probability vocabulary tokens to keep for top-k-filtering. 
            num_to_generate (:obj: int, :default: 256):
                The number of tokens to generate using the language model
            num_beams (:obj: int, :default: 5):
                The number of beams to keep during beam search
            no_repeat_ngram_size (:obj: int, :default: 2):
                The number of repeat n gram allowed
            early_stopping: (:obj: bool, :default: True):
                If the model should early stop if the probabilty drops a certain threshold
            num_return_sequences: (:obj: int, :default: 1):
                How many sequences should the model return
            do_sample (:obj: bool, :default: False):
                If the model should do sample its probablity during generation
            top_p (:obj: float, :default: 0.95): 
                probability cutoff for top p sampling
            temperature: (:obj: float, :default: 1.0):
                The value used to module the next token probabilities for the softmax calculation
            repetition_penalty (:obj: float, :default: 1.0):
                The parameter for repetition penalty. 1.0 means no penalty.
            length_penalty (:obj: float, :default: 1.0): 
                The parameter for length penalty. 0.0 means no penalty, <0 to encourage longer sequences.
            max_time (:obj: float, :default: 150): 
                The maximum time that a server can use to generate
            num_beam_groups (:obj: int, :default: 1):
                Number of groups to divide num_beams into in order to ensure diversity among different groups of beams. 
        Returns:
            codes (:obj:`List[str]`, `required`):
                Parsed codes from each endpoint from query.

            times (:obj:`List[float]`, `required`):
                Query times for each call from each endpoint.

            generations (:obj:`List[str]`, `required`):
                Generations from each endpoint.
        """
        tokenizer = bittensor.tokenizer()
        response = self.text ( 
            endpoints=endpoints, 
            inputs = prompt,
            synapses = [
                synapse.TextSeq2Seq(
                    topk = topk,
                    num_to_generate = num_to_generate,
                    num_beams = num_beams,
                    no_repeat_ngram_size = no_repeat_ngram_size,
                    early_stopping = early_stopping,
                    num_return_sequences = num_return_sequences,
                    do_sample = do_sample,
                    top_p = top_p,
                    temperature = temperature,
                    repetition_penalty = repetition_penalty,
                    length_penalty = length_penalty,
                    max_time = max_time,
                    num_beam_groups = num_beam_groups,
                )
            ],
            timeout = timeout
        )
        # Parse responses to natural language.
        generations = []
        for text_tensor in response[0]:
            generations.append( tokenizer.decode( text_tensor[0][0].long() ) )
        codes = []
        for code_tensor in response[1]:
            codes.append( bittensor.utils.codes.code_to_string( code_tensor ) )
        times = []
        for time_tensor in response[2]:
            times.append( time_tensor.item() )
        return codes, times, generations

    def text (
        self,
        endpoints: Union[ torch.LongTensor, List[torch.LongTensor], List['bittensor.Endpoint'], 'bittensor.Endpoint' ],
        synapses: List[ 'bittensor.Synapse' ],
        inputs: Union[str, List[str], List[torch.LongTensor], torch.LongTensor],
        timeout: int = None,
        requires_grad: bool = None,
    ) -> Tuple[ Union[List[torch.FloatTensor], torch.FloatTensor], torch.LongTensor, torch.FloatTensor]:
        r""" Forward text inputs to a list of neuron endpoints and returns logit encodings or timeout.

                Args:
                    endpoints (:obj:`Union[torch.LongTensor, List[torch.LongTensor], List[bittensor.Endpoint], bittensor.Endpoint]` of shape :obj:`(num_endpoints)`, `required`):
                        Endpoints to send inputs to. Endpoint can be one of the following types:
                            - a single endpoint tensor shape [250]
                            - a set of endpoint tensors shape [n, 250]
                            - a list of endpoints tensors each of shape [250]
                            - a single endpoint object. Inputs will be sent to this endpoint alone.
                            - a list of endpoint objects. All inputs will be sent to these endpoints.

                    synapses (:obj:`List[ 'bittensor.Synapse' ]` of shape :obj:`(num_synapses)`, `required`):
                        Bittensor synapse objects with arguments. Each corresponds to a synapse function on the axon.
                        Responses are packed in this ordering. 

                    inputs (:obj:`Union[str,  List[str], List[torch.LongTensor], torch.LongTensor]` of shape :obj:`(num_endpoints * [batch_size, sequence_len])`, `required`):
                        Tokenized sentences to send on the wire. Inputs can be one of the following types:
                            - a single string: the string will be tokenized using the bittensor tokenizer.
                            - a list of strings: the strings will be tokenized using the bittensor tokenizer.
                            - a tensor with shape [batch_size, sequence_len], assumed to be the output of bittensor tokenizer.
                            - a tensor with shape [n, batch_size, sequence_len], the operation will unbind the tensor and pass inputs to endpoints.
                            - a list of tensors of type long each representing a tokenized sentence to be sent to each endpoint.
                        If inputs are tensors they will be cast to int64 format before sending on the wire.

                    timeout (:type:`int`, default = dendrite.timeout `optional`):
                        Request timeout. Queries that do not respond will be replaced by zeros.

                    requires_grad (:type:`int`, default = dendrite.requires_grad, `optional`):
                        If true, the backward pass triggers passing gradients on the wire.

                Returns:
                    outputs (:obj:`List[ List[ torch.FloatTensor ] ]` of shape :obj:`num_synapses * ( num_endpoints * ( -1, -1, -1 ) )`, `required`):
                        List of outputs from synapses, each a list of size num_endpoints of tensors with relevant size. Non-responses are zeroes of relevant 
                        synapse shape.

                    codes (:obj:`List [ torch.LongTensor ]` of shape :obj:`[ num_endpoints ]`, `required`):
                        Return code per call per synapse.

                    times (:obj:`List [ torch.FloatTensor ]` of shape :obj:`[ num_endpoints ]`, `required`):
                        Times per call per synapse.
            """
        formatted_endpoints, formatted_inputs = self.format_text_inputs ( 
            endpoints = endpoints, 
            inputs = inputs
        )
        outputs, codes, times = self._forward (
            endpoints = formatted_endpoints,
            synapses = synapses,
            inputs = formatted_inputs,
            timeout = timeout,
            requires_grad = requires_grad,
        )
        # Return.
        self.update_stats( formatted_endpoints, synapses, formatted_inputs, outputs, codes, times )
        return outputs, codes, times

    def text_causal_lm (
        self,
        endpoints: Union [ torch.LongTensor, List [ torch.LongTensor ], List[ 'bittensor.Endpoint' ], 'bittensor.Endpoint' ],
        inputs: Union [ str, List[ str ], List [ torch.LongTensor ], torch.LongTensor],
        synapse: Optional[ 'bittensor.synapse.TextCausalLM' ] = synapse.TextCausalLM(),
        timeout: Optional [ int ] = None,
        requires_grad: Optional [ bool ] = None,
    ) -> Tuple[Union[List[torch.FloatTensor], torch.FloatTensor], torch.LongTensor, torch.FloatTensor]:
        r""" Forward text inputs to a list of neuron endpoints and returns logit encodings or timeout.

                Args:
                    endpoints (:obj:`Union[torch.LongTensor, List[torch.LongTensor], List[bittensor.Endpoint], bittensor.Endpoint]` of shape :obj:`(num_endpoints)`, `required`):
                        Endpoints to send inputs to. Endpoint can be one of the following types:
                            - a single endpoint tensor shape [250]
                            - a set of endpoint tensors shape [n, 250]
                            - a list of endpoints tensors each of shape [250]
                            - a single endpoint object. Inputs will be sent to this endpoint alone.
                            - a list of endpoint objects. All inputs will be sent to these endpoints.


                    inputs (:obj:`Union[str,  List[str], List[torch.LongTensor], torch.LongTensor]` of shape :obj:`(num_endpoints * [batch_size, sequence_len])`, `required`):
                        Tokenized sentences to send on the wire. Inputs can be one of the following types:
                            - a single string: the string will be tokenized using the bittensor tokenizer.
                            - a list of strings: the strings will be tokenized using the bittensor tokenizer.
                            - a tensor with shape [batch_size, sequence_len], assumed to be the output of bittensor tokenizer.
                            - a tensor with shape [n, batch_size, sequence_len], the operation will unbind the tensor and pass inputs to endpoints.
                            - a list of tensors of type long each representing a tokenized sentence to be sent to each endpoint.
                        If inputs are tensors they will be cast to int64 format before sending on the wire.

                    synapse (:type:`'bittensor.synapse.TextCausalLM'`, default = bittensor.synapse.TextCausalLM(), `optional`):
                        Synapse axon function call which defaults to bittensor.synapse.TextCausalLM().
                    
                    timeout (:type:`int`, default = dendrite.timeout `optional`):
                        Request timeout. Queries that do not respond will be replaced by zeros.

                    requires_grad (:type:`int`, default = dendrite.requires_grad, `optional`):
                        If true, the backward pass triggers passing gradients on the wire.

                Returns:
                    outputs (:obj:`List[ torch.FloatTensor ]` of shape :obj:`num_endpoints * (batch_size, sequence_len, bittensor.__vocab_size__ )`, `required`):
                        List of output logit encodings of inputs produced by each remote endpoints. Non-responses are zeroes of input shape plus output dimension.
                        The first dimension will match the number of endpoints queried.

                    codes (:obj:`torch.LongTensor` of shape :obj:`[ num_endpoints ]`, `required`):
                        dendrite call return ops.

                    times (:obj:`torch.FloatTensor` of shape :obj:`[ num_endpoints ]`, `required`):
                        times per call.
        """
        if synapse.synapse_type != bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM:
            raise ValueError( "Passed synapse must have type: {} got {} instead".format( bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM, synapses.synapse_type ) )

        # Format inputs.
        formatted_endpoints, formatted_inputs = self.format_text_inputs ( 
            endpoints = endpoints, 
            inputs = inputs
        )
        # Optionally convert synapses and set typing info.
        synapses = [ synapse ]
        # Make calls.
        outputs, codes, times = self._forward (
            endpoints = formatted_endpoints,
            synapses = synapses,
            inputs = formatted_inputs,
            timeout = timeout,
            requires_grad = requires_grad,
        )
        # Return.
        self.update_stats( formatted_endpoints, synapses, formatted_inputs, outputs, codes, times )
        return outputs[0], codes[0], times[0]

    def text_causal_lm_next(
            self,
            endpoints: Union[torch.LongTensor, List[torch.LongTensor], List['bittensor.Endpoint'], 'bittensor.Endpoint'],
            inputs: Union[str, List[str], List[torch.LongTensor], torch.LongTensor],
            synapse: Optional['bittensor.synapse.TextCausalLMNext'] = synapse.TextCausalLMNext(),
            timeout: Optional[int] = None,
            requires_grad: Optional[bool] = None,
    ) -> Tuple[Union[List[torch.FloatTensor], torch.FloatTensor], torch.LongTensor, torch.FloatTensor]:
        r""" Forward text inputs to a list of neuron endpoints and returns logit encodings or timeout.

                Args:
                    endpoints (:obj:`Union[torch.LongTensor, List[torch.LongTensor], List[bittensor.Endpoint], bittensor.Endpoint]` of shape :obj:`(num_endpoints)`, `required`):
                        Endpoints to send inputs to. Endpoint can be one of the following types:
                            - a single endpoint tensor shape [250]
                            - a set of endpoint tensors shape [n, 250]
                            - a list of endpoints tensors each of shape [250]
                            - a single endpoint object. Inputs will be sent to this endpoint alone.
                            - a list of endpoint objects. All inputs will be sent to these endpoints.


                    inputs (:obj:`Union[str,  List[str], List[torch.LongTensor], torch.LongTensor]` of shape :obj:`(num_endpoints * [batch_size, sequence_len])`, `required`):
                        Tokenized sentences to send on the wire. Inputs can be one of the following types:
                            - a single string: the string will be tokenized using the bittensor tokenizer.
                            - a list of strings: the strings will be tokenized using the bittensor tokenizer.
                            - a tensor with shape [batch_size, sequence_len], assumed to be the output of bittensor tokenizer.
                            - a tensor with shape [n, batch_size, sequence_len], the operation will unbind the tensor and pass inputs to endpoints.
                            - a list of tensors of type long each representing a tokenized sentence to be sent to each endpoint.
                        If inputs are tensors they will be cast to int64 format before sending on the wire.

                    synapse (:type:`'bittensor.synapse.TextCausalLMNext'`, default = bittensor.synapse.TextCausalLMNext(), `optional`):
                        Synapse axon function call which defaults to bittensor.synapse.TextCausalLMNext().

                    timeout (:type:`int`, default = dendrite.timeout `optional`):
                        Request timeout. Queries that do not respond will be replaced by zeros.

                    requires_grad (:type:`int`, default = dendrite.requires_grad, `optional`):
                        If true, the backward pass triggers passing gradients on the wire.

                Returns:
                    outputs (:obj:`List[ torch.FloatTensor ]` of shape :obj:`num_endpoints * ( >= batch_size * (2 * topk + 1) )`, `required`):
                        List of output topk phrases encodings of inputs produced by each remote endpoints.
                        Non-responses are zeroes of input shape plus output dimension.
                        The first dimension will match the number of endpoints queried.

                    codes (:obj:`torch.LongTensor` of shape :obj:`[ num_endpoints ]`, `required`):
                        dendrite call return ops.

                    times (:obj:`torch.FloatTensor` of shape :obj:`[ num_endpoints ]`, `required`):
                        times per call.
        """
        if synapse.synapse_type != bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT:
            raise ValueError(f"Passed synapse must have type: {bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT} "
                             f"got {synapse.synapse_type} instead")

        # Format inputs.
        formatted_endpoints, formatted_inputs = self.format_text_inputs(
            endpoints=endpoints,
            inputs=inputs
        )
        # Optionally convert synapses and set typing info.
        synapses = [synapse]
        # Make calls.
        outputs, codes, times = self._forward(
            endpoints=formatted_endpoints,
            synapses=synapses,
            inputs=formatted_inputs,
            timeout=timeout,
            requires_grad=requires_grad,
        )
        # Return.
        self.update_stats(formatted_endpoints, synapses, formatted_inputs, outputs, codes, times)
        return outputs[0], codes[0], times[0]

    def text_last_hidden_state(
            self,
            endpoints: Union[ torch.LongTensor, List[torch.LongTensor], List['bittensor.Endpoint'], 'bittensor.Endpoint' ],
            inputs: Union[str, List[str], List[torch.LongTensor], torch.LongTensor],
            mask: Optional[ Union[int, List[int]] ] = None,
            synapse: Optional[ 'bittensor.synapse.TextLastHiddenState' ] = None,
            timeout: int = None,
            requires_grad: bool = None,
    ) -> Tuple[Union[List[torch.FloatTensor], torch.FloatTensor], torch.LongTensor, torch.FloatTensor]:
        r""" Forward text inputs to a list of neuron endpoints and block until last hidden state responses or timeout.

                Args:
                    endpoints (:obj:`Union[torch.LongTensor, List[torch.LongTensor], List[bittensor.Endpoint], bittensor.Endpoint]` of shape :obj:`(num_endpoints)`, `required`):
                        Endpoints to send inputs to. Endpoint can be one of the following types:
                            - a single endpoint tensor shape [250]
                            - a set of endpoint tensors shape [n, 250]
                            - a list of endpoints tensors each of shape [250]
                            - a single endpoint object. Inputs will be sent to this endpoint alone.
                            - a list of endpoint objects. All inputs will be sent to these endpoints.

                    inputs (:obj:`Union[str,  List[str], List[torch.LongTensor], torch.LongTensor]` of shape :obj:`(num_endpoints * [batch_size, sequence_len])`, `required`):
                        Tokenized sentences to send on the wire. Inputs can be one of the following types:
                            - a single string: the string will be tokenized using the bittensor tokenizer.
                            - a list of strings: the strings will be tokenized using the bittensor tokenizer.
                            - a tensor with shape [batch_size, sequence_len], assumed to be the output of bittensor tokenizer.
                            - a tensor with shape [n, batch_size, sequence_len], the operation will unbind the tensor and pass inputs to endpoints.
                        If inputs are tensors they will be cast to int64 format before sending on the wire.

                    mask (:obj:`Union[int,  List[int]]` of shape :obj:`-1`, default = [], `optional`):
                        A mask used to select which hidden states of the sequence are to be returned by the query.
                            - None or empty: none of the sequences are masked.
                            - a single int: referring to a single sequence index across all batches i.e 0 (first), 5(fifth), -1 (last)
                            - a list of ints: referring to a set of sequence indices across all batches to return. [ 0 (first), 5(fifth), -1 ( and last)]
                        The mask determines which items are non zero in the response.

                    synapse (:type:`'bittensor.synapse.TextLastHiddenState'`, default = bittensor.synapse.TextLastHiddenState(), `optional`):
                        Synapse axon function call which defaults to bittensor.synapse.TextLastHiddenState().

                    timeout (:type:`int`, default = dendrite.timeout `optional`):
                        Request timeout. Queries that do not respond will be replaced by zeros.

                    requires_grad (:type:`int`, default = dendrite.requires_grad, `optional`):
                        If true, the backward pass triggers passing gradients on the wire.

                Returns:
                    outputs (:obj:`List [ torch.FloatTensor ]` of shape :obj:` num_endpoints * ( -1, sequence_len, bittensor.__network_dim__ )`, `required`):
                        List of output last hidden state encodings of inputs produced by remote endpoints. Non-responses are zeroes of input shape plus output dimension.
                        The first dimension will match the number of endpoints queried.

                    codes (:obj:`torch.LongTensor` of shape :obj:`[ num_endpoints ]`, `required`):
                        dendrite call return ops.

                    times (:obj:`torch.FloatTensor` of shape :obj:`[ num_endpoints ]`, `required`):
                        times per call.
        """
        # If the synapse is None, create it.
        if synapse == None:
            synapse = bittensor.synapse.TextLastHiddenState()
        
        # Check past synapse.
        elif synapse.synapse_type != bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE:
            raise ValueError( "Passed synapse must have type:{} got:{} instead".format( bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE, synapses.synapse_type ) )
        
        # Set the synapse mask.
        if mask != None:
            if isinstance(mask, int):
                synapse.mask = [mask]
            elif isinstance(mask, list):
                synapse.mask = mask
            else:
                raise ValueError( "Passed mask must have type: int or List[int] got:{} instead".format( type( mask) ) )

        # Format inputs.
        formatted_endpoints, formatted_inputs = self.format_text_inputs ( 
            endpoints = endpoints, 
            inputs = inputs
        )
        synapses = [ synapse ]

        # Make calls.
        outputs, codes, times = self._forward (
            endpoints = formatted_endpoints,
            synapses = synapses,
            inputs = formatted_inputs,
            timeout = timeout,
            requires_grad = requires_grad,
        )
        # Return.
        self.update_stats( formatted_endpoints, synapses, formatted_inputs, outputs, codes, times )
        return outputs[0], codes[0], times[0]

    def format_text_inputs (
        self,
        endpoints: Union[ torch.LongTensor, List[torch.LongTensor], List['bittensor.Endpoint'], 'bittensor.Endpoint' ],
        inputs: Union[str, List[str], List[torch.LongTensor], torch.LongTensor],
    ) -> Tuple[ 'bittensor.Endpoint', List[torch.LongTensor] ]:
        r""" Formats endpoint and inputs args to a common format.
            Args:
                endpoints (:obj:`Union[torch.LongTensor, List[torch.LongTensor], List[bittensor.Endpoint], bittensor.Endpoint]` of shape :obj:`(num_endpoints)`, `required`):
                    Endpoints to send inputs to. Endpoint can be one of the following types:
                        - a single endpoint tensor shape [250]
                        - a set of endpoint tensors shape [n, 250]
                        - a list of endpoints tensors each of shape [250]
                        - a single endpoint object. Inputs will be sent to this endpoint alone.
                        - a list of endpoint objects. All inputs will be sent to these endpoints.

                inputs (:obj:`Union[str,  List[str], List[torch.LongTensor], torch.LongTensor]` of shape :obj:`(num_endpoints * [batch_size, sequence_len])`, `required`):
                    Tokenized sentences to send on the wire. Inputs can be one of the following types:
                        - a single string: the string will be tokenized using the bittensor tokenizer.
                        - a list of strings: the strings will be tokenized using the bittensor tokenizer.
                        - a tensor with shape [batch_size, sequence_len], assumed to be the output of bittensor tokenizer.
                        - a tensor with shape [n, batch_size, sequence_len], the operation will unbind the tensor and pass inputs to endpoints.
                    If inputs are tensors they will be cast to int64 format before sending on the wire.

            Returns:
                formatted_endpoints (:obj:`Union[torch.LongTensor, List[torch.LongTensor], List[bittensor.Endpoint], bittensor.Endpoint]` of shape :obj:`(num_endpoints)`, `required`):
                    A list of endpoint objects. All inputs will be sent to these endpoints.

                formatted_inputs (:obj:`Union[str,  List[str], List[torch.LongTensor], torch.LongTensor]` of shape :obj:`(num_endpoints * [batch_size, sequence_len])`, `required`):
                    A list of tensor of type long each representing a tokenized sentence to be sent to each endpoint.
        """
        # To be filled. Inputs and endpoint must be list with the same number of elements.
        formatted_inputs = []
        formatted_endpoints = []

        # <<Helper function>> optional casts and then checks shape of inputs.
        def cast_and_check_tensor_input(tensor_input) -> torch.LongTensor:
            if not isinstance(tensor_input, torch.LongTensor):
                try:
                    tensor_input = tensor_input.to(torch.int64)
                except Exception as E:
                    error_msg = 'Error while casting tensor input {} to int64 {}'.format(tensor_input, E)
                    raise ValueError(error_msg) from ValueError()
            if not (isinstance(tensor_input, torch.cuda.LongTensor) or isinstance(tensor_input, torch.LongTensor)):
                raise ValueError(
                    'input {} must be of type torch.LongTensor. Got {}'.format(tensor_input, type(tensor_input)))
            # Expand shape if it is a singular dimension.
            if len(tensor_input.shape) == 1:
                tensor_input = tensor_input.view(1, -1)

            # Check shape.
            if len(tensor_input.shape) != 2:
                error_msg = 'Text inputs should be rank 2 with semantic shape: [batch_size, sequence_len]'
                raise ValueError(error_msg)
            return tensor_input

        # ---- Endpoints is singular.
        if isinstance(endpoints, bittensor.Endpoint):
            formatted_endpoints = [endpoints]

        # ---- Endpoints is a list of Endpoints.
        elif isinstance(endpoints, list) and len(endpoints) > 0 and isinstance(endpoints[0], bittensor.Endpoint):
            formatted_endpoints = endpoints

        # ---- Endpoints is a torch tensor.
        elif isinstance(endpoints, torch.LongTensor):
            if len(endpoints.shape) == 1:
                formatted_endpoints = [bittensor.endpoint.from_tensor(endpoints)]
            elif len(endpoints.shape) == 2:
                formatted_endpoints = [bittensor.endpoint.from_tensor(row) for row in endpoints]
            else:
                error_msg = 'Endpoints tensor should have semantic shape [n, 250], got {}'.format(endpoints)
                raise ValueError(error_msg)

        # ---- Endpoints is a list of tensors.
        elif isinstance(endpoints, list) and len(endpoints) > 0 and isinstance(endpoints[0], torch.LongTensor):
            for tensor in endpoints:
                if len(tensor.shape) == 1:
                    formatted_endpoints.append(bittensor.endpoint.from_tensor(tensor))
                elif len(tensor.shape) == 2:
                    for row in tensor:
                        formatted_endpoints.append(bittensor.endpoint.from_tensor(row))
                else:
                    error_msg = 'Endpoints tensor should have semantic shape [n, 250], got {}'.format(tensor)
                    raise ValueError(error_msg)
        else:
            error_msg = """ Endpoints should have one of the following types.
                            - a single endpoint tensor shape [250]
                            - a set of endpoint tensors shape [n, 250]
                            - a list of endpoints tensors each of shape [250]
                            - a single endpoint object. Inputs will be sent to this endpoint alone.
                            - a list of endpoint objects. All inputs will be sent to these endpoints.
                        Got {} """.format(endpoints)
            raise ValueError(error_msg)

        # ---- Inputs is a string
        if isinstance(inputs, str):
            # Encode to tensors.
            tokenizer = bittensor.tokenizer()
            inputs_list = tokenizer(inputs)['input_ids']
            inputs_tensor = cast_and_check_tensor_input(torch.tensor([inputs_list], dtype=torch.int64))
            # Expand to length.
            formatted_inputs = [inputs_tensor for _ in formatted_endpoints]

        # ---- Inputs is a list of strings.
        elif isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], str):
            # Encode to tensors.
            tokenizer = bittensor.tokenizer()
            tokenized_sentences = tokenizer(inputs, padding = True, truncation=True)['input_ids']
            tokenizer_tensor = cast_and_check_tensor_input(torch.tensor(tokenized_sentences, dtype=torch.int64))
            formatted_inputs = [tokenizer_tensor for _ in formatted_endpoints]

        # ---- Inputs is a single tensor
        elif isinstance(inputs, torch.Tensor) and len(inputs.shape) <= 2:
            inputs = cast_and_check_tensor_input(inputs)
            # Expand to length.
            formatted_inputs = [inputs for _ in formatted_endpoints]

        # ---- Inputs is tensor with shape [n_endpoints, batch_size, sequence_len]
        elif isinstance(inputs, torch.Tensor) and len(inputs.shape) == 3 and inputs.shape[0] == len(
                formatted_endpoints):
            # Unbind inputs into list the same length as endpoints.
            formatted_inputs = [cast_and_check_tensor_input(input) for input in torch.unbind(inputs)]

        # ---- Inputs is a list of tensors
        elif isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
            formatted_inputs = [cast_and_check_tensor_input(input) for input in inputs]

        else:
            error_msg = """ Inputs should have one of the following types:
                            - a single string: the string will be tokenized using the bittensor tokenizer.
                            - a list of strings: the strings will be tokenized using the bittensor tokenizer.
                            - a tensor with shape [batch_size, sequence_len], assumed to be the output of bittensor tokenizer.
                            - a tensor with shape [n, batch_size, sequence_len], the operation will unbind the tensor and pass inputs to endpoints.
                        Got {} """.format(inputs)
            raise ValueError(error_msg)

        # ---- Check length.
        if len(formatted_inputs) != len(formatted_endpoints):
            error_msg = 'List of text inputs should have the same length as passed destination endpoints, got {} and {}'.format(
                len(inputs), len(endpoints))
            raise ValueError(error_msg)

        return formatted_endpoints, formatted_inputs

    def _init_stats(self):
        return SimpleNamespace(
            total_requests = 0,
            # queries on dendrite per second.
            qps = stat_utils.EventsPerSecondRollingAverage( 0, 0.01 ),
            # total bytes recieved by this dendrite per second.
            avg_in_bytes_per_second = stat_utils.AmountPerSecondRollingAverage( 0, 0.01 ),
            # total sent by this dendrite per second.
            avg_out_bytes_per_second = stat_utils.AmountPerSecondRollingAverage( 0, 0.01 ),
            # Codes recieved per pubkey.
            codes_per_pubkey = {},
            # Number of requests per pubkey.
            requests_per_pubkey = {},
            # Success rate per pubkey.
            successes_per_pubkey = {},
            # Query time per pubkey.
            query_times_per_pubkey = {},
            # Bytes recieved per pubkey.
            avg_in_bytes_per_pubkey = {},
            # Bytes sent per pubkey.
            avg_out_bytes_per_pubkey = {},
            # QPS per pubkey.
            qps_per_pubkey = {},
        )

    def update_stats(
            self, 
            endpoints: List[ 'bittensor.Endpoint'], 
            synapses: List[ 'bittensor.proto.Synapse' ], 
            inputs: List[torch.Tensor],
            outputs: List[ List[ torch.Tensor ] ],
            codes: List [ List[ torch.LongTensor ] ],
            times: List [ List[ torch.FloatTensor ] ]
        ):
        r""" Update dendrite stat according to the response we get from peers. Updates were saved to self.stats.
            Args:
                endpoints (:obj:`List[bittensor.Endpoint]` of shape :obj:`(num_endpoints)`, `required`):
                    The set of endpoints that dendrite sent request to.

                synapses (:obj:`List[ 'bittensor.Synapse' ]` of shape :obj:`(num_synapses)`, `required`):
                    Bittensor synapse objects with arguments. Each corresponds to a synapse function on the axon.
                    Responses are packed in this ordering. 

                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(n_endpoints)`, `required`):
                    List of torch tensors to be sent to the associated endpoints.

                outputs (:obj:`List[ List[ torch.FloatTensor ] ]` of shape :obj:`num_synapses * ( num_endpoints * ( -1, -1, -1 ) )`, `required`):
                    List of outputs from synapses, each a list of size num_endpoints of tensors with relevant size. Non-responses are zeroes of relevant 
                    synapse shape.

                codes (:obj:`List [ torch.LongTensor ]` of shape :obj:`[ num_endpoints ]`, `required`):
                    Return code per call per synapse.

                times (:obj:`List [ torch.FloatTensor ]` of shape :obj:`[ num_endpoints ]`, `required`):
                    Times per call per synapse.
        """
        self.stats.qps.event()
        self.stats.total_requests += 1
        total_in_bytes_per_second = 0
        self.stats.avg_out_bytes_per_second.event( float(sys.getsizeof(inputs)) )
        for (end_i, syn_i, inps_i, outs_i, codes_i, times_i) in list( zip ( endpoints, synapses, inputs, outputs, codes, times ) ):
            pubkey = end_i.hotkey
            # First time for this pubkey we create a new entry.
            if pubkey not in self.stats.requests_per_pubkey:
                self.stats.requests_per_pubkey[pubkey] = 0
                self.stats.successes_per_pubkey[pubkey] = 0
                self.stats.codes_per_pubkey[pubkey] = dict([(k,0) for k in bittensor.proto.ReturnCode.keys()])
                self.stats.query_times_per_pubkey[pubkey] = stat_utils.AmountPerSecondRollingAverage( 0, 0.01 )
                self.stats.avg_in_bytes_per_pubkey[pubkey] = stat_utils.AmountPerSecondRollingAverage( 0, 0.01 )
                self.stats.avg_out_bytes_per_pubkey[pubkey] = stat_utils.AmountPerSecondRollingAverage( 0, 0.01 )
                self.stats.qps_per_pubkey[pubkey] = stat_utils.EventsPerSecondRollingAverage( 0, 0.01 )

            self.stats.requests_per_pubkey[pubkey] += 1
            self.stats.successes_per_pubkey[pubkey] += (codes_i == 1).sum().int()
            self.stats.query_times_per_pubkey[pubkey].event( float( times_i.max() ) )
            self.stats.avg_in_bytes_per_pubkey[pubkey].event( float(sys.getsizeof( outs_i )) )
            self.stats.avg_out_bytes_per_pubkey[pubkey].event( float(sys.getsizeof( inps_i )) )
            self.stats.qps_per_pubkey[pubkey].event()
            total_in_bytes_per_second += sys.getsizeof(outs_i) if (codes_i == 1).sum().int() == len( synapses ) else 0 
            try:
                for code_i_s in codes_i:
                    if bittensor.proto.ReturnCode.Name(code_i_s) in self.stats.codes_per_pubkey[pubkey].keys():
                        self.stats.codes_per_pubkey[pubkey][bittensor.proto.ReturnCode.Name(code_i_s)] += 1
            except:
                # Code may be faulty.
                pass

        self.stats.avg_in_bytes_per_second.event( float( total_in_bytes_per_second ) )

    def to_dataframe ( self, metagraph ):
        r""" Return a stats info as a pandas dataframe indexed by the metagraph or pubkey if not existend.
            Args:
                metagraph: (bittensor.Metagraph):
                    Indexes the stats data using metagraph hotkeys.
            Return:
                dataframe (:obj:`pandas.Dataframe`)
        """
        try:
            index = [ metagraph.hotkeys.index(pubkey) for pubkey in self.stats.requests_per_pubkey.keys() if pubkey in metagraph.hotkeys]
            columns = [ 'dendrite_n_requested', 'dendrite_n_success', 'dendrite_query_time', 'dendrite_avg_inbytes', 'dendrite_avg_outbytes', 'dendrite_qps' ]
            dataframe = pandas.DataFrame(columns = columns, index = index)
            for pubkey in self.stats.requests_per_pubkey.keys():
                if pubkey in metagraph.hotkeys:
                    uid = metagraph.hotkeys.index(pubkey)
                    dataframe.loc[ uid ] = pandas.Series( {
                        'dendrite_n_requested': int(self.stats.requests_per_pubkey[pubkey]),
                        'dendrite_n_success': int(self.stats.successes_per_pubkey[pubkey]),
                        'dendrite_query_time': float(self.stats.query_times_per_pubkey[pubkey].get()),               
                        'dendrite_avg_inbytes': float(self.stats.avg_in_bytes_per_pubkey[pubkey].get()),
                        'dendrite_avg_outbytes': float(self.stats.avg_out_bytes_per_pubkey[pubkey].get()),
                        'dendrite_qps': float(self.stats.qps_per_pubkey[pubkey].get())
                    } )
            return dataframe

        except Exception as e:
            bittensor.logging.error( prefix='failed dendrite.to_dataframe()', sufix=str(e) )
            return pandas.DataFrame()


    def to_wandb( self ):
        r""" Return a dictionary of dendrite stats as wandb logging info.
            Args:
                metagraph: (bittensor.Metagraph):
                    If not None, indexes the wandb data using int uids rather than string pubkeys.
            Return:
                wandb_info (:obj:`Dict`)
        """
        try:
            wandb_info = {
                'dendrite/qps': self.stats.qps.get(),
                'dendrite/total_requests' : self.receptor_pool.get_total_requests(),
                'dendrite/avg_in_bytes_per_second' : self.stats.avg_in_bytes_per_second.get(),
                'dendrite/avg_out_bytes_per_second' : self.stats.avg_out_bytes_per_second.get(),
                'dendrite/Total unique queries': len(self.stats.requests_per_pubkey.keys()),
            }
            return wandb_info
        except Exception as e:
            bittensor.logging.error( prefix='failed dendrite.to_wandb()', sufix = str(e))
            return {}