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

from torch.autograd.function import once_differentiable
from loguru import logger
from transformers.utils.logging import enable_explicit_format

import bittensor
from bittensor._endpoint.endpoint_impl import Endpoint
import bittensor.utils.stats as stat_utils
import bittensor.utils.codes as codes

import wandb

logger = logger.opt(colors=True)

# dummy tensor that triggers autograd 
DUMMY = torch.empty(0, requires_grad=True)


# Helper function for filling nill (zero) responses on failures.
def nill_response_for(inputs):
    """ Get zero matrix with the same size as inputs
    """
    if torch.numel(inputs) == 0:
        return torch.tensor([])
    return torch.zeros((inputs.size(0), inputs.size(1), bittensor.__network_dim__), dtype=torch.float32)

class Dendrite(torch.autograd.Function):
    r""" This is the implementation class for a bittensor.dendrite(). The dendrite class operates as a normal torch autograd friendly operation
    which accepts a list of bittensor.endpoints and a list of torch tensors. The passed endpoints are queried with the passed inputs and either return
    results or zeros. The operation is fully differentiable with a torch computation graph such that calls to loss.backward() produce Backward calls on
    the passed endpoints.

    Args:
        config (:obj:`bittensor.Config`, `optional`, defaults to bittensor.dendrite.config()):
            config namespace object created by calling bittensor.dendrite.config()
        wallet (:obj:`bittensor.Wallet`, `optional`, defaults to bittensor.wallet( name = 'default', hotkey = 'default')):
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
            synapses: List[ 'bittensor.proto.Synapse' ],
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

                synapses (:obj:`List[ bittensor.proto.Synapse ]` of shape :obj:`(num_synapses)`, `required`):
                    Protos specifiying the synapses to call, or synapse types with args. Each corresponds to a synapse function on the axon and args.
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
        inputs = [x.cpu().clone().detach() for x in inputs]
        forward_outputs, forward_codes, forward_times = ctx.receptor_pool.forward (
            endpoints = endpoints,
            synapses = synapses,
            inputs = inputs,
            timeout = timeout,
        )
        ctx.forward_codes = forward_codes
        forward_times = [-1 if t is None else t for t in forward_times]
        return (torch.tensor(forward_codes, dtype=torch.int64), 
                torch.tensor(forward_times, dtype=torch.float32),
                *forward_outputs)

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
            
            Returns:
                DUMMY, None, None, None,
                outputs (:obj:`List[torch.FloatTensor], `optional`):
                    Gradient results for each input.

        """
        if ctx.does_requires_grad:
            grads_cpu = [x.cpu().clone().detach() for x in output_grads]
            input_grads, _, _ = ctx.receptor_pool.backward(
                endpoints=ctx.endpoints,
                inputs_x=ctx.inputs,
                grads_dy=grads_cpu,
                modality=ctx.modality,
                timeout=ctx.timeout,
            )
            return (None, None, None, None, None, None, *input_grads)
        else:
            input_grads = [nill_response_for(inp) for inp in ctx.inputs]
            return (None, None, None, None, None, None, *input_grads)

    def _forward(
            self,
            endpoints: List [ 'bittensor.Endpoint' ],
            synapses: List[ 'bittensor.proto.Synapse' ],
            inputs: List [ torch.Tensor ],
            timeout: Optional [ int ]  = None,
            requires_grad: Optional [ bool ] = None,
    ) -> Tuple [ List[ torch.Tensor ], List[ torch.LongTensor ], List [ torch.FloatTensor ]]:
        r""" Internal Forward tensor inputs to a list of neuron endpoints.

            Args:
                endpoints (:obj:`List[bittensor.Endpoint]` of shape :obj:`(num_endpoints)`, `required`):
                    List of remote endpoints which match length of inputs. Tensors from inputs are sent forward to these endpoints.

                synapses (:obj:`List[ bittensor.proto.Synapse ]` of shape :obj:`(num_synapses)`, `required`):
                    Protos specifiying the synapses to call, or synapse types with args. Each corresponds to a synapse function on the axon and args.
                    Responses are packed in this ordering. 

                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(num_endpoints * [shape])`, `required`):
                    List of tensors to send to corresponding endpoints. Tensors are of arbitrary type and shape depending on the
                    modality.

                timeout (int, default = dendrite.timeout, `optional`):
                    request timeout.

                requires_grad (int, default = dendrite.requires_grad, `optional`):
                    If true, the backward pass triggers passing gradients on the wire.

            Returns:
                responses (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                    Output encodings of inputs produced by the remote endpoints. Non-responses are zeroes of common shape.

                codes (:obj:`List[torch.LongTensor]` of shape :obj:`[num_endpoints]`, `required`):
                    Return codes per endpoint per synapse.

                times (:obj:`torch.FloatTensor` of shape :obj:`[ num_endpoints ]`, `required`):
                    Call times per endpoint per synapse.

        """
        timeout = timeout if timeout is not None else self.config.dendrite.timeout
        requires_grad = requires_grad if requires_grad is not None else self.config.dendrite.requires_grad
        forward_response = Dendrite.apply (
            self,
            DUMMY,
            endpoints,
            synapses,
            timeout,
            requires_grad,
            *inputs
        )
        codes = forward_response[0]
        times = forward_response[1]
        responses = forward_response[2:]
        return responses, codes, times

    def text (
        self,
        endpoints: Union[ torch.LongTensor, List[torch.LongTensor], List['bittensor.Endpoint'], 'bittensor.Endpoint' ],
        synapses: Union [ List[ 'bittensor.proto.Synapse' ], List[ Tuple[ 'bittensor.proto.Synapse.Type', dict ] ]],
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

                    synapses (:obj:`Union[ List[ bittensor.proto.Synapse ], List[ Tuple[ bittensor.proto.Synapse.Type, dict ] ]]` of shape :obj:`(num_synapses)`, `required`):
                        Protos specifiying the synapses to call, or synapse types with args. Each corresponds to a synapse function on the axon and args.
                        Responses are packed in this ordering. 
                            - a list of synapse protos
                            - a list of tuples containing a synapse type and a dictionary of synapse arguments.

                    inputs (:obj:`Union[str,  List[str], List[torch.LongTensor], torch.LongTensor]` of shape :obj:`(num_endpoints * [batch_size, sequence_len])`, `required`):
                        Tokenized sentences to send on the wire. Inputs can be one of the following types:
                            - a single string: the string will be tokenized using the bittensor tokenizer.
                            - a list of strings: the strings will be tokenized using the bittensor tokenizer.
                            - a tensor with shape [batch_size, sequence_len], assumed to be the output of bittensor tokenizer.
                            - a tensor with shape [n, batch_size, sequence_len], the operation will unbind the tensor and pass inputs to endpoints.
                            - a list of tensors of type long each representing a tokenized sentence to be sent to each endpoint.
                        If inputs are tensors they will be cast to int64 format before sending on the wire.

                    synapses (:obj:`List[ Union[ int, bittensor.proto.Synapse.Type] ]` of shape :obj:`(num_synapses)`, `required`):
                        List of protos specifiying the synapses to call, each type corresponds to a synapse function on the axon.
                        responses are packed in this ordering. 

                    synapses_args (:obj:`List[ dict ]` of shape :obj:`(num_synapses)`, `required`):
                        List of dictionaries carrying synapse arguments. to be passed through to
                        the specific synapse functions on the axon side.

                    timeout (:type:`int`, default = dendrite.timeout `optional`):
                        Request timeout. Queries that do not respond will be replaced by zeros.

                    requires_grad (:type:`int`, default = dendrite.requires_grad, `optional`):
                        If true, the backward pass triggers passing gradients on the wire.

                Returns:
                    responses (:obj:`List[ List[ torch.FloatTensor ] ]` of shape :obj:`num_synapses * ( num_endpoints * ( -1, -1, -1 ) )`, `required`):
                        List of outputs from synapses, each a list of size num_endpoints of tensors with relevant size. Non-responses are zeroes of relevant 
                        synapse shape.

                    codes (:obj:`List [ torch.LongTensor ]` of shape :obj:`[ num_endpoints ]`, `required`):
                        Return code per call per synapse.

                    times (:obj:`List [ torch.FloatTensor ]` of shape :obj:`[ num_endpoints ]`, `required`):
                        Times per call per synapse.
            """
        formatted_endpoints, formatted_inputs = self.__format_text_inputs ( 
            endpoints = endpoints, 
            inputs = inputs
        )
        synapses = bittensor.Synapse_Serializer.format_synapses( synapses )
        responses, codes, times = self._forward (
            endpoints = formatted_endpoints,
            synapses = synapses,
            inputs = formatted_inputs,
            timeout = timeout,
            requires_grad = requires_grad,
        )
        # Return.
        self.update_stats( formatted_endpoints, formatted_inputs, responses, codes, times )
        return responses, codes, times

    def text_causal_lm(
        self,
        endpoints: Union[ torch.LongTensor, List[torch.LongTensor], List['bittensor.Endpoint'], 'bittensor.Endpoint' ],
        inputs: Union[str, List[str], List[torch.LongTensor], torch.LongTensor],
        topk_encoding_length: Optional[int] = 50,
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

                    topk_encoding_length (:type:`int`, default = dendrite.causal_lm.topk_encoding_length, `optional`):
                        Number of top logits to return from remote endpoints, the remaining logits values are 
                        set to zero.

                    timeout (:type:`int`, default = dendrite.timeout `optional`):
                        Request timeout. Queries that do not respond will be replaced by zeros.

                    requires_grad (:type:`int`, default = dendrite.requires_grad, `optional`):
                        If true, the backward pass triggers passing gradients on the wire.

                Returns:
                    responses (:obj:`List[ torch.FloatTensor ]` of shape :obj:`num_endpoints * (-1, sequence_len, bittensor.__vocab_size__ )`, `required`):
                        List of output logit encodings of inputs produced by each remote endpoints. Non-responses are zeroes of input shape plus output dimension.
                        The first dimension will match the number of endpoints queried.

                    codes (:obj:`torch.LongTensor` of shape :obj:`[ num_endpoints ]`, `required`):
                        dendrite call return ops.

                    times (:obj:`torch.FloatTensor` of shape :obj:`[ num_endpoints ]`, `required`):
                        times per call.
        """
        # Format inputs.
        formatted_endpoints, formatted_inputs = self.__format_text_inputs ( 
            endpoints = endpoints, 
            inputs = inputs
        )
        # Optionally convert synapses and set typing info.
        synapses = bittensor.Synapse_Serializer.format_synapses( [ ( bittensor.proto.Synapse.Type.TEXT_CAUSAL_LM , { "topk_encoding_length": topk_encoding_length, "synapse_type" : bittensor.proto.Synapse.Type.TEXT_CAUSAL_LM } ) ] )
        # Make calls.
        responses, codes, times = self._forward (
            endpoints = formatted_endpoints,
            synapses = synapses,
            inputs = formatted_inputs,
            timeout = timeout,
            requires_grad = requires_grad,
        )
        # Return.
        self.update_stats( formatted_endpoints, formatted_inputs, responses, codes, times )
        return responses[0], codes[0], times[0]


    def text_last_hidden_state( 
            self,
            endpoints: Union[ torch.LongTensor, List[torch.LongTensor], List['bittensor.Endpoint'], 'bittensor.Endpoint' ],
            inputs: Union[str, List[str], List[torch.LongTensor], torch.LongTensor],
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

                    timeout (:type:`int`, default = dendrite.timeout `optional`):
                        Request timeout. Queries that do not respond will be replaced by zeros.

                    requires_grad (:type:`int`, default = dendrite.requires_grad, `optional`):
                        If true, the backward pass triggers passing gradients on the wire.

                Returns:
                    responses (:obj:`List [ torch.FloatTensor ]` of shape :obj:` num_endpoints * ( -1, sequence_len, bittensor.__network_dim__ )`, `required`):
                        List of output last hidden state encodings of inputs produced by remote endpoints. Non-responses are zeroes of input shape plus output dimension.
                        The first dimension will match the number of endpoints queried.

                    codes (:obj:`torch.LongTensor` of shape :obj:`[ num_endpoints ]`, `required`):
                        dendrite call return ops.

                    times (:obj:`torch.FloatTensor` of shape :obj:`[ num_endpoints ]`, `required`):
                        times per call.
        """
        # Format inputs.
        formatted_endpoints, formatted_inputs = self.__format_text_inputs ( 
            endpoints = endpoints, 
            inputs = inputs
        )
        # Optionally convert synapses and set typing info.
        synapses = bittensor.Synapse_Serializer.format_synapses( [(bittensor.proto.Synapse.Type.TEXT_LAST_HIDDEN_STATE , { "synapse_type" : bittensor.proto.Synapse.Type.TEXT_LAST_HIDDEN_STATE })] )
        # Make calls.
        responses, codes, times = self._forward (
            endpoints = formatted_endpoints,
            synapses = synapses,
            inputs = formatted_inputs,
            modality = bittensor.proto.Modality.TEXT,
            timeout = timeout,
            requires_grad = requires_grad,
        )
        # Return.
        self.update_stats( formatted_endpoints, formatted_inputs, responses, codes, times )
        return responses[0], codes[0], times[0]


    def __format_text_inputs (
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
            tokenized_sentences = tokenizer(inputs, padding=True, truncation=True)['input_ids']
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

    def update_stats(self, endpoints, requests, responses, return_ops, query_times):
        r""" Update dendrite stat according to the response we get from peers. Updates were saved to self.stats.
            Args:
                endpoints (:obj:`List[bittensor.Endpoint]` of shape :obj:`(num_endpoints)`, `required`):
                    The set of endpoints that dendrite sent request to.

                requests (List[torch.Tensor] of shape :obj:`[ num_endpoints ]`, `required`):
                    Requests from the call.

                responses (List[torch.FloatTensor] of shape :obj:`[ num_endpoints ]`, `required`):
                    Responses from the call.

                return_ops (:obj:`torch.LongTensor` of shape :obj:`[ num_endpoints ]`, `required`):
                    Dendrite call return ops.

                query_times (:obj:`torch.FloatTensor` of shape :obj:`[ num_endpoints ]`, `required`):
                    Times per call.
        """
        self.stats.qps.event()
        self.stats.total_requests += 1
        total_in_bytes_per_second = 0
        self.stats.avg_out_bytes_per_second.event( float(sys.getsizeof(requests)) )
        for (e_i, req_i, resp_i, code_i, time_i) in list(zip(endpoints, requests, responses, return_ops.tolist(), query_times.tolist())):
            pubkey = e_i.hotkey

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
            self.stats.successes_per_pubkey[pubkey] += 1 if code_i == 1 else 0
            self.stats.query_times_per_pubkey[pubkey].event( float(time_i) )
            self.stats.avg_in_bytes_per_pubkey[pubkey].event( float(sys.getsizeof(resp_i)) )
            self.stats.avg_out_bytes_per_pubkey[pubkey].event( float(sys.getsizeof(req_i)) )
            self.stats.qps_per_pubkey[pubkey].event()
            total_in_bytes_per_second += sys.getsizeof(resp_i) if code_i == 1 else 0 
            try:
                if bittensor.proto.ReturnCode.Name(code_i) in self.stats.codes_per_pubkey[pubkey].keys():
                    self.stats.codes_per_pubkey[pubkey][bittensor.proto.ReturnCode.Name(code_i)] += 1
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


    def forward_text_seq2seq(
            self,
            endpoints,
            inputs,
            synapses_args,
            timeout: int = None,
            requires_grad: bool = None,
            ):

        self.forward_text(
                    endpoints= endpoints,
                    inputs = inputs,
                    syanpse = [bittensor.proto.SynapseType.TEXT_SEQ_2_SEQ],
                    synapse_args = synapse_args,
                    timeout=timeout,
                    requires_grad = requires_grad
                    )

    
    def forward_text_causal_LM(
            self,
            endpoints,
            inputs,
            synapses_args,
            timeout: int = None,
            requires_grad: bool = None,
            ):

        self.forward_text(
                    endpoints= endpoints,
                    inputs = inputs,
                    syanpse = [bittensor.proto.SynapseType.TEXT_CAUSAL_LM],
                    synapse_args = synapse_args,
                    timeout=timeout,
                    requires_grad = requires_grad
                    )