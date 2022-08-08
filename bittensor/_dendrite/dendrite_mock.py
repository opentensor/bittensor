
""" Implementation of mock class dendrite, which quries endpoints with tensors.
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

# dummy tensor that triggers autograd 
DUMMY = torch.empty(0, requires_grad=True)


class DendriteMock(torch.autograd.Function):

    r""" Mocked Dendrite returns random results 50% of the time.
    """

    def __init__(
            self,
            config: 'bittensor.Config',
            wallet: 'bittensor.Wallet',
    ):
        r""" Initializes a new Mock Dendrite entry point.
        """
        self.config = config
        self.wallet = wallet
        self.stats = self._init_stats()

    def __str__(self):
        return "MockDendrite({})".format(self.wallet.hotkey.ss58_address)

    def __repr__(self):
        return self.__str__()

    def __del__(self):
        pass

    @staticmethod
    def forward(
            ctx,
            dendrite: 'bittensor.Dendrite',
            dummy: torch.Tensor,
            endpoints: List['bittensor.Endpoint'],
            modality: bittensor.proto.Modality,
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

                modality (:obj:`bittensor.proto.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality or type ENUM [TEXT, IMAGE, TENSOR]

                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(n_endpoints)`, `required`):
                    List of torch tensors to be sent to the associated endpoints.

                timeout (int):
                    request timeout.

                requires_grad (int, default = dendrite.requires_grad, `optional`):
                    If true, the backward pass triggers passing gradients on the wire.

            Returns:
                codes (:obj:`torch.LongTensor` of shape :obj:`(n_endpoints)` `required`):
                    Return code associated with forward call.

                times (:obj:`torch.FloatTensor` of shape :obj:`[ num_endpoints ]`, `required`):
                    times per call.
                
                outputs (:obj:`List[torch.FloatTensor]` of shape :obj:`n_endpoints * (batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Output encodings of inputs produced by the remote endpoints. Non-responses are zeroes of common shape.
        """
        ctx.endpoints, ctx.inputs, ctx.modality, ctx.timeout, ctx.does_requires_grad = endpoints, inputs, modality, timeout, requires_grad
        inputs = [x.cpu().clone().detach() for x in inputs]

        # MOCK response generator.        
        forward_outputs = []
        forward_codes = []
        forward_times = []
        for tensor in inputs:
            if random.random() < 0.5:
                forward_outputs.append( torch.randn( list(tensor.shape) + [bittensor.__network_dim__], dtype=torch.float32) )
                forward_codes.append( 1 )
                forward_times.append( 0 )
            else:
                forward_outputs.append( torch.zeros( list(tensor.shape) + [bittensor.__network_dim__], dtype=torch.float32) )
                forward_codes.append( 1 )
                forward_times.append( 0 )
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
            input_grads = []
            for idx, tensor in enumerate( ctx.inputs ):
                if ctx.forward_codes[idx]:
                    input_grads.append( torch.randn( list(tensor.shape) + [bittensor.__network_dim__], dtype=torch.float32) )
                else:
                    input_grads.append( torch.zeros( list(tensor.shape) + [bittensor.__network_dim__], dtype=torch.float32) )
            return (None, None, None, None, None, None, *input_grads)
        else:
            input_grads = [nill_response_for(inp) for inp in ctx.inputs]
            return (None, None, None, None, None, None, *input_grads)

    def _forward(
            self,
            endpoints: List['bittensor.Endpoint'],
            inputs: List[torch.Tensor],
            modality: bittensor.proto.Modality,
            timeout: int = None,
            requires_grad: bool = None
    ) -> Tuple[List[torch.Tensor], torch.LongTensor, torch.FloatTensor]:
        r""" Internal Forward tensor inputs to a list of neuron endpoints.

            Args:
                endpoints (:obj:`List[bittensor.Endpoint]` of shape :obj:`(num_endpoints)`, `required`):
                    List of remote endpoints which match length of inputs. Tensors from inputs are sent forward to these endpoints.

                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(num_endpoints * [shape])`, `required`):
                    List of tensors to send to corresponding endpoints. Tensors are of arbitrary type and shape depending on the
                    modality.

                modality (:obj:`bittensor.proto.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality type. Enum in [TEXT, IMAGE, TENSOR]

                timeout (int, default = dendrite.timeout, `required`):
                    request timeout.

                requires_grad (int, default = dendrite.requires_grad, `optional`):
                    If true, the backward pass triggers passing gradients on the wire.

            Returns:
                responses (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                    Output encodings of inputs produced by the remote endpoints. Non-responses are zeroes of common shape.

                codes (:obj:`List[torch.LongTensor]` of shape :obj:`[num_endpoints]`, `required`):
                    dendrite call return codes.

                times (:obj:`torch.FloatTensor` of shape :obj:`[ num_endpoints ]`, `required`):
                    times per call.

        """
        timeout = timeout if timeout is not None else self.config.dendrite.timeout
        requires_grad = requires_grad if requires_grad is not None else self.config.dendrite.requires_grad
        forward_response = DendriteMock.apply(
            self,
            DUMMY,
            endpoints,
            modality,
            timeout,
            requires_grad,
            *inputs
        )
        codes = forward_response[0]
        times = forward_response[1]
        responses = forward_response[2:]
        return responses, codes, times

    def forward_image(
            self,
            endpoints: Union[List['bittensor.Endpoint'], 'bittensor.Endpoint'],
            inputs: List[torch.FloatTensor],
            timeout: int = None,
            requires_grad: bool = None
    ) -> Tuple[Union[List[torch.FloatTensor], torch.FloatTensor], torch.LongTensor, torch.FloatTensor]:
        r""" Forward image inputs to endpoints.

          Args:
                endpoints (:obj:`Union[List[bittensor.Endpoint], bittensor.Endpoint]` of shape :obj:`(num_endpoints)`, `required`):
                    List or single of endpoints which match the length of inputs. Inputs are sent forward to these endpoints.

                inputs (:obj:`Union[List[torch.FloatTensor], torch.FloatTensor]` of shape :obj:`(num_endpoints * [ batch_size, sequence_len, channels, rows, cols ])`, `required`):
                    List or single of image-tensors to send to corresponding endpoints. Tensors are images encoded using the
                    torch.toTensor() or other encoding which produces the shape [batch_size, channels, rows, cols].

                timeout (int, default = dendrite.timeout `optional`):
                    Request timeout.

                requires_grad (int, default = dendrite.requires_grad, `optional`):
                    If true, the backward pass triggers passing gradients on the wire.

            Returns:
                responses (:obj:`Union[ List[torch.FloatTensor], torch.FloatTensor] ` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                    Output encodings of inputs produced by remote endpoints. Non-responses are zeroes of input shape plus output dimension.

                codes (:obj:`torch.LongTensor` of shape :obj:`[ num_endpoints ]`, `required`):
                    dendrite call return ops.

                times (:obj:`torch.FloatTensor` of shape :obj:`[ num_endpoints ]`, `required`):
                    times per call.
        """
        # Check types.
        if not isinstance(endpoints, list) and not isinstance(endpoints, Endpoint):
            raise ValueError('endpoints must be of type list or bittensor.Endpoint. Got {}'.format(type(endpoints)))

        if not isinstance(inputs, list) and not isinstance(inputs, torch.FloatTensor):
            raise ValueError(
                'inputs must be of type list[torch.FloatTensor] or torch.FloatTensor. Got {}'.format(type(inputs)))

        # Format to list.
        non_list_inputs = False
        if not isinstance(inputs, list):
            non_list_inputs = True
            inputs = [inputs]

        # Format to list.
        if not isinstance(endpoints, list):
            endpoints = [endpoints]

        # Catch inputs != List and endpoints == List
        elif non_list_inputs and isinstance(endpoints, list):
            raise ValueError(
                'endpoints and inputs must be of same type. Got endpoints {} and inputs {} '.format(type(endpoints),
                                                                                                    type(inputs[0])))

        # Check length.
        if len(inputs) < 1:
            raise ValueError('inputs list must have at least one element. Got len {}'.format(len(inputs)))
        if len(endpoints) < 1:
            raise ValueError('endpoints list must have at least one item. Got len {}'.format(len(endpoints)))
        if len(inputs) != len(endpoints):
            error_msg = 'List of tensor inputs should have the same length as passed destination endpoints, got {} and {}'.format(
                len(inputs), len(endpoints))
            raise ValueError(error_msg)

        # Check list types.
        if not isinstance(inputs[0], torch.FloatTensor):
            raise ValueError('inputs must be of type torch.FloatTensor. Got {}'.format(type(inputs[0])))
        if not isinstance(endpoints[0], Endpoint):
            raise ValueError('endpoints must be of type bittensor.Endpoint. Got {}'.format(type(endpoints)))

        # Check shape.
        if len(inputs[0].shape) != 5:
            error_msg = 'Image inputs should be rank 5 with semantic shape: [batch_size, sequence_len, channels, rows, cols], got {}'.format(
                inputs[0].shape)
            raise ValueError(error_msg)

        # Make calls.
        responses, codes, times = self._forward(
            endpoints=endpoints,
            inputs=inputs,
            modality=bittensor.proto.Modality.IMAGE,
            timeout=timeout,
            requires_grad=requires_grad
        )

        # Format to singletons.
        if non_list_inputs:
            responses = responses[0]

        # Return.
        self.update_stats( endpoints, inputs, responses, codes, times )
        return responses, codes, times

    def forward_tensor(
            self,
            endpoints: Union[List['bittensor.Endpoint'], 'bittensor.Endpoint'],
            inputs: List[torch.FloatTensor],
            timeout: int = None,
            requires_grad: bool = None
    ) -> Tuple[Union[List[torch.FloatTensor], torch.FloatTensor], torch.LongTensor, torch.FloatTensor]:
        r""" Forward tensor inputs to endpoints.

            Args:
                endpoints (:obj:`Union[List[bittensor.Endpoint], bittensor.Endpoint]` of shape :obj:`(num_endpoints)`, `required`):
                    List or single of endpoints which match the length of inputs. Inputs are sent forward to these endpoints.

                inputs (:obj:`Union[List[torch.LongTensor], torch.LongTensor]` of shape :obj:`(num_endpoints * [batch_size, sequence_len])`, `required`):
                    List or single tensors to send to corresponding endpoints. Tensors are of float type and
                    with shape [batch_size, sequence_len, bittensor.__network_dim__].

                timeout (int, default = dendrite.timeout `optional`):
                    Request timeout.

                requires_grad (int, default = dendrite.requires_grad, `optional`):
                    If true, the backward pass triggers passing gradients on the wire.

            Returns:
                responses (:obj:`Union[ List[torch.FloatTensor], torch.FloatTensor] ` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                    Output encodings of inputs produced by remote endpoints. Non-responses are zeroes of input shape plus output dimension.

                codes (:obj:`torch.LongTensor` of shape :obj:`[ num_endpoints ]`, `required`):
                    dendrite call return ops.

                times (:obj:`torch.FloatTensor` of shape :obj:`[ num_endpoints ]`, `required`):
                    times per call.
        """
        # Check types.
        if not isinstance(endpoints, list) and not isinstance(endpoints, Endpoint):
            raise ValueError('endpoints must be of type list or bittensor.Endpoint. Got {}'.format(type(endpoints)))

        if not isinstance(inputs, list) and not isinstance(inputs, torch.FloatTensor):
            raise ValueError(
                'inputs must be of type list[torch.FloatTensor] or torch.FloatTensor. Got {}'.format(type(inputs)))

        # Format to list.
        non_list_inputs = False
        if not isinstance(inputs, list):
            non_list_inputs = True
            inputs = [inputs]

        # Format to list.
        if not isinstance(endpoints, list):
            endpoints = [endpoints]

        # Catch inputs != List and endpoints == List
        elif non_list_inputs and isinstance(endpoints, list):
            raise ValueError(
                'endpoints and inputs must be of same type. Got endpoints {} and inputs {} '.format(type(endpoints),
                                                                                                    type(inputs[0])))

        # Check length.
        if len(inputs) < 1:
            raise ValueError('inputs list must have at least one element. Got len {}'.format(len(inputs)))
        if len(endpoints) < 1:
            raise ValueError('endpoints list must have at least one item. Got len {}'.format(len(endpoints)))
        if len(inputs) != len(endpoints):
            error_msg = 'List of tensor inputs should have the same length as passed destination endpoints, got {} and {}'.format(
                len(inputs), len(endpoints))
            raise ValueError(error_msg)

        # Check list types.
        if not isinstance(inputs[0], torch.FloatTensor):
            raise ValueError('inputs must be of type torch.FloatTensor. Got {}'.format(type(inputs[0])))
        if not isinstance(endpoints[0], Endpoint):
            raise ValueError('endpoints must be of type bittensor.Endpoint. Got {}'.format(type(endpoints)))

        # Check shape.
        if len(inputs[0].shape) != 3:
            error_msg = 'Tensor inputs should be rank 3 with semantic shape: [batch_size, sequence_len, bittensor.__network_dim__]'
            raise ValueError(error_msg)
        if inputs[0].shape[2] != bittensor.__network_dim__:
            error_msg = 'Passed tensor must have last dimension {} got {}'.format(bittensor.__network_dim__,
                                                                                  inputs[0].shape[2])
            raise ValueError(error_msg)

        # Make calls.
        responses, codes, times = self._forward(
            endpoints=endpoints,
            inputs=inputs,
            modality=bittensor.proto.Modality.TENSOR,
            timeout=timeout,
            requires_grad=requires_grad
        )

        # Format to singletons.
        if non_list_inputs:
            responses = responses[0]

        # Return.
        self.update_stats( endpoints, inputs, responses, codes, times )
        return responses, codes, times

    def forward_text(
            self,
            endpoints: Union[
                torch.LongTensor, List[torch.LongTensor], List['bittensor.Endpoint'], 'bittensor.Endpoint'],
            inputs: Union[str, List[str], List[torch.LongTensor], torch.LongTensor],
            timeout: int = None,
            requires_grad: bool = None
    ) -> Tuple[Union[List[torch.FloatTensor], torch.FloatTensor], torch.LongTensor, torch.FloatTensor]:
        r""" Forward text inputs to a list of neuron endpoints and block until responses or timeout.

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
                    responses (:obj:`torch.FloatTensor` of shape :obj:`(n, batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Output encodings of inputs produced by remote endpoints. Non-responses are zeroes of input shape plus output dimension.
                        The first dimension will match the number of endpoints queried.

                    codes (:obj:`torch.LongTensor` of shape :obj:`[ num_endpoints ]`, `required`):
                        dendrite call return ops.

                    times (:obj:`torch.FloatTensor` of shape :obj:`[ num_endpoints ]`, `required`):
                        times per call.
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
            tokenized_sentences = tokenizer(inputs, truncation=True)['input_ids']
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

        # Make calls.
        responses, codes, times = self._forward(
            endpoints=formatted_endpoints,
            inputs=formatted_inputs,
            modality=bittensor.proto.Modality.TEXT,
            timeout=timeout,
            requires_grad=requires_grad,
        )

        # Return.
        self.update_stats( formatted_endpoints, formatted_inputs, responses, codes, times )
        return responses, codes, times

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
                'dendrite/total_requests' : self.stats.total_requests,
                'dendrite/avg_in_bytes_per_second' : self.stats.avg_in_bytes_per_second.get(),
                'dendrite/avg_out_bytes_per_second' : self.stats.avg_out_bytes_per_second.get(),
                'dendrite/Total unique queries': len(self.stats.requests_per_pubkey.keys()),
            }
            return wandb_info
        except Exception as e:
            bittensor.logging.error( prefix='failed dendrite.to_wandb()', sufix = str(e))
            return {}