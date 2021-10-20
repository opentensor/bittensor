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

import torch
from torch.autograd.function import once_differentiable
from loguru import logger

import bittensor
from bittensor._endpoint.endpoint_impl import Endpoint
import bittensor.utils.stats as stat_utils

logger = logger.opt(colors=True)

# dummy tensor that triggers autograd 
DUMMY = torch.empty(0, requires_grad=True)

# Helper function for filling nill (zero) responses on failures.
def nill_response_for(inputs):
    """ Get zero matrix with the same size as inputs
    """
    if torch.numel(inputs) == 0:
        return torch.tensor([])
    return torch.zeros( (inputs.size(0), inputs.size(1), bittensor.__network_dim__), dtype=torch.float32)

class Dendrite( torch.autograd.Function ):
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
        ):
        r""" Initializes a new Dendrite entry point.
            Args:
                receptor_pool (:obj:`bittensor.ReceptorPool`, `required`):
                    bittensor receptor pool
        """
        super().__init__()
        self.config = config
        self.wallet = wallet
        self.receptor_pool = receptor_pool

        # ---- Dendrite stats
        # num of time we have sent request to a peer, received successful respond, and the respond time
        self.stats = SimpleNamespace(
            requested_peers_count = {},
            responded_peers_count = {},
            peers_respond_time = {}
        )

    def __str__(self):
        return "Dendrite({}, {})".format(self.wallet.hotkey.ss58_address, self.receptor_pool)

    def __repr__(self):
        return self.__str__()

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
        ) -> Tuple[ torch.Tensor, ... ] :
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
        ctx.receptor_pool = dendrite.receptor_pool
        ctx.endpoints, ctx.inputs, ctx.modality, ctx.timeout, ctx.does_requires_grad = endpoints, inputs, modality, timeout, requires_grad
        inputs = [ x.cpu().clone().detach() for x in inputs ]
        forward_outputs, forward_codes, forward_times = ctx.receptor_pool.forward(
            endpoints = endpoints, 
            inputs = inputs, 
            modality = modality,
            timeout = timeout
        )
        ctx.forward_codes = forward_codes
        return (torch.tensor(forward_codes, dtype=torch.int64), torch.tensor(forward_times, dtype=torch.float32), *forward_outputs)

    @staticmethod
    @once_differentiable
    def backward( 
            ctx, 
            unused_code_grads: torch.FloatTensor,
            unused_time_grads: torch.FloatTensor,
            *output_grads: torch.FloatTensor
        ) -> Tuple[ Optional[torch.Tensor], ... ]:
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
            grads_cpu = [ x.cpu().clone().detach() for x in output_grads ]
            input_grads, _, _ =  ctx.receptor_pool.backward (
                endpoints = ctx.endpoints, 
                inputs_x = ctx.inputs, 
                grads_dy = grads_cpu, 
                modality = ctx.modality,
                timeout = ctx.timeout,
            )
            return (None, None, None, None, None, None, *input_grads)
        else:
            input_grads = [ nill_response_for( inp ) for inp in ctx.inputs ]
            return (None, None, None, None, None, None, *input_grads)


    def _forward(
                self,
                endpoints: List['bittensor.Endpoint'],
                inputs: List[torch.Tensor],
                modality: bittensor.proto.Modality,
                timeout: int = None,
                requires_grad: bool = None
            ) -> Tuple[torch.LongTensor, List[torch.Tensor]]:
        r""" Internal Forward tensor inputs to a list of neuron endpoints.

            Args:
                endpoints (:obj:`List[bittensor.Endpoint]` of shape :obj:`(num_endpoints)`, `required`):
                    List of remote endpoints which match length of inputs. Tensors from inputs are sent forward to these endpoints.

                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(num_endpoints * [shape])`, `required`):
                    List of tensors to send to corresponsing endpoints. Tensors are of arbitrary type and shape depending on the
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
        timeout = timeout if timeout != None else self.config.dendrite.timeout 
        requires_grad = requires_grad if requires_grad != None else self.config.dendrite.requires_grad 
        forward_response = Dendrite.apply(
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
        tensors = forward_response[2:]
        return tensors, codes, times

    def forward_image(
        self, 
        endpoints: Union[ List['bittensor.Endpoint'], 'bittensor.Endpoint'],
        inputs: List[ torch.FloatTensor ],
        timeout: int = None,
        requires_grad: bool = None
    ) -> Tuple[ Union[List[torch.FloatTensor], torch.FloatTensor], torch.LongTensor, torch.FloatTensor]:
        r""" Forward image inputs to endpoints.

          Args:
                endpoints (:obj:`Union[List[bittensor.Endpoint], bittensor.Endpoint]` of shape :obj:`(num_endpoints)`, `required`):
                    List or single of endpoints which match the length of inputs. Inputs are sent forward to these endpoints.

                inputs (:obj:`Union[List[torch.FloatTensor], torch.FloatTensor]` of shape :obj:`(num_endpoints * [ batch_size, sequence_len, channels, rows, cols ])`, `required`):
                    List or single of image-tensors to send to corresponsing endpoints. Tensors are images encoded using the
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
            raise ValueError('inputs must be of type list[torch.FloatTensor] or torch.FloatTensor. Got {}'.format(type(inputs)))
        
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
            raise ValueError('endpoints and inputs must be of same type. Got endpoints {} and inputs {} '.format( type(endpoints), type(inputs[0]) ))

        # Check length.
        if len(inputs) < 1:
            raise ValueError('inputs list must have atleast one element. Got len {}'.format(len(inputs)))
        if len(endpoints) < 1:
            raise ValueError('endpoints list must have atleast one item. Got len {}'.format(len(endpoints)))
        if len( inputs ) != len( endpoints ):
            error_msg = 'List of tensor inputs should have the same length as passed destination endpoints, got {} and {}'.format(len( inputs ), len( endpoints ))
            raise ValueError(error_msg)
            
        # Check list types.
        if not isinstance(inputs[0], torch.FloatTensor):
            raise ValueError('inputs must be of type torch.FloatTensor. Got {}'.format(type(inputs[0])))
        if not isinstance(endpoints[0], Endpoint):
            raise ValueError('endpoints must be of type bittensor.Endpoint. Got {}'.format(type(endpoints)))

        # Check shape.
        if len( inputs[0].shape ) != 5:
            error_msg = 'Image inputs should be rank 5 with semantic shape: [batch_size, sequence_len, channels, rows, cols], got {}'.format( inputs[0].shape )
            raise ValueError(error_msg)
        
        # Make calls.
        responses, codes, times = self._forward(
            endpoints = endpoints, 
            inputs = inputs, 
            modality = bittensor.proto.Modality.IMAGE,
            timeout = timeout,
            requires_grad = requires_grad
        )

        # Format to singletons.
        if non_list_inputs:
            responses = responses[0]

        # Return.
        self.update_stat(endpoints, codes, times)
        return responses, codes, times

    def forward_tensor(
            self, 
            endpoints: Union[ List['bittensor.Endpoint'], 'bittensor.Endpoint'] ,
            inputs: List[ torch.FloatTensor ],
            timeout: int = None,
            requires_grad: bool = None
        ) -> Tuple[ Union[List[torch.FloatTensor], torch.FloatTensor], torch.LongTensor, torch.FloatTensor]:
        r""" Forward tensor inputs to endpoints.

            Args:
                endpoints (:obj:`Union[List[bittensor.Endpoint], bittensor.Endpoint]` of shape :obj:`(num_endpoints)`, `required`):
                    List or single of endpoints which match the length of inputs. Inputs are sent forward to these endpoints.

                inputs (:obj:`Union[List[torch.LongTensor], torch.LongTensor]` of shape :obj:`(num_endpoints * [batch_size, sequence_len])`, `required`):
                    List or single tensors to send to corresponsing endpoints. Tensors are of float type and
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
            raise ValueError('inputs must be of type list[torch.FloatTensor] or torch.FloatTensor. Got {}'.format(type(inputs)))
        
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
            raise ValueError('endpoints and inputs must be of same type. Got endpoints {} and inputs {} '.format( type(endpoints), type(inputs[0]) ))

        # Check length.
        if len(inputs) < 1:
            raise ValueError('inputs list must have atleast one element. Got len {}'.format(len(inputs)))
        if len(endpoints) < 1:
            raise ValueError('endpoints list must have atleast one item. Got len {}'.format(len(endpoints)))
        if len( inputs ) != len( endpoints ):
            error_msg = 'List of tensor inputs should have the same length as passed destination endpoints, got {} and {}'.format(len( inputs ), len( endpoints ))
            raise ValueError(error_msg)
            
        # Check list types.
        if not isinstance(inputs[0], torch.FloatTensor):
            raise ValueError('inputs must be of type torch.FloatTensor. Got {}'.format(type(inputs[0])))
        if not isinstance(endpoints[0], Endpoint):
            raise ValueError('endpoints must be of type bittensor.Endpoint. Got {}'.format(type(endpoints)))

        # Check shape.
        if len( inputs[0].shape ) != 3:
            error_msg = 'Tensor inputs should be rank 3 with semantic shape: [batch_size, sequence_len, bittensor.__network_dim__]'
            raise ValueError(error_msg)
        if inputs[0].shape[2] != bittensor.__network_dim__:
            error_msg = 'Passed tensor must have last dimension {} got {}'.format(bittensor.__network_dim__, inputs[0].shape[2])
            raise ValueError(error_msg)

        # Make calls.
        responses, codes, times = self._forward(
            endpoints = endpoints, 
            inputs = inputs, 
            modality = bittensor.proto.Modality.TENSOR,
            timeout = timeout,
            requires_grad = requires_grad
        )

        # Format to singletons.
        if non_list_inputs:
            responses = responses[0]

        # Return.
        self.update_stat(endpoints, codes, times)
        return responses, codes, times

    def forward_text(
            self,
            endpoints: Union[torch.LongTensor, List[torch.LongTensor], List['bittensor.Endpoint'], 'bittensor.Endpoint'],
            inputs: Union[str, List[str], List[torch.LongTensor], torch.LongTensor],
            timeout: int = None,
            requires_grad: bool = None
        ) -> Tuple[ Union[List[torch.FloatTensor], torch.FloatTensor], torch.LongTensor, torch.FloatTensor]:
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
        def cast_and_check_tensor_input( tensor_input ) -> torch.LongTensor:
            if not isinstance ( tensor_input, torch.LongTensor):
                try:
                    tensor_input = tensor_input.to( torch.int64 )
                except Exception as E:
                    error_msg = 'Error while casting tensor input {} to int64 {}'.format(tensor_input, E)
                    raise ValueError(error_msg) from ValueError()
            if not ( isinstance(tensor_input, torch.cuda.LongTensor) or isinstance(tensor_input, torch.LongTensor)) :
                raise ValueError('input {} must be of type torch.LongTensor. Got {}'.format(tensor_input, type(tensor_input)))
            # Expand shape if it is a singlular dimension.
            if len( tensor_input.shape ) == 1:
                tensor_input = tensor_input.view(1, -1)

            # Check shape.
            if len( tensor_input.shape ) != 2:
                error_msg = 'Text inputs should be rank 2 with semantic shape: [batch_size, sequence_len]'
                raise ValueError(error_msg)
            return tensor_input 

        # ---- Endpoints is singular.
        if isinstance( endpoints, bittensor.Endpoint ):
            formatted_endpoints = [endpoints]

        # ---- Endpoints is a list of Endpoints.
        elif isinstance( endpoints, list ) and len( endpoints ) > 0 and isinstance( endpoints[0], bittensor.Endpoint ):
            formatted_endpoints = endpoints

        # ---- Endpoints is a torch tensor.
        elif isinstance( endpoints, torch.LongTensor ):
            if len(endpoints.shape) == 1:
                formatted_endpoints = [ bittensor.endpoint.from_tensor( endpoints ) ]
            elif len(endpoints.shape) == 2:
                formatted_endpoints = [ bittensor.endpoint.from_tensor( row ) for row in endpoints ]
            else:
                error_msg = 'Endpoints tensor should have semantic shape [n, 250], got {}'.format( endpoints )
                raise ValueError(error_msg)

        # ---- Endpoints is a list of tensors.
        elif isinstance( endpoints, list ) and len( endpoints ) > 0 and isinstance( endpoints[0], torch.LongTensor ):
            for tensor in endpoints:
                if len(tensor.shape) == 1:
                    formatted_endpoints.append( bittensor.endpoint.from_tensor( tensor ) )
                elif len(tensor.shape) == 2:
                    for row in tensor:
                        formatted_endpoints.append( bittensor.endpoint.from_tensor( row ) )
                else:
                    error_msg = 'Endpoints tensor should have semantic shape [n, 250], got {}'.format( tensor )
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
        if isinstance( inputs, str ):
            # Encode to tensors.
            tokenizer = bittensor.tokenizer()
            inputs_list = tokenizer( inputs )['input_ids']
            inputs_tensor = cast_and_check_tensor_input ( torch.tensor( [inputs_list], dtype=torch.int64 ) )
            # Expand to length.
            formatted_inputs = [ inputs_tensor for _ in formatted_endpoints ]

        # ---- Inputs is a list of strings.
        elif isinstance ( inputs, list ) and len( inputs ) > 0 and isinstance( inputs[0], str ):
            # Encode to tensors.
            tokenizer = bittensor.tokenizer()
            tokenized_sentences = tokenizer( inputs, padding=True, truncation=True)['input_ids']
            tokenizer_tensor = cast_and_check_tensor_input( torch.tensor( tokenized_sentences, dtype=torch.int64 ) )
            formatted_inputs = [ tokenizer_tensor for _ in formatted_endpoints ]

        # ---- Inputs is a single tensor
        elif isinstance ( inputs, torch.Tensor ) and len(inputs.shape) <= 2:
            inputs = cast_and_check_tensor_input( inputs )
            # Expand to length.
            formatted_inputs = [ inputs for _ in formatted_endpoints]

        # ---- Inputs is tensor with shape [n_endpoints, batch_size, sequence_len]
        elif isinstance ( inputs, torch.Tensor ) and len( inputs.shape ) == 3 and inputs.shape[0] == len( formatted_endpoints ):
            # Unbind inputs into list the same length as endpoints.
            formatted_inputs = [ cast_and_check_tensor_input(input) for input in torch.unbind( inputs ) ]

        # ---- Inputs is a list of tensors
        elif isinstance ( inputs, list ) and len( inputs ) > 0 and isinstance( inputs[0], torch.Tensor ):
            formatted_inputs = [ cast_and_check_tensor_input(input) for input in inputs ]

        else:
            error_msg = """ Inputs should have one of the following types:
                            - a single string: the string will be tokenized using the bittensor tokenizer.
                            - a list of strings: the strings will be tokenized using the bittensor tokenizer.
                            - a tensor with shape [batch_size, sequence_len], assumed to be the output of bittensor tokenizer.
                            - a tensor with shape [n, batch_size, sequence_len], the operation will unbind the tensor and pass inputs to endpoints.
                        Got {} """.format(inputs)
            raise ValueError(error_msg)
            
        # ---- Check length.
        if len( formatted_inputs ) != len( formatted_endpoints ):
            error_msg = 'List of text inputs should have the same length as passed destination endpoints, got {} and {}'.format(len( inputs ), len( endpoints ))
            raise ValueError(error_msg)

        # Make calls.
        responses, codes, times = self._forward(
            endpoints = formatted_endpoints, 
            inputs = formatted_inputs, 
            modality = bittensor.proto.Modality.TEXT,
            timeout = timeout,
            requires_grad = requires_grad,
        )

        # Return.
        self.update_stat(formatted_endpoints, codes, times)
        return responses, codes, times

    def update_stat(self, endpoints, return_ops, query_times):
        r""" Update dendrite stat according to the response we get from peers.
        Updates were saved to self.stats.
            Args:
                endpoints (:obj:`List[bittensor.Endpoint]` of shape :obj:`(num_endpoints)`, `required`):
                    The set of endpoints that dendrite sent request to.

                return_ops (:obj:`torch.LongTensor` of shape :obj:`[ num_endpoints ]`, `required`):
                    Dendrite call return ops.

                query_times (:obj:`torch.FloatTensor` of shape :obj:`[ num_endpoints ]`, `required`):
                    Times per call.
        """
        # ---- uids that we have sent request to.
        uids = torch.tensor([e.uid for e in endpoints])

        # ---- uids that gave us successful respond.
        success_ids= torch.where( return_ops == bittensor.proto.ReturnCode.Success )[0]
        
        # ---- For each uid, check we have a stats column for this peer and aggregate to stats.
        for uid, time in zip(uids, query_times):
            if uid in self.stats.requested_peers_count.keys():
                self.stats.requested_peers_count[uid].update(1)
                self.stats.peers_respond_time[uid].update(time)

            else:
                self.stats.requested_peers_count[uid] = stat_utils.timed_rolling_avg(1, 0.01)
                self.stats.responded_peers_count[uid] = stat_utils.timed_rolling_avg(1, 0.01)
                self.stats.peers_respond_time[uid] = stat_utils.timed_rolling_avg(time, 0.01)

        
        # --- Aggregating result to stats 
        for uid in uids[success_ids]:
            if uid in self.stats.requested_peers_count.keys():
                self.stats.responded_peers_count[uid].update(1)
            else:
                self.stats.responded_peers_count[uid] = stat_utils.timed_rolling_avg(1, 0.01)

    def to_wandb(self):
        r""" Return a dictionary of axon stat for wandb logging
            
            Return:
                wandb_info (:obj:`Dict`)
        """
        wandb_info = {}

        # ---- Dendrite stats per pubkey for wandb 
        for uid in self.stats.requested_peers_count.keys():
            respond_rate = self.stats.responded_peers_count[uid].value / self.stats.requested_peers_count[uid].value
           
            uid_str = str(uid.item()).zfill(3)
            wandb_info[f'dend_quested uid: {uid_str}']= self.stats.requested_peers_count[uid].value
            wandb_info[f'dend_responded uid: {uid_str}']= self.stats.responded_peers_count[uid].value
            wandb_info[f'dend_respond_time uid: {uid_str}']= self.stats.peers_respond_time[uid].value
            wandb_info[f'dend_respond_rate uid: {uid_str}']= respond_rate

        return wandb_info