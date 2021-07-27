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

import torch
from typing import Tuple, List, Union, Optional
from torch.autograd.function import once_differentiable
import bittensor

from loguru import logger
logger = logger.opt(colors=True)

# dummy tensor that triggers autograd 
DUMMY = torch.empty(0, requires_grad=True)

# Helper function for filling nill (zero) responses on failures.
def nill_response_for(inputs):
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
        self.config = config
        self.wallet = wallet
        self.receptor_pool = receptor_pool

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
                
                outputs (:obj:`List[torch.FloatTensor]` of shape :obj:`n_endpoints * (batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Output encodings of inputs produced by the remote endpoints. Non-responses are zeroes of common shape.
        """
        ctx.receptor_pool = dendrite.receptor_pool
        ctx.endpoints, ctx.inputs, ctx.modality, ctx.timeout, ctx.does_requires_grad = endpoints, inputs, modality, timeout, requires_grad
        inputs = [ x.cpu().clone().detach() for x in inputs ]
        forward_outputs, forward_codes = ctx.receptor_pool.forward(
            endpoints = endpoints, 
            inputs = inputs, 
            modality = modality,
            timeout = timeout
        )
        ctx.forward_codes = forward_codes
        return (torch.tensor(forward_codes, dtype=torch.int64), *forward_outputs)

    @staticmethod
    @once_differentiable
    def backward( 
            ctx, 
            unused_code_grads: torch.FloatTensor,
            *output_grads: torch.FloatTensor
        ) -> Tuple[ Optional[torch.Tensor], ... ]:
        """ Internal autograd-friendly Backward RPC call to a list of neuron endpoints.

            Args:
                ctx: (:obj:`torch.autograd.ctx`, `required`):
                    Autograd context, saves state information between forward and backward calls. i.e. inputs for gradient computation.

                unused_code_grads: (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    Gradients of this function's codes. (Unused)

                grads (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    Gradients of this function's outputs computed during the loss.backward() call.
            
            Returns:
                DUMMY, None, None, None,
                outputs (:obj:`List[torch.FloatTensor], `optional`):
                    Gradient results for each input.

        """
        if ctx.does_requires_grad:
            grads_cpu = [ x.cpu().clone().detach() for x in output_grads ]
            input_grads, _ =  ctx.receptor_pool.backward (
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
                    codes (:obj:`List[torch.LongTensor]` of shape :obj:`[num_endpoints]`, `required`):
                        dendrite call return codes.

                    responses (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Output encodings of inputs produced by the remote endpoints. Non-responses are zeroes of common shape.
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
            tensors = forward_response[1:]
            return tensors, codes

    def forward_image(
        self, 
        endpoints: Union[ List['bittensor.Endpoint'], 'bittensor.Endpoint'],
        inputs: List[ torch.FloatTensor ],
        timeout: int = None,
        requires_grad: bool = None
    ) -> Tuple[ Union[List[torch.FloatTensor], torch.FloatTensor], torch.LongTensor]:
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
        """
         # Check types.
        if not isinstance(endpoints, list) and not isinstance(endpoints, bittensor._endpoint.endpoint_impl.Endpoint):
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
        if not isinstance(endpoints[0], bittensor._endpoint.endpoint_impl.Endpoint):
            raise ValueError('endpoints must be of type bittensor.Endpoint. Got {}'.format(type(endpoints)))

        # Check shape.
        if len( inputs[0].shape ) != 5:
            error_msg = 'Image inputs should be rank 5 with semantic shape: [batch_size, sequence_len, channels, rows, cols], got {}'.format( inputs[0].shape )
            raise ValueError(error_msg)
        
        # Make calls.
        responses, codes = self._forward(
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
        return responses, codes

    def forward_tensor(
            self, 
            endpoints: Union[ List['bittensor.Endpoint'], 'bittensor.Endpoint'] ,
            inputs: List[ torch.FloatTensor ],
            timeout: int = None,
            requires_grad: bool = None
        ) -> Tuple[ Union[List[torch.FloatTensor], torch.FloatTensor], torch.LongTensor]:
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
        """
        # Check types.
        if not isinstance(endpoints, list) and not isinstance(endpoints, bittensor._endpoint.endpoint_impl.Endpoint):
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
        if not isinstance(endpoints[0], bittensor._endpoint.endpoint_impl.Endpoint):
            raise ValueError('endpoints must be of type bittensor.Endpoint. Got {}'.format(type(endpoints)))

        # Check shape.
        if len( inputs[0].shape ) != 3:
            error_msg = 'Tensor inputs should be rank 3 with semantic shape: [batch_size, sequence_len, bittensor.__network_dim__]'
            raise ValueError(error_msg)
        if inputs[0].shape[2] != bittensor.__network_dim__:
            error_msg = 'Passed tensor must have last dimension {} got {}'.format(bittensor.__network_dim__, inputs[0].shape[2])
            raise ValueError(error_msg)

        # Make calls.
        responses, codes = self._forward(
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
        return responses, codes


    def forward_text(
            self,
            endpoints: Union[ List['bittensor.Endpoint'], 'bittensor.Endpoint'] ,
            inputs: List[ torch.LongTensor ],
            timeout: int = None,
            requires_grad: bool = None
        ) -> Tuple[ Union[List[torch.FloatTensor], torch.FloatTensor], torch.LongTensor]:
        r""" Forward text inputs to a list of neuron endpoints and block until responses or timeout.

                Args:
                    endpoints (:obj:`Union[List[bittensor.Endpoint], bittensor.Endpoint]` of shape :obj:`(num_endpoints)`, `required`):
                        List or single of endpoints which match the length of inputs. Inputs are sent forward to these endpoints.

                    inputs (:obj:`Union[List[torch.LongTensor], torch.LongTensor]` of shape :obj:`(num_endpoints * [batch_size, sequence_len])`, `required`):
                        List or single tensors to send to corresponsing neurons. Tensors are text input_ids encoded using the
                        bittensor tokenizer with shape [batch_size, sequence_len].

                    timeout (:type:`int`, default = dendrite.timeout `optional`):
                        Request timeout.

                    requires_grad (:type:`int`, default = dendrite.requires_grad, `optional`):
                        If true, the backward pass triggers passing gradients on the wire.

                Returns:
                    responses (:obj:`Union[ List[torch.FloatTensor], torch.FloatTensor] ` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Output encodings of inputs produced by remote endpoints. Non-responses are zeroes of input shape plus output dimension.

                    codes (:obj:`torch.LongTensor` of shape :obj:`[ num_endpoints ]`, `required`):
                        dendrite call return ops.

            """
        
        # Check types.
        if not isinstance(endpoints, list) and not isinstance(endpoints, bittensor._endpoint.endpoint_impl.Endpoint):
            raise ValueError('endpoints must be of type list or bittensor.Endpoint. Got {}'.format(type(endpoints)))

        if not isinstance(inputs, list) and not isinstance(inputs, torch.LongTensor):
            raise ValueError('inputs must be of type list[torch.LongTensor] or torch.LongTensor. Got {}'.format(type(inputs)))
        
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
            error_msg = 'List of text inputs should have the same length as passed destination endpoints, got {} and {}'.format(len( inputs ), len( endpoints ))
            raise ValueError(error_msg)
            
        # Check list types.
        if not ( isinstance(inputs[0], torch.cuda.LongTensor) or isinstance(inputs[0], torch.LongTensor)) :
            raise ValueError('inputs must be of type torch.LongTensor. Got {}'.format(type(inputs[0])))
        if not isinstance(endpoints[0], bittensor._endpoint.endpoint_impl.Endpoint):
            raise ValueError('endpoints must be of type bittensor.Endpoint. Got {}'.format(type(endpoints)))

        # Check shape.
        if len( inputs[0].shape ) != 2:
            error_msg = 'Text inputs should be rank 2 with semantic shape: [batch_size, sequence_len]'
            raise ValueError(error_msg)

        # Make calls.
        responses, codes = self._forward(
            endpoints = endpoints, 
            inputs = inputs, 
            modality = bittensor.proto.Modality.TEXT,
            timeout = timeout,
            requires_grad = requires_grad,
        )

        # Format to singletons.
        if non_list_inputs:
            responses = responses[0]

        # Return.
        return responses, codes