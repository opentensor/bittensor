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
import pandas as pd
from termcolor import colored
from typing import Tuple, List, Union, Optional
from torch.autograd.function import once_differentiable
import bittensor

from loguru import logger
logger = logger.opt(colors=True)

# dummy tensor that triggers autograd 
DUMMY = torch.empty(0, requires_grad=True)

class Dendrite( torch.autograd.Function ):

    def __init__(
            self, 
            config: 'bittensor.Config',
            wallet: 'bittensor.Wallet',
            receptor_pool: 'bittensor.ReceptorPool'
        ):
        r""" Initializes a new Dendrite entry point.
            Args:
                config (:obj:`bittensor.Config`, `required`): 
                    bittensor.dendrite.config()
                wallet (:obj:`bittensor.Wallet`, `required`):
                    bittensor wallet with hotkey and coldkeypub.
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

            Returns:
                codes (:obj:`torch.LongTensor` of shape :obj:`(n_endpoints)` `required`):
                    Return code associated with forward call.
                
                outputs (:obj:`List[torch.FloatTensor]` of shape :obj:`n_endpoints * (batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Output encodings of inputs produced by the remote endpoints. Non-responses are zeroes of common shape.
        """
        ctx.receptor_pool = dendrite.receptor_pool
        ctx.endpoints, ctx.inputs, ctx.modality = endpoints, inputs, modality
        inputs = [ x.cpu().clone().detach() for x in inputs ]
        outputs, forward_codes = ctx.receptor_pool.forward(
            endpoints = endpoints, 
            inputs = inputs, 
            modality = modality
        )
        ctx.forward_codes = forward_codes
        return (torch.tensor(forward_codes, dtype=torch.int64), *outputs)

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
        grads_cpu = [ x.cpu().clone().detach() for x in output_grads ]
        input_grads, _ =  ctx.receptor_pool.backward (
            endpoints = ctx.endpoints, 
            inputs = ctx.inputs, 
            grads = grads_cpu, 
            modality = ctx.modality
        )
        return (None, None, None, None, *input_grads)

    def _forward(
                self,
                endpoints: List['bittensor.Endpoint'],
                inputs: List[torch.Tensor],
                modality: bittensor.proto.Modality
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

                Returns:
                    codes (:obj:`List[torch.LongTensor]` of shape :obj:`[num_endpoints]`, `required`):
                        dendrite call return codes.

                    responses (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Output encodings of inputs produced by the remote endpoints. Non-responses are zeroes of common shape.
            """
            forward_response = Dendrite.apply(
                self,
                DUMMY, 
                endpoints, 
                modality,
                *inputs
            )
            codes = forward_response[0]
            tensors = forward_response[1:]
            return tensors, codes

    def forward_text(
            self,
            endpoints: Union[ List['bittensor.Endpoint'], 'bittensor.Endpoint'] ,
            inputs: List[ torch.LongTensor ]
        ) -> Tuple[ Union[List[torch.FloatTensor], torch.FloatTensor], torch.LongTensor]:
        r""" Forward text inputs to a list of neuron endpoints and block until responses or timeout.

                Args:
                    endpoints (:obj:`Union[List[bittensor.Endpoint], bittensor.Endpoint]` of shape :obj:`(num_endpoints)`, `required`):
                        List or single of endpoints which match the length of inputs. Inputs are sent forward to these endpoints.

                    inputs (:obj:`Union[List[torch.LongTensor], torch.LongTensor]` of shape :obj:`(num_endpoints * [batch_size, sequence_len])`, `required`):
                        List or single tensors to send to corresponsing neurons. Tensors are text input_ids encoded using the
                        bittensor tokenizer with shape [batch_size, sequence_len].

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
        if not isinstance(inputs[0], torch.LongTensor):
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
            modality = bittensor.proto.Modality.TEXT
        )

        # Format to singletons.
        if non_list_inputs:
            responses = responses[0]

        # Return.
        return responses, codes
        

    # def forward_image(self, endpoints: List['bittensor.Endpoint'],
    #                   x: List[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
    #     r""" Forward image inputs to endpoints.

    #         Args:
    #             endpoints (:obj:`List[bittensor.Endpoint]` of shape :obj:`(num_endpoints)`, `required`):
    #                 List of remote endpoints which match length of x. Tensors from x are sent forward to these endpoints.

    #             x (:obj:`List[torch.Tensor]` of shape :obj:`(num_endpoints * [batch_size, sequence_len, channels, rows, cols])`, `required`):
    #                 List of image-tensors to send to corresponsing endpoints. Tensors are images encoded using the
    #                 torch.toTensor() or other encoding which produces the shape [batch_size, channels, rows, cols].

    #         Returns:
    #             forwad_output (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, sequence_len, bittensor.network_size)`, `required`):
    #                 Output encodings of images produced by remote endpoints. Non-responses are zeroes of common shape.

    #             return_codes (:obj:`torch.LongTensor` of shape :obj:`[num_endpoints]`, `required`):
    #                 dendrite call return ops.
    #     """
    #     # TODO(const): Checks across all tensors and other shape checks.
    #     if len(x[0].shape) != 5:
    #         error_msg = 'Image inputs should be rank 5 with semantic shape: [batch_size, sequence_dim, channels, rows, cols]'
    #         raise ValueError(error_msg)
    #     if len(x) != len(endpoints):
    #         error_msg = 'List of image inputs x should have the same length as passed destination endpoints, got {} and {}'.format(len(x), len(endpoints))
    #         raise ValueError(error_msg)
    #     if len(x) < 1:
    #         error_msg = 'Must pass more than 0 input for argument x, got {}'.format(len(x))
    #         raise ValueError(error_msg)
    #     return self.forward(endpoints, x, bittensor.proto.Modality.IMAGE)

    # def forward_tensor(
    #         self, 
    #         endpoints: List['bittensor.Endpoint'],
    #         x: List[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
    #     r""" Forward tensor inputs to endpoints.

    #         Args:
    #             endpoints (:obj:`List[bittensor.Endpoint]` of shape :obj:`(num_endpoints)`, `required`):
    #                 List of remote endpoints which match length of x. Tensors from x are sent forward to these endpoints.

    #             x (:obj:`List[torch.Tensor]` of shape :obj:`(num_endpoints * [batch_size, sequence_len, bittensor.__network_dim__])`, `required`):
    #                 List of tensors to send to corresponsing endpoints. Tensors are of arbitrary type and
    #                 with shape [batch_size, sequence_len, bittensor.__network_dim__].

    #         Returns:
    #             forwad_output (:obj:`List[torch.FloatTensor]` of shape :obj:`num_endpoints * (batch_size, sequence_len, bittensor.__network_dim__)]`, `required`):
    #                 Output encodings of tensors produced by remote endpoints. Non-responses are zeroes of common shape.

    #             return_codes (:obj:`torch.LongTensor` of shape :obj:`[num_endpoints]`, `required`):
    #                 dendrite call return ops.
    #     """
    #     if len(x[0].shape) != 3:
    #         error_msg = 'Tensor inputs should be rank 3 with semantic shape: [batch_size, sequence_len, feature_len]'
    #         raise ValueError(error_msg)
    #     if len(x) != len(endpoints):
    #         error_msg = 'List of tensor inputs x should have the same length as passed destination endpoints, got {} and {}'.format(len(x), len(endpoints))
    #         raise ValueError(error_msg)
    #     if x[0].shape[2] != bittensor.__network_dim__:
    #         error_msg = 'Passed tensor must have last dimension {} got {}'.format(bittensor.__network_dim__, x[0].shape[2])
    #         raise ValueError(error_msg)
    #     if len(x) == 0:
    #         error_msg = 'Must pass more than 0 input for argument x, got {}'.format(len(x))
    #         raise ValueError(error_msg)
    #     return self.forward(endpoints, x, bittensor.proto.Modality.TENSOR)

    # def forward(
    #         self, 
    #         endpoints: List['bittensor.Endpoint'],
    #         x: List[torch.Tensor],
    #         mode: bittensor.proto.Modality) -> Tuple[List[torch.Tensor], torch.LongTensor]:
    #     r""" Forward tensor inputs to endpoints.

    #         Args:
    #             endpoints (:obj:`List[bittensor.Endpoint]` of shape :obj:`(num_endpoints)`, `required`):
    #                 List of remote endpoints which match length of x. Tensors from x are sent forward to these endpoints.

    #             x (:obj:`List[torch.Tensor]` of shape :obj:`(num_endpoints * [shape])`, `required`):
    #                 List of tensors to send to corresponsing endpoints. Tensors are of arbitrary type and shape depending on the
    #                 modality.

    #             mode (:obj:`bittensor.proto.Modality` of shape :obj:`(1)`, `required`):
    #                 Bittensor forward modality type. Enum in [TEXT, IMAGE, TENSOR]

    #         Returns:
    #             forward_outputs (:obj:`List[torch.FloatTensor]` of shape :obj:`num_endpoints * (batch_size, sequence_len, bittensor.network_size)]`, `required`):
    #                 Output encodings of tensors produced by remote endpoints. Non-responses are zeroes of common shape.

    #             return_codes (:obj:`torch.LongTensor` of shape :obj:`[num_endpoints]`, `required`):
    #                 dendrite call return ops.
    #     """
    #     if len(x) != len(endpoints):
    #         error_msg = 'List of inputs x should have the same length as passed destination endpoints, got {} and {}'.format(len(x), len(endpoints))
    #         raise ValueError(error_msg)
    #     if len(x) < 1:
    #         error_msg = 'Must pass more than 0 input for argument x, got {}'.format(len(x))
    #         raise ValueError(error_msg)

    #     # ---- Stats ---
    #     self.stats.qps.update(1)

    #     # ---- Run threaded calls with executor ----
    #     tensor_results = []
    #     return_codes = []
        
    #     # --- Create calls ----
    #     def _call_receptor_with_args( receptor, inputs, mode ):
    #         return receptor.forward( inputs = inputs, mode = mode )

    #     # ---- Fill calls ----
    #     call_args = [ (self._get_or_create_receptor_for_endpoint( endpoint ), inputs, mode) for (inputs, endpoint) in list(zip( x, endpoints )) ]
    #     for result in self.thread_pool.map( lambda args: _call_receptor_with_args(*args), call_args ):
    #         tensor_results.append( result[0] )
    #         return_codes.append( result[1] )

    #     # ---- Kill receptors ----
    #     self._destroy_receptors_over_max_allowed()
        
    #     # ---- Return ----
    #     return_codes = torch.tensor(return_codes, dtype=torch.int64)
    #     return tensor_results, return_codes

    # def _destroy_receptors_over_max_allowed( self ):
    #     r""" Destroys receptors based on QPS until there are no more than max_active_receptors.
    #     """

    #     # ---- Finally: Kill receptors over max allowed ----
    #     while len(self._receptors) > self.config.dendrite.max_active_receptors:
    #         min_receptor_qps = math.inf
    #         receptor_to_remove = None
    #         for next_receptor in self._receptors.values():
    #             next_qps = next_receptor.stats.forward_qps.value
    #             if min_receptor_qps > next_qps:
    #                 receptor_to_remove = next_receptor
    #         if receptor_to_remove != None:
    #             logger.debug('<white>Destroy receptor for endpoint:</white> {}', receptor_to_remove.endpoint )
    #             del self._receptors[ receptor_to_remove.endpoint.hotkey ]

    # def _get_or_create_receptor_for_endpoint( self, endpoint: 'bittensor.Endpoint' ) -> 'bittensor.Receptor':
    #     r""" Finds or creates a receptor TCP connection associated with the passed Neuron Endpoint
    #         Returns
    #             receptor: (`bittensor.Receptor`):
    #                 receptor with tcp connection endpoint at endpoint.ip:endpoint.port
    #     """

    #     # ---- Find the active receptor for this endpoint ----
    #     if endpoint.hotkey in self._receptors:
    #         receptor = self._receptors[ endpoint.hotkey ]

    #         # Change receptor address.
    #         if receptor.endpoint.ip != endpoint.ip or receptor.endpoint.port != endpoint.port:
    #             del receptor
    #             logger.debug('<white>Update receptor for endpoint:</white> {}', endpoint )
    #             receptor = bittensor.receptor (
    #                 endpoint = endpoint, 
    #                 config = self.config, 
    #                 wallet = self.wallet
    #             )            
    #             self._receptors[ receptor.endpoint.hotkey ] = receptor

    #     # ---- Or: Create a new receptor ----
    #     else:
    #         logger.debug('<white>Create receptor for endpoint:</white> {}', endpoint )
    #         receptor = bittensor.receptor (
    #                 endpoint = endpoint, 
    #                 config = self.config, 
    #                 wallet = self.wallet
    #         )
    #         self._receptors[ receptor.endpoint.hotkey ] = receptor

    #     return receptor

    # def __str__(self):
    #     total_bytes_out = 0
    #     total_bytes_in = 0
    #     for receptor in self._receptors.values():
    #         total_bytes_out += receptor.stats.forward_bytes_out.value
    #         total_bytes_in += receptor.stats.forward_bytes_in.value
    #     qps_str = colored('{:.3f}'.format(self.stats.qps.value), 'blue')
    #     total_out_bytes_str = colored('\u290A{:.1f}'.format((total_bytes_out*8)/1000), 'green')
    #     total_in_bytes_str = colored('\u290B{:.1f}'.format((total_bytes_in*8)/1000), 'red')
    #     return "(" + qps_str + "q/s|" + total_in_bytes_str + "/" + total_out_bytes_str + "kB/s" + ")"

    # def __rich__(self): 
    #     total_bytes_out = 0
    #     total_bytes_in = 0
    #     for receptor in self._receptors.values():
    #         total_bytes_out += receptor.stats.forward_bytes_out.value
    #         total_bytes_in += receptor.stats.forward_bytes_in.value
    #     total_out_bytes_str = '[green]\u290A{:.1f}[/green]'.format((total_bytes_out * 8)/1000)
    #     total_in_bytes_str = '[red]\u290B{:.1f}[/red]'.format((total_bytes_in * 8)/1000)
    #     qps_str = "[blue]{:.3f}[/blue]".format(float(self.stats.qps.value))
    #     return "(" + qps_str + "q/s|" + total_out_bytes_str + "/" + total_in_bytes_str + "kB/s" + ")"

    # def __full_str__(self):
    #     uids = [receptor.endpoint.uid for receptor in self._receptors.values()]
    #     bytes_out = [receptor.stats.forward_bytes_out.value * (8/1000) for receptor in self._receptors.values()]
    #     bytes_in = [receptor.stats.forward_bytes_in.value * (8/1000) for receptor in self._receptors.values()]
    #     qps = [receptor.stats.forward_qps.value + receptor.stats.backward_qps.value for receptor in self._receptors.values()]
    #     rows = [bytes_out, bytes_in, qps]
    #     df = pd.DataFrame(rows, columns=uids)
    #     df = df.rename(index={df.index[0]: colored('\u290A kB/s', 'green')})
    #     df = df.rename(index={df.index[1]: colored('\u290B kB/s', 'red')})
    #     df = df.rename(index={df.index[2]: colored('Q/s', 'blue')})
    #     return '\nDendrite:\n' + df.to_string(max_rows=5000, max_cols=25, line_width=1000, float_format = lambda x: '%.2f' % x, col_space=1, justify='left')
    
    # def __to_tensorboard__(self, tensorboard, global_step):
    #     total_bytes_out = 0
    #     total_bytes_in = 0
    #     for receptor in self._receptors.values():
    #         total_bytes_out += receptor.stats.forward_bytes_out.value
    #         total_bytes_in += receptor.stats.forward_bytes_in.value
    #     total_in_bytes = (total_bytes_in*8)/1000
    #     total_out_bytes = (total_bytes_out*8)/1000
    #     tensorboard.add_scalar('Dendrite/Incoming bytes', total_in_bytes, global_step)
    #     tensorboard.add_scalar('Dendrite/Outgoing bytes', total_out_bytes, global_step)

    # @property
    # def receptors(self):
    #     return self._receptors.values()