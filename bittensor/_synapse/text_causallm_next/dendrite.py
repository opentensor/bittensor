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

from typing import List, Dict, Callable

import torch

import bittensor

import asyncio

class TextCausalLMNextDendrite(bittensor.Dendrite):
    """Dendrite for the text_last_hidden_state synapse."""

    # Dendrite name.
    name: str = "text_causallm_next"

    def __str__(self) -> str:
        return "TextCausalLMNext"

    def get_stub(self, channel) -> Callable:
        return bittensor.grpc.TextCausalLMNextStub(channel)

    def pre_process_forward_call_to_request_proto(
        self, forward_call: "bittensor.TextCausalLMNextForwardCall"
    ) -> "bittensor.ForwardTextCausalLMNextRequest":
        """Preprocesses the forward call to a request proto.
        --------------------------------------------
        Args:
            forward_call (:obj:`bittensor.TextCausalLMNextForwardCall`, `required`):
                forward_call to preprocess.
        Returns:
            request_proto (:obj:`bittensor.ForwardTextCausalLMNextRequest`, `required`):
                bittensor request proto object.
        """
        # Serialize text inputs.
        text_serializer = bittensor.serializer(
            serializer_type=forward_call.text_inputs_serializer_type
        )
        serialized_text = text_serializer.serialize(forward_call.text_inputs)

        # Return forward call.
        return bittensor.ForwardTextCausalLMNextRequest(
            timeout=forward_call.timeout,
            topk=forward_call.topk,
            serialized_text_inputs=serialized_text,
            text_inputs_serializer_type=forward_call.text_inputs_serializer_type,
            text_outputs_serializer_type=forward_call.text_outputs_serializer_type,
        )

    def post_process_response_proto_to_forward_call(
        self,
        forward_call: bittensor.TextCausalLMNextForwardCall,
        response_proto: bittensor.ForwardTextCausalLMNextResponse,
    ) -> bittensor.TextCausalLMNextForwardCall:
        """Postprocesses the response proto to fill forward call.
        --------------------------------------------
        Args:
            forward_call (:obj:`bittensor.TextCausalLMNextForwardCall`, `required`):
                bittensor forward call object to fill.
            response_proto (:obj:`bittensor.ForwardTextCausalLMNextResponse`, `required`):
                bittensor forward response proto.
        Returns:
            forward_call (:obj:`bittensor.TextCausalLMNextForwardCall`, `required`):
                filled bittensor forward call object.
        """

        forward_call.response_code = response_proto.return_code
        forward_call.response_message = response_proto.message

        if response_proto.return_code != bittensor.proto.ReturnCode.Success:
            forward_call.outputs = None
            return forward_call

        # Deserialize hidden states.
        text_outputs_serializer = bittensor.serializer(
            serializer_type=forward_call.text_outputs_serializer_type
        )
        text_outputs = text_outputs_serializer.deserialize(
            response_proto.serialized_text_outputs
        )

        # Unravel topk
        unraveled_topk = bittensor.unravel_topk_token_phrases(text_outputs, topk=forward_call.topk)

        # TODO: Compute loss calculation
        forward_call.outputs = unraveled_topk

        # Delete unnecessary proto(s)
        del forward_call.request_proto, forward_call.response_proto

        return forward_call

    def forward(
        self,
        text_inputs: torch.Tensor,
        timeout: float = bittensor.__blocktime__,
    ) -> "bittensor.TextPromptingForwardCall":
        loop = asyncio.get_event_loop()
        return loop.run_until_complete( 
            self._async_forward( 
                forward_call = bittensor.TextCausalLMForwardCall(
                    text_inputs=text_inputs,
                    timeout=timeout
                ) 
            ) 
        )
    
    def async_forward(
        self,
        text_inputs: torch.Tensor,
        timeout: float = bittensor.__blocktime__,
    ) -> "bittensor.TextPromptingForwardCall":
        return self._async_forward( 
                forward_call = bittensor.TextCausalLMForwardCall(
                    text_inputs=text_inputs,
                    timeout=timeout
            ) 
        )    

    def pre_process_backward_call_to_request_proto(
        self, backward_call: "bittensor.TextCausalLMNextBackwardCall"
    ) -> "bittensor.BackwardTextCausalLMNextRequest":
        """Preprocesses the forward call to a request proto.
        --------------------------------------------
        Args:
            forward_call (:obj:`bittensor.TextCausalLMNextBackwardCall`, `required`):
                backward_call to preprocess.
        Returns:
            request_proto (:obj:`bittensor.BackwardTextCausalLMNextRequest`, `required`):
                bittensor request proto object.
        """
        # Serialize text inputs.
        text_serializer = bittensor.serializer(
            serializer_type=backward_call.text_inputs_serializer_type
        )
        serialized_text_inputs = text_serializer.serialize(backward_call.text_inputs)

        if backward_call.mask != None:
            # Apply mask to hidden states.
            hidden_states = backward_call.hidden_states.reshape(-1, bittensor.__network_dim__)
            hidden_states = hidden_states[backward_call.mask.reshape(-1)]
        hidden_states_serializer = bittensor.serializer(
            serializer_type=backward_call.hidden_states_serializer_type
        )
        serialized_hidden_states = hidden_states_serializer.serialize(hidden_states)

        if backward_call.mask != None:
            # Apply mask to gradients.
            hidden_states_grads = backward_call.hidden_states_grads.reshape(
                -1, bittensor.__network_dim__
            )
            hidden_states_grads = hidden_states_grads[backward_call.mask.reshape(-1)]
        hidden_states_grads_serializer = bittensor.serializer(
            serializer_type=backward_call.hidden_states_grads_serializer_type
        )
        serialized_hidden_states_grads = hidden_states_grads_serializer.serialize(
            hidden_states_grads
        )

        if backward_call.mask != None:
            # serialize mask.
            mask_serializer = bittensor.serializer(
                serializer_type=backward_call.mask_serializer_type
            )
            serialized_mask = mask_serializer.serialize(backward_call.mask)
        else:
            serialized_mask = None

        # Return forward call.
        return bittensor.BackwardTextCausalLMNextRequest(
            serialized_mask=serialized_mask,
            serialized_text_inputs=serialized_text_inputs,
            serialized_hidden_states=serialized_hidden_states,
            serialized_hidden_states_grads=serialized_hidden_states_grads,
            mask_serializer_type=backward_call.mask_serializer_type,
            text_inputs_serializer_type=backward_call.text_inputs_serializer_type,
            hidden_states_serializer_type=backward_call.hidden_states_serializer_type,
            hidden_states_grads_serializer_type=backward_call.hidden_states_grads_serializer_type,
        )

    def backward(
        self,
        text_inputs: torch.FloatTensor,
        hidden_states: torch.FloatTensor,
        hidden_states_grads: torch.FloatTensor,
        mask: torch.BoolTensor = None,
        mask_serializer_type: "bittensor.serializer_type" = bittensor.proto.Serializer.MSGPACK,
        text_inputs_serializer_type: "bittensor.serializer_type" = bittensor.proto.Serializer.MSGPACK,
        hidden_states_serializer_type: "bittensor.serializer_type" = bittensor.proto.Serializer.MSGPACK,
        hidden_states_grads_serializer_type: "bittensor.serializer_type" = bittensor.proto.Serializer.MSGPACK,
    ):
        """Backward call to the receptor.
        Args:
            text_inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `required`):
                torch tensor of text inputs.
            hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, net_dim)`, `required`):
                torch tensor of hidden states against which the gradients have accumulated.
            hidden_states_grads (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, net_dim)`, `required`):
                torch tensor of grads for the hidden states.
            mask (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
                mask over returned hidden states.
            text_inputs_serializer_type (:obj:`bittensor.proto.Serializer`, `optional`, defaults to bittensor.proto.Serializer.MSGPACK):
                serializer type for text inputs.
            hidden_states_serializer_type (:obj:`bittensor.proto.Serializer`, `optional`, defaults to bittensor.proto.Serializer.MSGPACK):
                serializer type for hidden states.
            hidden_states_grads_serializer_type (:obj:`bittensor.proto.Serializer`, `optional`, defaults to bittensor.proto.Serializer.MSGPACK):
                serializer type for hidden states grads.
        Returns:
            None
        """
        return self._backward(
            backward_call=bittensor.TextCausalLMNextBackwardCall(
                text_inputs=text_inputs,
                hidden_states=hidden_states,
                hidden_states_grads=hidden_states_grads,
                mask=mask,
                mask_serializer_type=mask_serializer_type,
                text_inputs_serializer_type=text_inputs_serializer_type,
                hidden_states_serializer_type=hidden_states_serializer_type,
                hidden_states_grads_serializer_type=hidden_states_grads_serializer_type,
            )
        )
