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

from typing import Callable

import torch

import bittensor


class TextCausalLMDendrite(bittensor.Dendrite):
    """Dendrite for the text_last_hidden_state synapse."""

    # Dendrite name.
    name: str = "text_causallm"

    def __str__(self) -> str:
        return "TextCausalLM"

    def get_stub(self, channel) -> Callable:
        return bittensor.grpc.TextCausalLMStub(channel)

    def pre_process_forward_call_to_request_proto(
        self, forward_call: "bittensor.TextCausalLMForwardCall"
    ) -> "bittensor.ForwardTextCausalLMRequest":
        """Preprocesses the forward call to a request proto.
        --------------------------------------------
        Args:
            forward_call (:obj:`bittensor.TextCausalLMForwardCall`, `required`):
                forward_call to preprocess.
        Returns:
            request_proto (:obj:`bittensor.ForwardTextCausalLMRequest`, `required`):
                bittensor request proto object.
        """
        # Serialize text inputs.
        text_serializer = bittensor.serializer(
            serializer_type=forward_call.text_inputs_serializer_type
        )
        serialized_text = text_serializer.serialize(forward_call.text_inputs)
        # import pdb; pdb.set_trace()
        # Return forward call.
        return bittensor.ForwardTextCausalLMRequest(
            timeout=forward_call.timeout,
            topk=forward_call.topk,
            serialized_text_inputs=serialized_text,
            text_inputs_serializer_type=forward_call.text_inputs_serializer_type,
            text_outputs_serializer_type=forward_call.text_outputs_serializer_type,
        )

    def post_process_response_proto_to_forward_call(
        self,
        forward_call: bittensor.TextCausalLMForwardCall,
        response_proto: bittensor.ForwardTextCausalLMResponse,
    ) -> bittensor.TextCausalLMForwardCall:
        """Postprocesses the response proto to fill forward call.
        --------------------------------------------
        Args:
            forward_call (:obj:`bittensor.TextCausalLMForwardCall`, `required`):
                bittensor forward call object to fill.
            response_proto (:obj:`bittensor.ForwardTextCausalLMResponse`, `required`):
                bittensor forward response proto.
        Returns:
            forward_call (:obj:`bittensor.TextCausalLMForwardCall`, `required`):
                filled bittensor forward call object.
        """
        forward_call.response_code = response_proto.return_code
        forward_call.response_message = response_proto.message

        if response_proto.return_code != bittensor.proto.ReturnCode.Success:
            forward_call.hidden_states = None
            return forward_call

        # Deserialize hidden states.
        text_outputs_serializer = bittensor.serializer(
            serializer_type=forward_call.text_outputs_serializer_type
        )
        text_outputs = text_outputs_serializer.deserialize(
            response_proto.serialized_text_outputs
        )

        forward_call.outputs = text_outputs
        return forward_call

    def forward(
        self,
        text_inputs: torch.LongTensor,
        timeout: float = bittensor.__blocktime__,
        text_inputs_serializer_type: "bittensor.serializer_type" = bittensor.proto.Serializer.MSGPACK,
        text_outputs_serializer_type: "bittensor.serializer_type" = bittensor.proto.Serializer.MSGPACK,
    ) -> "bittensor.TextCausalLMForwardCall":
        """Forward call to the receptor.
        Args:
            text_inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `required`):
                torch tensor of text inputs.
            timeout (:obj:`float`, `optional`, defaults to 5 seconds):
                timeout for the forward call.
            text_inputs_serializer_type (:obj:`bittensor.proto.Serializer`, `optional`, defaults to bittensor.proto.Serializer.MSGPACK):
                serializer type for text inputs.
            text_outputs_serializer_type (:obj:`bittensor.proto.Serializer`, `optional`, defaults to bittensor.proto.Serializer.MSGPACK):
                serializer type for hidden states.
        Returns:
            bittensor.TextCausalLMForwardCall (:obj:`bittensor.TextCausalLMForwardCall`, `required`):
                bittensor forward call dataclass.
        """
        return self._forward(
            forward_call=bittensor.TextCausalLMForwardCall(
                text_inputs=text_inputs,
                timeout=timeout,
                text_inputs_serializer_type=text_inputs_serializer_type,
                text_outputs_serializer_type=text_outputs_serializer_type,
            )
        )

    def pre_process_backward_call_to_request_proto(
        self, backward_call: "bittensor.TextCausalLMBackwardCall"
    ) -> "bittensor.BackwardTextCausalLMRequest":
        """Preprocesses the forward call to a request proto.
        --------------------------------------------
        Args:
            forward_call (:obj:`bittensor.TextCausalLMBackwardCall`, `required`):
                backward_call to preprocess.
        Returns:
            request_proto (:obj:`bittensor.BackwardTextCausalLMRequest`, `required`):
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
        return bittensor.BackwardTextCausalLMRequest(
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
            backward_call=bittensor.TextCausalLMBackwardCall(
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
