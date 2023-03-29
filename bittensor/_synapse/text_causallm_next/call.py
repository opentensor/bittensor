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

import time
from typing import Union

import torch

import bittensor


class TextCausalLMNextForwardCall(bittensor.BittensorCall):
    """Call state for the text_last_hidden_state synapse."""

    # The name of the synapse call.
    synapse_name: str = "forward_text_causallm_next"
    outputs = None # To be filled by the forward call
    topk: int = 4096 # Static.

    def __str__(self) -> str:
        return """
bittensor.TextCausalLMNextForwardCall( 
    description: Returns the logits for the last predicted item in a given sequence.
    caller: {},
    topk: {},
    version: {},
    timeout = {}, 
    start_time = {},
    end_time = {},
    elapsed = {},
    Args:
    \ttext_inputs: torch.LongTensor = {}, 
    \ttext_outputs: torch.FloatTensor = {},
    \ttext_inputs_serializer_type: bittensor.serializer_type = {}, 
    \ttext_outputs_serializer_type: bittensor.serializer_type = {} 
)
""".format(
            self.hotkey,
            self.topk,
            self.version,
            self.timeout,
            self.start_time,
            self.end_time,
            time.time() - self.start_time,
            self.text_inputs,
            self.outputs if self.outputs is not None else "To be filled by the forward call.",
            self.text_inputs_serializer_type,
            self.text_outputs_serializer_type,
        )

    def __init__(
        self,
        text_inputs: torch.LongTensor,
        timeout: float = bittensor.__blocktime__,
        text_inputs_serializer_type: "bittensor.serializer_type" = bittensor.proto.Serializer.MSGPACK,
        text_outputs_serializer_type: "bittensor.serializer_type" = bittensor.proto.Serializer.MSGPACK,
    ):
        """Forward call to the receptor.
        Args:
            text_inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `required`):
                torch tensor of text inputs.
            timeout (:obj:`float`, `optional`, defaults to 5 seconds):
                timeout for the forward call.
            text_inputs_serializer_type (:obj:`bittensor.proto.Serializer`, `optional`, defaults to bittensor.proto.Serializer.MSGPACK):
                serializer type for text inputs.
            text_outputs_serializer_type (:obj:`bittensor.proto.Serializer`, `optional`, defaults to bittensor.proto.Serializer.MSGPACK):
                serializer type for text outputs.
        Returns:
            call.TextCausalLMNextForwardCall (:obj:`call.TextCausalLMNextForwardCall`, `required`):
                bittensor forward call dataclass.
        """
        super().__init__(timeout=timeout)
        self.text_inputs = text_inputs
        self.outputs = None
        self.text_inputs_serializer_type = text_inputs_serializer_type
        self.text_outputs_serializer_type = text_outputs_serializer_type

    def get_inputs_shape(self) -> Union[torch.Size, None]:
        if self.text_inputs is not None:
            return self.text_inputs.shape
        return None

    def get_outputs_shape(self) -> Union[torch.Size, None]:
        if self.outputs is not None:
            return self.outputs.shape
        return None


class TextCausalLMNextBackwardCall(bittensor.BittensorCall):
    """Backward call state for the text_last_hidden_state synapse."""

    # The name of the synapse call.
    synapse_name: str = "backward_text_last_hidden_state"
    outputs = None

    def __str__(self) -> str:
        return """
bittensor.TextCausalLMNextBackwardCall( 
    description: Sends the gradients of the last hidden state to the server.
    caller: {},
    version: {},
    timeout = {}, 
    start_time = {},
    end_time = {},
    elapsed = {},
    Args:
    \ttext_inputs: torch.LongTensor = {}, 
    \thidden_state: torch.FloatTensor = {},
    \thidden_state_grads: torch.FloatTensor = {}, 
    \tmask: torch.BoolTensor = {}, 
    \thidden_states: torch.FloatTensor = {},
    \tmask_serializer_type: bittensor.serializer_type = {}, 
    \ttext_inputs_serializer_type: bittensor.serializer_type = {}, 
    \thidden_states_serializer_type: bittensor.serializer_type = {}
    \thidden_states_grads_serializer_type: bittensor.serializer_type = {} 
)
""".format(
            self.hotkey,
            self.version,
            self.timeout,
            self.start_time,
            self.end_time,
            time.time() - self.start_time,
            self.text_inputs,
            self.hidden_states,
            self.hidden_states_grads,
            self.mask,
            self.hidden_states
            if self.hidden_states is not None
            else "To be filled by the forward call.",
            self.mask_serializer_type,
            self.text_inputs_serializer_type,
            self.hidden_states_serializer_type,
            self.hidden_states_grads_serializer_type,
        )

    def __init__(
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
        """Forward call to the receptor.
        Args:
            text_inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `required`):
                torch tensor of text inputs.
            hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, net_dim)`, `required`):
                torch tensor of hidden states against which the gradients have accumulated.
            hidden_states_grads (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, net_dim)`, `required`):
                torch tensor of grads for the hidden states.
            mask (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
                mask over returned hidden states.
            mask_serializer_type (:obj:`bittensor.proto.Serializer`, `optional`, defaults to bittensor.proto.Serializer.MSGPACK):
                serializer type for mask.
            text_inputs_serializer_type (:obj:`bittensor.proto.Serializer`, `optional`, defaults to bittensor.proto.Serializer.MSGPACK):
                serializer type for text inputs.
            hidden_states_serializer_type (:obj:`bittensor.proto.Serializer`, `optional`, defaults to bittensor.proto.Serializer.MSGPACK):
                serializer type for hidden states.
        Returns:
            call.TextCausalLMNextForwardCall (:obj:`call.TextCausalLMNextForwardCall`, `required`):
                bittensor forward call dataclass.
        """
        super().__init__()
        # Torch tensors.
        self.text_inputs = text_inputs
        self.hidden_states = hidden_states
        self.hidden_states_grads = hidden_states_grads
        self.mask = mask

        # Serializer types.
        self.mask_serializer_type = mask_serializer_type
        self.text_inputs_serializer_type = text_inputs_serializer_type
        self.hidden_states_serializer_type = hidden_states_serializer_type
        self.hidden_states_grads_serializer_type = hidden_states_grads_serializer_type

    def get_inputs_shape(self) -> torch.Size:
        if self.text_inputs is not None:
            return self.text_inputs.shape
        else:
            return None

    def get_outputs_shape(self) -> torch.Size:
        if self.outputs is not None:
            return self.outputs.shape
        else:
            return None
