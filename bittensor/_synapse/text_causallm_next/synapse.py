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

import copy

import torch

import bittensor

from abc import abstractmethod


class TextCausalLMNextSynapse(bittensor.Synapse, bittensor.grpc.TextCausalLMNextServicer):
    """TextCausalLMNextSynapse: A class for servicing text_causallm_next requests."""

    synapse_name: str = "text_causallm_next"
    default_blacklist_stake: float = 10

    def __init__(
        self,
        config: "bittensor.Config" = None,
    ):
        """ __init__: Initializes the synapse.
            Args:
                config (:obj:`bittensor.Config`, `optional`):
                    bittensor config object.
        """
        if config is None: config = self.config()
        TextCausalLMNextSynapse.check_config(config)
        super().__init__(config)
        self.config = copy.deepcopy(config)

    def __str__(self):
        return 'TextCausalLMNext'

    def _attach(self, axon: "bittensor.axon"):
        """_attach: Attaches the synapse to the axon."""
        bittensor.grpc.add_TextCausalLMNextServicer_to_server(self, axon.server)

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        ...

    def apply_forward_call(self, forward_call: bittensor.BittensorCall) -> "bittensor.BittensorCall":
        logits = self.forward(forward_call.text_inputs)
        logits = logits[:, -1, :]

        tokenizer = bittensor.tokenizer()
        bittensor.prep_tokenizer(tokenizer=tokenizer, std_tokenizer=tokenizer)

        topk_values = bittensor.topk_token_phrases(
            logits=logits, tokenizer=tokenizer, topk=forward_call.topk
        )
        compact_topk = bittensor.compact_topk_token_phrases(topk_values)

        forward_call.outputs = compact_topk
        return forward_call

    def pre_process_request_proto_to_forward_call(
        self, request_proto: bittensor.ForwardTextCausalLMNextRequest
    ) -> "bittensor.TextCausalLMNextForwardCall":
        """pre_process_request_proto_to_forward_call
        ------------------------------------------
        Args:
            request_proto (bittensor.ForwardTextCausalLMNextRequest):
                bittensor forward request proto.
        Returns:
            bittensor.TextCausalLMNextForwardCall (:obj:`bittensor.TextCausalLMNextForwardCall`, `required`):
                bittensor forward call dataclass.
        """
        # Deserialize text inputs.
        text_deserializer = bittensor.serializer(
            serializer_type=request_proto.text_inputs_serializer_type
        )
        text_inputs = text_deserializer.deserialize(request_proto.serialized_text_inputs)

        return bittensor.TextCausalLMNextForwardCall(
            text_inputs=text_inputs,
            timeout=request_proto.timeout,
            text_inputs_serializer_type=request_proto.text_inputs_serializer_type,
            text_outputs_serializer_type=request_proto.text_outputs_serializer_type,
        )

    def post_process_forward_call_to_response_proto(
        self, forward_call: "bittensor.TextCausalLMNextForwardCall"
    ) -> bittensor.ForwardTextCausalLMNextResponse:
        """post_process_forward_call_to_response_proto
        --------------------------------------------
        Args:
            forward_call (bittensor.TextCausalLMNextForwardCall):
                forward_call.text_inputs (torch.FloatTensor): text inputs.
                forward_call.timeout (float): timeout for the request.
                forward_call.text_inputs_serializer_type (bittensor.proto.SerializerType): text inputs serializer type.
                forward_call.hidden_states_serializer_type (bittensor.proto.SerializerType): hidden states serializer type.
                forward_call.hidden_states (torch.FloatTensor): hidden states.
        Returns:
            response (bittensor.ForwardTextCausalLMNextResponse):
                response.serialized_hidden_states (string): serialized hidden states.
        """
        # Serialize hidden states.
        outputs_serializer = bittensor.serializer(
            serializer_type=forward_call.text_outputs_serializer_type
        )

        # Check if response is successful
        if (forward_call.request_code != bittensor.proto.ReturnCode.Success) or (
            forward_call.response_code != bittensor.proto.ReturnCode.Success
        ):
            serialized_text_outputs = None
        else:
            text_outputs = forward_call.outputs
            serialized_text_outputs = outputs_serializer.serialize(text_outputs)

        # Return the forward response proto.
        return bittensor.ForwardTextCausalLMNextResponse(
            version=bittensor.__version_as_int__,
            hotkey=self.axon.wallet.hotkey.ss58_address,
            serialized_text_outputs=serialized_text_outputs,
            message=forward_call.request_message,
            return_code=forward_call.request_code,
        )

    def pre_process_request_proto_to_backward_call(
        self, request_proto: "bittensor.BackwardRequest"
    ) -> "bittensor.BittensorCall":
        """pre_process_request_proto_to_backward_call
        ------------------------------------------
        Args:
            request_proto (bittensor.BackwardRequest):
                request_proto to process in to a backward call.
        Returns:
            bittensor.BittensorCall (:obj:`bittensor.BittensorCall`, `required`):
                backward call processed from the request proto.
        """
        text_deserializer = bittensor.serializer(
            serializer_type=request_proto.text_inputs_serializer_type
        )
        text_inputs = text_deserializer.deserialize(request_proto.serialized_text_inputs)

        hidden_states_deserializer = bittensor.serializer(
            serializer_type=request_proto.hidden_states_serializer_type
        )
        hidden_states = hidden_states_deserializer.deserialize(
            request_proto.serialized_hidden_states
        )

        hidden_states_grads_deserializer = bittensor.serializer(
            serializer_type=request_proto.hidden_states_grads_serializer_type
        )
        hidden_states_grads = hidden_states_grads_deserializer.deserialize(
            request_proto.serialized_hidden_states_grads
        )

        # Optionally deserialize mask.
        try:
            mask_serializer = bittensor.serializer(
                serializer_type=request_proto.mask_serializer_type
            )
            mask = mask_serializer.serialize(request_proto.serialized_mask)
        except:
            mask = None

        # If the mask is not none, we need to expand the hidden states to the proper size.
        if mask is not None:
            # From the encode_forward_response function the forward_response_tensor is [ len(mask), net_dim ]
            # a set of rows from the stacked_forward_response_tensor = [ bs * seq, net_dim ]
            # We will load these rows into a destination tensor = [bs, seq, net_dim]
            hidden_states_destination = torch.zeros(
                [mask.size(0) * mask.size(1), bittensor.__network_dim__]
            )
            hidden_states_grads_destination = torch.zeros(
                [mask.size(0) * mask.size(1), bittensor.__network_dim__]
            )

            # Iterate through the mask and fill the destination tensor
            # with the hidden states from the forward call.
            counter = 0
            for i, not_masked in enumerate(mask.reshape(-1)):
                if not_masked:
                    hidden_states_destination[i, :] = hidden_states[counter, :]
                    hidden_states_grads_destination[i, :] = hidden_states_grads[counter, :]
                    counter += 1

            # Reshape the destination tensor to the proper expanded size.
            hidden_states = hidden_states_destination.reshape(
                (mask.size(0), mask.size(1), bittensor.__network_dim__)
            )
            hidden_states_grads = hidden_states_grads_destination.reshape(
                (mask.size(0), mask.size(1), bittensor.__network_dim__)
            )

        # Return backward call.
        return bittensor.TextCausalLMNextBackwardCall(
            mask=mask,
            text_inputs=text_inputs,
            hidden_states=hidden_states,
            hidden_states_grads=hidden_states_grads,
            mask_serializer_type=request_proto.mask_serializer_type,
            text_inputs_serializer_type=request_proto.text_inputs_serializer_type,
            hidden_states_serializer_type=request_proto.hidden_states_serializer_type,
            hidden_states_grads_serializer_type=request_proto.hidden_states_grads_serializer_type,
        )
