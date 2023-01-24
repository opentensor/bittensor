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

import bittensor
import torch
from .synapse_impl import Synapse
from bittensor.utils.tokenizer_utils import compact_topk_token_phrases, unravel_topk_token_phrases


class TextCausalLMNext(Synapse):
    """ TextCausalLMNext Synapse type for next token prediction from language models.
    """
    synapse_type: bittensor.proto.Synapse.SynapseType = bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT

    def __init__(
            self,
            topk: int = 4096,
            forward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
            forward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
            backward_request_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
            backward_response_serializer_type: 'bittensor.proto.Serializer.Type' = bittensor.proto.Serializer.MSGPACK,
    ):
        """ TextCausalLMNext Synapse initializer.
        Args:
            topk (:obj:`int`):
                Specifies the number of topk server token phrases to return.
            forward_request_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                Serializer used to pack torch tensors on forward request.
            forward_response_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                Serializer used to pack torch tensors on forward response.
            backward_request_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                Serializer used to pack torch tensors on forward request.
            backward_response_serializer_type (:obj:`bittensor.proto.Serializer.Type` of shape :obj:`(1)`, `optional`, :default: `bittensor.proto.Serializer.MSGPACK`):
                Serializer used to pack torch tensors on backward response.
        Returns:
            TextCausalLMNext (:obj:`TextCausalLMNext`, `required`):
                TextCausalLMNext instance adapter class.
    """
        super().__init__(
            forward_request_serializer_type,
            forward_response_serializer_type,
            backward_request_serializer_type,
            backward_response_serializer_type
        )
        self.topk = topk
        self.synapse_type = TextCausalLMNext.synapse_type

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return "TextCausalLMNext"

    @staticmethod
    def deserialize_from_instance_proto(instance_proto: bittensor.proto.Synapse) -> 'TextCausalLMNext':
        return TextCausalLMNext(
            topk=instance_proto.topk,
            forward_request_serializer_type=instance_proto.forward_request_serializer_type,
            forward_response_serializer_type=instance_proto.forward_response_serializer_type,
            backward_request_serializer_type=instance_proto.backward_request_serializer_type,
            backward_response_serializer_type=instance_proto.backward_response_serializer_type,
        )

    @staticmethod
    def deserialize_from_wire_proto(wire_proto: bittensor.proto.Synapse) -> 'TextCausalLMNext':
        instance_proto = bittensor.proto.Synapse.TextCausalLMNext()
        instance_proto.ParseFromString(wire_proto.synapse_data)
        return TextCausalLMNext.deserialize_from_instance_proto(instance_proto)

    def serialize_to_instance_proto(self) -> 'bittensor.proto.Synapse.TextCausalLMNext':
        return bittensor.proto.Synapse.TextCausalLMNext(
            topk=self.topk,
            forward_request_serializer_type=self.forward_request_serializer_type,
            forward_response_serializer_type=self.forward_response_serializer_type,
            backward_request_serializer_type=self.backward_request_serializer_type,
            backward_response_serializer_type=self.backward_response_serializer_type,
        )

    def serialize_to_wire_proto(self, code: 'bittensor.proto.ReturnCode' = 0,
                                message: str = '') -> bittensor.proto.Synapse:
        return bittensor.proto.Synapse(
            synapse_data=self.serialize_to_instance_proto().SerializeToString(),
            synapse_type=TextCausalLMNext.synapse_type,
            return_code=code,
            message=message
        )

    def check_forward_request_tensor(self, forward_request_tensor):
        # forward_request_tensor: [batch_size, sequence_len]
        if (
                len(forward_request_tensor.shape) != 2 or
                forward_request_tensor.shape[0] == 0 or
                forward_request_tensor.shape[1] == 0
        ):
            raise ValueError(f"forward_request_tensor.shape must be in [-1, -1], "
                             f"got: {list(forward_request_tensor.shape)} for synapse: {self}")

    def check_forward_response_tensor(self, forward_request_tensor, forward_response_tensor):
        # forward_request_tensor: [batch_size, sequence_len]
        # forward_response_tensor: [ >= batch_size * (2 * topk + 1)]
        if forward_response_tensor is None:
            raise ValueError("Empty Response")

        if (
                len(forward_response_tensor.shape) != 1 or
                forward_response_tensor.size(0) < forward_request_tensor.shape[0] * (2 * self.topk + 1)
        ):
            raise ValueError(f"forward_response_tensor.shape must be in "
                             f"[>={forward_request_tensor.shape[0]} x (2 x {self.topk} + 1)], "
                             f"got: {forward_response_tensor.size(0)} for synapse: {self}")

        atol = 1e-6  # absolute tolerance
        if (forward_response_tensor < -atol).any():
            raise ValueError("forward_response_tensor values below tolerance.")

    def check_backward_request_gradient(self, forward_request_tensor, backward_request_gradient):
        # forward_request_tensor: [batch_size, sequence_len]
        # backward_request_gradient: [batch_size, (topk + 1), max_len]
        if (
                len(backward_request_gradient.shape) != 3 or
                backward_request_gradient.size(0) != forward_request_tensor.shape[0] or
                backward_request_gradient.size(1) != (self.topk + 1)
        ):
            raise ValueError(f"backward_request_gradient.shape must be in "
                             f"[{forward_request_tensor.shape[0]}, ({self.topk} + 1), max_len], "
                             f"got: {backward_request_gradient.shape} for synapse: {self}")

    def encode_forward_request_tensor(self, forward_request_tensor: torch.Tensor) -> torch.Tensor:
        return forward_request_tensor

    def decode_forward_request_tensor(self, forward_request_tensor: torch.Tensor) -> torch.Tensor:
        return forward_request_tensor

    def encode_forward_response_tensor(self, forward_response_tensor: torch.Tensor) -> torch.Tensor:
        """ Compact [batch_size, (topk + 1), max_len] topk std_token_phrases to [ >= batch_size * (2 * topk + 1)]. """
        compact_topk = compact_topk_token_phrases(forward_response_tensor)
        # compact_topk: [sum_b(sum_k(len(phrase_k) + 1)_b)] Compacted 1-D tensor >= batch_size * (2 * topk + 1)
        return compact_topk

    def decode_forward_response_tensor(self, forward_request_tensor: torch.Tensor,
                                       forward_response_tensor: torch.Tensor) -> torch.Tensor:
        """ Unravel [ >= batch_size * (2 * topk + 1)] into [batch_size, (topk + 1), max_len] topk std_token_phrases. """
        topk_tensor = unravel_topk_token_phrases(forward_response_tensor, topk=self.topk)
        return topk_tensor  # [batch_size, (topk + 1), max_len]

    def encode_backward_response_gradient(self, backward_request_gradient: torch.Tensor) -> torch.Tensor:
        return backward_request_gradient

    def decode_backward_response_gradient(self, backward_request_gradient: torch.Tensor) -> torch.Tensor:
        return backward_request_gradient

    def encode_backward_request_gradient(self, backward_response_gradient: torch.Tensor) -> torch.Tensor:
        """ Compact gradients of [batch_size, (topk + 1), max_len] to [2 + batch_size * (topk + 1)]. """
        batch_size, topk_p1, max_len = backward_response_gradient.shape
        dims = torch.tensor([batch_size, max_len]).to(backward_response_gradient.device)
        prob_grads = backward_response_gradient[:, :, 0]  # [batch_size, topk + 1] first column w/ prob grads
        encoded_gradient = torch.hstack((dims, prob_grads.flatten()))  # [2 + batch_size * (topk + 1)]
        return encoded_gradient  # [2 + batch_size * (topk + 1)]

    def decode_backward_request_gradient(self, backward_response_gradient: torch.Tensor) -> torch.Tensor:
        """ Restructure [2 + batch_size * (topk + 1)] prob grads into [batch_size, (topk + 1), max_len]. """
        batch_size = int(backward_response_gradient[0].item())
        max_len = int(backward_response_gradient[1].item())
        decoded_gradient = torch.zeros((batch_size, self.topk + 1, max_len)).to(backward_response_gradient.device)
        decoded_gradient[:, :, 0] = backward_response_gradient[2:].reshape(batch_size, self.topk + 1)
        return decoded_gradient  # [batch_size, (topk + 1), max_len]

    def nill_forward_response_tensor(self, forward_request_tensor: torch.Tensor,
                                     encoded=False, ignore_index=-100) -> torch.Tensor:
        if forward_request_tensor.dim() == 0 or forward_request_tensor.shape[0] == 0:
            return torch.tensor([])

        forward_response_tensor = torch.zeros(forward_request_tensor.shape[0], (self.topk + 1), 1 + 1)
        forward_response_tensor[:, :, 1] = 2  # set 2 <= token_ids to preserve 0 <= probs <= 1 in column 0
        forward_response_tensor[:, self.topk::(self.topk + 1), 1] = ignore_index  # add ignore_index padding after floor_prob

        if encoded:
            return self.encode_forward_response_tensor(forward_response_tensor)

        return forward_response_tensor

    def nill_backward_response_tensor(self, forward_request_tensor: torch.Tensor) -> torch.Tensor:
        if forward_request_tensor.dim() == 0 or forward_request_tensor.shape[0] == 0:
            return torch.tensor([])
        return torch.zeros((forward_request_tensor.shape[0], (self.topk + 1), 1 + 1), dtype=torch.float32)
