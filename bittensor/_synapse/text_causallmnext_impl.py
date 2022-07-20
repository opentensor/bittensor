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

    def check_backward_request_gradient(self, forward_request_tensor, backward_request_gradient):
        # forward_request_tensor: [batch_size, sequence_len]
        # backward_request_gradient: [ >= batch_size * (2 * topk + 1)]
        if (
                len(backward_request_gradient.shape) != 1 or
                backward_request_gradient.size(0) < forward_request_tensor.shape[0] * (2 * self.topk + 1)
        ):
            raise ValueError(f"backward_request_gradient.shape must be in "
                             f"[>={forward_request_tensor.shape[0]} x (2 x {self.topk} + 1)], "
                             f"got: {backward_request_gradient.size(0)} for synapse: {self}")

    def encode_forward_request_tensor(self, forward_request_tensor: torch.Tensor) -> torch.Tensor:
        return forward_request_tensor

    def decode_forward_request_tensor(self, forward_request_tensor: torch.Tensor) -> torch.Tensor:
        return forward_request_tensor

    def encode_forward_response_tensor(self, forward_response_tensor: torch.Tensor) -> torch.Tensor:
        return forward_response_tensor

    def decode_forward_response_tensor(self, forward_response_tensor: torch.Tensor) -> torch.Tensor:
        return forward_response_tensor

    def encode_backward_response_gradient(self, backward_request_gradient: torch.Tensor) -> torch.Tensor:
        return backward_request_gradient

    def decode_backward_response_gradient(self, backward_request_gradient: torch.Tensor) -> torch.Tensor:
        return backward_request_gradient

    def encode_backward_request_gradient(self, backward_response_gradient: torch.Tensor) -> torch.Tensor:
        return backward_response_gradient

    def decode_backward_request_gradient(self, backward_response_gradient: torch.Tensor) -> torch.Tensor:
        return backward_response_gradient

    def nill_forward_response_tensor(self, forward_request_tensor: torch.Tensor) -> torch.Tensor:
        try:
            return torch.zeros((forward_request_tensor.shape[0] * (2 * self.topk + 1)), dtype=torch.float32)
        except:
            return torch.tensor([])

    def nill_backward_response_tensor(self, forward_request_tensor: torch.Tensor) -> torch.Tensor:
        try:
            return torch.zeros((forward_request_tensor.shape[0] * (2 * self.topk + 1)), dtype=torch.float32)
        except:
            return torch.tensor([])
