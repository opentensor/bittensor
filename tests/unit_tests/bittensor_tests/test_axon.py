# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2022 Opentensor Foundation

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
import unittest
import unittest.mock as mock
import uuid

import grpc
import pytest
import torch

import bittensor
from bittensor.utils.test_utils import get_random_unused_port
import concurrent

from concurrent.futures import ThreadPoolExecutor

wallet = bittensor.wallet.mock()
axon = bittensor.axon( netuid = -1, wallet = wallet)

sender_wallet = bittensor.wallet.mock()

def gen_nonce():
    return f"{time.monotonic_ns()}"


def sign_v2(sender_wallet, receiver_wallet):
    nonce, receptor_uid = gen_nonce(), str(uuid.uuid1())
    sender_hotkey = sender_wallet.hotkey.ss58_address
    receiver_hotkey = receiver_wallet.hotkey.ss58_address
    message = f"{nonce}.{sender_hotkey}.{receiver_hotkey}.{receptor_uid}"
    signature = f"0x{sender_wallet.hotkey.sign(message).hex()}"
    return ".".join([nonce, sender_hotkey, signature, receptor_uid])

def sign(sender_wallet, receiver_wallet, receiver_version):
    
    return sign_v2(sender_wallet, receiver_wallet)

def test_sign_v2():
    sign_v2(sender_wallet, wallet)

def test_forward_not_implemented():
    inputs_raw = torch.rand(3, 3)
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    synapses = [bittensor.synapse.TextLastHiddenState()]

    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        tensors=[inputs_serialized],
        synapses = [ syn.serialize_to_wire_proto() for syn in synapses ],
        hotkey = axon.wallet.hotkey.ss58_address,   
    )
    response, code, synapses = axon._forward( request )
    assert synapses[0].return_code == bittensor.proto.ReturnCode.NotImplemented

def test_forward_last_hidden_success():
    def forward( inputs_x: torch.FloatTensor, synapse , model_output = None):
        return None, dict(), torch.zeros( [inputs_x.shape[0], inputs_x.shape[1], bittensor.__network_dim__])
    axon.attach_synapse_callback( forward, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE)

    inputs_raw = torch.rand(3, 3)
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    synapses = [bittensor.synapse.TextLastHiddenState()]
    inputs_serialized = serializer.serialize(inputs_raw, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        tensors=[inputs_serialized],
        synapses = [ syn.serialize_to_wire_proto() for syn in synapses ],
        hotkey = axon.wallet.hotkey.ss58_address, 
    )
    response, code, synapses = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.Success
    assert synapses[0].return_code == bittensor.proto.ReturnCode.Success

def test_forward_causallm_success():
    def forward( inputs_x: torch.FloatTensor, synapse, model_output = None):
        return None, dict(), torch.zeros( [inputs_x.shape[0], inputs_x.shape[1], bittensor.__network_dim__])
    axon.attach_synapse_callback( forward, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM)

    inputs_raw = torch.rand(3, 3)
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    synapses = [bittensor.synapse.TextCausalLM()]
    inputs_serialized = serializer.serialize(inputs_raw, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        tensors=[inputs_serialized],
        synapses = [ syn.serialize_to_wire_proto() for syn in synapses ],
        hotkey = axon.wallet.hotkey.ss58_address,
    )
    response, code, synapses = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.Success

def test_forward_causallmnext_success():
    def forward(inputs_x: torch.FloatTensor, synapse, model_output=None):  # [batch_size, (topk + 1), max_len]
        return None, dict(), torch.zeros([inputs_x.shape[0], (synapse.topk + 1), 1 + 1])
    axon.attach_synapse_callback(forward, synapse_type=bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT)

    inputs_raw = torch.rand(3, 3)
    serializer = bittensor.serializer(serializer_type=bittensor.proto.Serializer.MSGPACK)
    synapses = [bittensor.synapse.TextCausalLMNext()]
    inputs_serialized = serializer.serialize(inputs_raw, from_type=bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        tensors=[inputs_serialized],
        synapses=[syn.serialize_to_wire_proto() for syn in synapses],
        hotkey=axon.wallet.hotkey.ss58_address,
    )
    response, code, synapses = axon._forward(request)
    assert code == bittensor.proto.ReturnCode.Success

def test_forward_seq_2_seq_success():
    def forward( inputs_x: torch.FloatTensor, synapse, model_output = None):
        return None, dict(), torch.zeros( [inputs_x.shape[0], synapse.num_to_generate])
    axon.attach_synapse_callback( forward, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ)

    inputs_raw = torch.rand(3, 3)
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    synapses = [bittensor.synapse.TextSeq2Seq()]
    inputs_serialized = serializer.serialize(inputs_raw, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        tensors=[inputs_serialized],
        synapses = [ syn.serialize_to_wire_proto() for syn in synapses ],
        hotkey = axon.wallet.hotkey.ss58_address,
    )
    response, code, synapses = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.Success

def test_forward_empty_request():
    inputs_raw = torch.rand(3, 3)
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, from_type = bittensor.proto.TensorType.TORCH)
  
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[]
    )
    response, code, synapses = axon._forward( request )
    assert code ==  bittensor.proto.ReturnCode.EmptyRequest

def test_forward_deserialization_error():
    x = dict()  # Not tensors that can be deserialized.
    synapses = [bittensor.synapse.TextLastHiddenState()]
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ x ],
        synapses= [ syn.serialize_to_wire_proto() for syn in synapses ]
    )
    response, code, synapses  = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.RequestDeserializationException

def test_forward_batch_shape_error():
    inputs_raw = torch.rand(0, 1)
    synapses = [bittensor.synapse.TextLastHiddenState()]
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized ],
        synapses= [ syn.serialize_to_wire_proto() for syn in synapses ]
    )
    response, code, synapses  = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_forward_seq_shape_error():
    inputs_raw = torch.rand(1, 0, 1)
    synapses = [bittensor.synapse.TextLastHiddenState()]
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized ],
        synapses= [ syn.serialize_to_wire_proto() for syn in synapses ]
    )
    response, code, synapses  = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_forward_last_hidden_shape_error():
    inputs_raw = torch.rand(1, 1, 1)
    synapses = [bittensor.synapse.TextLastHiddenState()]
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized ],
        synapses= [ syn.serialize_to_wire_proto() for syn in synapses ]
    )
    response, code, synapses  = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_forward_causallm_shape_error():
    inputs_raw = torch.rand(1, 1, 1)
    synapses = [bittensor.synapse.TextCausalLM()]
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized ],
        synapses= [ syn.serialize_to_wire_proto() for syn in synapses ]
    )
    response, code, synapses  = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_forward_causallmnext_shape_error():
    inputs_raw = torch.rand(1, 1, 1)
    synapses = [bittensor.synapse.TextCausalLMNext()]
    serializer = bittensor.serializer(serializer_type=bittensor.proto.Serializer.MSGPACK)
    inputs_serialized = serializer.serialize(inputs_raw, modality=bittensor.proto.Modality.TEXT,
                                             from_type=bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey=axon.wallet.hotkey.ss58_address,
        tensors=[inputs_serialized],
        synapses=[syn.serialize_to_wire_proto() for syn in synapses]
    )
    response, code, synapses = axon._forward(request)
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_forward_seq_2_seq_shape_error():
    inputs_raw = torch.rand(1, 1, 1)
    synapses = [bittensor.synapse.TextSeq2Seq()]
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized ],
        synapses= [ syn.serialize_to_wire_proto() for syn in synapses ]
    )
    response, code, synapses  = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException


def test_forward_deserialization_empty():
    def forward( inputs_x: torch.FloatTensor, synapse, model_output = None):
        return None, dict(), None
    axon.attach_synapse_callback( forward, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE)

    inputs_raw = torch.rand(3, 3)
    synapses = [bittensor.synapse.TextLastHiddenState()]
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, from_type = bittensor.proto.TensorType.TORCH)
  
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[inputs_serialized],
        synapses= [ syn.serialize_to_wire_proto() for syn in synapses ]
    )
    response, code, synapses = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.EmptyResponse

def test_forward_response_deserialization_error():
    def forward( inputs_x: torch.FloatTensor, synapse, model_output = None):
        return None, dict(), dict()
    axon.attach_synapse_callback( forward, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE)

    inputs_raw = torch.rand(3, 3)
    synapses = [bittensor.synapse.TextLastHiddenState()]
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[inputs_serialized],
        synapses= [ syn.serialize_to_wire_proto() for syn in synapses ]
    )

    def check(a, b, c):
        pass

    with mock.patch('bittensor.TextLastHiddenState.check_forward_response_tensor', new=check):
        response, code, synapses = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.ResponseSerializationException

def test_forward_last_hidden_state_exception():
    def forward( inputs_x: torch.FloatTensor , synapse , model_output = None):
        if inputs_x.size() == (1,1,1):
            return None
        else:
            raise Exception('Mock')
    axon.attach_synapse_callback( forward, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE)

    inputs_raw = torch.rand(3, 3)
    synapses = [bittensor.synapse.TextLastHiddenState()]
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        tensors=[inputs_serialized],
        hotkey= '123',
        synapses= [ syn.serialize_to_wire_proto() for syn in synapses ]
    )
    response, code, synapses = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.UnknownException

def test_forward_causal_lm_state_exception():
    def forward( inputs_x: torch.FloatTensor , synapse, model_output = None):
        if inputs_x.size() == (1,1,1):
            return None
        else:
            raise Exception('Mock')
    axon.attach_synapse_callback( forward, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM)

    inputs_raw = torch.rand(3, 3)
    synapses = [bittensor.synapse.TextCausalLM()]
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        tensors=[inputs_serialized],
        hotkey= '123',
        synapses= [ syn.serialize_to_wire_proto() for syn in synapses ]
    )
    response, code, synapses = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.UnknownException

def test_forward_causal_lm_next_state_exception():
    def forward(inputs_x: torch.FloatTensor, synapse, model_output=None):
        if inputs_x.size() == (1, 1, 1):
            return None
        else:
            raise Exception('Mock')
    axon.attach_synapse_callback(forward, synapse_type=bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT)

    inputs_raw = torch.rand(3, 3)
    synapses = [bittensor.synapse.TextCausalLMNext()]
    serializer = bittensor.serializer(serializer_type=bittensor.proto.Serializer.MSGPACK)
    inputs_serialized = serializer.serialize(inputs_raw, modality=bittensor.proto.Modality.TENSOR,
                                             from_type=bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        tensors=[inputs_serialized],
        hotkey='123',
        synapses=[syn.serialize_to_wire_proto() for syn in synapses]
    )
    response, code, synapses = axon._forward(request)
    assert code == bittensor.proto.ReturnCode.UnknownException

def test_forward_seq_2_seq_state_exception():
    def forward( inputs_x: torch.FloatTensor , synapse, model_output = None):
        if inputs_x.size() == (1,1,1):
            return None
        else:
            raise Exception('Mock')
    axon.attach_synapse_callback( forward, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ)

    inputs_raw = torch.rand(3, 3)
    synapses = [bittensor.synapse.TextSeq2Seq()]
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        tensors=[inputs_serialized],
        hotkey= '123',
        synapses= [ syn.serialize_to_wire_proto() for syn in synapses ]
    )
    response, code, synapses = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.UnknownException

def test_forward_seq_2_seq_success():
    def forward( inputs_x: torch.FloatTensor, synapse, model_output = None):
        return None, dict(), torch.zeros( [inputs_x.shape[0], synapse.num_to_generate])
    axon.attach_synapse_callback( forward, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ)

    inputs_raw = torch.rand(3, 3)
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    synapses = [bittensor.synapse.TextSeq2Seq()]
    inputs_serialized = serializer.serialize(inputs_raw, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        tensors=[inputs_serialized],
        synapses = [ syn.serialize_to_wire_proto() for syn in synapses ],
        hotkey = axon.wallet.hotkey.ss58_address,
    )
    response, code, synapses = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.Success

def test_forward_joint_success():
    def forward_generate( inputs_x: torch.FloatTensor , synapse, model_output = None):
        return None, None, torch.zeros( (inputs_x.shape[0], synapse.num_to_generate) )
    def forward_causal_lm( inputs_x: torch.FloatTensor , synapse, model_output = None):
        return None, None, torch.zeros(inputs_x.shape[0], inputs_x.shape[1], bittensor.__vocab_size__)
    def forward_causal_lm_next(inputs_x: torch.FloatTensor, synapse, model_output = None):
        return None, None, torch.zeros(inputs_x.shape[0], synapse.topk + 1, 1 + 1)
    def forward_hidden_state( inputs_x: torch.FloatTensor , synapse, model_output = None):
        return None, None, torch.zeros( inputs_x.shape[0], inputs_x.shape[1], bittensor.__network_dim__)

    axon.attach_synapse_callback( forward_generate, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ)
    axon.attach_synapse_callback( forward_causal_lm, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM)
    axon.attach_synapse_callback(forward_causal_lm_next, synapse_type=bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT)
    axon.attach_synapse_callback( forward_hidden_state, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE)

    inputs_raw = torch.rand(3, 3)
    synapses = [bittensor.synapse.TextCausalLM(), bittensor.synapse.TextCausalLMNext(),
                bittensor.synapse.TextLastHiddenState(), bittensor.synapse.TextSeq2Seq()]
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        tensors=[inputs_serialized] * len(synapses),
        hotkey= axon.wallet.hotkey.ss58_address,
        synapses= [ syn.serialize_to_wire_proto() for syn in synapses ]
    )
    response, code, synapses = axon._forward( request )
    assert [syn.return_code for syn in synapses] == [bittensor.proto.ReturnCode.Success] * len(synapses)

def test_forward_joint_missing_synapse():
    def forward_generate( inputs_x: torch.FloatTensor , synapse, model_output = None):
        return None, None, torch.zeros( (inputs_x.shape[0], synapse.num_to_generate) )
    def forward_causal_lm( inputs_x: torch.FloatTensor , synapse, model_output = None):
        return None, None, torch.zeros(inputs_x.shape[0], inputs_x.shape[1], bittensor.__vocab_size__)
    def forward_causal_lm_next(inputs_x: torch.FloatTensor, synapse, model_output = None):
        return None, None, torch.zeros(inputs_x.shape[0], synapse.topk + 1, 1 + 1)
    def forward_hidden_state( inputs_x: torch.FloatTensor , synapse, model_output = None):
        return None, None, torch.zeros( inputs_x.shape[0], inputs_x.shape[1], bittensor.__network_dim__)

    axon.attach_synapse_callback( forward_generate, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ)
    axon.attach_synapse_callback( forward_causal_lm, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM)
    axon.attach_synapse_callback(forward_causal_lm_next, synapse_type=bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT)
    axon.attach_synapse_callback( forward_hidden_state, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE)

    inputs_raw = torch.rand(3, 3)
    synapses = [bittensor.synapse.TextCausalLM(), bittensor.synapse.TextCausalLMNext(),
                bittensor.synapse.TextLastHiddenState(), bittensor.synapse.TextSeq2Seq()]
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        tensors=[inputs_serialized] * len(synapses),
        hotkey= axon.wallet.hotkey.ss58_address,
        synapses= [ syn.serialize_to_wire_proto() for syn in synapses ]
    )
    axon.attach_synapse_callback( None, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ)
    response, code, synapses = axon._forward( request )
    assert [syn.return_code for syn in synapses] == [bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.Success,
                                                     bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.NotImplemented]

    axon.attach_synapse_callback( None, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE)
    response, code, synapses = axon._forward( request )
    assert [syn.return_code for syn in synapses] == [bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.Success,
                                                     bittensor.proto.ReturnCode.NotImplemented, bittensor.proto.ReturnCode.NotImplemented]

    axon.attach_synapse_callback(None, synapse_type=bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT)
    response, code, synapses = axon._forward(request)
    assert [syn.return_code for syn in synapses] == [bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.NotImplemented,
                                                     bittensor.proto.ReturnCode.NotImplemented, bittensor.proto.ReturnCode.NotImplemented]

    axon.attach_synapse_callback( None, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM)
    response, code, synapses = axon._forward( request )
    assert [syn.return_code for syn in synapses] == [bittensor.proto.ReturnCode.NotImplemented] * len(synapses)

def test_forward_joint_faulty_synapse():
    def faulty( inputs_x: torch.FloatTensor , synapse, model_output = None):
        raise Exception
    def forward_causal_lm( inputs_x: torch.FloatTensor , synapse, model_output = None):
        return None, None, torch.zeros(inputs_x.shape[0], inputs_x.shape[1], bittensor.__vocab_size__)
    def forward_causal_lm_next(inputs_x: torch.FloatTensor, synapse, model_output = None):
        return None, None, torch.zeros(inputs_x.shape[0], synapse.topk + 1, 1 + 1)
    def forward_hidden_state( inputs_x: torch.FloatTensor , synapse, model_output = None):
        return None, None, torch.zeros( inputs_x.shape[0], inputs_x.shape[1], bittensor.__network_dim__)

    axon.attach_synapse_callback( forward_causal_lm, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM)
    axon.attach_synapse_callback( forward_hidden_state, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE)
    axon.attach_synapse_callback(forward_causal_lm_next, synapse_type=bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT)
    axon.attach_synapse_callback( faulty, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ)

    inputs_raw = torch.rand(3, 3)
    synapses = [bittensor.synapse.TextCausalLM(), bittensor.synapse.TextCausalLMNext(),
                bittensor.synapse.TextLastHiddenState(), bittensor.synapse.TextSeq2Seq()]
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        tensors=[inputs_serialized] * len(synapses),
        hotkey= axon.wallet.hotkey.ss58_address,
        synapses= [ syn.serialize_to_wire_proto() for syn in synapses ]
    )

    axon.attach_synapse_callback( faulty, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ)
    response, code, synapses = axon._forward( request )
    assert [syn.return_code for syn in synapses] == [bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.Success,
                                                     bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.UnknownException]

    axon.attach_synapse_callback( faulty, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE)
    response, code, synapses = axon._forward( request )
    assert [syn.return_code for syn in synapses] == [bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.Success,
                                                     bittensor.proto.ReturnCode.UnknownException, bittensor.proto.ReturnCode.UnknownException]

    axon.attach_synapse_callback(faulty, synapse_type=bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT)
    response, code, synapses = axon._forward(request)
    assert [syn.return_code for syn in synapses] == [bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.UnknownException,
                                                     bittensor.proto.ReturnCode.UnknownException, bittensor.proto.ReturnCode.UnknownException]

    axon.attach_synapse_callback( faulty, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM)
    response, code, synapses = axon._forward( request )
    assert [syn.return_code for syn in synapses] == [bittensor.proto.ReturnCode.UnknownException] * len(synapses)
    
def test_forward_timeout():
    def forward( inputs_x: torch.FloatTensor, synapses, hotkey):
        if inputs_x[0].size() == (3,3):
            return None
        else:
            raise concurrent.futures.TimeoutError('Timeout')

    axon.attach_forward_callback( forward)

    inputs_raw = torch.rand(1,1)
    synapses = [bittensor.synapse.TextLastHiddenState()]
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        tensors=[inputs_serialized],
        hotkey = axon.wallet.hotkey.ss58_address,
        synapses= [ syn.serialize_to_wire_proto() for syn in synapses ]
    )

    response, code, synapses = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.Timeout

def test_forward_unknown_error():
    def forward( inputs_x: torch.FloatTensor,modality, model_output = None):
        raise Exception('Unknown')

    with mock.patch.object(axon, 'forward_callback', new=forward):
        inputs_raw = torch.rand(3, 3)
        serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
        synapses = [bittensor.synapse.TextLastHiddenState()]
        inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
        request = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            tensors=[inputs_serialized],
            hotkey= '123',
            synapses= [ syn.serialize_to_wire_proto() for syn in synapses ]
        )

        response, code, synapses = axon._forward( request )
        assert code == bittensor.proto.ReturnCode.UnknownException

#--- backwards ---

def test_backward_invalid_request():
    inputs_raw = torch.rand(3, 3)
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, from_type = bittensor.proto.TensorType.TORCH)

    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[inputs_serialized]
    )
    response, code, synapses = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.InvalidRequest

def test_backward_deserialization_error():
    x = dict()  # Not tensors that can be deserialized.
    g = dict()
    synapses = [bittensor.synapse.TextLastHiddenState()]
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ x, g],
        synapses= [ syn.serialize_to_wire_proto() for syn in synapses ]
    )
    response, code, synapses  = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.RequestDeserializationException
    assert synapses[0].return_code == bittensor.proto.ReturnCode.RequestDeserializationException

def test_backward_last_hidden_shape_error():
    inputs_raw = torch.rand(1, 1, 1)
    grads_raw = torch.rand(1, 1, bittensor.__network_dim__)
    synapses = [bittensor.synapse.TextLastHiddenState()]
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized =  synapses[0].serialize_backward_request_gradient(inputs_raw, grads_raw)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized, grads_serialized],
        synapses= [ syn.serialize_to_wire_proto() for syn in synapses ]
    )
    response, code, synapses  = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException
    assert synapses[0].return_code == bittensor.proto.ReturnCode.RequestShapeException

def test_backward_causal_lm_shape_error():
    inputs_raw = torch.rand(1, 1, 1)
    grads_raw = torch.rand(1, 1, bittensor.__vocab_size__)
    synapses = [bittensor.synapse.TextCausalLM()]
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = synapses[0].serialize_backward_request_gradient(inputs_raw, grads_raw)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized, grads_serialized],
        synapses= [ syn.serialize_to_wire_proto() for syn in synapses ]
    )
    response, code, synapses  = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException
    assert synapses[0].return_code == bittensor.proto.ReturnCode.RequestShapeException


def test_backward_causal_lm_next_shape_error():
    synapses = [bittensor.synapse.TextCausalLMNext()]
    inputs_raw = torch.rand(1, 1, 1)
    grads_raw = torch.rand(1, synapses[0].topk + 1, 1 + 1)
    serializer = bittensor.serializer(serializer_type=bittensor.proto.Serializer.MSGPACK)
    inputs_serialized = serializer.serialize(inputs_raw, from_type=bittensor.proto.TensorType.TORCH)
    grads_serialized = synapses[0].serialize_backward_request_gradient(inputs_raw, grads_raw)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey=axon.wallet.hotkey.ss58_address,
        tensors=[inputs_serialized, grads_serialized],
        synapses=[syn.serialize_to_wire_proto() for syn in synapses]
    )
    response, code, synapses = axon._backward(request)
    assert code == bittensor.proto.ReturnCode.RequestShapeException
    assert synapses[0].return_code == bittensor.proto.ReturnCode.RequestShapeException


def test_backward_seq_2_seq_shape_error():
    inputs_raw = torch.rand(1, 1, 1)
    grads_raw = torch.tensor([])
    synapses = [bittensor.synapse.TextSeq2Seq()]
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = synapses[0].serialize_backward_request_gradient(inputs_raw, grads_raw)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized, grads_serialized],
        synapses= [ syn.serialize_to_wire_proto() for syn in synapses ]
    )
    response, code, synapses  = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException
    assert synapses[0].return_code == bittensor.proto.ReturnCode.RequestShapeException


def test_backward_grads_shape_error():
    inputs_raw = torch.rand(1, 1)
    grads_raw = torch.rand(1, 1, 1, bittensor.__network_dim__)
    synapses = [bittensor.synapse.TextLastHiddenState()]
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = synapses[0].serialize_forward_request_tensor(inputs_raw)
    grads_serialized = serializer.serialize(grads_raw, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized, grads_serialized],
        synapses= [ syn.serialize_to_wire_proto() for syn in synapses ]
    )
    response, code, synapses  = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException


def test_backward_response_success_hidden():
    def backward( inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, synapses):
        return [], [1], ['success']

    axon.attach_backward_callback(backward)
    inputs_raw = torch.ones(1, 1)
    grads_raw = torch.zeros(1, 1, bittensor.__network_dim__)
    synapses = [bittensor.synapse.TextLastHiddenState()]
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized =  synapses[0].serialize_forward_request_tensor(inputs_raw)
    grads_serialized = synapses[0].serialize_backward_request_gradient(inputs_raw, grads_raw)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized, grads_serialized],
        synapses= [ syn.serialize_to_wire_proto() for syn in synapses ]
    )
    response, code, synapses = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.Success

def test_backward_response_success_causal_lm():
    def backward( inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, synapses):
        return [], [1], ['success']

    axon.attach_backward_callback( backward)
    inputs_raw = torch.ones(1, 1)
    grads_raw = torch.zeros(1, 1, bittensor.__vocab_size__)
    synapses = [bittensor.synapse.TextCausalLM()]
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = synapses[0].serialize_forward_request_tensor(inputs_raw)
    grads_serialized = synapses[0].serialize_backward_request_gradient(inputs_raw,grads_raw)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized, grads_serialized],
        synapses= [ syn.serialize_to_wire_proto() for syn in synapses ]
    )
    response, code, synapses = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.Success

def test_backward_response_success_causal_lm_next():
    def backward( inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, synapses):
        return [], [1], ['success']

    axon.attach_backward_callback( backward)
    synapses = [bittensor.synapse.TextCausalLMNext()]

    inputs_raw = torch.ones(1, 1)
    grads_raw = torch.zeros([1, (synapses[0].topk + 1), 1 + 1])

    inputs_serialized = synapses[0].serialize_forward_request_tensor(inputs_raw)
    grads_serialized = synapses[0].serialize_backward_request_gradient(inputs_raw, grads_raw)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey=axon.wallet.hotkey.ss58_address,
        tensors=[inputs_serialized, grads_serialized],
        synapses=[syn.serialize_to_wire_proto() for syn in synapses]
    )
    response, code, synapses = axon._backward(request)
    assert code == bittensor.proto.ReturnCode.Success

def test_backward_response_timeout():
    def backward( inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, synapses):
        raise concurrent.futures.TimeoutError('Timeout')

    axon.attach_backward_callback( backward)
    inputs_raw = torch.rand(2, 2)
    grads_raw = torch.rand(2, 2, bittensor.__network_dim__)
    synapses = [bittensor.synapse.TextLastHiddenState()]
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized =  synapses[0].serialize_forward_request_tensor(inputs_raw)
    grads_serialized =  synapses[0].serialize_backward_request_gradient(inputs_raw, grads_raw)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized, grads_serialized],
        synapses= [ syn.serialize_to_wire_proto() for syn in synapses ]
    )
    response, code, synapses = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.Timeout

def test_backward_response_exception():
    def backward( inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, synapses):
        raise Exception('Timeout')

    axon.attach_backward_callback( backward)
    inputs_raw = torch.rand(2, 2)
    synapses = [bittensor.synapse.TextLastHiddenState()]
    grads_raw = torch.rand(2, 2, bittensor.__network_dim__)
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized =  synapses[0].serialize_forward_request_tensor(inputs_raw)
    grads_serialized = synapses[0].serialize_backward_request_gradient(inputs_raw, grads_raw)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized, grads_serialized],
        synapses= [ syn.serialize_to_wire_proto() for syn in synapses ]
    )
    response, code, synapses = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.UnknownException

# -- axon priority:

def test_forward_tensor_success_priority():
    
    def priority(pubkey:str, request_type:str, inputs_x):
        return 100

    axon = bittensor.axon( netuid = -1, wallet = wallet, priority= priority)

    def forward( inputs_x: torch.FloatTensor, synapses , model_output = None):
        return None, dict(), torch.zeros( [inputs_x.shape[0], inputs_x.shape[1], bittensor.__network_dim__])
    axon.attach_synapse_callback( forward, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE)
    inputs_raw = torch.rand(3, 3)
    synapses = [bittensor.synapse.TextLastHiddenState()]
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized =  synapses[0].serialize_forward_request_tensor(inputs_raw)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        tensors=[inputs_serialized],
        synapses= [ syn.serialize_to_wire_proto() for syn in synapses ],
        hotkey = axon.wallet.hotkey.ss58_address,
    )
    response, code, synapses = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.Success


def test_forward_priority_timeout():
    def priority(pubkey:str, request_type:str, inputs_x):
        return 100

    def forward( inputs_x: torch.FloatTensor, synapses, hotkey):
        time.sleep(15)

    axon = bittensor.axon( netuid = -1, wallet = wallet, priority= priority, forward_timeout = 5)
    axon.attach_forward_callback(forward)

    inputs_raw = torch.rand(1,1)
    synapses = [bittensor.synapse.TextLastHiddenState()]
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        tensors=[inputs_serialized],
        hotkey = axon.wallet.hotkey.ss58_address,
        synapses= [ syn.serialize_to_wire_proto() for syn in synapses ]
    )

    response, code, synapses = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.Timeout

    axon.stop()

def test_forward_priority_2nd_request_timeout():
    def priority(pubkey:str, request_type:str, inputs_x):
        return 100

    axon = bittensor.axon( netuid = -1, wallet = wallet, priority= priority, priority_threadpool = bittensor.prioritythreadpool(max_workers = 1))

    def forward( inputs_x: torch.FloatTensor, synapses , model_output = None):
        time.sleep(2)
        return None, dict(), torch.zeros( [inputs_x.shape[0], inputs_x.shape[1], bittensor.__network_dim__])
    
    axon.attach_synapse_callback( forward, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE)
    inputs_raw = torch.rand(3, 3)
    synapses = [bittensor.synapse.TextLastHiddenState()]
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized =  synapses[0].serialize_forward_request_tensor(inputs_raw)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        tensors=[inputs_serialized],
        synapses= [ syn.serialize_to_wire_proto() for syn in synapses ],
        hotkey = axon.wallet.hotkey.ss58_address,
    )
    executor = ThreadPoolExecutor(2)
    future = executor.submit(axon._forward, (request))
    future2 = executor.submit(axon._forward, (request))
    
    try: 
        future2.result(timeout = 1)
    except concurrent.futures.TimeoutError:
        pass
    else:
        raise AssertionError('Expected to Timeout')
    
    _, code, _ = future.result()
    assert code == bittensor.proto.ReturnCode.Success

    axon.stop()

def test_backward_response_success_text_priority():
        
    def priority(pubkey:str, request_type:str, inputs_x):
        return 100

    axon = bittensor.axon( netuid = -1, wallet = wallet, priority= priority)

    def forward( inputs_x: torch.FloatTensor, synapses, model_output = None):
        return None, dict(), torch.zeros( [inputs_x.shape[0], inputs_x.shape[1], bittensor.__network_dim__])
    axon.attach_synapse_callback( forward, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE)

    inputs_raw = torch.ones((1, 1))
    grads_raw = torch.zeros((1, 1, bittensor.__network_dim__))
    synapses = [bittensor.synapse.TextLastHiddenState()]

    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = synapses[0].serialize_forward_request_tensor(inputs_raw)
    grads_serialized = synapses[0].serialize_backward_request_gradient(inputs_raw, grads_raw)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.ss58_address,
        tensors=[ inputs_serialized, grads_serialized],
        synapses= [ syn.serialize_to_wire_proto() for syn in synapses ]
    )
    response, code, synapses = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.Success


def run_test_grpc_forward_works(receiver_version):
    def forward( inputs_x:torch.FloatTensor, synapse , model_output = None):
        return None, dict(), torch.zeros( [3, 3, bittensor.__network_dim__])
    axon = bittensor.axon (
        netuid = -1,
        port = 7084,
        ip = '127.0.0.1',
        wallet = wallet,
    )
    axon.attach_synapse_callback( forward, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE)
    axon.start()

    channel = grpc.insecure_channel(
            '127.0.0.1:7084',
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1)])
    stub = bittensor.grpc.BittensorStub( channel )

    inputs_raw = torch.rand(3, 3)
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    synapses = [bittensor.synapse.TextLastHiddenState()]

    inputs_serialized = synapses[0].serialize_forward_request_tensor(inputs_raw)

    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        hotkey = sender_wallet.hotkey.ss58_address,
        tensors = [inputs_serialized],
        synapses = [ syn.serialize_to_wire_proto() for syn in synapses ]
    )
    response = stub.Forward(request,
                            metadata = (
                                        ('rpc-auth-header','Bittensor'),
                                        ('bittensor-signature',sign(sender_wallet, wallet, receiver_version)),
                                        ('bittensor-version',str(bittensor.__version_as_int__)),
                                        ))

    outputs = synapses[0].deserialize_forward_response_proto (inputs_raw, response.tensors[0])
    assert outputs.size(2) ==  bittensor.__network_dim__
    assert response.return_code == bittensor.proto.ReturnCode.Success
    axon.stop()

def test_grpc_forward_works():
    for receiver_version in [bittensor.__new_signature_version__, bittensor.__version_as_int__]:
        run_test_grpc_forward_works(receiver_version)

def run_test_grpc_backward_works(receiver_version):
    def backward( inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, synapses):
        return [], [1], ['success']

    axon = bittensor.axon (
        netuid = -1,
        port = 7086,
        ip = '127.0.0.1',
        wallet = wallet,
    )
    axon.attach_backward_callback( backward)
    axon.start()

    channel = grpc.insecure_channel(
            '127.0.0.1:7086',
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1)])
    stub = bittensor.grpc.BittensorStub( channel )
    synapses = [bittensor.synapse.TextLastHiddenState()]
    inputs_raw = torch.rand(3, 3)
    grads_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = synapses[0].serialize_forward_request_tensor(inputs_raw)
    grads_serialized = synapses[0].serialize_backward_request_gradient(inputs_raw, grads_raw)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        hotkey = sender_wallet.hotkey.ss58_address,
        tensors = [inputs_serialized, grads_serialized],
        synapses = [ syn.serialize_to_wire_proto() for syn in synapses ]
    )
    response = stub.Backward(request,
                             metadata = (
                                    ('rpc-auth-header','Bittensor'),
                                    ('bittensor-signature',sign(sender_wallet, wallet, receiver_version)),
                                    ('bittensor-version',str(bittensor.__version_as_int__)),
                                    ))
    assert response.return_code == bittensor.proto.ReturnCode.Success
    axon.stop()

def test_grpc_backward_works():
    for receiver_version in [bittensor.__new_signature_version__, bittensor.__version_as_int__]:
        run_test_grpc_backward_works(receiver_version)

def test_grpc_forward_fails():
    def forward( inputs_x:torch.FloatTensor, synapse, model_output = None):
        return None, dict(), torch.zeros( [3, 3, bittensor.__network_dim__])
    axon = bittensor.axon (
        netuid = -1,
        port = 7084,
        ip = '127.0.0.1',
        wallet = wallet,
    )
    axon.attach_synapse_callback( forward, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE)
    axon.start()

    channel = grpc.insecure_channel(
            '127.0.0.1:7084',
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1)])
    stub = bittensor.grpc.BittensorStub( channel )

    inputs_raw = torch.rand(3, 3)
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    synapses = [bittensor.synapse.TextLastHiddenState()]

    inputs_serialized = synapses[0].serialize_forward_request_tensor(inputs_raw)

    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        hotkey = '1092310312914',
        tensors = [inputs_serialized],
        synapses = [ syn.serialize_to_wire_proto() for syn in synapses ]
    )
    try:
        response = stub.Forward(request)
    except grpc.RpcError as rpc_error_call:
        grpc_code = rpc_error_call.code()
        assert grpc_code == grpc.StatusCode.UNAUTHENTICATED

    axon.stop()

def test_grpc_backward_fails():
    def forward( inputs_x:torch.FloatTensor, synapse):
        return torch.zeros( [3, 3, bittensor.__network_dim__], requires_grad=True)

    axon = bittensor.axon (
        netuid = -1,
        port = 7086,
        ip = '127.0.0.1',
        wallet = wallet,
    )
    axon.attach_synapse_callback( forward, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE)
    axon.start()

    channel = grpc.insecure_channel(
            '127.0.0.1:7086',
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1)])
    stub = bittensor.grpc.BittensorStub( channel )
    synapses = [bittensor.synapse.TextLastHiddenState()]
    inputs_raw = torch.rand(3, 3)
    grads_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = synapses[0].serialize_forward_request_tensor(inputs_raw)
    grads_serialized = synapses[0].serialize_backward_request_gradient(inputs_raw, grads_raw)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        hotkey = '1092310312914',
        tensors = [inputs_serialized, grads_serialized],
        synapses = [ syn.serialize_to_wire_proto() for syn in synapses ]
    )
    
    try:
        response = stub.Backward(request)
    except grpc.RpcError as rpc_error_call:
        grpc_code = rpc_error_call.code()
        assert grpc_code == grpc.StatusCode.UNAUTHENTICATED

    axon.stop()

def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        val = s.connect_ex(('localhost', port))
        if val == 0:
            return True
        else:
            return False

def test_axon_is_destroyed():
    port = get_random_unused_port()
    assert is_port_in_use( port ) == False
    axon = bittensor.axon ( netuid = -1, port = port )
    assert is_port_in_use( port ) == True
    axon.start()
    assert is_port_in_use( port ) == True
    axon.stop()
    assert is_port_in_use( port ) == False
    axon.__del__()
    assert is_port_in_use( port ) == False

    port = get_random_unused_port()
    assert is_port_in_use( port ) == False
    axon2 = bittensor.axon ( netuid = -1, port = port )
    assert is_port_in_use( port ) == True
    axon2.start()
    assert is_port_in_use( port ) == True
    axon2.__del__()
    assert is_port_in_use( port ) == False

    port_3 = get_random_unused_port()
    assert is_port_in_use( port_3 ) == False
    axonA = bittensor.axon ( netuid = -1, port = port_3 )
    assert is_port_in_use( port_3 ) == True
    axonB = bittensor.axon ( netuid = -1, port = port_3 )
    assert axonA.server != axonB.server
    assert is_port_in_use( port_3 ) == True
    axonA.start()
    assert is_port_in_use( port_3 ) == True
    axonB.start()
    assert is_port_in_use( port_3 ) == True
    axonA.__del__()
    assert is_port_in_use( port ) == False
    axonB.__del__()
    assert is_port_in_use( port ) == False

# test external axon args
class TestExternalAxon(unittest.TestCase):
    """
    Tests the external axon config flags 
    `--axon.external_port` and `--axon.external_ip`
    Need to verify the external config is used when broadcasting to the network
    and the internal config is used when creating the grpc server

    Also test the default behaviour when no external axon config is provided
    (should use the internal axon config, like usual)
    """

    def test_external_ip_not_set_dont_use_internal_ip(self):
        # Verify that not setting the external ip arg will NOT default to the internal axon ip
        mock_add_insecure_port = mock.MagicMock(return_value=None)
        mock_server = mock.MagicMock(
            add_insecure_port=mock_add_insecure_port
        )

        mock_config = bittensor.axon.config()

        axon = bittensor.axon ( netuid = -1, ip = 'fake_ip', server=mock_server, config=mock_config )
        assert axon.external_ip != axon.ip # should be different
        assert axon.external_ip is None # should be None

    def test_external_port_not_set_use_internal_port(self):
        # Verify that not setting the external port arg will default to the internal axon port
        mock_config = bittensor.axon.config()

        mock_wallet = mock.MagicMock(
            hotkey = mock.MagicMock(
                ss58_address = 'fake_hotkey_address',
                spec = bittensor.Keypair
            ),
            spec = bittensor.Wallet
        )

        with mock.patch('bittensor.wallet') as mock_create_wallet:
            mock_create_wallet.return_value = mock_wallet

            axon = bittensor.axon ( netuid = -1, port = 1234, config=mock_config )
            assert axon.external_port == axon.port

    def test_external_port_set_full_address_internal(self):
        internal_port = 1234
        external_port = 5678

        mock_add_insecure_port = mock.MagicMock(return_value=None)
        mock_server = mock.MagicMock(
            add_insecure_port=mock_add_insecure_port
        )
        
        mock_config = bittensor.axon.config()

        _ = bittensor.axon( netuid = -1, port=internal_port, external_port=external_port, server=mock_server, config=mock_config )
        
        mock_add_insecure_port.assert_called_once()
        args, _ = mock_add_insecure_port.call_args
        full_address0 = args[0]

        assert f'{internal_port}' in full_address0 and f':{external_port}' not in full_address0

        mock_add_insecure_port.reset_mock()

        # Test using config
        mock_config = bittensor.axon.config()

        mock_config.axon.port = internal_port
        mock_config.axon.external_port = external_port

        _ = bittensor.axon( netuid = -1, config=mock_config, server=mock_server )
        
        mock_add_insecure_port.assert_called_once()
        args, _ = mock_add_insecure_port.call_args
        full_address0 = args[0]

        assert f'{internal_port}' in full_address0, f'{internal_port} was not found in {full_address0}'
        assert f':{external_port}' not in full_address0, f':{external_port} was found in {full_address0}'

    def test_external_ip_set_full_address_internal(self):
        internal_ip = 'fake_ip_internal'
        external_ip = 'fake_ip_external'

        mock_add_insecure_port = mock.MagicMock(return_value=None)
        mock_server = mock.MagicMock(
            add_insecure_port=mock_add_insecure_port
        )

        mock_config = bittensor.axon.config()

        _ = bittensor.axon( netuid = -1, ip=internal_ip, external_ip=external_ip, server=mock_server, config=mock_config )
        
        mock_add_insecure_port.assert_called_once()
        args, _ = mock_add_insecure_port.call_args
        full_address0 = args[0]

        assert f'{internal_ip}' in full_address0 and f'{external_ip}' not in full_address0

        mock_add_insecure_port.reset_mock()

        # Test using config
        mock_config = bittensor.axon.config()
        mock_config.axon.external_ip = external_ip
        mock_config.axon.ip = internal_ip

        _ = bittensor.axon( netuid = -1, config=mock_config, server=mock_server )
        
        mock_add_insecure_port.assert_called_once()
        args, _ = mock_add_insecure_port.call_args
        full_address0 = args[0]

        assert f'{internal_ip}' in full_address0, f'{internal_ip} was not found in {full_address0}'
        assert f'{external_ip}' not in full_address0, f'{external_ip} was found in {full_address0}'

    def test_external_ip_port_set_full_address_internal(self):
        internal_ip = 'fake_ip_internal'
        external_ip = 'fake_ip_external'
        internal_port = 1234
        external_port = 5678

        mock_add_insecure_port = mock.MagicMock(return_value=None)
        mock_server = mock.MagicMock(
            add_insecure_port=mock_add_insecure_port
        )

        mock_config = bittensor.axon.config()

        _ = bittensor.axon( netuid = -1, ip=internal_ip, external_ip=external_ip, port=internal_port, external_port=external_port, server=mock_server, config=mock_config )
        
        mock_add_insecure_port.assert_called_once()
        args, _ = mock_add_insecure_port.call_args
        full_address0 = args[0]

        assert f'{internal_ip}:{internal_port}' == full_address0 and f'{external_ip}:{external_port}' != full_address0

        mock_add_insecure_port.reset_mock()

        # Test using config
        mock_config = bittensor.axon.config()

        mock_config.axon.ip = internal_ip
        mock_config.axon.external_ip = external_ip
        mock_config.axon.port = internal_port
        mock_config.axon.external_port = external_port

        _ = bittensor.axon( netuid = -1, config=mock_config, server=mock_server )
        
        mock_add_insecure_port.assert_called_once()
        args, _ = mock_add_insecure_port.call_args
        full_address1 = args[0]

        assert f'{internal_ip}:{internal_port}' == full_address1, f'{internal_ip}:{internal_port} is not eq to {full_address1}'
        assert f'{external_ip}:{external_port}' != full_address1, f'{external_ip}:{external_port} is eq to {full_address1}'


if __name__ == "__main__":
    test_grpc_forward_fails()
    # test_forward_joint_success()
    # test_forward_joint_missing_synapse()
    # test_forward_priority_timeout()
    #test_forward_priority_2nd_request_timeout()
    # test_forward_joint_faulty_synapse()