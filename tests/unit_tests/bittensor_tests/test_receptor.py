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

from sys import version
import grpc
import torch
import bittensor
from unittest.mock import MagicMock
import unittest.mock as mock
import asyncio
from types import SimpleNamespace
import time as clock
from bittensor.utils.test_utils import get_random_unused_port

wallet = bittensor.wallet.mock()

endpoint = bittensor.endpoint(
    version = bittensor.__version_as_int__,
    uid = 0,
    ip = '0.0.0.0',
    ip_type = 4,
    port = get_random_unused_port(),
    hotkey = wallet.hotkey.ss58_address,
    coldkey = wallet.coldkey.ss58_address,
    protocol = 0
)
receptor = bittensor.receptor ( 
    endpoint = endpoint, 
    wallet = wallet,
)
channel = grpc.insecure_channel('localhost',
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1)])          
stub = bittensor.grpc.BittensorStub(channel)

synapses = [
    bittensor.synapse.TextLastHiddenState(),
    bittensor.synapse.TextCausalLM(), 
    bittensor.synapse.TextCausalLMNext(),
    bittensor.synapse.TextSeq2Seq(num_to_generate=70)
]

def test_print():
    print(receptor)
    print(str(receptor))

#-- dummy testing --

def test_dummy_forward():
    endpoint = bittensor.endpoint.dummy()
    dummy_receptor = bittensor.receptor ( endpoint= endpoint, wallet=wallet)
    assert dummy_receptor.endpoint.uid == 0
    x = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    out, ops, time = dummy_receptor.forward( synapses, x, timeout=1)
    assert ops == [bittensor.proto.ReturnCode.BadEndpoint for _ in synapses]
    assert [list(o.shape) for o in out] == [[2, 4,bittensor.__network_dim__],
                                            [2, 4, bittensor.__vocab_size__],
                                            [2, (bittensor.synapse.TextCausalLMNext().topk + 1), 1 + 1],
                                            [2, 70]]


def test_dummy_backward():
    endpoint = bittensor.endpoint.dummy()
    dummy_receptor = bittensor.receptor ( endpoint= endpoint, wallet=wallet)
    assert dummy_receptor.endpoint.uid == 0
    x = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    grads = [torch.ones((x.size(0),x.size(1),bittensor.__network_dim__))]*len(synapses)
    out, ops, time = dummy_receptor.backward( synapses, x,grads, timeout=1)
    assert ops == [bittensor.proto.ReturnCode.BadEndpoint for _ in synapses]
    assert [list(o.shape) for o in out] == [[2,4,bittensor.__network_dim__],
                                            [2,4, bittensor.__vocab_size__],
                                            [2, (bittensor.synapse.TextCausalLMNext().topk + 1), 1 + 1],
                                            [0]]

# -- request serialization --

def test_receptor_forward_request_serialize_error():    
    x = torch.tensor([[[1,2,3,4]]], dtype=torch.long)
    out, ops, time = receptor.forward( synapses, x, timeout=1)
    assert ops == [bittensor.proto.ReturnCode.RequestSerializationException]*len(synapses)


def test_receptor_backward_request_serialize_error():    
    x = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    grads = [torch.ones((x.size(0))),
            torch.ones((x.size(0))),
            torch.ones((x.size(0))),
            torch.ones((x.size(0))) ]
    out, ops, time = receptor.backward( synapses, x,grads, timeout=1)
    assert ops == [bittensor.proto.ReturnCode.RequestSerializationException]*len(synapses)


def test_receptor_neuron_text():
    x = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    out, ops, time = receptor.forward( synapses, x, timeout=1)
    assert ops == [bittensor.proto.ReturnCode.Unavailable]*len(synapses)
    assert [list(o.shape) for o in out] == [[2, 4,bittensor.__network_dim__],
                                            [2, 4, bittensor.__vocab_size__],
                                            [2, (bittensor.synapse.TextCausalLMNext().topk + 1), 1 + 1],
                                            [2, 70]]

def test_receptor_neuron_text_backward():
    x = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    hidden_grads = torch.ones((x.size(0), x.size(1), bittensor.__network_dim__))
    causal_grads = torch.ones((x.size(0), x.size(1), bittensor.__vocab_size__))
    causallmnext_grads = torch.ones((x.size(0), (bittensor.synapse.TextCausalLMNext().topk + 1), 1 + 1))
    seq_2_seq_grads = torch.tensor([])

    out, ops, time = receptor.backward(synapses, x, [hidden_grads, causal_grads, causallmnext_grads, seq_2_seq_grads], timeout=1)
    assert ops == [bittensor.proto.ReturnCode.Success] * len(synapses)

# -- forward testing --

def test_receptor_neuron_request_empty():
    x = torch.tensor([])
    out, ops, time  = receptor.forward( synapses, x, timeout=1)
    assert ops == [bittensor.proto.ReturnCode.EmptyRequest]*len(synapses)
    assert [list(o.shape) for o in out] == [[0]]*len(synapses)

def test_receptor_neuron_mock_server():
    y_hidden = torch.rand(3, 3, bittensor.__network_dim__)
    y_causallm = torch.rand(3, 3, bittensor.__network_dim__)
    y_causallmnext = bittensor.synapse.TextCausalLMNext().nill_forward_response_tensor(torch.ones(3), encoded=True)
    y_seq_2_seq = torch.rand(3, 70)
    
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    y_hidden_serialized = serializer.serialize(y_hidden, from_type = bittensor.proto.TensorType.TORCH)
    y_causallm_serialized = serializer.serialize(y_causallm, from_type = bittensor.proto.TensorType.TORCH)
    y_causallmnext_serialized = serializer.serialize(y_causallmnext, from_type=bittensor.proto.TensorType.TORCH)
    y_seq_2_seq_serialized = serializer.serialize(y_seq_2_seq, from_type = bittensor.proto.TensorType.TORCH)
            
    mock_return_tensor = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.ss58_address,
            synapses = [synapse.serialize_to_wire_proto(code = bittensor.proto.ReturnCode.Success, message= 'Success' ) for synapse in synapses],
            return_code = bittensor.proto.ReturnCode.Success,
            tensors=[y_hidden_serialized, y_causallm_serialized, y_causallmnext_serialized, y_seq_2_seq_serialized]
        )
    mock_result = asyncio.Future()
    mock_result.set_result( mock_return_tensor )
    stub.Forward = MagicMock( return_value = mock_result)
    receptor.stub = stub

    x = torch.rand(3, 3)
    out, ops, time  = receptor.forward( synapses, x, timeout=1)
    print([list(o.shape) for o in out])
    assert ops == [bittensor.proto.ReturnCode.Success] * len(synapses)
    assert [list(o.shape) for o in out] == [[3, 3, bittensor.__network_dim__],
                                            [3, 3, bittensor.__vocab_size__],
                                            [3, (bittensor.synapse.TextCausalLMNext().topk + 1), 1 + 1],
                                            [3, 70]]

def test_receptor_neuron_serve_timeout():
    y_hidden = torch.rand(3, 3, bittensor.__network_dim__)
    y_causallm = torch.rand(3, 3, bittensor.__network_dim__)
    y_causallmnext = bittensor.synapse.TextCausalLMNext().nill_forward_response_tensor(torch.ones(3), encoded=True)
    y_seq_2_seq = torch.rand(3, 70, bittensor.__network_dim__)
    
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    y_hidden_serialized = serializer.serialize(y_hidden, from_type = bittensor.proto.TensorType.TORCH)
    y_causallm_serialized = serializer.serialize(y_causallm, from_type = bittensor.proto.TensorType.TORCH)
    y_causallmnext_serialized = serializer.serialize(y_causallmnext, from_type=bittensor.proto.TensorType.TORCH)
    y_seq_2_seq_serialized = serializer.serialize(y_seq_2_seq, from_type = bittensor.proto.TensorType.TORCH)
            
    mock_return_tensor = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.ss58_address,
            synapses = [synapse.serialize_to_wire_proto(code = bittensor.proto.ReturnCode.Timeout, message= 'Timeout' ) for synapse in synapses],
            tensors=[y_hidden_serialized, y_causallm_serialized, y_causallmnext_serialized, y_seq_2_seq_serialized],
            return_code = bittensor.proto.ReturnCode.Timeout
    )
    mock_result = asyncio.Future()
    mock_result.set_result( mock_return_tensor )
    stub.Forward = MagicMock( return_value = mock_result )
    receptor.stub = stub

    x = torch.rand(3, 3)
    out, ops, time  = receptor.forward( synapses, x, timeout=1)
    assert ops == [bittensor.proto.ReturnCode.Timeout] * len(synapses)
    assert [list(o.shape) for o in out] == [[3, 3, bittensor.__network_dim__],
                                            [3, 3, bittensor.__vocab_size__],
                                            [3, (bittensor.synapse.TextCausalLMNext().topk + 1), 1 + 1],
                                            [3, 70]]

def test_receptor_neuron_mock_server_deserialization_error():
    y = dict() # bad response
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.ss58_address,
            synapses = [synapse.serialize_to_wire_proto(code = bittensor.proto.ReturnCode.Success, message= 'Success' ) for synapse in synapses],
            return_code = bittensor.proto.ReturnCode.Success,
            tensors=[y, y, y, y]
        )
    mock_result = asyncio.Future()
    mock_result.set_result( mock_return_val )

    stub.Forward = MagicMock( return_value = mock_result )
    receptor.stub = stub

    x = torch.rand(3, 3)
    out, ops, time  = receptor.forward( synapses, x, timeout=1)
    assert ops == [bittensor.proto.ReturnCode.ResponseDeserializationException] * len(synapses)
    assert [list(o.shape) for o in out] == [[3, 3, bittensor.__network_dim__],
                                            [3, 3, bittensor.__vocab_size__],
                                            [3, (bittensor.synapse.TextCausalLMNext().topk + 1), 1 + 1],
                                            [3, 70]]

def test_receptor_neuron_mock_server_shape_error():
    y = torch.rand(3, 3, bittensor.__network_dim__)

    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    y_serialized = serializer.serialize(y, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
   
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.ss58_address,
            return_code = bittensor.proto.ReturnCode.Success,
            tensors = [y_serialized],
            synapses = [synapse.serialize_to_wire_proto(code = bittensor.proto.ReturnCode.Success, message= 'Success' ) for synapse in synapses],
        )
    mock_result = asyncio.Future()
    mock_result.set_result( mock_return_val )


    stub.Forward = MagicMock( return_value = mock_result )
    receptor.stub = stub

    x = torch.rand(3, 3)

    out, ops, time  = receptor.forward( synapses, x, timeout=1)
    print(ops, bittensor.proto.ReturnCode.ResponseShapeException)
    assert ops == [bittensor.proto.ReturnCode.ResponseShapeException] * len(synapses)
    assert [list(o.shape) for o in out] == [[3, 3, bittensor.__network_dim__],
                                            [3, 3, bittensor.__vocab_size__],
                                            [3, (bittensor.synapse.TextCausalLMNext().topk + 1), 1 + 1],
                                            [3, 70]]

def test_receptor_neuron_server_response_with_nans():
    import numpy as np

    y_hidden = torch.rand(3, 3, bittensor.__network_dim__)
    y_causallm = torch.rand(3, 3, bittensor.__network_dim__)
    y_causallmnext = bittensor.synapse.TextCausalLMNext().nill_forward_response_tensor(torch.ones(3), encoded=True)
    y_seq_2_seq = torch.rand(3, 70)
    
    y_hidden[0][0][0] = np.nan
    y_causallm[0][0][0] = np.nan
    y_causallmnext[0] = np.nan  # unravel fails because demarcating probability is replaced by nan, ResponseDeserializationException
    y_seq_2_seq[0][0] = np.nan
    
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    y_hidden_serialized = serializer.serialize(y_hidden, from_type = bittensor.proto.TensorType.TORCH)
    y_causallm_serialized = serializer.serialize(y_causallm, from_type = bittensor.proto.TensorType.TORCH)
    y_causallmnext_serialized = serializer.serialize(y_causallmnext, from_type=bittensor.proto.TensorType.TORCH)
    y_seq_2_seq_serialized = serializer.serialize(y_seq_2_seq, from_type = bittensor.proto.TensorType.TORCH)

    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.ss58_address,
            return_code = bittensor.proto.ReturnCode.Success,
            synapses = [synapse.serialize_to_wire_proto(code = bittensor.proto.ReturnCode.Success, message= 'Success' ) for synapse in synapses],
            tensors = [y_hidden_serialized, y_causallm_serialized, y_causallmnext_serialized, y_seq_2_seq_serialized]
        )
    mock_result = asyncio.Future()
    mock_result.set_result( mock_return_val )

    stub.Forward = MagicMock( return_value = mock_result )
    receptor.stub = stub

    x = torch.rand(3, 3)
    out, ops, time  = receptor.forward( synapses, x, timeout=1)
    assert ops == [bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.Success,
                   bittensor.proto.ReturnCode.ResponseDeserializationException, bittensor.proto.ReturnCode.Success]
    assert out[0][0][0][0] != np.nan
    assert out[1][0][0][0] != np.nan
    assert out[3][0][0] != np.nan

# -- backwards testing --
def test_receptor_neuron_grads_misshape():
    x = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    grads = torch.zeros([0,1,2,3,4])
    out, ops, time = receptor.backward( synapses, x, [grads, grads, grads, grads], timeout=1)
    assert ops == [bittensor.proto.ReturnCode.RequestSerializationException] * len(synapses)


def test_receptor_neuron_mock_server_backward():
    y = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    y_serialized = serializer.serialize(y, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
            
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = "0x" + wallet.hotkey.public_key.hex(),
            return_code = bittensor.proto.ReturnCode.Success,
            synapses = [synapse.serialize_to_wire_proto(code = bittensor.proto.ReturnCode.Success, message= 'Success' ) for synapse in synapses],
            tensors = [y_serialized])

    mock_result = asyncio.Future()
    mock_result.set_result( mock_return_val )

    stub.Backward = MagicMock( return_value = mock_result )
    receptor.stub = stub

    x = torch.rand(3, 3)
    hidden_grads = torch.ones((x.size(0), x.size(1), bittensor.__network_dim__))
    causal_grads = torch.ones((x.size(0), x.size(1), bittensor.__vocab_size__))
    causallmnext_grads = torch.ones((x.size(0), (bittensor.synapse.TextCausalLMNext().topk + 1), 1 + 1))
    seq_2_seq_grads = torch.tensor([])
    out, ops, time = receptor.backward(synapses, x, [hidden_grads, causal_grads, causallmnext_grads, seq_2_seq_grads], timeout=1)
    assert ops == [bittensor.proto.ReturnCode.Success] * len(synapses)

# -- no return code -- 

def test_receptor_forward_no_return():
    y = torch.rand(3, 3, bittensor.__network_dim__)
    
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    y_serialized = serializer.serialize(y, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
            
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.ss58_address,
            synapses = [synapse.serialize_to_wire_proto(message= 'NoReturn' ) for synapse in synapses],
            tensors = [y_serialized]
        )
    mock_result = asyncio.Future()
    mock_result.set_result( mock_return_val )

    stub.Forward = MagicMock( return_value = mock_result )
    receptor.stub = stub

    x = torch.rand(3, 3)
    out, ops, time  = receptor.forward( synapses, x, timeout=1)
    assert ops == [bittensor.proto.ReturnCode.NoReturn] * len(synapses)

# -- no exception in response -- 

def test_receptor_forward_exception():
    y = torch.rand(3, 3, bittensor.__network_dim__)
    
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    y_serialized = serializer.serialize(y, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
            
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.ss58_address,
            return_code = bittensor.proto.ReturnCode.UnknownException,
            synapses = [synapse.serialize_to_wire_proto(code = bittensor.proto.ReturnCode.UnknownException, message= 'Success' ) for synapse in synapses],
            tensors = [y_serialized])
    mock_result = asyncio.Future()
    mock_result.set_result( mock_return_val )


    stub.Forward = MagicMock( return_value = mock_result )
    receptor.stub = stub

    x = torch.rand(3, 3)
    out, ops, time  = receptor.forward( synapses, x, timeout=1)
    assert ops == [bittensor.proto.ReturnCode.UnknownException] * len(synapses)


# -- stub erorr -- 

def test_receptor_forward_stub_exception():

    def forward_break():
        raise Exception('Mock')

    with mock.patch.object(receptor.stub, 'Forward', new=forward_break):
        x = torch.rand(3, 3)
        out, ops, time  = receptor.forward( synapses, x, timeout=1)
        assert ops == [bittensor.proto.ReturnCode.UnknownException] * len(synapses)

def test_receptor_backward_stub_exception():

    def backward_break():
        raise Exception('Mock')
    with mock.patch.object(receptor.stub, 'Backward', new=backward_break):
        x = torch.rand(3, 3)
        hidden_grads = torch.ones((x.size(0), x.size(1), bittensor.__network_dim__))
        causal_grads = torch.ones((x.size(0), x.size(1), bittensor.__vocab_size__))
        causallmnext_grads = torch.ones((x.size(0), (bittensor.synapse.TextCausalLMNext().topk + 1), 1 + 1))
        seq_2_seq_grads = torch.tensor([])
        out, ops, time = receptor.backward(synapses, x, [hidden_grads, causal_grads, causallmnext_grads, seq_2_seq_grads], timeout=1)
        assert ops == [bittensor.proto.ReturnCode.UnknownException] * len(synapses)


def test_receptor_forward_endpoint_exception():
    
    receptor = bittensor.receptor ( 
        endpoint = endpoint, 
        wallet = wallet,
    )
    
    def forward_break():
        raise Exception('Mock')

    with mock.patch.object(bittensor.proto, 'TensorMessage', new=forward_break):
        x = torch.rand(3, 3)
        out, ops, time  = receptor.forward( synapses, x, timeout=1)
        assert ops == [bittensor.proto.ReturnCode.UnknownException] * len(synapses)

def test_receptor_backward_endpoint_exception():
    
    receptor = bittensor.receptor ( 
        endpoint = endpoint, 
        wallet = wallet,
    )
    def backward_break():
        raise Exception('Mock')

    with mock.patch.object(bittensor.proto, 'TensorMessage', new=backward_break):
        x = torch.rand(3, 3)
        hidden_grads = torch.ones((x.size(0), x.size(1), bittensor.__network_dim__))
        causal_grads = torch.ones((x.size(0), x.size(1), bittensor.__vocab_size__))
        causallmnext_grads = torch.ones((x.size(0), (bittensor.synapse.TextCausalLMNext().topk + 1), 1 + 1))
        seq_2_seq_grads = torch.tensor([])
        out, ops, time = receptor.backward(synapses, x, [hidden_grads, causal_grads, causallmnext_grads, seq_2_seq_grads], timeout=1)
        assert ops == [bittensor.proto.ReturnCode.UnknownException] * len(synapses)

def test_receptor_signature_output():

    def verify_v2(signature: str):
        (nonce, sender_address, signature, receptor_uuid) = signature.split(".")
        assert nonce == "123"
        assert sender_address == "5Ey8t8pBJSYqLYCzeC3HiPJu5DxzXy2Dzheaj29wRHvhjoai"
        assert receptor_uuid == "6d8b8788-6b6a-11ed-916f-0242c0a85003"
        message = f"{nonce}.{sender_address}.5CSbZ7wG456oty4WoiX6a1J88VUbrCXLhrKVJ9q95BsYH4TZ.{receptor_uuid}"
        assert wallet.hotkey.verify(message, signature)

    matrix = {
        bittensor.__new_signature_version__: verify_v2,
    }

    for (receiver_version, verify) in matrix.items():
        endpoint = bittensor.endpoint(
            version=receiver_version,
            uid=0,
            ip="127.0.0.1",
            ip_type=4,
            port=65000,
            hotkey="5CSbZ7wG456oty4WoiX6a1J88VUbrCXLhrKVJ9q95BsYH4TZ",
            coldkey="5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm",
            protocol=0,
        )

        receptor = bittensor.receptor(
            endpoint=endpoint,
            wallet=wallet,
        )
        receptor.receptor_uid = "6d8b8788-6b6a-11ed-916f-0242c0a85003"
        receptor.nonce = lambda: 123

        verify(receptor.sign())

#-- axon receptor connection -- 

def run_test_axon_receptor_connection_forward_works(receiver_version):
    def forward_generate( input, synapse, model_output = None):
        return None, None, torch.zeros( [3, 70])

    def forward_hidden_state( input, synapse, model_output = None):
        return None, None, torch.zeros( [3, 3, bittensor.__network_dim__])

    def forward_casual_lm( input, synapse, model_output = None):
        return None, None, torch.zeros( [3, 3, bittensor.__vocab_size__])

    def forward_casual_lm_next(input, synapse, model_output=None):
        return None, None, torch.zeros([3, (synapse.topk + 1), 1 + 1])

    port = get_random_unused_port()
    axon = bittensor.axon (
        port = port,
        ip = '127.0.0.1',
        wallet = wallet,
        netuid = -1,
    )
    axon.attach_synapse_callback( forward_hidden_state,  synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE )
    axon.attach_synapse_callback( forward_generate,  synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ )
    axon.attach_synapse_callback( forward_casual_lm,  synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM )
    axon.attach_synapse_callback(forward_casual_lm_next, synapse_type=bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT)
    axon.start()
    
    endpoint = bittensor.endpoint(
        version = receiver_version,
        uid = 0,
        ip = '127.0.0.1',
        ip_type = 4,
        port = port,
        hotkey = wallet.hotkey.ss58_address,
        coldkey = wallet.coldkey.ss58_address,
        modality = 2,
        protocol = 0,
    )

    receptor = bittensor.receptor ( 
        endpoint = endpoint, 
        wallet = wallet,
    )

    x = torch.rand(3, 3)
    out, ops, time  = receptor.forward( synapses, x, timeout=1)
    assert ops == [bittensor.proto.ReturnCode.Success] * len(synapses)
    axon.stop()


def test_axon_receptor_connection_forward_works():
    for receiver_version in [341, bittensor.__new_signature_version__, bittensor.__version_as_int__]:
        run_test_axon_receptor_connection_forward_works(receiver_version)

def test_axon_receptor_connection_forward_unauthenticated():
    def forward_generate( input, synapse, model_output = None ):
        return None, None, torch.zeros( [3, 70])

    def forward_hidden_state( input, synapse, model_output = None ):
        return None, None, torch.zeros( [3, 3, bittensor.__network_dim__])

    def forward_casual_lm( input, synapse, model_output = None ):
        return None, None, torch.zeros( [3, 3, bittensor.__vocab_size__])

    def forward_casual_lm_next(input, synapse, model_output=None):
        return None, None, torch.zeros([3, (synapse.topk + 1), 1 + 1])

    axon = bittensor.axon (
        port = 8081,
        ip = '127.0.0.1',
        wallet = wallet,
        netuid = -1,
    )
    axon.attach_synapse_callback( forward_hidden_state,  synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE )
    axon.attach_synapse_callback( forward_generate,  synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ )
    axon.attach_synapse_callback( forward_casual_lm,  synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM )
    axon.attach_synapse_callback(forward_casual_lm_next, synapse_type=bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT)
    axon.start()
    
    endpoint = bittensor.endpoint(
        version = bittensor.__version_as_int__,
        uid = 0,
        ip = '127.0.0.1',
        ip_type = 4,
        port = 8081,
        hotkey = wallet.hotkey.ss58_address,
        coldkey = wallet.coldkey.ss58_address,
        protocol = 0
    )

    receptor = bittensor.receptor ( 
        endpoint = endpoint, 
        wallet = wallet,
    )

    x = torch.rand(3, 3)
    receptor.sign = MagicMock( return_value='mock' )
    out, ops, time  = receptor.forward( synapses, x, timeout=1)
    assert ops == [bittensor.proto.ReturnCode.Unauthenticated] * len(synapses)
    axon.stop()


# NOTE(const): This test should be removed because it is broken and breaks randomly depending on the
# speed at which the error propagates up the stack. The backward does NOT work on the axon since there
# is a trivial error in the default_backward_callback.
# def test_axon_receptor_connection_backward_works():
#     def forward_generate( input, synapse ):
#         return torch.zeros( [3, 70])

#     def forward_hidden_state( input, synapse ):
#         return torch.zeros( [3, 3, bittensor.__network_dim__])

#     def forward_casual_lm( input, synapse ):
#         return torch.zeros( [3, 3, bittensor.__vocab_size__])

#     def forward_casual_lm_next(input, synapse):
#         return torch.zeros([3, (synapse.topk + 1), 1 + 1])

#     axon = bittensor.axon (
#         port = 8082,
#         ip = '127.0.0.1',
#         wallet = wallet,
#     )
#     axon.attach_synapse_callback( forward_hidden_state,  synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE )
#     axon.attach_synapse_callback( forward_generate,  synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ )
#     axon.attach_synapse_callback( forward_casual_lm,  synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM )
#     axon.attach_synapse_callback(forward_casual_lm_next, synapse_type=bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT)
#     axon.start()
    
#     endpoint = bittensor.endpoint(
#         version = bittensor.__version_as_int__,
#         uid = 0,
#         ip = '127.0.0.1',
#         ip_type = 4,
#         port = 8082,
#         hotkey = wallet.hotkey.ss58_address,
#         coldkey = wallet.coldkey.ss58_address,
#         protocol = 0
#     )

#     receptor = bittensor.receptor ( 
#         endpoint = endpoint, 
#         wallet = wallet,
#     )
#     x = torch.rand(3, 3)
#     hidden_grads = torch.ones((x.size(0), x.size(1), bittensor.__network_dim__))
#     causal_grads = torch.ones((x.size(0), x.size(1), bittensor.__vocab_size__))
#     causallmnext_grads = torch.ones((x.size(0), (bittensor.synapse.TextCausalLMNext().topk + 1), 1 + 1))
#     seq_2_seq_grads = torch.tensor([])

#     out, ops, time = receptor.backward(synapses, x, [hidden_grads, causal_grads, causallmnext_grads, seq_2_seq_grads], timeout=1)
#     assert ops == [bittensor.proto.ReturnCode.Success] * len(synapses)
#     axon.stop()

def test_axon_receptor_connection_backward_unauthenticated():
    def forward_generate( input, synapse ):
        return torch.zeros( [3, 70])

    def forward_hidden_state( input, synapse ):
        return torch.zeros( [3, 3, bittensor.__network_dim__])

    def forward_casual_lm( input, synapse ):
        return torch.zeros( [3, 3, bittensor.__vocab_size__])

    def forward_casual_lm_next(input, synapse):
        return torch.zeros([3, (synapse.topk + 1), 1 + 1])

    axon = bittensor.axon (
        port = 8090,
        ip = '127.0.0.1',
        wallet = wallet,
        netuid = -1,
    )
    axon.attach_synapse_callback( forward_hidden_state,  synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE )
    axon.attach_synapse_callback( forward_generate,  synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ )
    axon.attach_synapse_callback( forward_casual_lm,  synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM )
    axon.attach_synapse_callback( forward_casual_lm_next, synapse_type=bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT)
    axon.start()
    
    endpoint = bittensor.endpoint(
        version = bittensor.__version_as_int__,
        uid = 0,
        ip = '127.0.0.1',
        ip_type = 4,
        port = 8090,
        hotkey = wallet.hotkey.ss58_address,
        coldkey = wallet.coldkey.ss58_address,
        protocol = 0
    )

    receptor = bittensor.receptor ( 
        endpoint = endpoint, 
        wallet = wallet,
    )

    x = torch.rand(3, 3)
    hidden_grads = torch.ones((x.size(0), x.size(1), bittensor.__network_dim__))
    causal_grads = torch.ones((x.size(0), x.size(1), bittensor.__vocab_size__))
    causallmnext_grads = torch.ones((x.size(0), (bittensor.synapse.TextCausalLMNext().topk + 1), 1 + 1))
    seq_2_seq_grads = torch.tensor([])

    receptor.sign = MagicMock( return_value='mock' )
    out, ops, time = receptor.backward(synapses, x, [hidden_grads, causal_grads, causallmnext_grads, seq_2_seq_grads], timeout=1)

    assert ops == [bittensor.proto.ReturnCode.Success] * len(synapses)
    axon.stop()

## --unimplemented error 

def test_axon_receptor_connection_forward_unimplemented():
    port = get_random_unused_port()
    axon = bittensor.axon (
        port = port,
        ip = '127.0.0.1',
        wallet = wallet,
        netuid = -1,
    )
    axon.start()
    
    endpoint = bittensor.endpoint(
        version = bittensor.__version_as_int__,
        uid = 0,
        ip = '127.0.0.1',
        ip_type = 4,
        port = port,
        hotkey = wallet.hotkey.ss58_address,
        coldkey = wallet.coldkey.ss58_address,
        protocol = 0
    )

    receptor = bittensor.receptor ( 
        endpoint = endpoint, 
        wallet = wallet,
    )

    x = torch.rand(3, 3)
    out, ops, time  = receptor.forward( synapses, x, timeout=1)
    assert ops == [bittensor.proto.ReturnCode.NotImplemented] * len(synapses)
    axon.stop()

## -- timeout error

def test_axon_receptor_connection_forward_timeout():

    def forward_generate( inputs, synapse, model_output = None):
        clock.sleep(5)
        raise TimeoutError('Timeout')

    def forward_hidden_state( inputs, synapse, model_output = None ):
        clock.sleep(5)
        raise TimeoutError('Timeout')

    def forward_casual_lm( inputs, synapse, model_output = None ):
        clock.sleep(5)
        raise TimeoutError('Timeout')

    def forward_casual_lm_next(inputs, synapse, model_output=None):
        clock.sleep(5)
        raise TimeoutError('Timeout')

    axon = bittensor.axon (
        port = 8085,
        ip = '127.0.0.1',
        wallet = wallet,
        netuid = -1,
    )
    axon.attach_synapse_callback( forward_hidden_state,  synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE )
    axon.attach_synapse_callback( forward_generate,  synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ )
    axon.attach_synapse_callback( forward_casual_lm,  synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM )
    axon.attach_synapse_callback(forward_casual_lm_next, synapse_type=bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT)
    axon.start()
    
    endpoint = bittensor.endpoint(
        version = bittensor.__version_as_int__,
        uid = 0,
        ip = '127.0.0.1',
        ip_type = 4,
        port = 8085,
        hotkey = wallet.hotkey.ss58_address,
        coldkey = wallet.coldkey.ss58_address,
        protocol = 0
    )

    receptor = bittensor.receptor ( 
        endpoint = endpoint, 
        wallet = wallet,
    )

    x = torch.rand(3, 3)
    out, ops, time  = receptor.forward( synapses, x, timeout=1)
    assert ops == [bittensor.proto.ReturnCode.Timeout] * len(synapses)
    axon.stop()

def test_axon_receptor_connection_backward_timeout():
    def forward_generate( inputs, synapse ):
        clock.sleep(5)
        raise TimeoutError('Timeout')

    def forward_hidden_state( inputs, synapse ):
        clock.sleep(5)
        raise TimeoutError('Timeout')

    def forward_casual_lm( inputs, synapse ):
        clock.sleep(5)
        raise TimeoutError('Timeout')

    def forward_casual_lm_next(inputs, synapse):
        clock.sleep(5)
        raise TimeoutError('Timeout')

    axon = bittensor.axon (
        port = 8088,
        ip = '127.0.0.1',
        wallet = wallet,
        netuid = -1,
    )
    axon.attach_synapse_callback( forward_hidden_state,  synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE )
    axon.attach_synapse_callback( forward_generate,  synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ )
    axon.attach_synapse_callback( forward_casual_lm,  synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM )
    axon.attach_synapse_callback(forward_casual_lm_next, synapse_type=bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT)
    axon.start()
    
    endpoint = bittensor.endpoint(
        version = bittensor.__version_as_int__,
        uid = 0,
        ip = '127.0.0.1',
        ip_type = 4,
        port = 8088,
        hotkey = wallet.hotkey.ss58_address,
        coldkey = wallet.coldkey.ss58_address,
        protocol = 0
    )

    receptor = bittensor.receptor ( 
        endpoint = endpoint, 
        wallet = wallet,
    )

    x = torch.rand(3, 3)
    hidden_grads = torch.ones((x.size(0), x.size(1), bittensor.__network_dim__))
    causal_grads = torch.ones((x.size(0), x.size(1), bittensor.__vocab_size__))
    causallmnext_grads = torch.ones((x.size(0), (bittensor.synapse.TextCausalLMNext().topk + 1), 1 + 1))
    seq_2_seq_grads = torch.tensor([])
    out, ops, time = receptor.backward(synapses, x, [hidden_grads, causal_grads, causallmnext_grads, seq_2_seq_grads], timeout=1)

    assert ops == [bittensor.proto.ReturnCode.Success] * len(synapses)
    axon.stop()

if __name__ == "__main__":
    # test_dummy_forward()
    # test_dummy_backward()
    # test_receptor_forward_request_serialize_error()
    # test_receptor_backward_request_serialize_error()
    # test_receptor_neuron_text()
    # test_receptor_neuron_image()
    # test_receptor_neuron_request_empty()
    #test_receptor_neuron_mock_server()
    # test_receptor_neuron_serve_timeout()
    #test_axon_receptor_connection_backward_unauthenticated()
    # test_receptor_neuron_mock_server_deserialization_error()
    # test_receptor_neuron_mock_server_shape_error()
    # test_receptor_neuron_server_response_with_nans()
    #test_receptor_neuron_text_backward()
    # test_receptor_neuron_grads_misshape()
    # test_receptor_neuron_mock_server_deserialization_error_backward()
    # test_receptor_neuron_backward_empty_response()
    # test_receptor_forward_no_return()
    # test_receptor_forward_exception()
    # test_axon_receptor_connection_forward_works()
    # test_receptor_neuron_mock_server()
    # test_receptor_neuron_server_response_with_nans()
    # test_axon_receptor_connection_forward_works()
    # test_axon_receptor_connection_forward_unauthenticated()
    #test_axon_receptor_connection_forward_timeout()
    test_receptor_neuron_text_backward()
    # test_axon_receptor_connection_backward_works()
    # test_axon_receptor_connection_backward_unimplemented()
    # test_axon_receptor_connection_forward_works()
    # test_receptor_neuron_mock_server()
    # test_receptor_neuron_mock_server_backward()
    # test_receptor_neuron_server_response_with_nans()
