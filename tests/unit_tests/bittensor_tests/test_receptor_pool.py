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
import time

from unittest.mock import MagicMock
import unittest.mock as mock
import asyncio

# --- Receptor Pool ---
wallet = bittensor.wallet.mock()
wallet2 = bittensor.wallet.mock()
wallet2.create_new_coldkey(use_password=False, overwrite = True)
wallet2.create_new_hotkey(use_password=False, overwrite = True)


neuron_obj = bittensor.endpoint(
    version = bittensor.__version_as_int__,
    uid = 0,
    ip = '0.0.0.0',
    ip_type = 4,
    port = 12345,
    hotkey = wallet.hotkey.ss58_address,
    coldkey = wallet.coldkey.ss58_address,
    protocol =0
)

receptor_pool = bittensor.receptor_pool(wallet=wallet)

synapses = [
    bittensor.synapse.TextLastHiddenState(),
    bittensor.synapse.TextCausalLM(), 
    bittensor.synapse.TextCausalLMNext(),
    bittensor.synapse.TextSeq2Seq(num_to_generate=70)
]

def test_receptor_pool_forward():
    endpoints = [neuron_obj]
    x = torch.ones( (1, 2 ,2) )
    resp1,  _, _ = receptor_pool.forward( endpoints, synapses, x, timeout=1)
    assert list(resp1[0][0].shape) == [2, 2, bittensor.__network_dim__]
    assert list(resp1[0][1].shape) == [2, 2, bittensor.__vocab_size__]
    assert list(resp1[0][2].shape) == [2, (bittensor.synapse.TextCausalLMNext().topk + 1), 1 + 1]
    assert list(resp1[0][3].shape) == [2, 70]

def test_receptor_pool_backward():
    endpoints = [neuron_obj]
    x = torch.ones( (1,2,2) )
    grads = [[torch.ones(2, 2, bittensor.__network_dim__),
              torch.ones(2, 2, bittensor.__vocab_size__),
              torch.ones(1, (bittensor.synapse.TextCausalLMNext().topk + 1), 1 + 1),
              torch.tensor([])]]
    receptor_pool.backward( endpoints, synapses, x, grads, timeout=1)

def test_receptor_pool_max_workers_forward():
    neuron_obj2 = bittensor.endpoint(
        version = bittensor.__version_as_int__,
        uid = 0,
        ip = '0.0.0.1',
        ip_type = 4,
        port = 12346,
        hotkey = wallet2.hotkey.ss58_address,
        coldkey = wallet2.coldkey.ss58_address,
        protocol =0
    )
    receptor_pool = bittensor.receptor_pool(wallet=wallet,max_active_receptors=1)
    endpoints = [neuron_obj,neuron_obj2]
    x = torch.ones( (2,2,2) )
    resp1,  _, _ = receptor_pool.forward( endpoints, synapses, x, timeout=1)
    assert list(resp1[0][0].shape) == [2, 2, bittensor.__network_dim__]
    assert list(resp1[0][1].shape) == [2, 2, bittensor.__vocab_size__]
    assert list(resp1[0][2].shape) == [2, (bittensor.synapse.TextCausalLMNext().topk + 1), 1 + 1]
    assert list(resp1[0][3].shape) == [2, 70]

def test_receptor_pool_forward_success():
    endpoints = [neuron_obj,neuron_obj]
    x = torch.ones( (2, 3, 3) )    

    y_hidden = torch.rand(3, 3, bittensor.__network_dim__)
    y_causallm = torch.rand(3, 3, bittensor.__network_dim__)
    y_causallmnext = bittensor.synapse.TextCausalLMNext().nill_forward_response_tensor(torch.ones(3), encoded=True)
    y_seq_2_seq = torch.rand(3, 70)
    
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    y_hidden_serialized = serializer.serialize(y_hidden, from_type = bittensor.proto.TensorType.TORCH)
    y_causallm_serialized = serializer.serialize(y_causallm, from_type = bittensor.proto.TensorType.TORCH)
    y_causallmnext_serialized = serializer.serialize(y_causallmnext, from_type=bittensor.proto.TensorType.TORCH)
    y_seq_2_seq_serialized = serializer.serialize(y_seq_2_seq, from_type = bittensor.proto.TensorType.TORCH)
            
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.ss58_address,
            synapses = [synapse.serialize_to_wire_proto(code = bittensor.proto.ReturnCode.Success, message= 'Success' ) for synapse in synapses],
            return_code = bittensor.proto.ReturnCode.Success,
            tensors = [y_hidden_serialized, y_causallm_serialized, y_causallmnext_serialized, y_seq_2_seq_serialized]
        )
    mock_result = asyncio.Future()
    mock_result.set_result( mock_return_val )


    receptor_pool = bittensor.receptor_pool(wallet=wallet,max_active_receptors=1)
    receptor_pool._get_or_create_receptor_for_endpoint(neuron_obj)
    receptor_pool.receptors[neuron_obj.hotkey].stub.Forward = MagicMock( return_value = mock_result )
    resp1,  codes, _ = receptor_pool.forward( endpoints, synapses, x, timeout=1)
    assert codes == [[bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.Success],
    [bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.Success]]

def test_receptor_pool_forward_timeout():
    endpoints = [neuron_obj,neuron_obj]
    x = torch.ones( (2, 3, 3) )    

    y_hidden = torch.rand(3, 3, bittensor.__network_dim__)
    y_causallm = torch.rand(3, 3, bittensor.__network_dim__)
    y_causallmnext = bittensor.synapse.TextCausalLMNext().nill_forward_response_tensor(torch.ones(3), encoded=True)
    y_seq_2_seq = torch.rand(3, 70)
    
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    y_hidden_serialized = serializer.serialize(y_hidden, from_type = bittensor.proto.TensorType.TORCH)
    y_causallm_serialized = serializer.serialize(y_causallm, from_type = bittensor.proto.TensorType.TORCH)
    y_causallmnext_serialized = serializer.serialize(y_causallmnext, from_type=bittensor.proto.TensorType.TORCH)
    y_seq_2_seq_serialized = serializer.serialize(y_seq_2_seq, from_type = bittensor.proto.TensorType.TORCH)
            
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.ss58_address,
            synapses = [synapse.serialize_to_wire_proto(code = bittensor.proto.ReturnCode.Timeout, message= 'Timeout' ) for synapse in synapses],
            return_code = bittensor.proto.ReturnCode.Timeout,
            tensors=[y_hidden_serialized, y_causallm_serialized, y_causallmnext_serialized, y_seq_2_seq_serialized]
        )
    mock_result = asyncio.Future()
    mock_result.set_result( mock_return_val )


    receptor_pool = bittensor.receptor_pool(wallet=wallet,max_active_receptors=1)
    receptor_pool._get_or_create_receptor_for_endpoint(neuron_obj)
    receptor_pool.receptors[neuron_obj.hotkey].stub.Forward = MagicMock( return_value = mock_result )
    resp1,  codes, _ = receptor_pool.forward( endpoints, synapses, x, timeout=1)
    assert codes == [
        [bittensor.proto.ReturnCode.Timeout, bittensor.proto.ReturnCode.Timeout, bittensor.proto.ReturnCode.Timeout,
         bittensor.proto.ReturnCode.Timeout],
        [bittensor.proto.ReturnCode.Timeout, bittensor.proto.ReturnCode.Timeout, bittensor.proto.ReturnCode.Timeout,
         bittensor.proto.ReturnCode.Timeout]]

def test_receptor_pool_forward_num_synapse_mismatch():
    endpoints = [neuron_obj,neuron_obj]
    x = torch.ones( (2, 3, 3) )    

    y_hidden = torch.rand(3, 3, bittensor.__network_dim__)
    y_causallm = torch.rand(3, 3, bittensor.__network_dim__)
    y_causallmnext = bittensor.synapse.TextCausalLMNext().nill_forward_response_tensor(torch.ones(3), encoded=True)
    y_seq_2_seq = torch.rand(3, 70)
    
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    y_hidden_serialized = serializer.serialize(y_hidden, from_type = bittensor.proto.TensorType.TORCH)
    y_causallm_serialized = serializer.serialize(y_causallm, from_type = bittensor.proto.TensorType.TORCH)
    y_causallmnext_serialized = serializer.serialize(y_causallmnext, from_type=bittensor.proto.TensorType.TORCH)
    y_seq_2_seq_serialized = serializer.serialize(y_seq_2_seq, from_type = bittensor.proto.TensorType.TORCH)
            
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.ss58_address,
            synapses = [synapse.serialize_to_wire_proto(code = bittensor.proto.ReturnCode.Success, message= 'Timeout' ) for synapse in synapses],
            return_code = bittensor.proto.ReturnCode.Success,
            tensors = [y_hidden_serialized, y_causallm_serialized, y_causallmnext_serialized]
        )

    receptor_pool = bittensor.receptor_pool(wallet=wallet,max_active_receptors=1)
    receptor_pool._get_or_create_receptor_for_endpoint(neuron_obj)

    mock_result = asyncio.Future()
    mock_result.set_result( mock_return_val )
    receptor_pool.receptors[neuron_obj.hotkey].stub.Forward = MagicMock( return_value = mock_result )
    resp1,  codes, _ = receptor_pool.forward( endpoints, synapses, x, timeout=1)
    assert codes == [[bittensor.proto.ReturnCode.ResponseShapeException, bittensor.proto.ReturnCode.ResponseShapeException, bittensor.proto.ReturnCode.ResponseShapeException, bittensor.proto.ReturnCode.ResponseShapeException],
    [bittensor.proto.ReturnCode.ResponseShapeException, bittensor.proto.ReturnCode.ResponseShapeException, bittensor.proto.ReturnCode.ResponseShapeException, bittensor.proto.ReturnCode.ResponseShapeException]]

def test_receptor_pool_forward_response_partial_shape_error():
    endpoints = [neuron_obj,neuron_obj]
    x = torch.ones( (2, 3, 3) )    

    y_hidden = torch.rand(3, 3, bittensor.__network_dim__)
    y_causallm = torch.rand(3, 3, bittensor.__network_dim__)
    y_causallmnext = bittensor.synapse.TextCausalLMNext().nill_forward_response_tensor(torch.ones(3), encoded=True)
    y_seq_2_seq = torch.rand(2, 70)
    
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    y_hidden_serialized = serializer.serialize(y_hidden, from_type = bittensor.proto.TensorType.TORCH)
    y_causallm_serialized = serializer.serialize(y_causallm, from_type = bittensor.proto.TensorType.TORCH)
    y_causallmnext_serialized = serializer.serialize(y_causallmnext, from_type=bittensor.proto.TensorType.TORCH)
    y_seq_2_seq_serialized = serializer.serialize(y_seq_2_seq, from_type = bittensor.proto.TensorType.TORCH)
            
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.ss58_address,
            synapses = [synapse.serialize_to_wire_proto(code = bittensor.proto.ReturnCode.Success, message= 'Success' ) for synapse in synapses],
            return_code = bittensor.proto.ReturnCode.Success,
            tensors = [y_hidden_serialized, y_causallm_serialized, y_causallmnext_serialized, y_seq_2_seq_serialized]
        )

    receptor_pool = bittensor.receptor_pool(wallet=wallet,max_active_receptors=1)
    receptor_pool._get_or_create_receptor_for_endpoint(neuron_obj)

    mock_result = asyncio.Future()
    mock_result.set_result( mock_return_val )

    receptor_pool.receptors[neuron_obj.hotkey].stub.Forward = MagicMock( return_value = mock_result )
    resp1,  codes, _ = receptor_pool.forward( endpoints, synapses, x, timeout=1)
    assert codes == [[bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.ResponseDeserializationException],
    [bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.ResponseDeserializationException]]

def test_receptor_pool_partial_remote_success_return_code():
    endpoints = [neuron_obj,neuron_obj]
    x = torch.ones( (2, 3, 3) )    

    y_hidden = torch.rand(3, 3, bittensor.__network_dim__)
    y_causallm = torch.rand(3, 3, bittensor.__network_dim__)
    y_causallmnext = bittensor.synapse.TextCausalLMNext().nill_forward_response_tensor(torch.ones(3), encoded=True)
    y_seq_2_seq = torch.rand(2, 70)
    
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    y_hidden_serialized = serializer.serialize(y_hidden, from_type = bittensor.proto.TensorType.TORCH)
    y_causallm_serialized = serializer.serialize(y_causallm, from_type = bittensor.proto.TensorType.TORCH)
    y_causallmnext_serialized = serializer.serialize(y_causallmnext, from_type=bittensor.proto.TensorType.TORCH)
    y_seq_2_seq_serialized = serializer.serialize(y_seq_2_seq, from_type = bittensor.proto.TensorType.TORCH)
            
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.ss58_address,
            synapses = [synapse.serialize_to_wire_proto(code = bittensor.proto.ReturnCode.Success, message= 'Success' ) for synapse in synapses[:-1]]
            + [synapses[-1].serialize_to_wire_proto(code = bittensor.proto.ReturnCode.UnknownException, message= 'UnknownException' )],
            return_code = bittensor.proto.ReturnCode.Success,
            tensors = [y_hidden_serialized, y_causallm_serialized, y_causallmnext_serialized, y_seq_2_seq_serialized]
        )

    receptor_pool = bittensor.receptor_pool(wallet=wallet,max_active_receptors=1)
    receptor_pool._get_or_create_receptor_for_endpoint(neuron_obj)

    mock_result = asyncio.Future()
    mock_result.set_result( mock_return_val )
    receptor_pool.receptors[neuron_obj.hotkey].stub.Forward = MagicMock( return_value = mock_result )
    resp1,  codes, _ = receptor_pool.forward( endpoints, synapses, x, timeout=1)
    assert codes == [[bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.UnknownException],
    [bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.UnknownException]]

def test_receptor_pool_missing_synapse():
    endpoints = [neuron_obj,neuron_obj]
    x = torch.ones( (2, 3, 3) )    

    y_hidden = torch.rand(3, 3, bittensor.__network_dim__)
    y_causallm = torch.rand(3, 3, bittensor.__network_dim__)
    y_causallmnext = bittensor.synapse.TextCausalLMNext().nill_forward_response_tensor(torch.ones(3), encoded=True)
    y_seq_2_seq = torch.rand(3, 70)
    
    serializer = bittensor.serializer( serializer_type = bittensor.proto.Serializer.MSGPACK )
    y_hidden_serialized = serializer.serialize(y_hidden, from_type = bittensor.proto.TensorType.TORCH)
    y_causallm_serialized = serializer.serialize(y_causallm, from_type = bittensor.proto.TensorType.TORCH)
    y_causallmnext_serialized = serializer.serialize(y_causallmnext, from_type=bittensor.proto.TensorType.TORCH)
    y_seq_2_seq_serialized = serializer.serialize(y_seq_2_seq, from_type = bittensor.proto.TensorType.TORCH)
            
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.ss58_address,
            synapses = [synapse.serialize_to_wire_proto(code = bittensor.proto.ReturnCode.Success, message= 'Success' ) for synapse in synapses[:2]],
            return_code = bittensor.proto.ReturnCode.Success,
            tensors = [y_hidden_serialized, y_causallm_serialized, y_causallmnext_serialized, y_seq_2_seq_serialized]
        )

    receptor_pool = bittensor.receptor_pool(wallet=wallet,max_active_receptors=1)
    receptor_pool._get_or_create_receptor_for_endpoint(neuron_obj)
    mock_result = asyncio.Future()
    mock_result.set_result( mock_return_val )
    receptor_pool.receptors[neuron_obj.hotkey].stub.Forward = MagicMock( return_value = mock_result )
    resp1,  codes, _ = receptor_pool.forward( endpoints, synapses, x, timeout=1)
    assert codes == [[bittensor.proto.ReturnCode.ResponseShapeException, bittensor.proto.ReturnCode.ResponseShapeException, bittensor.proto.ReturnCode.ResponseShapeException, bittensor.proto.ReturnCode.ResponseShapeException],
    [bittensor.proto.ReturnCode.ResponseShapeException, bittensor.proto.ReturnCode.ResponseShapeException, bittensor.proto.ReturnCode.ResponseShapeException, bittensor.proto.ReturnCode.ResponseShapeException]]

def test_receptor_pool_backward_hang():
    endpoints = [neuron_obj,neuron_obj]
    x = [ torch.ones( (2,2) ), torch.ones( (2,2) ) ]
    mock_return_val = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__,
            hotkey = wallet.hotkey.ss58_address,
            return_code = bittensor.proto.ReturnCode.Timeout,
            tensors = [])
    
    hidden_grads = torch.ones((x[0].size(0), x[0].size(1), bittensor.__network_dim__))
    causal_grads = torch.ones((x[0].size(0), x[0].size(1), bittensor.__vocab_size__))
    causallmnext_grads = torch.ones((x[0].size(0), (bittensor.synapse.TextCausalLMNext().topk + 1), 1 + 1))
    seq_2_seq_grads = torch.tensor([])

    receptor_pool = bittensor.receptor_pool(wallet=wallet,max_active_receptors=1)
    receptor_pool._get_or_create_receptor_for_endpoint(neuron_obj)

    mock_result = asyncio.Future()
    mock_result.set_result( mock_return_val )
    receptor_pool.receptors[neuron_obj.hotkey].stub.Backward = MagicMock( return_value = mock_result )

    receptor_pool.backward(endpoints, synapses, x, [[hidden_grads, causal_grads, causallmnext_grads, seq_2_seq_grads],
                                                    [hidden_grads, causal_grads, causallmnext_grads, seq_2_seq_grads]], timeout=1)

if __name__ == "__main__":
    #test_receptor_pool_forward()
    test_receptor_pool_backward_hang()
    # test_receptor_pool_forward_success()
    #t est_receptor_pool_forward_timeout()
    pass