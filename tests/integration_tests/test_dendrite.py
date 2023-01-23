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
import torch
import pytest
import bittensor
from bittensor._proto.bittensor_pb2 import UnknownException
from bittensor.utils.test_utils import get_random_unused_port
from . import constant

wallet = bittensor.wallet.mock()
dendrite = bittensor.dendrite( wallet = wallet )
port = get_random_unused_port()

neuron_obj = bittensor.endpoint(
    version = bittensor.__version_as_int__,
    uid = 0,
    ip = '0.0.0.0',
    ip_type = 4,
    port = port,
    hotkey = dendrite.wallet.hotkey.ss58_address,
    coldkey = dendrite.wallet.coldkey.ss58_address,
    modality = 0,
    protocol = 0,
)

synapses = [bittensor.synapse.TextLastHiddenState(),
            bittensor.synapse.TextCausalLM(),
            bittensor.synapse.TextCausalLMNext(),
            bittensor.synapse.TextSeq2Seq(num_to_generate=constant.synapse.num_to_generate)]

dataset = bittensor.dataset(num_batches=10, _mock=True)

def check_resp_shape(resp, num_resp, block_size, seq_len):
    assert len(resp) == num_resp
    assert list(resp[0][0].shape) == [block_size, seq_len, bittensor.__network_dim__]
    assert list(resp[0][1].shape) == [block_size, seq_len, bittensor.__vocab_size__]
    assert list(resp[0][2].shape) == [block_size, (synapses[2].topk + 1), 1 + 1]
    assert list(resp[0][3].shape) == [block_size, constant.synapse.num_to_generate]
    
def test_dendrite_forward_text_endpoints_tensor():
    endpoints = neuron_obj.to_tensor()
    x = torch.tensor( [[ 1,2,3 ], [ 1,2,3 ]] )
    resp,  _, _ = dendrite.text( endpoints = endpoints, inputs = x, synapses = synapses )
    check_resp_shape(resp,  num_resp = 1, block_size = 2, seq_len = 3 )
    assert dendrite.stats.total_requests == 1
    dendrite.to_wandb()

def test_dendrite_forward_text_multiple_endpoints_tensor():
    endpoints_1 = neuron_obj.to_tensor()
    endpoints_2 = neuron_obj.to_tensor()
    endpoints = torch.stack( [endpoints_1, endpoints_2], dim=0)
    x = torch.tensor( [[ 1,2,3 ], [ 1,2,3 ]] )
    resp,  _, _ = dendrite.text( endpoints = endpoints, inputs = x, synapses = synapses )
    check_resp_shape(resp,  num_resp = 2, block_size = 2, seq_len = 3 )

def test_dendrite_forward_text_multiple_endpoints_tensor_list():
    endpoints_1 = neuron_obj.to_tensor()
    endpoints_2 = neuron_obj.to_tensor()
    endpoints_3 = neuron_obj.to_tensor()
    endpoints = [torch.stack( [endpoints_1, endpoints_2], dim=0), endpoints_3]
    x = torch.tensor( [[ 1,2,3 ], [ 1,2,3 ]] )
    resp,  _, _ = dendrite.text( endpoints = endpoints, inputs = x, synapses = synapses )
    check_resp_shape(resp,  num_resp = 3, block_size = 2, seq_len = 3 )    

def test_dendrite_forward_text_singular():
    x = torch.tensor( [[ 1,2,3 ], [ 1,2,3 ]] )
    resp,  _, _ = dendrite.text( endpoints = [neuron_obj], inputs = x, synapses = synapses )
    check_resp_shape(resp,  num_resp = 1, block_size = 2, seq_len = 3 )

    resp,  _, _ = dendrite.text( endpoints = [neuron_obj], inputs = [x], synapses = synapses )
    check_resp_shape(resp,  num_resp = 1, block_size = 2, seq_len = 3 )

    resp,  _, _ = dendrite.text( endpoints = [neuron_obj, neuron_obj], inputs = x, synapses = synapses )
    check_resp_shape(resp,  num_resp = 2, block_size = 2, seq_len = 3 )

    with pytest.raises(ValueError):
        dendrite.text( endpoints = [neuron_obj, neuron_obj], inputs = [x], synapses = synapses )

def test_dendrite_forward_text_singular_no_batch_size():
    x = torch.tensor( [ 1,2,3 ] )
    resp,  _, _ = dendrite.text( endpoints = [neuron_obj], inputs = x, synapses = synapses )
    check_resp_shape(resp,  num_resp = 1, block_size = 1, seq_len = 3 )

    resp,  _, _ = dendrite.text( endpoints = [neuron_obj], inputs = [x], synapses = synapses )
    check_resp_shape(resp,  num_resp = 1, block_size = 1, seq_len = 3 )

    resp,  _, _ = dendrite.text( endpoints = [neuron_obj, neuron_obj], inputs = x, synapses = synapses )
    check_resp_shape(resp,  num_resp = 2, block_size = 1, seq_len = 3 )

    with pytest.raises(ValueError):
        dendrite.text( endpoints = [neuron_obj, neuron_obj], inputs = [x], synapses = synapses )

def test_dendrite_forward_text_tensor_list_singular():
    x = [ torch.tensor( [ 1,2,3 ] ) for _ in range(2) ]
    with pytest.raises(ValueError):
        resp,  _, _ = dendrite.text( endpoints = [neuron_obj], inputs = x, synapses = synapses )
    resp,  _, _ = dendrite.text( endpoints = [neuron_obj, neuron_obj], inputs = x, synapses = synapses )
    check_resp_shape(resp,  num_resp = 2, block_size = 1, seq_len = 3 )

def test_dendrite_forward_text_tensor_list():
    x = [ torch.tensor( [[ 1,2,3 ], [ 1,2,3 ]] ) for _ in range(2) ]
    with pytest.raises(ValueError):
        resp,  _, _ = dendrite.text( endpoints = [neuron_obj], inputs = x, synapses = synapses )
    resp,  _, _ = dendrite.text( endpoints = [neuron_obj, neuron_obj], inputs = x, synapses = synapses )
    check_resp_shape(resp,  num_resp = 2, block_size = 2, seq_len = 3 )

def test_dendrite_forward_text_singular_string():
    x = "the cat"
    resp,  _, _ = dendrite.text( endpoints = [neuron_obj], inputs = x, synapses = synapses )
    check_resp_shape(resp,  num_resp = 1, block_size = 1, seq_len = 2 )

    resp,  _, _ = dendrite.text( endpoints = [neuron_obj], inputs = [x], synapses = synapses )
    check_resp_shape(resp,  num_resp = 1, block_size = 1, seq_len = 2 )

    resp,  _, _ = dendrite.text( endpoints = [neuron_obj, neuron_obj], inputs = x, synapses = synapses )
    check_resp_shape(resp,  num_resp = 2, block_size = 1, seq_len = 2 )

    resp,  _, _ = dendrite.text( endpoints = [neuron_obj, neuron_obj], inputs = [x], synapses = synapses )
    check_resp_shape(resp,  num_resp = 2, block_size = 1, seq_len = 2 )

def test_dendrite_forward_text_list_string():
    x = ["the cat", 'the dog', 'the very long sentence that needs to be padded']
    resp, _, _ = dendrite.text( endpoints = [neuron_obj], inputs = x, synapses = synapses )
    check_resp_shape(resp,  num_resp = 1, block_size = 3, seq_len = 9 )

    resp,  _, _ = dendrite.text( endpoints = [neuron_obj, neuron_obj], inputs = x, synapses = synapses )
    check_resp_shape(resp,  num_resp = 2, block_size = 3, seq_len = 9 )

def test_dendrite_forward_tensor_shape_error():
    x = torch.rand(3, 3, 3, dtype=torch.float32)
    with pytest.raises(ValueError):
        dendrite.text( endpoints = [neuron_obj], inputs = [x], synapses = synapses)

def test_dendrite_forward_tensor_type_error():
    x = torch.zeros(3, 3, bittensor.__network_dim__, dtype=torch.int32)
    with pytest.raises(ValueError):
        dendrite.text( endpoints = [neuron_obj], inputs = x, synapses = synapses)

def test_dendrite_forward_tensor_endpoint_type_error():
    x = torch.rand(3, 3, bittensor.__network_dim__, dtype=torch.float32)
    with pytest.raises(ValueError):
        dendrite.text( endpoints = [dict()], inputs = [x], synapses = synapses)

def test_dendrite_forward_tensor_endpoint_len_error():
    x = torch.rand(3, 3, bittensor.__network_dim__, dtype=torch.float32)
    with pytest.raises(ValueError):
        dendrite.text( endpoints = [], inputs = [x], synapses = synapses)

def test_dendrite_forward_tensor_input_len_error():
    x = torch.rand(3, 3, bittensor.__network_dim__, dtype=torch.float32)
    with pytest.raises(ValueError):
        dendrite.text( endpoints = [neuron_obj], inputs = [], synapses = synapses)

def test_dendrite_forward_tensor_mismatch_len_error():
    x = torch.rand(3, 3, bittensor.__network_dim__, dtype=torch.float32)
    with pytest.raises(ValueError):
        dendrite.text( endpoints = [neuron_obj], inputs = [x, x], synapses = synapses)

def test_dendrite_forward_text_non_list():
    x = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    out, ops, times = dendrite.text( endpoints = neuron_obj, inputs = x, synapses = synapses )
    assert list(ops[0]) == [bittensor.proto.ReturnCode.Unavailable] * len(synapses)
    check_resp_shape(out, 1,2,4)

def test_dendrite_forward_text():
    x = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    out, ops, times = dendrite.text( endpoints = [neuron_obj], inputs = [x], synapses = synapses )
    assert list(ops[0]) == [bittensor.proto.ReturnCode.Unavailable] * len(synapses)
    check_resp_shape(out, 1,2,4)

def test_dendrite_forward_tensor():
    x = torch.rand(3, 3, dtype=torch.float32)
    out, ops, times = dendrite.text( endpoints = [neuron_obj], inputs = [x], synapses = synapses)
    assert list(ops[0]) == [bittensor.proto.ReturnCode.Unavailable] * len(synapses)
    check_resp_shape(out, 1, 3, 3)

def test_dendrite_backoff():
    _dendrite = bittensor.dendrite( wallet = wallet )
    port = get_random_unused_port()
    _endpoint_obj = bittensor.endpoint(
        version = bittensor.__version_as_int__,
        uid = 0,
        ip = '0.0.0.0',
        ip_type = 4,
        port = port,
        hotkey = _dendrite.wallet.hotkey.ss58_address,
        coldkey = _dendrite.wallet.coldkey.ss58_address,
        modality = 0,
        protocol = 0,
    )
    
    # Normal call.
    x = torch.rand(3, 3, dtype=torch.float32)
    out, ops, times = _dendrite.text( endpoints = [_endpoint_obj], inputs = [x], synapses = synapses)
    assert list(ops[0]) == [bittensor.proto.ReturnCode.Unavailable] * len(synapses)
    check_resp_shape(out, 1, 3, 3)
    del _dendrite


def test_dendrite_to_df():
    dendrite.to_dataframe(bittensor.metagraph(_mock=True).sync())

def test_successful_synapse():
    wallet = bittensor.wallet()
    def forward_generate( inputs_x, synapse, model_output = None):
        return None, None, torch.rand(inputs_x.shape[0], synapse.num_to_generate)

    def forward_hidden_state( inputs_x, synapse, model_output = None):
        return None, None, torch.rand(inputs_x.shape[0], inputs_x.shape[1], bittensor.__network_dim__)

    def forward_casual_lm(inputs_x, synapse, model_output = None):
        return None, None, torch.rand(inputs_x.shape[0], inputs_x.shape[1], bittensor.__vocab_size__)

    def forward_casual_lm_next(inputs_x, synapse, model_output=None):
        return None, None, synapse.nill_forward_response_tensor(inputs_x)

    port = get_random_unused_port()
    axon = bittensor.axon (
        port = port,
        ip = '0.0.0.0',
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
        hotkey = wallet.hotkey.ss58_address,
        ip = '0.0.0.0', 
        ip_type = 4, 
        port = port, 
        modality = 0, 
        coldkey = wallet.coldkeypub.ss58_address,
        protocol = 0,
    )

    dendrite = bittensor.dendrite()
    inputs = next(dataset)
    synapses = [bittensor.synapse.TextLastHiddenState(), bittensor.synapse.TextCausalLM(),
                bittensor.synapse.TextCausalLMNext(), bittensor.synapse.TextSeq2Seq(num_to_generate=20)]

    return_tensors, codes, times = dendrite.text( endpoints=endpoint, inputs = inputs, synapses=synapses)
    axon.stop()

    print(codes)
    assert list(codes[0]) == [bittensor.proto.ReturnCode.Success] * len(synapses)
    
def test_failing_synapse():
    wallet = bittensor.wallet()
    def faulty( inputs_x, synapse, model_output = None):
        raise UnknownException

    def forward_hidden_state( inputs_x, synapse, model_output = None):
        return None, None, torch.rand(inputs_x.shape[0], inputs_x.shape[1], bittensor.__network_dim__)

    def forward_casual_lm(inputs_x, synapse, model_output = None):
        return None, None, torch.rand(inputs_x.shape[0], inputs_x.shape[1], bittensor.__vocab_size__)

    def forward_casual_lm_next(inputs_x, synapse, model_output=None):
        return None, None, synapse.nill_forward_response_tensor(inputs_x)

    port = get_random_unused_port()
    axon = bittensor.axon (
        port = port,
        ip = '0.0.0.0',
        wallet = wallet,
        netuid = -1,
    )

    axon.attach_synapse_callback( forward_hidden_state,  synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE )
    axon.attach_synapse_callback( forward_casual_lm,  synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM )
    axon.attach_synapse_callback(forward_casual_lm_next, synapse_type=bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT)
    axon.attach_synapse_callback(faulty, synapse_type=bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ)
    axon.start()

    endpoint = bittensor.endpoint(
        version = bittensor.__version_as_int__,
        uid = 0,
        hotkey = wallet.hotkey.ss58_address,
        ip = '0.0.0.0', 
        ip_type = 4, 
        port = port, 
        modality = 0, 
        coldkey = wallet.coldkeypub.ss58_address,
        protocol = 0,
    )

    dendrite = bittensor.dendrite()
    inputs = next(dataset)
    synapses = [bittensor.synapse.TextLastHiddenState(), bittensor.synapse.TextCausalLM(),
                bittensor.synapse.TextCausalLMNext(), bittensor.synapse.TextSeq2Seq(num_to_generate=20)]

    return_tensors, codes, times = dendrite.text( endpoints=endpoint, inputs = inputs, synapses=synapses)
    assert list(codes[0]) == [bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.Success,
                              bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.UnknownException]

    axon.attach_synapse_callback( faulty,  synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE )
    return_tensors, codes, times = dendrite.text( endpoints=endpoint, inputs = inputs, synapses=synapses)
    assert list(codes[0]) == [bittensor.proto.ReturnCode.UnknownException, bittensor.proto.ReturnCode.Success,
                              bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.UnknownException]

    axon.attach_synapse_callback( faulty,  synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM )
    return_tensors, codes, times = dendrite.text( endpoints=endpoint, inputs = inputs, synapses=synapses)
    assert list(codes[0]) == [bittensor.proto.ReturnCode.UnknownException, bittensor.proto.ReturnCode.UnknownException,
                              bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.UnknownException]

    axon.attach_synapse_callback(faulty,  synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT)
    return_tensors, codes, times = dendrite.text(endpoints=endpoint, inputs=inputs, synapses=synapses)
    assert list(codes[0]) == [bittensor.proto.ReturnCode.UnknownException] * len(synapses)
    
    axon.stop()

def test_missing_synapse():
    wallet = bittensor.wallet()
    def forward_hidden_state( inputs_x, synapse, model_output = None):
        return None, None, torch.rand(inputs_x.shape[0], inputs_x.shape[1], bittensor.__network_dim__)

    def forward_casual_lm(inputs_x, synapse, model_output = None):
        return None, None, torch.rand(inputs_x.shape[0], inputs_x.shape[1], bittensor.__vocab_size__)

    def forward_casual_lm_next(inputs_x, synapse, model_output=None):
        return None, None, synapse.nill_forward_response_tensor(inputs_x)

    port = get_random_unused_port()
    axon = bittensor.axon (
        port = port,
        ip = '0.0.0.0',
        wallet = wallet,
        netuid = -1,
    )

    axon.start()

    endpoint = bittensor.endpoint(
        version = bittensor.__version_as_int__,
        uid = 0,
        hotkey = wallet.hotkey.ss58_address,
        ip = '0.0.0.0', 
        ip_type = 4, 
        port = port, 
        modality = 0, 
        coldkey = wallet.coldkeypub.ss58_address,
        protocol = 0,
    )

    dendrite = bittensor.dendrite()
    inputs = next(dataset)
    synapses = [bittensor.synapse.TextLastHiddenState(), bittensor.synapse.TextCausalLM(),
                bittensor.synapse.TextCausalLMNext(), bittensor.synapse.TextSeq2Seq(num_to_generate=20)]

    return_tensors, codes, times = dendrite.text( endpoints=endpoint, inputs = inputs, synapses=synapses)
    assert list(codes[0]) == [bittensor.proto.ReturnCode.NotImplemented] * len(synapses)

    axon.attach_synapse_callback( forward_hidden_state,  synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE )
    return_tensors, codes, times = dendrite.text( endpoints=endpoint, inputs = inputs, synapses=synapses)
    assert list(codes[0]) == [bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.NotImplemented,
                              bittensor.proto.ReturnCode.NotImplemented, bittensor.proto.ReturnCode.NotImplemented]

    axon.attach_synapse_callback( forward_casual_lm,  synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM )
    return_tensors, codes, times = dendrite.text( endpoints=endpoint, inputs = inputs, synapses=synapses)
    assert list(codes[0]) == [bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.Success,
                              bittensor.proto.ReturnCode.NotImplemented, bittensor.proto.ReturnCode.NotImplemented]

    axon.attach_synapse_callback(forward_casual_lm_next, synapse_type=bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT)
    return_tensors, codes, times = dendrite.text( endpoints=endpoint, inputs=inputs, synapses=synapses)
    assert list(codes[0]) == [bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.Success,
                              bittensor.proto.ReturnCode.Success, bittensor.proto.ReturnCode.NotImplemented]

    axon.stop()

def test_dendrite_timeout():
    wallet = bittensor.wallet()
    def forward_hidden_state( inputs_x, synapse, model_output = None):
        time.sleep(3)
        return None, None, torch.rand(inputs_x.shape[0], inputs_x.shape[1], bittensor.__network_dim__)

    def forward_casual_lm(inputs_x, synapse, model_output = None):
        time.sleep(3)
        return None, None, torch.rand(inputs_x.shape[0], inputs_x.shape[1], bittensor.__vocab_size__)

    def forward_casual_lm_next(inputs_x, synapse, model_output=None):
        time.sleep(3)
        return None, None, synapse.nill_forward_response_tensor(inputs_x)

    port = get_random_unused_port()
    axon = bittensor.axon (
        port = port,
        ip = '0.0.0.0',
        wallet = wallet,
        netuid = -1,
    )

    axon.start()

    endpoint = bittensor.endpoint(
        version = bittensor.__version_as_int__,
        uid = 0,
        hotkey = wallet.hotkey.ss58_address,
        ip = '0.0.0.0', 
        ip_type = 4, 
        port = port, 
        modality = 0, 
        coldkey = wallet.coldkeypub.ss58_address,
        protocol = 0,
    )

    dendrite = bittensor.dendrite()
    inputs = next(dataset)
    synapses = [bittensor.synapse.TextLastHiddenState(), bittensor.synapse.TextCausalLM(),
                bittensor.synapse.TextCausalLMNext()]

    axon.attach_synapse_callback( forward_hidden_state,  synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE )
    axon.attach_synapse_callback( forward_casual_lm,  synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM )
    axon.attach_synapse_callback(forward_casual_lm_next, synapse_type=bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT)

    return_tensors, codes, times = dendrite.text( endpoints=endpoint, inputs=inputs, synapses=synapses, timeout = 2)
    assert list(codes[0]) == [bittensor.proto.ReturnCode.Timeout, bittensor.proto.ReturnCode.Timeout,
                              bittensor.proto.ReturnCode.Timeout]

    axon.stop()

def test_dend_del():
    dendrite.__del__()

def test_clear():
    dataset.close()
    
if __name__ == "__main__":
    test_dendrite_timeout()