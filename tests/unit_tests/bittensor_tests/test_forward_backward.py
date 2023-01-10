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
from unittest.mock import MagicMock

import pytest
import torch
from torch.autograd import Variable

import bittensor
from bittensor._endpoint import endpoint
from bittensor.utils.test_utils import get_random_unused_port

wallet = bittensor.wallet.mock()
dendrite = bittensor.dendrite(requires_grad=True)
dendrite_no_grad = bittensor.dendrite(requires_grad=False)
dendrite_mock = bittensor.dendrite(requires_grad=True)
dendrite_mock.receptor_pool.forward = MagicMock(return_value = [torch.tensor([]), [1], [0]]) 
dendrite_mock.receptor_pool.backward = MagicMock(return_value = [torch.tensor([]), [1], [0]]) 
endpoint = bittensor.endpoint(
    version = bittensor.__version_as_int__,
    uid = 0,
    hotkey = wallet.hotkey.ss58_address,
    ip = '0.0.0.0', 
    ip_type = 4, 
    port = 8080, 
    protocol = 0, 
    coldkey = wallet.coldkey.ss58_address
)

def test_dendrite_forward_causal_lm_shape_error():
    x = torch.rand(3, 3, 3)
    synapses = [bittensor.synapse.TextCausalLM()]
    with pytest.raises(ValueError):
        dendrite_mock.text( endpoints=[endpoint], inputs=[x], synapses=synapses)

def test_dendrite_forward_causal_lm_next_shape_error():
    x = torch.rand(3, 3, 3)
    synapses = [bittensor.synapse.TextCausalLMNext()]
    with pytest.raises(ValueError):
        dendrite_mock.text(endpoints=[endpoint], inputs=[x], synapses=synapses)

def test_dendrite_forward_last_hidden_shape_error():
    x = torch.rand(3, 3, 3)
    synapses = [bittensor.synapse.TextLastHiddenState()]
    with pytest.raises(ValueError):
        dendrite_mock.text( endpoints=[endpoint], inputs=[x], synapses=synapses)

def test_dendrite_forward_seq_2_seq_shape_error():
    x = torch.rand(3, 3, 3)
    synapses = [bittensor.synapse.TextSeq2Seq()]
    with pytest.raises(ValueError):
        dendrite_mock.text( endpoints=[endpoint], inputs=[x], synapses=synapses)

def test_dendrite_forward_text_causal_lm():
    x = torch.tensor([[1,2,3,4],[5,6,7,8]], dtype=torch.long)
    synapses = [bittensor.synapse.TextCausalLM()]
    dendrite_mock.receptor_pool.forward = MagicMock(return_value = [ [[torch.zeros([2, 4, bittensor.__network_dim__])]], [[1]], [[0]]]) 
    tensors, codes, times = dendrite_mock.text( endpoints=[endpoint], inputs=[x], synapses=synapses)
    assert codes[0].item() == bittensor.proto.ReturnCode.Success
    assert list(tensors[0][0].shape) == [2, 4, bittensor.__network_dim__]

def test_dendrite_forward_text_causal_lm_next():
    x = torch.LongTensor([[1, 2, 3, 4], [5, 6, 7, 8]])  # [2, 4]
    synapses = [bittensor.synapse.TextCausalLMNext()]
    dendrite_mock.receptor_pool.forward = MagicMock(return_value=[[[torch.zeros([2, (synapses[0].topk + 1), 1 + 1])]], [[1]], [[0]]])
    tensors, codes, times = dendrite_mock.text(endpoints=[endpoint], inputs=[x], synapses=synapses)
    assert codes[0].item() == bittensor.proto.ReturnCode.Success
    assert list(tensors[0][0].shape) == [2, (synapses[0].topk + 1), 1 + 1]  # [batch_size, (topk + 1), max_len]

def test_dendrite_forward_text_last_hidden():
    x = torch.tensor([[1],[8]])
    synapses = [bittensor.synapse.TextLastHiddenState()]
    dendrite_mock.receptor_pool.forward = MagicMock(return_value = [ [[torch.zeros([1, 1, bittensor.__network_dim__])]], [[1]], [[0]]]) 
    tensors, codes, times  = dendrite_mock.text( endpoints=[endpoint], inputs=[x], synapses=synapses)
    assert codes[0].item() == bittensor.proto.ReturnCode.Success
    assert list(tensors[0][0].shape) == [1, 1, bittensor.__network_dim__]

def test_dendrite_forward_text_seq_2_seq():
    x = torch.rand(3, 3)
    synapses = [bittensor.synapse.TextSeq2Seq()]
    dendrite_mock.receptor_pool.forward = MagicMock(return_value = [ [[torch.zeros([3, 3, bittensor.__network_dim__])]], [[1]], [[0]]]) 
    tensors, codes, times = dendrite_mock.text( endpoints=[endpoint], inputs=[x], synapses=synapses)
    assert codes[0].item() == bittensor.proto.ReturnCode.Success
    assert list(tensors[0][0].shape) == [3, 3, bittensor.__network_dim__]

def test_dendrite_forward_tensor_pass_through_text_causal_lm():
    x = torch.ones((3, 3), dtype=torch.int64)
    y = torch.zeros([3, 3, bittensor.__network_dim__])
    synapses = [bittensor.synapse.TextCausalLM()]
    dendrite_mock.receptor_pool.forward = MagicMock(return_value = [ [[y, y, y]] , [[1, 1, 1]], [[0,0,0]]]) 
    tensors, codes, times = dendrite_mock.text( endpoints=[endpoint, endpoint, endpoint], inputs=[x, x, x], synapses=synapses)
    assert codes[0][0].item() == bittensor.proto.ReturnCode.Success
    assert codes[1][0].item() == bittensor.proto.ReturnCode.Success
    assert codes[2][0].item() == bittensor.proto.ReturnCode.Success
    assert tensors[0][0].shape == y.shape
    assert tensors[1][0].shape == y.shape
    assert tensors[2][0].shape == y.shape

def test_dendrite_forward_tensor_pass_through_text_causal_lm_next():
    x = torch.ones((3, 3), dtype=torch.int64)
    synapses = [bittensor.synapse.TextCausalLMNext()]
    y = torch.zeros([3, (synapses[0].topk + 1), 1 + 1])
    dendrite_mock.receptor_pool.forward = MagicMock(return_value=[[[y, y, y]], [[1, 1, 1]], [[0, 0, 0]]])
    tensors, codes, times = dendrite_mock.text(endpoints=[endpoint, endpoint, endpoint], inputs=[x, x, x], synapses=synapses)
    assert codes[0][0].item() == bittensor.proto.ReturnCode.Success
    assert codes[1][0].item() == bittensor.proto.ReturnCode.Success
    assert codes[2][0].item() == bittensor.proto.ReturnCode.Success
    assert tensors[0][0].shape == y.shape
    assert tensors[1][0].shape == y.shape
    assert tensors[2][0].shape == y.shape

def test_dendrite_forward_tensor_pass_through_text_last_hidden():
    x = torch.ones((3, 3), dtype=torch.int64)
    y = torch.zeros([3, 3, bittensor.__network_dim__])
    synapses = [bittensor.synapse.TextLastHiddenState()]
    dendrite_mock.receptor_pool.forward = MagicMock(return_value = [ [[y, y, y]] , [[1, 1, 1]], [[0,0,0]]]) 
    tensors, codes, times = dendrite_mock.text( endpoints=[endpoint, endpoint, endpoint], inputs=[x, x, x], synapses=synapses)
    assert codes[0][0].item() == bittensor.proto.ReturnCode.Success
    assert codes[1][0].item() == bittensor.proto.ReturnCode.Success
    assert codes[2][0].item() == bittensor.proto.ReturnCode.Success
    assert tensors[0][0].shape == y.shape
    assert tensors[1][0].shape == y.shape
    assert tensors[2][0].shape == y.shape

def test_dendrite_forward_tensor_pass_through_text_seq_2_seq():
    x = torch.ones((3, 3), dtype=torch.int64)
    y = torch.zeros([3, 3, bittensor.__network_dim__])
    synapses = [bittensor.synapse.TextSeq2Seq()]
    dendrite_mock.receptor_pool.forward = MagicMock(return_value = [ [[y, y, y]] , [[1, 1, 1]], [[0,0,0]]]) 
    tensors, codes, times = dendrite_mock.text( endpoints=[endpoint, endpoint, endpoint], inputs=[x, x, x], synapses=synapses)
    assert codes[0][0].item() == bittensor.proto.ReturnCode.Success
    assert codes[1][0].item() == bittensor.proto.ReturnCode.Success
    assert codes[2][0].item() == bittensor.proto.ReturnCode.Success
    assert tensors[0][0].shape == y.shape
    assert tensors[1][0].shape == y.shape
    assert tensors[2][0].shape == y.shape

def test_dendrite_backward():
    x = Variable(torch.rand((2, 2), dtype=torch.float32), requires_grad=True)
    y = torch.ones((2, 2))
    dendrite_mock.receptor_pool.forward = MagicMock(return_value = [ [[y]], [[0]], [[0]]]) 
    dendrite_mock.receptor_pool.backward = MagicMock(return_value = [ [[y]], [[0]], [[0]]]) 
    dendrite_mock.format_text_inputs = MagicMock(return_value = ( [ endpoint ], [ x ] ))
    synapses = [bittensor.synapse.TextCausalLM()]
    tensors, codes, times = dendrite_mock.text( endpoints = [ endpoint ], inputs=[ x ], synapses=synapses)
    tensors[0][0].sum().backward()
    assert x.grad.shape == y.shape

def test_dendrite_backward_large():
    x = Variable(torch.rand((1, 1), dtype=torch.float32), requires_grad=True)
    y = torch.ones((1, 1))
    dendrite_mock.receptor_pool.forward = MagicMock(return_value = [ [[y]], [[0]], [[0]]]) 
    dendrite_mock.receptor_pool.backward = MagicMock(return_value = [ [[y]], [[0]], [[0]]])
    dendrite_mock.format_text_inputs = MagicMock(return_value = ( [ endpoint ], [ x ] ))
    synapses = [bittensor.synapse.TextCausalLM()]
    tensors, codes, times = dendrite_mock.text( endpoints = [ endpoint ], inputs=[ x ], synapses=synapses)
    tensors[0][0].sum().backward()
    assert x.grad.shape == y.shape
    assert x.grad.tolist() == y.tolist()

def test_dendrite_backward_no_grad():
    x = Variable(torch.rand((1, 1), dtype=torch.float32), requires_grad=True)
    y = torch.ones((1, 1))
    nill_response = torch.zeros((1, 1))
    dendrite_no_grad.receptor_pool.forward = MagicMock(return_value = [ [[y]], [[0]], [[0]]]) 
    dendrite_no_grad.receptor_pool.backward = MagicMock(return_value = [ [[y]], [[0]], [[0]]]) 
    dendrite_no_grad.format_text_inputs = MagicMock(return_value = ( [ endpoint ], [ x ] ))
    synapses = [bittensor.synapse.TextCausalLM()]
    tensors, codes, times = dendrite_no_grad.text( endpoints = [ endpoint ], inputs=[ x ], synapses=synapses)
    tensors[0][0].sum().backward()
    assert x.grad.shape == y.shape
    assert x.grad.tolist() == nill_response.tolist()


def test_dendrite_backward_multiple():
    x1 = Variable(torch.rand((1, 1), dtype=torch.float32), requires_grad=True)
    x2 = Variable(torch.rand((1, 1), dtype=torch.float32), requires_grad=True)
    x3 = Variable(torch.rand((1, 1), dtype=torch.float32), requires_grad=True)
    y1 = torch.ones(1, 1)
    y2 = torch.ones(1, 1)
    y3 = torch.ones(1, 1)

    dendrite_mock.receptor_pool.forward = MagicMock(return_value = [ [[y1], [y2], [y3]], [[1],[1],[1]], [[0],[0],[0]]]) 
    dendrite_mock.receptor_pool.backward = MagicMock(return_value = [ [[y1], [y2], [y3]], [[1],[1],[1]], [[0],[0],[0]]]) 
    dendrite_mock.format_text_inputs = MagicMock(return_value = ( [ endpoint, endpoint, endpoint ], [  x1, x2, x3  ] ))

    synapses = [bittensor.synapse.TextCausalLM()]
    tensors, codes, times = dendrite_mock.text( endpoints = [endpoint, endpoint, endpoint], inputs=[ x1, x2, x3 ], synapses=synapses)
    tensors[0][0].sum().backward()
    assert x1.grad.shape == y1.shape
    assert x2.grad.shape == y2.shape
    assert x3.grad.shape == y3.shape
    assert x1.grad.tolist() == y1.tolist()
    assert x2.grad.tolist() == y2.tolist()
    assert x3.grad.tolist() == y3.tolist()

def test_axon_receptor_forward_works():
    def forward( inputs_x: torch.FloatTensor, synapse , model_output = None):
        return None, dict(), torch.zeros( [3, 3, bittensor.__network_dim__])
    

    axon_port = get_random_unused_port()
    axon = bittensor.axon (
        port = axon_port,
        ip = '0.0.0.0',
        wallet = wallet,
        netuid = -1,
    )
    axon.attach_synapse_callback( forward, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE)
    axon.start()
    endpoint = bittensor.endpoint(
        version = bittensor.__version_as_int__,
        uid = 0,
        hotkey = wallet.hotkey.ss58_address,
        ip = '0.0.0.0',
        ip_type = 4,
        port = axon_port,
        modality = 2,
        coldkey = wallet.coldkey.ss58_address,
        protocol = 0,
    )
    endpoints = [endpoint]
    x = torch.zeros(3, 3)
    synapses = [bittensor.synapse.TextLastHiddenState()]

    tensors, codes, times = dendrite.text( endpoints=endpoints, inputs=[x for _ in endpoints], synapses=synapses)
    receptors_states = dendrite.receptor_pool.get_receptors_state()
    # TODO: Fails locally independent of multiprocessing.
    assert receptors_states[endpoint.hotkey] == receptors_states[endpoint.hotkey].READY
    assert codes[0][0].item() == bittensor.proto.ReturnCode.Success
    assert list(tensors[0][0].shape) == [3, 3, bittensor.__network_dim__]
    print('assertions passed')
    axon.stop()

def test_dendrite_call_time():
    def forward( inputs_x: torch.FloatTensor, synapse , model_output = None):
        time.sleep(12)
        return None, dict(), torch.zeros( [3, 3, bittensor.__network_dim__])
    
    axon_port = get_random_unused_port()
    axon = bittensor.axon (
        port = axon_port,
        ip = '0.0.0.0',
        wallet = wallet,
        netuid = -1,
    )
    axon.attach_synapse_callback( forward, synapse_type = bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE)
    axon.start()
    endpoints = []
    for i in range(10):
        wallet.create_new_hotkey( use_password=False, overwrite = True)
        endpoint = bittensor.endpoint(
            version = bittensor.__version_as_int__,
            uid = 0,
            hotkey = wallet.hotkey.ss58_address,
            ip = '0.0.0.0', 
            ip_type = 4, 
            port = axon_port,
            modality = 2, 
            coldkey = wallet.coldkey.ss58_address,
            protocol = 0,
        )
        endpoints += [endpoint]
    x = torch.zeros(3, 3)
    synapses = [bittensor.synapse.TextLastHiddenState()]
    start_time = time.time()

    tensors, codes, times = dendrite.text( endpoints=endpoints, inputs=[x for i in endpoints], synapses=synapses)
    total_time = time.time() - start_time
    axon.stop()

def test_dendrite_del():
    global dendrite, dendrite_no_grad, dendrite_mock
    del dendrite
    del dendrite_no_grad
    del dendrite_mock

if __name__  == "__main__":
    pass

