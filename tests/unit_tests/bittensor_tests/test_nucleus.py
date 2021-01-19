import bittensor
import torch
import random
import time
import concurrent.futures
from loguru import logger
from munch import Munch
from unittest.mock import MagicMock

nucleus = None

def test_init():
    global nucleus
    nucleus = bittensor.nucleus.Nucleus()

def test_stop():
    nucleus = bittensor.nucleus.Nucleus()
    nucleus.stop()

def test_not_implemented():
    nucleus = bittensor.nucleus.Nucleus()
    synapse = bittensor.synapse.Synapse()
    x = torch.tensor([])
    mode = bittensor.proto.Modality.TEXT
    outputs, _, code = nucleus.forward(synapse = synapse, inputs = x, mode = mode, priority = 1)
    assert outputs == None
    assert code == bittensor.proto.ReturnCode.NotImplemented

def test_forward_success():
    nucleus = bittensor.nucleus.Nucleus()
    synapse = bittensor.synapse.Synapse()
    x = torch.rand(3, 3)
    synapse.call_forward = MagicMock(return_value = x)
    mode = bittensor.proto.Modality.TEXT
    outputs, _, code = nucleus.forward(synapse = synapse, inputs = x, mode = mode, priority = 1)
    assert list(outputs.shape) == [3, 3]
    assert code == bittensor.proto.ReturnCode.Success

def test_multiple_forward_success():
    nucleus = bittensor.nucleus.Nucleus()
    synapse = bittensor.synapse.Synapse()
    x = torch.rand(3, 3, bittensor.__network_dim__)
    synapse.call_forward = MagicMock(return_value = x)
    mode = bittensor.proto.Modality.TEXT
    outputs1, _, code1 = nucleus.forward(synapse = synapse, inputs = x, mode = mode, priority = 1)
    outputs2, _, code2 = nucleus.forward(synapse = synapse, inputs = x, mode = mode, priority = 1)
    outputs3, _, code3 = nucleus.forward(synapse = synapse, inputs = x, mode = mode, priority = 1)
    outputs4, _, code4 = nucleus.forward(synapse = synapse, inputs = x, mode = mode, priority = 1)
    outputs5, _, code5 = nucleus.forward(synapse = synapse, inputs = x, mode = mode, priority = 1)

    assert list(outputs1.shape) == [3, 3, bittensor.__network_dim__]
    assert list(outputs2.shape) == [3, 3, bittensor.__network_dim__]
    assert list(outputs3.shape) == [3, 3, bittensor.__network_dim__]
    assert list(outputs4.shape) == [3, 3, bittensor.__network_dim__]
    assert list(outputs5.shape) == [3, 3, bittensor.__network_dim__]

    assert code1 == bittensor.proto.ReturnCode.Success
    assert code2 == bittensor.proto.ReturnCode.Success
    assert code3 == bittensor.proto.ReturnCode.Success
    assert code4 == bittensor.proto.ReturnCode.Success
    assert code5 == bittensor.proto.ReturnCode.Success

class SlowSynapse(bittensor.synapse.Synapse):
    def call_forward(self, a, b):
        time.sleep(1)

def test_queue_full():
    config = bittensor.nucleus.Nucleus.build_config()
    config.nucleus.queue_maxsize = 10
    nucleus = bittensor.nucleus.Nucleus( config )
    synapse = SlowSynapse()
    x = torch.rand(3, 3, bittensor.__network_dim__)
    mode = bittensor.proto.Modality.TEXT

    def _call_nucleus_forward():
        _, _, code = nucleus.forward(synapse = synapse, inputs = x, mode = mode, priority = random.random())
        return code

    # Create many futures.
    futures = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for _ in range(100):
            futures.append(executor.submit(_call_nucleus_forward))

    # Check future codes, should get at least one queue full.
    for f in futures:
        code = f.result()
        if code == bittensor.proto.ReturnCode.NucleusFull:
            return
    
    # One should be a timeout.
    assert False

def test_stress_test():
    n_to_call = 100
    nucleus = bittensor.nucleus.Nucleus()
    synapse = SlowSynapse()
    nucleus.config.nucleus.queue_maxsize = 10000
    nucleus.config.nucleus.queue_timeout = n_to_call

    x = torch.rand(3, 3, bittensor.__network_dim__)
    mode = bittensor.proto.Modality.TEXT

    def _call_nucleus_forward():
        _, _, code = nucleus.forward(synapse = synapse, inputs = x, mode = mode, priority = random.random())
        return code

    # Create many futures.
    futures = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for _ in range(n_to_call):
            futures.append(executor.submit(_call_nucleus_forward))

    # All should return success fully.
    for f in futures:
        code = f.result()
        print (code)
        if code != bittensor.proto.ReturnCode.Success:
            assert False

    assert True

def test_backward_success():
    nucleus = bittensor.nucleus.Nucleus()
    synapse = bittensor.synapse.Synapse(None)
    x = torch.rand(3, 3)
    synapse.call_backward = MagicMock(return_value = x)
    mode = bittensor.proto.Modality.TEXT
    nucleus.backward(synapse = synapse, inputs_x = x, grads_dy = x, mode = mode, priority = 1)
   