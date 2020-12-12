import bittensor
import torch
import random
import time
import concurrent.futures
from loguru import logger
from unittest.mock import MagicMock
from bittensor import bittensor_pb2
from bittensor.config import Config
from bittensor.subtensor import Keypair
from bittensor.nucleus import Nucleus
from bittensor.synapse import Synapse

config = Config.load()
mnemonic = Keypair.generate_mnemonic()
keypair = Keypair.create_from_mnemonic(mnemonic)


def test_init():
    nucleus = Nucleus(config)

def test_stop():
    nucleus = Nucleus(config)
    nucleus.stop()

def test_not_implemented():
    nucleus = Nucleus(config)
    synapse = Synapse(None, None)
    x = torch.tensor([])
    mode = bittensor_pb2.Modality.TEXT
    outputs, _, code = nucleus.forward(synapse = synapse, inputs = x, mode = mode, priority = 1)
    assert outputs == None
    assert code == bittensor_pb2.ReturnCode.NotImplemented

def test_forward_success():
    nucleus = Nucleus(config)
    synapse = Synapse(None, None)
    x = torch.rand(3, 3)
    synapse.call_forward = MagicMock(return_value = x)
    mode = bittensor_pb2.Modality.TEXT
    outputs, _, code = nucleus.forward(synapse = synapse, inputs = x, mode = mode, priority = 1)
    assert list(outputs.shape) == [3, 3]
    assert code == bittensor_pb2.ReturnCode.Success

def test_multiple_forward_success():
    nucleus = Nucleus(config)
    synapse = Synapse(None, None)
    x = torch.rand(3, 3, bittensor.__network_dim__)
    synapse.call_forward = MagicMock(return_value = x)
    mode = bittensor_pb2.Modality.TEXT
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

    assert code1 == bittensor_pb2.ReturnCode.Success
    assert code2 == bittensor_pb2.ReturnCode.Success
    assert code3 == bittensor_pb2.ReturnCode.Success
    assert code4 == bittensor_pb2.ReturnCode.Success
    assert code5 == bittensor_pb2.ReturnCode.Success

class SlowSynapse(bittensor.synapse.Synapse):
    def call_forward(self, a, b):
        time.sleep(1)

def test_queue_full():
    config.nucleus.queue_maxsize = 10
    nucleus = Nucleus(config)
    synapse = SlowSynapse(None, None)
    x = torch.rand(3, 3, bittensor.__network_dim__)
    mode = bittensor_pb2.Modality.TEXT

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
        if code == bittensor_pb2.ReturnCode.NucleusFull:
            return
    
    # One should be a timeout.
    assert False

def test_stress_test():
    config.nucleus.queue_maxsize = 10000
    nucleus = Nucleus(config)
    synapse = SlowSynapse(None, None)
    x = torch.rand(3, 3, bittensor.__network_dim__)
    mode = bittensor_pb2.Modality.TEXT

    def _call_nucleus_forward():
        _, _, code = nucleus.forward(synapse = synapse, inputs = x, mode = mode, priority = random.random())
        return code

    # Create many futures.
    futures = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for _ in range(1000):
            futures.append(executor.submit(_call_nucleus_forward))

    # All should return success fully.
    for f in futures:
        code = f.result()
        if code != bittensor_pb2.ReturnCode.Success:
            assert False
    assert True

def test_backward_success():
    nucleus = Nucleus(config)
    synapse = Synapse(None, None)
    x = torch.rand(3, 3)
    synapse.call_backward = MagicMock(return_value = x)
    mode = bittensor_pb2.Modality.TEXT
    outputs, _, code = nucleus.backward(synapse = synapse, inputs_x = x, grads_dy = x, priority = 1)
    assert list(outputs.shape) == [3, 3]
    assert code == bittensor_pb2.ReturnCode.Success

if __name__ == "__main__":
    test_init()
    test_stop()
    test_not_implemented()
    test_forward_success()
    test_multiple_forward_success()
    test_queue_full()
    test_stress_test()
    test_backward_success()
