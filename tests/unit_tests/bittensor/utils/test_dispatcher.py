from bittensor import bittensor_pb2_grpc as bittensor_grpc
from bittensor import bittensor_pb2
import bittensor

import os
import pytest
import random
import torch

def random_synapse():
    private_key = bittensor.crypto.Crypto.generate_private_ed25519()
    public_key = bittensor.crypto.Crypto.public_key_from_private(private_key)
    synapse = bittensor_pb2.Synapse(
        version = bittensor.__version__,
        neuron_key = bittensor.crypto.Crypto.public_key_to_string(public_key),
        synapse_key = bittensor.crypto.Crypto.public_key_to_string(public_key),
        address = '0.0.0.0',
        port = '12231',
        block_hash = None
    )
    return synapse

def test_init():
    bittensor.Router(x_dim = 10, key_dim = 100, topk = 10)
    assert True

def test_router_correct():  
    input_dim = 11
    output_dim = 13
    context_dim = 10
    topk = 10
    key_dim = 100
    n_synapses = 10
    batch_size = 2

    router = bittensor.Router(x_dim = context_dim, key_dim = key_dim, topk = topk)
     
    input_shape = [batch_size, input_dim]
    context_shape = [batch_size, context_dim]
    output_shape = [batch_size, output_dim]
    
    inputs  = torch.rand(input_shape)
    context = torch.rand(context_shape)
    outputs = [torch.rand(output_shape) for _ in range(topk)]
    synapses = [random_synapse() for _ in range(n_synapses)]

    _ , _ = router.route( synapses, context, inputs )
    _ = router.join( outputs )
    assert True


def test_router_large_inputs():  
    output_dim = 13
    context_dim = 10
    topk = 10
    key_dim = 100
    n_synapses = 10
    batch_size = 2
    router = bittensor.Router(x_dim = context_dim, key_dim = key_dim, topk = topk)
     
    n_tests = 10
    for _ in range(n_tests):
        input_shape = [batch_size] + [random.randint(1, 10) for _ in range(5)]
        context_shape = [batch_size, context_dim]
        output_shape = [batch_size] + [random.randint(1, 10) for _ in range(5)]
        
        inputs  = torch.rand(input_shape)
        context = torch.rand(context_shape)
        outputs = [torch.rand(output_shape) for _ in range(topk)]
        synapses = [random_synapse() for _ in range(n_synapses)]

        # Check routing shapes
        requests , _ = router.route( synapses, context, inputs )
        for i, dim in enumerate(requests[0].shape[1:]):
            assert dim == input_shape[i+1]

        # Check output shapes
        output = router.join( outputs )
        for i, dim in enumerate(output.shape[1:]):
            assert dim == output_shape[i+1]
    assert True

def test_router_fail_context_size():  
    input_dim = 11
    context_dim = 10
    topk = 10
    key_dim = 100
    n_synapses = 10
    batch_size = 2

    router = bittensor.Router(x_dim = context_dim, key_dim = key_dim, topk = topk)
     
    input_shape = [batch_size, input_dim]
    context_shape = [batch_size, context_dim + 1]
    
    inputs  = torch.rand(input_shape)
    context = torch.rand(context_shape)
    synapses = [random_synapse() for _ in range(n_synapses)]
    with pytest.raises(ValueError, match=r"Ensure that x.size"):
        _ , _ = router.route( synapses, context, inputs )


if __name__ == "__main__": 
    test_router_fail_context_size()
    test_router_large_inputs()
    test_router_correct()