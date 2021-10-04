from sys import version
import grpc
import torch
import bittensor

from unittest.mock import MagicMock

# --- Receptor Pool ---

wallet =  bittensor.wallet(
    path = '/tmp/pytest',
    name = 'pytest',
    hotkey = 'pytest',
) 
wallet.create_new_coldkey(use_password=False, overwrite = True)
wallet.create_new_hotkey(use_password=False, overwrite = True)

neuron_obj = bittensor.endpoint(
    version = bittensor.__version_as_int__,
    uid = 0,
    ip = '0.0.0.0',
    ip_type = 4,
    port = 12345,
    hotkey = wallet.hotkey.public_key,
    coldkey = wallet.coldkey.public_key,
    modality = 0
)

receptor_pool = bittensor.receptor_pool(wallet=wallet)


def test_receptor_pool_forward():
    endpoints = [neuron_obj]
    x = torch.ones( (1,2,2) )
    resp1,  _, _ = receptor_pool.forward( endpoints, x, bittensor.proto.Modality.TENSOR, timeout=1)
    assert list(torch.stack(resp1, dim=0).shape) == [1, 2, 2, bittensor.__network_dim__]

def test_receptor_pool_backward():
    endpoints = [neuron_obj]
    x = torch.ones( (1,2,2) )
    resp1,  _, _ = receptor_pool.backward( endpoints, x,x, bittensor.proto.Modality.TENSOR, timeout=1)
    assert list(torch.stack(resp1, dim=0).shape) == [1, 2, 2, bittensor.__network_dim__]


def test_receptor_pool_max_workers_forward():
    neuron_obj2 = bittensor.endpoint(
        version = bittensor.__version_as_int__,
        uid = 0,
        ip = '0.0.0.1',
        ip_type = 4,
        port = 12345,
        hotkey = wallet.hotkey.public_key,
        coldkey = wallet.coldkey.public_key,
        modality = 0
    )
    receptor_pool = bittensor.receptor_pool(wallet=wallet,max_active_receptors=1)
    endpoints = [neuron_obj,neuron_obj2]
    x = torch.ones( (2,2,2) )
    resp1,  _, _ = receptor_pool.forward( endpoints, x, bittensor.proto.Modality.TENSOR, timeout=1)
    assert list(torch.stack(resp1, dim=0).shape) == [2, 2, 2, bittensor.__network_dim__]
