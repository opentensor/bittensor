from bittensor.subtensor.interface import Keypair
from bittensor.metagraph import Metagraph
from munch import Munch
import torch

metagraph = Metagraph()

def test_convert_weight_order_should_work_last():
    MAX_INT_WEIGHT = 4294967295 # Max weight value on chain.
    metagraph.state.uids = torch.tensor([1,2,3,4])
    metagraph.uid = 4
    weights = [0.1,0.2,0.3,0.4]
    uids, vals = metagraph.convert_weights_to_emit(weights)
    print (uids, vals)
    assert uids == [4, 1, 2, 3]
    assert sum( vals ) == MAX_INT_WEIGHT

def test_convert_weight_order_should_work_first():
    MAX_INT_WEIGHT = 4294967295 # Max weight value on chain.
    metagraph.state.uids = torch.tensor([1,2,3,4])
    metagraph.uid = 1
    weights = [0.1,0.2,0.3,0.4]
    uids, vals = metagraph.convert_weights_to_emit(weights)
    print (uids, vals)
    assert uids == [1, 2, 3, 4]
    assert sum( vals ) == MAX_INT_WEIGHT

def test_convert_weight_order_no_self_uids():
    MAX_INT_WEIGHT = 4294967295 # Max weight value on chain.
    metagraph.state.uids = torch.tensor([1])
    metagraph.uid = 4
    weights = [0.1]
    uids, vals = metagraph.convert_weights_to_emit(weights)
    print (uids, vals)
    assert uids == [1]
    assert sum( vals ) == MAX_INT_WEIGHT

def test_convert_weight_order_no_uids():
    metagraph.state.uids = torch.tensor([])
    metagraph.uid = 4
    weights = []
    uids, vals = metagraph.convert_weights_to_emit(weights)
    print (uids, vals)
    assert uids == []
    assert sum( vals ) == 0

def test_convert_weight_order_negative():
    MAX_INT_WEIGHT = 4294967295 # Max weight value on chain.
    metagraph.state.uids = torch.tensor([1])
    metagraph.uid = 4
    weights = [-1]
    uids, vals = metagraph.convert_weights_to_emit(weights)
    print (uids, vals)
    assert uids == [1]
    assert sum( vals ) == MAX_INT_WEIGHT

def test_convert_weight_order_negative_positive():
    MAX_INT_WEIGHT = 4294967295 # Max weight value on chain.
    metagraph.state.uids = torch.tensor([1, 2])
    metagraph.uid = 4
    weights = [-1, 1]
    uids, vals = metagraph.convert_weights_to_emit(weights)
    print (uids, vals)
    assert uids == [1, 2]
    assert sum( vals ) == MAX_INT_WEIGHT

def test_convert_weight_order_same_uid():
    MAX_INT_WEIGHT = 4294967295 # Max weight value on chain.
    metagraph.state.uids = torch.tensor([1, 1])
    metagraph.uid = 4
    weights = [-1, 1]
    uids, vals = metagraph.convert_weights_to_emit(weights)
    print (uids, vals)
    assert uids == [1, 1]
    assert sum( vals ) == MAX_INT_WEIGHT


def test_convert_weight_removes_zeros():
    MAX_INT_WEIGHT = 4294967295 # Max weight value on chain.
    metagraph.state.uids = torch.tensor([1, 2, 3, 4])
    metagraph.uid = 4
    weights = [0, 0, 0, 1]
    uids, vals = metagraph.convert_weights_to_emit(weights)
    print (uids, vals)
    assert uids == [4]
    assert sum( vals ) == MAX_INT_WEIGHT

def test_convert_weight_zeros_adds_remainder_to_last_member():
    MAX_INT_WEIGHT = 4294967295 # Max weight value on chain.
    metagraph.state.uids = torch.tensor([1, 1])
    metagraph.uid = 4
    weights = [0, 0]
    uids, vals = metagraph.convert_weights_to_emit(weights)
    print (uids, vals)
    assert uids == [1]
    assert sum( vals ) == MAX_INT_WEIGHT


test_convert_weight_zeros_adds_remainder_to_last_member()