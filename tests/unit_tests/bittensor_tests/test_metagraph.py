import torch
import unittest
import pytest
from munch import Munch
import bittensor
from bittensor.subtensor.interface import Keypair

metagraph = None

def test_create( ):
    global metagraph
    metagraph = bittensor.metagraph.Metagraph()
    assert True

def test_convert_weight_order_should_work_last( ):
    MAX_INT_WEIGHT = 4294967295 # Max weight value on chain.
    metagraph.state.uids = torch.tensor([1,2,3,4])
    metagraph.uid = 4
    weights = [0.1,0.2,0.3,0.4]
    uids, vals = metagraph.convert_weights_to_emit(weights)
    print (uids, vals)
    assert uids == [4, 1, 2, 3]
    assert sum( vals ) == MAX_INT_WEIGHT

def test_convert_weight_order_should_work_first( ):
    MAX_INT_WEIGHT = 4294967295 # Max weight value on chain.
    metagraph.state.uids = torch.tensor([1,2,3,4])
    metagraph.uid = 1
    weights = [0.1,0.2,0.3,0.4]
    uids, vals = metagraph.convert_weights_to_emit(weights)
    print (uids, vals)
    assert uids == [1, 2, 3, 4]
    assert sum( vals ) == MAX_INT_WEIGHT

def test_convert_weight_order_no_self_uids( ):
    MAX_INT_WEIGHT = 4294967295 # Max weight value on chain.
    metagraph.state.uids = torch.tensor([1])
    metagraph.uid = 4
    weights = [0.1]
    uids, vals = metagraph.convert_weights_to_emit(weights)
    print (uids, vals)
    assert uids == [1]
    assert sum( vals ) == MAX_INT_WEIGHT

def test_convert_weight_order_no_uids( ):
    metagraph.state.uids = torch.tensor([])
    metagraph.uid = 4
    weights = []
    uids, vals = metagraph.convert_weights_to_emit(weights)
    print (uids, vals)
    assert uids == []
    assert sum( vals ) == 0

def test_convert_weight_order_negative( ):
    MAX_INT_WEIGHT = 4294967295 # Max weight value on chain.
    metagraph.state.uids = torch.tensor([1])
    metagraph.uid = 4
    weights = [-1]
    uids, vals = metagraph.convert_weights_to_emit(weights)
    print (uids, vals)
    assert uids == [1]
    assert sum( vals ) == MAX_INT_WEIGHT

def test_convert_weight_order_negative_positive( ):
    MAX_INT_WEIGHT = 4294967295 # Max weight value on chain.
    metagraph.state.uids = torch.tensor([1, 2])
    metagraph.uid = 4
    weights = [-1, 1]
    uids, vals = metagraph.convert_weights_to_emit(weights)
    print (uids, vals)
    assert uids == [1, 2]
    assert sum( vals ) == MAX_INT_WEIGHT

def test_convert_weight_order_same_uid( ):
    MAX_INT_WEIGHT = 4294967295 # Max weight value on chain.
    metagraph.state.uids = torch.tensor([1, 1])
    metagraph.uid = 4
    weights = [-1, 1]
    uids, vals = metagraph.convert_weights_to_emit(weights)
    print (uids, vals)
    assert uids == [1, 1]
    assert sum( vals ) == MAX_INT_WEIGHT

def test_convert_weight_removes_zeros( ):
    MAX_INT_WEIGHT = 4294967295 # Max weight value on chain.
    metagraph.state.uids = torch.tensor([1, 2, 3, 4])
    metagraph.uid = 4
    weights = [0, 0, 0, 1]
    uids, vals = metagraph.convert_weights_to_emit(weights)
    print (uids, vals)
    assert uids == [4]
    assert sum( vals ) == MAX_INT_WEIGHT

def test_convert_weight_zeros_adds_remainder_to_last_member( ):
    MAX_INT_WEIGHT = 4294967295 # Max weight value on chain.
    metagraph.state.uids = torch.tensor([1, 1])
    metagraph.uid = 4
    weights = [0, 0]
    uids, vals = metagraph.convert_weights_to_emit(weights)
    print (uids, vals)
    assert uids == [1]
    assert sum( vals ) == MAX_INT_WEIGHT


chain_state = None
def test_chain_state( ):
    global chain_state  
    chain_state = bittensor.metagraph.ChainState()
    
def test_add():
    global chain_state  
    chain_state.add_or_update(pubkey='0', ip=0, port=0, uid=0, ip_type=0, modality=0, lastemit=0, stake=1000000000, w_uids=[0], w_vals=[0])
    assert chain_state.index_for_uid[0] == 0
    assert chain_state.n == 1
    assert chain_state.stake[0] == 1
    assert chain_state.lastemit[0] == 0
    assert chain_state.weight_uids[0] == [0]
    assert chain_state.weight_vals[0] == [0]
    assert chain_state.neurons[0].public_key == '0'
    assert chain_state.neurons[0].port == 0
    assert chain_state.neurons[0].ip_type == 0
    assert chain_state.neurons[0].modality == 0
    assert chain_state.neurons[0].uid == 0
    assert chain_state.index_for_uid[0] == 0
    assert chain_state.index_for_pubkey['0'] == 0
    assert chain_state.pubkey_for_index[0] == '0'

def test_update( ):
    global chain_state  
    chain_state.add_or_update(pubkey='0', ip=1, port=1, uid=0, ip_type=1, modality=1, lastemit=1, stake=1000000000, w_uids=[1], w_vals=[1])
    assert chain_state.index_for_uid[0] == 0
    assert chain_state.n == 1
    assert chain_state.stake[0] == 1
    assert chain_state.lastemit[0] == 1
    assert chain_state.weight_uids[0] == [1]
    assert chain_state.weight_vals[0] == [1]
    assert chain_state.neurons[0].public_key == '0'
    assert chain_state.neurons[0].port == 1
    assert chain_state.neurons[0].ip_type == 1
    assert chain_state.neurons[0].modality == 1
    assert chain_state.neurons[0].uid == 0
    assert chain_state.index_for_uid[0] == 0
    assert chain_state.index_for_pubkey['0'] == 0
    assert chain_state.pubkey_for_index[0] == '0'

def test_update_is_inconsistent( ):
    global chain_state  
    with pytest.raises(ValueError):
        chain_state.add_or_update(pubkey='0', ip=1, port=1, uid=1, ip_type=1, modality=1, lastemit=1, stake=1000000000, w_uids=[1], w_vals=[1])
        
def test_add_many():
    global chain_state  
    n = 100
    for i in range(n):
        chain_state.add_or_update(pubkey=str(i), uid=i, ip=i, port=i, ip_type=i, modality=i, lastemit=i, stake=i*1000000000, w_uids=[i], w_vals=[1])
    assert len(chain_state.index_for_uid) == n
    assert len(chain_state.index_for_uid) == n
    assert len(chain_state.index_for_pubkey) == n
    assert len(chain_state.pubkey_for_index) == n
    assert len(chain_state.neurons) == n
    assert len(chain_state.stake) == n
    assert len(chain_state.lastemit) == n
    assert len(chain_state.weight_uids) == n
    assert len(chain_state.weight_vals) == n

state = None

def test_convert_to_torch_state():
    global state 
    global metagraph
    state = bittensor.metagraph.TorchChainState.from_cache(chain_state) 
    assert state.tau == 0.5
    assert state.block == 0
    assert state.n == 100
    assert torch.all(state.uids.eq(torch.tensor(list(range(100)))))
    assert torch.all(state.indices.eq(torch.tensor(list(range(100)))))
    assert torch.all(state.stake.eq(torch.tensor(list(range(100)))))
    assert torch.all(state.lastemit.eq(torch.tensor(list(range(100)))))
    metagraph.state = state
    assert torch.all(metagraph.uids.eq(torch.tensor(list(range(100)))))
    assert torch.all(metagraph.indices.eq(torch.tensor(list(range(100)))))
    assert torch.all(metagraph.stake.eq(torch.tensor(list(range(100)))))
    assert torch.all(metagraph.lastemit.eq(torch.tensor(list(range(100)))))
    assert torch.all(metagraph.S.eq(metagraph.stake))
    assert torch.all(metagraph.tau.eq(torch.tensor(0.5)))

    pubkeys = metagraph.public_keys
    for i in range(100):
        assert pubkeys[i] == str(i)
    for i in range(100):
        assert metagraph.uids_to_indices(torch.tensor([i])) == torch.tensor([i])

def test_uids_to_indices():
    assert torch.all(metagraph.uids_to_indices(metagraph.uids).eq(metagraph.indices))

def test_uids_to_neurons():
    for i in range(100):
        neuron = metagraph.uids_to_neurons(torch.tensor([i]))[0]
        assert neuron.public_key == str(i)

def test_neurons_to_uids():
    assert torch.all(metagraph.neurons_to_uids(metagraph.neurons).eq(metagraph.uids))


def test_ranks():
    assert torch.all(metagraph.ranks.eq(torch.tensor(list(range(100)))))
    assert torch.all(metagraph.R.eq(torch.tensor(list(range(100)))))

def test_I():
    Ipr = torch.tensor(list(range(100))) * torch.tensor(0.5) / torch.sum(torch.tensor(list(range(100))))
    assert torch.all(metagraph.I.eq(Ipr))

def test_W():
    for i in range(100):
        row = torch.zeros(100)
        row[i] = 1
        assert torch.all(metagraph.W[i, :].eq(row))

    for i in range(100):
        col = torch.zeros(100)
        col[i] = 1
        assert torch.all(metagraph.W[:, i].eq(col))

def test_row():
    for i in range(100):
        metagraph.uid = i
        row = torch.zeros(100)
        row[i] = 1
        assert torch.all(metagraph.row.eq(row))

def test_col():
    for i in range(100):
        metagraph.uid = i
        col = torch.zeros(100)
        col[i] = 1
        assert torch.all(metagraph.row.eq(col))











   
