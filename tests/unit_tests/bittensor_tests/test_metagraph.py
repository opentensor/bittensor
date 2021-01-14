from bittensor.subtensor.interface import Keypair
from bittensor.metagraph import Metagraph
from munch import Munch
import torch

config = {'session':
              {'datapath': 'data/', 'learning_rate': 0.01, 'momentum': 0.9, 'batch_size_train': 64,
               'batch_size_test': 64, 'log_interval': 10, 'sync_interval': 100, 'priority_interval': 100,
               'name': 'mnist', 'trial_id': '1608070667'},
          'synapse': {'target_dim': 10},
          'dendrite': {'key_dim': 100, 'topk': 10, 'stale_emit_filter': 10000, 'pass_gradients': True, 'timeout': 0.5,
                       'do_backoff': True, 'max_backoff': 100}, 'axon': {'local_port': 8091, 'external_ip': '191.97.53.53', 'max_workers': 5, 'max_gradients': 1000},
          'nucleus': {'max_workers': 5, 'queue_timeout': 5, 'queue_maxsize': 1000},
          'metagraph': {'chain_endpoint': '206.189.254.5:12345', 'stale_emit_filter': 10000},
          'meta_logger': {'log_dir': 'data/'},
          'neuron': {'keyfile': None, 'keypair': None }
          }


config = Munch.fromDict(config)
mnemonic = Keypair.generate_mnemonic()
keypair = Keypair.create_from_mnemonic(mnemonic)
config.neuron.keypair = keypair
metagraph = Metagraph(config)

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