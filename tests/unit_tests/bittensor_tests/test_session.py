
from loguru import logger
import bittensor
from bittensor.config import Config
from bittensor.subtensor.interface import Keypair
from munch import Munch

def new_session():
    # 1. Init Config item.
    config = {'neuron':
                  {'datapath': 'data/', 'learning_rate': 0.01, 'momentum': 0.9, 'batch_size_train': 64,
                   'batch_size_test': 64, 'log_interval': 10, 'sync_interval': 100, 'priority_interval': 100,
                   'name': 'mnist', 'trial_id': '1608070667'},
              'synapse': {'target_dim': 10},
              'dendrite': {'key_dim': 100, 'topk': 10, 'stale_emit_filter': 10000, 'pass_gradients': True,
                           'timeout': 0.5,
                           'do_backoff': True, 'max_backoff': 100}, 'axon': {'local_port': 8091, 'external_ip': '191.97.53.53', 'max_workers': 5, 'max_gradients': 1000},
              'nucleus': {'max_workers': 5, 'queue_timeout': 5, 'queue_maxsize': 1000},
              'metagraph': {'chain_endpoint': '206.189.254.5:12345', 'stale_emit_filter': 10000},
              'meta_logger': {'log_dir': 'data/'},
              'session': {'keyfile': None, 'keypair': None}
              }

    config = Munch.fromDict(config)

    logger.info(Config.toString(config))
    mnemonic = Keypair.generate_mnemonic()
    keypair = Keypair.create_from_mnemonic(mnemonic)
    session = bittensor.init(config)
    session.keypair = keypair
    return session

def test_new_session():
    new_session()

if __name__ == "__main__":
    test_new_session()