import bittensor
import pytest

def test_loaded_config():
    with pytest.raises(NotImplementedError):
        bittensor.Config(loaded_config=True)


def construct_config():
    defaults = bittensor.Config()
    bittensor.subtensor.add_defaults( defaults )
    bittensor.dendrite.add_defaults( defaults )
    bittensor.axon.add_defaults( defaults )
    bittensor.wallet.add_defaults( defaults )
    bittensor.dataset.add_defaults( defaults )
    bittensor.logging.add_defaults( defaults )
    bittensor.wandb.add_defaults( defaults )
    
    return defaults

def test_to_defaults():
    config = construct_config()
    config.to_defaults()