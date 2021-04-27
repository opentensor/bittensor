import bittensor
from munch import Munch
import pytest
import os

wallet = bittensor.wallet.Wallet (
    path = '/tmp/pytest',
    name = 'pytest',
    hotkey = 'pytest',
)
subtensor = bittensor.subtensor.Subtensor (
    network = 'boltzmann'
)
metagraph = bittensor.metagraph.Metagraph (
    subtensor = subtensor
)
executor = bittensor.executor.Executor( 
    wallet = wallet,
    subtensor = subtensor,
    metagraph = metagraph,  
)

try:
    os.remove(executor.wallet.coldkeyfile)
    os.remove(executor.wallet.hotkeyfile)
except:
    pass

def test_create_hotkey():
    executor.create_new_hotkey(
        n_words = 12,
        use_password=False
    )
    assert os.path.isfile(executor.wallet.hotkeyfile) 
    os.remove(executor.wallet.hotkeyfile)
    assert not os.path.isfile(executor.wallet.hotkeyfile) 

def test_create_coldkey():
    executor.create_new_coldkey(
        n_words = 12,
        use_password=False
    )
    assert os.path.isfile(executor.wallet.coldkeyfile) 
    os.remove(executor.wallet.coldkeyfile)
    os.remove(executor.wallet.coldkeypubfile)
    assert not os.path.isfile(executor.wallet.coldkeyfile) 
    assert not os.path.isfile(executor.wallet.coldkeypubfile) 

def test_regenerate_coldkey():
    executor.wallet.config.wallet.coldkey = 'pytest3'
    executor.regenerate_coldkey(
        mnemonic = ["cabin", "thing", "arch", "canvas", "game", "park", "motion", "snack", "advice", "arch", "parade", "climb"],
        use_password=False
    )
    assert os.path.isfile(executor.wallet.coldkeyfile) 
    os.remove(executor.wallet.coldkeyfile)
    os.remove(executor.wallet.coldkeypubfile)
    assert not os.path.isfile(executor.wallet.coldkeyfile) 
    assert not os.path.isfile(executor.wallet.coldkeypubfile) 

def test_regenerate_hotkey():
    executor.wallet.config.wallet.coldkey = 'pytest4'
    executor.regenerate_hotkey(
        mnemonic = ["cabin", "thing", "arch", "canvas", "game", "park", "motion", "snack", "advice", "arch", "parade", "climb"],
        use_password=False
    )
    assert os.path.isfile(executor.wallet.hotkeyfile) 
    os.remove(executor.wallet.hotkeyfile)
    assert not os.path.isfile(executor.wallet.hotkeyfile) 
