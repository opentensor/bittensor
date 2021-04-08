import bittensor
from munch import Munch
import pytest
import os

wallet = bittensor.Wallet (
    wallet_path = 'pytest',
    wallet_name = 'pytest',
    wallet_hotkey = 'pytest',
)
subtensor = bittensor.Subtensor (
    subtensor_network = 'boltzmann'
)
metagraph = bittensor.Subtensor (
    subtensor = subtensor
)
executor = bittensor.Executor( 
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
    assert not os.path.isfile(executor.wallet.coldkeyfile) 

def test_regenerate_coldkey():
    executor.wallet.config.wallet.coldkey = 'pytest3'
    executor.regenerate_coldkey(
        mnemonic = ["cabin", "thing", "arch", "canvas", "game", "park", "motion", "snack", "advice", "arch", "parade", "climb"],
        use_password=False
    )
    assert os.path.isfile(executor.wallet.coldkeyfile) 
    os.remove(executor.wallet.coldkeyfile)
    assert not os.path.isfile(executor.wallet.coldkeyfile) 

def test_regenerate_hotkey():
    executor.wallet.config.wallet.coldkey = 'pytest4'
    executor.regenerate_hotkey(
        mnemonic = ["cabin", "thing", "arch", "canvas", "game", "park", "motion", "snack", "advice", "arch", "parade", "climb"],
        use_password=False
    )
    assert os.path.isfile(executor.wallet.hotkeyfile) 
    os.remove(executor.wallet.hotkeyfile)
    assert not os.path.isfile(executor.wallet.hotkeyfile) 

def test_unstake():
    executor.regenerate_hotkey(
        mnemonic = ["cabin", "thing", "arch", "canvas", "game", "park", "motion", "snack", "advice", "arch", "parade", "climb"],
        use_password=False
    )
    assert os.path.isfile(executor.wallet.hotkeyfile) 
    os.remove(executor.wallet.hotkeyfile)
    assert not os.path.isfile(executor.wallet.hotkeyfile) 