import bittensor
from munch import Munch
import pytest
import os

def test_regen_hotkey( ):
    config = Munch()
    config.command = "regen_hotkey"
    config.debug = False
    config.mnemonic = ["cabin", "thing", "arch", "canvas", "game", "park", "motion", "snack", "advice", "arch", "parade", "climb"]
    config.subtensor = Munch()
    config.subtensor.network = "akira"
    config.wallet = Munch()
    config.wallet.hotkey = "pytest_hotkey"
    config.wallet.name = "test_wallet"
    config.wallet.path = "~/tmp/pytest_wallets/"
    executor = bittensor.Executor( config )
    try:
        os.remove(os.path.expanduser("~/tmp/pytest_wallets/test_wallet/hotkeys/pytest_hotkey"))
    except:
        pass    
    executor.run_command()
    os.remove(os.path.expanduser("~/tmp/pytest_wallets/test_wallet/hotkeys/pytest_hotkey"))

def test_create_hotkey( ):
    config = Munch()
    config.command = "new_hotkey"
    config.debug = False
    config.n_words = 12
    config.subtensor = Munch()
    config.subtensor.network = "akira"
    config.wallet = Munch()
    config.wallet.hotkey = "pytest_hotkey"
    config.wallet.name = "test_wallet"
    config.wallet.path = "~/tmp/pytest_wallets/"
    executor = bittensor.Executor( config )
    try:
        os.remove(os.path.expanduser("~/tmp/pytest_wallets/test_wallet/hotkeys/pytest_hotkey"))
    except:
        pass
    executor.run_command()
    os.remove(os.path.expanduser("~/tmp/pytest_wallets/test_wallet/hotkeys/pytest_hotkey"))

def test_regen_coldkey( ):
    config = Munch()
    config.command = "regen_coldkey"
    config.debug = False
    config.use_password = False
    config.mnemonic = ["cabin", "thing", "arch", "canvas", "game", "park", "motion", "snack", "advice", "arch", "parade", "climb"]
    config.subtensor = Munch()
    config.subtensor.network = "akira"
    config.wallet = Munch()
    config.wallet.name = "test_wallet"
    config.wallet.path = "~/tmp/pytest_wallets/"
    executor = bittensor.Executor( config )
    try:
        os.remove(os.path.expanduser("~/tmp/pytest_wallets/test_wallet/coldkey"))
        os.remove(os.path.expanduser("~/tmp/pytest_wallets/test_wallet/coldkeypub.txt"))
    except:
        pass    
    executor.run_command()
    os.remove(os.path.expanduser("~/tmp/pytest_wallets/test_wallet/coldkey"))
    os.remove(os.path.expanduser("~/tmp/pytest_wallets/test_wallet/coldkeypub.txt"))

def test_create_coldkey( ):
    config = Munch()
    config.command = "new_coldkey"
    config.debug = False
    config.n_words = 12
    config.use_password = False
    config.subtensor = Munch()
    config.subtensor.network = "akira"
    config.wallet = Munch()
    config.wallet.hotkey = "pytest_hotkey"
    config.wallet.name = "test_wallet"
    config.wallet.path = "~/tmp/pytest_wallets/"
    executor = bittensor.Executor( config )
    try:
        os.remove(os.path.expanduser("~/tmp/pytest_wallets/test_wallet/coldkey"))
        os.remove(os.path.expanduser("~/tmp/pytest_wallets/test_wallet/coldkeypub.txt"))
    except:
        pass    
    executor.run_command()
    os.remove(os.path.expanduser("~/tmp/pytest_wallets/test_wallet/coldkey"))
    os.remove(os.path.expanduser("~/tmp/pytest_wallets/test_wallet/coldkeypub.txt"))
