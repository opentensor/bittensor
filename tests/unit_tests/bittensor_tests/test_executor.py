import bittensor
import os,sys
from unittest.mock import MagicMock
import unittest.mock as mock

from bittensor.utils.balance import Balance

wallet =  bittensor.wallet(
    path = '/tmp/pytest',
    name = 'pytest',
    hotkey = 'pytest',
) 
executor = bittensor.executor( wallet = wallet )
executor.metagraph.sync()

def test_create_hotkey():
    executor.create_new_hotkey(
        n_words = 12,
        use_password = False,
        overwrite = True
    )
    assert executor.wallet.hotkey_file.exists_on_device()
    os.remove(executor.wallet.hotkey_file.path)
    assert not executor.wallet.hotkey_file.exists_on_device()

def test_create_coldkey():
    executor.create_new_coldkey(
        n_words = 12,
        use_password = False,
        overwrite = True
    )
    assert executor.wallet.coldkey_file.exists_on_device() 
    os.remove(executor.wallet.coldkey_file.path)
    os.remove(executor.wallet.coldkeypub_file.path)
    assert not executor.wallet.coldkey_file.exists_on_device() 
    assert not executor.wallet.coldkeypub_file.exists_on_device() 

def test_regenerate_coldkey():
    executor.regenerate_coldkey(
        mnemonic = ["cabin", "thing", "arch", "canvas", "game", "park", "motion", "snack", "advice", "arch", "parade", "climb"],
        use_password = False,
        overwrite = True
    )
    assert executor.wallet.coldkey_file.exists_on_device() 
    os.remove(executor.wallet.coldkey_file.path)
    os.remove(executor.wallet.coldkeypub_file.path)
    assert not executor.wallet.coldkey_file.exists_on_device() 
    assert not executor.wallet.coldkeypub_file.exists_on_device() 

def test_regenerate_hotkey():
    executor.regenerate_hotkey(
        mnemonic = ["cabin", "thing", "arch", "canvas", "game", "park", "motion", "snack", "advice", "arch", "parade", "climb"],
        use_password = False,
        overwrite = True
    )
    assert executor.wallet.hotkey_file.exists_on_device()
    os.remove(executor.wallet.hotkey_file.path)
    assert not executor.wallet.hotkey_file.exists_on_device()

def test_overview():
    wallet.create_new_coldkey(use_password=False, overwrite = True)
    wallet.create_new_hotkey(use_password=False, overwrite = True)
    executor.overview()

def test_overview_copy():
    class mock_cold_key():
        def __init__(self,coldkey):
            self.ss58_address = str(coldkey)
        def address(self):
            return self.ss58_address

    executor = bittensor.executor( wallet = wallet )
    wallet.create_new_coldkey(use_password=False, overwrite = True)
    wallet.create_new_hotkey(use_password=False, overwrite = True)
    executor.metagraph.sync()

    coldkey = executor.metagraph.coldkeys[0]
    bal = Balance.from_float(200)

    with mock.patch.object(executor.wallet, '_coldkeypub', new=mock_cold_key(coldkey)):
        executor.subtensor.get_balance = MagicMock(return_value = bal) 
        executor.overview()

def test_unstake_all():
    wallet.create_new_coldkey(use_password=False, overwrite = True)
    wallet.create_new_hotkey(use_password=False, overwrite = True)
    executor.subtensor.unstake = MagicMock(return_value = True) 
    executor.unstake_all()

def test_unstake_all_fail():
    wallet.create_new_coldkey(use_password=False, overwrite = True)
    wallet.create_new_hotkey(use_password=False, overwrite = True)
    executor.subtensor.unstake = MagicMock(return_value = False) 
    executor.unstake_all()


# -- cli ---

def test_create_cli_overview():
    sys.argv = [sys.argv[0], 'overview']
    cli = bittensor.cli(executor = executor)
    bal = Balance.from_float(200)
    executor.subtensor.get_balance = MagicMock(return_value = bal) 
    cli.run()

def test_create_cli_regen_coldkey():
    mnemonic = ["cabin", "thing", "arch", "canvas", "game", "park", "motion", "snack", "advice", "arch", "parade", "climb"]
    sys.argv = [sys.argv[0], 'regen_coldkey', '--mnemonic', ' '.join(mnemonic) ]
    cli = bittensor.cli(executor = executor)
    executor.regenerate_coldkey = MagicMock(return_value = True) 
    cli.run()

def test_create_cli_regen_hotkey():
    mnemonic = ["cabin", "thing", "arch", "canvas", "game", "park", "motion", "snack", "advice", "arch", "parade", "climb"]
    sys.argv = [sys.argv[0], 'regen_hotkey', '--mnemonic', ' '.join(mnemonic) ]
    cli = bittensor.cli(executor = executor)
    executor.regenerate_hotkey = MagicMock(return_value = True) 
    cli.run()

def test_create_cli_new_coldkey():
    sys.argv = [sys.argv[0], 'new_coldkey']
    cli = bittensor.cli(executor = executor)
    executor.create_new_coldkey = MagicMock(return_value = True) 
    cli.run()

def test_create_cli_new_hotkey():
    sys.argv = [sys.argv[0], 'new_hotkey']
    cli = bittensor.cli(executor = executor)
    executor.create_new_hotkey = MagicMock(return_value = True) 
    cli.run()


if __name__ == "__main__":
    test_overview_copy()