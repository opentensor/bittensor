import bittensor
import os,sys
from unittest.mock import MagicMock

wallet =  bittensor.wallet(
    path = '/tmp/pytest',
    name = 'pytest',
    hotkey = 'pytest',
) 
executor = bittensor.executor( wallet = wallet )

def test_create_hotkey():
    executor.create_new_hotkey(
        n_words = 12,
        use_password = False,
        overwrite = True
    )
    assert os.path.isfile(executor.wallet.hotkeyfile) 
    os.remove(executor.wallet.hotkeyfile)
    assert not os.path.isfile(executor.wallet.hotkeyfile) 

def test_create_coldkey():
    executor.create_new_coldkey(
        n_words = 12,
        use_password = False,
        overwrite = True
    )
    assert os.path.isfile(executor.wallet.coldkeyfile) 
    os.remove(executor.wallet.coldkeyfile)
    os.remove(executor.wallet.coldkeypubfile)
    assert not os.path.isfile(executor.wallet.coldkeyfile) 
    assert not os.path.isfile(executor.wallet.coldkeypubfile) 

def test_regenerate_coldkey():
    executor.regenerate_coldkey(
        mnemonic = ["cabin", "thing", "arch", "canvas", "game", "park", "motion", "snack", "advice", "arch", "parade", "climb"],
        use_password = False,
        overwrite = True
    )
    assert os.path.isfile(executor.wallet.coldkeyfile) 
    os.remove(executor.wallet.coldkeyfile)
    os.remove(executor.wallet.coldkeypubfile)
    assert not os.path.isfile(executor.wallet.coldkeyfile) 
    assert not os.path.isfile(executor.wallet.coldkeypubfile) 

def test_regenerate_hotkey():
    executor.regenerate_hotkey(
        mnemonic = ["cabin", "thing", "arch", "canvas", "game", "park", "motion", "snack", "advice", "arch", "parade", "climb"],
        use_password = False,
        overwrite = True
    )
    assert os.path.isfile(executor.wallet.hotkeyfile) 
    os.remove(executor.wallet.hotkeyfile)
    assert not os.path.isfile(executor.wallet.hotkeyfile) 

def test_overview():
    wallet.create_new_coldkey(use_password=False, overwrite = True)
    wallet.create_new_hotkey(use_password=False, overwrite = True)
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
    sys.argv = [sys.argv[0], '--overview']
    cli = bittensor.cli(executor = executor)

def test_create_cli_regen_coldkey():
    sys.argv = [sys.argv[0], '--regen_coldkey']
    cli = bittensor.cli(executor = executor)

def test_create_cli_regen_hotkey():
    sys.argv = [sys.argv[0], '--regen_hotkey']
    cli = bittensor.cli(executor = executor)

def test_create_cli_new_coldkey():
    sys.argv = [sys.argv[0], '--new_coldkey']
    cli = bittensor.cli(executor = executor)

def test_create_cli_new_hotkey():
    sys.argv = [sys.argv[0], '--new_hotkey']
    cli = bittensor.cli(executor = executor)
