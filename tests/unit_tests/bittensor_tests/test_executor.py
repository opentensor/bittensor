import bittensor
import os,sys
import torch
from unittest.mock import MagicMock
import unittest.mock as mock
import pytest

from bittensor.utils.balance import Balance

wallet =  bittensor.wallet(
    path = '/tmp/pytest',
    name = 'pytest',
    hotkey = 'pytest',
)

wallet = wallet.create(coldkey_use_password=False, hotkey_use_password=False)
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

    executor.subtensor.unstake = MagicMock(return_value = False) 
    executor.unstake_all()
    
    # through cli
    sys.argv = [sys.argv[0], 'unstake', '--all']
    cli = bittensor.cli(executor = executor)
    cli.run()

def test_unstake():
    wallet.create_new_coldkey(use_password=False, overwrite = True)
    wallet.create_new_hotkey(use_password=False, overwrite = True)
    executor.metagraph.sync()
    test_endpoint_obj = bittensor.endpoint.from_dict(
        {
            'version' : executor.metagraph._endpoint_objs[0].version,
            'uid' : executor.metagraph._endpoint_objs[-1].uid,
            'hotkey' : wallet.hotkey.ss58_address,
            'port' : executor.metagraph._endpoint_objs[0].port,
            'ip' : executor.metagraph._endpoint_objs[0].ip,
            'ip_type' : executor.metagraph._endpoint_objs[0].ip_type,
            'modality' : executor.metagraph._endpoint_objs[0].modality,
            'coldkey' : wallet.coldkeypub.ss58_address,
        }
    )
    executor.metagraph._endpoint_objs[-1] =  test_endpoint_obj
    executor.metagraph.endpoints[-1] = torch.tensor(test_endpoint_obj.to_tensor().tolist())
    executor.metagraph.load = MagicMock(return_value = True)
    executor.metagraph.sync = MagicMock(return_value = True)
    executor.metagraph.save = MagicMock(return_value = True)
    uid = executor.metagraph._endpoint_objs[-1].uid
    executor.metagraph.S[ uid ] = 20
    
    # successful unstake
    executor.subtensor.unstake = MagicMock(return_value = True)
    executor.unstake( amount_tao = 10, uid = uid)

    # staking with not enough tao
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        executor.unstake( amount_tao = 30, uid = uid)
    
    assert pytest_wrapped_e.type == SystemExit

    # staking with wrong uid
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        executor.unstake( amount_tao = 10, uid = uid-1)
    
    assert pytest_wrapped_e.type == SystemExit

    # failing at subtensor add_stake
    executor.subtensor.add_stake = MagicMock(return_value = False)
    executor.unstake( amount_tao = 10, uid = uid)
    
    # through cli
    sys.argv = [sys.argv[0], 'unstake', '--uid', str(uid), '--amount', '10']
    cli = bittensor.cli(executor = executor)
    cli.run()

def test_stake():
    wallet.create_new_coldkey(use_password=False, overwrite = True)
    wallet.create_new_hotkey(use_password=False, overwrite = True)
    executor.metagraph.sync()
    test_endpoint_obj = bittensor.endpoint.from_dict(
        {
            'version' : executor.metagraph._endpoint_objs[0].version,
            'uid' : executor.metagraph._endpoint_objs[-1].uid,
            'hotkey' : wallet.hotkey.ss58_address,
            'port' : executor.metagraph._endpoint_objs[0].port,
            'ip' : executor.metagraph._endpoint_objs[0].ip,
            'ip_type' : executor.metagraph._endpoint_objs[0].ip_type,
            'modality' : executor.metagraph._endpoint_objs[0].modality,
            'coldkey' : wallet.coldkeypub.ss58_address,
        }
    )
    executor.metagraph._endpoint_objs[-1] =  test_endpoint_obj
    executor.metagraph.endpoints[-1] = torch.tensor(test_endpoint_obj.to_tensor().tolist())
    executor.metagraph.load = MagicMock(return_value = True)
    executor.metagraph.sync = MagicMock(return_value = True)
    executor.metagraph.save = MagicMock(return_value = True)
    executor.subtensor.get_balance = MagicMock(return_value = Balance.from_float(20)) 
    uid = executor.metagraph._endpoint_objs[-1].uid
    
    # successful staking
    executor.subtensor.add_stake = MagicMock(return_value = True)
    executor.stake( amount_tao = 10, uid = uid)

    # staking with not enough tao
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        executor.stake( amount_tao = 30, uid = uid)
    
    assert pytest_wrapped_e.type == SystemExit

    # staking with wrong uid
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        executor.stake( amount_tao = 20, uid = uid-1)
    
    assert pytest_wrapped_e.type == SystemExit

    # failing at subtensor add_stake
    executor.subtensor.add_stake = MagicMock(return_value = False)
    executor.stake( amount_tao = 10, uid = uid)
    
    # through cli
    sys.argv = [sys.argv[0], 'stake', '--uid', str(uid), '--amount', '10']
    cli = bittensor.cli(executor = executor)
    cli.run()


def test_transfer():
    wallet.create_new_coldkey(use_password=False, overwrite = True)
    wallet.create_new_hotkey(use_password=False, overwrite = True)
    executor.subtensor.transfer = MagicMock(return_value = True)
    executor.subtensor.get_balance = MagicMock(return_value = Balance.from_float(20)) 
    executor.transfer( amount_tao = 10, destination = "")

    executor.subtensor.transfer = MagicMock(return_value = False)
    executor.transfer( amount_tao = 10, destination = "")

    # through cli
    sys.argv = [sys.argv[0], 'transfer', '--dest', '', '--amount', '10']
    cli = bittensor.cli(executor = executor)
    cli.run()

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