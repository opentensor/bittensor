import bittensor
from unittest.mock import MagicMock
import os
import shutil

from bittensor.utils.balance import Balance
subtensor = bittensor.subtensor()

def init_wallet():
    if os.path.exists('/tmp/pytest'):
        shutil.rmtree('/tmp/pytest')
    
    the_wallet = bittensor.wallet (
        path = '/tmp/pytest',
        name = 'pytest',
        hotkey = 'pytest',
    )
    
    return the_wallet

def check_keys_exists(the_wallet = None):

    # --- test file and key exists
    assert os.path.isfile(the_wallet.coldkey_file.path)
    assert os.path.isfile(the_wallet.hotkey_file.path)
    assert os.path.isfile(the_wallet.coldkeypub_file.path)
    
    assert the_wallet._hotkey != None
    assert the_wallet._coldkey != None
    
    # --- test _load_key()
    the_wallet._hotkey = None
    the_wallet._coldkey = None
    the_wallet._coldkeypub = None
    
    the_wallet.hotkey
    the_wallet.coldkey
    the_wallet.coldkeypub
    
    assert the_wallet._hotkey != None
    assert the_wallet._coldkey != None
    assert the_wallet._coldkeypub != None

def test_create_wallet():
    the_wallet = init_wallet().create(coldkey_use_password = False, hotkey_use_password = False)
    check_keys_exists(the_wallet)

def test_create_keys():
    the_wallet = init_wallet()
    the_wallet.create_new_coldkey( use_password=False, overwrite = True )
    the_wallet.create_new_hotkey( use_password=False, overwrite = True )
    check_keys_exists(the_wallet)
    
    the_wallet = init_wallet()
    the_wallet.new_coldkey( use_password=False, overwrite = True )
    the_wallet.new_hotkey( use_password=False, overwrite = True )
    check_keys_exists(the_wallet)
    
def test_wallet_uri():
    the_wallet = init_wallet()
    the_wallet.create_coldkey_from_uri( uri = "/Alice", use_password=False, overwrite = True )
    the_wallet.create_hotkey_from_uri( uri = "/Alice", use_password=False, overwrite = True )
    check_keys_exists(the_wallet)

def test_wallet_mnemonic_create():
    the_wallet = init_wallet()
    the_wallet.regenerate_coldkey( mnemonic = "solve arrive guilt syrup dust sea used phone flock vital narrow endorse",  use_password=False, overwrite = True )
    the_wallet.regenerate_coldkey( mnemonic = "solve arrive guilt syrup dust sea used phone flock vital narrow endorse".split(),  use_password=False, overwrite = True )
    the_wallet.regenerate_hotkey( mnemonic = "solve arrive guilt syrup dust sea used phone flock vital narrow endorse", use_password=False, overwrite = True )
    the_wallet.regenerate_hotkey( mnemonic = "solve arrive guilt syrup dust sea used phone flock vital narrow endorse".split(),  use_password=False, overwrite = True )
    check_keys_exists(the_wallet)

    the_wallet = init_wallet()
    the_wallet.regen_coldkey( mnemonic = "solve arrive guilt syrup dust sea used phone flock vital narrow endorse",  use_password=False, overwrite = True )
    the_wallet.regen_coldkey( mnemonic = "solve arrive guilt syrup dust sea used phone flock vital narrow endorse".split(),  use_password=False, overwrite = True )
    the_wallet.regen_hotkey( mnemonic = "solve arrive guilt syrup dust sea used phone flock vital narrow endorse", use_password=False, overwrite = True )
    the_wallet.regen_hotkey( mnemonic = "solve arrive guilt syrup dust sea used phone flock vital narrow endorse".split(),  use_password=False, overwrite = True )
    check_keys_exists(the_wallet)

def test_wallet_is_registered():
    the_wallet = init_wallet().create(coldkey_use_password = False, hotkey_use_password = False)
    the_wallet.is_registered = MagicMock(return_value = True)
    the_wallet.register( email = 'fake@email.com')
    check_keys_exists(the_wallet)

def test_wallet_prop():
    the_wallet = init_wallet().create(coldkey_use_password = False, hotkey_use_password = False)
    the_wallet.is_registered = MagicMock(return_value = True)
    str(the_wallet)
    repr(the_wallet)
    assert the_wallet.neuron != None
    assert the_wallet.trust != None
    assert the_wallet.rank != None
    assert the_wallet.incentive != None
    assert the_wallet.dividends != None
    assert the_wallet.consensus != None
    assert the_wallet.inflation != None
    assert the_wallet.ip != None
    assert the_wallet.last_update != None
    assert the_wallet.weights != None
    assert the_wallet.bonds != None
    assert the_wallet.uid != None
    assert the_wallet.stake is not None
    assert the_wallet.balance is not None
    
    the_wallet.is_registered = MagicMock(return_value = False)
    assert the_wallet.neuron == None
    assert the_wallet.uid == -1
    assert the_wallet.stake == Balance(0)

def test_wallet_register_wo_email():
    the_wallet = init_wallet().create(coldkey_use_password = False, hotkey_use_password = False)
    the_wallet.register()

def test_wallet_register():
    the_wallet = init_wallet().create(coldkey_use_password = False, hotkey_use_password = False)
    the_wallet._email = 'pytest@gmail.com'
    the_wallet.is_registered = MagicMock(return_value = False)
    the_wallet.register()

def test_wallet_add_stake():
    subtensor = bittensor.subtensor()
    the_wallet = init_wallet().create(coldkey_use_password = False, hotkey_use_password = False)
    subtensor.add_stake = MagicMock(return_value = True)
    the_wallet.is_registered = MagicMock(return_value = True)
    the_wallet.add_stake(subtensor = subtensor)

    # when not registered
    the_wallet.is_registered = MagicMock(return_value = False)
    the_wallet.add_stake(subtensor = subtensor)

def test_wallet_remove_stake():
    subtensor = bittensor.subtensor()
    the_wallet = init_wallet().create(coldkey_use_password = False, hotkey_use_password = False)
    subtensor.unstake = MagicMock(return_value = True)
    the_wallet.is_registered = MagicMock(return_value = True)
    the_wallet.remove_stake(subtensor = subtensor)
    
    #when not registered
    the_wallet.is_registered = MagicMock(return_value = False)
    the_wallet.remove_stake(subtensor = subtensor)

def test_wallet_transfer():
    subtensor = bittensor.subtensor()
    
    the_wallet = init_wallet().create(coldkey_use_password = False, hotkey_use_password = False)
    subtensor.transfer = MagicMock(return_value = True)
    
    # when registered
    the_wallet.is_registered = MagicMock(return_value = True)
    the_wallet.get_balance = MagicMock(return_value = Balance(20))
    the_wallet.transfer(amount = 10, subtensor = subtensor, dest = "")
    
    # when not enough tao
    the_wallet.get_balance = MagicMock(return_value = Balance(5))
    the_wallet.transfer(amount = 10, subtensor = subtensor, dest = "")
    
    # when not registered
    the_wallet.is_registered = MagicMock(return_value = False)
    the_wallet.remove_stake(subtensor = subtensor)
