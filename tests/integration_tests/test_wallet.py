# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.

import bittensor
import unittest
from unittest.mock import MagicMock
import os
import time
import shutil
from bittensor.utils.balance import Balance


def init_wallet():
    wallet_path = f"/tmp/pytest/{time.time()}"
    if os.path.exists(wallet_path):
        shutil.rmtree(wallet_path)
    
    the_wallet = bittensor.wallet (
        path = wallet_path,
        name = 'pytest',
        hotkey = 'pytest',
    )
    
    return the_wallet

def check_keys_exists(the_wallet = None, coldkey_exists = True, hotkey_exists = True, coldkeypub_exists = True):
    if coldkey_exists:
        # --- test file and key exists
        assert os.path.isfile(the_wallet.coldkey_file.path)
        assert the_wallet._coldkey != None
        # --- test _load_key()
        the_wallet._coldkey = None
        the_wallet.coldkey
        assert the_wallet._coldkey != None

    if hotkey_exists:
        # --- test file and key exists
        assert os.path.isfile(the_wallet.hotkey_file.path)
        assert the_wallet._hotkey != None
        # --- test _load_key()
        the_wallet._hotkey = None
        the_wallet.hotkey
        assert the_wallet._hotkey != None

    if coldkeypub_exists:
        # --- test file and key exists
        assert os.path.isfile(the_wallet.coldkeypub_file.path)
        # --- test _load_key()
        the_wallet._coldkeypub = None
        the_wallet.coldkeypub
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

def test_wallet_seed_create():
    the_wallet = init_wallet()
    the_wallet.regenerate_coldkey( seed = "0xf34aef01dbe821cbd8d4bbe71e01885c99fe0684886f902f8a88e0434e113041",  use_password=False, overwrite = True )
    the_wallet.regenerate_hotkey( mnemonic = "solve arrive guilt syrup dust sea used phone flock vital narrow endorse".split(),  use_password=False, overwrite = True )
    check_keys_exists(the_wallet)
    assert the_wallet.coldkeypub.ss58_address == "5DtstnxttQ1DRQMr5oWdh9B3vnqgHANN76S7s5G4ARjmgbph"

    the_wallet = init_wallet()
    the_wallet.regen_coldkey( seed = "0xee30507e209fb7630d6e7ff81b8fa74def94ba88705b288138d16159e00ce2a8",  use_password=False, overwrite = True )
    the_wallet.regen_hotkey( mnemonic = "solve arrive guilt syrup dust sea used phone flock vital narrow endorse".split(),  use_password=False, overwrite = True )
    check_keys_exists(the_wallet)
    assert the_wallet.coldkeypub.ss58_address == "5CtstubuSoVLJGCXkiWRNKrrGg2DVBZ9qMs2qYTLsZR4q1Wg"

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

def test_wallet_coldkeypub_create():
    the_wallet = init_wallet()
    public_key_hex_str = "0x32939b6abc4d81f02dff04d2b8d1d01cc8e71c5e4c7492e4fa6a238cdca3512f"
    the_wallet.regenerate_coldkeypub( public_key = public_key_hex_str, overwrite = True )
    check_keys_exists(the_wallet, coldkey_exists=False, hotkey_exists=False) # Don't check the coldkey or hotkey
    assert the_wallet.coldkeypub.ss58_address == "5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"

    the_wallet = init_wallet()
    ss58_address = "5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
    the_wallet.regenerate_coldkeypub( ss58_address = ss58_address, overwrite = True )
    check_keys_exists(the_wallet, coldkey_exists=False, hotkey_exists=False) # Don't check the coldkey or hotkey
    assert the_wallet.coldkeypub.ss58_address == "5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"

def test_wallet_add_stake():
    subtensor = bittensor.subtensor( _mock=True )
    subtensor.add_stake = MagicMock(return_value = True)

    the_wallet = init_wallet().create(coldkey_use_password = False, hotkey_use_password = False)
    subtensor.add_stake = MagicMock(return_value = True)
    the_wallet.is_registered = MagicMock(return_value = True)
    the_wallet.add_stake(subtensor = subtensor)

    # when not registered
    the_wallet.is_registered = MagicMock(return_value = False)
    the_wallet.add_stake(subtensor = subtensor)

def test_wallet_remove_stake():
    subtensor = bittensor.subtensor( _mock=True )
    subtensor.unstake = MagicMock(return_value = True)

    the_wallet = init_wallet().create(coldkey_use_password = False, hotkey_use_password = False)
    subtensor.unstake = MagicMock(return_value = True)
    the_wallet.is_registered = MagicMock(return_value = True)
    the_wallet.remove_stake(subtensor = subtensor)
    
    #when not registered
    the_wallet.is_registered = MagicMock(return_value = False)
    the_wallet.remove_stake(subtensor = subtensor)

def test_wallet_transfer():
    subtensor = bittensor.subtensor( _mock=True )
    subtensor.transfer = MagicMock(return_value = True)
    subtensor.unstake = MagicMock(return_value = True)

    the_wallet = init_wallet().create(coldkey_use_password = False, hotkey_use_password = False)

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

def test_wallet_mock():
    wallet = bittensor.wallet(_mock=True)
    assert wallet.hotkey_file.exists_on_device()
    assert wallet.coldkey_file.exists_on_device()
    assert wallet.coldkeypub_file.exists_on_device()
    assert wallet.hotkey
    assert wallet.coldkey
    assert wallet.coldkeypub

def test_wallet_mock_from_config():
    config = bittensor.wallet.config()
    config.wallet.name = 'mock'
    wallet = bittensor.wallet(config = config)
    assert wallet.hotkey_file.exists_on_device()
    assert wallet.coldkey_file.exists_on_device()
    assert wallet.coldkeypub_file.exists_on_device()
    assert wallet.hotkey
    assert wallet.coldkey
    assert wallet.coldkeypub

def test_wallet_mock_from_name():
    wallet = bittensor.wallet(name = 'mock')
    assert wallet.hotkey_file.exists_on_device()
    assert wallet.coldkey_file.exists_on_device()
    assert wallet.coldkeypub_file.exists_on_device()
    assert wallet.hotkey
    assert wallet.coldkey
    assert wallet.coldkeypub

def test_wallet_mock_from_func():
    wallet = bittensor.wallet.mock()
    assert wallet.hotkey_file.exists_on_device()
    assert wallet.coldkey_file.exists_on_device()
    assert wallet.coldkeypub_file.exists_on_device()
    assert wallet.hotkey
    assert wallet.coldkey
    assert wallet.coldkeypub


if __name__ == "__main__":
    unittest.main()