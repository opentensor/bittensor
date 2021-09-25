import bittensor
from unittest.mock import MagicMock
import os

the_wallet = bittensor.wallet (
    path = '/tmp/pytest',
    name = 'pytest',
    hotkey = 'pytest',
)

def test_create_wallet():
    the_wallet.create_new_coldkey( use_password=False, overwrite = True )
    the_wallet.create_new_hotkey( use_password=False, overwrite = True )
    the_wallet.new_coldkey( use_password=False, overwrite = True )
    the_wallet.new_hotkey( use_password=False, overwrite = True )
    assert os.path.isfile(the_wallet.coldkeyfile)
    assert os.path.isfile(the_wallet.hotkeyfile)
    assert os.path.isfile(the_wallet.coldkeypubfile)

def test_wallet_uri():
    the_wallet.create_coldkey_from_uri( uri = "/Alice", use_password=False, overwrite = True )
    the_wallet.create_hotkey_from_uri( uri = "/Alice", use_password=False, overwrite = True )
    assert os.path.isfile(the_wallet.coldkeyfile)
    assert os.path.isfile(the_wallet.hotkeyfile)
    assert os.path.isfile(the_wallet.coldkeypubfile)

def test_wallet_mnemonic_create():
    the_wallet.regenerate_coldkey( mnemonic = "solve arrive guilt syrup dust sea used phone flock vital narrow endorse",  use_password=False, overwrite = True )
    the_wallet.regenerate_coldkey( mnemonic = "solve arrive guilt syrup dust sea used phone flock vital narrow endorse".split(),  use_password=False, overwrite = True )
    the_wallet.regenerate_hotkey( mnemonic = "solve arrive guilt syrup dust sea used phone flock vital narrow endorse", use_password=False, overwrite = True )
    the_wallet.regenerate_hotkey( mnemonic = "solve arrive guilt syrup dust sea used phone flock vital narrow endorse".split(),  use_password=False, overwrite = True )

    the_wallet.regen_coldkey( mnemonic = "solve arrive guilt syrup dust sea used phone flock vital narrow endorse",  use_password=False, overwrite = True )
    the_wallet.regen_coldkey( mnemonic = "solve arrive guilt syrup dust sea used phone flock vital narrow endorse".split(),  use_password=False, overwrite = True )
    the_wallet.regen_hotkey( mnemonic = "solve arrive guilt syrup dust sea used phone flock vital narrow endorse", use_password=False, overwrite = True )
    the_wallet.regen_hotkey( mnemonic = "solve arrive guilt syrup dust sea used phone flock vital narrow endorse".split(),  use_password=False, overwrite = True )
    assert os.path.isfile(the_wallet.coldkeyfile)
    assert os.path.isfile(the_wallet.hotkeyfile)
    assert os.path.isfile(the_wallet.coldkeypubfile)

def test_wallet_keypair():  
    the_wallet.hotkey
    the_wallet.coldkeypub

test_create_wallet()
test_wallet_keypair()
test_wallet_keypair()
test_create_wallet()