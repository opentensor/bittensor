import bittensor
from unittest.mock import MagicMock
import os

the_wallet = bittensor.wallet(
    path = '/tmp/pytest',
    name = 'pytest',
    hotkey = 'pytest',
) 

def test_create_wallet():
    the_wallet.create_new_coldkey( use_password=False, overwrite = True )
    the_wallet.create_new_hotkey( use_password=False, overwrite = True )
    assert os.path.isfile(the_wallet.coldkeyfile)
    assert os.path.isfile(the_wallet.hotkeyfile)
    assert os.path.isfile(the_wallet.coldkeypubfile)

def test_wallet_keypair():  
    the_wallet.hotkey
    the_wallet.coldkeypub

def test_wallet_uid():
    uid = the_wallet.get_uid()
    assert uid == -1 

    s = bittensor.subtensor()
    s.get_uid_for_pubkey = MagicMock( return_value = 10 )
    
    uid = the_wallet.get_uid( subtensor = s)
    assert uid == 10 

def test_wallet_stake():
    stake = the_wallet.get_stake( )
    assert stake.rao == 0  # the stake balance is zero, it is not subscribed

    s = bittensor.subtensor()
    s.get_stake_for_uid = MagicMock( return_value = bittensor.Balance(10) )
    stake = the_wallet.get_stake( subtensor = s )
    assert stake.rao == 10  # the stake balance is zero, it is not subscribed

def test_wallet_balance():
    balance = the_wallet.get_balance()
    assert balance.rao == 0

    s = bittensor.subtensor()
    s.get_balance = MagicMock( return_value = bittensor.Balance(10) )
    balance = the_wallet.get_balance( subtensor = s )
    assert balance.rao == 10

test_create_wallet()
test_wallet_keypair()