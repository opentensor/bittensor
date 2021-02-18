import bittensor

the_wallet = None

def test_create_wallet():
    global the_wallet
    the_wallet = bittensor.wallet.Wallet() 

def test_wallet_config():
    config = bittensor.wallet.Wallet.build_config()
    config.wallet.name
    config.wallet.path
    config.wallet.hotkey

def test_wallet_keypair():  
    the_wallet.hotkey
    the_wallet.coldkeypub

test_create_wallet()
test_wallet_config()
test_wallet_keypair()