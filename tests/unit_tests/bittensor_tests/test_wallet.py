import bittensor

the_wallet = None

def test_create_wallet():
    global the_wallet
    the_wallet = bittensor.wallet.Wallet() 


def test_wallet_config():
    config = bittensor.wallet.Wallet.build_config()
    config.wallet.hotkeyfile
    config.wallet.coldkeyfile

def test_wallet_keypair():  
    the_wallet.keypair
    the_wallet.coldkey

test_create_wallet()
test_wallet_keypair()