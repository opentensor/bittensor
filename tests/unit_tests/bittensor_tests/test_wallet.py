import bittensor
import bittensor.wallet as wallet

the_wallet = None

def test_create_wallet():
    global the_wallet
    the_wallet = wallet.Wallet() 

def test_wallet_keypair():  
    the_wallet.keypair

test_create_wallet()
test_wallet_keypair()