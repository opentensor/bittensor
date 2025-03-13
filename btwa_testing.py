from bittensor_wallet import Keypair, Keyfile, Wallet, Config



wallet = Wallet(name="tmp")
wallet.coldkeyfile.save_to_env

print(wallet.coldkey)
