from substrateinterface import Keypair
import bittensor


def setup_wallet(uri: str):
    keypair = Keypair.create_from_uri(uri)
    wallet = bittensor.wallet(path="/tmp/btcli-e2e-wallet-{}".format(uri))
    wallet.set_coldkey(keypair=keypair, encrypt=False, overwrite=True)
    wallet.set_coldkeypub(keypair=keypair, encrypt=False, overwrite=True)
    wallet.set_hotkey(keypair=keypair, encrypt=False, overwrite=True)
    return keypair
