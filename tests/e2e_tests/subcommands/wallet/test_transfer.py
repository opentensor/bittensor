from ...utils import get_wallet
from bittensor import wallet
from bittensor.subtensor import subtensor
from substrateinterface import Keypair

# Example test using the local_chain fixture
def test_transfer(local_chain: subtensor):
    wallet = get_wallet("//Alice", "//Bob")
    amount = 1
    assert local_chain.transfer(wallet, wallet.hotkey.ss58_address, amount=amount, wait_for_finalization=True, wait_for_inclusion=True)
   