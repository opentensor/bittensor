from bittensor.core.subtensor import Subtensor
from bittensor.utils.balance import Balance
from tests.e2e_tests.utils.e2e_test_utils import setup_wallet


def test_transfer(local_chain):
    """
    Test the transfer mechanism on the chain

    Steps:
        1. Create a wallet for Alice
        2. Calculate existing balance and transfer 2 Tao
        3. Calculate balance after transfer call and verify calculations
    Raises:
        AssertionError: If any of the checks or verifications fail
    """

    print("Testing test_transfer")

    # Set up Alice wallet
    keypair, wallet = setup_wallet("//Alice")
    subtensor = Subtensor(network="ws://localhost:9945")
    transfer_value = Balance.from_tao(2)
    dest_coldkey = "5GpzQgpiAKHMWNSH3RN4GLf96GVTDct9QxYEFAY7LWcVzTbx"

    # Fetch transfer fee
    transfer_fee = subtensor.get_transfer_fee(
        wallet=wallet,
        dest=dest_coldkey,
        value=transfer_value,
    )

    # Account details before transfer
    balance_before = subtensor.get_balance(wallet.coldkeypub.ss58_address)

    # Transfer Tao
    assert subtensor.transfer(
        wallet=wallet,
        dest=dest_coldkey,
        amount=transfer_value,
        wait_for_finalization=True,
        wait_for_inclusion=True,
    )
    # Account details after transfer
    balance_after = subtensor.get_balance(wallet.coldkeypub.ss58_address)

    # Assert correct transfer calculations
    assert (
        balance_before - transfer_fee - transfer_value == balance_after
    ), f"Expected {balance_before - transfer_value - transfer_fee}, got {balance_after}"

    print("✅ Passed test_transfer")
