from bittensor import Subtensor, logging
from bittensor.core.subtensor import transfer_extrinsic
from tests.e2e_tests.utils.e2e_test_utils import setup_wallet


def test_transfer(local_chain):
    """
    Test the transfer mechanism on the chain

    Steps:
        1. Create a wallet for Alice
        2. Calculate existing balance and transfer 2 Tao
        3. Calculate balance after extrinsic call and verify calculations
    Raises:
        AssertionError: If any of the checks or verifications fail
    """

    logging.info("Testing test_transfer")

    # Set up Alice wallet
    keypair, wallet = setup_wallet("//Alice")

    # Account details before transfer
    acc_before = local_chain.query("System", "Account", [keypair.ss58_address])

    # Transfer Tao using extrinsic
    subtensor = Subtensor(network="ws://localhost:9945")
    transfer_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        dest="5GpzQgpiAKHMWNSH3RN4GLf96GVTDct9QxYEFAY7LWcVzTbx",
        amount=2,
        wait_for_finalization=True,
        wait_for_inclusion=True,
        prompt=False,
    )

    # Account details after transfer
    acc_after = local_chain.query("System", "Account", [keypair.ss58_address])

    # Transfer calculation assertions
    expected_transfer = 2_000_000_000
    tolerance = 200_000  # Tx fee tolerance

    actual_difference = (
        acc_before.value["data"]["free"] - acc_after.value["data"]["free"]
    )
    assert (
        expected_transfer <= actual_difference <= expected_transfer + tolerance
    ), f"Expected transfer with tolerance: {expected_transfer} <= {actual_difference} <= {expected_transfer + tolerance}"

    logging.info("✅ Passed test_transfer")
