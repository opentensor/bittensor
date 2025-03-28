import pytest
from bittensor.utils.balance import Balance
from bittensor import logging

logging.set_trace()


@pytest.mark.asyncio
async def test_transfer(subtensor, alice_wallet):
    """
    Test the transfer mechanism on the chain

    Steps:
        1. Calculate existing balance and transfer 2 Tao
        2. Calculate balance after transfer call and verify calculations
    Raises:
        AssertionError: If any of the checks or verifications fail
    """

    print("Testing test_transfer")

    transfer_value = Balance.from_tao(2)
    dest_coldkey = "5GpzQgpiAKHMWNSH3RN4GLf96GVTDct9QxYEFAY7LWcVzTbx"

    # Fetch transfer fee
    transfer_fee = await subtensor.get_transfer_fee(
        wallet=alice_wallet,
        dest=dest_coldkey,
        value=transfer_value,
    )

    # Account details before transfer
    balance_before = await subtensor.get_balance(alice_wallet.coldkeypub.ss58_address)

    # Transfer Tao
    assert await subtensor.transfer(
        wallet=alice_wallet,
        dest=dest_coldkey,
        amount=transfer_value,
        wait_for_finalization=True,
        wait_for_inclusion=True,
    )
    # Account details after transfer
    balance_after = await subtensor.get_balance(alice_wallet.coldkeypub.ss58_address)

    # Assert correct transfer calculations
    assert (
        balance_before - transfer_fee - transfer_value == balance_after
    ), f"Expected {balance_before - transfer_value - transfer_fee}, got {balance_after}"

    print("✅ Passed test_transfer")
