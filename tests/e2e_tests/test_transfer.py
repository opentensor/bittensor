import typing

from bittensor_wallet import Wallet
import pytest

from bittensor.utils.balance import Balance
from bittensor import logging

if typing.TYPE_CHECKING:
    from bittensor.extras import SubtensorApi


def test_transfer(subtensor, alice_wallet):
    """
    Test the transfer mechanism on the chain.

    Steps:
        1. Attempt transfer with insufficient balance (should fail)
        2. Transfer 2 Tao successfully
        3. Verify balance calculations
    """

    dest_coldkey = "5GpzQgpiAKHMWNSH3RN4GLf96GVTDct9QxYEFAY7LWcVzTbx"

    # Account balance before any transfer
    balance_before = subtensor.wallets.get_balance(alice_wallet.coldkeypub.ss58_address)

    # --- Insufficient balance case (NEW) ---
    insufficient_amount = balance_before + Balance.from_tao(1)

    response = subtensor.extrinsics.transfer(
        wallet=alice_wallet,
        destination_ss58=dest_coldkey,
        amount=insufficient_amount,
        wait_for_finalization=True,
        wait_for_inclusion=True,
    )
    assert "insufficient Funds, Top Up!" in response.message.lower()

    # --- Successful transfer (EXISTING FLOW) ---
    transfer_value = Balance.from_tao(2)

    response = subtensor.extrinsics.transfer(
        wallet=alice_wallet,
        destination_ss58=dest_coldkey,
        amount=transfer_value,
        wait_for_finalization=True,
        wait_for_inclusion=True,
    )

    assert response.success, response.message

    balance_after = subtensor.wallets.get_balance(alice_wallet.coldkeypub.ss58_address)

    assert balance_before - response.extrinsic_fee - transfer_value == balance_after, (
        f"Expected {balance_before - transfer_value - response.extrinsic_fee}, got {balance_after}"
    )


@pytest.mark.asyncio
async def test_transfer_async(async_subtensor, alice_wallet):
    """
    Async version of transfer test.

    Steps:
        1. Attempt transfer with insufficient balance (should fail)
        2. Transfer 2 Tao successfully
        3. Verify balance calculations
    """

    dest_coldkey = "5GpzQgpiAKHMWNSH3RN4GLf96GVTDct9QxYEFAY7LWcVzTbx"

    balance_before = await async_subtensor.wallets.get_balance(
        alice_wallet.coldkeypub.ss58_address
    )

    # --- Insufficient balance case (NEW) ---
    insufficient_amount = balance_before + Balance.from_tao(1)

    response = await async_subtensor.extrinsics.transfer(
        wallet=alice_wallet,
        destination_ss58=dest_coldkey,
        amount=insufficient_amount,
        wait_for_finalization=True,
        wait_for_inclusion=True,
    )

    assert not response.success
    assert "insufficient" in response.message.lower()

    # --- Successful transfer (EXISTING FLOW) ---
    transfer_value = Balance.from_tao(2)

    response = await async_subtensor.extrinsics.transfer(
        wallet=alice_wallet,
        destination_ss58=dest_coldkey,
        amount=transfer_value,
        wait_for_finalization=True,
        wait_for_inclusion=True,
    )

    assert response.success, response.message

    balance_after = await async_subtensor.wallets.get_balance(
        alice_wallet.coldkeypub.ss58_address
    )

    assert balance_before - response.extrinsic_fee - transfer_value == balance_after, (
        f"Expected {balance_before - transfer_value - response.extrinsic_fee}, got {balance_after}"
    )


def test_transfer_all(subtensor, alice_wallet):
    # create two dummy accounts we can drain
    dummy_account_1 = Wallet(path="/tmp/bittensor-dummy-account-1")
    dummy_account_2 = Wallet(path="/tmp/bittensor-dummy-account-2")
    dummy_account_1.create_new_coldkey(use_password=False, overwrite=True)
    dummy_account_2.create_new_coldkey(use_password=False, overwrite=True)

    # fund the first dummy account
    assert subtensor.extrinsics.transfer(
        wallet=alice_wallet,
        destination_ss58=dummy_account_1.coldkeypub.ss58_address,
        amount=Balance.from_tao(2.0),
        wait_for_finalization=True,
        wait_for_inclusion=True,
    ).success

    existential_deposit = subtensor.chain.get_existential_deposit()

    assert subtensor.extrinsics.transfer(
        wallet=dummy_account_1,
        destination_ss58=dummy_account_2.coldkeypub.ss58_address,
        amount=None,
        transfer_all=True,
        wait_for_finalization=True,
        wait_for_inclusion=True,
        keep_alive=True,
    ).success

    balance_after = subtensor.wallets.get_balance(
        dummy_account_1.coldkeypub.ss58_address
    )
    assert balance_after == existential_deposit

    assert subtensor.extrinsics.transfer(
        wallet=dummy_account_2,
        destination_ss58=alice_wallet.coldkeypub.ss58_address,
        amount=None,
        transfer_all=True,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        keep_alive=False,
    ).success

    balance_after = subtensor.wallets.get_balance(
        dummy_account_2.coldkeypub.ss58_address
    )
    assert balance_after == Balance(0)


@pytest.mark.asyncio
async def test_transfer_all_async(async_subtensor, alice_wallet):
    dummy_account_1 = Wallet(path="/tmp/bittensor-dummy-account-3")
    dummy_account_2 = Wallet(path="/tmp/bittensor-dummy-account-4")
    dummy_account_1.create_new_coldkey(use_password=False, overwrite=True)
    dummy_account_2.create_new_coldkey(use_password=False, overwrite=True)

    assert (
        await async_subtensor.extrinsics.transfer(
            wallet=alice_wallet,
            destination_ss58=dummy_account_1.coldkeypub.ss58_address,
            amount=Balance.from_tao(2.0),
            wait_for_finalization=True,
            wait_for_inclusion=True,
        )
    ).success

    existential_deposit = await async_subtensor.chain.get_existential_deposit()

    assert (
        await async_subtensor.extrinsics.transfer(
            wallet=dummy_account_1,
            destination_ss58=dummy_account_2.coldkeypub.ss58_address,
            amount=None,
            transfer_all=True,
            wait_for_finalization=True,
            wait_for_inclusion=True,
            keep_alive=True,
        )
    ).success

    balance_after = await async_subtensor.wallets.get_balance(
        dummy_account_1.coldkeypub.ss58_address
    )
    assert balance_after == existential_deposit

    assert (
        await async_subtensor.extrinsics.transfer(
            wallet=dummy_account_2,
            destination_ss58=alice_wallet.coldkeypub.ss58_address,
            amount=None,
            transfer_all=True,
            wait_for_inclusion=True,
            wait_for_finalization=True,
            keep_alive=False,
        )
    ).success

    balance_after = await async_subtensor.wallets.get_balance(
        dummy_account_2.coldkeypub.ss58_address
    )
    assert balance_after == Balance(0)
