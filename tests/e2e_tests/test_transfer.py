import typing

from bittensor_wallet import Wallet
import pytest

from bittensor.utils.balance import Balance
from bittensor import logging

if typing.TYPE_CHECKING:
    from bittensor.core.subtensor_api import SubtensorApi


def test_transfer(subtensor, alice_wallet):
    """
    Test the transfer mechanism on the chain

    Steps:
        1. Calculate existing balance and transfer 2 Tao
        2. Calculate balance after transfer call and verify calculations
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    transfer_value = Balance.from_tao(2)
    dest_coldkey = "5GpzQgpiAKHMWNSH3RN4GLf96GVTDct9QxYEFAY7LWcVzTbx"

    # Fetch transfer fee
    transfer_fee = subtensor.wallets.get_transfer_fee(
        wallet=alice_wallet,
        dest=dest_coldkey,
        value=transfer_value,
    )

    # Account details before transfer
    balance_before = subtensor.wallets.get_balance(alice_wallet.coldkeypub.ss58_address)

    # Transfer Tao
    assert subtensor.extrinsics.transfer(
        wallet=alice_wallet,
        destination=dest_coldkey,
        amount=transfer_value,
        wait_for_finalization=True,
        wait_for_inclusion=True,
    ).success
    # Account details after transfer
    balance_after = subtensor.wallets.get_balance(alice_wallet.coldkeypub.ss58_address)

    # Assert correct transfer calculations
    assert balance_before - transfer_fee - transfer_value == balance_after, (
        f"Expected {balance_before - transfer_value - transfer_fee}, got {balance_after}"
    )


@pytest.mark.asyncio
async def test_transfer_async(async_subtensor, alice_wallet):
    """
    Test the transfer mechanism on the chain

    Steps:
        1. Calculate existing balance and transfer 2 Tao
        2. Calculate balance after transfer call and verify calculations
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    transfer_value = Balance.from_tao(2)
    dest_coldkey = "5GpzQgpiAKHMWNSH3RN4GLf96GVTDct9QxYEFAY7LWcVzTbx"

    # Fetch transfer fee
    transfer_fee = await async_subtensor.wallets.get_transfer_fee(
        wallet=alice_wallet,
        dest=dest_coldkey,
        value=transfer_value,
    )

    # Account details before transfer
    balance_before = await async_subtensor.wallets.get_balance(
        alice_wallet.coldkeypub.ss58_address
    )

    # Transfer Tao
    assert (
        await async_subtensor.extrinsics.transfer(
            wallet=alice_wallet,
            destination=dest_coldkey,
            amount=transfer_value,
            wait_for_finalization=True,
            wait_for_inclusion=True,
        )
    ).success
    # Account details after transfer
    balance_after = await async_subtensor.wallets.get_balance(
        alice_wallet.coldkeypub.ss58_address
    )

    # Assert correct transfer calculations
    assert balance_before - transfer_fee - transfer_value == balance_after, (
        f"Expected {balance_before - transfer_value - transfer_fee}, got {balance_after}"
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
        destination=dummy_account_1.coldkeypub.ss58_address,
        amount=Balance.from_tao(2.0),
        wait_for_finalization=True,
        wait_for_inclusion=True,
    ).success
    # Account details before transfer
    existential_deposit = subtensor.chain.get_existential_deposit()
    assert subtensor.extrinsics.transfer(
        wallet=dummy_account_1,
        destination=dummy_account_2.coldkeypub.ss58_address,
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
        destination=alice_wallet.coldkeypub.ss58_address,
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
    # create two dummy accounts we can drain
    dummy_account_1 = Wallet(path="/tmp/bittensor-dummy-account-3")
    dummy_account_2 = Wallet(path="/tmp/bittensor-dummy-account-4")
    dummy_account_1.create_new_coldkey(use_password=False, overwrite=True)
    dummy_account_2.create_new_coldkey(use_password=False, overwrite=True)

    # fund the first dummy account
    assert (
        await async_subtensor.extrinsics.transfer(
            wallet=alice_wallet,
            destination=dummy_account_1.coldkeypub.ss58_address,
            amount=Balance.from_tao(2.0),
            wait_for_finalization=True,
            wait_for_inclusion=True,
        )
    ).success
    # Account details before transfer
    existential_deposit = await async_subtensor.chain.get_existential_deposit()
    assert (
        await async_subtensor.extrinsics.transfer(
            wallet=dummy_account_1,
            destination=dummy_account_2.coldkeypub.ss58_address,
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
            destination=alice_wallet.coldkeypub.ss58_address,
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
