"""E2E tests for MEV Shield functionality."""

import pytest

from bittensor.core.extrinsics import pallets
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging
from tests.e2e_tests.utils import (
    TestSubnet,
    ACTIVATE_SUBNET,
    REGISTER_NEURON,
    REGISTER_SUBNET,
    SUDO_SET_TEMPO,
    NETUID,
    AdminUtils,
)

TEMPO_TO_SET = 3


# @pytest.mark.parametrize("local_chain", [False], indirect=True)
def test_mev_shield_happy_path(
    subtensor, alice_wallet, bob_wallet, charlie_wallet, local_chain
):
    """Tests MEV Shield functionality with add_stake inner call.

    This test verifies the complete MEV Shield flow: encrypting a transaction, submitting it, and verifying that
    validators decrypt and execute it correctly. The test covers two scenarios:
        - using default signer (wallet.coldkey)
        - explicit signer keypair

    Steps:
        - Register a subnet through Bob and activate it
        - Register Charlie's neuron on the subnet
        - Wait until the third epoch (MEV Shield logic requires at least 3 epochs with fast blocks)
        - For each signer scenario (None/default and explicit dave_wallet.coldkey):
            - Get the current stake before the transaction
            - Create an add_stake_limit call for Charlie's hotkey
            - Submit the call encrypted through MEV Shield using mev_submit_encrypted
            - Wait for validators to decrypt and execute the transaction (3 blocks)
            - Verify that the stake has increased after execution
    """
    bob_sn = TestSubnet(subtensor)
    bob_sn.execute_steps(
        [
            REGISTER_SUBNET(bob_wallet),
            SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, TEMPO_TO_SET),
            ACTIVATE_SUBNET(bob_wallet),
            REGISTER_NEURON(charlie_wallet),
        ]
    )

    if subtensor.chain.is_fast_blocks():
        # MeV Res logic works not from before third epoch with fast blocks, so we need to wait for it
        next_epoch_start_block = subtensor.subnets.get_next_epoch_start_block(
            bob_sn.netuid
        )
        subtensor.wait_for_block(
            next_epoch_start_block + subtensor.subnets.tempo(bob_sn.netuid) * 2
        )

    stake_before = subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=charlie_wallet.hotkey.ss58_address,
        netuid=bob_sn.netuid,
    )
    logging.console.info(f"Stake before: {stake_before}")

    subnet_price = subtensor.subnets.get_subnet_price(2)
    limit_price = (subnet_price * 2).rao

    call = pallets.SubtensorModule(subtensor).add_stake_limit(
        netuid=bob_sn.netuid,
        hotkey=charlie_wallet.hotkey.ss58_address,
        amount_staked=Balance.from_tao(5).rao,
        allow_partial=True,
        limit_price=limit_price,
    )

    response = subtensor.mev_shield.mev_submit_encrypted(
        wallet=alice_wallet,
        call=call,
        raise_error=True,
    )
    assert response.success, response.message

    stake_after = subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=charlie_wallet.hotkey.ss58_address,
        netuid=bob_sn.netuid,
    )
    logging.console.info(f"Stake after: {stake_after}")
    assert stake_after > stake_before


# @pytest.mark.parametrize("local_chain", [False], indirect=True)
@pytest.mark.asyncio
async def test_mev_shield_happy_path_async(
    async_subtensor, alice_wallet, bob_wallet, charlie_wallet, local_chain
):
    """Async tests MEV Shield functionality with add_stake inner call.

    This test verifies the complete MEV Shield flow: encrypting a transaction, submitting it, and verifying that
    validators decrypt and execute it correctly. The test covers two scenarios:
        - using default signer (wallet.coldkey)
        - explicit signer keypair

    Steps:
        - Register a subnet through Bob and activate it
        - Register Charlie's neuron on the subnet
        - Wait until the third epoch (MEV Shield logic requires at least 3 epochs with fast blocks)
        - For each signer scenario (None/default and explicit dave_wallet.coldkey):
            - Get the current stake before the transaction
            - Create an add_stake_limit call for Charlie's hotkey
            - Submit the call encrypted through MEV Shield using mev_submit_encrypted
            - Wait for validators to decrypt and execute the transaction (3 blocks)
            - Verify that the stake has increased after execution
    """

    bob_sn = TestSubnet(async_subtensor)
    await bob_sn.async_execute_steps(
        [
            REGISTER_SUBNET(bob_wallet),
            SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, TEMPO_TO_SET),
            ACTIVATE_SUBNET(bob_wallet),
            REGISTER_NEURON(charlie_wallet),
        ]
    )

    if await async_subtensor.chain.is_fast_blocks():
        # MeV Res logic works not from before third epoch with fast blocks, so we need to wait for it
        next_epoch_start_block = (
            await async_subtensor.subnets.get_next_epoch_start_block(bob_sn.netuid)
        )
        await async_subtensor.wait_for_block(
            next_epoch_start_block
            + await async_subtensor.subnets.tempo(bob_sn.netuid) * 2
        )

    stake_before = await async_subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=charlie_wallet.hotkey.ss58_address,
        netuid=bob_sn.netuid,
    )
    logging.console.info(f"Stake before: {stake_before}")

    subnet_price = await async_subtensor.subnets.get_subnet_price(2)
    limit_price = (subnet_price * 2).rao

    call = await pallets.SubtensorModule(async_subtensor).add_stake_limit(
        netuid=bob_sn.netuid,
        hotkey=charlie_wallet.hotkey.ss58_address,
        amount_staked=Balance.from_tao(5).rao,
        allow_partial=True,
        limit_price=limit_price,
    )

    response = await async_subtensor.mev_shield.mev_submit_encrypted(
        wallet=alice_wallet,
        call=call,
        raise_error=True,
    )
    assert response.success, response.message

    stake_after = await async_subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=charlie_wallet.hotkey.ss58_address,
        netuid=bob_sn.netuid,
    )
    logging.console.info(f"Stake after: {stake_after}")
    assert stake_after > stake_before
