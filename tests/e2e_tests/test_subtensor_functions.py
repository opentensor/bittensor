import asyncio
import re
import pytest

from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging
from tests.e2e_tests.utils import (
    TestSubnet,
    REGISTER_SUBNET,
    ACTIVATE_SUBNET,
    REGISTER_NEURON,
)

"""
Verifies:

* get_all_subnets_netuid()
* get_total_subnets()
* subnet_exists()
* get_netuids_for_hotkey()
* is_hotkey_registered_any()
* is_hotkey_registered_on_subnet()
* get_uid_for_hotkey_on_subnet()
* get_neuron_for_pubkey_and_subnet()
* get_balance()
* get_subnet_burn_cost()
* difficulty()
* burned_register()
* recycle()
* get_existential_deposit() 
* get_all_subnets_info()
"""


@pytest.mark.asyncio
async def test_subtensor_extrinsics(subtensor, templates, alice_wallet, bob_wallet):
    """
    Tests subtensor extrinsics

    Steps:
        1. Validate subnets in the chain before/after registering netuid = 1
        2. Register Alice's neuron
        3. Verify Alice and Bob's participation in subnets (individually and global)
        4. Verify uids of Alice and Bob gets populated correctly
        5. Start Alice as a validator and verify NeuronInfo before/after is different
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    netuid = subtensor.subnets.get_total_subnets()  # 22
    # Initial balance for Alice, defined in the genesis file of localnet
    initial_alice_balance = Balance.from_tao(1_000_000)
    # Current Existential deposit for all accounts in bittensor
    existential_deposit = Balance.from_tao(0.000_000_500)

    # Subnets 0 and 1 are bootstrapped from the start
    assert subtensor.subnets.get_all_subnets_netuid() == [0, 1]
    assert subtensor.subnets.get_total_subnets() == 2

    # Assert correct balance is fetched for Alice
    alice_balance = subtensor.wallets.get_balance(alice_wallet.coldkeypub.ss58_address)
    assert alice_balance == initial_alice_balance, (
        "Balance for Alice wallet doesn't match with pre-def value."
    )

    # Subnet burn cost is initially lower before we register a subnet
    pre_subnet_creation_cost = subtensor.subnets.get_subnet_burn_cost()

    # Register subnet
    alice_sn = TestSubnet(subtensor)
    response = alice_sn.execute_one(REGISTER_SUBNET(alice_wallet))
    netuid = alice_sn.netuid

    # Subnet burn cost is increased immediately after a subnet is registered
    post_subnet_creation_cost = subtensor.subnets.get_subnet_burn_cost()

    # Assert that the burn cost changed after registering a subnet
    assert Balance.from_tao(pre_subnet_creation_cost) < Balance.from_tao(
        post_subnet_creation_cost
    ), "Burn cost did not change after subnet creation"

    # Assert amount is deducted once a subnetwork is registered by Alice
    alice_balance_post_sn = subtensor.wallets.get_balance(
        alice_wallet.coldkeypub.ss58_address
    )
    assert (
        alice_balance_post_sn + pre_subnet_creation_cost + response.extrinsic_fee
        == initial_alice_balance
    ), "Balance is the same even after registering a subnet."

    # Subnet 2 is added after registration
    assert subtensor.subnets.get_all_subnets_netuid() == [0, 1, 2]
    assert subtensor.subnets.get_total_subnets() == 3

    # Verify subnet 2 created successfully
    assert subtensor.subnets.subnet_exists(netuid)

    # Default subnetwork difficulty
    assert subtensor.subnets.difficulty(netuid) == 10_000_000, (
        "Couldn't fetch correct subnet difficulty"
    )

    # Verify Alice is registered to netuid 2 and Bob isn't registered to any
    assert subtensor.wallets.get_netuids_for_hotkey(
        hotkey_ss58=alice_wallet.hotkey.ss58_address
    ) == [
        netuid,
    ], "Alice is not registered to netuid 2 as expected"
    assert (
        subtensor.wallets.get_netuids_for_hotkey(
            hotkey_ss58=bob_wallet.hotkey.ss58_address
        )
        == []
    ), "Bob is unexpectedly registered to some netuid"

    # Verify Alice's hotkey is registered to any subnet (currently netuid = 2)
    assert subtensor.wallets.is_hotkey_registered_any(
        hotkey_ss58=alice_wallet.hotkey.ss58_address
    ), "Alice's hotkey is not registered to any subnet"
    assert not subtensor.wallets.is_hotkey_registered_any(
        hotkey_ss58=bob_wallet.hotkey.ss58_address
    ), "Bob's hotkey is unexpectedly registered to a subnet"

    # Verify netuid = 2 only has Alice registered and not Bob
    assert subtensor.wallets.is_hotkey_registered_on_subnet(
        netuid=netuid, hotkey_ss58=alice_wallet.hotkey.ss58_address
    ), "Alice's hotkey is not registered on netuid 1"
    assert not subtensor.wallets.is_hotkey_registered_on_subnet(
        netuid=netuid, hotkey_ss58=bob_wallet.hotkey.ss58_address
    ), "Bob's hotkey is unexpectedly registered on netuid 1"

    # Verify Alice's UID on netuid 2 is 0
    assert (
        subtensor.subnets.get_uid_for_hotkey_on_subnet(
            hotkey_ss58=alice_wallet.hotkey.ss58_address, netuid=netuid
        )
        == 0
    ), "UID for Alice's hotkey on netuid 2 is not 0 as expected"

    bob_balance = subtensor.wallets.get_balance(bob_wallet.coldkeypub.ss58_address)

    alice_sn.execute_steps(
        [
            ACTIVATE_SUBNET(alice_wallet),
            REGISTER_NEURON(bob_wallet),
        ]
    )

    # Verify Bob's UID on netuid 2 is 1
    assert (
        subtensor.subnets.get_uid_for_hotkey_on_subnet(
            hotkey_ss58=bob_wallet.hotkey.ss58_address, netuid=netuid
        )
        == 1
    ), "UID for Bob's hotkey on netuid 2 is not 1 as expected"

    # Fetch recycle_amount to register to the subnet
    recycle_amount = subtensor.subnets.recycle(netuid)
    fee = alice_sn.calls[-1].response.extrinsic_fee
    bob_balance_post_reg = subtensor.wallets.get_balance(
        bob_wallet.coldkeypub.ss58_address
    )

    # Ensure recycled amount is only deducted from the balance after registration
    assert bob_balance - recycle_amount - fee == bob_balance_post_reg, (
        "Balance for Bob is not correct after burned register"
    )

    async with templates.validator(alice_wallet, netuid):
        await asyncio.sleep(
            5
        )  # wait for 5 seconds for the metagraph and subtensor to refresh with latest data

        alice_sn.wait_next_epoch()

    # Fetch and assert existential deposit for an account in the network
    assert subtensor.chain.get_existential_deposit() == existential_deposit, (
        "Existential deposit value doesn't match with pre-defined value"
    )

    # Fetching all subnets in the network
    all_subnets = subtensor.subnets.get_all_subnets_info()

    # Assert all netuids are present in all_subnets
    expected_netuids = [0, 1, 2]
    actual_netuids = [subnet.netuid for subnet in all_subnets]
    assert actual_netuids == expected_netuids, (
        f"Expected netuids {expected_netuids}, but found {actual_netuids}"
    )

    # Assert that the owner_ss58 of subnet 2 matches Alice's coldkey address
    expected_owner = alice_wallet.coldkeypub.ss58_address
    subnet_2 = next((subnet for subnet in all_subnets if subnet.netuid == netuid), None)
    actual_owner = subnet_2.owner_ss58
    assert actual_owner == expected_owner, (
        f"Expected owner {expected_owner}, but found {actual_owner}"
    )


@pytest.mark.asyncio
async def test_subtensor_extrinsics_async(
    async_subtensor, templates, alice_wallet, bob_wallet
):
    """
    Tests subtensor extrinsics

    Steps:
        1. Validate subnets in the chain before/after registering netuid = 1
        2. Register Alice's neuron
        3. Verify Alice and Bob's participation in subnets (individually and global)
        4. Verify uids of Alice and Bob gets populated correctly
        5. Start Alice as a validator and verify NeuronInfo before/after is different
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    netuid = await async_subtensor.subnets.get_total_subnets()  # 22
    # Initial balance for Alice, defined in the genesis file of localnet
    initial_alice_balance = Balance.from_tao(1_000_000)
    # Current Existential deposit for all accounts in bittensor
    existential_deposit = Balance.from_tao(0.000_000_500)

    # Subnets 0 and 1 are bootstrapped from the start
    assert await async_subtensor.subnets.get_all_subnets_netuid() == [0, 1]
    assert await async_subtensor.subnets.get_total_subnets() == 2

    # Assert correct balance is fetched for Alice
    alice_balance = await async_subtensor.wallets.get_balance(
        alice_wallet.coldkeypub.ss58_address
    )
    assert alice_balance == initial_alice_balance, (
        "Balance for Alice wallet doesn't match with pre-def value"
    )

    # Subnet burn cost is initially lower before we register a subnet
    pre_subnet_creation_cost = await async_subtensor.subnets.get_subnet_burn_cost()

    # Register subnet
    alice_sn = TestSubnet(async_subtensor)
    response = await alice_sn.async_execute_one(REGISTER_SUBNET(alice_wallet))
    netuid = alice_sn.netuid

    # Subnet burn cost is increased immediately after a subnet is registered
    post_subnet_creation_cost = await async_subtensor.subnets.get_subnet_burn_cost()

    # Assert that the burn cost changed after registering a subnet
    assert Balance.from_tao(pre_subnet_creation_cost) < Balance.from_tao(
        post_subnet_creation_cost
    ), "Burn cost did not change after subnet creation."

    # Assert amount is deducted once a subnetwork is registered by Alice
    alice_balance_post_sn = await async_subtensor.wallets.get_balance(
        alice_wallet.coldkeypub.ss58_address
    )
    assert (
        alice_balance_post_sn + pre_subnet_creation_cost + response.extrinsic_fee
        == initial_alice_balance
    ), "Balance is the same even after registering a subnet."

    # Subnet 2 is added after registration
    assert await async_subtensor.subnets.get_all_subnets_netuid() == [0, 1, 2]
    assert await async_subtensor.subnets.get_total_subnets() == 3

    # Verify subnet 2 created successfully
    assert await async_subtensor.subnets.subnet_exists(netuid)

    # Default subnetwork difficulty
    assert await async_subtensor.subnets.difficulty(netuid) == 10_000_000, (
        "Couldn't fetch correct subnet difficulty."
    )

    # Verify Alice is registered to netuid 2 and Bob isn't registered to any
    assert await async_subtensor.wallets.get_netuids_for_hotkey(
        hotkey_ss58=alice_wallet.hotkey.ss58_address
    ) == [
        netuid,
    ], "Alice is not registered to netuid 2 as expected."
    assert (
        await async_subtensor.wallets.get_netuids_for_hotkey(
            hotkey_ss58=bob_wallet.hotkey.ss58_address
        )
        == []
    ), "Bob is unexpectedly registered to some netuid."

    # Verify Alice's hotkey is registered to any subnet (currently netuid = 2)
    assert await async_subtensor.wallets.is_hotkey_registered_any(
        hotkey_ss58=alice_wallet.hotkey.ss58_address
    ), "Alice's hotkey is not registered to any subnet."
    assert not await async_subtensor.wallets.is_hotkey_registered_any(
        hotkey_ss58=bob_wallet.hotkey.ss58_address
    ), "Bob's hotkey is unexpectedly registered to a subnet."

    # Verify netuid = 2 only has Alice registered and not Bob
    assert await async_subtensor.wallets.is_hotkey_registered_on_subnet(
        netuid=netuid, hotkey_ss58=alice_wallet.hotkey.ss58_address
    ), f"Alice's hotkey is not registered on netuid {netuid}"
    assert not await async_subtensor.wallets.is_hotkey_registered_on_subnet(
        netuid=netuid, hotkey_ss58=bob_wallet.hotkey.ss58_address
    ), f"Bob's hotkey is unexpectedly registered on netuid {netuid}"

    # Verify Alice's UID on netuid 2 is 0
    assert (
        await async_subtensor.subnets.get_uid_for_hotkey_on_subnet(
            hotkey_ss58=alice_wallet.hotkey.ss58_address, netuid=netuid
        )
        == 0
    ), "UID for Alice's hotkey on netuid 2 is not 0 as expected."

    bob_balance = await async_subtensor.wallets.get_balance(
        bob_wallet.coldkeypub.ss58_address
    )

    await alice_sn.async_execute_steps(
        [
            ACTIVATE_SUBNET(alice_wallet),
            REGISTER_NEURON(bob_wallet),
        ]
    )

    # Verify Bob's UID on netuid 2 is 1
    assert (
        await async_subtensor.subnets.get_uid_for_hotkey_on_subnet(
            hotkey_ss58=bob_wallet.hotkey.ss58_address, netuid=netuid
        )
        == 1
    ), "UID for Bob's hotkey on netuid 2 is not 1 as expected."

    # Fetch recycle_amount to register to the subnet
    recycle_amount = await async_subtensor.subnets.recycle(netuid)
    fee = alice_sn.calls[-1].response.extrinsic_fee
    bob_balance_post_reg = await async_subtensor.wallets.get_balance(
        bob_wallet.coldkeypub.ss58_address
    )

    # Ensure recycled amount is only deducted from the balance after registration
    assert bob_balance - recycle_amount - fee == bob_balance_post_reg, (
        "Balance for Bob is not correct after burned register"
    )

    # neuron_info_old = subtensor.get_neuron_for_pubkey_and_subnet(
    #     alice_wallet.hotkey.ss58_address, netuid=netuid
    # )

    async with templates.validator(alice_wallet, netuid):
        await asyncio.sleep(
            5
        )  # wait for 5 seconds for the metagraph and subtensor to refresh with latest data

        await alice_sn.async_wait_next_epoch()

    # Fetch and assert existential deposit for an account in the network
    assert (
        await async_subtensor.chain.get_existential_deposit() == existential_deposit
    ), "Existential deposit value doesn't match with pre-defined value"

    # Fetching all subnets in the network
    all_subnets = await async_subtensor.subnets.get_all_subnets_info()

    # Assert all netuids are present in all_subnets
    expected_netuids = [0, 1, 2]
    actual_netuids = [subnet.netuid for subnet in all_subnets]
    assert actual_netuids == expected_netuids, (
        f"Expected netuids {expected_netuids}, but found {actual_netuids}"
    )

    # Assert that the owner_ss58 of subnet 2 matches Alice's coldkey address
    expected_owner = alice_wallet.coldkeypub.ss58_address
    subnet_2 = next((subnet for subnet in all_subnets if subnet.netuid == netuid), None)
    actual_owner = subnet_2.owner_ss58
    assert actual_owner == expected_owner, (
        f"Expected owner {expected_owner}, but found {actual_owner}"
    )


def test_blocks(subtensor):
    """
    Tests:
    - Get current block
    - Get block hash
    - Wait for block
    """
    get_current_block = subtensor.chain.get_current_block()
    block = subtensor.block

    # Several random tests fail during the block finalization period. Fast blocks of 0.25 seconds (very fast)
    assert get_current_block in [block, block + 1]

    block_hash = subtensor.chain.get_block_hash(block)
    assert re.match("0x[a-z0-9]{64}", block_hash)

    subtensor.wait_for_block(block + 10)
    assert subtensor.chain.get_current_block() == block + 10

    logging.console.info("✅ Passed [blue]test_blocks[/blue]")


@pytest.mark.asyncio
async def test_blocks_async(subtensor):
    """
    Async tests:
    - Get current block
    - Get block hash
    - Wait for block
    """
    block = subtensor.chain.get_current_block()
    assert block == subtensor.block

    block_hash = subtensor.chain.get_block_hash(block)
    assert re.match("0x[a-z0-9]{64}", block_hash)

    subtensor.wait_for_block(block + 10)
    assert subtensor.chain.get_current_block() in [block + 10, block + 11]
    logging.console.info("✅ Passed [blue]test_blocks_async[/blue]")


@pytest.mark.parametrize(
    "block, block_hash, result",
    [
        (None, None, True),
        (1, None, True),
        (None, "SOME_HASH", True),
        (1, "SOME_HASH", False),
    ],
)
def test_block_info(subtensor, block, block_hash, result):
    """Tests sync get_block_info."""
    if block_hash:
        block_hash = subtensor.chain.get_block_hash()

    subtensor.wait_for_block(2)

    try:
        res = subtensor.chain.get_block_info(block=block, block_hash=block_hash)
        assert (res is not None) == result
    except Exception as e:
        assert "Either block_hash or block_number should be set" in str(e)


@pytest.mark.parametrize(
    "block, block_hash, result",
    [
        (None, None, True),
        (1, None, True),
        (None, "SOME_HASH", True),
        (1, "SOME_HASH", False),
    ],
)
@pytest.mark.asyncio
async def test_block_info(async_subtensor, block, block_hash, result):
    """Tests async get_block_info."""
    if block_hash:
        block_hash = await async_subtensor.chain.get_block_hash()

    await async_subtensor.wait_for_block(2)

    try:
        res = await async_subtensor.chain.get_block_info(
            block=block, block_hash=block_hash
        )
        assert (res is not None) == result
    except Exception as e:
        assert "Either block_hash or block_number should be set" in str(e)
