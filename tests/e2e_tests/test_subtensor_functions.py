import asyncio
import time

import pytest
from bittensor.utils.btlogging import logging
from bittensor.core.extrinsics.utils import get_extrinsic_fee
from bittensor.utils.balance import Balance
from tests.e2e_tests.utils.chain_interactions import (
    async_wait_epoch,
    wait_epoch,
)
from tests.e2e_tests.utils.e2e_test_utils import (
    async_wait_to_start_call,
    wait_to_start_call,
)

"""
Verifies:

* get_subnets()
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
    logging.console.info("Testing [blue]test_subtensor_extrinsics[/blue]")
    netuid = subtensor.subnets.get_total_subnets()  # 22
    # Initial balance for Alice, defined in the genesis file of localnet
    initial_alice_balance = Balance.from_tao(1_000_000)
    # Current Existential deposit for all accounts in bittensor
    existential_deposit = Balance.from_tao(0.000_000_500)

    # Subnets 0 and 1 are bootstrapped from the start
    assert subtensor.subnets.get_subnets() == [0, 1]
    assert subtensor.subnets.get_total_subnets() == 2

    # Assert correct balance is fetched for Alice
    alice_balance = subtensor.wallets.get_balance(alice_wallet.coldkeypub.ss58_address)
    assert alice_balance == initial_alice_balance, (
        "Balance for Alice wallet doesn't match with pre-def value"
    )

    # Subnet burn cost is initially lower before we register a subnet
    pre_subnet_creation_cost = subtensor.subnets.get_subnet_burn_cost()

    # Register subnet
    assert subtensor.subnets.register_subnet(alice_wallet, True, True), (
        "Unable to register the subnet"
    )

    # Subnet burn cost is increased immediately after a subnet is registered
    post_subnet_creation_cost = subtensor.subnets.get_subnet_burn_cost()

    # TODO: in SDKv10 replace this logic with using `ExtrinsicResponse.extrinsic_fee`
    call = subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="register_network",
        call_params={
            "hotkey": alice_wallet.hotkey.ss58_address,
            "mechid": 1,
        },
    )
    register_fee = get_extrinsic_fee(call, alice_wallet.hotkey, subtensor)

    # Assert that the burn cost changed after registering a subnet
    assert Balance.from_tao(pre_subnet_creation_cost) < Balance.from_tao(
        post_subnet_creation_cost
    ), "Burn cost did not change after subnet creation"

    # Assert amount is deducted once a subnetwork is registered by Alice
    alice_balance_post_sn = subtensor.wallets.get_balance(
        alice_wallet.coldkeypub.ss58_address
    )
    assert alice_balance_post_sn + pre_subnet_creation_cost + register_fee == initial_alice_balance, (
        "Balance is the same even after registering a subnet"
    )

    # Subnet 2 is added after registration
    assert subtensor.subnets.get_subnets() == [0, 1, 2]
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

    assert wait_to_start_call(subtensor, alice_wallet, netuid)

    # Register Bob to the subnet
    assert subtensor.subnets.burned_register(bob_wallet, netuid), (
        "Unable to register Bob as a neuron"
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
    call = subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="burned_register",
        call_params={
            "netuid": netuid,
            "hotkey": bob_wallet.hotkey.ss58_address,
        },
    )
    payment_info = subtensor.substrate.get_payment_info(call, bob_wallet.coldkeypub)
    fee = Balance.from_rao(payment_info["partial_fee"])
    bob_balance_post_reg = subtensor.wallets.get_balance(
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

        await wait_epoch(subtensor, netuid)

    # Verify neuron info is updated after running as a validator
    # neuron_info = subtensor.get_neuron_for_pubkey_and_subnet(
    #     alice_wallet.hotkey.ss58_address, netuid=netuid
    # )
    # assert (
    #     neuron_info_old.dividends != neuron_info.dividends
    # ), "Neuron info not updated after running validator"

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

    logging.console.success("✅ Passed [blue]test_subtensor_extrinsics[/blue]")


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
    logging.console.info("Testing [blue]test_subtensor_extrinsics[/blue]")
    netuid = await async_subtensor.subnets.get_total_subnets()  # 22
    # Initial balance for Alice, defined in the genesis file of localnet
    initial_alice_balance = Balance.from_tao(1_000_000)
    # Current Existential deposit for all accounts in bittensor
    existential_deposit = Balance.from_tao(0.000_000_500)

    # Subnets 0 and 1 are bootstrapped from the start
    assert await async_subtensor.subnets.get_subnets() == [0, 1]
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
    assert await async_subtensor.subnets.register_subnet(alice_wallet, True, True), (
        "Unable to register the subnet"
    )

    # Subnet burn cost is increased immediately after a subnet is registered
    post_subnet_creation_cost = await async_subtensor.subnets.get_subnet_burn_cost()

    # Assert that the burn cost changed after registering a subnet
    assert Balance.from_tao(pre_subnet_creation_cost) < Balance.from_tao(
        post_subnet_creation_cost
    ), "Burn cost did not change after subnet creation"

    # Assert amount is deducted once a subnetwork is registered by Alice
    alice_balance_post_sn = await async_subtensor.wallets.get_balance(
        alice_wallet.coldkeypub.ss58_address
    )
    assert alice_balance_post_sn + pre_subnet_creation_cost == initial_alice_balance, (
        "Balance is the same even after registering a subnet"
    )

    # Subnet 2 is added after registration
    assert await async_subtensor.subnets.get_subnets() == [0, 1, 2]
    assert await async_subtensor.subnets.get_total_subnets() == 3

    # Verify subnet 2 created successfully
    assert await async_subtensor.subnets.subnet_exists(netuid)

    # Default subnetwork difficulty
    assert await async_subtensor.subnets.difficulty(netuid) == 10_000_000, (
        "Couldn't fetch correct subnet difficulty"
    )

    # Verify Alice is registered to netuid 2 and Bob isn't registered to any
    assert await async_subtensor.wallets.get_netuids_for_hotkey(
        hotkey_ss58=alice_wallet.hotkey.ss58_address
    ) == [
        netuid,
    ], "Alice is not registered to netuid 2 as expected"
    assert (
        await async_subtensor.wallets.get_netuids_for_hotkey(
            hotkey_ss58=bob_wallet.hotkey.ss58_address
        )
        == []
    ), "Bob is unexpectedly registered to some netuid"

    # Verify Alice's hotkey is registered to any subnet (currently netuid = 2)
    assert await async_subtensor.wallets.is_hotkey_registered_any(
        hotkey_ss58=alice_wallet.hotkey.ss58_address
    ), "Alice's hotkey is not registered to any subnet"
    assert not await async_subtensor.wallets.is_hotkey_registered_any(
        hotkey_ss58=bob_wallet.hotkey.ss58_address
    ), "Bob's hotkey is unexpectedly registered to a subnet"

    # Verify netuid = 2 only has Alice registered and not Bob
    assert await async_subtensor.wallets.is_hotkey_registered_on_subnet(
        netuid=netuid, hotkey_ss58=alice_wallet.hotkey.ss58_address
    ), "Alice's hotkey is not registered on netuid 1"
    assert not await async_subtensor.wallets.is_hotkey_registered_on_subnet(
        netuid=netuid, hotkey_ss58=bob_wallet.hotkey.ss58_address
    ), "Bob's hotkey is unexpectedly registered on netuid 1"

    # Verify Alice's UID on netuid 2 is 0
    assert (
        await async_subtensor.subnets.get_uid_for_hotkey_on_subnet(
            hotkey_ss58=alice_wallet.hotkey.ss58_address, netuid=netuid
        )
        == 0
    ), "UID for Alice's hotkey on netuid 2 is not 0 as expected"

    bob_balance = await async_subtensor.wallets.get_balance(
        bob_wallet.coldkeypub.ss58_address
    )

    assert await async_wait_to_start_call(async_subtensor, alice_wallet, netuid)

    # Register Bob to the subnet
    assert await async_subtensor.subnets.burned_register(bob_wallet, netuid), (
        "Unable to register Bob as a neuron"
    )

    # Verify Bob's UID on netuid 2 is 1
    assert (
        await async_subtensor.subnets.get_uid_for_hotkey_on_subnet(
            hotkey_ss58=bob_wallet.hotkey.ss58_address, netuid=netuid
        )
        == 1
    ), "UID for Bob's hotkey on netuid 2 is not 1 as expected"

    # Fetch recycle_amount to register to the subnet
    recycle_amount = await async_subtensor.subnets.recycle(netuid)
    call = await async_subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="burned_register",
        call_params={
            "netuid": netuid,
            "hotkey": bob_wallet.hotkey.ss58_address,
        },
    )
    payment_info = await async_subtensor.substrate.get_payment_info(
        call, bob_wallet.coldkeypub
    )
    fee = Balance.from_rao(payment_info["partial_fee"])
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

        await async_wait_epoch(async_subtensor, netuid)

    # Verify neuron info is updated after running as a validator
    # neuron_info = subtensor.get_neuron_for_pubkey_and_subnet(
    #     alice_wallet.hotkey.ss58_address, netuid=netuid
    # )
    # assert (
    #     neuron_info_old.dividends != neuron_info.dividends
    # ), "Neuron info not updated after running validator"

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

    logging.console.success("✅ Passed [blue]test_subtensor_extrinsics[/blue]")
