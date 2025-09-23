import pytest
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging


def test_stake_fee_api(subtensor, alice_wallet, bob_wallet):
    """
    Tests the stake fee calculation mechanism for various staking operations

    Steps:
        1. Register a subnet through Alice
        2. Test stake fees for:
            - Adding new stake
            - Removing stake
            - Moving stake between hotkeys/subnets/coldkeys
    """
    logging.console.info("Testing [blue]test_stake_fee_api[/blue]")

    netuid = 2
    root_netuid = 0
    stake_amount = Balance.from_tao(100)  # 100 TAO
    min_stake_fee = Balance.from_tao(0.050354772)

    # Register subnet as Alice
    assert subtensor.subnets.register_subnet(alice_wallet), (
        "Unable to register the subnet"
    )
    assert subtensor.subnets.subnet_exists(netuid), "Subnet wasn't created successfully"

    # Test add_stake fee
    stake_fee_0 = subtensor.staking.get_stake_add_fee(
        amount=stake_amount,
        netuid=netuid,
    )
    assert isinstance(stake_fee_0, Balance), "Stake fee should be a Balance object."
    assert stake_fee_0 == min_stake_fee, (
        "Stake fee should be equal the minimum stake fee."
    )

    # Test unstake fee
    unstake_fee_root = subtensor.staking.get_unstake_fee(
        amount=stake_amount,
        netuid=root_netuid,
    )
    assert isinstance(unstake_fee_root, Balance), (
        "Stake fee should be a Balance object."
    )
    assert unstake_fee_root == min_stake_fee, (
        "Root unstake fee should be equal the minimum stake fee."
    )

    # Test various stake movement scenarios
    movement_scenarios = [
        # Move from root to non-root
        {
            "origin_netuid": root_netuid,
            "stake_fee": min_stake_fee,
        },
        # Move between hotkeys on root
        {
            "origin_netuid": root_netuid,
            "stake_fee": 0,
        },
        # Move between coldkeys on root
        {
            "origin_netuid": root_netuid,
            "stake_fee": 0,
        },
        # Move between coldkeys on non-root
        {
            "origin_netuid": netuid,
            "stake_fee": min_stake_fee,
        },
    ]

    for scenario in movement_scenarios:
        stake_fee = subtensor.staking.get_stake_movement_fee(
            amount=stake_amount,
            origin_netuid=scenario["origin_netuid"],
        )
        assert isinstance(stake_fee, Balance), "Stake fee should be a Balance object"
        assert stake_fee >= scenario["stake_fee"], (
            "Stake fee should be greater than the minimum stake fee"
        )

    # Test cross-subnet movement
    netuid2 = 3
    assert subtensor.subnets.register_subnet(alice_wallet), (
        "Unable to register the second subnet"
    )
    assert subtensor.subnets.subnet_exists(netuid2), (
        "Second subnet wasn't created successfully"
    )

    stake_fee = subtensor.staking.get_stake_movement_fee(
        amount=stake_amount,
        origin_netuid=netuid,
    )
    assert isinstance(stake_fee, Balance), "Stake fee should be a Balance object"
    assert stake_fee >= min_stake_fee, (
        "Stake fee should be greater than the minimum stake fee"
    )
    logging.console.success("✅ Passed [blue]test_stake_fee_api[/blue]")


@pytest.mark.asyncio
async def test_stake_fee_api_async(async_subtensor, alice_wallet, bob_wallet):
    """
    Tests the stake fee calculation mechanism for various staking operations

    Steps:
        1. Register a subnet through Alice
        2. Test stake fees for:
            - Adding new stake
            - Removing stake
            - Moving stake between hotkeys/subnets/coldkeys
    """
    logging.console.info("Testing [blue]test_stake_fee_api_async[/blue]")

    netuid = 2
    root_netuid = 0
    stake_amount = Balance.from_tao(100)  # 100 TAO
    min_stake_fee = Balance.from_tao(0.050354772)

    # Register subnet as Alice
    assert await async_subtensor.subnets.register_subnet(alice_wallet), (
        "Unable to register the subnet"
    )
    assert await async_subtensor.subnets.subnet_exists(netuid), (
        "Subnet wasn't created successfully"
    )

    # Test add_stake fee
    stake_fee_0 = await async_subtensor.staking.get_stake_add_fee(
        amount=stake_amount,
        netuid=netuid,
    )
    assert isinstance(stake_fee_0, Balance), "Stake fee should be a Balance object."
    assert stake_fee_0 == min_stake_fee, (
        "Stake fee should be equal the minimum stake fee."
    )

    # Test unstake fee
    unstake_fee_root = await async_subtensor.staking.get_unstake_fee(
        amount=stake_amount,
        netuid=root_netuid,
    )
    assert isinstance(unstake_fee_root, Balance), (
        "Stake fee should be a Balance object."
    )
    assert unstake_fee_root == min_stake_fee, (
        "Root unstake fee should be equal the minimum stake fee."
    )

    # Test various stake movement scenarios
    movement_scenarios = [
        # Move from root to non-root
        {
            "origin_netuid": root_netuid,
            "stake_fee": min_stake_fee,
        },
        # Move between hotkeys on root
        {
            "origin_netuid": root_netuid,
            "stake_fee": 0,
        },
        # Move between coldkeys on root
        {
            "origin_netuid": root_netuid,
            "stake_fee": 0,
        },
        # Move between coldkeys on non-root
        {
            "origin_netuid": netuid,
            "stake_fee": min_stake_fee,
        },
    ]

    for scenario in movement_scenarios:
        stake_fee = await async_subtensor.staking.get_stake_movement_fee(
            amount=stake_amount,
            origin_netuid=scenario["origin_netuid"],
        )
        assert isinstance(stake_fee, Balance), "Stake fee should be a Balance object"
        assert stake_fee >= scenario["stake_fee"], (
            "Stake fee should be greater than the minimum stake fee"
        )

    # Test cross-subnet movement
    netuid2 = 3
    assert await async_subtensor.subnets.register_subnet(alice_wallet), (
        "Unable to register the second subnet"
    )
    assert await async_subtensor.subnets.subnet_exists(netuid2), (
        "Second subnet wasn't created successfully"
    )

    stake_fee = await async_subtensor.staking.get_stake_movement_fee(
        amount=stake_amount,
        origin_netuid=netuid,
    )
    assert isinstance(stake_fee, Balance), "Stake fee should be a Balance object"
    assert stake_fee >= min_stake_fee, (
        "Stake fee should be greater than the minimum stake fee"
    )
    logging.console.success("✅ Passed [blue]test_stake_fee_api_async[/blue]")
