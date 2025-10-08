import pytest

from bittensor.utils.btlogging import logging
from bittensor.utils.balance import Balance
from tests.e2e_tests.utils import TestSubnet, REGISTER_SUBNET


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
    root_netuid = 0
    stake_amount = Balance.from_tao(1)  # 1 TAO
    min_stake_fee = Balance.from_tao(0.000503547)

    sn2 = TestSubnet(subtensor)
    sn2.execute_one(REGISTER_SUBNET(alice_wallet))

    # Test cross-subnet movement
    sn3 = TestSubnet(subtensor)
    sn3.execute_one(REGISTER_SUBNET(bob_wallet))

    # Test add_stake fee
    stake_fee_0 = subtensor.staking.get_stake_add_fee(
        amount=stake_amount,
        netuid=sn2.netuid,
    )
    assert isinstance(stake_fee_0, Balance), "Stake fee should be a Balance object."
    assert stake_fee_0 == min_stake_fee, (
        "Stake fee should be equal the minimum stake fee."
    )

    # Test unstake fee
    unstake_fee_root = subtensor.staking.get_unstake_fee(
        netuid=root_netuid,
        amount=stake_amount,
    )
    assert isinstance(unstake_fee_root, Balance), (
        "Stake fee should be a Balance object."
    )
    assert unstake_fee_root == Balance.from_tao(0), (
        "Root unstake fee should be equal o TAO fee."
    )

    # Test various stake movement scenarios
    movement_scenarios = [
        {
            "title": "Move from root to non-root",
            "origin_netuid": root_netuid,
            "destination_netuid": sn2.netuid,
            "stake_fee": min_stake_fee,
        },

        {
            "title": "Move between hotkeys on root",
            "origin_netuid": root_netuid,
            "destination_netuid": root_netuid,
            "stake_fee": 0,
        },

        {
            "title": "Move between coldkeys on root",
            "origin_netuid": root_netuid,
            "destination_netuid": root_netuid,
            "stake_fee": 0,
        },

        {
            "title": "Move between coldkeys on non-root",
            "origin_netuid": sn2.netuid,
            "destination_netuid": sn2.netuid,
            "stake_fee": min_stake_fee,
        },

        {
            "title": "Move between different subnets",
            "origin_netuid": sn2.netuid,
            "destination_netuid": sn3.netuid,
            "stake_fee": min_stake_fee,
        },
    ]

    for scenario in movement_scenarios:
        logging.console.info(f"Scenario: {scenario.get("title")}")
        stake_fee = subtensor.staking.get_stake_movement_fee(
            origin_netuid=scenario.get("origin_netuid"),
            destination_netuid=scenario.get("destination_netuid"),
            amount=stake_amount,
        )
        assert isinstance(stake_fee, Balance), "Stake fee should be a Balance object"
        assert scenario["stake_fee"] >= stake_fee


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
    root_netuid = 0
    stake_amount = Balance.from_tao(1)  # 1 TAO
    min_stake_fee = Balance.from_tao(0.000503547)

    sn2 = TestSubnet(async_subtensor)
    await sn2.async_execute_one(REGISTER_SUBNET(bob_wallet))

    # Test cross-subnet movement
    sn3 = TestSubnet(async_subtensor)
    await sn3.async_execute_one(REGISTER_SUBNET(bob_wallet))

    # Test add_stake fee
    stake_fee_0 = await async_subtensor.staking.get_stake_add_fee(
        amount=stake_amount,
        netuid=sn2.netuid,
    )
    assert isinstance(stake_fee_0, Balance), "Stake fee should be a Balance object."
    assert stake_fee_0 == min_stake_fee, (
        "Stake fee should be equal the minimum stake fee."
    )

    # Test unstake fee
    unstake_fee_root = await async_subtensor.staking.get_unstake_fee(
        netuid=root_netuid,
        amount=stake_amount,
    )
    assert isinstance(unstake_fee_root, Balance), (
        "Stake fee should be a Balance object."
    )
    assert unstake_fee_root == Balance.from_tao(0), (
        "Root unstake fee should be equal the minimum stake fee."
    )

    # Test various stake movement scenarios
    movement_scenarios = [
        {
            "title": "Move from root to non-root",
            "origin_netuid": root_netuid,
            "destination_netuid": sn2.netuid,
            "stake_fee": min_stake_fee,
        },

        {
            "title": "Move between hotkeys on root",
            "origin_netuid": root_netuid,
            "destination_netuid": root_netuid,
            "stake_fee": 0,
        },

        {
            "title": "Move between coldkeys on root",
            "origin_netuid": root_netuid,
            "destination_netuid": root_netuid,
            "stake_fee": 0,
        },

        {
            "title": "Move between coldkeys on non-root",
            "origin_netuid": sn2.netuid,
            "destination_netuid": sn2.netuid,
            "stake_fee": min_stake_fee,
        },

        {
            "title": "Move between different subnets",
            "origin_netuid": sn2.netuid,
            "destination_netuid": sn3.netuid,
            "stake_fee": min_stake_fee,
        },
    ]

    for scenario in movement_scenarios:
        logging.console.info(f"Scenario: {scenario.get("title")}")
        stake_fee = await async_subtensor.staking.get_stake_movement_fee(
            origin_netuid=scenario.get("origin_netuid"),
            destination_netuid=scenario.get("destination_netuid"),
            amount=stake_amount,
        )
        assert isinstance(stake_fee, Balance), "Stake fee should be a Balance object"
        assert scenario["stake_fee"] >= stake_fee
