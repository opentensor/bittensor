import pytest
from bittensor import Balance


@pytest.mark.parametrize("local_chain", [False], indirect=True)
@pytest.mark.asyncio
async def test_stake_fee_api(local_chain, subtensor, alice_wallet, bob_wallet):
    """
    Tests the stake fee calculation mechanism for various staking operations

    Steps:
        1. Register a subnet through Alice
        2. Test stake fees for:
            - Adding new stake
            - Removing stake
            - Moving stake between hotkeys/subnets/coldkeys
    """

    netuid = 2
    root_netuid = 0
    stake_amount = Balance.from_tao(100)  # 100 TAO
    min_stake_fee = Balance.from_tao(0.050354772)

    # Register subnet as Alice
    assert subtensor.register_subnet(alice_wallet), "Unable to register the subnet"
    assert subtensor.subnet_exists(netuid), "Subnet wasn't created successfully"

    # Test add_stake fee
    stake_fee_0 = subtensor.get_stake_add_fee(
        amount=stake_amount,
        netuid=netuid,
        coldkey_ss58=alice_wallet.coldkeypub.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
    )
    assert isinstance(stake_fee_0, Balance), "Stake fee should be a Balance object."
    assert stake_fee_0 == min_stake_fee, (
        "Stake fee should be equal the minimum stake fee."
    )

    # Test unstake fee
    unstake_fee_root = subtensor.get_unstake_fee(
        amount=stake_amount,
        netuid=root_netuid,
        coldkey_ss58=alice_wallet.coldkeypub.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
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
            "origin_hotkey": alice_wallet.hotkey.ss58_address,
            "origin_coldkey": alice_wallet.coldkeypub.ss58_address,
            "dest_netuid": netuid,
            "dest_hotkey": alice_wallet.hotkey.ss58_address,
            "dest_coldkey": alice_wallet.coldkeypub.ss58_address,
            "stake_fee": min_stake_fee,
        },
        # Move between hotkeys on root
        {
            "origin_netuid": root_netuid,
            "origin_hotkey": alice_wallet.hotkey.ss58_address,
            "origin_coldkey": alice_wallet.coldkeypub.ss58_address,
            "dest_netuid": root_netuid,
            "dest_hotkey": bob_wallet.hotkey.ss58_address,
            "dest_coldkey": alice_wallet.coldkeypub.ss58_address,
            "stake_fee": 0,
        },
        # Move between coldkeys on root
        {
            "origin_netuid": root_netuid,
            "origin_hotkey": bob_wallet.hotkey.ss58_address,
            "origin_coldkey": alice_wallet.coldkeypub.ss58_address,
            "dest_netuid": root_netuid,
            "dest_hotkey": bob_wallet.hotkey.ss58_address,
            "dest_coldkey": bob_wallet.coldkeypub.ss58_address,
            "stake_fee": 0,
        },
        # Move between coldkeys on non-root
        {
            "origin_netuid": netuid,
            "origin_hotkey": bob_wallet.hotkey.ss58_address,
            "origin_coldkey": alice_wallet.coldkeypub.ss58_address,
            "dest_netuid": netuid,
            "dest_hotkey": bob_wallet.hotkey.ss58_address,
            "dest_coldkey": bob_wallet.coldkeypub.ss58_address,
            "stake_fee": min_stake_fee,
        },
    ]

    for scenario in movement_scenarios:
        stake_fee = subtensor.get_stake_movement_fee(
            amount=stake_amount,
            origin_netuid=scenario["origin_netuid"],
            origin_hotkey_ss58=scenario["origin_hotkey"],
            origin_coldkey_ss58=scenario["origin_coldkey"],
            destination_netuid=scenario["dest_netuid"],
            destination_hotkey_ss58=scenario["dest_hotkey"],
            destination_coldkey_ss58=scenario["dest_coldkey"],
        )
        assert isinstance(stake_fee, Balance), "Stake fee should be a Balance object"
        assert stake_fee >= scenario["stake_fee"], (
            "Stake fee should be greater than the minimum stake fee"
        )

    # Test cross-subnet movement
    netuid2 = 3
    assert subtensor.register_subnet(alice_wallet), (
        "Unable to register the second subnet"
    )
    assert subtensor.subnet_exists(netuid2), "Second subnet wasn't created successfully"

    stake_fee = subtensor.get_stake_movement_fee(
        amount=stake_amount,
        origin_netuid=netuid,
        origin_hotkey_ss58=bob_wallet.hotkey.ss58_address,
        origin_coldkey_ss58=alice_wallet.coldkeypub.ss58_address,
        destination_netuid=netuid2,
        destination_hotkey_ss58=bob_wallet.hotkey.ss58_address,
        destination_coldkey_ss58=alice_wallet.coldkeypub.ss58_address,
    )
    assert isinstance(stake_fee, Balance), "Stake fee should be a Balance object"
    assert stake_fee >= min_stake_fee, (
        "Stake fee should be greater than the minimum stake fee"
    )
