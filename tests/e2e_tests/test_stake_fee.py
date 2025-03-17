import pytest
from bittensor import Balance


@pytest.mark.parametrize("local_chain", [False], indirect=True)
@pytest.mark.asyncio
async def test_stake_fee_api(local_chain, subtensor, alice_wallet, bob_wallet):
    """
    Tests the stake fee calculation mechanism for various staking operations

    Steps:
        1. Register a subnet through Alice
        2. Set up test parameters
        3. Test stake fees for different scenarios:
            - Adding new stake
            - Removing stake
            - Moving between subnets
            - Moving between hotkeys
            - Moving between coldkeys
            - Swapping between subnets
    """
    MIN_STAKE_FEE = Balance.from_rao(50_000)
    netuid = 2
    root_netuid = 0
    stake_amount = Balance.from_tao(100).rao  # 100 TAO

    # Register subnet as Alice
    assert subtensor.register_subnet(alice_wallet), "Unable to register the subnet"
    assert subtensor.subnet_exists(netuid), "Subnet wasn't created successfully"

    # Add_stake (new stake)
    stake_fee_0 = subtensor.get_stake_fee(
        origin_hotkey_ss58=None,
        origin_netuid=None,
        origin_coldkey_ss58=alice_wallet.coldkeypub.ss58_address,
        destination_hotkey_ss58=alice_wallet.hotkey.ss58_address,
        destination_netuid=netuid,
        destination_coldkey_ss58=alice_wallet.coldkeypub.ss58_address,
        amount=stake_amount,
    )
    assert isinstance(stake_fee_0, Balance), "Stake fee should be a Balance object"

    # Remove stake
    stake_fee_1 = subtensor.get_stake_fee(
        origin_hotkey_ss58=bob_wallet.hotkey.ss58_address,
        origin_netuid=root_netuid,
        origin_coldkey_ss58=alice_wallet.coldkeypub.ss58_address,
        destination_hotkey_ss58=None,
        destination_netuid=None,
        destination_coldkey_ss58=alice_wallet.coldkeypub.ss58_address,
        amount=stake_amount,
    )
    assert isinstance(stake_fee_1, Balance), "Stake fee should be a Balance object"

    # Move from root to non-root
    stake_fee_2 = subtensor.get_stake_fee(
        origin_hotkey_ss58=alice_wallet.hotkey.ss58_address,
        origin_netuid=root_netuid,
        origin_coldkey_ss58=alice_wallet.coldkeypub.ss58_address,
        destination_hotkey_ss58=alice_wallet.hotkey.ss58_address,
        destination_netuid=netuid,
        destination_coldkey_ss58=alice_wallet.coldkeypub.ss58_address,
        amount=stake_amount,
    )
    assert isinstance(stake_fee_2, Balance), "Stake fee should be a Balance object"

    # Move between hotkeys on root
    stake_fee_3 = subtensor.get_stake_fee(
        origin_hotkey_ss58=alice_wallet.hotkey.ss58_address,
        origin_netuid=root_netuid,
        origin_coldkey_ss58=alice_wallet.coldkeypub.ss58_address,
        destination_hotkey_ss58=bob_wallet.hotkey.ss58_address,
        destination_netuid=root_netuid,
        destination_coldkey_ss58=alice_wallet.coldkeypub.ss58_address,
        amount=stake_amount,
    )
    assert isinstance(stake_fee_3, Balance), "Stake fee should be a Balance object"

    # Move between coldkeys on root
    stake_fee_4 = subtensor.get_stake_fee(
        origin_hotkey_ss58=bob_wallet.hotkey.ss58_address,
        origin_netuid=root_netuid,
        origin_coldkey_ss58=alice_wallet.coldkeypub.ss58_address,
        destination_hotkey_ss58=bob_wallet.hotkey.ss58_address,
        destination_netuid=root_netuid,
        destination_coldkey_ss58=bob_wallet.coldkeypub.ss58_address,
        amount=stake_amount,
    )
    assert isinstance(stake_fee_4, Balance), "Stake fee should be a Balance object"

    # Swap from non-root to root
    stake_fee_5 = subtensor.get_stake_fee(
        origin_hotkey_ss58=bob_wallet.hotkey.ss58_address,
        origin_netuid=netuid,
        origin_coldkey_ss58=alice_wallet.coldkeypub.ss58_address,
        destination_hotkey_ss58=bob_wallet.hotkey.ss58_address,
        destination_netuid=root_netuid,
        destination_coldkey_ss58=alice_wallet.coldkeypub.ss58_address,
        amount=stake_amount,
    )
    assert isinstance(stake_fee_5, Balance), "Stake fee should be a Balance object"

    # Move between hotkeys on non-root
    stake_fee_6 = subtensor.get_stake_fee(
        origin_hotkey_ss58=bob_wallet.hotkey.ss58_address,
        origin_netuid=netuid,
        origin_coldkey_ss58=alice_wallet.coldkeypub.ss58_address,
        destination_hotkey_ss58=alice_wallet.hotkey.ss58_address,
        destination_netuid=netuid,
        destination_coldkey_ss58=alice_wallet.coldkeypub.ss58_address,
        amount=stake_amount,
    )
    assert isinstance(stake_fee_6, Balance), "Stake fee should be a Balance object"

    # Move between coldkeys on non-root
    stake_fee_7 = subtensor.get_stake_fee(
        origin_hotkey_ss58=bob_wallet.hotkey.ss58_address,
        origin_netuid=netuid,
        origin_coldkey_ss58=alice_wallet.coldkeypub.ss58_address,
        destination_hotkey_ss58=bob_wallet.hotkey.ss58_address,
        destination_netuid=netuid,
        destination_coldkey_ss58=bob_wallet.coldkeypub.ss58_address,
        amount=stake_amount,
    )
    assert isinstance(stake_fee_7, Balance), "Stake fee should be a Balance object"

    # Swap from non-root to non-root (between subnets)
    netuid2 = 3
    assert subtensor.register_subnet(
        alice_wallet
    ), "Unable to register the second subnet"
    assert subtensor.subnet_exists(netuid2), "Second subnet wasn't created successfully"

    stake_fee_8 = subtensor.get_stake_fee(
        origin_hotkey_ss58=bob_wallet.hotkey.ss58_address,
        origin_netuid=netuid,
        origin_coldkey_ss58=alice_wallet.coldkeypub.ss58_address,
        destination_hotkey_ss58=bob_wallet.hotkey.ss58_address,
        destination_netuid=netuid2,
        destination_coldkey_ss58=alice_wallet.coldkeypub.ss58_address,
        amount=stake_amount,
    )
    assert isinstance(stake_fee_8, Balance), "Stake fee should be a Balance object"

    # Verify all fees are non-zero
    assert stake_fee_0 >= MIN_STAKE_FEE, "Stake fee should be greater than 0"
    assert stake_fee_1 >= MIN_STAKE_FEE, "Stake fee should be greater than 0"
    assert stake_fee_2 >= MIN_STAKE_FEE, "Stake fee should be greater than 0"
    assert stake_fee_3 >= MIN_STAKE_FEE, "Stake fee should be greater than 0"
    assert stake_fee_4 >= MIN_STAKE_FEE, "Stake fee should be greater than 0"
    assert stake_fee_5 >= MIN_STAKE_FEE, "Stake fee should be greater than 0"
    assert stake_fee_6 >= MIN_STAKE_FEE, "Stake fee should be greater than 0"
    assert stake_fee_7 >= MIN_STAKE_FEE, "Stake fee should be greater than 0"
    assert stake_fee_8 >= MIN_STAKE_FEE, "Stake fee should be greater than 0"
