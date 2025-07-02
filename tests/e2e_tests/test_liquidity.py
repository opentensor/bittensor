import pytest

from bittensor import Balance
from tests.e2e_tests.utils.e2e_test_utils import wait_to_start_call
from bittensor.utils.liquidity import LiquidityPosition


@pytest.mark.asyncio
async def test_liquidity(local_chain, subtensor, alice_wallet, bob_wallet):
    """
    Tests the liquidity mechanism

    Steps:
        1. Check `get_liquidity_list` return None if SN doesn't exist.
        2. Register a subnet through Alice.
        3. Make sure `get_liquidity_list` return None without activ SN.
        4. Wait until start call availability and do this call.
        5. Add liquidity to the subnet and check `get_liquidity_list` return liquidity positions.
        6. Modify liquidity and check `get_liquidity_list` return new liquidity positions with modified liquidity value.
        7. Add second liquidity position and check `get_liquidity_list` return new liquidity positions with 0 index.
        8. Remove all liquidity positions and check `get_liquidity_list` return empty list.
    """

    alice_subnet_netuid = subtensor.get_total_subnets()  # 2

    # Make sure `get_liquidity_list` return None if SN doesn't exist
    assert (
        subtensor.get_liquidity_list(wallet=alice_wallet, netuid=alice_subnet_netuid)
        is None
    ), "❌ `get_liquidity_list` is not None for unexisting subnet."

    # Register root as Alice
    assert subtensor.register_subnet(alice_wallet), "❌ Unable to register the subnet"

    # Verify subnet 2 created successfully
    assert subtensor.subnets.subnet_exists(alice_subnet_netuid), (
        f"❌ Subnet {alice_subnet_netuid} wasn't created successfully"
    )

    # Make sure `get_liquidity_list` return None without activ SN
    assert (
        subtensor.subnets.get_liquidity_list(
            wallet=alice_wallet, netuid=alice_subnet_netuid
        )
        is None
    ), "❌ `get_liquidity_list` is not None when no activ subnet."

    # Wait until start call availability and do this call
    assert wait_to_start_call(subtensor, alice_wallet, alice_subnet_netuid)

    # Make sure `get_liquidity_list` return None without activ SN
    assert (
        subtensor.subnets.get_liquidity_list(
            wallet=alice_wallet, netuid=alice_subnet_netuid
        )
        == []
    ), "❌ `get_liquidity_list` is not empty list before fist liquidity add."

    # enable user liquidity in SN
    success, message = subtensor.extrinsics.toggle_user_liquidity(
        wallet=alice_wallet,
        netuid=alice_subnet_netuid,
        enable=True,
    )
    assert success, message
    assert message == "", "❌ Cannot enable user liquidity."

    # Add stake to herself to have Alpha (affect non-fast-blocks chain)
    assert subtensor.extrinsics.add_stake(
        wallet=alice_wallet,
        netuid=alice_subnet_netuid,
        amount=Balance.from_tao(100),
        wait_for_inclusion=True,
        wait_for_finalization=True,
    ), "❌ Cannot add stake."

    # Add liquidity
    success, message = subtensor.extrinsics.add_liquidity(
        wallet=alice_wallet,
        netuid=alice_subnet_netuid,
        liquidity=Balance.from_tao(1000),
        price_low=Balance.from_tao(0.9),
        price_high=Balance.from_tao(1.1),
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert success, message
    assert message == "", "❌ Cannot add liquidity."

    # Add liquidity
    liquidity_positions = subtensor.subnets.get_liquidity_list(
        wallet=alice_wallet, netuid=alice_subnet_netuid
    )

    assert len(liquidity_positions) == 1, (
        "❌ liquidity_positions has more than one element."
    )

    # Check if liquidity is correct
    liquidity_position = liquidity_positions[0]
    assert liquidity_position == LiquidityPosition(
        id=2,
        price_low=liquidity_position.price_low,
        price_high=liquidity_position.price_high,
        liquidity=Balance.from_tao(1000),
        fees_tao=Balance.from_tao(0),
        fees_alpha=Balance.from_tao(0, netuid=alice_subnet_netuid),
        netuid=alice_subnet_netuid,
    ), "❌ `get_liquidity_list` still empty list after liquidity add."

    # Modify liquidity position with positive value
    success, message = subtensor.extrinsics.modify_liquidity(
        wallet=alice_wallet,
        netuid=alice_subnet_netuid,
        position_id=liquidity_position.id,
        liquidity_delta=Balance.from_tao(500),
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert success, message
    assert message == "", "❌ cannot modify liquidity position."

    liquidity_positions = subtensor.subnets.get_liquidity_list(
        wallet=alice_wallet, netuid=alice_subnet_netuid
    )

    assert len(liquidity_positions) == 1, (
        "❌ liquidity_positions has more than one element."
    )
    liquidity_position = liquidity_positions[0]

    assert liquidity_position == LiquidityPosition(
        id=2,
        price_low=liquidity_position.price_low,
        price_high=liquidity_position.price_high,
        liquidity=Balance.from_tao(1500),
        fees_tao=Balance.from_tao(0),
        fees_alpha=Balance.from_tao(0, netuid=alice_subnet_netuid),
        netuid=alice_subnet_netuid,
    )

    # Modify liquidity position with negative value
    success, message = subtensor.extrinsics.modify_liquidity(
        wallet=alice_wallet,
        netuid=alice_subnet_netuid,
        position_id=liquidity_position.id,
        liquidity_delta=-Balance.from_tao(750),
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert success, message
    assert message == "", "❌ cannot modify liquidity position."

    liquidity_positions = subtensor.subnets.get_liquidity_list(
        wallet=alice_wallet, netuid=alice_subnet_netuid
    )

    assert len(liquidity_positions) == 1, (
        "❌ liquidity_positions has more than one element."
    )
    liquidity_position = liquidity_positions[0]

    assert liquidity_position == LiquidityPosition(
        id=2,
        price_low=liquidity_position.price_low,
        price_high=liquidity_position.price_high,
        liquidity=Balance.from_tao(750),
        fees_tao=Balance.from_tao(0),
        fees_alpha=Balance.from_tao(0, netuid=alice_subnet_netuid),
        netuid=alice_subnet_netuid,
    )

    # Add second liquidity position
    success, message = subtensor.extrinsics.add_liquidity(
        wallet=alice_wallet,
        netuid=alice_subnet_netuid,
        liquidity=Balance.from_tao(150),
        price_low=Balance.from_tao(0.8),
        price_high=Balance.from_tao(1.2),
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert success, message
    assert message == "", "❌ Cannot add liquidity."

    liquidity_positions = subtensor.subnets.get_liquidity_list(
        wallet=alice_wallet, netuid=alice_subnet_netuid
    )

    assert len(liquidity_positions) == 2, (
        f"❌ liquidity_positions should have 2 elements, but has only {len(liquidity_positions)} element."
    )

    # All new liquidity inserts on the 0 index
    liquidity_position = liquidity_positions[0]
    assert liquidity_position == LiquidityPosition(
        id=3,
        price_low=liquidity_position.price_low,
        price_high=liquidity_position.price_high,
        liquidity=Balance.from_tao(150),
        fees_tao=Balance.from_tao(0),
        fees_alpha=Balance.from_tao(0, netuid=alice_subnet_netuid),
        netuid=alice_subnet_netuid,
    )

    # Remove all liquidity positions
    for p in liquidity_positions:
        success, message = subtensor.extrinsics.remove_liquidity(
            wallet=alice_wallet,
            netuid=alice_subnet_netuid,
            position_id=p.id,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
        assert success, message
        assert message == "", "❌ Cannot remove liquidity."

    # Make sure all liquidity positions removed
    assert (
        subtensor.subnets.get_liquidity_list(
            wallet=alice_wallet, netuid=alice_subnet_netuid
        )
        == []
    ), "❌ Not all liquidity positions removed."
