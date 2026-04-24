import pytest

from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging
from tests.e2e_tests.utils import (
    TestSubnet,
    ACTIVATE_SUBNET,
    REGISTER_SUBNET,
)


def test_register_limit_success(subtensor, alice_wallet, bob_wallet):
    """Tests successful registration with register_limit when limit_price is above burn."""
    alice_sn = TestSubnet(subtensor)
    alice_sn.execute_steps(
        [
            REGISTER_SUBNET(alice_wallet),
            ACTIVATE_SUBNET(alice_wallet),
        ]
    )

    recycle = subtensor.subnets.recycle(alice_sn.netuid)
    assert recycle is not None, (
        "Recycle amount should not be None after subnet activation"
    )

    limit_price = recycle * 2
    logging.console.info(
        f"Registering Bob with register_limit on SN #{alice_sn.netuid}, "
        f"recycle={recycle}, limit_price={limit_price}"
    )
    result = subtensor.subnets.register_limit(bob_wallet, alice_sn.netuid, limit_price)
    assert result.success, "register_limit should succeed with limit above burn price"


def test_register_limit_price_exceeded(subtensor, alice_wallet, bob_wallet):
    """Tests that register_limit fails when limit_price is below the current burn price."""
    alice_sn = TestSubnet(subtensor)
    alice_sn.execute_steps(
        [
            REGISTER_SUBNET(alice_wallet),
            ACTIVATE_SUBNET(alice_wallet),
        ]
    )

    logging.console.info(
        f"Attempting register_limit with limit_price=1 on SN #{alice_sn.netuid}"
    )
    result = subtensor.subnets.register_limit(
        bob_wallet, alice_sn.netuid, limit_price=Balance.from_rao(1)
    )
    assert not result.success, (
        "register_limit should fail with limit_price=1 (below burn)"
    )


@pytest.mark.asyncio
async def test_register_limit_success_async(async_subtensor, alice_wallet, bob_wallet):
    """Tests successful async registration with register_limit when limit_price is above burn."""
    alice_sn = TestSubnet(async_subtensor)
    await alice_sn.async_execute_steps(
        [
            REGISTER_SUBNET(alice_wallet),
            ACTIVATE_SUBNET(alice_wallet),
        ]
    )

    recycle = await async_subtensor.subnets.recycle(alice_sn.netuid)
    assert recycle is not None, (
        "Recycle amount should not be None after subnet activation"
    )

    limit_price = recycle * 2
    logging.console.info(
        f"Registering Bob with register_limit on SN #{alice_sn.netuid}, "
        f"recycle={recycle}, limit_price={limit_price}"
    )
    result = await async_subtensor.subnets.register_limit(
        bob_wallet, alice_sn.netuid, limit_price
    )
    assert result.success, "register_limit should succeed with limit above burn price"


@pytest.mark.asyncio
async def test_register_limit_price_exceeded_async(
    async_subtensor, alice_wallet, bob_wallet
):
    """Tests that async register_limit fails when limit_price is below the current burn price."""
    alice_sn = TestSubnet(async_subtensor)
    await alice_sn.async_execute_steps(
        [
            REGISTER_SUBNET(alice_wallet),
            ACTIVATE_SUBNET(alice_wallet),
        ]
    )

    logging.console.info(
        f"Attempting register_limit with limit_price=1 on SN #{alice_sn.netuid}"
    )
    result = await async_subtensor.subnets.register_limit(
        bob_wallet, alice_sn.netuid, limit_price=Balance.from_rao(1)
    )
    assert not result.success, (
        "register_limit should fail with limit_price=1 (below burn)"
    )


def test_register_auto_limit_price(subtensor, alice_wallet, bob_wallet):
    """Tests successful registration via register() with auto-calculated limit_price."""
    alice_sn = TestSubnet(subtensor)
    alice_sn.execute_steps(
        [
            REGISTER_SUBNET(alice_wallet),
            ACTIVATE_SUBNET(alice_wallet),
        ]
    )

    logging.console.info(
        f"Registering Bob with register (auto limit_price) on SN #{alice_sn.netuid}"
    )
    result = subtensor.subnets.register(bob_wallet, alice_sn.netuid)
    assert result.success, "register should succeed with auto-calculated limit_price"


@pytest.mark.asyncio
async def test_register_auto_limit_price_async(
    async_subtensor, alice_wallet, bob_wallet
):
    """Tests successful async registration via register() with auto-calculated limit_price."""
    alice_sn = TestSubnet(async_subtensor)
    await alice_sn.async_execute_steps(
        [
            REGISTER_SUBNET(alice_wallet),
            ACTIVATE_SUBNET(alice_wallet),
        ]
    )

    logging.console.info(
        f"Registering Bob with register (auto limit_price) on SN #{alice_sn.netuid}"
    )
    result = await async_subtensor.subnets.register(bob_wallet, alice_sn.netuid)
    assert result.success, "register should succeed with auto-calculated limit_price"
