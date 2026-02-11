import pytest

from bittensor.utils.balance import Balance
from tests.e2e_tests.utils import (
    ACTIVATE_SUBNET,
    REGISTER_NEURON,
    REGISTER_SUBNET,
    TestSubnet,
)


def test_add_stake_burn(subtensor, alice_wallet, bob_wallet):
    """Tests subnet buyback without limit price.

    Steps:
    - Create subnet and register neuron for the target hotkey
    - Verify no stake before buyback
    - Execute subnet buyback as subnet owner
    - Confirm stake is burned and coldkey balance decreases
    """
    alice_sn = TestSubnet(subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
        REGISTER_NEURON(bob_wallet),
    ]
    alice_sn.execute_steps(steps)

    # no stake before buyback
    stake_before = subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    assert stake_before == Balance(0).set_unit(alice_sn.netuid)

    # track coldkey balance before buyback
    balance_before = subtensor.wallets.get_balance(
        address=alice_wallet.coldkeypub.ss58_address
    )

    response = subtensor.staking.add_stake_burn(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=Balance.from_tao(10),
        period=16,
    )
    assert response.success, response.message

    # stake is burned immediately after buyback
    stake_after = subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    assert stake_after == Balance(0).set_unit(alice_sn.netuid)

    # buyback spends TAO from the subnet owner coldkey
    balance_after = subtensor.wallets.get_balance(
        address=alice_wallet.coldkeypub.ss58_address
    )
    assert balance_after < balance_before


@pytest.mark.asyncio
async def test_add_stake_burn_async(async_subtensor, alice_wallet, bob_wallet):
    """Tests subnet buyback without limit price (async).

    Steps:
    - Create subnet and register neuron for the target hotkey
    - Verify no stake before buyback
    - Execute subnet buyback as subnet owner
    - Confirm stake is burned and coldkey balance decreases
    """
    alice_sn = TestSubnet(async_subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
        REGISTER_NEURON(bob_wallet),
    ]
    await alice_sn.async_execute_steps(steps)

    # no stake before buyback
    stake_before = await async_subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    assert stake_before == Balance(0).set_unit(alice_sn.netuid)

    # track coldkey balance before buyback
    balance_before = await async_subtensor.wallets.get_balance(
        address=alice_wallet.coldkeypub.ss58_address
    )

    response = await async_subtensor.staking.add_stake_burn(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=Balance.from_tao(10),
        period=16,
    )
    assert response.success, response.message

    # stake is burned immediately after buyback
    stake_after = await async_subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    assert stake_after == Balance(0).set_unit(alice_sn.netuid)

    # buyback spends TAO from the subnet owner coldkey
    balance_after = await async_subtensor.wallets.get_balance(
        address=alice_wallet.coldkeypub.ss58_address
    )
    assert balance_after < balance_before


def test_add_stake_burn_with_limit_price(subtensor, alice_wallet, bob_wallet):
    """Tests subnet buyback with limit price.

    Steps:
    - Create subnet and register neuron for the target hotkey
    - Verify no stake before buyback
    - Execute subnet buyback with limit price as subnet owner
    - Confirm stake is burned and coldkey balance decreases
    """
    alice_sn = TestSubnet(subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
        REGISTER_NEURON(bob_wallet),
    ]
    alice_sn.execute_steps(steps)

    # no stake before buyback
    stake_before = subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    assert stake_before == Balance(0).set_unit(alice_sn.netuid)

    # track coldkey balance before buyback
    balance_before = subtensor.wallets.get_balance(
        address=alice_wallet.coldkeypub.ss58_address
    )

    response = subtensor.staking.add_stake_burn(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=Balance.from_tao(10),
        limit_price=Balance.from_tao(2),
        period=16,
    )
    assert response.success, response.message

    # stake is burned immediately after buyback
    stake_after = subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    assert stake_after == Balance(0).set_unit(alice_sn.netuid)

    # buyback spends TAO from the subnet owner coldkey
    balance_after = subtensor.wallets.get_balance(
        address=alice_wallet.coldkeypub.ss58_address
    )
    assert balance_after < balance_before


@pytest.mark.asyncio
async def test_add_stake_burn_with_limit_price_async(
    async_subtensor, alice_wallet, bob_wallet
):
    """Tests subnet buyback with limit price (async).

    Steps:
    - Create subnet and register neuron for the target hotkey
    - Verify no stake before buyback
    - Execute subnet buyback with limit price as subnet owner
    - Confirm stake is burned and coldkey balance decreases
    """
    alice_sn = TestSubnet(async_subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
        REGISTER_NEURON(bob_wallet),
    ]
    await alice_sn.async_execute_steps(steps)

    # no stake before buyback
    stake_before = await async_subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    assert stake_before == Balance(0).set_unit(alice_sn.netuid)

    # track coldkey balance before buyback
    balance_before = await async_subtensor.wallets.get_balance(
        address=alice_wallet.coldkeypub.ss58_address
    )

    response = await async_subtensor.staking.add_stake_burn(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=Balance.from_tao(10),
        limit_price=Balance.from_tao(2),
        period=16,
    )
    assert response.success, response.message

    # stake is burned immediately after buyback
    stake_after = await async_subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    assert stake_after == Balance(0).set_unit(alice_sn.netuid)

    # buyback spends TAO from the subnet owner coldkey
    balance_after = await async_subtensor.wallets.get_balance(
        address=alice_wallet.coldkeypub.ss58_address
    )
    assert balance_after < balance_before
