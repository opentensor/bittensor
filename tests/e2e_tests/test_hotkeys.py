import pytest

import bittensor
from tests.e2e_tests.utils.chain_interactions import (
    sudo_set_admin_utils,
    wait_epoch,
)


SET_CHILDREN_COOLDOWN_PERIOD = 15
SET_CHILDREN_RATE_LIMIT = 150


@pytest.mark.asyncio
async def test_hotkeys(subtensor, alice_wallet):
    """
    Tests:
    - Check if Hotkey exists
    - Check if Hotkey is registered
    """

    coldkey = alice_wallet.coldkeypub.ss58_address
    hotkey = alice_wallet.hotkey.ss58_address

    with pytest.raises(ValueError, match="Invalid checksum"):
        await subtensor.does_hotkey_exist("fake")

    assert await subtensor.does_hotkey_exist(hotkey) is False
    assert await subtensor.get_hotkey_owner(hotkey) is None

    assert await subtensor.is_hotkey_registered(hotkey) is False
    assert await subtensor.is_hotkey_registered_any(hotkey) is False
    assert (
        await subtensor.is_hotkey_registered_on_subnet(
            hotkey,
            netuid=1,
        )
        is False
    )

    await subtensor.burned_register(
        alice_wallet,
        netuid=1,
    )

    assert await subtensor.does_hotkey_exist(hotkey) is True
    assert await subtensor.get_hotkey_owner(hotkey) == coldkey

    assert await subtensor.is_hotkey_registered(hotkey) is True
    assert await subtensor.is_hotkey_registered_any(hotkey) is True
    assert (
        await subtensor.is_hotkey_registered_on_subnet(
            hotkey,
            netuid=1,
        )
        is True
    )


@pytest.mark.asyncio
async def test_children(local_chain, subtensor, alice_wallet, bob_wallet):
    """
    Tests:
    - Get default children (empty list)
    - Update children list
    - Checking pending children
    - Checking cooldown period
    - Trigger rate limit
    - Clear children list
    """

    with pytest.raises(bittensor.RegistrationNotPermittedOnRootSubnet):
        await subtensor.set_children(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            netuid=0,
            children=[],
            raise_error=True,
        )

    with pytest.raises(bittensor.NonAssociatedColdKey):
        await subtensor.set_children(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            netuid=1,
            children=[],
            raise_error=True,
        )

    with pytest.raises(bittensor.SubNetworkDoesNotExist):
        await subtensor.set_children(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            netuid=2,
            children=[],
            raise_error=True,
        )

    await subtensor.burned_register(
        alice_wallet,
        netuid=1,
    )
    await subtensor.burned_register(
        bob_wallet,
        netuid=1,
    )

    success, children, error = await subtensor.get_children(
        alice_wallet.hotkey.ss58_address,
        netuid=1,
    )

    assert error == ""
    assert success is True
    assert children == []

    with pytest.raises(bittensor.InvalidChild):
        await subtensor.set_children(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            netuid=1,
            children=[
                (
                    1.0,
                    alice_wallet.hotkey.ss58_address,
                ),
            ],
            raise_error=True,
        )

    with pytest.raises(bittensor.TooManyChildren):
        await subtensor.set_children(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            netuid=1,
            children=[
                (
                    0.1,
                    bob_wallet.hotkey.ss58_address,
                )
                for _ in range(10)
            ],
            raise_error=True,
        )

    with pytest.raises(bittensor.ProportionOverflow):
        await subtensor.set_children(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            netuid=1,
            children=[
                (
                    1.0,
                    bob_wallet.hotkey.ss58_address,
                ),
                (
                    1.0,
                    "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
                ),
            ],
            raise_error=True,
        )

    with pytest.raises(bittensor.DuplicateChild):
        await subtensor.set_children(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            netuid=1,
            children=[
                (
                    0.5,
                    bob_wallet.hotkey.ss58_address,
                ),
                (
                    0.5,
                    bob_wallet.hotkey.ss58_address,
                ),
            ],
            raise_error=True,
        )

    await subtensor.set_children(
        alice_wallet,
        alice_wallet.hotkey.ss58_address,
        netuid=1,
        children=[
            (
                1.0,
                bob_wallet.hotkey.ss58_address,
            ),
        ],
        raise_error=True,
    )

    assert error == ""
    assert success is True

    set_children_block = await subtensor.get_current_block()

    # children not set yet (have to wait cool-down period)
    success, children, error = await subtensor.get_children(
        alice_wallet.hotkey.ss58_address,
        block=set_children_block,
        netuid=1,
    )

    assert success is True
    assert children == []
    assert error == ""

    # children are in pending state
    pending, cooldown = await subtensor.get_children_pending(
        alice_wallet.hotkey.ss58_address,
        netuid=1,
    )

    assert pending == [
        (
            1.0,
            "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        ),
    ]

    await subtensor.wait_for_block(cooldown)

    await wait_epoch(subtensor, netuid=1)

    success, children, error = await subtensor.get_children(
        alice_wallet.hotkey.ss58_address,
        netuid=1,
    )

    assert error == ""
    assert success is True
    assert children == [
        (
            1.0,
            bob_wallet.hotkey.ss58_address,
        )
    ]

    # pending queue is empty
    pending, cooldown = await subtensor.get_children_pending(
        alice_wallet.hotkey.ss58_address,
        netuid=1,
    )

    assert pending == []

    with pytest.raises(bittensor.TxRateLimitExceeded):
        await subtensor.set_children(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            netuid=1,
            children=[],
            raise_error=True,
        )

    await subtensor.wait_for_block(set_children_block + SET_CHILDREN_RATE_LIMIT)

    await subtensor.set_children(
        alice_wallet,
        alice_wallet.hotkey.ss58_address,
        netuid=1,
        children=[],
        raise_error=True,
    )
    set_children_block = await subtensor.get_current_block()

    pending, cooldown = await subtensor.get_children_pending(
        alice_wallet.hotkey.ss58_address,
        netuid=1,
    )

    assert pending == []

    await subtensor.wait_for_block(cooldown)

    await wait_epoch(subtensor, netuid=1)

    success, children, error = await subtensor.get_children(
        alice_wallet.hotkey.ss58_address,
        netuid=1,
    )

    assert error == ""
    assert success is True
    assert children == []

    await subtensor.wait_for_block(set_children_block + SET_CHILDREN_RATE_LIMIT)

    sudo_set_admin_utils(
        local_chain,
        alice_wallet,
        call_function="sudo_set_stake_threshold",
        call_params={
            "min_stake": 1_000_000_000_000,
        },
    )

    with pytest.raises(bittensor.NotEnoughStakeToSetChildkeys):
        await subtensor.set_children(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            netuid=1,
            children=[
                (
                    1.0,
                    bob_wallet.hotkey.ss58_address,
                ),
            ],
            raise_error=True,
        )
