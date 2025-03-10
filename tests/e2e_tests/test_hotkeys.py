import pytest

from tests.e2e_tests.utils.chain_interactions import (
    set_children,
    wait_epoch,
)


SET_CHILDREN_COOLDOWN_PERIOD = 15
SET_CHILDREN_RATE_LIMIT = 150


def test_hotkeys(subtensor, alice_wallet):
    """
    Tests:
    - Check if Hotkey exists
    - Check if Hotkey is registered
    """

    coldkey = alice_wallet.coldkeypub.ss58_address
    hotkey = alice_wallet.hotkey.ss58_address

    with pytest.raises(ValueError, match="Invalid checksum"):
        subtensor.does_hotkey_exist("fake")

    assert subtensor.does_hotkey_exist(hotkey) is False
    assert subtensor.get_hotkey_owner(hotkey) is None

    assert subtensor.is_hotkey_registered(hotkey) is False
    assert subtensor.is_hotkey_registered_any(hotkey) is False
    assert (
        subtensor.is_hotkey_registered_on_subnet(
            hotkey,
            netuid=1,
        )
        is False
    )

    subtensor.burned_register(
        alice_wallet,
        netuid=1,
    )

    assert subtensor.does_hotkey_exist(hotkey) is True
    assert subtensor.get_hotkey_owner(hotkey) == coldkey

    assert subtensor.is_hotkey_registered(hotkey) is True
    assert subtensor.is_hotkey_registered_any(hotkey) is True
    assert (
        subtensor.is_hotkey_registered_on_subnet(
            hotkey,
            netuid=1,
        )
        is True
    )


@pytest.mark.asyncio
async def test_children(subtensor, alice_wallet, bob_wallet):
    """
    Tests:
    - Get default children (empty list)
    - Update children list
    - Checking cooldown period
    - Trigger rate limit
    - Clear children list
    """

    subtensor.burned_register(
        alice_wallet,
        netuid=1,
    )
    subtensor.burned_register(
        bob_wallet,
        netuid=1,
    )

    success, children, error = subtensor.get_children(
        alice_wallet.hotkey.ss58_address,
        netuid=1,
    )

    assert error == ""
    assert success is True
    assert children == []

    success, error = set_children(
        subtensor,
        alice_wallet,
        netuid=1,
        children=[
            (
                2**64 - 1,
                bob_wallet.hotkey.ss58_address,
            ),
        ],
    )

    assert error == ""
    assert success is True

    set_children_block = subtensor.get_current_block()

    success, children, error = subtensor.get_children(
        alice_wallet.hotkey.ss58_address,
        block=set_children_block,
        netuid=1,
    )

    assert success is True
    assert children == []
    assert error == ""

    subtensor.wait_for_block(set_children_block + SET_CHILDREN_COOLDOWN_PERIOD)

    await wait_epoch(subtensor, netuid=1)

    success, children, error = subtensor.get_children(
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

    success, error = set_children(
        subtensor,
        alice_wallet,
        netuid=1,
        children=[],
    )

    assert "`TxRateLimitExceeded(Module)`" in error
    assert success is False

    subtensor.wait_for_block(set_children_block + SET_CHILDREN_RATE_LIMIT)

    success, error = set_children(
        subtensor,
        alice_wallet,
        netuid=1,
        children=[],
    )

    assert error == ""
    assert success is True

    await wait_epoch(subtensor, netuid=1)

    success, children, error = subtensor.get_children(
        alice_wallet.hotkey.ss58_address,
        netuid=1,
    )

    assert error == ""
    assert success is True
    assert children == []
