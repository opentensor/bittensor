import pytest

from bittensor.core.errors import (
    NotEnoughStakeToSetChildkeys,
    RegistrationNotPermittedOnRootSubnet,
    SubnetNotExists,
    InvalidChild,
    TooManyChildren,
    ProportionOverflow,
    DuplicateChild,
    TxRateLimitExceeded,
    NonAssociatedColdKey,
)
from bittensor.core.extrinsics.sudo import (
    sudo_set_admin_freez_window_extrinsic,
)
from bittensor.utils.btlogging import logging
from tests.e2e_tests.utils.chain_interactions import sudo_set_admin_utils
from tests.e2e_tests.utils.e2e_test_utils import wait_to_start_call

SET_CHILDREN_RATE_LIMIT = 15
ROOT_COOLDOWN = 15  # blocks


def test_hotkeys(subtensor, alice_wallet, dave_wallet):
    """
    Tests:
    - Check if Hotkey exists
    - Check if Hotkey is registered
    """
    dave_subnet_netuid = subtensor.get_total_subnets()  # 2
    assert subtensor.register_subnet(dave_wallet, True, True)
    assert subtensor.subnet_exists(dave_subnet_netuid), (
        f"Subnet #{dave_subnet_netuid} does not exist."
    )

    assert wait_to_start_call(subtensor, dave_wallet, dave_subnet_netuid)

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
            netuid=dave_subnet_netuid,
        )
        is False
    )

    subtensor.burned_register(
        alice_wallet,
        netuid=dave_subnet_netuid,
    )

    assert subtensor.does_hotkey_exist(hotkey) is True
    assert subtensor.get_hotkey_owner(hotkey) == coldkey

    assert subtensor.is_hotkey_registered(hotkey) is True
    assert subtensor.is_hotkey_registered_any(hotkey) is True
    assert (
        subtensor.is_hotkey_registered_on_subnet(
            hotkey,
            netuid=dave_subnet_netuid,
        )
        is True
    )
    logging.console.success("✅ Test [green]test_hotkeys[/green] passed")


@pytest.mark.asyncio
async def test_children(local_chain, subtensor, alice_wallet, bob_wallet, dave_wallet):
    """
    Tests:
    - Get default children (empty list)
    - Call `root_set_pending_childkey_cooldown` extrinsic.
    - Update children list
    - Checking pending children
    - Checking cooldown period
    - Trigger rate limit
    - Clear children list
    """

    # turn off admin freeze window limit for testing
    assert sudo_set_admin_freez_window_extrinsic(subtensor, alice_wallet, 0)

    dave_subnet_netuid = subtensor.get_total_subnets()  # 2
    set_tempo = 10  # affect to non-fast-blocks mode

    # Set cooldown
    success, message = subtensor.extrinsics.root_set_pending_childkey_cooldown(
        wallet=alice_wallet, cooldown=ROOT_COOLDOWN
    )
    assert success, f"Call `root_set_pending_childkey_cooldown` failed: {message}"
    assert (
        message
        == "Success with `root_set_pending_childkey_cooldown_extrinsic` response."
    )

    assert subtensor.register_subnet(dave_wallet, True, True)
    assert subtensor.subnet_exists(dave_subnet_netuid), (
        f"Subnet #{dave_subnet_netuid} does not exist."
    )

    assert wait_to_start_call(subtensor, dave_wallet, dave_subnet_netuid)

    # set the same tempo for both type of nodes (to avoid tests timeout)
    if not subtensor.is_fast_blocks():
        assert (
            sudo_set_admin_utils(
                local_chain,
                alice_wallet,
                call_function="sudo_set_tempo",
                call_params={"netuid": dave_subnet_netuid, "tempo": set_tempo},
            )[0]
            is True
        )

        assert (
            sudo_set_admin_utils(
                local_chain,
                alice_wallet,
                call_function="sudo_set_tx_rate_limit",
                call_params={"tx_rate_limit": 0},
            )[0]
            is True
        )

    with pytest.raises(RegistrationNotPermittedOnRootSubnet):
        subtensor.set_children(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            netuid=0,
            children=[],
            raise_error=True,
        )

    with pytest.raises(NonAssociatedColdKey):
        subtensor.set_children(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            netuid=1,
            children=[],
            raise_error=True,
        )

    with pytest.raises(SubnetNotExists):
        subtensor.set_children(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            netuid=3,
            children=[],
            raise_error=True,
        )

    subtensor.burned_register(
        alice_wallet,
        netuid=dave_subnet_netuid,
    )
    logging.console.success(f"Alice registered on subnet {dave_subnet_netuid}")
    subtensor.burned_register(
        bob_wallet,
        netuid=dave_subnet_netuid,
    )
    logging.console.success(f"Bob registered on subnet {dave_subnet_netuid}")

    success, children, error = subtensor.get_children(
        alice_wallet.hotkey.ss58_address,
        netuid=dave_subnet_netuid,
    )

    assert error == ""
    assert success is True
    assert children == []

    with pytest.raises(InvalidChild):
        subtensor.set_children(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            netuid=dave_subnet_netuid,
            children=[
                (
                    1.0,
                    alice_wallet.hotkey.ss58_address,
                ),
            ],
            raise_error=True,
        )

    with pytest.raises(TooManyChildren):
        subtensor.set_children(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            netuid=dave_subnet_netuid,
            children=[
                (
                    0.1,
                    bob_wallet.hotkey.ss58_address,
                )
                for _ in range(10)
            ],
            raise_error=True,
        )

    with pytest.raises(ProportionOverflow):
        subtensor.set_children(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            netuid=dave_subnet_netuid,
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

    with pytest.raises(DuplicateChild):
        subtensor.set_children(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            netuid=dave_subnet_netuid,
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

    success, error = subtensor.set_children(
        wallet=alice_wallet,
        hotkey=alice_wallet.hotkey.ss58_address,
        netuid=dave_subnet_netuid,
        children=[
            (
                1.0,
                bob_wallet.hotkey.ss58_address,
            ),
        ],
        raise_error=True,
    )

    assert error == "Success with `set_children_extrinsic` response."
    assert success is True

    set_children_block = subtensor.get_current_block()

    # children not set yet (have to wait cool-down period)
    success, children, error = subtensor.get_children(
        hotkey=alice_wallet.hotkey.ss58_address,
        block=set_children_block,
        netuid=dave_subnet_netuid,
    )

    assert success is True
    assert children == []
    assert error == ""

    # children are in pending state
    pending, cooldown = subtensor.get_children_pending(
        hotkey=alice_wallet.hotkey.ss58_address,
        netuid=dave_subnet_netuid,
    )

    assert pending == [(1.0, bob_wallet.hotkey.ss58_address)]

    # we use `*2` to ensure that the chain has time to process
    subtensor.wait_for_block(cooldown + SET_CHILDREN_RATE_LIMIT * 2)

    success, children, error = subtensor.get_children(
        hotkey=alice_wallet.hotkey.ss58_address,
        netuid=dave_subnet_netuid,
    )

    assert error == ""
    assert success is True
    assert children == [(1.0, bob_wallet.hotkey.ss58_address)]

    parent_ = subtensor.get_parents(bob_wallet.hotkey.ss58_address, dave_subnet_netuid)

    assert parent_ == [(1.0, alice_wallet.hotkey.ss58_address)]

    # pending queue is empty
    pending, cooldown = subtensor.get_children_pending(
        hotkey=alice_wallet.hotkey.ss58_address,
        netuid=dave_subnet_netuid,
    )
    assert pending == []

    with pytest.raises(TxRateLimitExceeded):
        set_children_block = subtensor.get_current_block()
        subtensor.set_children(
            wallet=alice_wallet,
            hotkey=alice_wallet.hotkey.ss58_address,
            netuid=dave_subnet_netuid,
            children=[],
            raise_error=True,
        )
        subtensor.set_children(
            wallet=alice_wallet,
            hotkey=alice_wallet.hotkey.ss58_address,
            netuid=dave_subnet_netuid,
            children=[],
            raise_error=True,
        )

    # wait for rate limit to expire + 1 block to ensure that the rate limit is expired
    subtensor.wait_for_block(set_children_block + SET_CHILDREN_RATE_LIMIT + 1)

    subtensor.set_children(
        wallet=alice_wallet,
        hotkey=alice_wallet.hotkey.ss58_address,
        netuid=dave_subnet_netuid,
        children=[],
        raise_error=True,
    )
    set_children_block = subtensor.get_current_block()

    pending, cooldown = subtensor.get_children_pending(
        hotkey=alice_wallet.hotkey.ss58_address,
        netuid=dave_subnet_netuid,
    )

    assert pending == []

    subtensor.wait_for_block(cooldown + 1)

    success, children, error = subtensor.get_children(
        hotkey=alice_wallet.hotkey.ss58_address,
        netuid=dave_subnet_netuid,
    )

    assert error == ""
    assert success is True
    assert children == [(1.0, bob_wallet.hotkey.ss58_address)]

    subtensor.wait_for_block(set_children_block + SET_CHILDREN_RATE_LIMIT)

    sudo_set_admin_utils(
        local_chain,
        alice_wallet,
        call_function="sudo_set_stake_threshold",
        call_params={
            "min_stake": 1_000_000_000_000,
        },
    )

    with pytest.raises(NotEnoughStakeToSetChildkeys):
        subtensor.set_children(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            netuid=dave_subnet_netuid,
            children=[
                (
                    1.0,
                    bob_wallet.hotkey.ss58_address,
                ),
            ],
            raise_error=True,
        )

    logging.console.success(f"✅ Test [green]test_children[/green] passed")
