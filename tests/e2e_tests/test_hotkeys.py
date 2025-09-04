import pytest

from bittensor.core.errors import (
    NotEnoughStakeToSetChildkeys,
    RegistrationNotPermittedOnRootSubnet,
    SubNetworkDoesNotExist,
    InvalidChild,
    TooManyChildren,
    ProportionOverflow,
    DuplicateChild,
    TxRateLimitExceeded,
    NonAssociatedColdKey,
)
from bittensor.utils.btlogging import logging
from tests.e2e_tests.utils.chain_interactions import (
    async_sudo_set_admin_utils,
    sudo_set_admin_utils,
)
from tests.e2e_tests.utils.e2e_test_utils import (
    async_wait_to_start_call,
    wait_to_start_call,
)

SET_CHILDREN_RATE_LIMIT = 30
ROOT_COOLDOWN = 50  # blocks


def test_hotkeys(subtensor, alice_wallet, dave_wallet):
    """
    Tests:
    - Check if Hotkey exists
    - Check if Hotkey is registered
    """
    logging.console.info("Testing [green]test_hotkeys[/green].")

    dave_subnet_netuid = subtensor.subnets.get_total_subnets()  # 2
    assert subtensor.subnets.register_subnet(dave_wallet)
    assert subtensor.subnets.subnet_exists(dave_subnet_netuid), (
        f"Subnet #{dave_subnet_netuid} does not exist."
    )

    assert wait_to_start_call(subtensor, dave_wallet, dave_subnet_netuid)

    coldkey = alice_wallet.coldkeypub.ss58_address
    hotkey = alice_wallet.hotkey.ss58_address

    with pytest.raises(ValueError, match="Invalid checksum"):
        subtensor.wallets.does_hotkey_exist("fake")

    assert subtensor.wallets.does_hotkey_exist(hotkey) is False
    assert subtensor.wallets.get_hotkey_owner(hotkey) is None

    assert subtensor.wallets.is_hotkey_registered(hotkey) is False
    assert subtensor.wallets.is_hotkey_registered_any(hotkey) is False
    assert (
        subtensor.wallets.is_hotkey_registered_on_subnet(
            hotkey_ss58=hotkey,
            netuid=dave_subnet_netuid,
        )
        is False
    )

    subtensor.subnets.burned_register(
        alice_wallet,
        netuid=dave_subnet_netuid,
    )

    assert subtensor.wallets.does_hotkey_exist(hotkey) is True
    assert subtensor.wallets.get_hotkey_owner(hotkey) == coldkey

    assert subtensor.wallets.is_hotkey_registered(hotkey) is True
    assert subtensor.wallets.is_hotkey_registered_any(hotkey) is True
    assert (
        subtensor.wallets.is_hotkey_registered_on_subnet(
            hotkey_ss58=hotkey,
            netuid=dave_subnet_netuid,
        )
        is True
    )
    logging.console.success("✅ Test [green]test_hotkeys[/green] passed")


@pytest.mark.asyncio
async def test_hotkeys_async(async_subtensor, alice_wallet, dave_wallet):
    """
    Async tests:
    - Check if Hotkey exists
    - Check if Hotkey is registered
    """
    logging.console.info("Testing [green]test_hotkeys_async[/green].")

    dave_subnet_netuid = await async_subtensor.subnets.get_total_subnets()  # 2
    assert await async_subtensor.subnets.register_subnet(dave_wallet)
    assert await async_subtensor.subnets.subnet_exists(dave_subnet_netuid), (
        f"Subnet #{dave_subnet_netuid} does not exist."
    )

    assert await async_wait_to_start_call(
        async_subtensor, dave_wallet, dave_subnet_netuid
    )

    coldkey = alice_wallet.coldkeypub.ss58_address
    hotkey = alice_wallet.hotkey.ss58_address

    with pytest.raises(ValueError, match="Invalid checksum"):
        await async_subtensor.wallets.does_hotkey_exist("fake")

    assert await async_subtensor.wallets.does_hotkey_exist(hotkey) is False
    assert await async_subtensor.wallets.get_hotkey_owner(hotkey) is None

    assert await async_subtensor.wallets.is_hotkey_registered(hotkey) is False
    assert await async_subtensor.wallets.is_hotkey_registered_any(hotkey) is False
    assert (
        await async_subtensor.wallets.is_hotkey_registered_on_subnet(
            hotkey_ss58=hotkey,
            netuid=dave_subnet_netuid,
        )
        is False
    )

    assert await async_subtensor.subnets.burned_register(
        alice_wallet,
        netuid=dave_subnet_netuid,
    )

    assert await async_subtensor.wallets.does_hotkey_exist(hotkey) is True
    assert await async_subtensor.wallets.get_hotkey_owner(hotkey) == coldkey

    assert await async_subtensor.wallets.is_hotkey_registered(hotkey) is True
    assert await async_subtensor.wallets.is_hotkey_registered_any(hotkey) is True
    assert (
        await async_subtensor.wallets.is_hotkey_registered_on_subnet(
            hotkey_ss58=hotkey,
            netuid=dave_subnet_netuid,
        )
        is True
    )
    logging.console.success("✅ Test [green]test_hotkeys[/green] passed")


def test_children(subtensor, alice_wallet, bob_wallet, dave_wallet):
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

    logging.console.info("Testing [green]test_children[/green].")

    dave_subnet_netuid = subtensor.subnets.get_total_subnets()  # 2
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

    assert subtensor.subnets.register_subnet(dave_wallet)
    assert subtensor.subnets.subnet_exists(dave_subnet_netuid), (
        f"Subnet #{dave_subnet_netuid} does not exist."
    )

    assert wait_to_start_call(subtensor, dave_wallet, dave_subnet_netuid)

    # set the same tempo for both type of nodes (to avoid tests timeout)
    if not subtensor.chain.is_fast_blocks():
        assert (
            sudo_set_admin_utils(
                substrate=subtensor.substrate,
                wallet=alice_wallet,
                call_function="sudo_set_tempo",
                call_params={"netuid": dave_subnet_netuid, "tempo": set_tempo},
            )[0]
            is True
        )

        assert (
            sudo_set_admin_utils(
                substrate=subtensor.substrate,
                wallet=alice_wallet,
                call_function="sudo_set_tx_rate_limit",
                call_params={"tx_rate_limit": 0},
            )[0]
            is True
        )

    with pytest.raises(RegistrationNotPermittedOnRootSubnet):
        subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey=alice_wallet.hotkey.ss58_address,
            netuid=0,
            children=[],
            raise_error=True,
        )

    with pytest.raises(NonAssociatedColdKey):
        subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey=alice_wallet.hotkey.ss58_address,
            netuid=1,
            children=[],
            raise_error=True,
        )

    with pytest.raises(SubNetworkDoesNotExist):
        subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey=alice_wallet.hotkey.ss58_address,
            netuid=3,
            children=[],
            raise_error=True,
        )

    assert subtensor.subnets.burned_register(
        wallet=alice_wallet,
        netuid=dave_subnet_netuid,
    )
    logging.console.success(f"Alice registered on subnet {dave_subnet_netuid}")

    assert subtensor.subnets.burned_register(
        wallet=bob_wallet,
        netuid=dave_subnet_netuid,
    )
    logging.console.success(f"Bob registered on subnet {dave_subnet_netuid}")

    success, children, error = subtensor.wallets.get_children(
        hotkey=alice_wallet.hotkey.ss58_address,
        netuid=dave_subnet_netuid,
    )

    assert error == ""
    assert success is True
    assert children == []

    with pytest.raises(InvalidChild):
        subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey=alice_wallet.hotkey.ss58_address,
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
        subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey=alice_wallet.hotkey.ss58_address,
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
        subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey=alice_wallet.hotkey.ss58_address,
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
        subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey=alice_wallet.hotkey.ss58_address,
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

    success, message = subtensor.extrinsics.set_children(
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
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert message == "Success with `set_children_extrinsic` response."
    assert success is True

    set_children_block = subtensor.block

    # children not set yet (have to wait cool-down period)
    success, children, error = subtensor.wallets.get_children(
        hotkey=alice_wallet.hotkey.ss58_address,
        block=set_children_block,
        netuid=dave_subnet_netuid,
    )

    assert success is True
    assert children == []
    assert error == ""

    # children are in pending state
    pending, cooldown = subtensor.wallets.get_children_pending(
        hotkey=alice_wallet.hotkey.ss58_address,
        netuid=dave_subnet_netuid,
    )

    logging.console.info(
        f"[orange]block: {subtensor.block}, cooldown: {cooldown}[/orange]"
    )

    assert pending == [(1.0, bob_wallet.hotkey.ss58_address)]

    # we use `*2` to ensure that the chain has time to process
    subtensor.wait_for_block(cooldown + SET_CHILDREN_RATE_LIMIT * 2)

    success, children, error = subtensor.wallets.get_children(
        hotkey=alice_wallet.hotkey.ss58_address,
        netuid=dave_subnet_netuid,
    )

    assert error == ""
    assert success is True
    assert children == [(1.0, bob_wallet.hotkey.ss58_address)]

    parent_ = subtensor.wallets.get_parents(
        bob_wallet.hotkey.ss58_address, dave_subnet_netuid
    )

    assert parent_ == [(1.0, alice_wallet.hotkey.ss58_address)]

    # pending queue is empty
    pending, cooldown = subtensor.wallets.get_children_pending(
        hotkey=alice_wallet.hotkey.ss58_address,
        netuid=dave_subnet_netuid,
    )
    assert pending == []
    logging.console.info(
        f"[orange]block: {subtensor.block}, cooldown: {cooldown}[/orange]"
    )

    with pytest.raises(TxRateLimitExceeded):
        set_children_block = subtensor.block
        subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey=alice_wallet.hotkey.ss58_address,
            netuid=dave_subnet_netuid,
            children=[],
            raise_error=True,
        )
        subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey=alice_wallet.hotkey.ss58_address,
            netuid=dave_subnet_netuid,
            children=[],
            raise_error=True,
        )

    subtensor.wait_for_block(set_children_block + SET_CHILDREN_RATE_LIMIT)

    success, message = subtensor.extrinsics.set_children(
        wallet=alice_wallet,
        hotkey=alice_wallet.hotkey.ss58_address,
        netuid=dave_subnet_netuid,
        children=[],
        raise_error=True,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert success, message

    set_children_block = subtensor.block

    pending, cooldown = subtensor.wallets.get_children_pending(
        hotkey=alice_wallet.hotkey.ss58_address,
        netuid=dave_subnet_netuid,
    )

    assert pending == []
    logging.console.info(
        f"[orange]block: {subtensor.block}, cooldown: {cooldown}[/orange]"
    )

    subtensor.wait_for_block(cooldown)

    success, children, error = subtensor.wallets.get_children(
        hotkey=alice_wallet.hotkey.ss58_address,
        netuid=dave_subnet_netuid,
    )

    assert error == ""
    assert success is True
    assert children == [(1.0, bob_wallet.hotkey.ss58_address)]

    subtensor.wait_for_block(set_children_block + SET_CHILDREN_RATE_LIMIT)

    sudo_set_admin_utils(
        substrate=subtensor.substrate,
        wallet=alice_wallet,
        call_function="sudo_set_stake_threshold",
        call_params={
            "min_stake": 1_000_000_000_000,
        },
    )

    with pytest.raises(NotEnoughStakeToSetChildkeys):
        subtensor.extrinsics.set_children(
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

    logging.console.success(f"✅ Test [green]test_children[/green] passed")


@pytest.mark.asyncio
async def test_children_async(async_subtensor, alice_wallet, bob_wallet, dave_wallet):
    """
    Async tests:
    - Get default children (empty list)
    - Call `root_set_pending_childkey_cooldown` extrinsic.
    - Update children list
    - Checking pending children
    - Checking cooldown period
    - Trigger rate limit
    - Clear children list
    """

    logging.console.info("Testing [green]test_children_async[/green].")

    dave_subnet_netuid = await async_subtensor.subnets.get_total_subnets()  # 2
    set_tempo = 10  # affect to non-fast-blocks mode

    # Set cooldown
    (
        success,
        message,
    ) = await async_subtensor.extrinsics.root_set_pending_childkey_cooldown(
        wallet=alice_wallet, cooldown=ROOT_COOLDOWN
    )
    assert success, f"Call `root_set_pending_childkey_cooldown` failed: {message}"
    assert (
        message
        == "Success with `root_set_pending_childkey_cooldown_extrinsic` response."
    )

    assert await async_subtensor.subnets.register_subnet(dave_wallet)
    assert await async_subtensor.subnets.subnet_exists(dave_subnet_netuid), (
        f"Subnet #{dave_subnet_netuid} does not exist."
    )

    assert (
        await async_wait_to_start_call(async_subtensor, dave_wallet, dave_subnet_netuid)
        is True
    )

    # set the same tempo for both type of nodes (to avoid tests timeout)
    if not await async_subtensor.chain.is_fast_blocks():
        assert (
            await async_sudo_set_admin_utils(
                substrate=async_subtensor.substrate,
                wallet=alice_wallet,
                call_function="sudo_set_tempo",
                call_params={"netuid": dave_subnet_netuid, "tempo": set_tempo},
            )
        )[0] is True

        assert (
            await async_sudo_set_admin_utils(
                substrate=async_subtensor.substrate,
                wallet=alice_wallet,
                call_function="sudo_set_tx_rate_limit",
                call_params={"tx_rate_limit": 0},
            )
        )[0] is True
        assert await async_subtensor.subnets.tempo(dave_subnet_netuid) == set_tempo
        assert await async_subtensor.chain.tx_rate_limit(dave_subnet_netuid) == 0

    with pytest.raises(RegistrationNotPermittedOnRootSubnet):
        await async_subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey=alice_wallet.hotkey.ss58_address,
            netuid=0,
            children=[],
            raise_error=True,
        )

    with pytest.raises(NonAssociatedColdKey):
        await async_subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey=alice_wallet.hotkey.ss58_address,
            netuid=1,
            children=[],
            raise_error=True,
        )

    with pytest.raises(SubNetworkDoesNotExist):
        await async_subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey=alice_wallet.hotkey.ss58_address,
            netuid=3,
            children=[],
            raise_error=True,
        )

    assert await async_subtensor.subnets.burned_register(
        wallet=alice_wallet,
        netuid=dave_subnet_netuid,
    )
    logging.console.success(f"Alice registered on subnet {dave_subnet_netuid}")

    assert await async_subtensor.subnets.burned_register(
        wallet=bob_wallet,
        netuid=dave_subnet_netuid,
    )
    logging.console.success(f"Bob registered on subnet {dave_subnet_netuid}")

    success, children, error = await async_subtensor.wallets.get_children(
        hotkey=alice_wallet.hotkey.ss58_address,
        netuid=dave_subnet_netuid,
    )

    assert error == ""
    assert success is True
    assert children == []

    with pytest.raises(InvalidChild):
        await async_subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey=alice_wallet.hotkey.ss58_address,
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
        await async_subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey=alice_wallet.hotkey.ss58_address,
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
        await async_subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey=alice_wallet.hotkey.ss58_address,
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
        await async_subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey=alice_wallet.hotkey.ss58_address,
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

    success, message = await async_subtensor.extrinsics.set_children(
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
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert message == "Success with `set_children_extrinsic` response."
    assert success is True
    logging.console.info(f"[orange]success: {success}, message: {message}[/orange]")

    set_children_block = await async_subtensor.chain.get_current_block()

    # children not set yet (have to wait cool-down period)
    success, children, error = await async_subtensor.wallets.get_children(
        hotkey=alice_wallet.hotkey.ss58_address,
        block=set_children_block,
        netuid=dave_subnet_netuid,
    )

    assert success is True
    assert children == []
    assert error == ""

    # children are in pending state
    pending, cooldown = await async_subtensor.wallets.get_children_pending(
        hotkey=alice_wallet.hotkey.ss58_address,
        netuid=dave_subnet_netuid,
    )

    logging.console.info(
        f"[orange]block: {await async_subtensor.block}, cooldown: {cooldown}[/orange]"
    )

    assert pending == [(1.0, bob_wallet.hotkey.ss58_address)]

    # we use `*2` to ensure that the chain has time to process
    await async_subtensor.wait_for_block(cooldown + SET_CHILDREN_RATE_LIMIT * 2)

    success, children, error = await async_subtensor.wallets.get_children(
        hotkey=alice_wallet.hotkey.ss58_address,
        netuid=dave_subnet_netuid,
    )

    assert error == ""
    assert success is True
    assert children == [(1.0, bob_wallet.hotkey.ss58_address)]

    parent_ = await async_subtensor.wallets.get_parents(
        bob_wallet.hotkey.ss58_address, dave_subnet_netuid
    )

    assert parent_ == [(1.0, alice_wallet.hotkey.ss58_address)]

    # pending queue is empty
    pending, cooldown = await async_subtensor.wallets.get_children_pending(
        hotkey=alice_wallet.hotkey.ss58_address,
        netuid=dave_subnet_netuid,
    )
    assert pending == []
    logging.console.info(
        f"[orange]block: {await async_subtensor.block}, cooldown: {cooldown}[/orange]"
    )

    with pytest.raises(TxRateLimitExceeded):
        set_children_block = await async_subtensor.chain.get_current_block()
        await async_subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey=alice_wallet.hotkey.ss58_address,
            netuid=dave_subnet_netuid,
            children=[],
            raise_error=True,
        )
        await async_subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey=alice_wallet.hotkey.ss58_address,
            netuid=dave_subnet_netuid,
            children=[],
            raise_error=True,
        )

    await async_subtensor.wait_for_block(set_children_block + SET_CHILDREN_RATE_LIMIT)

    success, message = await async_subtensor.extrinsics.set_children(
        wallet=alice_wallet,
        hotkey=alice_wallet.hotkey.ss58_address,
        netuid=dave_subnet_netuid,
        children=[],
        raise_error=True,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert message == "Success with `set_children_extrinsic` response."
    assert success is True
    logging.console.info(f"[orange]success: {success}, message: {message}[/orange]")

    set_children_block = await async_subtensor.chain.get_current_block()

    pending, cooldown = await async_subtensor.wallets.get_children_pending(
        hotkey=alice_wallet.hotkey.ss58_address,
        netuid=dave_subnet_netuid,
    )

    assert pending == []
    logging.console.info(
        f"[orange]block: {await async_subtensor.block}, cooldown: {cooldown}[/orange]"
    )

    await async_subtensor.wait_for_block(cooldown)

    success, children, error = await async_subtensor.wallets.get_children(
        hotkey=alice_wallet.hotkey.ss58_address,
        netuid=dave_subnet_netuid,
    )

    assert error == ""
    assert success is True
    assert children == [(1.0, bob_wallet.hotkey.ss58_address)]

    await async_subtensor.wait_for_block(set_children_block + SET_CHILDREN_RATE_LIMIT)

    await async_sudo_set_admin_utils(
        substrate=async_subtensor.substrate,
        wallet=alice_wallet,
        call_function="sudo_set_stake_threshold",
        call_params={
            "min_stake": 1_000_000_000_000,
        },
    )

    with pytest.raises(NotEnoughStakeToSetChildkeys):
        await async_subtensor.extrinsics.set_children(
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

    logging.console.success(f"✅ Test [green]test_children_async[/green] passed")
