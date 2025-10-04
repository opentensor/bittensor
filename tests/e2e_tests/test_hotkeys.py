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
from bittensor.utils.btlogging import logging
from tests.e2e_tests.utils import (
    TestSubnet,
    AdminUtils,
    NETUID,
    ACTIVATE_SUBNET,
    REGISTER_NEURON,
    REGISTER_SUBNET,
    SUDO_SET_ADMIN_FREEZE_WINDOW,
    SUDO_SET_TEMPO,
    SUDO_SET_TX_RATE_LIMIT,
    SUDO_SET_STAKE_THRESHOLD,
)

# all values are in blocks
SET_CHILDREN_RATE_LIMIT = 50
ROOT_COOLDOWN = 30
FAST_RUNTIME_TEMPO = 100
NON_FAST_RUNTIME_TEMPO = 10


def test_hotkeys(subtensor, alice_wallet, dave_wallet):
    """
    Tests:
    - Check if Hotkey exists
    - Check if Hotkey is registered
    """
    dave_sn = TestSubnet(subtensor)
    dave_sn.execute_steps(
        [
            REGISTER_SUBNET(dave_wallet),
            ACTIVATE_SUBNET(dave_wallet),
        ]
    )

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
            netuid=dave_sn.netuid,
        )
        is False
    )

    assert subtensor.subnets.burned_register(
        wallet=alice_wallet,
        netuid=dave_sn.netuid,
    ).success

    assert subtensor.wallets.does_hotkey_exist(hotkey) is True
    assert subtensor.wallets.get_hotkey_owner(hotkey) == coldkey

    assert subtensor.wallets.is_hotkey_registered(hotkey) is True
    assert subtensor.wallets.is_hotkey_registered_any(hotkey) is True
    assert (
        subtensor.wallets.is_hotkey_registered_on_subnet(
            hotkey_ss58=hotkey,
            netuid=dave_sn.netuid,
        )
        is True
    )


@pytest.mark.asyncio
async def test_hotkeys_async(async_subtensor, alice_wallet, dave_wallet):
    """
    Async tests:
    - Check if Hotkey exists
    - Check if Hotkey is registered
    """
    dave_sn = TestSubnet(async_subtensor)
    await dave_sn.async_execute_steps(
        [
            REGISTER_SUBNET(dave_wallet),
            ACTIVATE_SUBNET(dave_wallet),
        ]
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
            netuid=dave_sn.netuid,
        )
        is False
    )

    assert (
        await async_subtensor.subnets.burned_register(
            wallet=alice_wallet,
            netuid=dave_sn.netuid,
        )
    ).success

    assert await async_subtensor.wallets.does_hotkey_exist(hotkey) is True
    assert await async_subtensor.wallets.get_hotkey_owner(hotkey) == coldkey

    assert await async_subtensor.wallets.is_hotkey_registered(hotkey) is True
    assert await async_subtensor.wallets.is_hotkey_registered_any(hotkey) is True
    assert (
        await async_subtensor.wallets.is_hotkey_registered_on_subnet(
            hotkey_ss58=hotkey,
            netuid=dave_sn.netuid,
        )
        is True
    )


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
    TEMPO_TO_SET = (
        FAST_RUNTIME_TEMPO
        if subtensor.chain.is_fast_blocks()
        else NON_FAST_RUNTIME_TEMPO
    )

    # set PendingChildKeyCooldown to SET_CHILDREN_RATE_LIMIT before everything
    subtensor.extrinsics.root_set_pending_childkey_cooldown(alice_wallet, ROOT_COOLDOWN)

    dave_sn = TestSubnet(subtensor)
    steps = [
        SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
        REGISTER_SUBNET(dave_wallet),
        ACTIVATE_SUBNET(dave_wallet),
        SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, TEMPO_TO_SET),
        SUDO_SET_TX_RATE_LIMIT(alice_wallet, AdminUtils, True, 0),
    ]
    dave_sn.execute_steps(steps)

    with pytest.raises(RegistrationNotPermittedOnRootSubnet):
        subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=0,
            children=[],
            raise_error=True,
        )

    with pytest.raises(NonAssociatedColdKey):
        subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=1,
            children=[],
            raise_error=True,
        )

    with pytest.raises(SubnetNotExists):
        subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=3,
            children=[],
            raise_error=True,
        )

    dave_sn.execute_steps(
        [
            REGISTER_NEURON(alice_wallet),
            REGISTER_NEURON(bob_wallet),
        ]
    )

    success, children, error = subtensor.wallets.get_children(
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=dave_sn.netuid,
    )

    assert error == ""
    assert success is True
    assert children == []

    with pytest.raises(InvalidChild):
        subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=dave_sn.netuid,
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
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=dave_sn.netuid,
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
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=dave_sn.netuid,
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
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=dave_sn.netuid,
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
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=dave_sn.netuid,
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
    assert success is True, message
    assert message == "Success"

    # children not set yet (have to wait cool-down period)
    success, children, error = subtensor.wallets.get_children(
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=dave_sn.netuid,
    )
    assert success is True
    assert children == []
    assert error == ""

    # children are in pending state
    pending, cooldown = subtensor.wallets.get_children_pending(
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=dave_sn.netuid,
    )
    assert pending == [(1.0, bob_wallet.hotkey.ss58_address)]

    # Wait for first block of the next tempo after the cooldown's tempo
    block = subtensor.block
    extra_blocks = block // TEMPO_TO_SET * 3
    wait_to_block = (
        cooldown
        - subtensor.subnets.blocks_since_last_step(cooldown)
        + TEMPO_TO_SET
        + extra_blocks
    )
    logging.console.info(
        f"[orange]block: {block}, cooldown: {cooldown} wait_to_block: {wait_to_block}[/orange]"
    )
    subtensor.wait_for_block(wait_to_block)

    success, children, error = subtensor.wallets.get_children(
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=dave_sn.netuid,
    )
    logging.console.info(f"[orange]block get_children: {subtensor.block}")

    assert error == ""
    assert success is True
    assert children == [(1.0, bob_wallet.hotkey.ss58_address)]

    parent_ = subtensor.wallets.get_parents(
        bob_wallet.hotkey.ss58_address, dave_sn.netuid
    )

    assert parent_ == [(1.0, alice_wallet.hotkey.ss58_address)]

    # pending queue is empty
    pending, cooldown = subtensor.wallets.get_children_pending(
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=dave_sn.netuid,
    )
    assert pending == []
    logging.console.info(
        f"[orange]block get_children_pending: {subtensor.block}, cooldown: {cooldown}[/orange]"
    )

    with pytest.raises(TxRateLimitExceeded):
        set_children_block = subtensor.block
        # first passed
        assert subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=dave_sn.netuid,
            children=[],
            raise_error=True,
        ).success

        # second raise the error
        subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=dave_sn.netuid,
            children=[],
            raise_error=True,
        )

    # wait for rate limit to expire + 1 block to ensure that the rate limit is expired
    subtensor.wait_for_block(set_children_block + SET_CHILDREN_RATE_LIMIT + 1)

    response = subtensor.extrinsics.set_children(
        wallet=alice_wallet,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=dave_sn.netuid,
        children=[],
        raise_error=True,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response.success, response.message

    pending, cooldown = subtensor.wallets.get_children_pending(
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=dave_sn.netuid,
    )
    assert pending == []

    # sometimes we need to wait some amount of blocks to ensure that children are posted on chain
    # than slower the machine then longer need to wait. But no longer than one tempo.
    # Actually this is additional protection for fast runtime note test.
    block = subtensor.block
    extra_blocks = block // TEMPO_TO_SET * 3
    wait_to_block = (
        cooldown
        - subtensor.subnets.blocks_since_last_step(cooldown)
        + TEMPO_TO_SET
        + extra_blocks
    )
    logging.console.info(
        f"[orange]block: {block}, cooldown: {cooldown} wait_to_block: {wait_to_block}[/orange]"
    )
    subtensor.wait_for_block(wait_to_block)

    start_block = subtensor.block
    while not children:
        success, children, error = subtensor.wallets.get_children(
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=dave_sn.netuid,
        )
        block = subtensor.block
        if block - start_block > TEMPO_TO_SET:
            break
        logging.console.info(f"block get_children: {subtensor.block}")
        subtensor.wait_for_block()

    assert error == ""
    assert success is True
    assert children == [(1.0, bob_wallet.hotkey.ss58_address)]

    subtensor.wait_for_block(set_children_block + SET_CHILDREN_RATE_LIMIT + 1)

    dave_sn.execute_one(
        SUDO_SET_STAKE_THRESHOLD(alice_wallet, AdminUtils, True, 1_000_000_000_000)
    )

    with pytest.raises(NotEnoughStakeToSetChildkeys):
        subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=dave_sn.netuid,
            children=[
                (
                    1.0,
                    bob_wallet.hotkey.ss58_address,
                ),
            ],
            raise_error=True,
        )


@pytest.mark.asyncio
async def test_children_async(async_subtensor, alice_wallet, bob_wallet, dave_wallet):
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
    TEMPO_TO_SET = (
        FAST_RUNTIME_TEMPO
        if await async_subtensor.chain.is_fast_blocks()
        else NON_FAST_RUNTIME_TEMPO
    )

    # set PendingChildKeyCooldown to SET_CHILDREN_RATE_LIMIT before everything
    await async_subtensor.extrinsics.root_set_pending_childkey_cooldown(
        alice_wallet, ROOT_COOLDOWN
    )

    dave_sn = TestSubnet(async_subtensor)
    steps = [
        SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
        REGISTER_SUBNET(dave_wallet),
        ACTIVATE_SUBNET(dave_wallet),
        SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, TEMPO_TO_SET),
        SUDO_SET_TX_RATE_LIMIT(alice_wallet, AdminUtils, True, 0),
    ]
    await dave_sn.async_execute_steps(steps)

    with pytest.raises(RegistrationNotPermittedOnRootSubnet):
        await async_subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=0,
            children=[],
            raise_error=True,
        )

    with pytest.raises(NonAssociatedColdKey):
        await async_subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=1,
            children=[],
            raise_error=True,
        )

    with pytest.raises(SubnetNotExists):
        await async_subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=3,
            children=[],
            raise_error=True,
        )

    await dave_sn.async_execute_steps(
        [
            REGISTER_NEURON(alice_wallet),
            REGISTER_NEURON(bob_wallet),
        ]
    )

    success, children, error = await async_subtensor.wallets.get_children(
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=dave_sn.netuid,
    )
    assert error == ""
    assert success is True
    assert children == []

    with pytest.raises(InvalidChild):
        await async_subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=dave_sn.netuid,
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
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=dave_sn.netuid,
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
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=dave_sn.netuid,
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
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=dave_sn.netuid,
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
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=dave_sn.netuid,
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
    assert success is True, message
    assert message == "Success"

    # children not set yet (have to wait cool-down period)
    success, children, error = await async_subtensor.wallets.get_children(
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=dave_sn.netuid,
    )
    assert success is True
    assert children == []
    assert error == ""

    # children are in pending state
    pending, cooldown = await async_subtensor.wallets.get_children_pending(
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=dave_sn.netuid,
    )
    assert pending == [(1.0, bob_wallet.hotkey.ss58_address)]

    # Wait for first block of the next tempo after the cooldown's tempo
    block = await async_subtensor.block
    extra_blocks = block // TEMPO_TO_SET * 3
    wait_to_block = (
        cooldown
        - await async_subtensor.subnets.blocks_since_last_step(cooldown)
        + TEMPO_TO_SET
        + extra_blocks
    )
    logging.console.info(
        f"[orange]block: {block}, cooldown: {cooldown} wait_to_block: {wait_to_block}[/orange]"
    )
    await async_subtensor.wait_for_block(wait_to_block)

    success, children, error = await async_subtensor.wallets.get_children(
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=dave_sn.netuid,
    )
    logging.console.info(f"[orange]block get_children: {await async_subtensor.block}")

    assert error == ""
    assert success is True
    assert children == [(1.0, bob_wallet.hotkey.ss58_address)]

    parent_ = await async_subtensor.wallets.get_parents(
        bob_wallet.hotkey.ss58_address, dave_sn.netuid
    )

    assert parent_ == [(1.0, alice_wallet.hotkey.ss58_address)]

    # pending queue is empty
    pending, cooldown = await async_subtensor.wallets.get_children_pending(
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=dave_sn.netuid,
    )
    assert pending == []
    logging.console.info(
        f"[orange]block get_children_pending: {await async_subtensor.block}, cooldown: {cooldown}[/orange]"
    )

    with pytest.raises(TxRateLimitExceeded):
        set_children_block = await async_subtensor.block
        # first passed
        await async_subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=dave_sn.netuid,
            children=[],
            raise_error=True,
            wait_for_finalization=False,
        )
        # second raise the error
        await async_subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=dave_sn.netuid,
            children=[],
            raise_error=True,
            wait_for_finalization=False,
        )

    # wait for rate limit to expire + 1 block to ensure that the rate limit is expired
    await async_subtensor.wait_for_block(
        set_children_block + SET_CHILDREN_RATE_LIMIT + 1
    )

    response = await async_subtensor.extrinsics.set_children(
        wallet=alice_wallet,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=dave_sn.netuid,
        children=[],
        raise_error=True,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response.success, response.message

    pending, cooldown = await async_subtensor.wallets.get_children_pending(
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=dave_sn.netuid,
    )
    assert pending == []

    # sometimes we need to wait some amount of blocks to ensure that children are posted on chain
    # than slower the machine then longer need to wait. But no longer than one tempo.
    # Actually this is additional protection for fast runtime note test.
    block = await async_subtensor.block
    extra_blocks = block // TEMPO_TO_SET * 3
    wait_to_block = (
        cooldown
        - await async_subtensor.subnets.blocks_since_last_step(cooldown)
        + TEMPO_TO_SET
        + extra_blocks
    )
    logging.console.info(
        f"[orange]block: {block}, cooldown: {cooldown} wait_to_block: {wait_to_block}[/orange]"
    )
    await async_subtensor.wait_for_block(wait_to_block)

    start_block = await async_subtensor.block
    while not children:
        success, children, error = await async_subtensor.wallets.get_children(
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=dave_sn.netuid,
        )
        block = await async_subtensor.block
        if block - start_block > TEMPO_TO_SET:
            break
        logging.console.info(f"block get_children: {block}")
        await async_subtensor.wait_for_block()

    assert error == ""
    assert success is True
    assert children == [(1.0, bob_wallet.hotkey.ss58_address)]

    await async_subtensor.wait_for_block(
        set_children_block + SET_CHILDREN_RATE_LIMIT + 1
    )

    await dave_sn.async_execute_one(
        SUDO_SET_STAKE_THRESHOLD(alice_wallet, AdminUtils, True, 1_000_000_000_000)
    )

    with pytest.raises(NotEnoughStakeToSetChildkeys):
        await async_subtensor.extrinsics.set_children(
            wallet=alice_wallet,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=dave_sn.netuid,
            children=[
                (
                    1.0,
                    bob_wallet.hotkey.ss58_address,
                ),
            ],
            raise_error=True,
        )
