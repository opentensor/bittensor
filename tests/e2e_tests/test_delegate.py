import pytest

from bittensor.core.chain_data.chain_identity import ChainIdentity
from bittensor.core.chain_data.delegate_info import DelegatedInfo, DelegateInfo
from bittensor.core.chain_data.proposal_vote_data import ProposalVoteData
from bittensor.core.errors import (
    DelegateTakeTooHigh,
    DelegateTxRateLimitExceeded,
    HotKeyAccountNotExists,
    NonAssociatedColdKey,
)
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging
from tests.e2e_tests.utils.chain_interactions import (
    async_propose,
    async_set_identity,
    async_sudo_set_admin_utils,
    async_vote,
    get_dynamic_balance,
    propose,
    set_identity,
    sudo_set_admin_utils,
    vote,
)
from tests.e2e_tests.utils.e2e_test_utils import (
    async_wait_to_start_call,
    wait_to_start_call,
)
from tests.helpers.helpers import CloseInValue

DEFAULT_DELEGATE_TAKE = 0.179995422293431


def test_identity(subtensor, alice_wallet, bob_wallet):
    """
    Tests:
    - Check Delegate's default identity
    - Update Delegate's identity
    """
    logging.console.info("Testing [green]test_identity[/green].")

    identity = subtensor.neurons.query_identity(alice_wallet.coldkeypub.ss58_address)
    assert identity is None

    identities = subtensor.delegates.get_delegate_identities()
    assert alice_wallet.coldkey.ss58_address not in identities

    subtensor.extrinsics.root_register(
        alice_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    identities = subtensor.delegates.get_delegate_identities()
    assert alice_wallet.coldkey.ss58_address not in identities

    success, error = set_identity(
        subtensor=subtensor,
        wallet=alice_wallet,
        name="Alice",
        url="https://www.example.com",
        github_repo="https://github.com/opentensor/bittensor",
        description="Local Chain",
    )
    assert error == ""
    assert success is True

    identity = subtensor.neurons.query_identity(alice_wallet.coldkeypub.ss58_address)
    assert identity == ChainIdentity(
        additional="",
        description="Local Chain",
        discord="",
        github="https://github.com/opentensor/bittensor",
        image="",
        name="Alice",
        url="https://www.example.com",
    )

    identities = subtensor.delegates.get_delegate_identities()
    assert alice_wallet.coldkey.ss58_address in identities

    identity = identities[alice_wallet.coldkey.ss58_address]
    assert identity == ChainIdentity(
        additional="",
        description="Local Chain",
        discord="",
        github="https://github.com/opentensor/bittensor",
        image="",
        name="Alice",
        url="https://www.example.com",
    )
    logging.console.success("Test [green]test_identity[/green] passed.")


@pytest.mark.asyncio
async def test_identity_async(async_subtensor, alice_wallet, bob_wallet):
    """
    Async tests:
    - Check Delegate's default identity
    - Update Delegate's identity
    """
    logging.console.info("Testing [green]test_identity_async[/green].")

    identity = await async_subtensor.neurons.query_identity(
        alice_wallet.coldkeypub.ss58_address
    )
    assert identity is None

    identities = await async_subtensor.delegates.get_delegate_identities()
    assert alice_wallet.coldkey.ss58_address not in identities

    await async_subtensor.extrinsics.root_register(
        alice_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    identities = await async_subtensor.delegates.get_delegate_identities()
    assert alice_wallet.coldkey.ss58_address not in identities

    success, error = await async_set_identity(
        subtensor=async_subtensor,
        wallet=alice_wallet,
        name="Alice",
        url="https://www.example.com",
        github_repo="https://github.com/opentensor/bittensor",
        description="Local Chain",
    )

    assert error == ""
    assert success is True

    identity = await async_subtensor.neurons.query_identity(
        alice_wallet.coldkeypub.ss58_address
    )

    assert identity == ChainIdentity(
        additional="",
        description="Local Chain",
        discord="",
        github="https://github.com/opentensor/bittensor",
        image="",
        name="Alice",
        url="https://www.example.com",
    )

    identities = await async_subtensor.delegates.get_delegate_identities()
    assert alice_wallet.coldkey.ss58_address in identities

    identity = identities[alice_wallet.coldkey.ss58_address]
    assert identity == ChainIdentity(
        additional="",
        description="Local Chain",
        discord="",
        github="https://github.com/opentensor/bittensor",
        image="",
        name="Alice",
        url="https://www.example.com",
    )
    logging.console.success("Test [green]test_identity_async[/green] passed.")


def test_change_take(subtensor, alice_wallet, bob_wallet):
    """
    Tests:
    - Get default Delegate's take once registered in root subnet
    - Increase and decreased Delegate's take
    - Try corner cases (increase/decrease beyond allowed min/max)
    """

    logging.console.info("Testing [green]test_change_take[/green].")
    with pytest.raises(HotKeyAccountNotExists):
        subtensor.delegates.set_delegate_take(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            0.1,
            raise_error=True,
        )

    subtensor.extrinsics.root_register(
        alice_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    take = subtensor.delegates.get_delegate_take(alice_wallet.hotkey.ss58_address)
    assert take == DEFAULT_DELEGATE_TAKE

    with pytest.raises(NonAssociatedColdKey):
        subtensor.delegates.set_delegate_take(
            bob_wallet,
            alice_wallet.hotkey.ss58_address,
            0.1,
            raise_error=True,
        )

    with pytest.raises(DelegateTakeTooHigh):
        subtensor.delegates.set_delegate_take(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            0.5,
            raise_error=True,
        )

    subtensor.delegates.set_delegate_take(
        alice_wallet,
        alice_wallet.hotkey.ss58_address,
        0.1,
        raise_error=True,
    )

    take = subtensor.delegates.get_delegate_take(alice_wallet.hotkey.ss58_address)
    assert take == 0.09999237048905166

    with pytest.raises(DelegateTxRateLimitExceeded):
        subtensor.delegates.set_delegate_take(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            0.15,
            raise_error=True,
        )

    take = subtensor.delegates.get_delegate_take(alice_wallet.hotkey.ss58_address)
    assert take == 0.09999237048905166

    sudo_set_admin_utils(
        substrate=subtensor.substrate,
        wallet=alice_wallet,
        call_function="sudo_set_tx_delegate_take_rate_limit",
        call_params={
            "tx_rate_limit": 0,
        },
    )

    subtensor.delegates.set_delegate_take(
        wallet=alice_wallet,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        take=0.15,
        raise_error=True,
    )

    take = subtensor.delegates.get_delegate_take(alice_wallet.hotkey.ss58_address)
    assert take == 0.14999618524452582

    logging.console.success("Test [green]test_change_take[/green] passed.")


@pytest.mark.asyncio
async def test_change_take_async(async_subtensor, alice_wallet, bob_wallet):
    """
    Async tests:
    - Get default Delegate's take once registered in root subnet
    - Increase and decreased Delegate's take
    - Try corner cases (increase/decrease beyond allowed min/max)
    """

    logging.console.info("Testing [green]test_change_take_async[/green].")
    with pytest.raises(HotKeyAccountNotExists):
        await async_subtensor.delegates.set_delegate_take(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            0.1,
            raise_error=True,
        )

    await async_subtensor.extrinsics.root_register(
        alice_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    take = await async_subtensor.delegates.get_delegate_take(
        alice_wallet.hotkey.ss58_address
    )
    assert take == DEFAULT_DELEGATE_TAKE

    with pytest.raises(NonAssociatedColdKey):
        await async_subtensor.delegates.set_delegate_take(
            bob_wallet,
            alice_wallet.hotkey.ss58_address,
            0.1,
            raise_error=True,
        )

    with pytest.raises(DelegateTakeTooHigh):
        await async_subtensor.delegates.set_delegate_take(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            0.5,
            raise_error=True,
        )

    await async_subtensor.delegates.set_delegate_take(
        alice_wallet,
        alice_wallet.hotkey.ss58_address,
        0.1,
        raise_error=True,
    )

    take = await async_subtensor.delegates.get_delegate_take(
        alice_wallet.hotkey.ss58_address
    )
    assert take == 0.09999237048905166

    with pytest.raises(DelegateTxRateLimitExceeded):
        await async_subtensor.delegates.set_delegate_take(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            0.15,
            raise_error=True,
        )

    take = await async_subtensor.delegates.get_delegate_take(
        alice_wallet.hotkey.ss58_address
    )
    assert take == 0.09999237048905166

    await async_sudo_set_admin_utils(
        substrate=async_subtensor.substrate,
        wallet=alice_wallet,
        call_function="sudo_set_tx_delegate_take_rate_limit",
        call_params={
            "tx_rate_limit": 0,
        },
    )

    await async_subtensor.delegates.set_delegate_take(
        wallet=alice_wallet,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        take=0.15,
        raise_error=True,
    )

    take = await async_subtensor.delegates.get_delegate_take(
        alice_wallet.hotkey.ss58_address
    )
    assert take == 0.14999618524452582

    logging.console.success("Test [green]test_change_take_async[/green] passed.")


def test_delegates(subtensor, alice_wallet, bob_wallet):
    """
    Tests:
    - Check default Delegates
    - Register Delegates
    - Check if Hotkey is a Delegate
    - Nominator Staking
    """
    logging.console.info("Testing [green]test_delegates[/green].")

    assert subtensor.delegates.get_delegates() == []
    assert subtensor.delegates.get_delegated(alice_wallet.coldkey.ss58_address) == []
    assert (
        subtensor.delegates.get_delegate_by_hotkey(alice_wallet.hotkey.ss58_address)
        is None
    )
    assert (
        subtensor.delegates.get_delegate_by_hotkey(bob_wallet.hotkey.ss58_address)
        is None
    )

    assert (
        subtensor.delegates.is_hotkey_delegate(alice_wallet.hotkey.ss58_address)
        is False
    )
    assert (
        subtensor.delegates.is_hotkey_delegate(bob_wallet.hotkey.ss58_address) is False
    )

    subtensor.extrinsics.root_register(
        alice_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    subtensor.extrinsics.root_register(
        bob_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert (
        subtensor.delegates.is_hotkey_delegate(alice_wallet.hotkey.ss58_address) is True
    )
    assert (
        subtensor.delegates.is_hotkey_delegate(bob_wallet.hotkey.ss58_address) is True
    )

    alice_delegate = subtensor.delegates.get_delegate_by_hotkey(
        alice_wallet.hotkey.ss58_address
    )

    assert alice_delegate == DelegateInfo(
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        owner_ss58=alice_wallet.coldkey.ss58_address,
        take=DEFAULT_DELEGATE_TAKE,
        validator_permits=[],
        registrations=[0],
        return_per_1000=Balance(0),
        total_daily_return=Balance(0),
        total_stake={},
        nominators={},
    )

    bob_delegate = subtensor.delegates.get_delegate_by_hotkey(
        bob_wallet.hotkey.ss58_address
    )

    assert bob_delegate == DelegateInfo(
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        owner_ss58=bob_wallet.coldkey.ss58_address,
        take=DEFAULT_DELEGATE_TAKE,
        validator_permits=[],
        registrations=[0],
        return_per_1000=Balance(0),
        total_daily_return=Balance(0),
        total_stake={},
        nominators={},
    )

    delegates = subtensor.delegates.get_delegates()

    assert delegates == [
        bob_delegate,
        alice_delegate,
    ]

    assert subtensor.delegates.get_delegated(bob_wallet.coldkey.ss58_address) == []

    alice_subnet_netuid = subtensor.subnets.get_total_subnets()  # 2
    set_tempo = 10
    # Register a subnet, netuid 2
    assert subtensor.subnets.register_subnet(alice_wallet), "Subnet wasn't created"

    # Verify subnet <netuid> created successfully
    assert subtensor.subnets.subnet_exists(alice_subnet_netuid), (
        "Subnet wasn't created successfully"
    )

    assert wait_to_start_call(subtensor, alice_wallet, alice_subnet_netuid)

    # set the same tempo for both type of nodes (fast and non-fast blocks)
    assert (
        sudo_set_admin_utils(
            substrate=subtensor.substrate,
            wallet=alice_wallet,
            call_function="sudo_set_tempo",
            call_params={"netuid": alice_subnet_netuid, "tempo": set_tempo},
        )[0]
        is True
    )

    subtensor.staking.add_stake(
        wallet=bob_wallet,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
        amount=Balance.from_tao(10_000),
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # let chain update validator_permits
    subtensor.wait_for_block(subtensor.block + set_tempo + 1)

    bob_delegated = subtensor.delegates.get_delegated(bob_wallet.coldkey.ss58_address)
    assert bob_delegated == [
        DelegatedInfo(
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            owner_ss58=alice_wallet.coldkey.ss58_address,
            take=DEFAULT_DELEGATE_TAKE,
            validator_permits=[alice_subnet_netuid],
            registrations=[0, alice_subnet_netuid],
            return_per_1000=Balance(0),
            total_daily_return=get_dynamic_balance(
                bob_delegated[0].total_daily_return.rao
            ),
            netuid=alice_subnet_netuid,
            stake=get_dynamic_balance(bob_delegated[0].stake.rao, alice_subnet_netuid),
        ),
    ]
    logging.console.success("Test [green]test_delegates[/green] passed.")


@pytest.mark.asyncio
async def test_delegates_async(async_subtensor, alice_wallet, bob_wallet):
    """
    Tests:
    - Check default Delegates
    - Register Delegates
    - Check if Hotkey is a Delegate
    - Nominator Staking
    """
    logging.console.info("Testing [green]test_delegates_async[/green].")

    assert await async_subtensor.delegates.get_delegates() == []
    assert (
        await async_subtensor.delegates.get_delegated(alice_wallet.coldkey.ss58_address)
        == []
    )
    assert (
        await async_subtensor.delegates.get_delegate_by_hotkey(
            alice_wallet.hotkey.ss58_address
        )
        is None
    )
    assert (
        await async_subtensor.delegates.get_delegate_by_hotkey(
            bob_wallet.hotkey.ss58_address
        )
        is None
    )

    assert (
        await async_subtensor.delegates.is_hotkey_delegate(
            alice_wallet.hotkey.ss58_address
        )
        is False
    )
    assert (
        await async_subtensor.delegates.is_hotkey_delegate(
            bob_wallet.hotkey.ss58_address
        )
        is False
    )

    await async_subtensor.extrinsics.root_register(
        alice_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    await async_subtensor.extrinsics.root_register(
        bob_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert (
        await async_subtensor.delegates.is_hotkey_delegate(
            alice_wallet.hotkey.ss58_address
        )
        is True
    )
    assert (
        await async_subtensor.delegates.is_hotkey_delegate(
            bob_wallet.hotkey.ss58_address
        )
        is True
    )

    alice_delegate = await async_subtensor.delegates.get_delegate_by_hotkey(
        alice_wallet.hotkey.ss58_address
    )

    assert alice_delegate == DelegateInfo(
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        owner_ss58=alice_wallet.coldkey.ss58_address,
        take=DEFAULT_DELEGATE_TAKE,
        validator_permits=[],
        registrations=[0],
        return_per_1000=Balance(0),
        total_daily_return=Balance(0),
        total_stake={},
        nominators={},
    )

    bob_delegate = await async_subtensor.delegates.get_delegate_by_hotkey(
        bob_wallet.hotkey.ss58_address
    )

    assert bob_delegate == DelegateInfo(
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        owner_ss58=bob_wallet.coldkey.ss58_address,
        take=DEFAULT_DELEGATE_TAKE,
        validator_permits=[],
        registrations=[0],
        return_per_1000=Balance(0),
        total_daily_return=Balance(0),
        total_stake={},
        nominators={},
    )

    delegates = await async_subtensor.delegates.get_delegates()

    assert delegates == [
        bob_delegate,
        alice_delegate,
    ]

    assert (
        await async_subtensor.delegates.get_delegated(bob_wallet.coldkey.ss58_address)
        == []
    )

    alice_subnet_netuid = await async_subtensor.subnets.get_total_subnets()  # 2
    set_tempo = 10
    # Register a subnet, netuid 2
    assert await async_subtensor.subnets.register_subnet(alice_wallet), (
        "Subnet wasn't created"
    )

    # Verify subnet <netuid> created successfully
    assert await async_subtensor.subnets.subnet_exists(alice_subnet_netuid), (
        "Subnet wasn't created successfully"
    )

    assert await async_wait_to_start_call(
        async_subtensor, alice_wallet, alice_subnet_netuid
    )

    # set the same tempo for both type of nodes (fast and non-fast blocks)
    assert (
        await async_sudo_set_admin_utils(
            substrate=async_subtensor.substrate,
            wallet=alice_wallet,
            call_function="sudo_set_tempo",
            call_params={"netuid": alice_subnet_netuid, "tempo": set_tempo},
        )
    )[0] is True

    await async_subtensor.staking.add_stake(
        wallet=bob_wallet,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
        amount=Balance.from_tao(10_000),
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # let chain update validator_permits
    await async_subtensor.wait_for_block(await async_subtensor.block + set_tempo + 1)

    bob_delegated = await async_subtensor.delegates.get_delegated(
        bob_wallet.coldkey.ss58_address
    )
    assert bob_delegated == [
        DelegatedInfo(
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            owner_ss58=alice_wallet.coldkey.ss58_address,
            take=DEFAULT_DELEGATE_TAKE,
            validator_permits=[alice_subnet_netuid],
            registrations=[0, alice_subnet_netuid],
            return_per_1000=Balance(0),
            total_daily_return=get_dynamic_balance(
                bob_delegated[0].total_daily_return.rao
            ),
            netuid=alice_subnet_netuid,
            stake=get_dynamic_balance(bob_delegated[0].stake.rao, alice_subnet_netuid),
        ),
    ]
    logging.console.success("Test [green]test_delegates_async[/green] passed.")


def test_nominator_min_required_stake(subtensor, alice_wallet, bob_wallet, dave_wallet):
    """
    Tests:
    - Check default NominatorMinRequiredStake
    - Add Stake to Nominate from Dave to Bob
    - Update NominatorMinRequiredStake
    - Check Nominator is removed
    """
    logging.console.info("Testing [green]test_delegates_async[/green].")

    alice_subnet_netuid = subtensor.subnets.get_total_subnets()  # 2

    # Register a subnet, netuid 2
    assert subtensor.subnets.register_subnet(alice_wallet), "Subnet wasn't created"

    # Verify subnet <netuid> created successfully
    assert subtensor.subnets.subnet_exists(alice_subnet_netuid), (
        "Subnet wasn't created successfully"
    )
    assert wait_to_start_call(subtensor, alice_wallet, alice_subnet_netuid)

    minimum_required_stake = subtensor.staking.get_minimum_required_stake()
    assert minimum_required_stake == Balance(0)

    subtensor.subnets.burned_register(
        wallet=bob_wallet,
        netuid=alice_subnet_netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    subtensor.subnets.burned_register(
        wallet=dave_wallet,
        netuid=alice_subnet_netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    success = subtensor.staking.add_stake(
        wallet=dave_wallet,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
        amount=Balance.from_tao(1000),
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert success is True

    stake = subtensor.staking.get_stake(
        coldkey_ss58=dave_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )
    assert stake > 0

    # this will trigger clear_small_nominations
    sudo_set_admin_utils(
        substrate=subtensor.substrate,
        wallet=alice_wallet,
        call_function="sudo_set_nominator_min_required_stake",
        call_params={
            "min_stake": "100000000000000",
        },
    )

    minimum_required_stake = subtensor.staking.get_minimum_required_stake()
    assert minimum_required_stake == Balance.from_tao(100_000)

    stake = subtensor.staking.get_stake(
        coldkey_ss58=dave_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )
    assert stake == Balance(0)

    logging.console.success(
        "Test [green]test_nominator_min_required_stake[/green] passed."
    )


@pytest.mark.asyncio
async def test_nominator_min_required_stake_async(
    async_subtensor, alice_wallet, bob_wallet, dave_wallet
):
    """
    Async tests:
    - Check default NominatorMinRequiredStake
    - Add Stake to Nominate from Dave to Bob
    - Update NominatorMinRequiredStake
    - Check Nominator is removed
    """
    logging.console.info(
        "Testing [green]test_nominator_min_required_stake_async[/green]."
    )

    alice_subnet_netuid = await async_subtensor.subnets.get_total_subnets()  # 2

    # Register a subnet, netuid 2
    assert await async_subtensor.subnets.register_subnet(alice_wallet), (
        "Subnet wasn't created"
    )

    # Verify subnet <netuid> created successfully
    assert await async_subtensor.subnets.subnet_exists(alice_subnet_netuid), (
        "Subnet wasn't created successfully"
    )
    assert await async_wait_to_start_call(
        async_subtensor, alice_wallet, alice_subnet_netuid
    )

    minimum_required_stake = await async_subtensor.staking.get_minimum_required_stake()
    assert minimum_required_stake == Balance(0)

    await async_subtensor.subnets.burned_register(
        wallet=bob_wallet,
        netuid=alice_subnet_netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    await async_subtensor.subnets.burned_register(
        wallet=dave_wallet,
        netuid=alice_subnet_netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    success = await async_subtensor.staking.add_stake(
        wallet=dave_wallet,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
        amount=Balance.from_tao(1000),
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert success is True

    stake = await async_subtensor.staking.get_stake(
        coldkey_ss58=dave_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )
    assert stake > 0

    # this will trigger clear_small_nominations
    await async_sudo_set_admin_utils(
        substrate=async_subtensor.substrate,
        wallet=alice_wallet,
        call_function="sudo_set_nominator_min_required_stake",
        call_params={
            "min_stake": "100000000000000",
        },
    )

    minimum_required_stake = await async_subtensor.staking.get_minimum_required_stake()
    assert minimum_required_stake == Balance.from_tao(100_000)

    stake = await async_subtensor.staking.get_stake(
        coldkey_ss58=dave_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )
    assert stake == Balance(0)

    logging.console.success(
        "Test [green]test_nominator_min_required_stake_async[/green] passed."
    )


def test_get_vote_data(subtensor, alice_wallet):
    """
    Tests:
    - Sends Propose
    - Checks existing Proposals
    - Votes
    - Checks Proposal is updated
    """
    logging.console.info("Testing [green]test_get_vote_data[/green].")

    assert subtensor.extrinsics.root_register(alice_wallet), (
        "Can not register Alice in root SN."
    )

    proposals = subtensor.queries.query_map(
        "Triumvirate",
        "ProposalOf",
        params=[],
    )

    assert proposals.records == []

    success, error = propose(
        subtensor=subtensor,
        wallet=alice_wallet,
        proposal=subtensor.substrate.compose_call(
            call_module="Triumvirate",
            call_function="set_members",
            call_params={
                "new_members": [],
                "prime": None,
                "old_count": 0,
            },
        ),
        duration=1_000_000,
    )

    assert error == ""
    assert success is True

    proposals = subtensor.queries.query_map(
        module="Triumvirate",
        name="ProposalOf",
        params=[],
    )
    proposals = {
        bytes(proposal_hash[0]): proposal.value for proposal_hash, proposal in proposals
    }

    assert list(proposals.values()) == [
        {
            "Triumvirate": (
                {
                    "set_members": {
                        "new_members": (),
                        "prime": None,
                        "old_count": 0,
                    },
                },
            ),
        },
    ]

    proposal_hash = list(proposals.keys())[0]
    proposal_hash = f"0x{proposal_hash.hex()}"

    proposal = subtensor.chain.get_vote_data(
        proposal_hash,
    )

    assert proposal == ProposalVoteData(
        ayes=[],
        end=CloseInValue(1_000_000, subtensor.block),
        index=0,
        nays=[],
        threshold=3,
    )

    success, error = vote(
        subtensor=subtensor,
        wallet=alice_wallet,
        hotkey=alice_wallet.hotkey.ss58_address,
        proposal=proposal_hash,
        index=0,
        approve=True,
    )

    assert error == ""
    assert success is True

    proposal = subtensor.chain.get_vote_data(
        proposal_hash=proposal_hash,
    )

    assert proposal == ProposalVoteData(
        ayes=[
            alice_wallet.hotkey.ss58_address,
        ],
        end=CloseInValue(1_000_000, subtensor.block),
        index=0,
        nays=[],
        threshold=3,
    )
    logging.console.success("Test [green]test_get_vote_data[/green] passed.")


@pytest.mark.asyncio
async def test_get_vote_data_async(async_subtensor, alice_wallet):
    """
    Async tests:
    - Sends Propose
    - Checks existing Proposals
    - Votes
    - Checks Proposal is updated
    """
    logging.console.info("Testing [green]test_get_vote_data_async[/green].")

    assert await async_subtensor.extrinsics.root_register(alice_wallet), (
        "Can not register Alice in root SN."
    )

    proposals = await async_subtensor.queries.query_map(
        "Triumvirate",
        "ProposalOf",
        params=[],
    )

    assert proposals.records == []

    success, error = await async_propose(
        subtensor=async_subtensor,
        wallet=alice_wallet,
        proposal=await async_subtensor.substrate.compose_call(
            call_module="Triumvirate",
            call_function="set_members",
            call_params={
                "new_members": [],
                "prime": None,
                "old_count": 0,
            },
        ),
        duration=1_000_000,
    )

    assert error == ""
    assert success is True

    proposals = await async_subtensor.queries.query_map(
        module="Triumvirate",
        name="ProposalOf",
        params=[],
    )
    proposals = {
        bytes(proposal_hash[0]): proposal.value
        async for proposal_hash, proposal in proposals
    }

    assert list(proposals.values()) == [
        {
            "Triumvirate": (
                {
                    "set_members": {
                        "new_members": (),
                        "prime": None,
                        "old_count": 0,
                    },
                },
            ),
        },
    ]

    proposal_hash = list(proposals.keys())[0]
    proposal_hash = f"0x{proposal_hash.hex()}"

    proposal = await async_subtensor.chain.get_vote_data(
        proposal_hash,
    )

    assert proposal == ProposalVoteData(
        ayes=[],
        end=CloseInValue(1_000_000, await async_subtensor.block),
        index=0,
        nays=[],
        threshold=3,
    )

    success, error = await async_vote(
        subtensor=async_subtensor,
        wallet=alice_wallet,
        hotkey=alice_wallet.hotkey.ss58_address,
        proposal=proposal_hash,
        index=0,
        approve=True,
    )

    assert error == ""
    assert success is True

    proposal = await async_subtensor.chain.get_vote_data(
        proposal_hash=proposal_hash,
    )

    assert proposal == ProposalVoteData(
        ayes=[
            alice_wallet.hotkey.ss58_address,
        ],
        end=CloseInValue(1_000_000, await async_subtensor.block),
        index=0,
        nays=[],
        threshold=3,
    )
    logging.console.success("Test [green]test_get_vote_data_async[/green] passed.")
