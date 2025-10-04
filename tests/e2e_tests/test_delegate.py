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
from tests.e2e_tests.utils import (
    async_propose,
    async_set_identity,
    async_vote,
    get_dynamic_balance,
    propose,
    set_identity,
    vote,
    TestSubnet,
    AdminUtils,
    ACTIVATE_SUBNET,
    REGISTER_SUBNET,
    REGISTER_NEURON,
    SUDO_SET_NOMINATOR_MIN_REQUIRED_STAKE,
    SUDO_SET_TX_DELEGATE_TAKE_RATE_LIMIT,
)
from tests.helpers.helpers import CloseInValue

DEFAULT_DELEGATE_TAKE = 0.179995422293431


def test_identity(subtensor, alice_wallet, bob_wallet):
    """
    Tests:
    - Check Delegate's default identity
    - Update Delegate's identity
    """
    identity = subtensor.neurons.query_identity(alice_wallet.coldkeypub.ss58_address)
    assert identity is None

    identities = subtensor.delegates.get_delegate_identities()
    assert alice_wallet.coldkey.ss58_address not in identities

    assert subtensor.extrinsics.root_register(
        wallet=alice_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    ).success

    identities = subtensor.delegates.get_delegate_identities()
    assert alice_wallet.coldkey.ss58_address not in identities

    success, message = set_identity(
        subtensor=subtensor,
        wallet=alice_wallet,
        name="Alice",
        url="https://www.example.com",
        github_repo="https://github.com/opentensor/bittensor",
        description="Local Chain",
    )
    assert success is True, message
    assert message == "Success"

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


@pytest.mark.asyncio
async def test_identity_async(async_subtensor, alice_wallet, bob_wallet):
    """
    Async tests:
    - Check Delegate's default identity
    - Update Delegate's identity
    """
    identity = await async_subtensor.neurons.query_identity(
        alice_wallet.coldkeypub.ss58_address
    )
    assert identity is None

    identities = await async_subtensor.delegates.get_delegate_identities()
    assert alice_wallet.coldkey.ss58_address not in identities

    assert (
        await async_subtensor.extrinsics.root_register(
            wallet=alice_wallet,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
    ).success

    identities = await async_subtensor.delegates.get_delegate_identities()
    assert alice_wallet.coldkey.ss58_address not in identities

    success, message = await async_set_identity(
        subtensor=async_subtensor,
        wallet=alice_wallet,
        name="Alice",
        url="https://www.example.com",
        github_repo="https://github.com/opentensor/bittensor",
        description="Local Chain",
    )

    assert success is True, message
    assert message == "Success"

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


def test_change_take(subtensor, alice_wallet, bob_wallet):
    """
    Tests:
    - Get default Delegate's take once registered in root subnet
    - Increase and decreased Delegate's take
    - Try corner cases (increase/decrease beyond allowed min/max)
    """
    with pytest.raises(HotKeyAccountNotExists):
        subtensor.delegates.set_delegate_take(
            wallet=alice_wallet,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            take=0.1,
            raise_error=True,
        )

    assert subtensor.extrinsics.root_register(
        wallet=alice_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    ).success

    take = subtensor.delegates.get_delegate_take(alice_wallet.hotkey.ss58_address)
    assert take == DEFAULT_DELEGATE_TAKE

    with pytest.raises(NonAssociatedColdKey):
        subtensor.delegates.set_delegate_take(
            wallet=bob_wallet,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            take=0.1,
            raise_error=True,
        )

    with pytest.raises(DelegateTakeTooHigh):
        subtensor.delegates.set_delegate_take(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            0.5,
            raise_error=True,
        )

    assert subtensor.delegates.set_delegate_take(
        alice_wallet,
        alice_wallet.hotkey.ss58_address,
        0.1,
        raise_error=True,
    ).success

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

    TestSubnet(subtensor).execute_one(
        SUDO_SET_TX_DELEGATE_TAKE_RATE_LIMIT(alice_wallet, AdminUtils, True, 0)
    )

    assert subtensor.delegates.set_delegate_take(
        wallet=alice_wallet,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        take=0.15,
        raise_error=True,
    ).success

    take = subtensor.delegates.get_delegate_take(alice_wallet.hotkey.ss58_address)
    assert take == 0.14999618524452582


@pytest.mark.asyncio
async def test_change_take_async(async_subtensor, alice_wallet, bob_wallet):
    """
    Async tests:
    - Get default Delegate's take once registered in root subnet
    - Increase and decreased Delegate's take
    - Try corner cases (increase/decrease beyond allowed min/max)
    """
    with pytest.raises(HotKeyAccountNotExists):
        await async_subtensor.delegates.set_delegate_take(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            0.1,
            raise_error=True,
        )

    assert (
        await async_subtensor.extrinsics.root_register(
            wallet=alice_wallet,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
    ).success

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

    assert (
        await async_subtensor.delegates.set_delegate_take(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            0.1,
            raise_error=True,
        )
    ).success

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

    await TestSubnet(async_subtensor).async_execute_one(
        SUDO_SET_TX_DELEGATE_TAKE_RATE_LIMIT(alice_wallet, AdminUtils, True, 0)
    )

    assert (
        await async_subtensor.delegates.set_delegate_take(
            wallet=alice_wallet,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            take=0.15,
            raise_error=True,
        )
    ).success

    take = await async_subtensor.delegates.get_delegate_take(
        alice_wallet.hotkey.ss58_address
    )
    assert take == 0.14999618524452582


def test_delegates(subtensor, alice_wallet, bob_wallet):
    """
    Tests:
    - Check default Delegates
    - Register Delegates
    - Check if Hotkey is a Delegate
    - Nominator Staking
    """
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

    assert subtensor.extrinsics.root_register(
        wallet=alice_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    ).success
    assert subtensor.extrinsics.root_register(
        wallet=bob_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    ).success

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
        total_stake={},
        nominators={},
    )

    delegates = subtensor.delegates.get_delegates()

    assert delegates == [
        bob_delegate,
        alice_delegate,
    ]

    assert subtensor.delegates.get_delegated(bob_wallet.coldkey.ss58_address) == []

    TEMPO_TO_SET = 10
    alice_sn = TestSubnet(subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
    ]
    alice_sn.execute_steps(steps)

    assert subtensor.staking.add_stake(
        wallet=bob_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        amount=Balance.from_tao(10_000),
    ).success

    # let chain update validator_permits
    subtensor.wait_for_block(subtensor.block + TEMPO_TO_SET + 1)

    bob_delegated = subtensor.delegates.get_delegated(bob_wallet.coldkey.ss58_address)
    assert bob_delegated == [
        DelegatedInfo(
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            owner_ss58=alice_wallet.coldkey.ss58_address,
            take=DEFAULT_DELEGATE_TAKE,
            validator_permits=[alice_sn.netuid],
            registrations=[0, alice_sn.netuid],
            return_per_1000=Balance(0),
            netuid=alice_sn.netuid,
            stake=get_dynamic_balance(bob_delegated[0].stake.rao, alice_sn.netuid),
        ),
    ]


@pytest.mark.asyncio
async def test_delegates_async(async_subtensor, alice_wallet, bob_wallet):
    """
    Tests:
    - Check default Delegates
    - Register Delegates
    - Check if Hotkey is a Delegate
    - Nominator Staking
    """
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

    assert (
        await async_subtensor.extrinsics.root_register(
            wallet=alice_wallet,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
    ).success
    assert (
        await async_subtensor.extrinsics.root_register(
            wallet=bob_wallet,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
    ).success

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

    TEMPO_TO_SET = 10
    alice_sn = TestSubnet(async_subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
    ]
    await alice_sn.async_execute_steps(steps)

    assert (
        await async_subtensor.staking.add_stake(
            wallet=bob_wallet,
            netuid=alice_sn.netuid,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            amount=Balance.from_tao(10_000),
        )
    ).success

    # let chain update validator_permits
    await async_subtensor.wait_for_block(await async_subtensor.block + TEMPO_TO_SET + 1)

    bob_delegated = await async_subtensor.delegates.get_delegated(
        bob_wallet.coldkey.ss58_address
    )
    assert bob_delegated == [
        DelegatedInfo(
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            owner_ss58=alice_wallet.coldkey.ss58_address,
            take=DEFAULT_DELEGATE_TAKE,
            validator_permits=[alice_sn.netuid],
            registrations=[0, alice_sn.netuid],
            return_per_1000=Balance(0),
            netuid=alice_sn.netuid,
            stake=get_dynamic_balance(bob_delegated[0].stake.rao, alice_sn.netuid),
        ),
    ]


def test_nominator_min_required_stake(subtensor, alice_wallet, bob_wallet, dave_wallet):
    """
    Tests:
    - Check default NominatorMinRequiredStake
    - Add Stake to Nominate from Dave to Bob
    - Update NominatorMinRequiredStake
    - Check Nominator is removed
    """
    alice_sn = TestSubnet(subtensor)
    alice_sn.execute_steps(
        [
            REGISTER_SUBNET(alice_wallet),
            ACTIVATE_SUBNET(alice_wallet),
            REGISTER_NEURON(bob_wallet),
            REGISTER_NEURON(dave_wallet),
        ]
    )

    minimum_required_stake = subtensor.staking.get_minimum_required_stake()
    assert minimum_required_stake == Balance(0)

    assert subtensor.staking.add_stake(
        wallet=dave_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=Balance.from_tao(1000),
    ).success

    stake = subtensor.staking.get_stake(
        coldkey_ss58=dave_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    assert stake > 0

    # this will trigger clear_small_nominations
    alice_sn.execute_one(
        SUDO_SET_NOMINATOR_MIN_REQUIRED_STAKE(
            alice_wallet, AdminUtils, True, 100000000000000
        )
    )

    minimum_required_stake = subtensor.staking.get_minimum_required_stake()
    assert minimum_required_stake == Balance.from_tao(100_000)

    stake = subtensor.staking.get_stake(
        coldkey_ss58=dave_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    assert stake == Balance.from_tao(0, alice_sn.netuid)


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
    alice_sn = TestSubnet(async_subtensor)
    await alice_sn.async_execute_steps(
        [
            REGISTER_SUBNET(alice_wallet),
            ACTIVATE_SUBNET(alice_wallet),
            REGISTER_NEURON(bob_wallet),
            REGISTER_NEURON(dave_wallet),
        ]
    )

    minimum_required_stake = await async_subtensor.staking.get_minimum_required_stake()
    assert minimum_required_stake == Balance(0)

    assert (
        await async_subtensor.subnets.burned_register(
            wallet=bob_wallet,
            netuid=alice_sn.netuid,
        )
    ).success

    assert (
        await async_subtensor.subnets.burned_register(
            wallet=dave_wallet,
            netuid=alice_sn.netuid,
        )
    ).success

    assert (
        await async_subtensor.staking.add_stake(
            wallet=dave_wallet,
            netuid=alice_sn.netuid,
            hotkey_ss58=bob_wallet.hotkey.ss58_address,
            amount=Balance.from_tao(1000),
        )
    ).success

    stake = await async_subtensor.staking.get_stake(
        coldkey_ss58=dave_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    assert stake > 0

    # this will trigger clear_small_nominations
    await alice_sn.async_execute_one(
        SUDO_SET_NOMINATOR_MIN_REQUIRED_STAKE(
            alice_wallet, AdminUtils, True, 100000000000000
        )
    )

    minimum_required_stake = await async_subtensor.staking.get_minimum_required_stake()
    assert minimum_required_stake == Balance.from_tao(100_000)

    stake = await async_subtensor.staking.get_stake(
        coldkey_ss58=dave_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    assert stake == Balance.from_tao(0, alice_sn.netuid)


def test_get_vote_data(subtensor, alice_wallet):
    """
    Tests:
    - Sends Propose
    - Checks existing Proposals
    - Votes
    - Checks Proposal is updated
    """
    assert subtensor.extrinsics.root_register(alice_wallet).success, (
        "Can not register Alice in root SN."
    )

    proposals = subtensor.queries.query_map(
        "Triumvirate",
        "ProposalOf",
        params=[],
    )

    assert proposals.records == []

    success, message = propose(
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

    assert success is True, message
    assert message == "Success"

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

    success, message = vote(
        subtensor=subtensor,
        wallet=alice_wallet,
        hotkey=alice_wallet.hotkey.ss58_address,
        proposal=proposal_hash,
        index=0,
        approve=True,
    )

    assert success is True, message
    assert message == "Success"

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


@pytest.mark.asyncio
async def test_get_vote_data_async(async_subtensor, alice_wallet):
    """
    Async tests:
    - Sends Propose
    - Checks existing Proposals
    - Votes
    - Checks Proposal is updated
    """
    assert (await async_subtensor.extrinsics.root_register(alice_wallet)).success, (
        "Can not register Alice in root SN."
    )

    proposals = await async_subtensor.queries.query_map(
        "Triumvirate",
        "ProposalOf",
        params=[],
    )

    assert proposals.records == []

    success, message = await async_propose(
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

    assert success is True
    assert message == "Success"

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

    success, message = await async_vote(
        subtensor=async_subtensor,
        wallet=alice_wallet,
        hotkey=alice_wallet.hotkey.ss58_address,
        proposal=proposal_hash,
        index=0,
        approve=True,
    )

    assert success is True, message
    assert message == "Success"

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
