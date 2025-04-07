import pytest

import bittensor
from bittensor.core.chain_data.chain_identity import ChainIdentity
from bittensor.core.chain_data.delegate_info import DelegatedInfo, DelegateInfo
from bittensor.core.chain_data.proposal_vote_data import ProposalVoteData
from bittensor.utils.balance import Balance
from tests.e2e_tests.utils.chain_interactions import (
    propose,
    set_identity,
    sudo_set_admin_utils,
    vote,
)
from tests.helpers.helpers import CLOSE_IN_VALUE

DEFAULT_DELEGATE_TAKE = 0.179995422293431


@pytest.mark.asyncio
async def test_identity(subtensor, alice_wallet, bob_wallet):
    """
    Tests:
    - Check Delegate's default identity
    - Update Delegate's identity
    """

    identity = await subtensor.query_identity(alice_wallet.coldkeypub.ss58_address)

    assert identity is None

    identities = await subtensor.get_delegate_identities()

    assert alice_wallet.coldkey.ss58_address not in identities

    await subtensor.root_register(
        alice_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    identities = await subtensor.get_delegate_identities()

    assert alice_wallet.coldkey.ss58_address not in identities

    success, error = await set_identity(
        subtensor,
        alice_wallet,
        name="Alice",
        url="https://www.example.com",
        github_repo="https://github.com/opentensor/bittensor",
        description="Local Chain",
    )

    assert error == ""
    assert success is True

    identity = await subtensor.query_identity(alice_wallet.coldkeypub.ss58_address)

    assert identity == ChainIdentity(
        additional="",
        description="Local Chain",
        discord="",
        github="https://github.com/opentensor/bittensor",
        image="",
        name="Alice",
        url="https://www.example.com",
    )

    identities = await subtensor.get_delegate_identities()

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
async def test_change_take(local_chain, subtensor, alice_wallet, bob_wallet):
    """
    Tests:
    - Get default Delegate's take once registered in root subnet
    - Increase and decreased Delegate's take
    - Try corner cases (increase/decrease beyond allowed min/max)
    """

    with pytest.raises(bittensor.HotKeyAccountNotExists):
        await subtensor.set_delegate_take(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            0.1,
            raise_error=True,
        )

    await subtensor.root_register(
        alice_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    take = await subtensor.get_delegate_take(alice_wallet.hotkey.ss58_address)

    assert take == DEFAULT_DELEGATE_TAKE

    with pytest.raises(bittensor.NonAssociatedColdKey):
        await subtensor.set_delegate_take(
            bob_wallet,
            alice_wallet.hotkey.ss58_address,
            0.1,
            raise_error=True,
        )

    with pytest.raises(bittensor.DelegateTakeTooHigh):
        await subtensor.set_delegate_take(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            0.5,
            raise_error=True,
        )

    await subtensor.set_delegate_take(
        alice_wallet,
        alice_wallet.hotkey.ss58_address,
        0.1,
        raise_error=True,
    )

    take = await subtensor.get_delegate_take(alice_wallet.hotkey.ss58_address)

    assert take == 0.09999237048905166

    with pytest.raises(bittensor.DelegateTxRateLimitExceeded):
        await subtensor.set_delegate_take(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            0.15,
            raise_error=True,
        )

    take = await subtensor.get_delegate_take(alice_wallet.hotkey.ss58_address)

    assert take == 0.09999237048905166

    sudo_set_admin_utils(
        local_chain,
        alice_wallet,
        call_function="sudo_set_tx_delegate_take_rate_limit",
        call_params={
            "tx_rate_limit": 0,
        },
    )

    await subtensor.set_delegate_take(
        alice_wallet,
        alice_wallet.hotkey.ss58_address,
        0.15,
        raise_error=True,
    )

    take = await subtensor.get_delegate_take(alice_wallet.hotkey.ss58_address)

    assert take == 0.14999618524452582


@pytest.mark.asyncio
async def test_delegates(subtensor, alice_wallet, bob_wallet):
    """
    Tests:
    - Check default Delegates
    - Register Delegates
    - Check if Hotkey is a Delegate
    - Nominator Staking
    """

    assert await subtensor.get_delegates() == []
    assert await subtensor.get_delegated(alice_wallet.coldkey.ss58_address) == []
    assert (
        await subtensor.get_delegate_by_hotkey(alice_wallet.hotkey.ss58_address) is None
    )
    assert (
        await subtensor.get_delegate_by_hotkey(bob_wallet.hotkey.ss58_address) is None
    )

    assert await subtensor.is_hotkey_delegate(alice_wallet.hotkey.ss58_address) is False
    assert await subtensor.is_hotkey_delegate(bob_wallet.hotkey.ss58_address) is False

    await subtensor.root_register(
        alice_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    await subtensor.root_register(
        bob_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert await subtensor.is_hotkey_delegate(alice_wallet.hotkey.ss58_address) is True
    assert await subtensor.is_hotkey_delegate(bob_wallet.hotkey.ss58_address) is True

    alice_delegate = await subtensor.get_delegate_by_hotkey(
        alice_wallet.hotkey.ss58_address,
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

    bob_delegate = await subtensor.get_delegate_by_hotkey(
        bob_wallet.hotkey.ss58_address,
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

    delegates = await subtensor.get_delegates()

    assert delegates == [
        bob_delegate,
        alice_delegate,
    ]

    assert await subtensor.get_delegated(bob_wallet.coldkey.ss58_address) == []

    await subtensor.add_stake(
        bob_wallet,
        alice_wallet.hotkey.ss58_address,
        netuid=0,
        amount=Balance.from_tao(10_000),
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert await subtensor.get_delegated(bob_wallet.coldkey.ss58_address) == [
        DelegatedInfo(
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            owner_ss58=alice_wallet.coldkey.ss58_address,
            take=DEFAULT_DELEGATE_TAKE,
            validator_permits=[],
            registrations=[0],
            return_per_1000=Balance(0),
            total_daily_return=Balance(0),
            netuid=0,
            stake=Balance.from_tao(9_999.99995),
        ),
    ]


@pytest.mark.asyncio
async def test_nominator_min_required_stake(
    local_chain, subtensor, alice_wallet, bob_wallet
):
    """
    Tests:
    - Check default NominatorMinRequiredStake
    - Add Stake to Nominate
    - Update NominatorMinRequiredStake
    - Check Nominator is removed
    """

    minimum_required_stake = await subtensor.get_minimum_required_stake()

    assert minimum_required_stake == Balance(0)

    await subtensor.root_register(
        alice_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    await subtensor.root_register(
        bob_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    success = await subtensor.add_stake(
        alice_wallet,
        bob_wallet.hotkey.ss58_address,
        netuid=0,
        amount=Balance.from_tao(10_000),
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert success is True

    stake = await subtensor.get_stake(
        alice_wallet.coldkey.ss58_address,
        bob_wallet.hotkey.ss58_address,
        netuid=0,
    )

    assert stake == Balance.from_tao(9_999.99995)

    # this will trigger clear_small_nominations
    sudo_set_admin_utils(
        local_chain,
        alice_wallet,
        call_function="sudo_set_nominator_min_required_stake",
        call_params={
            "min_stake": "100000000000000",
        },
    )

    minimum_required_stake = await subtensor.get_minimum_required_stake()

    assert minimum_required_stake == Balance.from_tao(100_000)

    stake = await subtensor.get_stake(
        alice_wallet.coldkey.ss58_address,
        bob_wallet.hotkey.ss58_address,
        netuid=0,
    )

    assert stake == Balance(0)


@pytest.mark.asyncio
async def test_get_vote_data(subtensor, alice_wallet):
    """
    Tests:
    - Sends Propose
    - Checks existing Proposals
    - Votes
    - Checks Proposal is updated
    """

    await subtensor.root_register(alice_wallet)

    proposals = await subtensor.query_map(
        "Triumvirate",
        "ProposalOf",
        params=[],
    )

    assert proposals.records == []

    success, error = await propose(
        subtensor,
        alice_wallet,
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

    proposals = await subtensor.query_map(
        "Triumvirate",
        "ProposalOf",
        params=[],
    )
    proposals = {
        bytes(proposal_hash[0]): proposal.value
        for proposal_hash, proposal in proposals.records
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

    proposal = await subtensor.get_vote_data(
        proposal_hash,
    )

    assert proposal == ProposalVoteData(
        ayes=[],
        end=CLOSE_IN_VALUE(1_000_000, await subtensor.block),
        index=0,
        nays=[],
        threshold=3,
    )

    success, error = await vote(
        subtensor,
        alice_wallet,
        alice_wallet.hotkey.ss58_address,
        proposal_hash,
        index=0,
        approve=True,
    )

    assert error == ""
    assert success is True

    proposal = await subtensor.get_vote_data(
        proposal_hash,
    )

    assert proposal == ProposalVoteData(
        ayes=[
            alice_wallet.hotkey.ss58_address,
        ],
        end=CLOSE_IN_VALUE(1_000_000, await subtensor.block),
        index=0,
        nays=[],
        threshold=3,
    )
