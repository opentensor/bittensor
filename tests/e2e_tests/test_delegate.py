import pytest

import bittensor
from bittensor.core.chain_data.chain_identity import ChainIdentity
from bittensor.core.chain_data.delegate_info import DelegatedInfo, DelegateInfo
from bittensor.core.chain_data.proposal_vote_data import ProposalVoteData
from bittensor.utils.balance import Balance
from tests.e2e_tests.utils.chain_interactions import (
    get_dynamic_balance,
    propose,
    set_identity,
    sudo_set_admin_utils,
    vote,
)
from tests.helpers.helpers import CLOSE_IN_VALUE

DEFAULT_DELEGATE_TAKE = 0.179995422293431


def test_identity(subtensor, alice_wallet, bob_wallet):
    """
    Tests:
    - Check Delegate's default identity
    - Update Delegate's identity
    """

    identity = subtensor.query_identity(alice_wallet.coldkeypub.ss58_address)

    assert identity is None

    identities = subtensor.get_delegate_identities()

    assert alice_wallet.coldkey.ss58_address not in identities

    subtensor.root_register(
        alice_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    identities = subtensor.get_delegate_identities()

    assert alice_wallet.coldkey.ss58_address not in identities

    success, error = set_identity(
        subtensor,
        alice_wallet,
        name="Alice",
        url="https://www.example.com",
        github_repo="https://github.com/opentensor/bittensor",
        description="Local Chain",
    )

    assert error == ""
    assert success is True

    identity = subtensor.query_identity(alice_wallet.coldkeypub.ss58_address)

    assert identity == ChainIdentity(
        additional="",
        description="Local Chain",
        discord="",
        github="https://github.com/opentensor/bittensor",
        image="",
        name="Alice",
        url="https://www.example.com",
    )

    identities = subtensor.get_delegate_identities()

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


def test_change_take(local_chain, subtensor, alice_wallet, bob_wallet):
    """
    Tests:
    - Get default Delegate's take once registered in root subnet
    - Increase and decreased Delegate's take
    - Try corner cases (increase/decrease beyond allowed min/max)
    """

    with pytest.raises(bittensor.HotKeyAccountNotExists):
        subtensor.set_delegate_take(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            0.1,
            raise_error=True,
        )

    subtensor.root_register(
        alice_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    take = subtensor.get_delegate_take(alice_wallet.hotkey.ss58_address)

    assert take == DEFAULT_DELEGATE_TAKE

    with pytest.raises(bittensor.NonAssociatedColdKey):
        subtensor.set_delegate_take(
            bob_wallet,
            alice_wallet.hotkey.ss58_address,
            0.1,
            raise_error=True,
        )

    with pytest.raises(bittensor.DelegateTakeTooHigh):
        subtensor.set_delegate_take(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            0.5,
            raise_error=True,
        )

    subtensor.set_delegate_take(
        alice_wallet,
        alice_wallet.hotkey.ss58_address,
        0.1,
        raise_error=True,
    )

    take = subtensor.get_delegate_take(alice_wallet.hotkey.ss58_address)

    assert take == 0.09999237048905166

    with pytest.raises(bittensor.DelegateTxRateLimitExceeded):
        subtensor.set_delegate_take(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            0.15,
            raise_error=True,
        )

    take = subtensor.get_delegate_take(alice_wallet.hotkey.ss58_address)

    assert take == 0.09999237048905166

    sudo_set_admin_utils(
        local_chain,
        alice_wallet,
        call_function="sudo_set_tx_delegate_take_rate_limit",
        call_params={
            "tx_rate_limit": 0,
        },
    )

    subtensor.set_delegate_take(
        alice_wallet,
        alice_wallet.hotkey.ss58_address,
        0.15,
        raise_error=True,
    )

    take = subtensor.get_delegate_take(alice_wallet.hotkey.ss58_address)

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

    assert subtensor.get_delegates() == []
    assert subtensor.get_delegated(alice_wallet.coldkey.ss58_address) == []
    assert subtensor.get_delegate_by_hotkey(alice_wallet.hotkey.ss58_address) is None
    assert subtensor.get_delegate_by_hotkey(bob_wallet.hotkey.ss58_address) is None

    assert subtensor.is_hotkey_delegate(alice_wallet.hotkey.ss58_address) is False
    assert subtensor.is_hotkey_delegate(bob_wallet.hotkey.ss58_address) is False

    subtensor.root_register(
        alice_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    subtensor.root_register(
        bob_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert subtensor.is_hotkey_delegate(alice_wallet.hotkey.ss58_address) is True
    assert subtensor.is_hotkey_delegate(bob_wallet.hotkey.ss58_address) is True

    alice_delegate = subtensor.get_delegate_by_hotkey(alice_wallet.hotkey.ss58_address)

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

    bob_delegate = subtensor.get_delegate_by_hotkey(bob_wallet.hotkey.ss58_address)

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

    delegates = subtensor.get_delegates()

    assert delegates == [
        bob_delegate,
        alice_delegate,
    ]

    assert subtensor.get_delegated(bob_wallet.coldkey.ss58_address) == []

    alice_subnet_netuid = subtensor.get_total_subnets()  # 2
    # Register a subnet, netuid 2
    assert subtensor.register_subnet(alice_wallet), "Subnet wasn't created"

    # Verify subnet <netuid> created successfully
    assert subtensor.subnet_exists(
        alice_subnet_netuid
    ), "Subnet wasn't created successfully"

    # make sure we passed start_call limit
    subtensor.wait_for_block(subtensor.block + 20)
    status, message = subtensor.start_call(
        alice_wallet, alice_subnet_netuid, True, True
    )
    assert status, message

    subtensor.add_stake(
        bob_wallet,
        alice_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
        amount=Balance.from_tao(10_000),
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    bob_delegated = subtensor.get_delegated(bob_wallet.coldkey.ss58_address)
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
            stake=get_dynamic_balance(bob_delegated[0].stake.rao),
        ),
    ]


def test_nominator_min_required_stake(local_chain, subtensor, alice_wallet, bob_wallet):
    """
    Tests:
    - Check default NominatorMinRequiredStake
    - Add Stake to Nominate
    - Update NominatorMinRequiredStake
    - Check Nominator is removed
    """

    alice_subnet_netuid = subtensor.get_total_subnets()  # 2

    # Register a subnet, netuid 2
    assert subtensor.register_subnet(
        alice_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    ), "Subnet wasn't created"

    # Verify subnet <netuid> created successfully
    assert subtensor.subnet_exists(
        alice_subnet_netuid
    ), "Subnet wasn't created successfully"

    # make sure we passed start_call limit
    subtensor.wait_for_block(subtensor.block + 20)
    status, message = subtensor.start_call(
        alice_wallet, alice_subnet_netuid, True, True
    )
    assert status, message

    minimum_required_stake = subtensor.get_minimum_required_stake()

    assert minimum_required_stake == Balance(0)

    # subtensor.root_register(
    #     alice_wallet,
    #     wait_for_inclusion=True,
    #     wait_for_finalization=True,
    # )
    subtensor.burned_register(
        bob_wallet,
        alice_subnet_netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    success = subtensor.add_stake(
        alice_wallet,
        bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
        amount=Balance.from_tao(10_000),
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert success is True

    stake = subtensor.get_stake(
        alice_wallet.coldkey.ss58_address,
        bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )

    assert stake > 0

    # this will trigger clear_small_nominations
    sudo_set_admin_utils(
        local_chain,
        alice_wallet,
        call_function="sudo_set_nominator_min_required_stake",
        call_params={
            "min_stake": "100000000000000",
        },
    )

    minimum_required_stake = subtensor.get_minimum_required_stake()

    assert minimum_required_stake == Balance.from_tao(100_000)

    stake = subtensor.get_stake(
        alice_wallet.coldkey.ss58_address,
        bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )

    assert stake == Balance(0)


def test_get_vote_data(subtensor, alice_wallet):
    """
    Tests:
    - Sends Propose
    - Checks existing Proposals
    - Votes
    - Checks Proposal is updated
    """

    subtensor.root_register(alice_wallet)

    proposals = subtensor.query_map(
        "Triumvirate",
        "ProposalOf",
        params=[],
    )

    assert proposals.records == []

    success, error = propose(
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

    proposals = subtensor.query_map(
        "Triumvirate",
        "ProposalOf",
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

    proposal = subtensor.get_vote_data(
        proposal_hash,
    )

    assert proposal == ProposalVoteData(
        ayes=[],
        end=CLOSE_IN_VALUE(1_000_000, subtensor.block),
        index=0,
        nays=[],
        threshold=3,
    )

    success, error = vote(
        subtensor,
        alice_wallet,
        alice_wallet.hotkey.ss58_address,
        proposal_hash,
        index=0,
        approve=True,
    )

    assert error == ""
    assert success is True

    proposal = subtensor.get_vote_data(
        proposal_hash,
    )

    assert proposal == ProposalVoteData(
        ayes=[
            alice_wallet.hotkey.ss58_address,
        ],
        end=CLOSE_IN_VALUE(1_000_000, subtensor.block),
        index=0,
        nays=[],
        threshold=3,
    )
