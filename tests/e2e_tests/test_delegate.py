import async_substrate_interface.errors
import pytest

from bittensor.core.chain_data.chain_identity import ChainIdentity
from bittensor.core.chain_data.delegate_info import DelegatedInfo, DelegateInfo
from bittensor.utils.balance import Balance
from tests.e2e_tests.utils.chain_interactions import (
    set_identity,
    sudo_set_admin_utils,
)

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

    with pytest.raises(async_substrate_interface.errors.HotKeyAccountNotExists):
        subtensor.set_delegate_take(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            0.1,
        )

    subtensor.root_register(
        alice_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    take = subtensor.get_delegate_take(alice_wallet.hotkey.ss58_address)

    assert take == DEFAULT_DELEGATE_TAKE

    with pytest.raises(async_substrate_interface.errors.NonAssociatedColdKey):
        subtensor.set_delegate_take(
            bob_wallet,
            alice_wallet.hotkey.ss58_address,
            0.1,
        )

    with pytest.raises(async_substrate_interface.errors.DelegateTakeTooHigh):
        subtensor.set_delegate_take(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            0.5,
        )

    subtensor.set_delegate_take(
        alice_wallet,
        alice_wallet.hotkey.ss58_address,
        0.1,
    )

    take = subtensor.get_delegate_take(alice_wallet.hotkey.ss58_address)

    assert take == 0.09999237048905166

    with pytest.raises(async_substrate_interface.errors.DelegateTxRateLimitExceeded):
        subtensor.set_delegate_take(
            alice_wallet,
            alice_wallet.hotkey.ss58_address,
            0.15,
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

    subtensor.add_stake(
        bob_wallet,
        alice_wallet.hotkey.ss58_address,
        netuid=0,
        amount=Balance.from_tao(10_000),
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert subtensor.get_delegated(bob_wallet.coldkey.ss58_address) == [
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


def test_nominator_min_required_stake(local_chain, subtensor, alice_wallet, bob_wallet):
    """
    Tests:
    - Check default NominatorMinRequiredStake
    - Add Stake to Nominate
    - Update NominatorMinRequiredStake
    - Check Nominator is removed
    """

    minimum_required_stake = subtensor.get_minimum_required_stake()

    assert minimum_required_stake == Balance(0)

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

    success = subtensor.add_stake(
        alice_wallet,
        bob_wallet.hotkey.ss58_address,
        netuid=0,
        amount=Balance.from_tao(10_000),
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert success is True

    stake = subtensor.get_stake(
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
        return_error_message=True,
    )

    minimum_required_stake = subtensor.get_minimum_required_stake()

    assert minimum_required_stake == Balance.from_tao(100_000)

    stake = subtensor.get_stake(
        alice_wallet.coldkey.ss58_address,
        bob_wallet.hotkey.ss58_address,
        netuid=0,
    )

    assert stake == Balance(0)
