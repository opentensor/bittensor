import pytest

from bittensor.core.chain_data.stake_info import StakeInfo
from bittensor.utils.balance import Balance
from tests.e2e_tests.utils.chain_interactions import ANY_BALANCE


def test_single_operation(subtensor, alice_wallet, bob_wallet):
    """
    Tests:
    - Staking using `add_stake`
    - Unstaking using `unstake`
    - Checks StakeInfo
    """

    subtensor.burned_register(
        alice_wallet,
        netuid=1,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    subtensor.burned_register(
        bob_wallet,
        netuid=1,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    stake = subtensor.get_stake(
        alice_wallet.coldkey.ss58_address,
        bob_wallet.hotkey.ss58_address,
        netuid=1,
    )

    assert stake == Balance(0)

    success = subtensor.add_stake(
        alice_wallet,
        bob_wallet.hotkey.ss58_address,
        netuid=1,
        amount=Balance.from_tao(10_000),
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert success is True

    stake = subtensor.get_stake(
        alice_wallet.coldkey.ss58_address,
        bob_wallet.hotkey.ss58_address,
        netuid=1,
    )

    assert stake > Balance(0)

    stakes = subtensor.get_stake_for_coldkey(alice_wallet.coldkey.ss58_address)

    assert stakes == [
        StakeInfo(
            hotkey_ss58=bob_wallet.hotkey.ss58_address,
            coldkey_ss58=alice_wallet.coldkey.ss58_address,
            netuid=1,
            stake=stake,
            locked=Balance(0),
            emission=Balance(0),
            drain=0,
            is_registered=True,
        ),
    ]

    stakes = subtensor.get_stake_info_for_coldkey(alice_wallet.coldkey.ss58_address)

    assert stakes == [
        StakeInfo(
            hotkey_ss58=bob_wallet.hotkey.ss58_address,
            coldkey_ss58=alice_wallet.coldkey.ss58_address,
            netuid=1,
            stake=stake,
            locked=Balance(0),
            emission=Balance(0),
            drain=0,
            is_registered=True,
        ),
    ]

    stakes = subtensor.get_stake_for_coldkey_and_hotkey(
        alice_wallet.coldkey.ss58_address,
        bob_wallet.hotkey.ss58_address,
    )

    assert stakes == {
        0: StakeInfo(
            hotkey_ss58=bob_wallet.hotkey.ss58_address,
            coldkey_ss58=alice_wallet.coldkey.ss58_address,
            netuid=0,
            stake=Balance(0),
            locked=Balance(0),
            emission=Balance(0),
            drain=0,
            is_registered=False,
        ),
        1: StakeInfo(
            hotkey_ss58=bob_wallet.hotkey.ss58_address,
            coldkey_ss58=alice_wallet.coldkey.ss58_address,
            netuid=1,
            stake=stake,
            locked=Balance.from_tao(0, netuid=1),
            emission=Balance.from_tao(0, netuid=1),
            drain=0,
            is_registered=True,
        ),
    }

    success = subtensor.unstake(
        alice_wallet,
        bob_wallet.hotkey.ss58_address,
        netuid=1,
        amount=stake,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert success is True

    stake = subtensor.get_stake(
        alice_wallet.coldkey.ss58_address,
        bob_wallet.hotkey.ss58_address,
        netuid=1,
    )

    assert stake == Balance(0)


@pytest.mark.skip(
    reason="add_stake_multiple and unstake_multiple doesn't return (just hangs)",
)
def test_batch_operations(subtensor, alice_wallet, bob_wallet):
    """
    Tests:
    - Staking using `add_stake_multiple`
    - Unstaking using `unstake_multiple`
    - Checks StakeInfo
    - Checks Accounts Balance
    """

    netuids = [
        2,
        3,
    ]

    for _ in netuids:
        subtensor.register_subnet(
            alice_wallet,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )

    for netuid in netuids:
        subtensor.burned_register(
            bob_wallet,
            netuid,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )

    for netuid in netuids:
        stake = subtensor.get_stake(
            alice_wallet.coldkey.ss58_address,
            bob_wallet.hotkey.ss58_address,
            netuid=netuid,
        )

        assert stake == Balance(0), f"netuid={netuid} stake={stake}"

    balances = subtensor.get_balances(
        alice_wallet.coldkey.ss58_address,
        bob_wallet.coldkey.ss58_address,
    )

    assert balances == {
        alice_wallet.coldkey.ss58_address: ANY_BALANCE,
        bob_wallet.coldkey.ss58_address: Balance.from_tao(999_998),
    }

    alice_balance = balances[alice_wallet.coldkey.ss58_address]

    success = subtensor.add_stake_multiple(
        alice_wallet,
        hotkey_ss58s=[bob_wallet.hotkey.ss58_address for _ in netuids],
        netuids=netuids,
        amounts=[Balance.from_tao(10_000) for _ in netuids],
    )

    assert success is True

    stakes = [
        subtensor.get_stake(
            alice_wallet.coldkey.ss58_address,
            bob_wallet.hotkey.ss58_address,
            netuid=netuid,
        )
        for netuid in netuids
    ]

    for netuid, stake in zip(netuids, stakes):
        assert stake > Balance(0), f"netuid={netuid} stake={stake}"

    alice_balance -= len(netuids) * Balance.from_tao(10_000)

    balances = subtensor.get_balances(
        alice_wallet.coldkey.ss58_address,
        bob_wallet.coldkey.ss58_address,
    )

    assert balances == {
        alice_wallet.coldkey.ss58_address: alice_balance,
        bob_wallet.coldkey.ss58_address: Balance.from_tao(999_998),
    }

    success = subtensor.unstake_multiple(
        alice_wallet,
        hotkey_ss58s=[bob_wallet.hotkey.ss58_address for _ in netuids],
        netuids=netuids,
        amounts=[Balance.from_tao(100) for _ in netuids],
    )

    assert success is True

    for netuid, old_stake in zip(netuids, stakes):
        stake = subtensor.get_stake(
            alice_wallet.coldkey.ss58_address,
            bob_wallet.hotkey.ss58_address,
            netuid=netuid,
        )

        assert stake < old_stake, f"netuid={netuid} stake={stake}"

    balances = subtensor.get_balances(
        alice_wallet.coldkey.ss58_address,
        bob_wallet.coldkey.ss58_address,
    )

    assert balances == {
        alice_wallet.coldkey.ss58_address: ANY_BALANCE,
        bob_wallet.coldkey.ss58_address: Balance.from_tao(999_998),
    }
    assert balances[alice_wallet.coldkey.ss58_address] > alice_balance
