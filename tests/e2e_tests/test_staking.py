from bittensor import logging
from bittensor.core.chain_data.stake_info import StakeInfo
from bittensor.utils.balance import Balance
from tests.e2e_tests.utils.chain_interactions import ANY_BALANCE
from tests.helpers.helpers import ApproxBalance

logging.enable_info()


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
        alice_wallet.coldkey.ss58_address: ApproxBalance(alice_balance.rao),
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


def test_safe_staking_scenarios(subtensor, alice_wallet, bob_wallet):
    """
    Tests safe staking scenarios with different parameters.

    For both staking and unstaking:
    1. Fails with strict threshold (0.5%) and no partial staking
    2. Succeeds with strict threshold (0.5%) and partial staking allowed
    3. Succeeds with lenient threshold (10% and 30%) and no partial staking
    """
    netuid = 2
    # Register root as Alice - the subnet owner and validator
    assert subtensor.register_subnet(alice_wallet)

    # Verify subnet created successfully
    assert subtensor.subnet_exists(netuid), "Subnet wasn't created successfully"

    subtensor.burned_register(
        alice_wallet,
        netuid=netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    subtensor.burned_register(
        bob_wallet,
        netuid=netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    initial_stake = subtensor.get_stake(
        alice_wallet.coldkey.ss58_address,
        bob_wallet.hotkey.ss58_address,
        netuid=netuid,
    )
    assert initial_stake == Balance(0)

    # Test Staking Scenarios
    stake_amount = Balance.from_tao(100)

    # 1. Strict params - should fail
    success = subtensor.add_stake(
        alice_wallet,
        bob_wallet.hotkey.ss58_address,
        netuid=netuid,
        amount=stake_amount,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        safe_staking=True,
        rate_threshold=0.005,  # 0.5%
        allow_partial_stake=False,
    )
    assert success is False

    current_stake = subtensor.get_stake(
        alice_wallet.coldkey.ss58_address,
        bob_wallet.hotkey.ss58_address,
        netuid=netuid,
    )
    assert current_stake == Balance(0), "Stake should not change after failed attempt"

    # 2. Partial allowed - should succeed partially
    success = subtensor.add_stake(
        alice_wallet,
        bob_wallet.hotkey.ss58_address,
        netuid=netuid,
        amount=stake_amount,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        safe_staking=True,
        rate_threshold=0.005,  # 0.5%
        allow_partial_stake=True,
    )
    assert success is True

    partial_stake = subtensor.get_stake(
        alice_wallet.coldkey.ss58_address,
        bob_wallet.hotkey.ss58_address,
        netuid=netuid,
    )
    assert partial_stake > Balance(0), "Partial stake should be added"
    assert (
        partial_stake < stake_amount
    ), "Partial stake should be less than requested amount"

    # 3. Higher threshold - should succeed fully
    amount = Balance.from_tao(100)
    success = subtensor.add_stake(
        alice_wallet,
        bob_wallet.hotkey.ss58_address,
        netuid=netuid,
        amount=amount,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        safe_staking=True,
        rate_threshold=0.1,  # 10%
        allow_partial_stake=False,
    )
    assert success is True

    full_stake = subtensor.get_stake(
        alice_wallet.coldkey.ss58_address,
        bob_wallet.hotkey.ss58_address,
        netuid=netuid,
    )

    # Test Unstaking Scenarios
    # 1. Strict params - should fail
    success = subtensor.unstake(
        alice_wallet,
        bob_wallet.hotkey.ss58_address,
        netuid=netuid,
        amount=stake_amount,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        safe_staking=True,
        rate_threshold=0.005,  # 0.5%
        allow_partial_stake=False,
    )
    assert success is False

    current_stake = subtensor.get_stake(
        alice_wallet.coldkey.ss58_address,
        bob_wallet.hotkey.ss58_address,
        netuid=netuid,
    )
    assert (
        current_stake == full_stake
    ), "Stake should not change after failed unstake attempt"

    # 2. Partial allowed - should succeed partially
    success = subtensor.unstake(
        alice_wallet,
        bob_wallet.hotkey.ss58_address,
        netuid=netuid,
        amount=current_stake,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        safe_staking=True,
        rate_threshold=0.005,  # 0.5%
        allow_partial_stake=True,
    )
    assert success is True

    partial_unstake = subtensor.get_stake(
        alice_wallet.coldkey.ss58_address,
        bob_wallet.hotkey.ss58_address,
        netuid=netuid,
    )
    assert partial_unstake > Balance(0), "Some stake should remain"

    # 3. Higher threshold - should succeed fully
    success = subtensor.unstake(
        alice_wallet,
        bob_wallet.hotkey.ss58_address,
        netuid=netuid,
        amount=partial_unstake,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        safe_staking=True,
        rate_threshold=0.3,  # 30%
        allow_partial_stake=False,
    )
    assert success is True


def test_safe_swap_stake_scenarios(subtensor, alice_wallet, bob_wallet):
    """
    Tests safe swap stake scenarios with different parameters.

    Tests:
    1. Fails with strict threshold (0.5%)
    2. Succeeds with lenient threshold (10%)
    """
    # Create new subnet (netuid 2) and register Alice
    origin_netuid = 2
    assert subtensor.register_subnet(bob_wallet)
    assert subtensor.subnet_exists(origin_netuid), "Subnet wasn't created successfully"
    dest_netuid = 3
    assert subtensor.register_subnet(bob_wallet)
    assert subtensor.subnet_exists(dest_netuid), "Subnet wasn't created successfully"

    # Register Alice on both subnets
    subtensor.burned_register(
        alice_wallet,
        netuid=origin_netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    subtensor.burned_register(
        alice_wallet,
        netuid=dest_netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    # Add initial stake to swap from
    initial_stake_amount = Balance.from_tao(10_000)
    success = subtensor.add_stake(
        alice_wallet,
        alice_wallet.hotkey.ss58_address,
        netuid=origin_netuid,
        amount=initial_stake_amount,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert success is True

    origin_stake = subtensor.get_stake(
        alice_wallet.coldkey.ss58_address,
        alice_wallet.hotkey.ss58_address,
        netuid=origin_netuid,
    )
    assert origin_stake > Balance(0), "Origin stake should be non-zero"

    stake_swap_amount = Balance.from_tao(10_000)
    # 1. Try swap with strict threshold and big amount- should fail
    success = subtensor.swap_stake(
        wallet=alice_wallet,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        origin_netuid=origin_netuid,
        destination_netuid=dest_netuid,
        amount=stake_swap_amount,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        safe_staking=True,
        rate_threshold=0.005,  # 0.5%
        allow_partial_stake=False,
    )
    assert success is False

    # Verify no stake was moved
    dest_stake = subtensor.get_stake(
        alice_wallet.coldkey.ss58_address,
        alice_wallet.hotkey.ss58_address,
        netuid=dest_netuid,
    )
    assert dest_stake == Balance(
        0
    ), "Destination stake should remain 0 after failed swap"

    # 2. Try swap with higher threshold and less amount - should succeed
    stake_swap_amount = Balance.from_tao(100)
    success = subtensor.swap_stake(
        wallet=alice_wallet,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        origin_netuid=origin_netuid,
        destination_netuid=dest_netuid,
        amount=stake_swap_amount,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        safe_staking=True,
        rate_threshold=0.3,  # 30%
        allow_partial_stake=True,
    )
    assert success is True

    # Verify stake was moved
    origin_stake = subtensor.get_stake(
        alice_wallet.coldkey.ss58_address,
        alice_wallet.hotkey.ss58_address,
        netuid=origin_netuid,
    )
    dest_stake = subtensor.get_stake(
        alice_wallet.coldkey.ss58_address,
        alice_wallet.hotkey.ss58_address,
        netuid=dest_netuid,
    )
    assert dest_stake > Balance(
        0
    ), "Destination stake should be non-zero after successful swap"
