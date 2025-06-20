from bittensor import logging
from bittensor.core.chain_data.stake_info import StakeInfo
from bittensor.utils.balance import Balance
from tests.e2e_tests.utils.chain_interactions import get_dynamic_balance
from tests.helpers.helpers import ApproxBalance
from tests.e2e_tests.utils.e2e_test_utils import wait_to_start_call


def test_single_operation(subtensor, alice_wallet, bob_wallet):
    """
    Tests:
    - Staking using `add_stake`
    - Unstaking using `unstake`
    - Checks StakeInfo
    """
    alice_subnet_netuid = subtensor.get_total_subnets()  # 2

    # Register root as Alice - the subnet owner and validator
    assert subtensor.register_subnet(alice_wallet, True, True)

    # Verify subnet <netuid> created successfully
    assert subtensor.subnet_exists(alice_subnet_netuid), (
        "Subnet wasn't created successfully"
    )

    assert wait_to_start_call(subtensor, alice_wallet, alice_subnet_netuid)

    subtensor.burned_register(
        wallet=alice_wallet,
        netuid=alice_subnet_netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    logging.console.success(f"Alice is registered in subnet {alice_subnet_netuid}")
    subtensor.burned_register(
        wallet=bob_wallet,
        netuid=alice_subnet_netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    logging.console.success(f"Bob is registered in subnet {alice_subnet_netuid}")

    stake = subtensor.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )

    assert stake == Balance(0).set_unit(alice_subnet_netuid)

    success = subtensor.add_stake(
        wallet=alice_wallet,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
        amount=Balance.from_tao(1),
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=16,
    )

    assert success is True

    stake_alice = subtensor.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )
    logging.console.info(f"Alice stake: {stake_alice}")

    stake_bob = subtensor.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )

    logging.console.info(f"Bob stake: {stake_bob}")
    assert stake_bob > Balance(0).set_unit(alice_subnet_netuid)

    stakes = subtensor.get_stake_for_coldkey(alice_wallet.coldkey.ss58_address)

    expected_stakes = [
        StakeInfo(
            hotkey_ss58=stakes[0].hotkey_ss58,
            coldkey_ss58=alice_wallet.coldkey.ss58_address,
            netuid=alice_subnet_netuid,
            stake=get_dynamic_balance(stakes[0].stake.rao, alice_subnet_netuid),
            locked=Balance(0).set_unit(alice_subnet_netuid),
            emission=get_dynamic_balance(stakes[0].emission.rao, alice_subnet_netuid),
            drain=0,
            is_registered=True,
        ),
    ]

    fast_blocks_stake = (
        [
            StakeInfo(
                hotkey_ss58=stakes[1].hotkey_ss58,
                coldkey_ss58=alice_wallet.coldkey.ss58_address,
                netuid=alice_subnet_netuid,
                stake=get_dynamic_balance(stakes[1].stake.rao, alice_subnet_netuid),
                locked=Balance(0).set_unit(alice_subnet_netuid),
                emission=get_dynamic_balance(
                    stakes[1].emission.rao, alice_subnet_netuid
                ),
                drain=0,
                is_registered=True,
            )
        ]
        if subtensor.is_fast_blocks()
        else []
    )

    expected_stakes += fast_blocks_stake

    assert stakes == expected_stakes
    assert subtensor.get_stake_for_coldkey == subtensor.get_stake_info_for_coldkey

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
            stake=stake.set_unit(1),
            locked=Balance.from_tao(0, netuid=1),
            emission=Balance.from_tao(0, netuid=1),
            drain=0,
            is_registered=False,
        ),
        2: StakeInfo(
            hotkey_ss58=bob_wallet.hotkey.ss58_address,
            coldkey_ss58=alice_wallet.coldkey.ss58_address,
            netuid=alice_subnet_netuid,
            stake=get_dynamic_balance(stakes[2].stake.rao, alice_subnet_netuid),
            locked=Balance.from_tao(0, netuid=alice_subnet_netuid),
            emission=get_dynamic_balance(stakes[2].emission.rao, alice_subnet_netuid),
            drain=0,
            is_registered=True,
        ),
    }

    # unstale all to check in later
    success = subtensor.unstake(
        wallet=alice_wallet,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        period=16,
    )

    assert success is True

    stake = subtensor.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )

    # all balances have been unstaked
    assert stake == Balance(0).set_unit(alice_subnet_netuid)
    logging.console.success(f"✅ Test [green]test_single_operation[/green] passed")


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

    # make sure we passed start_call limit for both subnets
    for netuid in netuids:
        assert wait_to_start_call(subtensor, alice_wallet, netuid)

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

    expected_balances = {
        alice_wallet.coldkey.ss58_address: get_dynamic_balance(
            balances[alice_wallet.coldkey.ss58_address].rao
        ),
        bob_wallet.coldkey.ss58_address: get_dynamic_balance(
            balances[bob_wallet.coldkey.ss58_address].rao
        ),
    }

    assert balances == expected_balances

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

    expected_balances = {
        alice_wallet.coldkey.ss58_address: get_dynamic_balance(
            balances[alice_wallet.coldkey.ss58_address].rao
        ),
        bob_wallet.coldkey.ss58_address: get_dynamic_balance(
            balances[bob_wallet.coldkey.ss58_address].rao
        ),
    }

    assert balances == expected_balances

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

    expected_balances = {
        alice_wallet.coldkey.ss58_address: get_dynamic_balance(
            balances[alice_wallet.coldkey.ss58_address].rao,
        ),
        bob_wallet.coldkey.ss58_address: Balance.from_tao(999_999.8),
    }

    assert balances == expected_balances

    assert balances[alice_wallet.coldkey.ss58_address] > alice_balance
    logging.console.success(f"✅ Test [green]test_batch_operations[/green] passed")


def test_safe_staking_scenarios(subtensor, alice_wallet, bob_wallet):
    """
    Tests safe staking scenarios with different parameters.

    For both staking and unstaking:
    1. Fails with strict threshold (0.5%) and no partial staking
    2. Succeeds with strict threshold (0.5%) and partial staking allowed
    3. Succeeds with lenient threshold (10% and 30%) and no partial staking
    """
    alice_subnet_netuid = subtensor.get_total_subnets()  # 2
    # Register root as Alice - the subnet owner and validator
    assert subtensor.register_subnet(alice_wallet, True, True)

    # Verify subnet created successfully
    assert subtensor.subnet_exists(alice_subnet_netuid), (
        "Subnet wasn't created successfully"
    )

    assert wait_to_start_call(subtensor, alice_wallet, alice_subnet_netuid)

    subtensor.burned_register(
        alice_wallet,
        netuid=alice_subnet_netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    subtensor.burned_register(
        bob_wallet,
        netuid=alice_subnet_netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    initial_stake = subtensor.get_stake(
        alice_wallet.coldkey.ss58_address,
        bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )
    assert initial_stake == Balance(0)

    # Test Staking Scenarios
    stake_amount = Balance.from_tao(100)

    # 1. Strict params - should fail
    success = subtensor.add_stake(
        alice_wallet,
        bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
        amount=stake_amount,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        safe_staking=True,
        rate_tolerance=0.005,  # 0.5%
        allow_partial_stake=False,
    )
    assert success is False

    current_stake = subtensor.get_stake(
        alice_wallet.coldkey.ss58_address,
        bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )
    assert current_stake == Balance(0), "Stake should not change after failed attempt"

    # 2. Partial allowed - should succeed partially
    success = subtensor.add_stake(
        alice_wallet,
        bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
        amount=stake_amount,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        safe_staking=True,
        rate_tolerance=0.005,  # 0.5%
        allow_partial_stake=True,
    )
    assert success is True

    partial_stake = subtensor.get_stake(
        alice_wallet.coldkey.ss58_address,
        bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )
    assert partial_stake > Balance(0), "Partial stake should be added"
    assert partial_stake < stake_amount, (
        "Partial stake should be less than requested amount"
    )

    # 3. Higher threshold - should succeed fully
    amount = Balance.from_tao(100)
    success = subtensor.add_stake(
        alice_wallet,
        bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
        amount=amount,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        safe_staking=True,
        rate_tolerance=0.22,  # 22%
        allow_partial_stake=False,
    )
    assert success is True

    full_stake = subtensor.get_stake(
        alice_wallet.coldkey.ss58_address,
        bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )

    # Test Unstaking Scenarios
    # 1. Strict params - should fail
    success = subtensor.unstake(
        alice_wallet,
        bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
        amount=full_stake,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        safe_staking=True,
        rate_tolerance=0.005,  # 0.5%
        allow_partial_stake=False,
    )
    assert success is False, "Unstake should fail."

    current_stake = subtensor.get_stake(
        alice_wallet.coldkey.ss58_address,
        bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )

    logging.console.info(f"[orange]Current stake: {current_stake}[orange]")
    logging.console.info(f"[orange]Full stake: {full_stake}[orange]")

    assert current_stake == full_stake, (
        "Stake should not change after failed unstake attempt"
    )

    # 2. Partial allowed - should succeed partially
    success = subtensor.unstake(
        alice_wallet,
        bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
        amount=current_stake,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        safe_staking=True,
        rate_tolerance=0.005,  # 0.5%
        allow_partial_stake=True,
    )
    assert success is True

    partial_unstake = subtensor.get_stake(
        alice_wallet.coldkey.ss58_address,
        bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )
    logging.console.info(f"[orange]Partial unstake: {partial_unstake}[orange]")
    assert partial_unstake > Balance(0), "Some stake should remain"

    # 3. Higher threshold - should succeed fully
    success = subtensor.unstake(
        alice_wallet,
        bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
        amount=partial_unstake,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        safe_staking=True,
        rate_tolerance=0.3,  # 30%
        allow_partial_stake=False,
    )
    assert success is True, "Unstake should succeed"
    logging.console.success(
        f"✅ Test [green]test_safe_staking_scenarios[/green] passed"
    )


def test_safe_swap_stake_scenarios(subtensor, alice_wallet, bob_wallet):
    """
    Tests safe swap stake scenarios with different parameters.

    Tests:
    1. Fails with strict threshold (0.5%)
    2. Succeeds with lenient threshold (10%)
    """
    # Create new subnet (netuid 2) and register Alice
    origin_netuid = 2
    assert subtensor.register_subnet(bob_wallet, True, True)
    assert subtensor.subnet_exists(origin_netuid), "Subnet wasn't created successfully"
    dest_netuid = 3
    assert subtensor.register_subnet(bob_wallet, True, True)
    assert subtensor.subnet_exists(dest_netuid), "Subnet wasn't created successfully"

    # make sure we passed start_call limit for both subnets
    assert wait_to_start_call(subtensor, bob_wallet, origin_netuid)
    assert wait_to_start_call(subtensor, bob_wallet, dest_netuid)

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
        rate_tolerance=0.005,  # 0.5%
        allow_partial_stake=False,
    )
    assert success is False

    # Verify no stake was moved
    dest_stake = subtensor.get_stake(
        alice_wallet.coldkey.ss58_address,
        alice_wallet.hotkey.ss58_address,
        netuid=dest_netuid,
    )
    assert dest_stake == Balance(0), (
        "Destination stake should remain 0 after failed swap"
    )

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
        rate_tolerance=0.3,  # 30%
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
    assert dest_stake > Balance(0), (
        "Destination stake should be non-zero after successful swap"
    )
    logging.console.success(
        f"✅ Test [green]test_safe_swap_stake_scenarios[/green] passed"
    )


def test_move_stake(subtensor, alice_wallet, bob_wallet):
    """
    Tests:
    - Adding stake
    - Moving stake from one hotkey-subnet pair to another
    """

    alice_subnet_netuid = subtensor.get_total_subnets()  # 2
    assert subtensor.register_subnet(alice_wallet, True, True)
    assert subtensor.subnet_exists(alice_subnet_netuid), (
        "Subnet wasn't created successfully"
    )

    assert wait_to_start_call(subtensor, alice_wallet, alice_subnet_netuid)

    subtensor.burned_register(
        alice_wallet,
        netuid=alice_subnet_netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert subtensor.add_stake(
        alice_wallet,
        alice_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
        amount=Balance.from_tao(1_000),
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    stakes = subtensor.get_stake_for_coldkey(alice_wallet.coldkey.ss58_address)

    assert stakes == [
        StakeInfo(
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            coldkey_ss58=alice_wallet.coldkey.ss58_address,
            netuid=alice_subnet_netuid,
            stake=get_dynamic_balance(stakes[0].stake.rao, alice_subnet_netuid),
            locked=Balance(0),
            emission=get_dynamic_balance(stakes[0].emission.rao, alice_subnet_netuid),
            drain=0,
            is_registered=True,
        ),
    ]

    bob_subnet_netuid = subtensor.get_total_subnets()  # 3
    subtensor.register_subnet(bob_wallet, True, True)
    assert subtensor.subnet_exists(bob_subnet_netuid), (
        "Subnet wasn't created successfully"
    )

    assert wait_to_start_call(subtensor, bob_wallet, bob_subnet_netuid)

    assert subtensor.move_stake(
        alice_wallet,
        origin_hotkey=alice_wallet.hotkey.ss58_address,
        origin_netuid=alice_subnet_netuid,
        destination_hotkey=bob_wallet.hotkey.ss58_address,
        destination_netuid=bob_subnet_netuid,
        amount=stakes[0].stake,
        wait_for_finalization=True,
        wait_for_inclusion=True,
    )

    stakes = subtensor.get_stake_for_coldkey(alice_wallet.coldkey.ss58_address)

    expected_stakes = [
        StakeInfo(
            hotkey_ss58=stakes[0].hotkey_ss58,
            coldkey_ss58=alice_wallet.coldkey.ss58_address,
            netuid=alice_subnet_netuid
            if subtensor.is_fast_blocks()
            else bob_subnet_netuid,
            stake=get_dynamic_balance(stakes[0].stake.rao, bob_subnet_netuid),
            locked=Balance(0).set_unit(bob_subnet_netuid),
            emission=get_dynamic_balance(stakes[0].emission.rao, bob_subnet_netuid),
            drain=0,
            is_registered=True,
        )
    ]

    fast_block_stake = (
        [
            StakeInfo(
                hotkey_ss58=stakes[1].hotkey_ss58,
                coldkey_ss58=alice_wallet.coldkey.ss58_address,
                netuid=bob_subnet_netuid,
                stake=get_dynamic_balance(stakes[1].stake.rao, bob_subnet_netuid),
                locked=Balance(0).set_unit(bob_subnet_netuid),
                emission=get_dynamic_balance(stakes[1].emission.rao, bob_subnet_netuid),
                drain=0,
                is_registered=True,
            ),
        ]
        if subtensor.is_fast_blocks()
        else []
    )

    expected_stakes += fast_block_stake

    assert stakes == expected_stakes
    logging.console.success(f"✅ Test [green]test_move_stake[/green] passed")


def test_transfer_stake(subtensor, alice_wallet, bob_wallet, dave_wallet):
    """
    Tests:
    - Adding stake
    - Transferring stake from one coldkey-subnet pair to another
    """
    alice_subnet_netuid = subtensor.get_total_subnets()  # 2

    assert subtensor.register_subnet(alice_wallet, True, True)
    assert subtensor.subnet_exists(alice_subnet_netuid), (
        "Subnet wasn't created successfully"
    )

    assert wait_to_start_call(subtensor, alice_wallet, alice_subnet_netuid)

    subtensor.burned_register(
        alice_wallet,
        netuid=alice_subnet_netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert subtensor.add_stake(
        alice_wallet,
        alice_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
        amount=Balance.from_tao(1_000),
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    alice_stakes = subtensor.get_stake_for_coldkey(alice_wallet.coldkey.ss58_address)

    assert alice_stakes == [
        StakeInfo(
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            coldkey_ss58=alice_wallet.coldkey.ss58_address,
            netuid=alice_subnet_netuid,
            stake=get_dynamic_balance(alice_stakes[0].stake.rao, alice_subnet_netuid),
            locked=Balance(0),
            emission=get_dynamic_balance(
                alice_stakes[0].emission.rao, alice_subnet_netuid
            ),
            drain=0,
            is_registered=True,
        ),
    ]

    bob_stakes = subtensor.get_stake_for_coldkey(bob_wallet.coldkey.ss58_address)

    assert bob_stakes == []

    dave_subnet_netuid = subtensor.get_total_subnets()  # 3
    subtensor.register_subnet(dave_wallet, True, True)

    assert wait_to_start_call(subtensor, dave_wallet, dave_subnet_netuid)

    subtensor.burned_register(
        bob_wallet,
        netuid=dave_subnet_netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    assert subtensor.transfer_stake(
        alice_wallet,
        destination_coldkey_ss58=bob_wallet.coldkey.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        origin_netuid=alice_subnet_netuid,
        destination_netuid=dave_subnet_netuid,
        amount=alice_stakes[0].stake,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    alice_stakes = subtensor.get_stake_for_coldkey(alice_wallet.coldkey.ss58_address)

    expected_alice_stake = (
        [
            StakeInfo(
                hotkey_ss58=alice_wallet.hotkey.ss58_address,
                coldkey_ss58=alice_wallet.coldkey.ss58_address,
                netuid=alice_subnet_netuid,
                stake=get_dynamic_balance(
                    alice_stakes[0].stake.rao, alice_subnet_netuid
                ),
                locked=Balance(0).set_unit(alice_subnet_netuid),
                emission=get_dynamic_balance(
                    alice_stakes[0].emission.rao, alice_subnet_netuid
                ),
                drain=0,
                is_registered=True,
            ),
        ]
        if subtensor.is_fast_blocks()
        else []
    )

    assert alice_stakes == expected_alice_stake

    bob_stakes = subtensor.get_stake_for_coldkey(bob_wallet.coldkey.ss58_address)

    expected_bob_stake = [
        StakeInfo(
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            coldkey_ss58=bob_wallet.coldkey.ss58_address,
            netuid=dave_subnet_netuid,
            stake=get_dynamic_balance(bob_stakes[0].stake.rao, dave_subnet_netuid),
            locked=Balance(0),
            emission=get_dynamic_balance(
                bob_stakes[0].emission.rao, dave_subnet_netuid
            ),
            drain=0,
            is_registered=False,
        ),
    ]
    assert bob_stakes == expected_bob_stake
    logging.console.success(f"✅ Test [green]test_transfer_stake[/green] passed")
