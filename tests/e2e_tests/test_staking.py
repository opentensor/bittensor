import pytest
import asyncio

from bittensor import logging
from bittensor.core.chain_data.stake_info import StakeInfo
from bittensor.core.errors import ChainError
from bittensor.utils.balance import Balance
from tests.e2e_tests.utils.chain_interactions import (
    async_sudo_set_admin_utils,
    get_dynamic_balance,
    sudo_set_admin_utils,
)
from tests.e2e_tests.utils.e2e_test_utils import (
    async_wait_to_start_call,
    wait_to_start_call,
)
from tests.helpers.helpers import CloseInValue


def test_single_operation(subtensor, alice_wallet, bob_wallet):
    """
    Tests:
    - Staking using `add_stake`
    - Unstaking using `unstake`
    - Checks StakeInfo
    """
    logging.console.info("Testing [blue]test_single_operation[/blue]")

    alice_subnet_netuid = subtensor.subnets.get_total_subnets()  # 2

    # Register root as Alice - the subnet owner and validator
    assert subtensor.subnets.register_subnet(alice_wallet).success

    # Verify subnet <netuid> created successfully
    assert subtensor.subnets.subnet_exists(alice_subnet_netuid), (
        "Subnet wasn't created successfully"
    )

    assert wait_to_start_call(subtensor, alice_wallet, alice_subnet_netuid)

    assert subtensor.subnets.burned_register(
        wallet=alice_wallet,
        netuid=alice_subnet_netuid,
    ).success
    logging.console.success(f"Alice is registered in subnet {alice_subnet_netuid}")
    assert subtensor.subnets.burned_register(
        wallet=bob_wallet,
        netuid=alice_subnet_netuid,
    ).success
    logging.console.success(f"Bob is registered in subnet {alice_subnet_netuid}")

    stake = subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )

    assert stake == Balance(0).set_unit(alice_subnet_netuid)

    success = subtensor.staking.add_stake(
        wallet=alice_wallet,
        netuid=alice_subnet_netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=Balance.from_tao(1),
        period=16,
    )

    assert success is True

    stake_alice = subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )
    logging.console.info(f"Alice stake: {stake_alice}")

    stake_bob = subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )

    logging.console.info(f"Bob stake: {stake_bob}")
    assert stake_bob > Balance(0).set_unit(alice_subnet_netuid)

    stakes = subtensor.staking.get_stake_for_coldkey(alice_wallet.coldkey.ss58_address)

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
        if subtensor.chain.is_fast_blocks()
        else []
    )

    expected_stakes += fast_blocks_stake

    assert stakes == expected_stakes
    assert (
        subtensor.staking.get_stake_for_coldkey
        == subtensor.staking.get_stake_info_for_coldkey
    )

    stakes = subtensor.staking.get_stake_for_coldkey_and_hotkey(
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

    stake = subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )
    logging.console.info(f"Alice stake before unstake: {stake}")

    # unstale all to check in later
    success = subtensor.staking.unstake(
        wallet=alice_wallet,
        netuid=alice_subnet_netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=stake,
        period=16,
    )

    assert success is True

    stake = subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )

    # all balances have been unstaked
    assert stake == Balance(0).set_unit(alice_subnet_netuid)
    logging.console.success("✅ Test [green]test_single_operation[/green] passed")


@pytest.mark.asyncio
async def test_single_operation_async(async_subtensor, alice_wallet, bob_wallet):
    """
    Async ests:
    - Staking using `add_stake`
    - Unstaking using `unstake`
    - Checks StakeInfo
    """
    logging.console.info("Testing [blue]test_single_operation_async[/blue]")

    alice_subnet_netuid = await async_subtensor.subnets.get_total_subnets()  # 2

    # Register root as Alice - the subnet owner and validator
    assert await async_subtensor.subnets.register_subnet(alice_wallet).success

    # Verify subnet <netuid> created successfully
    assert await async_subtensor.subnets.subnet_exists(alice_subnet_netuid), (
        "Subnet wasn't created successfully"
    )

    assert await async_wait_to_start_call(
        async_subtensor, alice_wallet, alice_subnet_netuid
    )

    assert (
        await async_subtensor.subnets.burned_register(
            wallet=alice_wallet,
            netuid=alice_subnet_netuid,
        )
    ).success
    logging.console.success(f"Alice is registered in subnet {alice_subnet_netuid}")
    assert (
        await async_subtensor.subnets.burned_register(
            wallet=bob_wallet,
            netuid=alice_subnet_netuid,
        )
    ).success
    logging.console.success(f"Bob is registered in subnet {alice_subnet_netuid}")

    stake = await async_subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )

    assert stake == Balance(0).set_unit(alice_subnet_netuid)

    success = await async_subtensor.staking.add_stake(
        wallet=alice_wallet,
        netuid=alice_subnet_netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=Balance.from_tao(1),
        period=16,
    )

    assert success is True

    stake_alice = await async_subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )
    logging.console.info(f"Alice stake: {stake_alice}")

    stake_bob = await async_subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )

    logging.console.info(f"Bob stake: {stake_bob}")
    assert stake_bob > Balance(0).set_unit(alice_subnet_netuid)

    stakes = await async_subtensor.staking.get_stake_for_coldkey(
        alice_wallet.coldkey.ss58_address
    )

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
        if await async_subtensor.chain.is_fast_blocks()
        else []
    )

    expected_stakes += fast_blocks_stake

    assert stakes == expected_stakes
    assert (
        async_subtensor.staking.get_stake_for_coldkey
        == async_subtensor.staking.get_stake_info_for_coldkey
    )

    stakes = await async_subtensor.staking.get_stake_for_coldkey_and_hotkey(
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

    stake = await async_subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )
    logging.console.info(f"Alice stake before unstake: {stake}")

    # unstale all to check in later
    success, message = await async_subtensor.staking.unstake_all(
        wallet=alice_wallet,
        netuid=alice_subnet_netuid,
        hotkey=bob_wallet.hotkey.ss58_address,
        period=16,
    )
    assert success is True, message

    stake = await async_subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )
    logging.console.info(f"Alice stake after unstake: {stake}")

    # all balances have been unstaked
    assert stake == Balance(0).set_unit(alice_subnet_netuid)

    logging.console.success("✅ Test [green]test_single_operation_async[/green] passed")


def test_batch_operations(subtensor, alice_wallet, bob_wallet):
    """
    Tests:
    - Staking using `add_stake_multiple`
    - Unstaking using `unstake_multiple`
    - Checks StakeInfo
    - Checks Accounts Balance
    """
    logging.console.info("Testing [blue]test_batch_operations[/blue]")

    netuids = [
        2,
        3,
    ]

    for _ in netuids:
        assert subtensor.subnets.register_subnet(alice_wallet).success

    # make sure we passed start_call limit for both subnets
    for netuid in netuids:
        assert wait_to_start_call(subtensor, alice_wallet, netuid)

    for netuid in netuids:
        assert subtensor.subnets.burned_register(
            wallet=bob_wallet,
            netuid=netuid,
        ).success

    for netuid in netuids:
        stake = subtensor.staking.get_stake(
            alice_wallet.coldkey.ss58_address,
            bob_wallet.hotkey.ss58_address,
            netuid=netuid,
        )

        assert stake == Balance(0).set_unit(netuid), f"netuid={netuid} stake={stake}"

    balances = subtensor.wallets.get_balances(
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

    success = subtensor.staking.add_stake_multiple(
        wallet=alice_wallet,
        hotkey_ss58s=[bob_wallet.hotkey.ss58_address for _ in netuids],
        netuids=netuids,
        amounts=[Balance.from_tao(10_000) for _ in netuids],
    )

    assert success is True

    stakes = [
        subtensor.staking.get_stake(
            coldkey_ss58=alice_wallet.coldkey.ss58_address,
            hotkey_ss58=bob_wallet.hotkey.ss58_address,
            netuid=netuid,
        )
        for netuid in netuids
    ]

    for netuid, stake in zip(netuids, stakes):
        assert stake > Balance(0).set_unit(netuid), f"netuid={netuid} stake={stake}"

    alice_balance -= len(netuids) * Balance.from_tao(10_000)

    balances = subtensor.wallets.get_balances(
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

    expected_fee_paid = Balance(0)
    for netuid in netuids:
        call = subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="remove_stake",
            call_params={
                "hotkey": bob_wallet.hotkey.ss58_address,
                "amount_unstaked": Balance.from_tao(100).rao,
                "netuid": netuid,
            },
        )
        payment_info = subtensor.substrate.get_payment_info(
            call, alice_wallet.coldkeypub
        )
        fee_alpha = Balance.from_rao(payment_info["partial_fee"]).set_unit(netuid)
        dynamic_info = subtensor.subnets.subnet(netuid)
        fee_tao = dynamic_info.alpha_to_tao(fee_alpha)
        expected_fee_paid += fee_tao

    success = subtensor.staking.unstake_multiple(
        wallet=alice_wallet,
        netuids=netuids,
        hotkey_ss58s=[bob_wallet.hotkey.ss58_address for _ in netuids],
        amounts=[Balance.from_tao(100) for _ in netuids],
    )

    assert success is True

    for netuid, old_stake in zip(netuids, stakes):
        stake = subtensor.staking.get_stake(
            coldkey_ss58=alice_wallet.coldkey.ss58_address,
            hotkey_ss58=bob_wallet.hotkey.ss58_address,
            netuid=netuid,
        )

        assert stake < old_stake, f"netuid={netuid} stake={stake}"

    balances = subtensor.wallets.get_balances(
        alice_wallet.coldkey.ss58_address,
        bob_wallet.coldkey.ss58_address,
    )

    assert CloseInValue(  # Make sure we are within 0.0001 TAO due to tx fees
        balances[bob_wallet.coldkey.ss58_address], Balance.from_rao(100_000)
    ) == Balance.from_tao(999_999.7994)

    assert balances[alice_wallet.coldkey.ss58_address] > alice_balance
    logging.console.success("✅ Test [green]test_batch_operations[/green] passed")


@pytest.mark.asyncio
async def test_batch_operations_async(async_subtensor, alice_wallet, bob_wallet):
    """
    Async tests:
    - Staking using `add_stake_multiple`
    - Unstaking using `unstake_multiple`
    - Checks StakeInfo
    - Checks Accounts Balance
    """
    logging.console.info("Testing [blue]test_batch_operations_async[/blue]")

    netuids = [
        2,
        3,
    ]

    for _ in netuids:
        assert (await async_subtensor.subnets.register_subnet(alice_wallet)).success

    # make sure we passed start_call limit for both subnets
    for netuid in netuids:
        assert await async_wait_to_start_call(async_subtensor, alice_wallet, netuid)

    for netuid in netuids:
        assert (
            await async_subtensor.subnets.burned_register(
                wallet=bob_wallet,
                netuid=netuid,
            )
        ).success

    for netuid in netuids:
        stake = await async_subtensor.staking.get_stake(
            coldkey_ss58=alice_wallet.coldkey.ss58_address,
            hotkey_ss58=bob_wallet.hotkey.ss58_address,
            netuid=netuid,
        )

        assert stake == Balance(0).set_unit(netuid), f"netuid={netuid} stake={stake}"

    balances = await async_subtensor.wallets.get_balances(
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

    success = await async_subtensor.staking.add_stake_multiple(
        alice_wallet,
        hotkey_ss58s=[bob_wallet.hotkey.ss58_address for _ in netuids],
        netuids=netuids,
        amounts=[Balance.from_tao(10_000) for _ in netuids],
    )

    assert success is True

    stakes = [
        await async_subtensor.staking.get_stake(
            alice_wallet.coldkey.ss58_address,
            bob_wallet.hotkey.ss58_address,
            netuid=netuid,
        )
        for netuid in netuids
    ]

    for netuid, stake in zip(netuids, stakes):
        assert stake > Balance(0).set_unit(netuid), f"netuid={netuid} stake={stake}"

    alice_balance -= len(netuids) * Balance.from_tao(10_000)

    balances = await async_subtensor.wallets.get_balances(
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

    expected_fee_paid = Balance(0)
    for netuid in netuids:
        call = await async_subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="remove_stake",
            call_params={
                "hotkey": bob_wallet.hotkey.ss58_address,
                "amount_unstaked": Balance.from_tao(100).rao,
                "netuid": netuid,
            },
        )
        payment_info = await async_subtensor.substrate.get_payment_info(
            call, alice_wallet.coldkeypub
        )
        fee_alpha = Balance.from_rao(payment_info["partial_fee"]).set_unit(netuid)
        dynamic_info = await async_subtensor.subnets.subnet(netuid)
        fee_tao = dynamic_info.alpha_to_tao(fee_alpha)
        expected_fee_paid += fee_tao

    success = await async_subtensor.staking.unstake_multiple(
        wallet=alice_wallet,
        netuids=netuids,
        hotkey_ss58s=[bob_wallet.hotkey.ss58_address for _ in netuids],
        amounts=[Balance.from_tao(100) for _ in netuids],
    )

    assert success is True

    for netuid, old_stake in zip(netuids, stakes):
        stake = await async_subtensor.staking.get_stake(
            coldkey_ss58=alice_wallet.coldkey.ss58_address,
            hotkey_ss58=bob_wallet.hotkey.ss58_address,
            netuid=netuid,
        )

        assert stake < old_stake, f"netuid={netuid} stake={stake}"

    balances = await async_subtensor.wallets.get_balances(
        alice_wallet.coldkey.ss58_address,
        bob_wallet.coldkey.ss58_address,
    )

    assert CloseInValue(  # Make sure we are within 0.0001 TAO due to tx fees
        balances[bob_wallet.coldkey.ss58_address], Balance.from_rao(100_000)
    ) == Balance.from_tao(999_999.7994)

    assert balances[alice_wallet.coldkey.ss58_address] > alice_balance
    logging.console.success("✅ Test [green]test_batch_operations_async[/green] passed")


def test_safe_staking_scenarios(subtensor, alice_wallet, bob_wallet, eve_wallet):
    """
    Tests safe staking scenarios with different parameters.

    For both staking and unstaking:
    1. Fails with strict threshold (0.5%) and no partial staking
    2. Succeeds with strict threshold (0.5%) and partial staking allowed
    3. Succeeds with lenient threshold (10% and 30%) and no partial staking
    """
    logging.console.info("Testing [blue]test_safe_staking_scenarios[/blue]")


    # turn off admin freeze window limit for testing
    assert (
        sudo_set_admin_utils(
            substrate=subtensor.substrate,
            wallet=alice_wallet,
            call_function="sudo_set_admin_freeze_window",
            call_params={"window": 0},
        )[0]
        is True
    ), "Failed to set admin freeze window to 0"

    alice_subnet_netuid = subtensor.subnets.get_total_subnets()  # 2
    # Register root as Alice - the subnet owner and validator
    assert subtensor.extrinsics.register_subnet(alice_wallet)

    # Verify subnet created successfully
    assert subtensor.subnets.subnet_exists(alice_subnet_netuid), (
        "Subnet wasn't created successfully"
    )

    # Change the tempo of the subnet
    TEMPO_TO_SET = 100 if subtensor.chain.is_fast_blocks() else 20
    assert (
        sudo_set_admin_utils(
            substrate=subtensor.substrate,
            wallet=alice_wallet,
            call_function="sudo_set_tempo",
            call_params={"netuid": alice_subnet_netuid, "tempo": TEMPO_TO_SET},
        )[0]
        is True
    )
    tempo = subtensor.subnets.get_subnet_hyperparameters(
        netuid=alice_subnet_netuid
    ).tempo
    assert tempo == TEMPO_TO_SET, "SN tempos has not been changed."
    logging.console.success(f"SN #{alice_subnet_netuid} tempo set to {TEMPO_TO_SET}")

    assert wait_to_start_call(subtensor, alice_wallet, alice_subnet_netuid)

    assert subtensor.extrinsics.burned_register(
        wallet=bob_wallet,
        netuid=alice_subnet_netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    ).success

    initial_stake = subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )
    assert initial_stake == Balance(0).set_unit(alice_subnet_netuid)
    logging.console.info(f"[orange]Initial stake: {initial_stake}[orange]")

    # Test Staking Scenarios
    stake_amount = Balance.from_tao(100)

    # 1. Strict params - should fail
    success = subtensor.staking.add_stake(
        wallet=alice_wallet,
        netuid=alice_subnet_netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=stake_amount,
        safe_staking=True,
        rate_tolerance=0.005,  # 0.5%
        allow_partial_stake=False,
    )
    assert success is False

    current_stake = subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )
    assert current_stake == Balance(0).set_unit(alice_subnet_netuid), (
        "Stake should not change after failed attempt"
    )
    logging.console.info(f"[orange]Current stake: {current_stake}[orange]")

    # 2. Partial allowed - should succeed partially
    success = subtensor.staking.add_stake(
        wallet=alice_wallet,
        netuid=alice_subnet_netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=stake_amount,
        safe_staking=True,
        rate_tolerance=0.005,  # 0.5%
        allow_partial_stake=True,
    )
    assert success is True

    partial_stake = subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )
    assert partial_stake > Balance(0).set_unit(alice_subnet_netuid), (
        "Partial stake should be added"
    )
    assert partial_stake < stake_amount, (
        "Partial stake should be less than requested amount"
    )

    # 3. Higher threshold - should succeed fully
    amount = Balance.from_tao(100)
    success = subtensor.staking.add_stake(
        wallet=alice_wallet,
        netuid=alice_subnet_netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=amount,
        safe_staking=True,
        rate_tolerance=0.22,  # 22%
        allow_partial_stake=False,
    )
    assert success is True

    full_stake = subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )

    # Test Unstaking Scenarios
    # 1. Strict params - should fail
    success = subtensor.staking.unstake(
        wallet=alice_wallet,
        netuid=alice_subnet_netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=full_stake,
        safe_unstaking=True,
        rate_tolerance=0.005,  # 0.5%
        allow_partial_stake=False,
    )
    assert success is False, "Unstake should fail."

    current_stake = subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )

    logging.console.info(f"[orange]Current stake: {current_stake}[orange]")
    logging.console.info(f"[orange]Full stake: {full_stake}[orange]")

    assert current_stake == full_stake, (
        "Stake should not change after failed unstake attempt"
    )

    # 2. Partial allowed - should succeed partially
    success = subtensor.staking.unstake(
        wallet=alice_wallet,
        netuid=alice_subnet_netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=current_stake,
        safe_unstaking=True,
        rate_tolerance=0.005,  # 0.5%
        allow_partial_stake=True,
    )
    assert success is True

    partial_unstake = subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )
    logging.console.info(f"[orange]Partial unstake: {partial_unstake}[orange]")
    assert partial_unstake > Balance(0).set_unit(alice_subnet_netuid), (
        "Some stake should remain"
    )

    # 3. Higher threshold - should succeed fully
    success = subtensor.staking.unstake(
        wallet=alice_wallet,
        netuid=alice_subnet_netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=partial_unstake,
        safe_unstaking=True,
        rate_tolerance=0.3,  # 30%
        allow_partial_stake=False,
    )
    assert success is True, "Unstake should succeed"
    logging.console.success("✅ Test [green]test_safe_staking_scenarios[/green] passed")


@pytest.mark.asyncio
async def test_safe_staking_scenarios_async(
    async_subtensor, alice_wallet, bob_wallet, eve_wallet
):
    """
    Tests safe staking scenarios with different parameters.

    For both staking and unstaking:
    1. Fails with strict threshold (0.5%) and no partial staking
    2. Succeeds with strict threshold (0.5%) and partial staking allowed
    3. Succeeds with lenient threshold (10% and 30%) and no partial staking
    """
    logging.console.info("Testing [blue]test_safe_staking_scenarios_async[/blue]")

    alice_subnet_netuid = await async_subtensor.subnets.get_total_subnets()  # 2
    # Register root as Alice - the subnet owner and validator
    assert await async_subtensor.extrinsics.register_subnet(alice_wallet)

    # Verify subnet created successfully
    assert await async_subtensor.subnets.subnet_exists(alice_subnet_netuid), (
        "Subnet wasn't created successfully"
    )

    # Change the tempo of the subnet
    TEMPO_TO_SET = 100 if await async_subtensor.chain.is_fast_blocks() else 20
    assert (
        await async_sudo_set_admin_utils(
            substrate=async_subtensor.substrate,
            wallet=alice_wallet,
            call_function="sudo_set_tempo",
            call_params={"netuid": alice_subnet_netuid, "tempo": TEMPO_TO_SET},
        )
    )[0] is True
    tempo = (
        await async_subtensor.subnets.get_subnet_hyperparameters(
            netuid=alice_subnet_netuid
        )
    ).tempo
    assert tempo == TEMPO_TO_SET, "SN tempos has not been changed."
    logging.console.success(f"SN #{alice_subnet_netuid} tempo set to {TEMPO_TO_SET}")

    assert await async_wait_to_start_call(
        async_subtensor, alice_wallet, alice_subnet_netuid
    )

    assert (
        await async_subtensor.extrinsics.burned_register(
            wallet=bob_wallet,
            netuid=alice_subnet_netuid,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
    ).success

    initial_stake = await async_subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )
    assert initial_stake == Balance(0).set_unit(alice_subnet_netuid)
    logging.console.info(f"[orange]Initial stake: {initial_stake}[orange]")

    # Test Staking Scenarios
    stake_amount = Balance.from_tao(100)

    # 1. Strict params - should fail
    success = await async_subtensor.staking.add_stake(
        wallet=alice_wallet,
        netuid=alice_subnet_netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=stake_amount,
        safe_staking=True,
        rate_tolerance=0.005,  # 0.5%
        allow_partial_stake=False,
    )
    assert success is False

    current_stake = await async_subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )
    assert current_stake == Balance(0).set_unit(alice_subnet_netuid), (
        "Stake should not change after failed attempt"
    )
    logging.console.info(f"[orange]Current stake: {current_stake}[orange]")

    # 2. Partial allowed - should succeed partially
    success = await async_subtensor.staking.add_stake(
        wallet=alice_wallet,
        netuid=alice_subnet_netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=stake_amount,
        safe_staking=True,
        rate_tolerance=0.005,  # 0.5%
        allow_partial_stake=True,
    )
    assert success is True

    partial_stake = await async_subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )
    assert partial_stake > Balance(0).set_unit(alice_subnet_netuid), (
        "Partial stake should be added"
    )
    assert partial_stake < stake_amount, (
        "Partial stake should be less than requested amount"
    )

    # 3. Higher threshold - should succeed fully
    amount = Balance.from_tao(100)
    success = await async_subtensor.staking.add_stake(
        wallet=alice_wallet,
        netuid=alice_subnet_netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=amount,
        safe_staking=True,
        rate_tolerance=0.22,  # 22%
        allow_partial_stake=False,
    )
    assert success is True

    full_stake = await async_subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )

    # Test Unstaking Scenarios
    # 1. Strict params - should fail
    success = await async_subtensor.staking.unstake(
        wallet=alice_wallet,
        netuid=alice_subnet_netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=full_stake,
        safe_unstaking=True,
        rate_tolerance=0.005,  # 0.5%
        allow_partial_stake=False,
    )
    assert success is False, "Unstake should fail."

    current_stake = await async_subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )

    logging.console.info(f"[orange]Current stake: {current_stake}[orange]")
    logging.console.info(f"[orange]Full stake: {full_stake}[orange]")

    assert current_stake == full_stake, (
        "Stake should not change after failed unstake attempt"
    )

    # 2. Partial allowed - should succeed partially
    success = await async_subtensor.staking.unstake(
        wallet=alice_wallet,
        netuid=alice_subnet_netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=current_stake,
        safe_unstaking=True,
        rate_tolerance=0.005,  # 0.5%
        allow_partial_stake=True,
    )
    assert success is True

    partial_unstake = await async_subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid,
    )
    logging.console.info(f"[orange]Partial unstake: {partial_unstake}[orange]")
    assert partial_unstake > Balance(0).set_unit(alice_subnet_netuid), (
        "Some stake should remain"
    )

    # 3. Higher threshold - should succeed fully
    success = await async_subtensor.staking.unstake(
        wallet=alice_wallet,
        netuid=alice_subnet_netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=partial_unstake,
        safe_unstaking=True,
        rate_tolerance=0.3,  # 30%
        allow_partial_stake=False,
    )
    assert success is True, "Unstake should succeed"
    logging.console.success(
        "✅ Test [green]test_safe_staking_scenarios_async[/green] passed"
    )


def test_safe_swap_stake_scenarios(subtensor, alice_wallet, bob_wallet):
    """
    Tests safe swap stake scenarios with different parameters.

    Tests:
    1. Fails with strict threshold (0.5%)
    2. Succeeds with lenient threshold (10%)
    """
    logging.console.info("Testing [blue]test_safe_swap_stake_scenarios[/blue]")

    # Create new subnet (netuid 2) and register Alice
    origin_netuid = 2
    assert subtensor.subnets.register_subnet(bob_wallet).success
    assert subtensor.subnets.subnet_exists(origin_netuid), (
        "Subnet wasn't created successfully"
    )
    dest_netuid = 3
    assert subtensor.subnets.register_subnet(bob_wallet).success
    assert subtensor.subnets.subnet_exists(dest_netuid), (
        "Subnet wasn't created successfully"
    )

    # make sure we passed start_call limit for both subnets
    assert wait_to_start_call(subtensor, bob_wallet, origin_netuid)
    assert wait_to_start_call(subtensor, bob_wallet, dest_netuid)

    # Register Alice on both subnets
    assert subtensor.subnets.burned_register(
        wallet=alice_wallet,
        netuid=origin_netuid,
    ).success
    assert subtensor.subnets.burned_register(
        wallet=alice_wallet,
        netuid=dest_netuid,
    ).success

    # Add initial stake to swap from
    initial_stake_amount = Balance.from_tao(10_000)
    success = subtensor.staking.add_stake(
        wallet=alice_wallet,
        netuid=origin_netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        amount=initial_stake_amount,
    )
    assert success is True

    origin_stake = subtensor.staking.get_stake(
        alice_wallet.coldkey.ss58_address,
        alice_wallet.hotkey.ss58_address,
        netuid=origin_netuid,
    )
    assert origin_stake > Balance(0).set_unit(origin_netuid), (
        "Origin stake should be non-zero"
    )

    stake_swap_amount = Balance.from_tao(10_000)
    # 1. Try swap with strict threshold and big amount- should fail
    response = subtensor.staking.swap_stake(
        wallet=alice_wallet,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        origin_netuid=origin_netuid,
        destination_netuid=dest_netuid,
        amount=stake_swap_amount,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        safe_swapping=True,
        rate_tolerance=0.005,  # 0.5%
        allow_partial_stake=False,
    )
    assert response.success is False

    # Verify no stake was moved
    dest_stake = subtensor.staking.get_stake(
        alice_wallet.coldkey.ss58_address,
        alice_wallet.hotkey.ss58_address,
        netuid=dest_netuid,
    )
    assert dest_stake == Balance(0).set_unit(dest_netuid), (
        "Destination stake should remain 0 after failed swap"
    )

    # 2. Try swap with higher threshold and less amount - should succeed
    stake_swap_amount = Balance.from_tao(100)
    response = subtensor.staking.swap_stake(
        wallet=alice_wallet,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        origin_netuid=origin_netuid,
        destination_netuid=dest_netuid,
        amount=stake_swap_amount,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        safe_swapping=True,
        rate_tolerance=0.3,  # 30%
        allow_partial_stake=True,
    )
    assert response.success is True

    # Verify stake was moved
    dest_stake = subtensor.staking.get_stake(
        alice_wallet.coldkey.ss58_address,
        alice_wallet.hotkey.ss58_address,
        netuid=dest_netuid,
    )
    assert dest_stake > Balance(0).set_unit(dest_netuid), (
        "Destination stake should be non-zero after successful swap"
    )
    logging.console.success(
        "✅ Test [green]test_safe_swap_stake_scenarios[/green] passed"
    )


@pytest.mark.asyncio
async def test_safe_swap_stake_scenarios_async(
    async_subtensor, alice_wallet, bob_wallet
):
    """
    Tests safe swap stake scenarios with different parameters.

    Tests:
    1. Fails with strict threshold (0.5%)
    2. Succeeds with lenient threshold (10%)
    """
    logging.console.info("Testing [blue]test_safe_swap_stake_scenarios_async[/blue]")

    # Create new subnet (netuid 2) and register Alice
    origin_netuid = 2
    assert (await async_subtensor.subnets.register_subnet(bob_wallet)).success
    assert await async_subtensor.subnets.subnet_exists(origin_netuid), (
        "Subnet wasn't created successfully"
    )
    dest_netuid = 3
    assert (await async_subtensor.subnets.register_subnet(bob_wallet)).success
    assert await async_subtensor.subnets.subnet_exists(dest_netuid), (
        "Subnet wasn't created successfully"
    )

    # make sure we passed start_call limit for both subnets
    assert await async_wait_to_start_call(async_subtensor, bob_wallet, origin_netuid)
    assert await async_wait_to_start_call(async_subtensor, bob_wallet, dest_netuid)

    # Register Alice on both subnets
    assert (
        await async_subtensor.subnets.burned_register(
            wallet=alice_wallet,
            netuid=origin_netuid,
        )
    ).success
    assert (
        await async_subtensor.subnets.burned_register(
            wallet=alice_wallet,
            netuid=dest_netuid,
        )
    ).success

    # Add initial stake to swap from
    initial_stake_amount = Balance.from_tao(10_000)
    success = await async_subtensor.staking.add_stake(
        wallet=alice_wallet,
        netuid=origin_netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        amount=initial_stake_amount,
    )
    assert success is True

    origin_stake = await async_subtensor.staking.get_stake(
        alice_wallet.coldkey.ss58_address,
        alice_wallet.hotkey.ss58_address,
        netuid=origin_netuid,
    )
    assert origin_stake > Balance(0).set_unit(origin_netuid), (
        "Origin stake should be non-zero"
    )

    stake_swap_amount = Balance.from_tao(10_000)
    # 1. Try swap with strict threshold and big amount- should fail
    response = await async_subtensor.staking.swap_stake(
        wallet=alice_wallet,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        origin_netuid=origin_netuid,
        destination_netuid=dest_netuid,
        amount=stake_swap_amount,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        safe_swapping=True,
        rate_tolerance=0.005,  # 0.5%
        allow_partial_stake=False,
    )
    assert response.success is False

    # Verify no stake was moved
    dest_stake = await async_subtensor.staking.get_stake(
        alice_wallet.coldkey.ss58_address,
        alice_wallet.hotkey.ss58_address,
        netuid=dest_netuid,
    )
    assert dest_stake == Balance(0).set_unit(dest_netuid), (
        "Destination stake should remain 0 after failed swap"
    )

    # 2. Try swap with higher threshold and less amount - should succeed
    stake_swap_amount = Balance.from_tao(100)
    response = await async_subtensor.staking.swap_stake(
        wallet=alice_wallet,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        origin_netuid=origin_netuid,
        destination_netuid=dest_netuid,
        amount=stake_swap_amount,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        safe_swapping=True,
        rate_tolerance=0.3,  # 30%
        allow_partial_stake=True,
    )
    assert response.success is True

    # Verify stake was moved
    dest_stake = await async_subtensor.staking.get_stake(
        alice_wallet.coldkey.ss58_address,
        alice_wallet.hotkey.ss58_address,
        netuid=dest_netuid,
    )
    assert dest_stake > Balance(0).set_unit(dest_netuid), (
        "Destination stake should be non-zero after successful swap"
    )
    logging.console.success(
        "✅ Test [green]test_safe_swap_stake_scenarios_async[/green] passed"
    )


def test_move_stake(subtensor, alice_wallet, bob_wallet, dave_wallet):
    """
    Tests:
    - Adding stake
    - Moving stake from one hotkey-subnet pair to another
    - Testing `move_stake` method with `move_all_stake=True` flag.
    """
    logging.console.info("Testing [blue]test_move_stake[/blue]")

    alice_subnet_netuid = subtensor.subnets.get_total_subnets()  # 2
    assert subtensor.subnets.register_subnet(alice_wallet).success
    assert subtensor.subnets.subnet_exists(alice_subnet_netuid), (
        "Subnet wasn't created successfully"
    )

    assert wait_to_start_call(subtensor, alice_wallet, alice_subnet_netuid)

    assert subtensor.staking.add_stake(
        wallet=alice_wallet,
        netuid=alice_subnet_netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        amount=Balance.from_tao(1_000),
    )

    stakes = subtensor.staking.get_stake_for_coldkey(alice_wallet.coldkey.ss58_address)

    assert stakes == [
        StakeInfo(
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            coldkey_ss58=alice_wallet.coldkey.ss58_address,
            netuid=alice_subnet_netuid,
            stake=get_dynamic_balance(stakes[0].stake.rao, alice_subnet_netuid),
            locked=Balance(0).set_unit(alice_subnet_netuid),
            emission=get_dynamic_balance(stakes[0].emission.rao, alice_subnet_netuid),
            drain=0,
            is_registered=True,
        ),
    ]

    bob_subnet_netuid = subtensor.subnets.get_total_subnets()  # 3
    assert subtensor.subnets.register_subnet(bob_wallet).success
    assert subtensor.subnets.subnet_exists(bob_subnet_netuid), (
        "Subnet wasn't created successfully"
    )

    assert wait_to_start_call(subtensor, bob_wallet, bob_subnet_netuid)

    assert subtensor.subnets.burned_register(
        wallet=bob_wallet,
        netuid=alice_subnet_netuid,
    ).success

    assert subtensor.subnets.burned_register(
        wallet=dave_wallet,
        netuid=alice_subnet_netuid,
    ).success

    response = subtensor.staking.move_stake(
        wallet=alice_wallet,
        origin_hotkey_ss58=alice_wallet.hotkey.ss58_address,
        origin_netuid=alice_subnet_netuid,
        destination_hotkey_ss58=bob_wallet.hotkey.ss58_address,
        destination_netuid=bob_subnet_netuid,
        amount=stakes[0].stake,
        wait_for_finalization=True,
        wait_for_inclusion=True,
    )
    assert response.success is True

    stakes = subtensor.staking.get_stake_for_coldkey(alice_wallet.coldkey.ss58_address)

    expected_stakes = [
        StakeInfo(
            hotkey_ss58=stakes[0].hotkey_ss58,
            coldkey_ss58=alice_wallet.coldkey.ss58_address,
            netuid=alice_subnet_netuid
            if subtensor.chain.is_fast_blocks()
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
        if subtensor.chain.is_fast_blocks()
        else []
    )

    expected_stakes += fast_block_stake
    assert stakes == expected_stakes

    # test move_stake with move_all_stake=True
    dave_stake = subtensor.staking.get_stake(
        coldkey_ss58=dave_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=bob_subnet_netuid,
    )
    logging.console.info(f"[orange]Dave stake before adding: {dave_stake}[orange]")

    assert subtensor.staking.add_stake(
        wallet=dave_wallet,
        netuid=bob_subnet_netuid,
        hotkey_ss58=dave_wallet.hotkey.ss58_address,
        amount=Balance.from_tao(1000),
        allow_partial_stake=True,
    )

    dave_stake = subtensor.staking.get_stake(
        coldkey_ss58=dave_wallet.coldkey.ss58_address,
        hotkey_ss58=dave_wallet.hotkey.ss58_address,
        netuid=bob_subnet_netuid,
    )
    logging.console.info(f"[orange]Dave stake after adding: {dave_stake}[orange]")

    # let chain to process the transaction
    subtensor.wait_for_block(
        subtensor.block + subtensor.subnets.tempo(netuid=bob_subnet_netuid)
    )

    response = subtensor.staking.move_stake(
        wallet=dave_wallet,
        origin_hotkey_ss58=dave_wallet.hotkey.ss58_address,
        origin_netuid=bob_subnet_netuid,
        destination_hotkey_ss58=bob_wallet.hotkey.ss58_address,
        destination_netuid=bob_subnet_netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        move_all_stake=True,
    )
    assert response.success is True

    dave_stake = subtensor.staking.get_stake(
        coldkey_ss58=dave_wallet.coldkey.ss58_address,
        hotkey_ss58=dave_wallet.hotkey.ss58_address,
        netuid=bob_subnet_netuid,
    )
    logging.console.info(f"[orange]Dave stake after moving all: {dave_stake}[orange]")

    assert dave_stake.rao == CloseInValue(0, 0.00001)

    logging.console.success("✅ Test [green]test_move_stake[/green] passed.")


@pytest.mark.asyncio
async def test_move_stake_async(async_subtensor, alice_wallet, bob_wallet, dave_wallet):
    """
    Tests:
    - Adding stake
    - Moving stake from one hotkey-subnet pair to another
    - Testing `move_stake` method with `move_all_stake=True` flag.
    """
    logging.console.info("Testing [blue]test_move_stake_async[/blue]")

    alice_subnet_netuid = await async_subtensor.subnets.get_total_subnets()  # 2
    assert (await async_subtensor.subnets.register_subnet(alice_wallet)).success
    assert await async_subtensor.subnets.subnet_exists(alice_subnet_netuid), (
        "Subnet wasn't created successfully"
    )

    assert await async_wait_to_start_call(
        async_subtensor, alice_wallet, alice_subnet_netuid
    )

    assert await async_subtensor.staking.add_stake(
        wallet=alice_wallet,
        netuid=alice_subnet_netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        amount=Balance.from_tao(1_000),
    )

    stakes = await async_subtensor.staking.get_stake_for_coldkey(
        alice_wallet.coldkey.ss58_address
    )

    assert stakes == [
        StakeInfo(
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            coldkey_ss58=alice_wallet.coldkey.ss58_address,
            netuid=alice_subnet_netuid,
            stake=get_dynamic_balance(stakes[0].stake.rao, alice_subnet_netuid),
            locked=Balance(0).set_unit(alice_subnet_netuid),
            emission=get_dynamic_balance(stakes[0].emission.rao, alice_subnet_netuid),
            drain=0,
            is_registered=True,
        ),
    ]

    bob_subnet_netuid = await async_subtensor.subnets.get_total_subnets()  # 3
    assert (await async_subtensor.subnets.register_subnet(bob_wallet)).success
    assert await async_subtensor.subnets.subnet_exists(bob_subnet_netuid), (
        "Subnet wasn't created successfully"
    )

    assert await async_wait_to_start_call(
        async_subtensor, bob_wallet, bob_subnet_netuid
    )

    assert (
        await async_subtensor.subnets.burned_register(
            wallet=bob_wallet,
            netuid=alice_subnet_netuid,
        )
    ).success

    assert (
        await async_subtensor.subnets.burned_register(
            wallet=dave_wallet,
            netuid=alice_subnet_netuid,
        )
    ).success

    response = await async_subtensor.staking.move_stake(
        wallet=alice_wallet,
        origin_hotkey_ss58=alice_wallet.hotkey.ss58_address,
        origin_netuid=alice_subnet_netuid,
        destination_hotkey_ss58=bob_wallet.hotkey.ss58_address,
        destination_netuid=bob_subnet_netuid,
        amount=stakes[0].stake,
        wait_for_finalization=True,
        wait_for_inclusion=True,
    )
    assert response.success is True

    stakes = await async_subtensor.staking.get_stake_for_coldkey(
        alice_wallet.coldkey.ss58_address
    )

    expected_stakes = [
        StakeInfo(
            hotkey_ss58=stakes[0].hotkey_ss58,
            coldkey_ss58=alice_wallet.coldkey.ss58_address,
            netuid=alice_subnet_netuid
            if await async_subtensor.chain.is_fast_blocks()
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
        if await async_subtensor.chain.is_fast_blocks()
        else []
    )

    expected_stakes += fast_block_stake
    assert stakes == expected_stakes

    # test move_stake with move_all_stake=True
    dave_stake = await async_subtensor.staking.get_stake(
        coldkey_ss58=dave_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=bob_subnet_netuid,
    )
    logging.console.info(f"[orange]Dave stake before adding: {dave_stake}[orange]")

    assert await async_subtensor.staking.add_stake(
        wallet=dave_wallet,
        hotkey_ss58=dave_wallet.hotkey.ss58_address,
        netuid=bob_subnet_netuid,
        amount=Balance.from_tao(1000),
        allow_partial_stake=True,
    )

    dave_stake = await async_subtensor.staking.get_stake(
        coldkey_ss58=dave_wallet.coldkey.ss58_address,
        hotkey_ss58=dave_wallet.hotkey.ss58_address,
        netuid=bob_subnet_netuid,
    )
    logging.console.info(f"[orange]Dave stake after adding: {dave_stake}[orange]")

    block_, tampo_ = await asyncio.gather(
        async_subtensor.block, async_subtensor.subnets.tempo(netuid=bob_subnet_netuid)
    )
    # let chain to process the transaction
    await async_subtensor.wait_for_block(block_ + tampo_)

    response = await async_subtensor.staking.move_stake(
        wallet=dave_wallet,
        origin_hotkey_ss58=dave_wallet.hotkey.ss58_address,
        origin_netuid=bob_subnet_netuid,
        destination_hotkey_ss58=bob_wallet.hotkey.ss58_address,
        destination_netuid=bob_subnet_netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        move_all_stake=True,
    )
    assert response.success is True

    dave_stake = await async_subtensor.staking.get_stake(
        coldkey_ss58=dave_wallet.coldkey.ss58_address,
        hotkey_ss58=dave_wallet.hotkey.ss58_address,
        netuid=bob_subnet_netuid,
    )
    logging.console.info(f"[orange]Dave stake after moving all: {dave_stake}[orange]")

    assert dave_stake.rao == CloseInValue(0, 0.00001)

    logging.console.success("✅ Test [green]test_move_stake_async[/green] passed.")


def test_transfer_stake(subtensor, alice_wallet, bob_wallet, dave_wallet):
    """
    Tests:
    - Adding stake
    - Transferring stake from one coldkey-subnet pair to another
    """
    logging.console.info("Testing [blue]test_transfer_stake[/blue]")

    alice_subnet_netuid = subtensor.subnets.get_total_subnets()  # 2

    assert subtensor.subnets.register_subnet(alice_wallet).success
    assert subtensor.subnets.subnet_exists(alice_subnet_netuid), (
        "Subnet wasn't created successfully"
    )

    assert wait_to_start_call(subtensor, alice_wallet, alice_subnet_netuid)

    assert subtensor.subnets.burned_register(
        wallet=alice_wallet,
        netuid=alice_subnet_netuid,
    ).success

    assert subtensor.staking.add_stake(
        wallet=alice_wallet,
        netuid=alice_subnet_netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        amount=Balance.from_tao(1_000),
    )

    alice_stakes = subtensor.staking.get_stake_for_coldkey(
        alice_wallet.coldkey.ss58_address
    )

    assert alice_stakes == [
        StakeInfo(
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            coldkey_ss58=alice_wallet.coldkey.ss58_address,
            netuid=alice_subnet_netuid,
            stake=get_dynamic_balance(alice_stakes[0].stake.rao, alice_subnet_netuid),
            locked=Balance(0).set_unit(alice_subnet_netuid),
            emission=get_dynamic_balance(
                alice_stakes[0].emission.rao, alice_subnet_netuid
            ),
            drain=0,
            is_registered=True,
        ),
    ]

    bob_stakes = subtensor.staking.get_stake_for_coldkey(
        bob_wallet.coldkey.ss58_address
    )

    assert bob_stakes == []

    dave_subnet_netuid = subtensor.subnets.get_total_subnets()  # 3
    assert subtensor.subnets.register_subnet(dave_wallet).success

    assert wait_to_start_call(subtensor, dave_wallet, dave_subnet_netuid)

    assert subtensor.subnets.burned_register(
        wallet=bob_wallet,
        netuid=dave_subnet_netuid,
    ).success

    response = subtensor.staking.transfer_stake(
        alice_wallet,
        destination_coldkey_ss58=bob_wallet.coldkey.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        origin_netuid=alice_subnet_netuid,
        destination_netuid=dave_subnet_netuid,
        amount=alice_stakes[0].stake,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response.success is True

    alice_stakes = subtensor.staking.get_stake_for_coldkey(
        alice_wallet.coldkey.ss58_address
    )

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
        if subtensor.chain.is_fast_blocks()
        else []
    )

    assert alice_stakes == expected_alice_stake

    bob_stakes = subtensor.staking.get_stake_for_coldkey(
        bob_wallet.coldkey.ss58_address
    )

    expected_bob_stake = [
        StakeInfo(
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            coldkey_ss58=bob_wallet.coldkey.ss58_address,
            netuid=dave_subnet_netuid,
            stake=get_dynamic_balance(bob_stakes[0].stake.rao, dave_subnet_netuid),
            locked=Balance(0).set_unit(dave_subnet_netuid),
            emission=get_dynamic_balance(
                bob_stakes[0].emission.rao, dave_subnet_netuid
            ),
            drain=0,
            is_registered=False,
        ),
    ]
    assert bob_stakes == expected_bob_stake
    logging.console.success("✅ Test [green]test_transfer_stake[/green] passed")


@pytest.mark.asyncio
async def test_transfer_stake_async(
    async_subtensor, alice_wallet, bob_wallet, dave_wallet
):
    """
    Tests:
    - Adding stake
    - Transferring stake from one coldkey-subnet pair to another
    """
    logging.console.info("Testing [blue]test_transfer_stake_async[/blue]")

    alice_subnet_netuid = await async_subtensor.subnets.get_total_subnets()  # 2

    assert (await async_subtensor.subnets.register_subnet(alice_wallet)).success
    assert await async_subtensor.subnets.subnet_exists(alice_subnet_netuid), (
        "Subnet wasn't created successfully"
    )

    assert await async_wait_to_start_call(
        async_subtensor, alice_wallet, alice_subnet_netuid
    )

    assert (
        await async_subtensor.subnets.burned_register(
            wallet=alice_wallet,
            netuid=alice_subnet_netuid,
        )
    ).success

    assert await async_subtensor.staking.add_stake(
        wallet=alice_wallet,
        netuid=alice_subnet_netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        amount=Balance.from_tao(1_000),
    )

    alice_stakes = await async_subtensor.staking.get_stake_for_coldkey(
        alice_wallet.coldkey.ss58_address
    )

    assert alice_stakes == [
        StakeInfo(
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            coldkey_ss58=alice_wallet.coldkey.ss58_address,
            netuid=alice_subnet_netuid,
            stake=get_dynamic_balance(alice_stakes[0].stake.rao, alice_subnet_netuid),
            locked=Balance(0).set_unit(alice_subnet_netuid),
            emission=get_dynamic_balance(
                alice_stakes[0].emission.rao, alice_subnet_netuid
            ),
            drain=0,
            is_registered=True,
        ),
    ]

    bob_stakes = await async_subtensor.staking.get_stake_for_coldkey(
        bob_wallet.coldkey.ss58_address
    )

    assert bob_stakes == []

    dave_subnet_netuid = await async_subtensor.subnets.get_total_subnets()  # 3
    assert (await async_subtensor.subnets.register_subnet(dave_wallet)).success

    assert await async_wait_to_start_call(
        async_subtensor, dave_wallet, dave_subnet_netuid
    )

    assert (
        await async_subtensor.subnets.burned_register(
            wallet=bob_wallet,
            netuid=dave_subnet_netuid,
        )
    ).success

    response = await async_subtensor.staking.transfer_stake(
        alice_wallet,
        destination_coldkey_ss58=bob_wallet.coldkey.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        origin_netuid=alice_subnet_netuid,
        destination_netuid=dave_subnet_netuid,
        amount=alice_stakes[0].stake,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response.success is True

    alice_stakes = await async_subtensor.staking.get_stake_for_coldkey(
        alice_wallet.coldkey.ss58_address
    )

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
        if await async_subtensor.chain.is_fast_blocks()
        else []
    )

    assert alice_stakes == expected_alice_stake

    bob_stakes = await async_subtensor.staking.get_stake_for_coldkey(
        bob_wallet.coldkey.ss58_address
    )

    expected_bob_stake = [
        StakeInfo(
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            coldkey_ss58=bob_wallet.coldkey.ss58_address,
            netuid=dave_subnet_netuid,
            stake=get_dynamic_balance(bob_stakes[0].stake.rao, dave_subnet_netuid),
            locked=Balance(0).set_unit(dave_subnet_netuid),
            emission=get_dynamic_balance(
                bob_stakes[0].emission.rao, dave_subnet_netuid
            ),
            drain=0,
            is_registered=False,
        ),
    ]
    assert bob_stakes == expected_bob_stake
    logging.console.success("✅ Test [green]test_transfer_stake_async[/green] passed")


# For test we set rate_tolerance=0.7 (70%) because of price is highly dynamic for fast-blocks and 2 SN to avoid `
# Slippage is too high for the transaction`. This logic controls by the chain.
# Also this test implementation works with non-fast-blocks run.
@pytest.mark.parametrize(
    "rate_tolerance",
    [None, 1.0],
    ids=[
        "Without price limit",
        "With price limit",
    ],
)
def test_unstaking_with_limit(
    subtensor, alice_wallet, bob_wallet, dave_wallet, rate_tolerance
):
    """Test unstaking with limits goes well for all subnets with and without price limit."""
    logging.console.info("Testing [blue]test_unstaking_with_limit[/blue]")

    # Register first SN
    alice_subnet_netuid_2 = subtensor.subnets.get_total_subnets()  # 2
    assert subtensor.subnets.register_subnet(alice_wallet).success
    assert subtensor.subnets.subnet_exists(alice_subnet_netuid_2), (
        "Subnet wasn't created successfully"
    )

    wait_to_start_call(subtensor, alice_wallet, alice_subnet_netuid_2)

    # Register Bob and Dave in SN2
    assert subtensor.subnets.burned_register(
        wallet=bob_wallet,
        netuid=alice_subnet_netuid_2,
    ).success

    assert subtensor.subnets.burned_register(
        wallet=dave_wallet,
        netuid=alice_subnet_netuid_2,
    ).success

    # Register second SN
    alice_subnet_netuid_3 = subtensor.subnets.get_total_subnets()  # 3
    assert subtensor.subnets.register_subnet(alice_wallet).success
    assert subtensor.subnets.subnet_exists(alice_subnet_netuid_3), (
        "Subnet wasn't created successfully"
    )

    wait_to_start_call(subtensor, alice_wallet, alice_subnet_netuid_3)

    # Register Bob and Dave in SN3
    assert subtensor.subnets.burned_register(
        wallet=bob_wallet,
        netuid=alice_subnet_netuid_3,
    ).success

    assert subtensor.subnets.burned_register(
        wallet=dave_wallet,
        netuid=alice_subnet_netuid_3,
    ).success

    # Check Bob's stakes are empty.
    assert (
        subtensor.staking.get_stake_info_for_coldkey(bob_wallet.coldkey.ss58_address)
        == []
    )

    # Bob stakes to Dave in both SNs

    assert subtensor.staking.add_stake(
        wallet=bob_wallet,
        hotkey_ss58=dave_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid_2,
        amount=Balance.from_tao(10000),
        period=16,
    ), f"Cant add stake to dave in SN {alice_subnet_netuid_2}"
    assert subtensor.staking.add_stake(
        wallet=bob_wallet,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=alice_subnet_netuid_3,
        amount=Balance.from_tao(15000),
        period=16,
    ), f"Cant add stake to dave in SN {alice_subnet_netuid_3}"

    # Check that both stakes are presented in result
    bob_stakes = subtensor.staking.get_stake_info_for_coldkey(
        bob_wallet.coldkey.ss58_address
    )
    assert len(bob_stakes) == 2

    if rate_tolerance == 0.0001:
        # Raise the error
        with pytest.raises(
            ChainError, match="Slippage is too high for the transaction"
        ):
            subtensor.staking.unstake_all(
                wallet=bob_wallet,
                hotkey=bob_stakes[0].hotkey_ss58,
                netuid=bob_stakes[0].netuid,
                rate_tolerance=rate_tolerance,
            )
    else:
        # Successful cases
        for si in bob_stakes:
            assert subtensor.staking.unstake_all(
                wallet=bob_wallet,
                hotkey=si.hotkey_ss58,
                netuid=si.netuid,
                rate_tolerance=rate_tolerance,
            )[0]

        # Make sure both unstake were successful.
        bob_stakes = subtensor.staking.get_stake_info_for_coldkey(
            bob_wallet.coldkey.ss58_address
        )
        assert len(bob_stakes) == 0


@pytest.mark.parametrize(
    "rate_tolerance",
    [None, 1.0],
    ids=[
        "Without price limit",
        "With price limit",
    ],
)
@pytest.mark.asyncio
async def test_unstaking_with_limit_async(
    async_subtensor, alice_wallet, bob_wallet, dave_wallet, rate_tolerance
):
    """Test unstaking with limits goes well for all subnets with and without price limit."""
    logging.console.info("Testing [blue]test_unstaking_with_limit_async[/blue]")

    # Register first SN
    alice_subnet_netuid_2 = await async_subtensor.subnets.get_total_subnets()  # 2
    assert (await async_subtensor.subnets.register_subnet(alice_wallet)).success
    assert await async_subtensor.subnets.subnet_exists(alice_subnet_netuid_2), (
        "Subnet wasn't created successfully"
    )

    assert await async_wait_to_start_call(
        async_subtensor, alice_wallet, alice_subnet_netuid_2
    )

    # Register Bob and Dave in SN2
    assert (
        await async_subtensor.subnets.burned_register(
            wallet=bob_wallet,
            netuid=alice_subnet_netuid_2,
        )
    ).success

    assert (
        await async_subtensor.subnets.burned_register(
            wallet=dave_wallet,
            netuid=alice_subnet_netuid_2,
        )
    ).success

    # Register second SN
    alice_subnet_netuid_3 = await async_subtensor.subnets.get_total_subnets()  # 3
    assert (await async_subtensor.subnets.register_subnet(alice_wallet)).success
    assert await async_subtensor.subnets.subnet_exists(alice_subnet_netuid_3), (
        "Subnet wasn't created successfully"
    )

    await async_wait_to_start_call(async_subtensor, alice_wallet, alice_subnet_netuid_3)

    # Register Bob and Dave in SN3
    assert (
        await async_subtensor.subnets.burned_register(
            wallet=bob_wallet,
            netuid=alice_subnet_netuid_3,
        )
    ).success

    assert (
        await async_subtensor.subnets.burned_register(
            wallet=dave_wallet,
            netuid=alice_subnet_netuid_3,
        )
    ).success

    # Check Bob's stakes are empty.
    assert (
        await async_subtensor.staking.get_stake_info_for_coldkey(
            bob_wallet.coldkey.ss58_address
        )
        == []
    )

    # Bob stakes to Dave in both SNs

    assert await async_subtensor.staking.add_stake(
        wallet=bob_wallet,
        netuid=alice_subnet_netuid_2,
        hotkey_ss58=dave_wallet.hotkey.ss58_address,
        amount=Balance.from_tao(10000),
        period=16,
    ), f"Cant add stake to dave in SN {alice_subnet_netuid_2}"
    assert await async_subtensor.staking.add_stake(
        wallet=bob_wallet,
        netuid=alice_subnet_netuid_3,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        amount=Balance.from_tao(15000),
        period=16,
    ), f"Cant add stake to dave in SN {alice_subnet_netuid_3}"

    # Check that both stakes are presented in result
    bob_stakes = await async_subtensor.staking.get_stake_info_for_coldkey(
        bob_wallet.coldkey.ss58_address
    )
    assert len(bob_stakes) == 2

    if rate_tolerance == 0.0001:
        # Raise the error
        with pytest.raises(
            ChainError, match="Slippage is too high for the transaction"
        ):
            await async_subtensor.staking.unstake_all(
                wallet=bob_wallet,
                netuid=bob_stakes[0].netuid,
                hotkey=bob_stakes[0].hotkey_ss58,
                rate_tolerance=rate_tolerance,
            )
    else:
        # Successful cases
        for si in bob_stakes:
            assert (
                await async_subtensor.staking.unstake_all(
                    wallet=bob_wallet,
                    netuid=si.netuid,
                    hotkey=si.hotkey_ss58,
                    rate_tolerance=rate_tolerance,
                )
            )[0]

        # Make sure both unstake were successful.
        bob_stakes = await async_subtensor.staking.get_stake_info_for_coldkey(
            bob_wallet.coldkey.ss58_address
        )
        assert len(bob_stakes) == 0

        logging.console.success(
            "✅ Test [green]test_unstaking_with_limit_async[/green] passed"
        )
