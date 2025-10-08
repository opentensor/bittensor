import asyncio

import pytest

from bittensor import logging
from bittensor.core.chain_data.stake_info import StakeInfo
from bittensor.core.errors import ChainError
from bittensor.utils.balance import Balance
from tests.e2e_tests.utils import (
    get_dynamic_balance,
    AdminUtils,
    TestSubnet,
    ACTIVATE_SUBNET,
    REGISTER_SUBNET,
    REGISTER_NEURON,
    NETUID,
    SUDO_SET_ADMIN_FREEZE_WINDOW,
    SUDO_SET_TEMPO,
)
from tests.helpers.helpers import CloseInValue


def test_single_operation(subtensor, alice_wallet, bob_wallet):
    """
    Tests:
    - Staking using `add_stake`
    - Unstaking using `unstake`
    - Checks StakeInfo
    """
    alice_sn = TestSubnet(subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
        REGISTER_NEURON(bob_wallet),
    ]
    alice_sn.execute_steps(steps)

    stake = subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )

    assert stake == Balance(0).set_unit(alice_sn.netuid)

    assert subtensor.staking.add_stake(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=Balance.from_tao(1),
        period=16,
    ).success

    stake_alice = subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    logging.console.info(f"Alice stake: {stake_alice}")

    stake_bob = subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )

    logging.console.info(f"Bob stake: {stake_bob}")
    assert stake_bob > Balance(0).set_unit(alice_sn.netuid)

    stakes = subtensor.staking.get_stake_info_for_coldkey(
        alice_wallet.coldkey.ss58_address
    )

    expected_stakes = [
        StakeInfo(
            hotkey_ss58=stakes[0].hotkey_ss58,
            coldkey_ss58=alice_wallet.coldkey.ss58_address,
            netuid=alice_sn.netuid,
            stake=get_dynamic_balance(stakes[0].stake.rao, alice_sn.netuid),
            locked=Balance(0).set_unit(alice_sn.netuid),
            emission=get_dynamic_balance(stakes[0].emission.rao, alice_sn.netuid),
            drain=0,
            is_registered=True,
        ),
    ]

    fast_blocks_stake = (
        [
            StakeInfo(
                hotkey_ss58=stakes[1].hotkey_ss58,
                coldkey_ss58=alice_wallet.coldkey.ss58_address,
                netuid=alice_sn.netuid,
                stake=get_dynamic_balance(stakes[1].stake.rao, alice_sn.netuid),
                locked=Balance(0).set_unit(alice_sn.netuid),
                emission=get_dynamic_balance(stakes[1].emission.rao, alice_sn.netuid),
                drain=0,
                is_registered=True,
            )
        ]
        if subtensor.chain.is_fast_blocks()
        else []
    )

    expected_stakes += fast_blocks_stake

    assert stakes == expected_stakes

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
            netuid=alice_sn.netuid,
            stake=get_dynamic_balance(stakes[2].stake.rao, alice_sn.netuid),
            locked=Balance.from_tao(0, netuid=alice_sn.netuid),
            emission=get_dynamic_balance(stakes[2].emission.rao, alice_sn.netuid),
            drain=0,
            is_registered=True,
        ),
    }

    stake = subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    logging.console.info(f"Alice stake before unstake: {stake}")

    # unstale all to check in later
    response = subtensor.staking.unstake(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=stake,
        period=16,
    )

    assert response.success is True

    stake = subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )

    # all balances have been unstaked
    assert stake == Balance(0).set_unit(alice_sn.netuid)


@pytest.mark.asyncio
async def test_single_operation_async(async_subtensor, alice_wallet, bob_wallet):
    """
    Async ests:
    - Staking using `add_stake`
    - Unstaking using `unstake`
    - Checks StakeInfo
    """
    alice_sn = TestSubnet(async_subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
        REGISTER_NEURON(bob_wallet),
    ]
    await alice_sn.async_execute_steps(steps)

    stake = await async_subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )

    assert stake == Balance(0).set_unit(alice_sn.netuid)

    assert (
        await async_subtensor.staking.add_stake(
            wallet=alice_wallet,
            netuid=alice_sn.netuid,
            hotkey_ss58=bob_wallet.hotkey.ss58_address,
            amount=Balance.from_tao(1),
            period=16,
        )
    ).success

    stake_alice = await async_subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    logging.console.info(f"Alice stake: {stake_alice}")

    stake_bob = await async_subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )

    logging.console.info(f"Bob stake: {stake_bob}")
    assert stake_bob > Balance(0).set_unit(alice_sn.netuid)

    stakes = await async_subtensor.staking.get_stake_info_for_coldkey(
        alice_wallet.coldkey.ss58_address
    )

    expected_stakes = [
        StakeInfo(
            hotkey_ss58=stakes[0].hotkey_ss58,
            coldkey_ss58=alice_wallet.coldkey.ss58_address,
            netuid=alice_sn.netuid,
            stake=get_dynamic_balance(stakes[0].stake.rao, alice_sn.netuid),
            locked=Balance(0).set_unit(alice_sn.netuid),
            emission=get_dynamic_balance(stakes[0].emission.rao, alice_sn.netuid),
            drain=0,
            is_registered=True,
        ),
    ]

    fast_blocks_stake = (
        [
            StakeInfo(
                hotkey_ss58=stakes[1].hotkey_ss58,
                coldkey_ss58=alice_wallet.coldkey.ss58_address,
                netuid=alice_sn.netuid,
                stake=get_dynamic_balance(stakes[1].stake.rao, alice_sn.netuid),
                locked=Balance(0).set_unit(alice_sn.netuid),
                emission=get_dynamic_balance(stakes[1].emission.rao, alice_sn.netuid),
                drain=0,
                is_registered=True,
            )
        ]
        if await async_subtensor.chain.is_fast_blocks()
        else []
    )

    expected_stakes += fast_blocks_stake

    assert stakes == expected_stakes

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
            netuid=alice_sn.netuid,
            stake=get_dynamic_balance(stakes[2].stake.rao, alice_sn.netuid),
            locked=Balance.from_tao(0, netuid=alice_sn.netuid),
            emission=get_dynamic_balance(stakes[2].emission.rao, alice_sn.netuid),
            drain=0,
            is_registered=True,
        ),
    }

    stake = await async_subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    logging.console.info(f"Alice stake before unstake: {stake}")

    # unstale all to check in later
    success, message = await async_subtensor.staking.unstake_all(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        period=16,
    )
    assert success is True, message

    stake = await async_subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    logging.console.info(f"Alice stake after unstake: {stake}")

    # all balances have been unstaked
    assert stake == Balance(0).set_unit(alice_sn.netuid)


def test_batch_operations(subtensor, alice_wallet, bob_wallet):
    """
    Tests:
    - Staking using `add_stake_multiple`
    - Unstaking using `unstake_multiple`
    - Checks StakeInfo
    - Checks Accounts Balance
    """
    subnets_tested = 2

    sns = [TestSubnet(subtensor) for _ in range(subnets_tested)]

    sn_steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
        REGISTER_NEURON(bob_wallet),
    ]

    for sn in sns:
        sn.execute_steps(sn_steps)

        stake = subtensor.staking.get_stake(
            alice_wallet.coldkey.ss58_address,
            bob_wallet.hotkey.ss58_address,
            netuid=sn.netuid,
        )

        assert stake == Balance(0).set_unit(sn.netuid), (
            f"netuid={sn.netuid} stake={stake}"
        )

    netuids = [sn.netuid for sn in sns]

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

    assert subtensor.staking.add_stake_multiple(
        wallet=alice_wallet,
        hotkey_ss58s=[bob_wallet.hotkey.ss58_address for _ in netuids],
        netuids=netuids,
        amounts=[Balance.from_tao(10_000) for _ in netuids],
    ).success

    stakes = [
        subtensor.staking.get_stake(
            coldkey_ss58=alice_wallet.coldkey.ss58_address,
            hotkey_ss58=bob_wallet.hotkey.ss58_address,
            netuid=sn.netuid,
        )
        for sn in sns
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

    response = subtensor.staking.unstake_multiple(
        wallet=alice_wallet,
        netuids=[sn.netuid for sn in sns],
        hotkey_ss58s=[bob_wallet.hotkey.ss58_address for _ in netuids],
        amounts=[Balance.from_tao(100) for _ in netuids],
        raise_error=True,
    )
    assert response.success, response.message
    total_fee = sum(
        [
            v.extrinsic_fee
            for _, v in response.data.items()
            if hasattr(v, "extrinsic_fee")
        ]
    )
    logging.console.info(f"Total fee: [blue]{total_fee}[/blue]")

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


@pytest.mark.asyncio
async def test_batch_operations_async(async_subtensor, alice_wallet, bob_wallet):
    """
    Async tests:
    - Staking using `add_stake_multiple`
    - Unstaking using `unstake_multiple`
    - Checks StakeInfo
    - Checks Accounts Balance
    """
    subnets_tested = 2

    sns = [TestSubnet(async_subtensor) for _ in range(subnets_tested)]

    sn_steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
        REGISTER_NEURON(bob_wallet),
    ]

    for sn in sns:
        await sn.async_execute_steps(sn_steps)

        stake = await async_subtensor.staking.get_stake(
            alice_wallet.coldkey.ss58_address,
            bob_wallet.hotkey.ss58_address,
            netuid=sn.netuid,
        )

        assert stake == Balance(0).set_unit(sn.netuid), (
            f"netuid={sn.netuid} stake={stake}"
        )

    netuids = [sn.netuid for sn in sns]

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

    response = await async_subtensor.staking.add_stake_multiple(
        wallet=alice_wallet,
        hotkey_ss58s=[bob_wallet.hotkey.ss58_address for _ in netuids],
        netuids=netuids,
        amounts=[Balance.from_tao(10_000) for _ in netuids],
    )
    assert response.success

    stakes = [
        await async_subtensor.staking.get_stake(
            coldkey_ss58=alice_wallet.coldkey.ss58_address,
            hotkey_ss58=bob_wallet.hotkey.ss58_address,
            netuid=sn.netuid,
        )
        for sn in sns
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

    response = await async_subtensor.staking.unstake_multiple(
        wallet=alice_wallet,
        netuids=netuids,
        hotkey_ss58s=[bob_wallet.hotkey.ss58_address for _ in netuids],
        amounts=[Balance.from_tao(100) for _ in netuids],
    )
    assert response.success, response.message
    total_fee = sum(
        [
            v.extrinsic_fee
            for _, v in response.data.items()
            if hasattr(v, "extrinsic_fee")
        ]
    )
    logging.console.info(f"Total fee: [blue]{total_fee}[/blue]")

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


def test_safe_staking_scenarios(subtensor, alice_wallet, bob_wallet, eve_wallet):
    """
    Tests safe staking scenarios with different parameters.

    For both staking and unstaking:
    1. Fails with strict threshold (0.5%) and no partial staking
    2. Succeeds with strict threshold (0.5%) and partial staking allowed
    3. Succeeds with lenient threshold (10% and 30%) and no partial staking
    """
    TEMPO_TO_SET = 50 if subtensor.chain.is_fast_blocks() else 20

    alice_sn = TestSubnet(subtensor)
    steps = [
        SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
        REGISTER_SUBNET(alice_wallet),
        SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, TEMPO_TO_SET),
        ACTIVATE_SUBNET(alice_wallet),
        REGISTER_NEURON(bob_wallet),
    ]
    alice_sn.execute_steps(steps)

    tempo = subtensor.subnets.get_subnet_hyperparameters(netuid=alice_sn.netuid).tempo
    assert tempo == TEMPO_TO_SET, "SN tempos has not been changed."
    logging.console.success(f"SN #{alice_sn.netuid} tempo set to {TEMPO_TO_SET}")

    initial_stake = subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    assert initial_stake == Balance(0).set_unit(alice_sn.netuid)
    logging.console.info(f"[orange]Initial stake: {initial_stake}[orange]")

    # Test Staking Scenarios
    stake_amount = Balance.from_tao(100)

    # 1. Strict params - should fail
    assert (
        subtensor.staking.add_stake(
            wallet=alice_wallet,
            netuid=alice_sn.netuid,
            hotkey_ss58=bob_wallet.hotkey.ss58_address,
            amount=stake_amount,
            safe_staking=True,
            rate_tolerance=0.005,  # 0.5%
            allow_partial_stake=False,
        ).success
        is False
    ), "Staking should fail."

    current_stake = subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    assert current_stake == Balance(0).set_unit(alice_sn.netuid), (
        "Stake should not change after failed attempt"
    )
    logging.console.info(f"[orange]Current stake: {current_stake}[orange]")

    # 2. Partial allowed - should succeed partially
    assert subtensor.staking.add_stake(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=stake_amount,
        safe_staking=True,
        rate_tolerance=0.005,  # 0.5%
        allow_partial_stake=True,
    ).success

    partial_stake = subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    assert partial_stake > Balance(0).set_unit(alice_sn.netuid), (
        "Partial stake should be added"
    )
    assert partial_stake < Balance.from_tao(stake_amount.tao).set_unit(
        alice_sn.netuid
    ), "Partial stake should be less than requested amount"

    # 3. Higher threshold - should succeed fully
    amount = Balance.from_tao(100)
    assert subtensor.staking.add_stake(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=amount,
        safe_staking=True,
        rate_tolerance=0.22,  # 22%
        allow_partial_stake=False,
    ).success

    full_stake = subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )

    # Test Unstaking Scenarios
    # 1. Strict params - should fail
    response = subtensor.staking.unstake(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=full_stake,
        safe_unstaking=True,
        rate_tolerance=0.005,  # 0.5%
        allow_partial_stake=False,
    )
    assert response.success is False

    current_stake = subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )

    logging.console.info(f"[orange]Current stake: {current_stake}[orange]")
    logging.console.info(f"[orange]Full stake: {full_stake}[orange]")

    # The current stake may be greater or equal, but not less. It may be greater due to rapid emissions.
    assert current_stake >= full_stake, (
        "Stake should not change after failed unstake attempt."
    )

    # 2. Partial allowed - should succeed partially
    response = subtensor.staking.unstake(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=current_stake,
        safe_unstaking=True,
        rate_tolerance=0.005,  # 0.5%
        allow_partial_stake=True,
    )
    assert response.success, response.message

    partial_unstake = subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    logging.console.info(f"[orange]Partial unstake: {partial_unstake}[orange]")
    assert partial_unstake > Balance(0).set_unit(alice_sn.netuid), (
        "Some stake should remain"
    )

    # 3. Higher threshold - should succeed fully
    response = subtensor.staking.unstake(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=partial_unstake,
        safe_unstaking=True,
        rate_tolerance=0.3,  # 30%
        allow_partial_stake=False,
    )
    assert response.success is True, "Unstake should succeed"


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
    TEMPO_TO_SET = 50 if await async_subtensor.chain.is_fast_blocks() else 20

    alice_sn = TestSubnet(async_subtensor)
    steps = [
        SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
        REGISTER_SUBNET(alice_wallet),
        SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, TEMPO_TO_SET),
        ACTIVATE_SUBNET(alice_wallet),
        REGISTER_NEURON(bob_wallet),
    ]
    await alice_sn.async_execute_steps(steps)

    tempo = (
        await async_subtensor.subnets.get_subnet_hyperparameters(netuid=alice_sn.netuid)
    ).tempo
    assert tempo == TEMPO_TO_SET, "SN tempos has not been changed."
    logging.console.success(f"SN #{alice_sn.netuid} tempo set to {TEMPO_TO_SET}")

    initial_stake = await async_subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    assert initial_stake == Balance(0).set_unit(alice_sn.netuid)
    logging.console.info(f"[orange]Initial stake: {initial_stake}[orange]")

    # Test Staking Scenarios
    stake_amount = Balance.from_tao(100)

    # 1. Strict params - should fail
    assert (
        await async_subtensor.staking.add_stake(
            wallet=alice_wallet,
            netuid=alice_sn.netuid,
            hotkey_ss58=bob_wallet.hotkey.ss58_address,
            amount=stake_amount,
            safe_staking=True,
            rate_tolerance=0.005,  # 0.5%
            allow_partial_stake=False,
        )
    ).success is False, "Staking should fail."

    current_stake = await async_subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    assert current_stake == Balance(0).set_unit(alice_sn.netuid), (
        "Stake should not change after failed attempt"
    )
    logging.console.info(f"[orange]Current stake: {current_stake}[orange]")

    # 2. Partial allowed - should succeed partially
    assert (
        await async_subtensor.staking.add_stake(
            wallet=alice_wallet,
            netuid=alice_sn.netuid,
            hotkey_ss58=bob_wallet.hotkey.ss58_address,
            amount=stake_amount,
            safe_staking=True,
            rate_tolerance=0.005,  # 0.5%
            allow_partial_stake=True,
        )
    ).success

    partial_stake = await async_subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    assert partial_stake > Balance(0).set_unit(alice_sn.netuid), (
        "Partial stake should be added"
    )
    assert partial_stake < Balance.from_tao(stake_amount.tao).set_unit(
        alice_sn.netuid
    ), "Partial stake should be less than requested amount"

    # 3. Higher threshold - should succeed fully
    amount = Balance.from_tao(100)
    assert (
        await async_subtensor.staking.add_stake(
            wallet=alice_wallet,
            netuid=alice_sn.netuid,
            hotkey_ss58=bob_wallet.hotkey.ss58_address,
            amount=amount,
            safe_staking=True,
            rate_tolerance=0.22,  # 22%
            allow_partial_stake=False,
        )
    ).success

    full_stake = await async_subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )

    # Test Unstaking Scenarios
    # 1. Strict params - should fail
    response = await async_subtensor.staking.unstake(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=full_stake,
        safe_unstaking=True,
        rate_tolerance=0.005,  # 0.5%
        allow_partial_stake=False,
    )
    assert response.success is False, "Unstake should fail."

    current_stake = await async_subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )

    logging.console.info(f"[orange]Current stake: {current_stake}[orange]")
    logging.console.info(f"[orange]Full stake: {full_stake}[orange]")

    # The current stake may be greater or equal, but not less. It may be greater due to rapid emissions.
    assert current_stake >= full_stake, (
        "Stake should not change after failed unstake attempt"
    )

    # 2. Partial allowed - should succeed partially
    response = await async_subtensor.staking.unstake(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=current_stake,
        safe_unstaking=True,
        rate_tolerance=0.005,  # 0.5%
        allow_partial_stake=True,
    )
    assert response.success is True

    partial_unstake = await async_subtensor.staking.get_stake(
        coldkey_ss58=alice_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=alice_sn.netuid,
    )
    logging.console.info(f"[orange]Partial unstake: {partial_unstake}[orange]")
    assert partial_unstake > Balance(0).set_unit(alice_sn.netuid), (
        "Some stake should remain"
    )

    # 3. Higher threshold - should succeed fully
    response = await async_subtensor.staking.unstake(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        amount=partial_unstake,
        safe_unstaking=True,
        rate_tolerance=0.3,  # 30%
        allow_partial_stake=False,
    )
    assert response.success is True, "Unstake should succeed"


def test_safe_swap_stake_scenarios(subtensor, alice_wallet, bob_wallet):
    """
    Tests safe swap stake scenarios with different parameters.

    Tests:
    1. Fails with strict threshold (0.5%)
    2. Succeeds with lenient threshold (10%)
    """
    origin_sn = TestSubnet(subtensor)
    dest_sn = TestSubnet(subtensor)

    sns = [origin_sn, dest_sn]
    steps = [
        REGISTER_SUBNET(bob_wallet),
        ACTIVATE_SUBNET(bob_wallet),
        REGISTER_NEURON(alice_wallet),
    ]
    [sn.execute_steps(steps) for sn in sns]

    # Add initial stake to swap from
    initial_stake_amount = Balance.from_tao(10_000)
    assert subtensor.staking.add_stake(
        wallet=alice_wallet,
        netuid=origin_sn.netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        amount=initial_stake_amount,
    ).success

    origin_stake = subtensor.staking.get_stake(
        alice_wallet.coldkey.ss58_address,
        alice_wallet.hotkey.ss58_address,
        netuid=origin_sn.netuid,
    )
    assert origin_stake > Balance(0).set_unit(origin_sn.netuid), (
        "Origin stake should be non-zero"
    )

    stake_swap_amount = Balance.from_tao(10_000)
    # 1. Try swap with strict threshold and big amount- should fail
    response = subtensor.staking.swap_stake(
        wallet=alice_wallet,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        origin_netuid=origin_sn.netuid,
        destination_netuid=dest_sn.netuid,
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
        netuid=dest_sn.netuid,
    )
    assert dest_stake == Balance(0).set_unit(dest_sn.netuid), (
        "Destination stake should remain 0 after failed swap"
    )

    # 2. Try swap with higher threshold and less amount - should succeed
    stake_swap_amount = Balance.from_tao(100)
    response = subtensor.staking.swap_stake(
        wallet=alice_wallet,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        origin_netuid=origin_sn.netuid,
        destination_netuid=dest_sn.netuid,
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
        netuid=dest_sn.netuid,
    )
    assert dest_stake > Balance(0).set_unit(dest_sn.netuid), (
        "Destination stake should be non-zero after successful swap"
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
    origin_sn = TestSubnet(async_subtensor)
    dest_sn = TestSubnet(async_subtensor)

    sns = [origin_sn, dest_sn]
    steps = [
        REGISTER_SUBNET(bob_wallet),
        ACTIVATE_SUBNET(bob_wallet),
        REGISTER_NEURON(alice_wallet),
    ]
    [await sn.async_execute_steps(steps) for sn in sns]

    # Add initial stake to swap from
    initial_stake_amount = Balance.from_tao(10_000)
    assert (
        await async_subtensor.staking.add_stake(
            wallet=alice_wallet,
            netuid=origin_sn.netuid,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            amount=initial_stake_amount,
        )
    ).success

    origin_stake = await async_subtensor.staking.get_stake(
        alice_wallet.coldkey.ss58_address,
        alice_wallet.hotkey.ss58_address,
        netuid=origin_sn.netuid,
    )
    assert origin_stake > Balance(0).set_unit(origin_sn.netuid), (
        "Origin stake should be non-zero"
    )

    stake_swap_amount = Balance.from_tao(10_000)
    # 1. Try swap with strict threshold and big amount- should fail
    response = await async_subtensor.staking.swap_stake(
        wallet=alice_wallet,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        origin_netuid=origin_sn.netuid,
        destination_netuid=dest_sn.netuid,
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
        netuid=dest_sn.netuid,
    )
    assert dest_stake == Balance(0).set_unit(dest_sn.netuid), (
        "Destination stake should remain 0 after failed swap"
    )

    # 2. Try swap with higher threshold and less amount - should succeed
    stake_swap_amount = Balance.from_tao(100)
    response = await async_subtensor.staking.swap_stake(
        wallet=alice_wallet,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        origin_netuid=origin_sn.netuid,
        destination_netuid=dest_sn.netuid,
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
        netuid=dest_sn.netuid,
    )
    assert dest_stake > Balance(0).set_unit(dest_sn.netuid), (
        "Destination stake should be non-zero after successful swap"
    )


def test_move_stake(subtensor, alice_wallet, bob_wallet, dave_wallet):
    """
    Tests:
    - Adding stake
    - Moving stake from one hotkey-subnet pair to another
    - Testing `move_stake` method with `move_all_stake=True` flag.
    """
    alice_sn = TestSubnet(subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
    ]
    alice_sn.execute_steps(steps)

    assert subtensor.staking.add_stake(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        amount=Balance.from_tao(1_000),
    ).success

    stakes = subtensor.staking.get_stake_info_for_coldkey(
        alice_wallet.coldkey.ss58_address
    )

    assert stakes == [
        StakeInfo(
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            coldkey_ss58=alice_wallet.coldkey.ss58_address,
            netuid=alice_sn.netuid,
            stake=get_dynamic_balance(stakes[0].stake.rao, alice_sn.netuid),
            locked=Balance(0).set_unit(alice_sn.netuid),
            emission=get_dynamic_balance(stakes[0].emission.rao, alice_sn.netuid),
            drain=0,
            is_registered=True,
        ),
    ]

    bob_sn = TestSubnet(subtensor)
    steps = [
        REGISTER_SUBNET(bob_wallet),
        ACTIVATE_SUBNET(bob_wallet),
    ]
    bob_sn.execute_steps(steps)

    alice_sn.execute_steps([REGISTER_NEURON(bob_wallet), REGISTER_NEURON(dave_wallet)])

    response = subtensor.staking.move_stake(
        wallet=alice_wallet,
        origin_hotkey_ss58=alice_wallet.hotkey.ss58_address,
        origin_netuid=alice_sn.netuid,
        destination_hotkey_ss58=bob_wallet.hotkey.ss58_address,
        destination_netuid=bob_sn.netuid,
        amount=stakes[0].stake,
        wait_for_finalization=True,
        wait_for_inclusion=True,
    )
    assert response.success is True

    stakes = subtensor.staking.get_stake_info_for_coldkey(
        alice_wallet.coldkey.ss58_address
    )

    expected_stakes = [
        StakeInfo(
            hotkey_ss58=stakes[0].hotkey_ss58,
            coldkey_ss58=alice_wallet.coldkey.ss58_address,
            netuid=alice_sn.netuid
            if subtensor.chain.is_fast_blocks()
            else bob_sn.netuid,
            stake=get_dynamic_balance(stakes[0].stake.rao, alice_sn.netuid),
            locked=Balance(0).set_unit(alice_sn.netuid),
            emission=get_dynamic_balance(stakes[0].emission.rao, alice_sn.netuid),
            drain=0,
            is_registered=True,
        )
    ]

    fast_block_stake = (
        [
            StakeInfo(
                hotkey_ss58=stakes[1].hotkey_ss58,
                coldkey_ss58=alice_wallet.coldkey.ss58_address,
                netuid=bob_sn.netuid,
                stake=get_dynamic_balance(stakes[1].stake.rao, bob_sn.netuid),
                locked=Balance(0).set_unit(bob_sn.netuid),
                emission=get_dynamic_balance(stakes[1].emission.rao, bob_sn.netuid),
                drain=0,
                is_registered=True,
            ),
        ]
        if subtensor.chain.is_fast_blocks()
        else []
    )

    expected_stakes += fast_block_stake
    logging.console.info(f"[orange]FS: {fast_block_stake}[/orange]")
    logging.console.info(f"[orange]RS: {stakes}[/orange]")
    logging.console.info(f"[orange]ES: {expected_stakes}[/orange]")
    assert stakes == expected_stakes

    # test move_stake with move_all_stake=True
    dave_stake = subtensor.staking.get_stake(
        coldkey_ss58=dave_wallet.coldkey.ss58_address,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
        netuid=bob_sn.netuid,
    )
    logging.console.info(f"[orange]Dave stake before adding: {dave_stake}[orange]")

    assert subtensor.staking.add_stake(
        wallet=dave_wallet,
        netuid=bob_sn.netuid,
        hotkey_ss58=dave_wallet.hotkey.ss58_address,
        amount=Balance.from_tao(1000),
        allow_partial_stake=True,
    ).success

    dave_stake = subtensor.staking.get_stake(
        coldkey_ss58=dave_wallet.coldkey.ss58_address,
        hotkey_ss58=dave_wallet.hotkey.ss58_address,
        netuid=bob_sn.netuid,
    )
    logging.console.info(f"[orange]Dave stake after adding: {dave_stake}[orange]")

    # let chain to process the transaction
    subtensor.wait_for_block(
        subtensor.block + subtensor.subnets.tempo(netuid=bob_sn.netuid)
    )

    response = subtensor.staking.move_stake(
        wallet=dave_wallet,
        origin_hotkey_ss58=dave_wallet.hotkey.ss58_address,
        origin_netuid=bob_sn.netuid,
        destination_hotkey_ss58=bob_wallet.hotkey.ss58_address,
        destination_netuid=bob_sn.netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        move_all_stake=True,
    )
    assert response.success is True

    dave_stake = subtensor.staking.get_stake(
        coldkey_ss58=dave_wallet.coldkey.ss58_address,
        hotkey_ss58=dave_wallet.hotkey.ss58_address,
        netuid=bob_sn.netuid,
    )
    logging.console.info(f"[orange]Dave stake after moving all: {dave_stake}[orange]")

    assert dave_stake.rao == CloseInValue(0, 0.00001)


@pytest.mark.asyncio
async def test_move_stake_async(async_subtensor, alice_wallet, bob_wallet, dave_wallet):
    """
    Tests:
    - Adding stake
    - Moving stake from one hotkey-subnet pair to another
    - Testing `move_stake` method with `move_all_stake=True` flag.
    """
    alice_sn = TestSubnet(async_subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
    ]
    await alice_sn.async_execute_steps(steps)

    assert (
        await async_subtensor.staking.add_stake(
            wallet=alice_wallet,
            netuid=alice_sn.netuid,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            amount=Balance.from_tao(1_000),
        )
    ).success

    stakes = await async_subtensor.staking.get_stake_info_for_coldkey(
        alice_wallet.coldkey.ss58_address
    )

    assert stakes == [
        StakeInfo(
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            coldkey_ss58=alice_wallet.coldkey.ss58_address,
            netuid=alice_sn.netuid,
            stake=get_dynamic_balance(stakes[0].stake.rao, alice_sn.netuid),
            locked=Balance(0).set_unit(alice_sn.netuid),
            emission=get_dynamic_balance(stakes[0].emission.rao, alice_sn.netuid),
            drain=0,
            is_registered=True,
        ),
    ]

    bob_sn = TestSubnet(async_subtensor)
    steps = [
        REGISTER_SUBNET(bob_wallet),
        ACTIVATE_SUBNET(bob_wallet),
    ]
    await bob_sn.async_execute_steps(steps)

    await alice_sn.async_execute_steps(
        [REGISTER_NEURON(bob_wallet), REGISTER_NEURON(dave_wallet)]
    )

    response = await async_subtensor.staking.move_stake(
        wallet=alice_wallet,
        origin_hotkey_ss58=alice_wallet.hotkey.ss58_address,
        origin_netuid=alice_sn.netuid,
        destination_hotkey_ss58=bob_wallet.hotkey.ss58_address,
        destination_netuid=bob_sn.netuid,
        amount=stakes[0].stake,
        wait_for_finalization=True,
        wait_for_inclusion=True,
    )
    assert response.success is True

    stakes = await async_subtensor.staking.get_stake_info_for_coldkey(
        alice_wallet.coldkey.ss58_address
    )

    expected_stakes = [
        StakeInfo(
            hotkey_ss58=stakes[0].hotkey_ss58,
            coldkey_ss58=alice_wallet.coldkey.ss58_address,
            netuid=alice_sn.netuid
            if await async_subtensor.chain.is_fast_blocks()
            else bob_sn.netuid,
            stake=get_dynamic_balance(stakes[0].stake.rao, alice_sn.netuid),
            locked=Balance(0).set_unit(alice_sn.netuid),
            emission=get_dynamic_balance(stakes[0].emission.rao, alice_sn.netuid),
            drain=0,
            is_registered=True,
        )
    ]

    fast_block_stake = (
        [
            StakeInfo(
                hotkey_ss58=stakes[1].hotkey_ss58,
                coldkey_ss58=alice_wallet.coldkey.ss58_address,
                netuid=bob_sn.netuid,
                stake=get_dynamic_balance(stakes[1].stake.rao, bob_sn.netuid),
                locked=Balance(0).set_unit(bob_sn.netuid),
                emission=get_dynamic_balance(stakes[1].emission.rao, bob_sn.netuid),
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
        netuid=bob_sn.netuid,
    )
    logging.console.info(f"[orange]Dave stake before adding: {dave_stake}[orange]")

    assert (
        await async_subtensor.staking.add_stake(
            wallet=dave_wallet,
            hotkey_ss58=dave_wallet.hotkey.ss58_address,
            netuid=bob_sn.netuid,
            amount=Balance.from_tao(1000),
            allow_partial_stake=True,
        )
    ).success

    dave_stake = await async_subtensor.staking.get_stake(
        coldkey_ss58=dave_wallet.coldkey.ss58_address,
        hotkey_ss58=dave_wallet.hotkey.ss58_address,
        netuid=bob_sn.netuid,
    )
    logging.console.info(f"[orange]Dave stake after adding: {dave_stake}[orange]")

    block_, tampo_ = await asyncio.gather(
        async_subtensor.block, async_subtensor.subnets.tempo(netuid=bob_sn.netuid)
    )
    # let chain to process the transaction
    await async_subtensor.wait_for_block(block_ + tampo_)

    response = await async_subtensor.staking.move_stake(
        wallet=dave_wallet,
        origin_hotkey_ss58=dave_wallet.hotkey.ss58_address,
        origin_netuid=bob_sn.netuid,
        destination_hotkey_ss58=bob_wallet.hotkey.ss58_address,
        destination_netuid=bob_sn.netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
        move_all_stake=True,
    )
    assert response.success, response.error_message

    dave_stake = await async_subtensor.staking.get_stake(
        coldkey_ss58=dave_wallet.coldkey.ss58_address,
        hotkey_ss58=dave_wallet.hotkey.ss58_address,
        netuid=bob_sn.netuid,
    )
    logging.console.info(f"[orange]Dave stake after moving all: {dave_stake}[orange]")

    assert dave_stake.rao == CloseInValue(0, 0.00001)


def test_transfer_stake(subtensor, alice_wallet, bob_wallet, dave_wallet):
    """
    Tests:
    - Adding stake
    - Transferring stake from one coldkey-subnet pair to another
    """
    alice_sn = TestSubnet(subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
    ]
    alice_sn.execute_steps(steps)

    assert subtensor.staking.add_stake(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        amount=Balance.from_tao(1_000),
    ).success

    alice_stakes = subtensor.staking.get_stake_info_for_coldkey(
        alice_wallet.coldkey.ss58_address
    )

    assert alice_stakes == [
        StakeInfo(
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            coldkey_ss58=alice_wallet.coldkey.ss58_address,
            netuid=alice_sn.netuid,
            stake=get_dynamic_balance(alice_stakes[0].stake.rao, alice_sn.netuid),
            locked=Balance(0).set_unit(alice_sn.netuid),
            emission=get_dynamic_balance(alice_stakes[0].emission.rao, alice_sn.netuid),
            drain=0,
            is_registered=True,
        ),
    ]

    bob_stakes = subtensor.staking.get_stake_info_for_coldkey(
        bob_wallet.coldkey.ss58_address
    )
    assert bob_stakes == []

    dave_sn = TestSubnet(subtensor)
    steps = [
        REGISTER_SUBNET(dave_wallet),
        ACTIVATE_SUBNET(dave_wallet),
        REGISTER_NEURON(bob_wallet),
    ]
    dave_sn.execute_steps(steps)

    response = subtensor.staking.transfer_stake(
        alice_wallet,
        destination_coldkey_ss58=bob_wallet.coldkey.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        origin_netuid=alice_sn.netuid,
        destination_netuid=dave_sn.netuid,
        amount=alice_stakes[0].stake,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response.success is True

    alice_stakes = subtensor.staking.get_stake_info_for_coldkey(
        alice_wallet.coldkey.ss58_address
    )

    expected_alice_stake = (
        [
            StakeInfo(
                hotkey_ss58=alice_wallet.hotkey.ss58_address,
                coldkey_ss58=alice_wallet.coldkey.ss58_address,
                netuid=alice_sn.netuid,
                stake=get_dynamic_balance(alice_stakes[0].stake.rao, alice_sn.netuid),
                locked=Balance(0).set_unit(alice_sn.netuid),
                emission=get_dynamic_balance(
                    alice_stakes[0].emission.rao, alice_sn.netuid
                ),
                drain=0,
                is_registered=True,
            ),
        ]
        if subtensor.chain.is_fast_blocks()
        else []
    )

    assert alice_stakes == expected_alice_stake

    bob_stakes = subtensor.staking.get_stake_info_for_coldkey(
        bob_wallet.coldkey.ss58_address
    )

    expected_bob_stake = [
        StakeInfo(
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            coldkey_ss58=bob_wallet.coldkey.ss58_address,
            netuid=dave_sn.netuid,
            stake=get_dynamic_balance(bob_stakes[0].stake.rao, dave_sn.netuid),
            locked=Balance(0).set_unit(dave_sn.netuid),
            emission=get_dynamic_balance(bob_stakes[0].emission.rao, dave_sn.netuid),
            drain=0,
            is_registered=False,
        ),
    ]
    assert bob_stakes == expected_bob_stake


@pytest.mark.asyncio
async def test_transfer_stake_async(
    async_subtensor, alice_wallet, bob_wallet, dave_wallet
):
    """
    Tests:
    - Adding stake
    - Transferring stake from one coldkey-subnet pair to another
    """
    alice_sn = TestSubnet(async_subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
    ]
    await alice_sn.async_execute_steps(steps)

    assert (
        await async_subtensor.staking.add_stake(
            wallet=alice_wallet,
            netuid=alice_sn.netuid,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            amount=Balance.from_tao(1_000),
        )
    ).success

    alice_stakes = await async_subtensor.staking.get_stake_info_for_coldkey(
        alice_wallet.coldkey.ss58_address
    )

    assert alice_stakes == [
        StakeInfo(
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            coldkey_ss58=alice_wallet.coldkey.ss58_address,
            netuid=alice_sn.netuid,
            stake=get_dynamic_balance(alice_stakes[0].stake.rao, alice_sn.netuid),
            locked=Balance(0).set_unit(alice_sn.netuid),
            emission=get_dynamic_balance(alice_stakes[0].emission.rao, alice_sn.netuid),
            drain=0,
            is_registered=True,
        ),
    ]

    bob_stakes = await async_subtensor.staking.get_stake_info_for_coldkey(
        bob_wallet.coldkey.ss58_address
    )
    assert bob_stakes == []

    dave_sn = TestSubnet(async_subtensor)
    steps = [
        REGISTER_SUBNET(dave_wallet),
        ACTIVATE_SUBNET(dave_wallet),
        REGISTER_NEURON(bob_wallet),
    ]
    await dave_sn.async_execute_steps(steps)

    response = await async_subtensor.staking.transfer_stake(
        alice_wallet,
        destination_coldkey_ss58=bob_wallet.coldkey.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        origin_netuid=alice_sn.netuid,
        destination_netuid=dave_sn.netuid,
        amount=alice_stakes[0].stake,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert response.success is True

    alice_stakes = await async_subtensor.staking.get_stake_info_for_coldkey(
        alice_wallet.coldkey.ss58_address
    )

    expected_alice_stake = (
        [
            StakeInfo(
                hotkey_ss58=alice_wallet.hotkey.ss58_address,
                coldkey_ss58=alice_wallet.coldkey.ss58_address,
                netuid=alice_sn.netuid,
                stake=get_dynamic_balance(alice_stakes[0].stake.rao, alice_sn.netuid),
                locked=Balance(0).set_unit(alice_sn.netuid),
                emission=get_dynamic_balance(
                    alice_stakes[0].emission.rao, alice_sn.netuid
                ),
                drain=0,
                is_registered=True,
            ),
        ]
        if await async_subtensor.chain.is_fast_blocks()
        else []
    )

    assert alice_stakes == expected_alice_stake

    bob_stakes = await async_subtensor.staking.get_stake_info_for_coldkey(
        bob_wallet.coldkey.ss58_address
    )

    expected_bob_stake = [
        StakeInfo(
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            coldkey_ss58=bob_wallet.coldkey.ss58_address,
            netuid=dave_sn.netuid,
            stake=get_dynamic_balance(bob_stakes[0].stake.rao, dave_sn.netuid),
            locked=Balance(0).set_unit(dave_sn.netuid),
            emission=get_dynamic_balance(bob_stakes[0].emission.rao, dave_sn.netuid),
            drain=0,
            is_registered=False,
        ),
    ]
    assert bob_stakes == expected_bob_stake


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
    # Register first SN
    alice_sn_2 = TestSubnet(subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
        REGISTER_NEURON(bob_wallet),
        REGISTER_NEURON(dave_wallet),
    ]
    alice_sn_2.execute_steps(steps)

    # Register second SN
    alice_sn_3 = TestSubnet(subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
        REGISTER_NEURON(bob_wallet),
        REGISTER_NEURON(dave_wallet),
    ]
    alice_sn_3.execute_steps(steps)

    # Check Bob's stakes are empty.
    assert (
        subtensor.staking.get_stake_info_for_coldkey(bob_wallet.coldkey.ss58_address)
        == []
    )

    # Bob stakes to Dave in both SNs

    assert subtensor.staking.add_stake(
        wallet=bob_wallet,
        hotkey_ss58=dave_wallet.hotkey.ss58_address,
        netuid=alice_sn_2.netuid,
        amount=Balance.from_tao(10000),
        period=16,
    ).success, f"Cant add stake to dave in SN {alice_sn_2.netuid}"

    assert subtensor.staking.add_stake(
        wallet=bob_wallet,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=alice_sn_3.netuid,
        amount=Balance.from_tao(15000),
        period=16,
    ).success, f"Cant add stake to dave in SN {alice_sn_3.netuid}"

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
                netuid=bob_stakes[0].netuid,
                hotkey_ss58=bob_stakes[0].hotkey_ss58,
                rate_tolerance=rate_tolerance,
            )
    else:
        # Successful cases
        for si in bob_stakes:
            assert subtensor.staking.unstake_all(
                wallet=bob_wallet,
                netuid=si.netuid,
                hotkey_ss58=si.hotkey_ss58,
                rate_tolerance=rate_tolerance,
            ).success

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
    # Register first SN
    alice_sn_2 = TestSubnet(async_subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
        REGISTER_NEURON(bob_wallet),
        REGISTER_NEURON(dave_wallet),
    ]
    await alice_sn_2.async_execute_steps(steps)

    # Register second SN
    alice_sn_3 = TestSubnet(async_subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
        REGISTER_NEURON(bob_wallet),
        REGISTER_NEURON(dave_wallet),
    ]
    await alice_sn_3.async_execute_steps(steps)
    # Check Bob's stakes are empty.
    assert (
        await async_subtensor.staking.get_stake_info_for_coldkey(
            bob_wallet.coldkey.ss58_address
        )
        == []
    )

    # Bob stakes to Dave in both SNs

    assert (
        await async_subtensor.staking.add_stake(
            wallet=bob_wallet,
            netuid=alice_sn_2.netuid,
            hotkey_ss58=dave_wallet.hotkey.ss58_address,
            amount=Balance.from_tao(10000),
            period=16,
        )
    ).success, f"Cant add stake to dave in SN {alice_sn_2.netuid}"

    assert (
        await async_subtensor.staking.add_stake(
            wallet=bob_wallet,
            netuid=alice_sn_3.netuid,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            amount=Balance.from_tao(15000),
            period=16,
        )
    ).success, f"Cant add stake to dave in SN {alice_sn_3.netuid}"

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
                hotkey_ss58=bob_stakes[0].hotkey_ss58,
                rate_tolerance=rate_tolerance,
            )
    else:
        # Successful cases
        for si in bob_stakes:
            assert (
                await async_subtensor.staking.unstake_all(
                    wallet=bob_wallet,
                    hotkey_ss58=si.hotkey_ss58,
                    netuid=si.netuid,
                    rate_tolerance=rate_tolerance,
                )
            ).success

        # Make sure both unstake were successful.
        bob_stakes = await async_subtensor.staking.get_stake_info_for_coldkey(
            bob_wallet.coldkey.ss58_address
        )
        assert len(bob_stakes) == 0


def test_auto_staking(subtensor, alice_wallet, bob_wallet, eve_wallet):
    """Tests auto staking logic."""
    alice_sn = TestSubnet(subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
        REGISTER_NEURON(bob_wallet),
    ]
    alice_sn.execute_steps(steps)

    assert subtensor.staking.get_auto_stakes(alice_wallet.coldkey.ss58_address) == {}

    # set auto stake
    assert subtensor.staking.set_auto_stake(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=bob_wallet.hotkey.ss58_address,
    ).success

    # check auto stake
    assert subtensor.staking.get_auto_stakes(alice_wallet.coldkey.ss58_address) == {
        alice_sn.netuid: bob_wallet.hotkey.ss58_address
    }

    # set auto stake to nonexistent hotkey
    success, message = subtensor.staking.set_auto_stake(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=eve_wallet.hotkey.ss58_address,
    )
    assert success is False
    assert "HotKeyNotRegisteredInSubNet" in message

    # check auto stake
    assert subtensor.staking.get_auto_stakes(alice_wallet.coldkey.ss58_address) == {
        alice_sn.netuid: bob_wallet.hotkey.ss58_address
    }


@pytest.mark.asyncio
async def test_auto_staking_async(
    async_subtensor, alice_wallet, bob_wallet, eve_wallet
):
    """Tests auto staking logic."""
    alice_sn = TestSubnet(async_subtensor)
    steps = [
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
        REGISTER_NEURON(bob_wallet),
    ]
    await alice_sn.async_execute_steps(steps)

    assert (
        await async_subtensor.staking.get_auto_stakes(alice_wallet.coldkey.ss58_address)
        == {}
    )

    # set auto stake
    assert (
        await async_subtensor.staking.set_auto_stake(
            wallet=alice_wallet,
            netuid=alice_sn.netuid,
            hotkey_ss58=bob_wallet.hotkey.ss58_address,
        )
    ).success

    # check auto stake
    assert await async_subtensor.staking.get_auto_stakes(
        alice_wallet.coldkey.ss58_address
    ) == {alice_sn.netuid: bob_wallet.hotkey.ss58_address}

    # set auto stake to nonexistent hotkey
    success, message = await async_subtensor.staking.set_auto_stake(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=eve_wallet.hotkey.ss58_address,
    )
    assert success is False
    assert "HotKeyNotRegisteredInSubNet" in message

    # check auto stake
    assert await async_subtensor.staking.get_auto_stakes(
        alice_wallet.coldkey.ss58_address
    ) == {alice_sn.netuid: bob_wallet.hotkey.ss58_address}
