import pytest

from bittensor.utils import U16_MAX
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging
from tests.e2e_tests.utils import (
    TestSubnet,
    AdminUtils,
    NETUID,
    ACTIVATE_SUBNET,
    REGISTER_NEURON,
    REGISTER_SUBNET,
    SUDO_SET_ADMIN_FREEZE_WINDOW,
    SUDO_SET_ALPHA_VALUES,
    SUDO_SET_LIQUID_ALPHA_ENABLED,
)


def liquid_alpha_call_params(netuid: int, alpha_values: str):
    alpha_low, alpha_high = [v.strip() for v in alpha_values.split(",")]
    return {
        "netuid": netuid,
        "alpha_low": alpha_low,
        "alpha_high": alpha_high,
    }


def test_liquid_alpha(subtensor, alice_wallet):
    """
    Test the liquid alpha mechanism. By June 17 2025 the limits are `0.025 <= alpha_low <= alpha_high <= 1`

    Steps:
        1. Register a subnet through Alice
        2. Register Alice's neuron and add stake
        3. Verify we can't set alpha values without enabling liquid_alpha
        4. Test setting alpha values after enabling liquid_alpha
        5. Verify failures when setting incorrect values (upper and lower bounds)
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    alice_sn = TestSubnet(subtensor)
    steps = [
        SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
        REGISTER_NEURON(alice_wallet),
    ]
    alice_sn.execute_steps(steps)

    # Stake to become to top neuron after the first epoch
    assert subtensor.staking.add_stake(
        wallet=alice_wallet,
        netuid=alice_sn.netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        amount=Balance.from_tao(10_000),
    ).success, "Unable to stake to Alice neuron."

    # Assert liquid alpha is disabled
    assert (
        subtensor.subnets.get_subnet_hyperparameters(
            netuid=alice_sn.netuid
        ).liquid_alpha_enabled
        is False
    ), "Liquid alpha is enabled by default."

    # Attempt to set alpha high/low while disabled (should fail)
    alpha_values = "6553, 53083"
    call_params = liquid_alpha_call_params(alice_sn.netuid, alpha_values)

    response = alice_sn.execute_one(
        SUDO_SET_ALPHA_VALUES(alice_wallet, AdminUtils, False, **call_params)
    )
    assert response.success is False and "LiquidAlphaDisabled" in response.message, (
        "Alpha values set while being disabled."
    )

    alice_sn.execute_one(
        SUDO_SET_LIQUID_ALPHA_ENABLED(alice_wallet, AdminUtils, True, NETUID, True)
    )

    assert subtensor.subnets.get_subnet_hyperparameters(
        alice_sn.netuid,
    ).liquid_alpha_enabled, "Failed to enable liquid alpha."

    # Attempt to set alpha high & low after enabling the hyperparameter
    alpha_values = "26001, 54099"
    call_params = liquid_alpha_call_params(alice_sn.netuid, alpha_values)
    alice_sn.execute_one(
        SUDO_SET_ALPHA_VALUES(alice_wallet, AdminUtils, False, **call_params)
    )
    assert (
        subtensor.subnets.get_subnet_hyperparameters(alice_sn.netuid).alpha_high
        == 54099
    ), "Failed to set alpha high"
    assert (
        subtensor.subnets.get_subnet_hyperparameters(alice_sn.netuid).alpha_low == 26001
    ), "Failed to set alpha low."

    # Testing alpha high upper and lower bounds

    # 1. Test setting Alpha_high too low
    alpha_high_too_low = 87

    # Test needs to wait for the amount of tempo in the chain equal to OwnerHyperparamRateLimit
    owner_hyperparam_ratelimit = subtensor.queries.query_subtensor(
        "OwnerHyperparamRateLimit"
    ).value
    logging.console.info(
        f"OwnerHyperparamRateLimit is {owner_hyperparam_ratelimit} tempo(s)."
    )
    subtensor.wait_for_block(
        subtensor.block
        + subtensor.subnets.tempo(alice_sn.netuid) * owner_hyperparam_ratelimit
    )

    call_params = liquid_alpha_call_params(
        alice_sn.netuid, f"6553, {alpha_high_too_low}"
    )

    response = alice_sn.execute_one(
        SUDO_SET_ALPHA_VALUES(alice_wallet, AdminUtils, False, **call_params)
    )
    assert response.success is False and "AlphaHighTooLow" in response.message, (
        "Able to set incorrect alpha_high value."
    )

    # 2. Test setting Alpha_high too high
    alpha_high_too_high = U16_MAX + 1  # One more than the max acceptable value
    call_params = liquid_alpha_call_params(
        alice_sn.netuid, f"6553, {alpha_high_too_high}"
    )

    response = alice_sn.execute_one(
        SUDO_SET_ALPHA_VALUES(alice_wallet, AdminUtils, False, **call_params)
    )
    assert (
        response.success is False and "65536 out of range for u16" in response.message
    ), f"Unexpected error: {response}"

    # 1. Test setting Alpha_low too low
    alpha_low_too_low = 0
    call_params = liquid_alpha_call_params(
        alice_sn.netuid, f"{alpha_low_too_low}, 53083"
    )

    response = alice_sn.execute_one(
        SUDO_SET_ALPHA_VALUES(alice_wallet, AdminUtils, False, **call_params)
    )
    assert response.success is False and "AlphaLowOutOfRange" in response.message, (
        "Able to set incorrect alpha_low value."
    )

    # 2. Test setting Alpha_low too high
    alpha_low_too_high = 53084
    call_params = liquid_alpha_call_params(
        alice_sn.netuid, f"{alpha_low_too_high}, 53083"
    )
    response = alice_sn.execute_one(
        SUDO_SET_ALPHA_VALUES(alice_wallet, AdminUtils, False, **call_params)
    )
    assert response.success is False and "AlphaLowOutOfRange" in response.message, (
        "Able to set incorrect alpha_low value."
    )

    # Setting normal alpha values
    alpha_values = "6553, 53083"
    call_params = liquid_alpha_call_params(alice_sn.netuid, alpha_values)
    alice_sn.execute_one(
        SUDO_SET_ALPHA_VALUES(alice_wallet, AdminUtils, False, **call_params)
    )

    assert (
        subtensor.subnets.get_subnet_hyperparameters(alice_sn.netuid).alpha_high
        == 53083
    ), "Failed to set alpha high."
    assert (
        subtensor.subnets.get_subnet_hyperparameters(alice_sn.netuid).alpha_low == 6553
    ), "Failed to set alpha low."

    # Disable Liquid Alpha
    alice_sn.execute_one(
        SUDO_SET_LIQUID_ALPHA_ENABLED(alice_wallet, AdminUtils, True, NETUID, False)
    )

    assert (
        subtensor.subnets.get_subnet_hyperparameters(
            alice_sn.netuid
        ).liquid_alpha_enabled
        is False
    ), "Failed to disable liquid alpha."


@pytest.mark.asyncio
async def test_liquid_alpha_async(async_subtensor, alice_wallet):
    """
    Async test the liquid alpha mechanism. By June 17 2025 the limits are `0.025 <= alpha_low <= alpha_high <= 1`

    Steps:
        1. Register a subnet through Alice
        2. Register Alice's neuron and add stake
        3. Verify we can't set alpha values without enabling liquid_alpha
        4. Test setting alpha values after enabling liquid_alpha
        5. Verify failures when setting incorrect values (upper and lower bounds)
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    alice_sn = TestSubnet(async_subtensor)
    steps = [
        SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
        REGISTER_SUBNET(alice_wallet),
        ACTIVATE_SUBNET(alice_wallet),
        REGISTER_NEURON(alice_wallet),
    ]
    await alice_sn.async_execute_steps(steps)

    # Stake to become to top neuron after the first epoch
    assert (
        await async_subtensor.staking.add_stake(
            wallet=alice_wallet,
            netuid=alice_sn.netuid,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            amount=Balance.from_tao(10_000),
        )
    ).success, "Unable to stake to Alice neuron."

    # Assert liquid alpha is disabled
    assert (
        await async_subtensor.subnets.get_subnet_hyperparameters(netuid=alice_sn.netuid)
    ).liquid_alpha_enabled is False, "Liquid alpha is enabled by default."

    # Attempt to set alpha high/low while disabled (should fail)
    alpha_values = "6553, 53083"
    call_params = liquid_alpha_call_params(alice_sn.netuid, alpha_values)

    response = await alice_sn.async_execute_one(
        SUDO_SET_ALPHA_VALUES(alice_wallet, AdminUtils, False, **call_params)
    )
    assert response.success is False and "LiquidAlphaDisabled" in response.message, (
        "Alpha values set while being disabled."
    )

    await alice_sn.async_execute_one(
        SUDO_SET_LIQUID_ALPHA_ENABLED(alice_wallet, AdminUtils, True, NETUID, True)
    )

    assert (
        await async_subtensor.subnets.get_subnet_hyperparameters(alice_sn.netuid)
    ).liquid_alpha_enabled, "Failed to enable liquid alpha."

    # Attempt to set alpha high & low after enabling the hyperparameter
    alpha_values = "26001, 54099"
    call_params = liquid_alpha_call_params(alice_sn.netuid, alpha_values)
    await alice_sn.async_execute_one(
        SUDO_SET_ALPHA_VALUES(alice_wallet, AdminUtils, False, **call_params)
    )
    assert (
        await async_subtensor.subnets.get_subnet_hyperparameters(alice_sn.netuid)
    ).alpha_high == 54099, "Failed to set alpha high"
    assert (
        await async_subtensor.subnets.get_subnet_hyperparameters(alice_sn.netuid)
    ).alpha_low == 26001, "Failed to set alpha low."

    # 1. Test setting Alpha_high too low
    alpha_high_too_low = 87

    # Test needs to wait for the amount of tempo in the chain equal to OwnerHyperparamRateLimit
    owner_hyperparam_ratelimit = (
        await async_subtensor.queries.query_subtensor("OwnerHyperparamRateLimit")
    ).value
    logging.console.info(
        f"OwnerHyperparamRateLimit is {owner_hyperparam_ratelimit} tempo(s)."
    )
    await async_subtensor.wait_for_block(
        await async_subtensor.block
        + await async_subtensor.subnets.tempo(alice_sn.netuid)
        * owner_hyperparam_ratelimit
    )

    call_params = liquid_alpha_call_params(
        alice_sn.netuid, f"6553, {alpha_high_too_low}"
    )

    response = await alice_sn.async_execute_one(
        SUDO_SET_ALPHA_VALUES(alice_wallet, AdminUtils, False, **call_params)
    )
    assert response.success is False and "AlphaHighTooLow" in response.message, (
        "Able to set incorrect alpha_high value."
    )

    # 2. Test setting Alpha_high too high
    alpha_high_too_high = U16_MAX + 1  # One more than the max acceptable value
    call_params = liquid_alpha_call_params(
        alice_sn.netuid, f"6553, {alpha_high_too_high}"
    )

    response = await alice_sn.async_execute_one(
        SUDO_SET_ALPHA_VALUES(alice_wallet, AdminUtils, False, **call_params)
    )
    assert (
        response.success is False and "65536 out of range for u16" in response.message
    ), f"Unexpected error: {response}"

    # 1. Test setting Alpha_low too low
    alpha_low_too_low = 0
    call_params = liquid_alpha_call_params(
        alice_sn.netuid, f"{alpha_low_too_low}, 53083"
    )

    response = await alice_sn.async_execute_one(
        SUDO_SET_ALPHA_VALUES(alice_wallet, AdminUtils, False, **call_params)
    )
    assert response.success is False and "AlphaLowOutOfRange" in response.message, (
        "Able to set incorrect alpha_low value."
    )

    # 2. Test setting Alpha_low too high
    alpha_low_too_high = 53084
    call_params = liquid_alpha_call_params(
        alice_sn.netuid, f"{alpha_low_too_high}, 53083"
    )
    response = await alice_sn.async_execute_one(
        SUDO_SET_ALPHA_VALUES(alice_wallet, AdminUtils, False, **call_params)
    )
    assert response.success is False and "AlphaLowOutOfRange" in response.message, (
        "Able to set incorrect alpha_low value."
    )

    # Setting normal alpha values
    alpha_values = "6553, 53083"
    call_params = liquid_alpha_call_params(alice_sn.netuid, alpha_values)
    await alice_sn.async_execute_one(
        SUDO_SET_ALPHA_VALUES(alice_wallet, AdminUtils, False, **call_params)
    )

    assert (
        await async_subtensor.subnets.get_subnet_hyperparameters(alice_sn.netuid)
    ).alpha_high == 53083, "Failed to set alpha high."
    assert (
        await async_subtensor.subnets.get_subnet_hyperparameters(alice_sn.netuid)
    ).alpha_low == 6553, "Failed to set alpha low."

    # Disable Liquid Alpha
    await alice_sn.async_execute_one(
        SUDO_SET_LIQUID_ALPHA_ENABLED(alice_wallet, AdminUtils, True, NETUID, False)
    )

    assert (
        await async_subtensor.subnets.get_subnet_hyperparameters(alice_sn.netuid)
    ).liquid_alpha_enabled is False, "Failed to disable liquid alpha."
