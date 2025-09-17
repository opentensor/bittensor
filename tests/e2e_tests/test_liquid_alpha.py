from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging
from tests.e2e_tests.utils.chain_interactions import (
    sudo_set_hyperparameter_bool,
    sudo_set_hyperparameter_values,
    sudo_set_admin_utils,
)
from tests.e2e_tests.utils.e2e_test_utils import wait_to_start_call


def liquid_alpha_call_params(netuid: int, alpha_values: str):
    alpha_low, alpha_high = [v.strip() for v in alpha_values.split(",")]
    return {
        "netuid": netuid,
        "alpha_low": alpha_low,
        "alpha_high": alpha_high,
    }


def test_liquid_alpha(local_chain, subtensor, alice_wallet):
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
    # turn off admin freeze window limit for testing
    assert (
        sudo_set_admin_utils(
            local_chain,
            alice_wallet,
            call_function="sudo_set_admin_freeze_window",
            call_params={"window": 0},
        )[0]
        is True
    ), "Failed to set admin freeze window to 0"

    u16_max = 65535
    netuid = 2
    logging.console.info("Testing test_liquid_alpha_enabled")

    # Register root as Alice
    assert subtensor.register_subnet(alice_wallet), "Unable to register the subnet."

    # Verify subnet created successfully
    assert subtensor.subnet_exists(netuid)

    assert wait_to_start_call(subtensor, alice_wallet, netuid), (
        "Subnet failed to start."
    )

    # Register a neuron (Alice) to the subnet
    assert subtensor.burned_register(alice_wallet, netuid), (
        "Unable to register Alice as a neuron"
    )

    # Stake to become to top neuron after the first epoch
    assert subtensor.add_stake(
        wallet=alice_wallet,
        netuid=netuid,
        amount=Balance.from_tao(10_000),
    ), "Failed to stake"

    # Assert liquid alpha is disabled
    assert (
        subtensor.get_subnet_hyperparameters(netuid=netuid).liquid_alpha_enabled
        is False
    ), "Liquid alpha is enabled by default."

    # Attempt to set alpha high/low while disabled (should fail)
    alpha_values = "6553, 53083"
    call_params = liquid_alpha_call_params(netuid, alpha_values)
    result, error_message = sudo_set_hyperparameter_values(
        local_chain,
        alice_wallet,
        call_function="sudo_set_alpha_values",
        call_params=call_params,
        return_error_message=True,
    )
    assert result is False, "Alpha values set while being disabled."
    assert error_message["name"] == "LiquidAlphaDisabled"

    # Enabled liquid alpha on the subnet
    assert sudo_set_hyperparameter_bool(
        local_chain, alice_wallet, "sudo_set_liquid_alpha_enabled", True, netuid
    ), "Unable to enable liquid alpha"

    assert subtensor.get_subnet_hyperparameters(
        netuid,
    ).liquid_alpha_enabled, "Failed to enable liquid alpha."

    # Attempt to set alpha high & low after enabling the hyperparameter
    alpha_values = "26001, 54099"
    call_params = liquid_alpha_call_params(netuid, alpha_values)
    assert sudo_set_hyperparameter_values(
        local_chain,
        alice_wallet,
        call_function="sudo_set_alpha_values",
        call_params=call_params,
    ), "Unable to set alpha_values."
    assert subtensor.get_subnet_hyperparameters(netuid).alpha_high == 54099, (
        "Failed to set alpha high."
    )
    assert subtensor.get_subnet_hyperparameters(netuid).alpha_low == 26001, (
        "Failed to set alpha low."
    )

    # Testing alpha high upper and lower bounds

    # 1. Test setting Alpha_high too low
    alpha_high_too_low = 87

    # Test needs to wait for the amount of tempo in the chain equal to OwnerHyperparamRateLimit
    owner_hyperparam_ratelimit = subtensor.substrate.query(
        module="SubtensorModule", storage_function="OwnerHyperparamRateLimit"
    )
    logging.console.info(
        f"OwnerHyperparamRateLimit is {owner_hyperparam_ratelimit} tempo(s)."
    )
    subtensor.wait_for_block(
        subtensor.block + subtensor.tempo(netuid) * owner_hyperparam_ratelimit
    )

    call_params = liquid_alpha_call_params(netuid, f"6553, {alpha_high_too_low}")
    result, error_message = sudo_set_hyperparameter_values(
        local_chain,
        alice_wallet,
        call_function="sudo_set_alpha_values",
        call_params=call_params,
        return_error_message=True,
    )

    assert result is False, "Able to set incorrect alpha_high value."
    assert error_message["name"] == "AlphaHighTooLow"

    # 2. Test setting Alpha_high too high
    alpha_high_too_high = u16_max + 1  # One more than the max acceptable value
    call_params = liquid_alpha_call_params(netuid, f"6553, {alpha_high_too_high}")
    try:
        sudo_set_hyperparameter_values(
            local_chain,
            alice_wallet,
            call_function="sudo_set_alpha_values",
            call_params=call_params,
            return_error_message=True,
        )
    except Exception as e:
        assert str(e) == "65536 out of range for u16", f"Unexpected error: {e}"

    # Testing alpha low upper and lower bounds

    # 1. Test setting Alpha_low too low
    alpha_low_too_low = 0
    call_params = liquid_alpha_call_params(netuid, f"{alpha_low_too_low}, 53083")
    result, error_message = sudo_set_hyperparameter_values(
        local_chain,
        alice_wallet,
        call_function="sudo_set_alpha_values",
        call_params=call_params,
        return_error_message=True,
    )
    assert result is False, "Able to set incorrect alpha_low value."
    assert error_message["name"] == "AlphaLowOutOfRange"

    # 2. Test setting Alpha_low too high
    alpha_low_too_high = 53084
    call_params = liquid_alpha_call_params(netuid, f"{alpha_low_too_high}, 53083")
    result, error_message = sudo_set_hyperparameter_values(
        local_chain,
        alice_wallet,
        call_function="sudo_set_alpha_values",
        call_params=call_params,
        return_error_message=True,
    )
    assert result is False, "Able to set incorrect alpha_low value."
    assert error_message["name"] == "AlphaLowOutOfRange"

    # Setting normal alpha values
    alpha_values = "6553, 53083"
    call_params = liquid_alpha_call_params(netuid, alpha_values)
    assert sudo_set_hyperparameter_values(
        local_chain,
        alice_wallet,
        call_function="sudo_set_alpha_values",
        call_params=call_params,
    ), "Unable to set liquid alpha values."

    assert subtensor.get_subnet_hyperparameters(netuid).alpha_high == 53083, (
        "Failed to set alpha high."
    )
    assert subtensor.get_subnet_hyperparameters(netuid).alpha_low == 6553, (
        "Failed to set alpha low."
    )

    # Disable Liquid Alpha
    assert sudo_set_hyperparameter_bool(
        local_chain, alice_wallet, "sudo_set_liquid_alpha_enabled", False, netuid
    ), "Unable to disable liquid alpha."

    assert subtensor.get_subnet_hyperparameters(netuid).liquid_alpha_enabled is False, (
        "Failed to disable liquid alpha."
    )
    logging.console.info("✅ Passed test_liquid_alpha")
