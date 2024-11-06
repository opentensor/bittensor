from bittensor.core.subtensor import Subtensor
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging
from tests.e2e_tests.utils.chain_interactions import (
    add_stake,
    register_neuron,
    register_subnet,
    sudo_set_hyperparameter_bool,
    sudo_set_hyperparameter_values,
)
from tests.e2e_tests.utils.e2e_test_utils import setup_wallet


def liquid_alpha_call_params(netuid: int, alpha_values: str):
    alpha_low, alpha_high = [v.strip() for v in alpha_values.split(",")]
    return {
        "netuid": netuid,
        "alpha_low": alpha_low,
        "alpha_high": alpha_high,
    }


def test_liquid_alpha(local_chain):
    """
    Test the liquid alpha mechanism

    Steps:
        1. Register a subnet through Alice
        2. Register Alice's neuron and add stake
        3. Verify we can't set alpha values without enabling liquid_alpha
        4. Test setting alpha values after enabling liquid_alpha
        5. Verify failures when setting incorrect values (upper and lower bounds)
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    u16_max = 65535
    netuid = 1
    logging.info("Testing test_liquid_alpha_enabled")

    # Register root as Alice
    keypair, alice_wallet = setup_wallet("//Alice")
    register_subnet(local_chain, alice_wallet), "Unable to register the subnet"

    # Verify subnet 1 created successfully
    assert local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()

    # Register a neuron to the subnet
    assert register_neuron(
        local_chain, alice_wallet, netuid
    ), "Unable to register Alice as a neuron"

    # Stake to become to top neuron after the first epoch
    add_stake(local_chain, alice_wallet, Balance.from_tao(100_000))

    # Assert liquid alpha is disabled
    subtensor = Subtensor(network="ws://localhost:9945")
    assert (
        subtensor.get_subnet_hyperparameters(netuid=netuid).liquid_alpha_enabled
        is False
    ), "Liquid alpha is enabled by default"

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
    assert result is False, "Alpha values set while being disabled"
    assert error_message["name"] == "LiquidAlphaDisabled"

    # Enabled liquid alpha on the subnet
    assert sudo_set_hyperparameter_bool(
        local_chain, alice_wallet, "sudo_set_liquid_alpha_enabled", True, netuid
    ), "Unable to enable liquid alpha"

    assert subtensor.get_subnet_hyperparameters(
        netuid=1
    ).liquid_alpha_enabled, "Failed to enable liquid alpha"

    # Attempt to set alpha high & low after enabling the hyperparameter
    alpha_values = "87, 54099"
    call_params = liquid_alpha_call_params(netuid, alpha_values)
    assert sudo_set_hyperparameter_values(
        local_chain,
        alice_wallet,
        call_function="sudo_set_alpha_values",
        call_params=call_params,
    ), "Unable to set alpha_values"
    assert (
        subtensor.get_subnet_hyperparameters(netuid=1).alpha_high == 54099
    ), "Failed to set alpha high"
    assert (
        subtensor.get_subnet_hyperparameters(netuid=1).alpha_low == 87
    ), "Failed to set alpha low"

    # Testing alpha high upper and lower bounds

    # 1. Test setting Alpha_high too low
    alpha_high_too_low = (
        u16_max * 4 // 5
    ) - 1  # One less than the minimum acceptable value
    call_params = liquid_alpha_call_params(netuid, f"6553, {alpha_high_too_low}")
    result, error_message = sudo_set_hyperparameter_values(
        local_chain,
        alice_wallet,
        call_function="sudo_set_alpha_values",
        call_params=call_params,
        return_error_message=True,
    )

    assert result is False, "Able to set incorrect alpha_high value"
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
    assert result is False, "Able to set incorrect alpha_low value"
    assert error_message["name"] == "AlphaLowOutOfRange"

    # 2. Test setting Alpha_low too high
    alpha_low_too_high = (
        u16_max * 4 // 5
    ) + 1  # One more than the maximum acceptable value
    call_params = liquid_alpha_call_params(netuid, f"{alpha_low_too_high}, 53083")
    result, error_message = sudo_set_hyperparameter_values(
        local_chain,
        alice_wallet,
        call_function="sudo_set_alpha_values",
        call_params=call_params,
        return_error_message=True,
    )
    assert result is False, "Able to set incorrect alpha_low value"
    assert error_message["name"] == "AlphaLowOutOfRange"

    # Setting normal alpha values
    alpha_values = "6553, 53083"
    call_params = liquid_alpha_call_params(netuid, alpha_values)
    assert sudo_set_hyperparameter_values(
        local_chain,
        alice_wallet,
        call_function="sudo_set_alpha_values",
        call_params=call_params,
    ), "Unable to set liquid alpha values"

    assert (
        subtensor.get_subnet_hyperparameters(netuid=1).alpha_high == 53083
    ), "Failed to set alpha high"
    assert (
        subtensor.get_subnet_hyperparameters(netuid=1).alpha_low == 6553
    ), "Failed to set alpha low"

    # Disable Liquid Alpha
    assert sudo_set_hyperparameter_bool(
        local_chain, alice_wallet, "sudo_set_liquid_alpha_enabled", False, netuid
    ), "Unable to disable liquid alpha"

    assert (
        subtensor.get_subnet_hyperparameters(netuid=1).liquid_alpha_enabled is False
    ), "Failed to disable liquid alpha"
    logging.info("✅ Passed test_liquid_alpha")
