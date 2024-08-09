import bittensor
from bittensor import logging
from bittensor.commands import (
    RegisterCommand,
    RegisterSubnetworkCommand,
    StakeCommand,
    SubnetSudoCommand,
)
from tests.e2e_tests.utils import setup_wallet

"""
Test the liquid alpha weights mechanism. 

Verify that:
* it can get enabled
* liquid alpha values cannot be set before the feature flag is set
* after feature flag, you can set alpha_high
* after feature flag, you can set alpha_low
"""


def test_liquid_alpha_enabled(local_chain, capsys):
    logging.info("Testing test_liquid_alpha_enabled")
    # Register root as Alice
    keypair, exec_command, wallet = setup_wallet("//Alice")
    exec_command(RegisterSubnetworkCommand, ["s", "create"])

    # hyperparameter values
    alpha_values = "6553, 53083"

    # Verify subnet 1 created successfully
    assert local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()

    # Register a neuron to the subnet
    exec_command(
        RegisterCommand,
        [
            "s",
            "register",
            "--netuid",
            "1",
        ],
    )

    # Stake to become to top neuron after the first epoch
    exec_command(
        StakeCommand,
        [
            "stake",
            "add",
            "--amount",
            "100000",
        ],
    )

    # Assert liquid alpha disabled
    subtensor = bittensor.subtensor(network="ws://localhost:9945")
    assert (
        subtensor.get_subnet_hyperparameters(netuid=1).liquid_alpha_enabled is False
    ), "Liquid alpha is enabled by default"

    # Attempt to set alpha high/low while disabled (should fail)
    result = subtensor.set_hyperparameter(
        wallet=wallet,
        netuid=1,
        parameter="alpha_values",
        value=list(map(int, alpha_values.split(","))),
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert result is None
    output = capsys.readouterr().out
    assert (
        "❌ Failed: Subtensor returned `LiquidAlphaDisabled (Module)` error. This means: \n`Attempting to set alpha high/low while disabled`"
        in output
    )

    # Enable Liquid Alpha
    exec_command(
        SubnetSudoCommand,
        [
            "sudo",
            "set",
            "hyperparameters",
            "--netuid",
            "1",
            "--wallet.name",
            wallet.name,
            "--param",
            "liquid_alpha_enabled",
            "--value",
            "True",
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    assert subtensor.get_subnet_hyperparameters(
        netuid=1
    ).liquid_alpha_enabled, "Failed to enable liquid alpha"

    output = capsys.readouterr().out
    assert "✅ Hyper parameter liquid_alpha_enabled changed to True" in output

    exec_command(
        SubnetSudoCommand,
        [
            "sudo",
            "set",
            "hyperparameters",
            "--netuid",
            "1",
            "--wallet.name",
            wallet.name,
            "--param",
            "alpha_values",
            "--value",
            "87, 54099",
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )
    assert (
        subtensor.get_subnet_hyperparameters(netuid=1).alpha_high == 54099
    ), "Failed to set alpha high"
    assert (
        subtensor.get_subnet_hyperparameters(netuid=1).alpha_low == 87
    ), "Failed to set alpha low"

    u16_max = 65535
    # Set alpha high too low
    alpha_high_too_low = (
        u16_max * 4 // 5
    ) - 1  # One less than the minimum acceptable value
    result = subtensor.set_hyperparameter(
        wallet=wallet,
        netuid=1,
        parameter="alpha_values",
        value=[6553, alpha_high_too_low],
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert result is None
    output = capsys.readouterr().out
    assert (
        "❌ Failed: Subtensor returned `AlphaHighTooLow (Module)` error. This means: \n`Alpha high is too low: alpha_high > 0.8`"
        in output
    )

    alpha_high_too_high = u16_max + 1  # One more than the max acceptable value
    try:
        result = subtensor.set_hyperparameter(
            wallet=wallet,
            netuid=1,
            parameter="alpha_values",
            value=[6553, alpha_high_too_high],
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
        assert result is None, "Expected not to be able to set alpha value above u16"
    except Exception as e:
        assert str(e) == "65536 out of range for u16", f"Unexpected error: {e}"

    # Set alpha low too low
    alpha_low_too_low = 0
    result = subtensor.set_hyperparameter(
        wallet=wallet,
        netuid=1,
        parameter="alpha_values",
        value=[alpha_low_too_low, 53083],
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert result is None
    output = capsys.readouterr().out
    assert (
        "❌ Failed: Subtensor returned `AlphaLowOutOfRange (Module)` error. This means: \n`Alpha low is out of range: alpha_low > 0 && alpha_low < 0.8`"
        in output
    )

    # Set alpha low too high
    alpha_low_too_high = (
        u16_max * 4 // 5
    ) + 1  # One more than the maximum acceptable value
    result = subtensor.set_hyperparameter(
        wallet=wallet,
        netuid=1,
        parameter="alpha_values",
        value=[alpha_low_too_high, 53083],
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert result is None
    output = capsys.readouterr().out
    assert (
        "❌ Failed: Subtensor returned `AlphaLowOutOfRange (Module)` error. This means: \n`Alpha low is out of range: alpha_low > 0 && alpha_low < 0.8`"
        in output
    )

    exec_command(
        SubnetSudoCommand,
        [
            "sudo",
            "set",
            "hyperparameters",
            "--netuid",
            "1",
            "--wallet.name",
            wallet.name,
            "--param",
            "alpha_values",
            "--value",
            alpha_values,
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    assert (
        subtensor.get_subnet_hyperparameters(netuid=1).alpha_high == 53083
    ), "Failed to set alpha high"
    assert (
        subtensor.get_subnet_hyperparameters(netuid=1).alpha_low == 6553
    ), "Failed to set alpha low"

    output = capsys.readouterr().out
    assert "✅ Hyper parameter alpha_values changed to [6553.0, 53083.0]" in output

    # Disable Liquid Alpha
    exec_command(
        SubnetSudoCommand,
        [
            "sudo",
            "set",
            "hyperparameters",
            "--netuid",
            "1",
            "--wallet.name",
            wallet.name,
            "--param",
            "liquid_alpha_enabled",
            "--value",
            "False",
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    assert (
        subtensor.get_subnet_hyperparameters(netuid=1).liquid_alpha_enabled is False
    ), "Failed to disable liquid alpha"

    output = capsys.readouterr().out
    assert "✅ Hyper parameter liquid_alpha_enabled changed to False" in output

    result = subtensor.set_hyperparameter(
        wallet=wallet,
        netuid=1,
        parameter="alpha_values",
        value=list(map(int, alpha_values.split(","))),
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert result is None
    output = capsys.readouterr().out
    assert (
        "❌ Failed: Subtensor returned `LiquidAlphaDisabled (Module)` error. This means: \n`Attempting to set alpha high/low while disabled`"
        in output
    )
    logging.info("Passed test_liquid_alpha_enabled")
