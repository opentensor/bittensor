import bittensor
from bittensor.commands import (
    RegisterCommand,
    StakeCommand,
    RegisterSubnetworkCommand,
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
    # Register root as Alice
    keypair, exec_command, wallet = setup_wallet("//Alice")
    exec_command(RegisterSubnetworkCommand, ["s", "create"])

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
    capsys.readouterr()
    subtensor.set_hyperparameter(
        wallet=wallet,
        netuid=1,
        parameter="alpha_high",
        value=0.3,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    output = capsys.readouterr()
    assert (
        output.out
        == "❌ Failed: Subtensor returned `LiquidAlphaDisabled (Module)` error. This means: \n`Attempting to set alpha high/low while disabled`\n"
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

    subtensor = bittensor.subtensor(network="ws://localhost:9945")
    assert subtensor.get_subnet_hyperparameters(
        netuid=1
    ).liquid_alpha_enabled, "Failed to enable liquid alpha"

    output = capsys.readouterr().out
    assert "✅ Hyper parameter liquid_alpha_enabled changed to True" in output

    # set high value
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
            "alpha_high",
            "--value",
            "53083",
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    subtensor = bittensor.subtensor(network="ws://localhost:9945")
    assert (
        subtensor.get_subnet_hyperparameters(netuid=1).alpha_high == 53083
    ), "Failed to set alpha high"

    output = capsys.readouterr().out
    assert "✅ Hyper parameter alpha_high changed to 53083" in output

    # Set low value
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
            "alpha_low",
            "--value",
            "6553",
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    subtensor = bittensor.subtensor(network="ws://localhost:9945")
    assert (
        subtensor.get_subnet_hyperparameters(netuid=1).alpha_low == 6553
    ), "Failed to set alpha low"

    output = capsys.readouterr().out
    assert "✅ Hyper parameter alpha_low changed to 6553" in output
