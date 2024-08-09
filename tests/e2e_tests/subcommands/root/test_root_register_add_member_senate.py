import bittensor
from bittensor import logging
from bittensor.commands import (
    NominateCommand,
    RegisterCommand,
    RegisterSubnetworkCommand,
    RootRegisterCommand,
    SetTakeCommand,
    StakeCommand,
)
from bittensor.commands.senate import SenateCommand

from ...utils import setup_wallet


def test_root_register_add_member_senate(local_chain, capsys):
    logging.info("Testing test_root_register_add_member_senate")
    # Register root as Alice - the subnet owner
    alice_keypair, exec_command, wallet = setup_wallet("//Alice")
    exec_command(RegisterSubnetworkCommand, ["s", "create"])

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
            "10000",
        ],
    )

    exec_command(NominateCommand, ["root", "nominate"])

    exec_command(SetTakeCommand, ["r", "set_take", "--take", "0.8"])

    captured = capsys.readouterr()
    # Verify subnet 1 created successfully
    assert local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()
    # Query local chain for senate members
    members = local_chain.query("SenateMembers", "Members").serialize()
    assert len(members) == 3, f"Expected 3 senate members, found {len(members)}"

    # Assert subtensor has 3 senate members
    subtensor = bittensor.subtensor(network="ws://localhost:9945")
    sub_senate = len(subtensor.get_senate_members())
    assert (
        sub_senate == 3
    ), f"Root senate expected 3 members but found {sub_senate} instead."

    # Execute command and capture output
    exec_command(
        SenateCommand,
        ["root", "senate"],
    )

    captured = capsys.readouterr()

    # assert output is graph Titling "Senate" with names and addresses
    assert "Senate" in captured.out
    assert "NAME" in captured.out
    assert "ADDRESS" in captured.out
    assert "5CiPPseXPECbkjWCa6MnjNokrgYjMqmKndv2rSnekmSK2DjL" in captured.out
    assert "5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy" in captured.out
    assert "5HGjWAeFDfFCWPsjFQdVV2Msvz2XtMktvgocEZcCj68kUMaw" in captured.out

    exec_command(
        RootRegisterCommand,
        [
            "root",
            "register",
            "--wallet.hotkey",
            "default",
            "--wallet.name",
            "default",
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )
    # sudo_call_add_senate_member(local_chain, wallet)

    members = local_chain.query("SenateMembers", "Members").serialize()
    assert len(members) == 4, f"Expected 4 senate members, found {len(members)}"

    # Assert subtensor has 4 senate members
    subtensor = bittensor.subtensor(network="ws://localhost:9945")
    sub_senate = len(subtensor.get_senate_members())
    assert (
        sub_senate == 4
    ), f"Root senate expected 3 members but found {sub_senate} instead."

    exec_command(
        SenateCommand,
        ["root", "senate"],
    )

    captured = capsys.readouterr()

    # assert output is graph Titling "Senate" with names and addresses

    assert "Senate" in captured.out
    assert "NAME" in captured.out
    assert "ADDRESS" in captured.out
    assert "5CiPPseXPECbkjWCa6MnjNokrgYjMqmKndv2rSnekmSK2DjL" in captured.out
    assert "5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy" in captured.out
    assert "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY" in captured.out
    assert "5HGjWAeFDfFCWPsjFQdVV2Msvz2XtMktvgocEZcCj68kUMaw" in captured.out
