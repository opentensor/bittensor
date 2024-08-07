import bittensor
from bittensor.commands import (
    RegisterSubnetworkCommand,
    RegisterCommand,
    StakeCommand,
    NominateCommand,
    SetTakeCommand,
    RootRegisterCommand,
)
from bittensor.commands.senate import SenateCommand
from ...utils import setup_wallet


def assert_sequence(lines, sequence):
    sequence_ptr = 0
    for line in lines:
        lineset = set(line.split())
        seqset = sequence[sequence_ptr]
        if lineset.intersection(seqset) == seqset:
            sequence_ptr += 1
            if sequence_ptr >= len(sequence):
                # all items seen
                break
    assert sequence_ptr == len(
        sequence
    ), f"Did not find sequence[{sequence_ptr}] = '{sequence[sequence_ptr]}' in output"


def test_root_register_add_member_senate(local_chain, capsys):
    netuid = 1

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
            str(netuid),
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

    # Verify subnet <netuid> created successfully
    assert local_chain.query("SubtensorModule", "NetworksAdded", [netuid]).serialize()
    # Query local chain for senate members
    members = local_chain.query("SenateMembers", "Members").serialize()
    assert len(members) == 3

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
    assert_sequence(
        lines,
        (
            {"Senate"},
            {"NAME", "ADDRESS"},
            {"5CiPPseXPECbkjWCa6MnjNokrgYjMqmKndv2rSnekmSK2DjL"},
            {"5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy"},
            {"5HGjWAeFDfFCWPsjFQdVV2Msvz2XtMktvgocEZcCj68kUMaw"},
        ),
    )

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
    assert len(members) == 4

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
    assert_sequence(
        lines,
        (
            {"Senate"},
            {"NAME", "ADDRESS"},
            {"5CiPPseXPECbkjWCa6MnjNokrgYjMqmKndv2rSnekmSK2DjL"},
            {"5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy"},
            {"5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"},
            {"5HGjWAeFDfFCWPsjFQdVV2Msvz2XtMktvgocEZcCj68kUMaw"},
        ),
    )
