import pytest

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


@pytest.mark.asyncio
async def test_root_register_add_member_senate(local_chain, capsys):
    # Register root as Alice - the subnet owner
    alice_keypair, exec_command, wallet = await setup_wallet("//Alice")
    await exec_command(RegisterSubnetworkCommand, ["s", "create"])

    # Register a neuron to the subnet
    await exec_command(
        RegisterCommand,
        [
            "s",
            "register",
            "--netuid",
            "1",
        ],
    )

    # Stake to become to top neuron after the first epoch
    await exec_command(
        StakeCommand,
        [
            "stake",
            "add",
            "--amount",
            "10000",
        ],
    )

    await exec_command(NominateCommand, ["root", "nominate"])

    await exec_command(SetTakeCommand, ["r", "set_take", "--take", "0.8"])

    # Verify subnet 1 created successfully
    assert local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()
    # Query local chain for senate members
    members = local_chain.query("SenateMembers", "Members").serialize()
    assert len(members) == 3

    # Assert subtensor has 3 senate members
    subtensor = bittensor.subtensor(network="ws://localhost:9945")
    sub_senate = len(await subtensor.get_senate_members())
    assert (
        sub_senate == 3
    ), f"Root senate expected 3 members but found {sub_senate} instead."

    # Execute command and capture output
    await exec_command(
        SenateCommand,
        [
            "root",
            "senate",
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    captured = capsys.readouterr()
    lines = captured.out.splitlines()

    # assert output is graph Titling "Senate" with names and addresses
    assert "Senate" in lines[14].strip().split()
    assert "NAME" in lines[15].strip().split()
    assert "ADDRESS" in lines[15].strip().split()
    assert (
        "5CiPPseXPECbkjWCa6MnjNokrgYjMqmKndv2rSnekmSK2DjL" in lines[16].strip().split()
    )
    assert (
        "5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy" in lines[17].strip().split()
    )
    assert (
        "5HGjWAeFDfFCWPsjFQdVV2Msvz2XtMktvgocEZcCj68kUMaw" in lines[18].strip().split()
    )

    await exec_command(
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
    sub_senate = len(await subtensor.get_senate_members())
    assert (
        sub_senate == 4
    ), f"Root senate expected 3 members but found {sub_senate} instead."

    await exec_command(
        SenateCommand,
        ["root", "senate"],
    )

    captured = capsys.readouterr()
    lines = captured.out.splitlines()

    # assert output is graph Titling "Senate" with names and addresses
    assert "Senate" in lines[2].strip().split()
    assert "NAME" in lines[3].strip().split()
    assert "ADDRESS" in lines[3].strip().split()
    assert (
        "5CiPPseXPECbkjWCa6MnjNokrgYjMqmKndv2rSnekmSK2DjL" in lines[4].strip().split()
    )
    assert (
        "5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy" in lines[5].strip().split()
    )
    assert (
        "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY" in lines[6].strip().split()
    )
    assert (
        "5HGjWAeFDfFCWPsjFQdVV2Msvz2XtMktvgocEZcCj68kUMaw" in lines[7].strip().split()
    )
