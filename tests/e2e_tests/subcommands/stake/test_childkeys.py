import bittensor
import pytest
from bittensor.commands import (
    RegisterCommand,
    StakeCommand,
    RegisterSubnetworkCommand,
    SetChildrenCommand,
    RevokeChildrenCommand,
    GetChildrenCommand,
)
from bittensor.commands.stake import SetChildKeyTakeCommand, GetChildKeyTakeCommand
from bittensor.extrinsics.staking import prepare_child_proportions
from tests.e2e_tests.utils import setup_wallet, wait_interval


@pytest.mark.asyncio
async def test_set_revoke_children_multiple(local_chain, capsys):
    """
    Test the setting and revoking of children hotkeys for staking.

    This test case covers the following scenarios:
    1. Setting multiple children hotkeys with specified proportions
    2. Retrieving children information
    3. Revoking all children hotkeys
    4. Verifying the absence of children after revocation

    The test uses three wallets (Alice, Bob, and Eve) and performs operations
    on a local blockchain.

    Args:
        local_chain: A fixture providing access to the local blockchain
        capsys: A pytest fixture for capturing stdout and stderr

    The test performs the following steps:
    - Set up wallets for Alice, Bob, and Eve
    - Create a subnet and register wallets
    - Add stake to Alice's wallet
    - Set Bob and Eve as children of Alice with specific proportions
    - Verify the children are set correctly
    - Get and verify children information
    - Revoke all children
    - Verify children are revoked
    - Check that no children exist after revocation

    This test ensures the proper functioning of setting children hotkeys,
    retrieving children information, and revoking children in the staking system.
    """
    # Setup
    alice_keypair, alice_exec_command, alice_wallet = setup_wallet("//Alice")
    bob_keypair, bob_exec_command, bob_wallet = setup_wallet("//Bob")
    eve_keypair, eve_exec_command, eve_wallet = setup_wallet("//Eve")

    alice_exec_command(RegisterSubnetworkCommand, ["s", "create"])
    assert local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()

    for exec_command in [alice_exec_command, bob_exec_command, eve_exec_command]:
        exec_command(RegisterCommand, ["s", "register", "--netuid", "1"])

    alice_exec_command(StakeCommand, ["stake", "add", "--amount", "100000"])

    async def wait():
        # wait rate limit, until we are allowed to get children

        rate_limit = (
                subtensor.query_constant(
                    module_name="SubtensorModule", constant_name="InitialTempo"
                ).value
                * 2
        )
        curr_block = subtensor.get_current_block()
        await wait_interval(rate_limit + curr_block + 1, subtensor)

    subtensor = bittensor.subtensor(network="ws://localhost:9945")

    await wait()

    children_with_proportions = [
        [0.4, bob_keypair.ss58_address],
        [0.2, eve_keypair.ss58_address],
    ]

    # Test 1: Set multiple children
    alice_exec_command(
        SetChildrenCommand,
        [
            "stake",
            "set_children",
            "--netuid",
            "1",
            "--children",
            f"{children_with_proportions[0][1]},{children_with_proportions[1][1]}",
            "--hotkey",
            str(alice_keypair.ss58_address),
            "--proportions",
            f"{children_with_proportions[0][0]},{children_with_proportions[1][0]}",
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    await wait()

    subtensor = bittensor.subtensor(network="ws://localhost:9945")
    children_info = subtensor.get_children(hotkey=alice_keypair.ss58_address, netuid=1)

    assert len(children_info) == 2, "Failed to set children hotkeys"

    normalized_proportions = prepare_child_proportions(children_with_proportions)
    assert (
            children_info[0][0] == normalized_proportions[0][0]
            and children_info[1][0] == normalized_proportions[1][0]
    ), "Incorrect proportions set"

    # Test 2: Get children information
    alice_exec_command(
        GetChildrenCommand,
        [
            "stake",
            "get_children",
            "--netuid",
            "1",
            "--hotkey",
            str(alice_keypair.ss58_address),
        ],
    )
    output = capsys.readouterr().out
    assert "ChildHotkey" in output
    assert "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92Uh… │ 40.0%" in output
    assert "5HGjWAeFDfFCWPsjFQdVV2Msvz2XtMktvgocEZcC… │ 20.0%" in output
    assert "Total                                     │     60.0%" in output

    await wait()

    # Test 3: Revoke all children
    alice_exec_command(
        RevokeChildrenCommand,
        [
            "stake",
            "revoke_children",
            "--netuid",
            "1",
            "--hotkey",
            str(alice_keypair.ss58_address),
            "--children",
            f"{children_with_proportions[0][1]},{children_with_proportions[1][1]}",
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    await wait()

    assert (
            subtensor.get_children(netuid=1, hotkey=alice_keypair.ss58_address) == []
    ), "Failed to revoke children hotkeys"

    await wait()
    # Test 4: Get children after revocation
    alice_exec_command(
        GetChildrenCommand,
        [
            "stake",
            "get_children",
            "--netuid",
            "1",
            "--hotkey",
            str(alice_keypair.ss58_address),
        ],
    )
    output = capsys.readouterr().out
    assert "There are currently no child hotkeys on subnet" in output


@pytest.mark.asyncio
async def test_set_revoke_childkey_take(local_chain, capsys):
    """
    Test the setting and retrieving of childkey take amounts for staking.

    This test case covers the following scenarios:
    1. Setting a childkey take amount for a specific hotkey
    2. Retrieving the childkey take amount
    3. Verifying the retrieved childkey take amount

    The test uses one wallet (Alice) and performs operations
    on a local blockchain.

    Args:
        local_chain: A fixture providing access to the local blockchain
        capsys: A pytest fixture for capturing stdout and stderr

    The test performs the following steps:
    - Set up wallets for Alice, Bob, and Eve
    - Create a subnet and register wallets
    - Set a childkey take amount for Alice
    - Verify the setting operation was successful
    - Retrieve the set childkey take amount
    - Verify the retrieved amount is correct 

    This test ensures the proper functioning of setting and retrieving
    childkey take amounts in the staking system.
    """
    # Setup
    alice_keypair, alice_exec_command, alice_wallet = setup_wallet("//Alice")

    alice_exec_command(RegisterSubnetworkCommand, ["s", "create"])
    assert local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()

    for exec_command in [alice_exec_command]:
        exec_command(RegisterCommand, ["s", "register", "--netuid", "1"])

    # Test 1: Set multiple children
    alice_exec_command(
        SetChildKeyTakeCommand,
        [
            "stake",
            "set_childkey_take",
            "--netuid",
            "1",
            "--hotkey",
            str(alice_keypair.ss58_address),
            "--take",
            "0.12",
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    output = capsys.readouterr().out
    assert "Set childkey take." in output

    # Test 1: Set multiple children
    alice_exec_command(
        GetChildKeyTakeCommand,
        [
            "stake",
            "get_childkey_take",
            "--netuid",
            "1",
            "--hotkey",
            str(alice_keypair.ss58_address),
        ],
    )

    output = capsys.readouterr().out
    assert f"The childkey take for {alice_keypair.ss58_address} is \n11.999694819562066%" in output
    
    
@pytest.mark.asyncio
async def test_set_revoke_children_singular(local_chain, capsys):
    """
    Test the setting and revoking of children hotkeys for staking.

    This test case covers the following scenarios:
    1. Setting multiple children hotkeys with specified proportions (set one at a time)
    2. Retrieving children information
    3. Revoking children hotkeys (one at a time)
    4. Verifying the absence of children after revocation

    The test uses three wallets (Alice, Bob, and Eve) and performs operations
    on a local blockchain.

    Args:
        local_chain: A fixture providing access to the local blockchain
        capsys: A pytest fixture for capturing stdout and stderr

    The test performs the following steps:
    - Set up wallets for Alice, Bob, and Eve
    - Create a subnet and register wallets
    - Add stake to Alice's wallet
    - Set Bob and Eve as children of Alice with specific proportions
    - Verify the children are set correctly
    - Get and verify children information
    - Revoke all children
    - Verify children are revoked
    - Check that no children exist after revocation

    This test ensures the proper functioning of setting children hotkeys,
    retrieving children information, and revoking children in the staking system.
    """
    # Setup
    alice_keypair, alice_exec_command, alice_wallet = setup_wallet("//Alice")
    bob_keypair, bob_exec_command, bob_wallet = setup_wallet("//Bob")
    eve_keypair, eve_exec_command, eve_wallet = setup_wallet("//Eve")

    alice_exec_command(RegisterSubnetworkCommand, ["s", "create"])
    assert local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()

    for exec_command in [alice_exec_command, bob_exec_command, eve_exec_command]:
        exec_command(RegisterCommand, ["s", "register", "--netuid", "1"])

    alice_exec_command(StakeCommand, ["stake", "add", "--amount", "100000"])

    async def wait():
        # wait rate limit, until we are allowed to get children

        rate_limit = (
                subtensor.query_constant(
                    module_name="SubtensorModule", constant_name="InitialTempo"
                ).value
                * 2
        )
        curr_block = subtensor.get_current_block()
        await wait_interval(rate_limit + curr_block + 1, subtensor)

    subtensor = bittensor.subtensor(network="ws://localhost:9945")

    await wait()

    children_with_proportions = [
        [0.6, bob_keypair.ss58_address],
        [0.4, eve_keypair.ss58_address],
    ]

    # Test 1: Set first children
    alice_exec_command(
        SetChildrenCommand,
        [
            "stake",
            "set_children",
            "--netuid",
            "1",
            "--children",
            f"{children_with_proportions[0][1]}",
            "--hotkey",
            str(alice_keypair.ss58_address),
            "--proportions",
            f"{children_with_proportions[0][0]}",
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    await wait()

    subtensor = bittensor.subtensor(network="ws://localhost:9945")
    children_info = subtensor.get_children(hotkey=alice_keypair.ss58_address, netuid=1)
    
    assert len(children_info) == 1, "Failed to set child hotkeys"
    
    # Test 2: Set second child
    alice_exec_command(
        SetChildrenCommand,
        [
            "stake",
            "set_children",
            "--netuid",
            "1",
            "--children",
            f"{children_with_proportions[1][1]}",
            "--hotkey",
            str(alice_keypair.ss58_address),
            "--proportions",
            f"{children_with_proportions[1][0]}",
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    await wait()

    subtensor = bittensor.subtensor(network="ws://localhost:9945")
    children_info = subtensor.get_children(hotkey=alice_keypair.ss58_address, netuid=1)

    assert len(children_info) == 2, "Failed to set child hotkey"
    
    normalized_proportions = prepare_child_proportions(children_with_proportions)
    assert (
            children_info[0][0] == normalized_proportions[1][0]
            and children_info[1][0] == normalized_proportions[0][0]
    ), "Incorrect proportions set"

    # Test 2: Get children information
    alice_exec_command(
        GetChildrenCommand,
        [
            "stake",
            "get_children",
            "--netuid",
            "1",
            "--hotkey",
            str(alice_keypair.ss58_address),
        ],
    )
    output = capsys.readouterr().out
    assert "ChildHotkey" in output
    assert "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92Uh… │ 60.0%" in output
    assert "5HGjWAeFDfFCWPsjFQdVV2Msvz2XtMktvgocEZcC… │ 40.0%" in output
    assert "Total                                     │    100.0%" in output

    await wait()

    # Test 3: Revoke 1 child
    alice_exec_command(
        RevokeChildrenCommand,
        [
            "stake",
            "revoke_children",
            "--netuid",
            "1",
            "--hotkey",
            str(alice_keypair.ss58_address),
            "--children",
            f"{children_with_proportions[0][1]}",
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )
    await wait()
    
    subtensor = bittensor.subtensor(network="ws://localhost:9945")
    children_info = subtensor.get_children(hotkey=alice_keypair.ss58_address, netuid=1)
    assert len(children_info) == 1, "Failed to revoke child hotkey"
    
    # Test 4: Revoke second child
    alice_exec_command(
        RevokeChildrenCommand,
        [
            "stake",
            "revoke_children",
            "--netuid",
            "1",
            "--hotkey",
            str(alice_keypair.ss58_address),
            "--children",
            f"{children_with_proportions[1][1]}",
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )
    await wait()
    subtensor = bittensor.subtensor(network="ws://localhost:9945")
    children_info = subtensor.get_children(hotkey=alice_keypair.ss58_address, netuid=1)
    assert len(children_info) == 0, "Failed to revoke child hotkey"
    # Test 4: Get children after revocation
    alice_exec_command(
        GetChildrenCommand,
        [
            "stake",
            "get_children",
            "--netuid",
            "1",
            "--hotkey",
            str(alice_keypair.ss58_address),
        ],
    )
    output = capsys.readouterr().out
    assert "There are currently no child hotkeys on subnet" in output

