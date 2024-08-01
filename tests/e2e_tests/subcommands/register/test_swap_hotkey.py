import asyncio
import sys
import logging
import uuid

import pytest

import bittensor
from bittensor.commands import (
    RegisterCommand,
    RegisterSubnetworkCommand,
    SwapHotkeyCommand,
    StakeCommand,
    RootRegisterCommand,
    NewHotkeyCommand,
    ListCommand,
)
from tests.e2e_tests.utils import (
    setup_wallet,
    template_path,
    templates_repo,
    wait_interval,
)

logging.basicConfig(level=logging.INFO)

"""
Test the swap_hotkey mechanism. 

Verify that:
* Alice - neuron is registered on network as a validator
* Bob - neuron is registered on network as a miner
* Swap hotkey of Alice via BTCLI
* verify that the hotkey is swapped
* verify that stake hotkey, delegates hotkey, UIDS and prometheus hotkey is swapped
"""


@pytest.mark.asyncio
async def test_swap_hotkey_validator_owner(local_chain):
    # Register root as Alice - the subnet owner and validator
    alice_keypair, alice_exec_command, alice_wallet = setup_wallet("//Alice")
    alice_exec_command(RegisterSubnetworkCommand, ["s", "create"])
    # Verify subnet 1 created successfully
    assert local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()

    # Register Bob as miner
    bob_keypair, bob_exec_command, bob_wallet = setup_wallet("//Bob")

    alice_old_hotkey_address = alice_wallet.hotkey.ss58_address

    # Register Alice as neuron to the subnet
    alice_exec_command(
        RegisterCommand,
        [
            "s",
            "register",
            "--netuid",
            "1",
        ],
    )

    # Register Bob as neuron to the subnet
    bob_exec_command(
        RegisterCommand,
        [
            "s",
            "register",
            "--netuid",
            "1",
        ],
    )

    subtensor = bittensor.subtensor(network="ws://localhost:9945")
    # assert two neurons are in network
    assert len(subtensor.neurons(netuid=1)) == 2

    # register Bob as miner
    cmd = " ".join(
        [
            f"{sys.executable}",
            f'"{template_path}{templates_repo}/neurons/miner.py"',
            "--no_prompt",
            "--netuid",
            "1",
            "--subtensor.network",
            "local",
            "--subtensor.chain_endpoint",
            "ws://localhost:9945",
            "--wallet.path",
            bob_wallet.path,
            "--wallet.name",
            bob_wallet.name,
            "--wallet.hotkey",
            "default",
            "--logging.trace",
        ]
    )

    await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    await asyncio.sleep(
        5
    )  # wait for 5 seconds for the metagraph to refresh with latest data

    # register Alice as validator
    cmd = " ".join(
        [
            f"{sys.executable}",
            f'"{template_path}{templates_repo}/neurons/validator.py"',
            "--no_prompt",
            "--netuid",
            "1",
            "--subtensor.network",
            "local",
            "--subtensor.chain_endpoint",
            "ws://localhost:9945",
            "--wallet.path",
            alice_wallet.path,
            "--wallet.name",
            alice_wallet.name,
            "--wallet.hotkey",
            "default",
            "--logging.trace",
        ]
    )
    # run validator in the background

    await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    await asyncio.sleep(
        5
    )  # wait for 5 seconds for the metagraph and subtensor to refresh with latest data

    # register validator with root network
    alice_exec_command(
        RootRegisterCommand,
        [
            "root",
            "register",
            "--netuid",
            "1",
            "--wallet.name",
            "default",
            "--wallet.hotkey",
            "default",
            "--subtensor.chain_endpoint",
            "ws://localhost:9945",
        ],
    )

    # Alice to stake to become to top neuron after the first epoch
    alice_exec_command(
        StakeCommand,
        [
            "stake",
            "add",
            "--amount",
            "10000",
        ],
    )

    # get latest metagraph
    metagraph = bittensor.metagraph(netuid=1, network="ws://localhost:9945")
    subtensor = bittensor.subtensor(network="ws://localhost:9945")

    # assert alice has old hotkey
    alice_neuron = metagraph.neurons[0]

    # get current number of hotkeys
    wallet_tree = alice_exec_command(ListCommand, ["w", "list"], "get_tree")
    num_hotkeys = len(wallet_tree.children[0].children)

    assert alice_neuron.coldkey == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    assert alice_neuron.hotkey == alice_old_hotkey_address
    assert (
        alice_neuron.stake_dict["5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"].tao
        == 10000.0
    )
    assert alice_neuron.hotkey == alice_neuron.coldkey
    assert alice_neuron.hotkey == subtensor.get_all_subnets_info()[1].owner_ss58
    assert alice_neuron.coldkey == subtensor.get_hotkey_owner(alice_old_hotkey_address)
    assert subtensor.is_hotkey_delegate(alice_neuron.hotkey) is True
    assert (
        subtensor.is_hotkey_registered_on_subnet(
            hotkey_ss58=alice_neuron.hotkey, netuid=1
        )
        is True
    )
    assert (
        subtensor.get_uid_for_hotkey_on_subnet(
            hotkey_ss58=alice_neuron.hotkey, netuid=1
        )
        == alice_neuron.uid
    )
    if num_hotkeys > 1:
        logging.info(f"You have {num_hotkeys} hotkeys for Alice.")

    # generate new guid name for hotkey
    new_hotkey_name = str(uuid.uuid4())

    # create a new hotkey
    alice_exec_command(
        NewHotkeyCommand,
        [
            "w",
            "new_hotkey",
            "--wallet.name",
            alice_wallet.name,
            "--wallet.hotkey",
            new_hotkey_name,
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    # wait rate limit, until we are allowed to change hotkeys
    rate_limit = subtensor.tx_rate_limit()
    curr_block = subtensor.get_current_block()
    await wait_interval(rate_limit + curr_block + 1, subtensor)

    # swap hotkey
    alice_exec_command(
        SwapHotkeyCommand,
        [
            "w",
            "swap_hotkey",
            "--wallet.name",
            alice_wallet.name,
            "--wallet.hotkey",
            alice_wallet.hotkey_str,
            "--wallet.hotkey_b",
            new_hotkey_name,
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    # get latest metagraph
    metagraph = bittensor.metagraph(netuid=1, network="ws://localhost:9945")
    subtensor = bittensor.subtensor(network="ws://localhost:9945")

    # assert Alice has new hotkey
    alice_neuron = metagraph.neurons[0]
    wallet_tree = alice_exec_command(ListCommand, ["w", "list"], "get_tree")
    new_num_hotkeys = len(wallet_tree.children[0].children)

    assert (
        alice_neuron.coldkey == "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    )  # cold key didnt change
    assert alice_neuron.hotkey != alice_old_hotkey_address
    assert alice_neuron.hotkey != alice_neuron.coldkey
    assert (
        alice_neuron.coldkey == subtensor.get_all_subnets_info()[1].owner_ss58
    )  # new hotkey address is subnet owner
    assert alice_neuron.coldkey != subtensor.get_hotkey_owner(
        alice_old_hotkey_address
    )  # old key is NOT owner
    assert alice_neuron.coldkey == subtensor.get_hotkey_owner(
        alice_neuron.hotkey
    )  # new key is owner
    assert (
        subtensor.is_hotkey_delegate(alice_neuron.hotkey) is True
    )  # new key is delegate
    assert (  # new key is registered on subnet
        subtensor.is_hotkey_registered_on_subnet(
            hotkey_ss58=alice_neuron.hotkey, netuid=1
        )
        is True
    )
    assert (  # old key is NOT registered on subnet
        subtensor.is_hotkey_registered_on_subnet(
            hotkey_ss58=alice_old_hotkey_address, netuid=1
        )
        is False
    )
    assert (  # uid is unchanged
        subtensor.get_uid_for_hotkey_on_subnet(
            hotkey_ss58=alice_neuron.hotkey, netuid=1
        )
        == alice_neuron.uid
    )
    assert new_num_hotkeys == num_hotkeys + 1


"""
Test the swap_hotkey mechanism. 

Verify that:
* Alice - neuron is registered on network as a validator
* Bob - neuron is registered on network as a miner
* Swap hotkey of Bob via BTCLI
* verify that the hotkey is swapped
* verify that stake hotkey, delegates hotkey, UIDS and prometheus hotkey is swapped
"""


@pytest.mark.asyncio
async def test_swap_hotkey_miner(local_chain):
    # Register root as Alice - the subnet owner and validator
    alice_keypair, alice_exec_command, alice_wallet = setup_wallet("//Alice")
    alice_exec_command(RegisterSubnetworkCommand, ["s", "create"])
    # Verify subnet 1 created successfully
    assert local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()

    # Register Bob as miner
    bob_keypair, bob_exec_command, bob_wallet = setup_wallet("//Bob")

    bob_old_hotkey_address = bob_wallet.hotkey.ss58_address

    # Register Alice as neuron to the subnet
    alice_exec_command(
        RegisterCommand,
        [
            "s",
            "register",
            "--netuid",
            "1",
        ],
    )

    # Register Bob as neuron to the subnet
    bob_exec_command(
        RegisterCommand,
        [
            "s",
            "register",
            "--netuid",
            "1",
        ],
    )

    subtensor = bittensor.subtensor(network="ws://localhost:9945")
    # assert two neurons are in network
    assert len(subtensor.neurons(netuid=1)) == 2

    # register Bob as miner
    cmd = " ".join(
        [
            f"{sys.executable}",
            f'"{template_path}{templates_repo}/neurons/miner.py"',
            "--no_prompt",
            "--netuid",
            "1",
            "--subtensor.network",
            "local",
            "--subtensor.chain_endpoint",
            "ws://localhost:9945",
            "--wallet.path",
            bob_wallet.path,
            "--wallet.name",
            bob_wallet.name,
            "--wallet.hotkey",
            "default",
            "--logging.trace",
        ]
    )

    await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # register Alice as validator
    cmd = " ".join(
        [
            f"{sys.executable}",
            f'"{template_path}{templates_repo}/neurons/validator.py"',
            "--no_prompt",
            "--netuid",
            "1",
            "--subtensor.network",
            "local",
            "--subtensor.chain_endpoint",
            "ws://localhost:9945",
            "--wallet.path",
            alice_wallet.path,
            "--wallet.name",
            alice_wallet.name,
            "--wallet.hotkey",
            "default",
            "--logging.trace",
        ]
    )
    # run validator in the background

    await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    await asyncio.sleep(
        5
    )  # wait for 5 seconds for the metagraph and subtensor to refresh with latest data

    # register validator with root network
    alice_exec_command(
        RootRegisterCommand,
        [
            "root",
            "register",
            "--netuid",
            "1",
        ],
    )

    # Alice to stake to become to top neuron after the first epoch
    alice_exec_command(
        StakeCommand,
        [
            "stake",
            "add",
            "--amount",
            "10000",
        ],
    )

    # get latest metagraph
    metagraph = bittensor.metagraph(netuid=1, network="ws://localhost:9945")
    subtensor = bittensor.subtensor(network="ws://localhost:9945")

    # assert bob has old hotkey
    bob_neuron = metagraph.neurons[1]

    # get current number of hotkeys
    wallet_tree = bob_exec_command(ListCommand, ["w", "list"], "get_tree")
    num_hotkeys = len(wallet_tree.children[0].children)

    assert bob_neuron.coldkey == "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    assert bob_neuron.hotkey == bob_old_hotkey_address
    assert bob_neuron.hotkey == bob_neuron.coldkey
    assert bob_neuron.coldkey == subtensor.get_hotkey_owner(bob_old_hotkey_address)
    assert subtensor.is_hotkey_delegate(bob_neuron.hotkey) is False
    assert (
        subtensor.is_hotkey_registered_on_subnet(
            hotkey_ss58=bob_neuron.hotkey, netuid=1
        )
        is True
    )
    assert (
        subtensor.get_uid_for_hotkey_on_subnet(hotkey_ss58=bob_neuron.hotkey, netuid=1)
        == bob_neuron.uid
    )
    if num_hotkeys > 1:
        logging.info(f"You have {num_hotkeys} hotkeys for Bob.")

    # generate new guid name for hotkey
    new_hotkey_name = str(uuid.uuid4())

    # create a new hotkey
    bob_exec_command(
        NewHotkeyCommand,
        [
            "w",
            "new_hotkey",
            "--wallet.name",
            bob_wallet.name,
            "--wallet.hotkey",
            new_hotkey_name,
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    # wait rate limit, until we are allowed to change hotkeys
    rate_limit = subtensor.tx_rate_limit()
    curr_block = subtensor.get_current_block()
    await wait_interval(rate_limit + curr_block + 1, subtensor)

    # swap hotkey
    bob_exec_command(
        SwapHotkeyCommand,
        [
            "w",
            "swap_hotkey",
            "--wallet.name",
            bob_wallet.name,
            "--wallet.hotkey",
            bob_wallet.hotkey_str,
            "--wallet.hotkey_b",
            new_hotkey_name,
            "--wait_for_inclusion",
            "True",
            "--wait_for_finalization",
            "True",
        ],
    )

    # get latest metagraph
    metagraph = bittensor.metagraph(netuid=1, network="ws://localhost:9945")
    subtensor = bittensor.subtensor(network="ws://localhost:9945")

    # assert bob has new hotkey
    bob_neuron = metagraph.neurons[1]
    wallet_tree = alice_exec_command(ListCommand, ["w", "list"], "get_tree")
    new_num_hotkeys = len(wallet_tree.children[0].children)

    assert (
        bob_neuron.coldkey == "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
    )  # cold key didn't change
    assert bob_neuron.hotkey != bob_old_hotkey_address
    assert bob_neuron.hotkey != bob_neuron.coldkey
    assert bob_neuron.coldkey == subtensor.get_hotkey_owner(
        bob_neuron.hotkey
    )  # new key is owner
    assert (
        subtensor.is_hotkey_delegate(bob_neuron.hotkey) is False
    )  # new key is delegate ??
    assert (  # new key is registered on subnet
        subtensor.is_hotkey_registered_on_subnet(
            hotkey_ss58=bob_neuron.hotkey, netuid=1
        )
        is True
    )
    assert (  # old key is NOT registered on subnet
        subtensor.is_hotkey_registered_on_subnet(
            hotkey_ss58=bob_old_hotkey_address, netuid=1
        )
        is False
    )
    assert (  # uid is unchanged
        subtensor.get_uid_for_hotkey_on_subnet(hotkey_ss58=bob_neuron.hotkey, netuid=1)
        == bob_neuron.uid
    )
    assert new_num_hotkeys == num_hotkeys + 1
