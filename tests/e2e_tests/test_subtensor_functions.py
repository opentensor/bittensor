import asyncio
import sys

import pytest

from bittensor.core.subtensor import Subtensor
from bittensor.utils.btlogging import logging
from tests.e2e_tests.utils.chain_interactions import (
    register_neuron,
    register_subnet,
)
from tests.e2e_tests.utils.e2e_test_utils import (
    setup_wallet,
    template_path,
    templates_repo,
)


@pytest.mark.asyncio
async def test_subtensor_extrinsics(local_chain):
    """
    Tests subtensor extrinsics

    Steps:
        1. Validate subnets in the chain before/after registering netuid = 1
        2. Register Alice's neuron
        3. Verify Alice and Bob's participation in subnets (individually and global)
        4. Verify uids of Alice and Bob gets populated correctly
        5. Start Alice as a validator and verify neuroninfo before/after is different
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    netuid = 2
    subtensor = Subtensor(network="ws://localhost:9945")

    # Subnets 0 and 3 are bootstrapped from the start
    assert subtensor.get_subnets() == [0, 1]
    assert subtensor.get_total_subnets() == 2

    # Add wallets for Alice and Bob
    alice_keypair, alice_wallet = setup_wallet("//Alice")
    bob_keypair, bob_wallet = setup_wallet("//Bob")

    # Register subnet
    register_subnet(local_chain, alice_wallet), "Unable to register the subnet"

    # Subnet 1 is added after registration
    assert subtensor.get_subnets() == [0, 1, netuid]
    assert subtensor.get_total_subnets() == 3

    # Verify subnet 1 created successfully
    assert local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()
    assert subtensor.subnet_exists(netuid)

    # Verify Alice is registered to netuid 1 and Bob isn't registered to any
    assert subtensor.get_netuids_for_hotkey(hotkey_ss58=alice_keypair.ss58_address) == [
        netuid
    ], "Alice is not registered to netuid 1 as expected"
    assert (
        subtensor.get_netuids_for_hotkey(hotkey_ss58=bob_keypair.ss58_address) == []
    ), "Bob is unexpectedly registered to some netuid"

    # Verify Alice's hotkey is registered to any subnet (currently netuid = 1)
    assert subtensor.is_hotkey_registered_any(
        hotkey_ss58=alice_keypair.ss58_address
    ), "Alice's hotkey is not registered to any subnet"
    assert not subtensor.is_hotkey_registered_any(
        hotkey_ss58=bob_keypair.ss58_address
    ), "Bob's hotkey is unexpectedly registered to a subnet"

    # Verify netuid = 1 only has Alice registered and not Bob
    assert subtensor.is_hotkey_registered_on_subnet(
        netuid=netuid, hotkey_ss58=alice_keypair.ss58_address
    ), "Alice's hotkey is not registered on netuid 1"
    assert not subtensor.is_hotkey_registered_on_subnet(
        netuid=netuid, hotkey_ss58=bob_keypair.ss58_address
    ), "Bob's hotkey is unexpectedly registered on netuid 1"

    # Verify Alice's UID on netuid 1 is 0
    assert (
        subtensor.get_uid_for_hotkey_on_subnet(
            hotkey_ss58=alice_keypair.ss58_address, netuid=netuid
        )
        == 0
    ), "UID for Alice's hotkey on netuid 1 is not 0 as expected"

    # Register Bob to the subnet
    assert register_neuron(
        local_chain, bob_wallet, netuid
    ), "Unable to register Alice as a neuron"

    # Verify Bob's UID on netuid 1 is 1
    assert (
        subtensor.get_uid_for_hotkey_on_subnet(
            hotkey_ss58=bob_keypair.ss58_address, netuid=netuid
        )
        == 1
    ), "UID for Bob's hotkey on netuid 1 is not 1 as expected"

    neuron_info_old = subtensor.get_neuron_for_pubkey_and_subnet(
        alice_keypair.ss58_address, netuid=netuid
    )

    # Prepare to run Alice as validator
    cmd = " ".join(
        [
            f"{sys.executable}",
            f'"{template_path}{templates_repo}/neurons/validator.py"',
            "--netuid",
            str(netuid),
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

    # Run Alice as validator in the background
    await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    logging.info("Neuron Alice is now validating")

    await asyncio.sleep(
        5
    )  # wait for 5 seconds for the metagraph and subtensor to refresh with latest data
    subtensor = Subtensor(network="ws://localhost:9945")

    # Verify neuron info is updated after running as a validator
    neuron_info = subtensor.get_neuron_for_pubkey_and_subnet(
        alice_keypair.ss58_address, netuid=netuid
    )
    
    assert (
        neuron_info_old.axon_info != neuron_info.axon_info
    ), "Neuron info not updated after running validator"

    logging.info("✅ Passed test_subtensor_extrinsics")
