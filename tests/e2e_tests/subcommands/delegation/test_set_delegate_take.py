from bittensor import logging
from bittensor.commands.delegates import NominateCommand, SetTakeCommand
from bittensor.commands.network import RegisterSubnetworkCommand
from bittensor.commands.register import RegisterCommand
from bittensor.commands.root import RootRegisterCommand
from tests.e2e_tests.utils import setup_wallet


def test_set_delegate_increase_take(local_chain):
    logging.info("Testing test_set_delegate_increase_take")
    # Register root as Alice
    keypair, exec_command, wallet = setup_wallet("//Alice")
    exec_command(RootRegisterCommand, ["root", "register"])

    # Create subnet 1 and verify created successfully
    assert not (
        local_chain.query("SubtensorModule", "NetworksAdded", [1]).serialize()
    ), "Subnet is already registered"

    exec_command(RegisterSubnetworkCommand, ["s", "create"])

    assert local_chain.query(
        "SubtensorModule", "NetworksAdded", [1]
    ).serialize(), "Subnet wasn't registered"

    # Register and nominate Bob
    keypair, exec_command, wallet = setup_wallet("//Bob")
    assert (
        local_chain.query(
            "SubtensorModule", "LastTxBlock", [keypair.ss58_address]
        ).serialize()
        == 0
    )

    assert (
        local_chain.query(
            "SubtensorModule", "LastTxBlockDelegateTake", [keypair.ss58_address]
        ).serialize()
        == 0
    )
    exec_command(RegisterCommand, ["s", "register", "--netuid", "1"])
    exec_command(NominateCommand, ["root", "nominate"])
    assert (
        local_chain.query(
            "SubtensorModule", "LastTxBlock", [keypair.ss58_address]
        ).serialize()
        > 0
    )
    assert (
        local_chain.query(
            "SubtensorModule", "LastTxBlockDelegateTake", [keypair.ss58_address]
        ).serialize()
        > 0
    )

    # Set delegate take for Bob
    exec_command(SetTakeCommand, ["r", "set_take", "--take", "0.15"])
    assert local_chain.query(
        "SubtensorModule", "Delegates", [keypair.ss58_address]
    ).value == int(0.15 * 65535), "Take value set incorrectly"
    logging.info("Passed test_set_delegate_increase_take")
