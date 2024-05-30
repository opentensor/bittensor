from bittensor.commands.root import RootRegisterCommand
from ...utils import new_wallet


# Example test using the local_chain fixture
def test_root_register_root_network(local_chain, capsys):
    (wallet, exec_command) = new_wallet("//Alice", "//Bob")

    uid = local_chain.query("SubtensorModule", "Uids", [0, wallet.hotkey.ss58_address])
    assert uid == None

    exec_command(
        RootRegisterCommand,
        ["root", "register"],
    )

    uid = local_chain.query("SubtensorModule", "Uids", [0, wallet.hotkey.ss58_address])
    assert uid != None
