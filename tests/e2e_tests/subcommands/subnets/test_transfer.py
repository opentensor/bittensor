from substrateinterface import Keypair

from bittensor.commands.transfer import TransferCommand
import bittensor


# Example test using the local_chain fixture
def test_example(local_chain):
    keypair = Keypair.create_from_uri("//Alice")
    wallet = bittensor.wallet(path="/tmp/btcli-wallet")
    wallet.set_coldkey(keypair=keypair, encrypt=False, overwrite=True)
    wallet.set_coldkeypub(keypair=keypair, encrypt=False, overwrite=True)
    wallet.set_hotkey(keypair=keypair, encrypt=False, overwrite=True)

    parser = bittensor.cli.__create_parser__()
    config = bittensor.config(
        parser=parser,
        args=[
            "wallet",
            "transfer",
            "--no_prompt",
            "--amount",
            "2",
            "--dest",
            "5GpzQgpiAKHMWNSH3RN4GLf96GVTDct9QxYEFAY7LWcVzTbx",
            "--subtensor.network",
            "local",
            "--subtensor.chain_endpoint",
            "ws://localhost:9945",
            "--wallet.path",
            "/tmp/btcli-wallet",
        ],
    )
    cli_instance = bittensor.cli(config)

    acc_before = local_chain.query('System', 'Account', [keypair.ss58_address])
    TransferCommand.run(cli_instance)
    acc_after = local_chain.query('System', 'Account', [keypair.ss58_address])

    expected_transfer = 2_000_000_000
    tolerance = 200_000  # Tx fee tolerance

    actual_difference = acc_before.value['data']['free'] - acc_after.value['data']['free']
    assert expected_transfer <= actual_difference <= expected_transfer + tolerance, f"Expected transfer with tolerance: {expected_transfer} <= {actual_difference} <= {expected_transfer + tolerance}"
