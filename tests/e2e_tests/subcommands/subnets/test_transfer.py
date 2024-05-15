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
    r = TransferCommand.run(cli_instance)

    print(r)
