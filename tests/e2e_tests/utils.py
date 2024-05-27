from substrateinterface import Keypair
from typing import List
import bittensor


def setup_wallet(uri: str):
    keypair = Keypair.create_from_uri(uri)
    wallet_path = "/tmp/btcli-e2e-wallet-{}".format(uri.strip("/"))
    wallet = bittensor.wallet(path=wallet_path)
    wallet.set_coldkey(keypair=keypair, encrypt=False, overwrite=True)
    wallet.set_coldkeypub(keypair=keypair, encrypt=False, overwrite=True)
    wallet.set_hotkey(keypair=keypair, encrypt=False, overwrite=True)

    def exec_command(command, extra_args: List[str]):
        parser = bittensor.cli.__create_parser__()
        args = extra_args + [
            "--no_prompt",
            "--subtensor.network",
            "local",
            "--subtensor.chain_endpoint",
            "ws://localhost:9945",
            "--wallet.path",
            wallet_path,
        ]
        config = bittensor.config(
            parser=parser,
            args=args,
        )
        cli_instance = bittensor.cli(config)
        command.run(cli_instance)

    return (keypair, exec_command)

def get_wallet(uri: str, uri2: str):
    cold_keypair = Keypair.create_from_uri(uri)
    hot_keypair = Keypair.create_from_uri(uri2)

    wallet_path = "/tmp/btcli-e2e-wallet-{}-{}".format(uri.strip("/"), uri2.strip("/"))
    wallet = bittensor.wallet(path=wallet_path)
    wallet.set_coldkey(keypair=cold_keypair, encrypt=False, overwrite=True)
    wallet.set_coldkeypub(keypair=cold_keypair, encrypt=False, overwrite=True)
    wallet.set_hotkey(keypair=hot_keypair, encrypt=False, overwrite=True)

    return wallet
