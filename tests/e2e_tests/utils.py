from substrateinterface import Keypair, SubstrateInterface
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


def new_wallet(uri: str, uri2: str):
    keypair_1 = Keypair.create_from_uri(uri)
    keypair_2 = Keypair.create_from_uri(uri2)
    wallet_path = "/tmp/btcli-e2e-wallet-{}-{}".format(uri.strip("/"), uri2.strip("/"))
    wallet = bittensor.wallet(path=wallet_path)
    wallet.set_coldkey(keypair=keypair_1, encrypt=False, overwrite=True)
    wallet.set_coldkeypub(keypair=keypair_1, encrypt=False, overwrite=True)
    wallet.set_hotkey(keypair=keypair_2, encrypt=False, overwrite=True)

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

    return (wallet, exec_command)


def sudo_call_set_network_limit(
    substrate: SubstrateInterface, wallet: bittensor.wallet
) -> bool:
    inner_call = substrate.compose_call(
        call_module="AdminUtils",
        call_function="sudo_set_network_rate_limit",
        call_params={"rate_limit": 1},
    )
    call = substrate.compose_call(
        call_module="Sudo",
        call_function="sudo",
        call_params={"call": inner_call},
    )

    extrinsic = substrate.create_signed_extrinsic(call=call, keypair=wallet.coldkey)
    response = substrate.submit_extrinsic(
        extrinsic,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    response.process_events()
    return response.is_success
