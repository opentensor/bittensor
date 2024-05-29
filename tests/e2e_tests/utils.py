from substrateinterface import Keypair
from typing import List
import bittensor
import io
import contextlib


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

        stdout_io = io.StringIO()
        stderr_io = io.StringIO()

        with contextlib.redirect_stdout(stdout_io), contextlib.redirect_stderr(
            stderr_io
        ):
            command.run(cli_instance)

        stdout_lines = stdout_io.getvalue().split("\n")
        stderr_lines = stderr_io.getvalue().split("\n")
        print("STDOUT: " + stdout_io.getvalue())
        print("STDERR: " + stderr_io.getvalue())

        return stdout_lines, stderr_lines

    return (keypair, exec_command)
