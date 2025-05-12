import asyncio
import os
import shutil
import subprocess
import sys

from bittensor_wallet import Keypair

import bittensor

template_path = os.getcwd() + "/neurons/"
templates_repo = "templates repository"


def setup_wallet(uri: str) -> tuple[Keypair, bittensor.Wallet]:
    """
    Sets up a wallet using the provided URI.

    This function creates a keypair from the given URI and initializes a wallet
    at a temporary path. It sets the coldkey, coldkeypub, and hotkey for the wallet
    using the generated keypair.

    Side Effects:
        - Creates a wallet in a temporary directory.
        - Sets keys in the wallet without encryption and with overwriting enabled.
    """
    keypair = Keypair.create_from_uri(uri)
    wallet_path = f"/tmp/btcli-e2e-wallet-{uri.strip('/')}"
    wallet = bittensor.Wallet(path=wallet_path)
    wallet.set_coldkey(keypair=keypair, encrypt=False, overwrite=True)
    wallet.set_coldkeypub(keypair=keypair, encrypt=False, overwrite=True)
    wallet.set_hotkey(keypair=keypair, encrypt=False, overwrite=True)
    return keypair, wallet


def clone_or_update_templates(specific_commit=None):
    """
    Clones or updates the Bittensor subnet template repository.

    This function clones the Bittensor subnet template repository if it does not
    already exist in the specified installation directory. If the repository already
    exists, it updates it by pulling the latest changes. Optionally, it can check out
    a specific commit if the `specific_commit` variable is set.
    """
    install_dir = template_path
    repo_mapping = {
        templates_repo: "https://github.com/opentensor/subnet-template.git",
    }

    cwd = os.getcwd()

    os.makedirs(install_dir, exist_ok=True)
    os.chdir(install_dir)

    for repo, git_link in repo_mapping.items():
        print(os.path.abspath(repo))
        if not os.path.exists(repo):
            print(f"\033[94mCloning {repo}...\033[0m")
            subprocess.run(["git", "clone", git_link, repo], check=True)
        else:
            print(f"\033[94mUpdating {repo}...\033[0m")
            os.chdir(repo)
            subprocess.run(["git", "pull"], check=True)
            os.chdir("..")

    # For pulling specific commit versions of repo
    if specific_commit:
        os.chdir(templates_repo)
        print(
            f"\033[94mChecking out commit {specific_commit} in {templates_repo}...\033[0m"
        )
        subprocess.run(["git", "checkout", specific_commit], check=True)
        os.chdir("..")

    os.chdir(cwd)

    return install_dir + templates_repo


def uninstall_templates(install_dir):
    # Delete everything in directory
    shutil.rmtree(install_dir)


class Templates:
    class Miner:
        def __init__(self, dir, wallet, netuid):
            self.dir = dir
            self.wallet = wallet
            self.netuid = netuid
            self.process = None

            self.started = asyncio.Event()

        async def __aenter__(self):
            env = os.environ.copy()
            env["BT_LOGGING_INFO"] = "1"
            self.process = await asyncio.create_subprocess_exec(
                sys.executable,
                f"{self.dir}/miner.py",
                "--netuid",
                str(self.netuid),
                "--subtensor.network",
                "local",
                "--subtensor.chain_endpoint",
                "ws://localhost:9944",
                "--wallet.path",
                self.wallet.path,
                "--wallet.name",
                self.wallet.name,
                "--wallet.hotkey",
                "default",
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            self.__reader_task = asyncio.create_task(self._reader())

            try:
                await asyncio.wait_for(self.started.wait(), 60)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
                raise RuntimeError("Miner failed to start within timeout")

            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            self.process.terminate()
            self.__reader_task.cancel()

            await self.process.wait()

        async def _reader(self):
            async for line in self.process.stdout:
                try:
                    bittensor.logging.console.info(
                        f"[green]MINER LOG: {line.split(b'|')[-1].strip().decode()}[/blue]"
                    )
                except BaseException:
                    # skipp empty lines
                    pass

                if b"Starting main loop" in line:
                    self.started.set()

    class Validator:
        def __init__(self, dir, wallet, netuid):
            self.dir = dir
            self.wallet = wallet
            self.netuid = netuid
            self.process = None

            self.started = asyncio.Event()
            self.set_weights = asyncio.Event()

        async def __aenter__(self):
            env = os.environ.copy()
            env["BT_LOGGING_INFO"] = "1"
            self.process = await asyncio.create_subprocess_exec(
                sys.executable,
                f"{self.dir}/validator.py",
                "--netuid",
                str(self.netuid),
                "--subtensor.network",
                "local",
                "--subtensor.chain_endpoint",
                "ws://localhost:9944",
                "--wallet.path",
                self.wallet.path,
                "--wallet.name",
                self.wallet.name,
                "--wallet.hotkey",
                "default",
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            self.__reader_task = asyncio.create_task(self._reader())

            try:
                await asyncio.wait_for(self.started.wait(), 60)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
                raise RuntimeError("Validator failed to start within timeout")

            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            self.process.terminate()
            self.__reader_task.cancel()

            await self.process.wait()

        async def _reader(self):
            async for line in self.process.stdout:
                try:
                    bittensor.logging.console.info(
                        f"[orange]VALIDATOR LOG: {line.split(b'|')[-1].strip().decode()}[/orange]"
                    )
                except BaseException:
                    # skipp empty lines
                    pass

                if b"Starting validator loop." in line:
                    bittensor.logging.console.info("Validator started.")
                    self.started.set()
                elif b"Successfully set weights and Finalized." in line:
                    bittensor.logging.console.info("Validator is setting weights.")
                    self.set_weights.set()

    def __init__(self):
        self.dir = clone_or_update_templates()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        uninstall_templates(self.dir)

    def miner(self, wallet, netuid):
        return self.Miner(self.dir, wallet, netuid)

    def validator(self, wallet, netuid):
        return self.Validator(self.dir, wallet, netuid)


def wait_to_start_call(
    subtensor: "bittensor.Subtensor",
    subnet_owner_wallet: "bittensor.Wallet",
    netuid: int,
    in_blocks: int = 10,
):
    """Waits for a certain number of blocks before making a start call."""
    if subtensor.is_fast_blocks() is False:
        in_blocks = 5
    bittensor.logging.console.info(
        f"Waiting for [blue]{in_blocks}[/blue] blocks before [red]start call[/red]. "
        f"Current block: [blue]{subtensor.block}[/blue]."
    )

    # make sure we passed start_call limit
    subtensor.wait_for_block(subtensor.block + in_blocks + 1)
    status, message = subtensor.start_call(
        wallet=subnet_owner_wallet,
        netuid=netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert status, message
    return True


async def async_wait_to_start_call(
    subtensor: "bittensor.AsyncSubtensor",
    subnet_owner_wallet: "bittensor.Wallet",
    netuid: int,
    in_blocks: int = 10,
):
    """Waits for a certain number of blocks before making a start call."""
    if await subtensor.is_fast_blocks() is False:
        in_blocks = 5

    bittensor.logging.console.info(
        f"Waiting for [blue]{in_blocks}[/blue] blocks before [red]start call[/red]. "
        f"Current block: [blue]{subtensor.block}[/blue]."
    )

    # make sure we passed start_call limit
    current_block = await subtensor.block
    await subtensor.wait_for_block(current_block + in_blocks + 1)
    status, message = await subtensor.start_call(
        wallet=subnet_owner_wallet,
        netuid=netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    assert status, message
    return True
