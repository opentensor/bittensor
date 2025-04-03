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
        subprocess.run(["git", "fetch", "origin", specific_commit], check=True)
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
                env={
                    "BT_LOGGING_INFO": "1",
                },
                stdout=asyncio.subprocess.PIPE,
            )

            self.__reader_task = asyncio.create_task(self._reader())

            await asyncio.wait_for(self.started.wait(), 30)

            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            self.process.terminate()
            self.__reader_task.cancel()

            await self.process.wait()

        async def _reader(self):
            async for line in self.process.stdout:
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
                env={
                    "BT_LOGGING_INFO": "1",
                },
                stdout=asyncio.subprocess.PIPE,
            )

            self.__reader_task = asyncio.create_task(self._reader())

            await asyncio.wait_for(self.started.wait(), 30)

            return self

        async def __aexit__(self, exc_type, exc_value, traceback):
            self.process.terminate()
            self.__reader_task.cancel()

            await self.process.wait()

        async def _reader(self):
            async for line in self.process.stdout:
                if b"Starting validator loop." in line:
                    self.started.set()
                elif b"Successfully set weights and Finalized." in line:
                    self.set_weights.set()

    def __init__(self):
        # set_weights_era
        self.dir = clone_or_update_templates("d757bcc350d48d316d04ee0dc1581c74abeeaf67")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        uninstall_templates(self.dir)

    def miner(self, wallet, netuid):
        return self.Miner(self.dir, wallet, netuid)

    def validator(self, wallet, netuid):
        return self.Validator(self.dir, wallet, netuid)
