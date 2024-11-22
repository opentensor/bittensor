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
        templates_repo: "https://github.com/opentensor/bittensor-subnet-template.git",
    }

    os.makedirs(install_dir, exist_ok=True)
    os.chdir(install_dir)

    for repo, git_link in repo_mapping.items():
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

    return install_dir + templates_repo + "/"


def install_templates(install_dir):
    subprocess.check_call([sys.executable, "-m", "pip", "install", install_dir])


def uninstall_templates(install_dir):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "uninstall", "bittensor_subnet_template", "-y"]
    )
    # Delete everything in directory
    shutil.rmtree(install_dir)
