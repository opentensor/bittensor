import logging
import os
import shutil
import subprocess
import sys
import time
from typing import List

import bittensor
from bittensor import Keypair

template_path = os.getcwd() + "/neurons/"
repo_name = "templates repository"


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

    return keypair, exec_command, wallet


def clone_or_update_templates():
    install_dir = template_path
    repo_mapping = {
        repo_name: "https://github.com/opentensor/bittensor-subnet-template.git",
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

    return install_dir + repo_name + "/"


def install_templates(install_dir):
    subprocess.check_call([sys.executable, "-m", "pip", "install", install_dir])


def uninstall_templates(install_dir):
    # uninstall templates
    subprocess.check_call(
        [sys.executable, "-m", "pip", "uninstall", "bittensor_subnet_template", "-y"]
    )
    # delete everything in directory
    shutil.rmtree(install_dir)


def wait_epoch(interval, subtensor):
    current_block = subtensor.get_current_block()
    next_tempo_block_start = (current_block - (current_block % interval)) + interval
    while current_block < next_tempo_block_start:
        time.sleep(1)  # Wait for 1 second before checking the block number again
        current_block = subtensor.get_current_block()
        if current_block % 10 == 0:
            print(
                f"Current Block: {current_block}  Next tempo at: {next_tempo_block_start}"
            )
            logging.info(
                f"Current Block: {current_block}  Next tempo at: {next_tempo_block_start}"
            )
