import logging
import asyncio
import os
import shutil
import subprocess
import sys
import time
from typing import List

from substrateinterface import SubstrateInterface

import bittensor
from bittensor import Keypair

template_path = os.getcwd() + "/neurons/"
templates_repo = "templates repository"


def setup_wallet(uri: str):
    keypair = Keypair.create_from_uri(uri)
    wallet_path = "/tmp/btcli-e2e-wallet-{}".format(uri.strip("/"))
    wallet = bittensor.wallet(path=wallet_path)
    wallet.set_coldkey(keypair=keypair, encrypt=False, overwrite=True)
    wallet.set_coldkeypub(keypair=keypair, encrypt=False, overwrite=True)
    wallet.set_hotkey(keypair=keypair, encrypt=False, overwrite=True)

    def exec_command(command, extra_args: List[str], function: str = "run"):
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
        logging.info(f'executing command: {command} {" ".join(args)}')
        config = bittensor.config(
            parser=parser,
            args=args,
        )
        cli_instance = bittensor.cli(config)
        # Dynamically call the specified function on the command
        result = getattr(command, function)(cli_instance)
        return result

    return keypair, exec_command, wallet


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


def sudo_call_set_target_stakes_per_interval(
    substrate: SubstrateInterface, wallet: bittensor.wallet
) -> bool:
    inner_call = substrate.compose_call(
        call_module="AdminUtils",
        call_function="sudo_set_target_stakes_per_interval",
        call_params={"target_stakes_per_interval": 100},
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


def call_add_proposal(substrate: SubstrateInterface, wallet: bittensor.wallet) -> bool:
    proposal_call = substrate.compose_call(
        call_module="System",
        call_function="remark",
        call_params={"remark": [0]},
    )
    call = substrate.compose_call(
        call_module="Triumvirate",
        call_function="propose",
        call_params={
            "proposal": proposal_call,
            "length_bound": 100_000,
            "duration": 100_000_000,
        },
    )

    extrinsic = substrate.create_signed_extrinsic(call=call, keypair=wallet.coldkey)
    response = substrate.submit_extrinsic(
        extrinsic,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    response.process_events()
    return response.is_success


async def wait_epoch(subtensor, netuid=1):
    q_tempo = [
        v.value
        for [k, v] in subtensor.query_map_subtensor("Tempo")
        if k.value == netuid
    ]
    if len(q_tempo) == 0:
        raise Exception("could not determine tempo")
    tempo = q_tempo[0]
    logging.info(f"tempo = {tempo}")
    await wait_interval(tempo, subtensor, netuid)


async def wait_interval(tempo, subtensor, netuid=1):
    interval = tempo + 1
    current_block = subtensor.get_current_block()
    last_epoch = current_block - 1 - (current_block + netuid + 1) % interval
    next_tempo_block_start = last_epoch + interval
    last_reported = None
    while current_block < next_tempo_block_start:
        await asyncio.sleep(
            1
        )  # Wait for 1 second before checking the block number again
        current_block = subtensor.get_current_block()
        if last_reported is None or current_block - last_reported >= 10:
            last_reported = current_block
            print(
                f"Current Block: {current_block}  Next tempo for netuid {netuid} at: {next_tempo_block_start}"
            )
            logging.info(
                f"Current Block: {current_block}  Next tempo for netuid {netuid} at: {next_tempo_block_start}"
            )


def clone_or_update_templates():
    specific_commit = None
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

    # here for pulling specific commit versions of repo
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
    # uninstall templates
    subprocess.check_call(
        [sys.executable, "-m", "pip", "uninstall", "bittensor_subnet_template", "-y"]
    )
    # delete everything in directory
    shutil.rmtree(install_dir)


async def write_output_log_to_file(name, stream):
    log_file = f"{name}.log"
    with open(log_file, "a") as f:
        while True:
            line = await stream.readline()
            if not line:
                break
            f.write(line.decode())
            f.flush()
