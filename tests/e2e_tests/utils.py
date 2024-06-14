from substrateinterface import Keypair, SubstrateInterface
from typing import List
import os
import shutil
import subprocess
import sys

from bittensor import Keypair
import bittensor

template_path = os.getcwd() + "/neurons/"
repo_name = "templates repository"


def setup_wallet(uri: str, with_path: bool = False):
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

    if with_path:
        return (keypair, exec_command, wallet_path)
    else:
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


def sudo_call_set_weight_limit(
    substrate: SubstrateInterface, wallet: bittensor.wallet, netuid: int
) -> bool:
    """Set the set weight limit for the network with netuid via sudo call."""
    inner_call = substrate.compose_call(
        call_module="AdminUtils",
        call_function="sudo_set_weights_set_rate_limit",
        call_params={"netuid": netuid, "weights_set_rate_limit": 1},
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


def sudo_call_set_min_stake(
    substrate: SubstrateInterface, wallet: bittensor.wallet, min_stake: int
) -> bool:
    """Set the minimum stake for the network via sudo call."""
    inner_call = substrate.compose_call(
        call_module="AdminUtils",
        call_function="sudo_set_weights_min_stake",
        call_params={"min_stake": min_stake},
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


def sudo_call_add_senate_member(
    substrate: SubstrateInterface, wallet: bittensor.wallet
) -> bool:
    inner_call = substrate.compose_call(
        call_module="SenateMembers",
        call_function="add_member",
        call_params={"who": wallet.hotkey.ss58_address},
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


def sudo_call_set_triumvirate_members(
    substrate: SubstrateInterface, wallet: bittensor.wallet
) -> bool:
    inner_call = substrate.compose_call(
        call_module="Triumvirate",
        call_function="set_members",
        call_params={
            "new_members": [wallet.hotkey.ss58_address],
            "prime": wallet.coldkey.ss58_address,
            "old_count": 0,
        },
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
    return keypair, exec_command, wallet_path


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
