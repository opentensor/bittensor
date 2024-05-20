# Standard Library
import argparse
import os

# 3rd Party
from rich.prompt import Prompt

# Bittensor
import bittensor

# Local
from . import defaults

SAVED_ATTRIBUTES = {
    "profile": ["name"],
    "subtensor": ["network", "chain_endpoint"],
    "wallet": ["name", "hotkey", "path"],
    "netuid": True,
}

class ProfileCommand:
    """
    Executes the ``create`` command.

    This class provides functionality to create a profile by prompting the user to enter various attributes.
    The entered attributes are then written to a profile file.

    """

    @staticmethod
    def run(cli):
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            ProfileCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")
        

    @staticmethod
    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        config = cli.config.copy()
        for (key, attributes) in SAVED_ATTRIBUTES.items():
            #if attributes isn't a list then just raw values
            if not isinstance(attributes, list):
                if config.no_prompt:
                    continue
                setattr(
                    config,
                    key,
                    Prompt.ask(
                        f"Enter {key}",
                        default=getattr(config, key, None)
                    ),
                )
                continue
            sub_attr = getattr(config, key, None)
            for attribute in attributes:
                if config.no_prompt:
                    continue
                attr_key = f"{key}.{attribute}"
                setattr(
                    config[key],
                    attribute,
                    Prompt.ask(
                        f"Enter {attr_key}",
                        choices=None if attribute != "network" else bittensor.__networks__,
                        default=getattr(sub_attr, attribute, None),
                    ),
                )
                #Set the chain_endpoint to match the network unless user defined.
                if attribute == "network" and config.subtensor.chain_endpoint is bittensor.__finney_entrypoint__:
                    (_, config.subtensor.chain_endpoint) = subtensor.determine_chain_endpoint_and_network(config.subtensor.network)
        ProfileCommand._write_profile(config)

    @staticmethod
    def _write_profile(config: "bittensor.config"):
        path = os.path.expanduser(config.profile.path)
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            bittensor.__console__.print(
                f":cross_mark: [red]Failed to write profile[/red]:[bold white] {e}"
            )
            return
        
        if os.path.exists(f"{path}{config.profile.name}") and not config.no_prompt:
            overwrite = None
            while overwrite not in ["y", "n"]:
                overwrite = Prompt.ask(f"Profile {config.profile.name} already exists. Overwrite?")
                if overwrite:
                    overwrite = overwrite.lower()
            if overwrite == "n":
                bittensor.__console__.print(
                    ":cross_mark: [red]Failed to write profile[/red]:[bold white] User denied."
                    )
                return
            
        profile = bittensor.config()
        #Create a profile clone with only the saved attributes
        for key in SAVED_ATTRIBUTES.keys():
            if isinstance(SAVED_ATTRIBUTES[key], list):
                getattr(profile, key, None)
                profile[key] = bittensor.config()
                for attribute in SAVED_ATTRIBUTES[key]:
                    setattr(profile[key], attribute, getattr(config[key], attribute))
            else:
                setattr(profile, key, getattr(config, key))
                
        try:
            with open(f"{path}{config.profile.name}", "w+") as f:
                f.write(str(profile))
        except Exception as e:
            bittensor.__console__.print(
                f":cross_mark: [red]Failed to write profile[/red]:[bold white] {e}"
            )
            return
        
        bittensor.__console__.print(
            f":white_check_mark: [bold green]Profile {config.profile.name} written to {path}[/bold green]"
        )

    @staticmethod
    def check_config(config: "bittensor.config"):
        return config is not None

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        list_parser = parser.add_parser("create", help="""Create profile""")
        list_parser.set_defaults(func=ProfileCommand.run)
        list_parser.add_argument(
            "--profile.name",
            type=str,
            default=defaults.profile.name,
            help="The name of the profile",
        )
        list_parser.add_argument(
            "--profile.path",
            type=str,
            default=defaults.profile.path,
            help="The path to the profile directory",
        )
        bittensor.subtensor.add_args(list_parser)
        bittensor.wallet.add_args(list_parser)

