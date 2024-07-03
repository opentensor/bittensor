# Standard Library
import argparse
import os
import yaml

# 3rd Party
from rich.prompt import Prompt
from rich.table import Table

# Bittensor
import bittensor

# Local
from ..defaults import defaults

SAVED_ATTRIBUTES = {
    "profile": ["name"],
    "subtensor": ["network", "chain_endpoint"],
    "wallet": ["name", "hotkey", "path"],
    "netuid": True,
}


def handle_error(message, exception):
    bittensor.__console__.print(f":cross_mark: [red]{message}[/red]:[bold white] {exception}")


def log_info(message):
    bittensor.__console__.print(f":white_check_mark: [bold green]{message}[/bold green]")


def log_warning(message):
    bittensor.__console__.print(f":warning: [yellow]{message}[/yellow]")


def log_error(message):
    bittensor.__console__.print(f":cross_mark: [red]{message}[/red]")


def list_profiles(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        handle_error("Failed to list profiles", e)
        return []

    try:
        profiles = os.listdir(path)
        profiles = [profile for profile in profiles if profile.endswith(".yaml") or profile.endswith(".yml")]
    except Exception as e:
        handle_error("Failed to list profiles", e)
        return []

    if not profiles:
        log_error(f"No profiles found in {path}")
        return []

    return profiles


def ask_for_profile(profiles, action):
    profiles_without_extension = [profile.replace('.yml', '').replace('.yaml', '') for profile in profiles]

    # If we have no profiles, return None
    if len(profiles_without_extension) == 0:
        return None

    profile_name = Prompt.ask(f"Enter profile name to {action}", choices=profiles_without_extension)
    return profile_name + ('.yml' if profile_name + '.yml' in profiles else '.yaml')


def get_profile_file_path(path, profile_name):
    """Return the profile file path with the correct extension."""
    profiles = list_profiles(path)
    profile_base_name = profile_name.replace('.yml', '').replace('.yaml', '')

    for profile in profiles:
        if profile.startswith(profile_base_name):
            return os.path.join(path, profile)

    handle_error(f"Profile {profile_name} not found in {path}", None)
    return None


def get_profile_path_from_config(cli, action):
    """Extract profile path from config and handle user prompt if necessary."""
    config = cli.config.copy()
    path = os.path.expanduser(config.profile.path)

    if not config.is_set("profile.name") and not config.no_prompt:
        profiles = list_profiles(path)
        if not profiles:
            return None, None

        profile_name = ask_for_profile(profiles, action)
        config.profile.name = str(profile_name)

    profile_name = config.profile.name
    profile_path = get_profile_file_path(path, profile_name)

    return config, profile_path


def open_profile(cli, action):
    """Open a profile and return its configuration and contents."""
    config, profile_path = get_profile_path_from_config(cli, action)

    if profile_path is None:
        return None, None, None

    try:
        with open(profile_path, "r") as f:
            config_content = f.read()
        contents = yaml.safe_load(config_content)
        return config, profile_path, contents
    except Exception as e:
        handle_error("Failed to read profile", e)
        return None, None, None


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
            # if attributes isn't a list then just raw values
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
                # Set the chain_endpoint to match the network unless user defined.
                if attribute == "network" and config.subtensor.chain_endpoint is bittensor.__finney_entrypoint__:
                    (_, config.subtensor.chain_endpoint) = subtensor.determine_chain_endpoint_and_network(
                        config.subtensor.network)
        ProfileCommand._write_profile(config)

    @staticmethod
    def _write_profile(config: "bittensor.config"):
        path = os.path.expanduser(config.profile.path)
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            handle_error("Failed to write profile", e)
            return

        if os.path.exists(f"{path}{config.profile.name}.yml") and not config.no_prompt:
            overwrite = None
            while overwrite not in ["y", "n"]:
                overwrite = Prompt.ask(f"Profile {config.profile.name} already exists. Overwrite?")
                if overwrite:
                    overwrite = overwrite.lower()
            if overwrite == "n":
                log_error("Failed to write profile: User denied.")
                return

        profile = bittensor.config()
        # Create a profile clone with only the saved attributes
        for key in SAVED_ATTRIBUTES.keys():
            if isinstance(SAVED_ATTRIBUTES[key], list):
                getattr(profile, key, None)
                profile[key] = bittensor.config()
                for attribute in SAVED_ATTRIBUTES[key]:
                    setattr(profile[key], attribute, getattr(config[key], attribute))
            else:
                setattr(profile, key, getattr(config, key))

        try:
            with open(f"{path}{config.profile.name}.yml", "w+") as f:
                f.write(str(profile))
        except Exception as e:
            handle_error("Failed to write profile", e)
            return

        log_info(f"Profile {config.profile.name} written to {path}")

    @staticmethod
    def check_config(config: "bittensor.config"):
        return config is not None

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        profile_parser = parser.add_parser("create", help="""Create profile""")
        profile_parser.set_defaults(func=ProfileCommand.run)
        profile_parser.add_argument(
            "--profile.name",
            type=str,
            default=defaults.profile.name,
            help="The name of the profile",
        )
        profile_parser.add_argument(
            "--profile.path",
            type=str,
            default=defaults.profile.path,
            help="The path to the profile directory",
        )
        bittensor.subtensor.add_args(profile_parser)
        bittensor.wallet.add_args(profile_parser)


class ProfileListCommand:
    @staticmethod
    def run(cli):
        ProfileListCommand._run(cli)

    @staticmethod
    def _run(cli: "bittensor.cli"):
        path = os.path.expanduser(cli.config.profile.path)
        profiles = list_profiles(path)
        if not profiles:
            return

        profile_content = []
        for profile in profiles:
            try:
                with open(f"{path}/{profile}", "r") as f:
                    config_content = f.read()
                config = yaml.safe_load(config_content)
                profile_content.append(
                    (" ", str(config['profile']['name']), str(config['subtensor']['network']), str(config['netuid'])))
            except Exception as e:
                handle_error("Failed to read profile", e)
                continue

        table = Table(show_footer=True, width=cli.config.get("width", None), pad_edge=True, box=None, show_edge=True)
        table.title = "[white]Profiles"
        table.add_column("A", style="red", justify="center", min_width=1)
        table.add_column("Name", style="white", justify="center", min_width=10)
        table.add_column("Network", style="white", justify="center", min_width=10)
        table.add_column("Netuid", style="white", justify="center", min_width=10)
        for profile in profile_content:
            table.add_row(*profile)

        bittensor.__console__.print(table)

    @staticmethod
    def check_config(config: "bittensor.config"):
        return config is not None

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        profile_parser = parser.add_parser("list", help="""List profiles""")
        profile_parser.set_defaults(func=ProfileListCommand.run)
        profile_parser.add_argument("--profile.path", type=str, default=defaults.profile.path,
                                    help="The path to the profile directory")


class ProfileShowCommand:
    @staticmethod
    def run(cli):
        ProfileShowCommand._run(cli)

    @staticmethod
    def _run(cli: "bittensor.cli"):
        config, _profile_path, contents = open_profile(cli, "show")

        if contents is None:
            # Error message already printed in get_profile_file_path
            return

        table = Table(show_footer=True, width=cli.config.get("width", None), pad_edge=True, box=None, show_edge=True)
        table.title = f"[white]Profile [bold white]{config.profile.name}"
        table.add_column("[overline white]PARAMETERS", style="bold white", justify="left", min_width=10)
        table.add_column("[overline white]VALUES", style="green", justify="left", min_width=10)
        for key in contents.keys():
            if isinstance(contents[key], dict):
                for subkey in contents[key].keys():
                    table.add_row(f"  [bold white]{key}.{subkey}", f"[green]{contents[key][subkey]}")
                continue
            table.add_row(f"  [bold white]{key}", f"[green]{contents[key]}")

        bittensor.__console__.print(table)

    @staticmethod
    def check_config(config: "bittensor.config"):
        return config is not None

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        profile_parser = parser.add_parser("show", help="""Show profile""")
        profile_parser.set_defaults(func=ProfileShowCommand.run)
        profile_parser.add_argument("--profile.name", type=str, help="The name of the profile")
        profile_parser.add_argument("--profile.path", type=str, default=defaults.profile.path,
                                    help="The path to the profile directory")


class ProfileDeleteCommand:
    @staticmethod
    def run(cli):
        ProfileDeleteCommand._run(cli)

    @staticmethod
    def _run(cli: "bittensor.cli"):
        config, profile_path = get_profile_path_from_config(cli, "delete")

        if profile_path is None:
            # Error message already printed in get_profile_file_path
            return

        try:
            os.remove(profile_path)
            log_info(f"Profile {config.profile.name} deleted from {os.path.expanduser(config.profile.path)}")
        except Exception as e:
            handle_error("Failed to delete profile", e)

    @staticmethod
    def check_config(config: "bittensor.config"):
        return config is not None

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        profile_parser = parser.add_parser("delete", help="""Delete profile""")
        profile_parser.set_defaults(func=ProfileDeleteCommand.run)
        profile_parser.add_argument("--profile.name", type=str, help="The name of the profile to delete")
        profile_parser.add_argument("--profile.path", type=str, default=defaults.profile.path,
                                    help="The path to the profile directory")


class ProfileSetValueCommand:
    @staticmethod
    def run(cli):
        ProfileSetValueCommand._run(cli)

    @staticmethod
    def _run(cli: "bittensor.cli"):
        config, profile_path, contents = open_profile(cli, "set_value")

        if profile_path is None:
            return

        # Parse the new value from the arguments
        for arg in cli.config.args:
            if "=" in arg:
                key, value = arg.split("=")
                if ProfileSetValueCommand._set_value(contents, key, value):
                    log_info(f"Variable {key} was updated to {value} in profile {config.profile.name}")
                else:
                    log_info(f"Variable {key} was created with value {value} in profile {config.profile.name}")

        try:
            with open(profile_path, "w") as f:
                yaml.safe_dump(contents, f)
        except Exception as e:
            handle_error("Failed to write profile", e)

    @staticmethod
    def _set_value(contents, key, value):
        keys = key.split(".")
        sub_content = contents
        for k in keys[:-1]:
            if k not in sub_content:
                sub_content[k] = {}
            sub_content = sub_content[k]
        updated = keys[-1] in sub_content
        sub_content[keys[-1]] = value
        return updated

    @staticmethod
    def check_config(config: "bittensor.config"):
        return config is not None

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        profile_parser = parser.add_parser("set_value", help="""Set or update profile values""")
        profile_parser.set_defaults(func=ProfileSetValueCommand.run)
        profile_parser.add_argument("--profile.name", type=str, help="The name of the profile to update")
        profile_parser.add_argument("--profile.path", type=str, default=defaults.profile.path,
                                    help="The path to the profile directory")
        profile_parser.add_argument("args", nargs=argparse.REMAINDER, help="The key-value pairs to set or update")


class ProfileDeleteValueCommand:
    @staticmethod
    def run(cli):
        ProfileDeleteValueCommand._run(cli)

    @staticmethod
    def _run(cli: "bittensor.cli"):
        config, profile_path, contents = open_profile(cli, "delete_value")

        if profile_path is None:
            return

        for arg in cli.config.args:
            key = arg
            if ProfileDeleteValueCommand._remove_key(contents, key):
                log_info(f"Variable {key} was removed from profile {config.profile.name}")
            else:
                log_warning(f"Variable {key} does not exist in profile {config.profile.name}")

        try:
            with open(profile_path, "w") as f:
                yaml.safe_dump(contents, f)
        except Exception as e:
            handle_error("Failed to write profile", e)

    @staticmethod
    def _remove_key(contents, key):
        keys = key.split(".")
        sub_content = contents
        for k in keys[:-1]:
            sub_content = sub_content.get(k, {})
        return sub_content.pop(keys[-1], None) is not None

    @staticmethod
    def check_config(config: "bittensor.config"):
        return config is not None

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        profile_parser = parser.add_parser("delete_value", help="""Delete profile values""")
        profile_parser.set_defaults(func=ProfileDeleteValueCommand.run)
        profile_parser.add_argument("--profile.name", type=str, help="The name of the profile to update")
        profile_parser.add_argument("--profile.path", type=str, default=defaults.profile.path,
                                    help="The path to the profile directory")
        profile_parser.add_argument("args", nargs=argparse.REMAINDER, help="The keys to delete")


class ProfileSetCommand:
    @staticmethod
    def run(cli):
        config, profile_path, contents = open_profile(cli, "set")

        if profile_path is None:
            return

        # Load generic config file and write active profile to it
        config_path = defaults.config.path
        config_file_yaml = os.path.expanduser(os.path.join(config_path, "btcliconfig.yaml"))
        config_file_yml = os.path.expanduser(os.path.join(config_path, "btcliconfig.yml"))

        try:
            generic_config = None
            generic_config_path = None
            if os.path.exists(config_file_yaml):
                with open(config_file_yaml, 'r') as file:
                    generic_config = yaml.safe_load(file)
                    generic_config_path = config_file_yaml
            elif os.path.exists(config_file_yml):
                with open(config_file_yml, 'r') as file:
                    generic_config = yaml.safe_load(file)
                    generic_config_path = config_file_yml

            if not generic_config_path:
                handle_error("Failed to read generic config", None)
                return

            if not generic_config:
                generic_config = {'profile': {'active': ''}}

            generic_config['profile']['active'] = config.profile.name.replace('.yml', '').replace('.yaml', '')

            with open(generic_config_path, 'w+') as file:
                yaml.safe_dump(generic_config, file)

            log_info(f"Profile {config.profile.name} set as active.")
        except Exception as e:
            handle_error("Failed to set active profile", e)

    @staticmethod
    def check_config(config: "bittensor.config"):
        return config is not None

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        profile_parser = parser.add_parser("set", help="""Set active profile""")
        profile_parser.set_defaults(func=ProfileSetCommand.run)
        profile_parser.add_argument("--profile.name", type=str, help="The name of the profile to set as active")
        profile_parser.add_argument("--profile.path", type=str, default=defaults.profile.path,
                                    help="The path to the profile directory")
