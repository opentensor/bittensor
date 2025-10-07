"""Implementation of the config class, which manages the configuration of different Bittensor modules.

Example:
    import argparse
    import bittensor as bt

    parser = argparse.ArgumentParser('Miner')
    bt.Axon.add_args(parser)
    bt.Subtensor.add_args(parser)
    bt.Async_subtensor.add_args(parser)
    bt.Wallet.add_args(parser)
    bt.logging.add_args(parser)
    bt.PriorityThreadPoolExecutor.add_args(parser)
    config = bt.config(parser)

    print(config)
"""

import argparse
import os
import sys
from copy import deepcopy
from typing import Any, Optional
from bittensor.core.settings import DEFAULTS
import yaml
from munch import DefaultMunch


def _filter_keys(obj):
    """Filters keys from an object, excluding private and certain internal properties."""
    if isinstance(obj, dict):
        return {
            k: _filter_keys(v)
            for k, v in obj.items()
            if not k.startswith("__") and not k.startswith("_Config__is_set")
        }
    elif isinstance(obj, (Config, DefaultMunch)):
        return _filter_keys(obj.toDict())
    return obj


class InvalidConfigFile(Exception):
    """Raised when there's an error loading the config file."""


class Config(DefaultMunch):
    """Manages configuration for Bittensor modules with nested namespace support."""

    def __init__(
        self,
        parser: argparse.ArgumentParser = None,
        args: Optional[list[str]] = None,
        strict: bool = False,
        default: Any = DEFAULTS,
    ) -> None:
        no_parse_cli = os.getenv("BT_NO_PARSE_CLI_ARGS", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        # Fallback to defaults if not provided
        default = deepcopy(default or DEFAULTS)

        if isinstance(default, DefaultMunch):
            # Initialize Munch with defaults (dict-safe)
            super().__init__(None, default.toDict())
        else:
            # if defaults passed as dict
            super().__init__(None, default)

        self.__is_set = {}

        # If CLI parsing disabled, stop here
        if no_parse_cli or parser is None:
            return

        self._add_default_arguments(parser)
        args = args or sys.argv[1:]
        self._validate_required_args(parser, args)

        config_params = self._parse_args(args, parser, strict=False)
        config_path = self._get_config_path(config_params)
        strict = strict or getattr(config_params, "strict", False)

        if config_path:
            self._load_config_file(parser, config_path)

        params = self._parse_args(args, parser, strict)
        self._build_config_tree(params)
        self._detect_set_parameters(parser, args)

    def __str__(self) -> str:
        """String representation without private keys, optimized to avoid deepcopy."""
        cleaned = _filter_keys(self.toDict())
        return "\n" + yaml.dump(cleaned, sort_keys=False, default_flow_style=False)

    def __repr__(self) -> str:
        """String representation of the Config."""
        return self.__str__()

    def _validate_required_args(
        self, parser: argparse.ArgumentParser, args: list[str]
    ) -> None:
        """Validates required arguments are present."""
        missing = self._find_missing_required_args(parser, args)
        if missing:
            raise ValueError(f"Missing required arguments: {', '.join(missing)}")

    def _find_missing_required_args(
        self, parser: argparse.ArgumentParser, args: list[str]
    ) -> list[str]:
        """Identifies missing required arguments."""
        required = {a.dest for a in parser._actions if a.required}
        provided = {a.split("=")[0].lstrip("-") for a in args if a.startswith("-")}
        return list(required - provided)

    def _get_config_path(self, params: DefaultMunch) -> Optional[str]:
        """Gets Config path from parameters."""
        return getattr(params, "config", None)

    def _load_config_file(self, parser: argparse.ArgumentParser, path: str) -> None:
        """Loads Config from YAML file."""
        try:
            with open(os.path.expanduser(path)) as f:
                config = yaml.safe_load(f)
                print(f"Loading config from: {path}")
                parser.set_defaults(**config)
        except Exception as e:
            raise InvalidConfigFile(f"Error loading config: {e}") from e

    def _build_config_tree(self, params: DefaultMunch) -> None:
        """Builds nested Config structure."""
        for key, value in params.items():
            if key in ["__is_set"]:
                continue
            current = self
            parts = key.split(".")
            for part in parts[:-1]:
                current = current.setdefault(part, Config())
            current[parts[-1]] = value

    def _detect_set_parameters(
        self, parser: argparse.ArgumentParser, args: list[str]
    ) -> None:
        """Detects which parameters were explicitly set."""
        temp_parser = self._create_non_default_parser(parser)
        detected = self._parse_args(args, temp_parser, strict=False)
        self.__is_set = DefaultMunch(**{k: True for k in detected.keys()})

    def _create_non_default_parser(
        self, original: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        """Creates a parser that ignores default values."""
        parser = deepcopy(original)
        for action in parser._actions:
            action.default = argparse.SUPPRESS
        return parser

    @staticmethod
    def _parse_args(
        args: list[str], parser: argparse.ArgumentParser, strict: bool
    ) -> DefaultMunch:
        """Parses args with error handling."""
        try:
            if strict:
                result = parser.parse_args(args)
                return DefaultMunch.fromDict(vars(result))

            result, unknown = parser.parse_known_args(args)
            for arg in unknown:
                if arg.startswith("--") and (name := arg[2:]) in vars(result):
                    setattr(result, name, True)
            return DefaultMunch.fromDict(vars(result))
        except Exception:
            raise ValueError("Invalid arguments provided.")

    def __deepcopy__(self, memo) -> "Config":
        """Creates a deep copy that maintains Config type."""
        new_config = Config()
        memo[id(self)] = new_config

        for key, value in self.items():
            new_config[key] = deepcopy(value, memo)

        new_config.__is_set = deepcopy(self.__is_set, memo)
        return new_config

    def merge(self, other: "Config") -> None:
        """Merges another Config into this one."""
        self.update(self._merge_dicts(self, other))
        self.__is_set.update(other.__is_set)

    @staticmethod
    def _merge_dicts(a: DefaultMunch, b: DefaultMunch) -> DefaultMunch:
        """Recursively merges two Config objects."""
        result = deepcopy(a)
        for key, value in b.items():
            if key in result:
                if isinstance(result[key], DefaultMunch) and isinstance(
                    value, DefaultMunch
                ):
                    result[key] = Config._merge_dicts(result[key], value)
                else:
                    result[key] = deepcopy(value)
            else:
                result[key] = deepcopy(value)
        return result

    def is_set(self, param_name: str) -> bool:
        """Checks if a parameter was explicitly set."""
        return self.__is_set.get(param_name, False)

    def to_dict(self) -> dict:
        """Returns the configuration as a dictionary."""
        return self.toDict()

    def _add_default_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Adds default arguments to the Config parser."""
        arguments = [
            (
                "--config",
                {
                    "type": str,
                    "help": "If set, defaults are overridden by passed file.",
                    "default": False,
                },
            ),
            (
                "--strict",
                {
                    "action": "store_true",
                    "help": "If flagged, config will check that only exact arguments have been set.",
                    "default": False,
                },
            ),
            (
                "--no_version_checking",
                {
                    "action": "store_true",
                    "help": "Set `true to stop cli version checking.",
                    "default": False,
                },
            ),
        ]

        for arg_name, kwargs in arguments:
            try:
                parser.add_argument(arg_name, **kwargs)
            except argparse.ArgumentError:
                # this can fail if argument has already been added.
                pass
