# The MIT License (MIT)
# Copyright © 2024 Opentensor Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Implementation of the config class, which manages the configuration of different Bittensor modules."""

import argparse
import copy
import os
import sys
from copy import deepcopy
from typing import Any, TypeVar, Type, Optional

import yaml
from munch import DefaultMunch


class InvalidConfigFile(Exception):
    """In place of YAMLError"""


class Config(DefaultMunch):
    """
    Implementation of the config class, which manages the configuration of different Bittensor modules.

    Translates the passed parser into a nested Bittensor config.

    Args:
        parser (argparse.ArgumentParser): Command line parser object.
        strict (bool): If ``true``, the command line arguments are strictly parsed.
        args (list of str): Command line arguments.
        default (Optional[Any]): Default value for the Config. Defaults to ``None``. This default will be returned for attributes that are undefined.

    Returns:
        config (bittensor.core.config.Config): Nested config object created from parser arguments.
    """

    __is_set: dict[str, bool]

    def __init__(
        self,
        parser: argparse.ArgumentParser = None,
        args: Optional[list[str]] = None,
        strict: bool = False,
        default: Optional[Any] = None,
    ) -> None:
        super().__init__(default)

        self["__is_set"] = {}

        if parser is None:
            return

        # Optionally add config specific arguments
        try:
            parser.add_argument(
                "--config",
                type=str,
                help="If set, defaults are overridden by passed file.",
            )
        except Exception:
            # this can fail if --config has already been added.
            pass

        try:
            parser.add_argument(
                "--strict",
                action="store_true",
                help="""If flagged, config will check that only exact arguments have been set.""",
                default=False,
            )
        except Exception:
            # this can fail if --strict has already been added.
            pass

        try:
            parser.add_argument(
                "--no_version_checking",
                action="store_true",
                help="Set ``true`` to stop cli version checking.",
                default=False,
            )
        except Exception:
            # this can fail if --no_version_checking has already been added.
            pass

        try:
            parser.add_argument(
                "--no_prompt",
                dest="no_prompt",
                action="store_true",
                help="Set ``true`` to stop cli from prompting the user.",
                default=False,
            )
        except Exception:
            # this can fail if --no_version_checking has already been added.
            pass

        # Get args from argv if not passed in.
        if args is None:
            args = sys.argv[1:]

        # Check for missing required arguments before proceeding
        missing_required_args = self.__check_for_missing_required_args(parser, args)
        if missing_required_args:
            # Handle missing required arguments gracefully
            raise ValueError(
                f"Missing required arguments: {', '.join(missing_required_args)}"
            )

        # 1.1 Optionally load defaults if the --config is set.
        try:
            config_file_path = (
                str(os.getcwd())
                + "/"
                + vars(parser.parse_known_args(args)[0])["config"]
            )
        except Exception as e:
            config_file_path = None

        # Parse args not strict
        config_params = Config.__parse_args__(args=args, parser=parser, strict=False)

        # 2. Optionally check for --strict
        # strict=True when passed in OR when --strict is set
        strict = config_params.strict or strict

        if config_file_path is not None:
            config_file_path = os.path.expanduser(config_file_path)
            try:
                with open(config_file_path) as f:
                    params_config = yaml.safe_load(f)
                    print(f"Loading config defaults from: {config_file_path}")
                    parser.set_defaults(**params_config)
            except Exception as e:
                print(f"Error in loading: {e} using default parser settings")

        # 2. Continue with loading in params.
        params = Config.__parse_args__(args=args, parser=parser, strict=strict)

        _config = self

        # Splits params and add to config
        Config.__split_params__(params=params, _config=_config)

        # Make the is_set map
        _config["__is_set"] = {}

        # Reparse args using default of unset
        parser_no_defaults = copy.deepcopy(parser)

        # Only command as the arg, else no args
        default_param_args = (
            [_config.get("command")]
            if _config.get("command") is not None and _config.get("subcommand") is None
            else []
        )
        if _config.get("command") is not None and _config.get("subcommand") is not None:
            default_param_args = [_config.get("command"), _config.get("subcommand")]

        # Get all args by name
        default_params = parser.parse_args(args=default_param_args)

        all_default_args = default_params.__dict__.keys() | []
        # Make a dict with keys as args and values as argparse.SUPPRESS
        defaults_as_suppress = {key: argparse.SUPPRESS for key in all_default_args}
        # Set the defaults to argparse.SUPPRESS, should remove them from the namespace
        parser_no_defaults.set_defaults(**defaults_as_suppress)
        parser_no_defaults._defaults.clear()  # Needed for quirk of argparse

        # Check for subparsers and do the same
        if parser_no_defaults._subparsers is not None:
            for action in parser_no_defaults._subparsers._actions:
                # Should only be the "command" subparser action
                if isinstance(action, argparse._SubParsersAction):
                    # Set the defaults to argparse.SUPPRESS, should remove them from the namespace
                    # Each choice is the keyword for a command, we need to set the defaults for each of these
                    # Note: we also need to clear the _defaults dict for each, this is a quirk of argparse
                    cmd_parser: argparse.ArgumentParser
                    for cmd_parser in action.choices.values():
                        # If this choice is also a subparser, set defaults recursively
                        if cmd_parser._subparsers:
                            for action in cmd_parser._subparsers._actions:
                                # Should only be the "command" subparser action
                                if isinstance(action, argparse._SubParsersAction):
                                    cmd_parser: argparse.ArgumentParser
                                    for cmd_parser in action.choices.values():
                                        cmd_parser.set_defaults(**defaults_as_suppress)
                                        cmd_parser._defaults.clear()  # Needed for quirk of argparse
                        else:
                            cmd_parser.set_defaults(**defaults_as_suppress)
                            cmd_parser._defaults.clear()  # Needed for quirk of argparse

        # Reparse the args, but this time with the defaults as argparse.SUPPRESS
        params_no_defaults = Config.__parse_args__(
            args=args, parser=parser_no_defaults, strict=strict
        )

        # Diff the params and params_no_defaults to get the is_set map
        _config["__is_set"] = {
            arg_key: True
            for arg_key in [
                k
                for k, _ in filter(
                    lambda kv: kv[1] != argparse.SUPPRESS,
                    params_no_defaults.__dict__.items(),
                )
            ]
        }

    @staticmethod
    def __split_params__(params: argparse.Namespace, _config: "Config"):
        # Splits params on dot syntax i.e. neuron.axon_port and adds to _config
        for arg_key, arg_val in params.__dict__.items():
            split_keys = arg_key.split(".")
            head = _config
            keys = split_keys
            while len(keys) > 1:
                if (
                    hasattr(head, keys[0]) and head[keys[0]] is not None
                ):  # Needs to be Config
                    head = getattr(head, keys[0])
                    keys = keys[1:]
                else:
                    head[keys[0]] = Config()
                    head = head[keys[0]]
                    keys = keys[1:]
            if len(keys) == 1:
                head[keys[0]] = arg_val

    @staticmethod
    def __parse_args__(
        args: list[str], parser: argparse.ArgumentParser = None, strict: bool = False
    ) -> argparse.Namespace:
        """Parses the passed args use the passed parser.

        Args:
            args (List[str]): List of arguments to parse.
            parser (argparse.ArgumentParser): Command line parser object.
            strict (bool): If ``true``, the command line arguments are strictly parsed.

        Returns:
            Namespace: Namespace object created from parser arguments.
        """
        if not strict:
            params, unrecognized = parser.parse_known_args(args=args)
            params_list = list(params.__dict__)
            # bug within argparse itself, does not correctly set value for boolean flags
            for unrec in unrecognized:
                if unrec.startswith("--") and unrec[2:] in params_list:
                    # Set the missing boolean value to true
                    setattr(params, unrec[2:], True)
        else:
            params = parser.parse_args(args=args)

        return params

    def __deepcopy__(self, memo) -> "Config":
        _default = self.__default__

        config_state = self.__getstate__()
        config_copy = Config()
        memo[id(self)] = config_copy

        config_copy.__setstate__(config_state)
        config_copy.__default__ = _default

        config_copy["__is_set"] = deepcopy(self["__is_set"], memo)

        return config_copy

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def _remove_private_keys(d):
        if "__parser" in d:
            d.pop("__parser", None)
        if "__is_set" in d:
            d.pop("__is_set", None)
        for k, v in list(d.items()):
            if isinstance(v, dict):
                Config._remove_private_keys(v)
        return d

    def __str__(self) -> str:
        # remove the parser and is_set map from the visible config
        visible = copy.deepcopy(self.toDict())
        visible.pop("__parser", None)
        visible.pop("__is_set", None)
        cleaned = Config._remove_private_keys(visible)
        return "\n" + yaml.dump(cleaned, sort_keys=False)

    def copy(self) -> "Config":
        return copy.deepcopy(self)

    @staticmethod
    def to_string(items) -> str:
        """Get string from items."""
        return "\n" + yaml.dump(items.toDict())

    def update_with_kwargs(self, kwargs):
        """Add config to self"""
        for key, val in kwargs.items():
            self[key] = val

    @classmethod
    def _merge(cls, a, b):
        """
        Merge two configurations recursively.
        If there is a conflict, the value from the second configuration will take precedence.
        """
        for key in b:
            if key in a:
                if isinstance(a[key], dict) and isinstance(b[key], dict):
                    a[key] = cls._merge(a[key], b[key])
                else:
                    a[key] = b[key]
            else:
                a[key] = b[key]
        return a

    def merge(self, b: "Config"):
        """
        Merges the current config with another config.

        Args:
            b (bittensor.core.config.Config): Another config to merge.
        """
        self._merge(self, b)

    @classmethod
    def merge_all(cls, configs: list["Config"]) -> "Config":
        """
        Merge all configs in the list into one config.
        If there is a conflict, the value from the last configuration in the list will take precedence.

        Args:
            configs (list[bittensor.core.config.Config]): List of configs to be merged.

        Returns:
            config (bittensor.core.config.Config): Merged config object.
        """
        result = cls()
        for cfg in configs:
            result.merge(cfg)
        return result

    def is_set(self, param_name: str) -> bool:
        """Returns a boolean indicating whether the parameter has been set or is still the default."""
        if param_name not in self.get("__is_set"):
            return False
        else:
            return self.get("__is_set")[param_name]

    def __check_for_missing_required_args(
        self, parser: argparse.ArgumentParser, args: list[str]
    ) -> list[str]:
        required_args = self.__get_required_args_from_parser(parser)
        missing_args = [arg for arg in required_args if not any(arg in s for s in args)]
        return missing_args

    @staticmethod
    def __get_required_args_from_parser(parser: argparse.ArgumentParser) -> list[str]:
        required_args = []
        for action in parser._actions:
            if action.required:
                # Prefix the argument with '--' if it's a long argument, or '-' if it's short
                prefix = "--" if len(action.dest) > 1 else "-"
                required_args.append(prefix + action.dest)
        return required_args


T = TypeVar("T", bound="DefaultConfig")


class DefaultConfig(Config):
    """A Config with a set of default values."""

    @classmethod
    def default(cls: Type[T]) -> T:
        """Get default config."""
        raise NotImplementedError("Function default is not implemented.")
