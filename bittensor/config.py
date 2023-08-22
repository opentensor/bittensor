"""
Implementation of the config class, which manages the config of different bittensor modules.
"""
# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2022 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import sys
import yaml
import copy
from munch import Munch
from typing import List, Optional
from argparse import ArgumentParser, Namespace, SUPPRESS, _SubParsersAction


class config(Munch):
    """
    Implementation of the config class, which manages the config of different bittensor modules.
    """

    def __init__(
        self,
        parser: ArgumentParser = None,
        args: Optional[List[str]] = None,
        strict: bool = False,
    ):
        """
        Initializes a new config object.

        Args:
            parser (argparse.Parser):
                Command line parser object.
            args (list of str):
                Command line arguments.
            strict (bool):
                If true, the command line arguments are strictly parsed.
        """
        super().__init__()

        # Base empty config
        if parser is None and args is None:
            return

        # Optionally add config specific arguments
        try:
            parser.add_argument(
                "--config",
                type=str,
                help="If set, defaults are overridden by the passed file.",
            )
        except:
            # this can fail if --config has already been added.
            pass

        try:
            parser.add_argument(
                "--strict",
                action="store_true",
                help="If flagged, config will check that only exact arguments have been set.",
                default=False,
            )
        except:
            # this can fail if --strict has already been added.
            pass

        # Get args from argv if not passed in.
        if args is None:
            args = sys.argv[1:]

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
        config_params = config.__parse_args__(args=args, parser=parser, strict=False)

        # 2. Optionally check for --strict, if strict we will parse the args strictly.
        ## strict=True when passed in OR when --strict is set
        strict = config_params.strict or strict

        if config_file_path is not None:
            config_file_path = os.path.expanduser(config_file_path)
            try:
                with open(config_file_path) as f:
                    params_config = yaml.safe_load(f)
                    print("Loading config defaults from: {}".format(config_file_path))
                    parser.set_defaults(**params_config)
            except Exception as e:
                print("Error in loading: {} using default parser settings".format(e))

        # 2. Continue with loading in params.
        params = config.__parse_args__(args=args, parser=parser, strict=strict)

        _config = self
        _config["__parser"] = parser

        # Make the is_set map
        _config["__is_set"] = {}

        # Splits params on dot syntax i.e neuron.axon_port
        # The is_set map only sets True if a value is different from the default values.
        for arg_key, arg_val in params.__dict__.items():
            default_val = parser.get_default(arg_key)
            split_keys = arg_key.split(".")
            head = _config
            keys = split_keys
            while len(keys) > 1:
                if hasattr(head, keys[0]):
                    head = getattr(head, keys[0])
                    keys = keys[1:]
                else:
                    head[keys[0]] = config()
                    head = head[keys[0]]
                    keys = keys[1:]
            if len(keys) == 1:
                head[keys[0]] = arg_val
                if arg_val != default_val:
                    _config["__is_set"][arg_key] = True

        ## Reparse args using default of unset
        parser_no_defaults = copy.deepcopy(parser)
        ## Get all args by name
        default_params = parser.parse_args(
            args=[_config.get("command")]  # Only command as the arg, else no args
            if _config.get("command") != None
            else []
        )
        all_default_args = default_params.__dict__.keys() | []
        ## Make a dict with keys as args and values as argparse.SUPPRESS
        defaults_as_suppress = {key: SUPPRESS for key in all_default_args}
        ## Set the defaults to argparse.SUPPRESS, should remove them from the namespace
        parser_no_defaults.set_defaults(**defaults_as_suppress)
        parser_no_defaults._defaults.clear()  # Needed for quirk of argparse

        ### Check for subparsers and do the same
        if parser_no_defaults._subparsers != None:
            for action in parser_no_defaults._subparsers._actions:
                # Should only be the "command" subparser action
                if isinstance(action, _SubParsersAction):
                    # Set the defaults to argparse.SUPPRESS, should remove them from the namespace
                    # Each choice is the keyword for a command, we need to set the defaults for each of these
                    ## Note: we also need to clear the _defaults dict for each, this is a quirk of argparse
                    cmd_parser: ArgumentParser
                    for cmd_parser in action.choices.values():
                        cmd_parser.set_defaults(**defaults_as_suppress)
                        cmd_parser._defaults.clear()  # Needed for quirk of argparse

        ## Reparse the args, but this time with the defaults as argparse.SUPPRESS
        params_no_defaults = self.__parse_args__(
            args=args, parser=parser_no_defaults, strict=strict
        )

        ## Diff the params and params_no_defaults to get the is_set map
        _config["__is_set"] = {
            arg_key: True
            for arg_key in [
                k
                for k, _ in filter(
                    lambda kv: kv[1] != SUPPRESS, params_no_defaults.__dict__.items()
                )
            ]
        }

    @staticmethod
    def __parse_args__(
        args: List[str], parser: ArgumentParser = None, strict: bool = False
    ) -> Namespace:
        """
        Parses the passed args using the passed parser.

        Args:
            args (List[str]):
                List of arguments to parse.
            parser (argparse.ArgumentParser):
                Command line parser object.
            strict (bool):
                If true, the command line arguments are strictly parsed.

        Returns:
            Namespace:
                Namespace object created from parser arguments.
        """
        if not strict:
            params = parser.parse_known_args(args=args)[0]
        else:
            params = parser.parse_args(args=args)

        return params

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return "\n" + yaml.dump(self.toDict())

    def copy(self) -> "config":
        return copy.deepcopy(self)

    def to_string(self, items) -> str:
        """
        Returns a string representation of the items.

        Args:
            items: Items to convert to a string.

        Returns:
            str: String representation of the items.
        """
        return "\n" + yaml.dump(items.toDict())

    def update_with_kwargs(self, kwargs):
        """
        Updates the config with the given keyword arguments.

        Args:
            kwargs: Keyword arguments to update the config.
        """
        for key, val in kwargs.items():
            self[key] = val

    @classmethod
    def _merge(cls, a, b):
        """
        Merge two configurations recursively.
        If there is a conflict, the value from the second configuration will take precedence.

        Args:
            a: First configuration to merge.
            b: Second configuration to merge.

        Returns:
            Merged configuration.
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

    def merge(self, b):
        """
        Merges the current config with another config.

        Args:
            b: Another config to merge.
        """
        self = self._merge(self, b)

    @classmethod
    def merge_all(cls, configs: List["config"]) -> "config":
        """
        Merge all configs in the list into one config.
        If there is a conflict, the value from the last configuration in the list will take precedence.

        Args:
            configs (list of config):
                List of configs to be merged.

        Returns:
            config:
                Merged config object.
        """
        result = cls()
        for cfg in configs:
            result.merge(cfg)
        return result

    def is_set(self, param_name: str) -> bool:
        """
        Returns a boolean indicating whether the parameter has been set or is still the default.
        """
        if param_name not in self.get("__is_set"):
            return False
        else:
            return self.get("__is_set")[param_name]
