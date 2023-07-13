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
from argparse import ArgumentParser, Namespace


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
            parser.add_argument('--config', type=str, help='If set, defaults are overridden by the passed file.')
        except:
            # this can fail if the --config has already been added.
            pass
        try:
            parser.add_argument('--strict', action='store_true',
                                help='If flagged, config will check that only exact arguments have been set.',
                                default=False)
        except:
            # this can fail if the --config has already been added.
            pass

        # Get args from argv if not passed in.
        if args is None:
            args = sys.argv[1:]

        # 1.1 Optionally load defaults if the --config is set.
        try:
            config_file_path = str(os.getcwd()) + '/' + vars(parser.parse_known_args(args)[0])['config']
        except Exception as e:
            config_file_path = None

        # Parse args not strict
        params = config.__parse_args__(args=args, parser=parser, strict=False)

        # 2. Optionally check for --strict, if strict we will parse the args strictly.
        strict = params.strict

        if config_file_path is not None:
            config_file_path = os.path.expanduser(config_file_path)
            try:
                with open(config_file_path) as f:
                    params_config = yaml.safe_load(f)
                    print('Loading config defaults from: {}'.format(config_file_path))
                    parser.set_defaults(**params_config)
            except Exception as e:
                print('Error in loading: {} using default parser settings'.format(e))

        # 2. Continue with loading in params.
        params = config.__parse_args__(args=args, parser=parser, strict=strict)

        _config = self

        # Splits params on dot syntax i.e neuron.axon_port
        for arg_key, arg_val in params.__dict__.items():
            split_keys = arg_key.split('.')
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

    @staticmethod
    def __parse_args__(args: List[str], parser: ArgumentParser = None, strict: bool = False) -> Namespace:
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

    def copy(self) -> 'config':
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
    def merge_all(cls, configs: List['config']) -> 'config':
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

        Args:
            param_name (str):
                The name of the parameter to check.

        Returns:
            bool:
                True if the parameter is set, False otherwise.
        """
        keys = param_name.split('.')
        if len(keys) == 1:
            return keys[0] in self
        else:
            next_config = self.get(keys[0])
            if isinstance(next_config, config):
                return next_config.is_set('.'.join(keys[1:]))
            else:
                return False
            
#########
# Tests #
#########         

import os
import yaml
import unittest
import argparse
import bittensor
from argparse import ArgumentParser
from unittest.mock import MagicMock

def test_strict():
    parser = argparse.ArgumentParser()
    # Positional/mandatory arguments don't play nice with multiprocessing.
    # When the CLI is used, the argument is just the 0th element or the filepath.
    # However with multiprocessing this function call actually comes from a subprocess, and so there
    # is no positional argument and this raises an exception when we try to parse the args later.
    # parser.add_argument("arg", help="Dummy Args")
    parser.add_argument("--cov", help="Dummy Args")
    parser.add_argument("--cov-append", action='store_true', help="Dummy Args")
    parser.add_argument("--cov-config",  help="Dummy Args")
    bittensor.logging.add_args( parser )
    bittensor.wallet.add_args( parser )
    bittensor.subtensor.add_args( parser )
    bittensor.axon.add_args( parser )
    bittensor.config( parser, strict=False)
    bittensor.config( parser, strict=True)

def test_prefix():
    # Test the use of prefixes to instantiate all of the bittensor objects.
    parser = argparse.ArgumentParser()

    mock_wallet = MagicMock(
        spec=bittensor.wallet,
        coldkey=MagicMock(),
        coldkeypub=MagicMock(
            # mock ss58 address
            ss58_address="5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
        ),
        hotkey=MagicMock(
            ss58_address="5CtstubuSoVLJGCXkiWRNKrrGg2DVBZ9qMs2qYTLsZR4q1Wg"
        ),
    )

    bittensor.logging.add_args( parser )
    bittensor.logging.add_args( parser, prefix = 'second' )

    bittensor.wallet.add_args( parser )
    bittensor.wallet.add_args( parser, prefix = 'second' )

    bittensor.subtensor.add_args( parser )
    bittensor.subtensor.add_args( parser, prefix = 'second'  )

    bittensor.axon.add_args( parser )
    bittensor.axon.add_args( parser, prefix = 'second' )

    config_non_strict = bittensor.config( parser, strict=False)
    config_strict = bittensor.config( parser, strict=True)

    bittensor.axon( wallet=mock_wallet, config=config_strict ).stop()
    bittensor.axon( wallet=mock_wallet, config=config_non_strict ).stop()
    bittensor.axon( wallet=mock_wallet, config=config_strict.second ).stop()
    bittensor.axon( wallet=mock_wallet, config=config_non_strict.second ).stop()

    bittensor.wallet( config_strict )
    bittensor.wallet( config_non_strict )
    bittensor.wallet( config_strict.second )
    bittensor.wallet( config_non_strict.second )

    bittensor.logging( config_strict )
    bittensor.logging( config_non_strict )
    bittensor.logging( config_strict.second )
    bittensor.logging( config_non_strict.second )


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.parser = ArgumentParser()
        self.parser.add_argument('--name', type=str, default='John')
        self.parser.add_argument('--age', type=int, default=30)
        self.parser.add_argument('--nested.param', type=float, default=1.23)

    def test_config_creation_from_parser(self):
        args = ['--name', 'Alice', '--age', '25', '--nested.param', '4.56']
        cfg = config(self.parser, args)
        
        # Check if the values are correctly set
        self.assertEqual(cfg.name, 'Alice')
        self.assertEqual(cfg.age, 25)
        self.assertEqual(cfg.nested.param, 4.56)

    def test_config_merge(self):
        # Create two config objects
        cfg1 = config(self.parser, ['--name', 'Alice', '--age', '25'])
        cfg2 = config(self.parser, ['--age', '30', '--nested.param', '4.56'])

        # Merge the two configs
        cfg2.merge(cfg1)

        # Check if the values are correctly merged
        self.assertEqual(cfg1.name, 'Alice')
        self.assertEqual(cfg1.age, 25)
        self.assertEqual(cfg1.nested.param, 1.23)

    def test_config_is_set(self):
        cfg = config(self.parser, ['--name', 'Alice'])
        
        # Check if the parameters are correctly set
        self.assertTrue(cfg.is_set('name'))
        self.assertTrue(cfg.is_set('age'))
        self.assertTrue(cfg.is_set('nested.param'))
    
        # Check if nested parameters are correctly set
        self.assertTrue(cfg.is_set('nested'))
        self.assertTrue(cfg.is_set('nested.param'))

if __name__ == '__main__':
    unittest.main()

