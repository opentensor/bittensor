# The MIT License (MIT)
# Copyright © 2022 Yuma Rao
# Copyright © 2022-2023 Opentensor Foundation

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


import unittest
from unittest.mock import MagicMock, patch
from typing import Any
import pytest
from copy import deepcopy
import re

from tests.helpers import _get_mock_coldkey

import bittensor


class TestCLINoNetwork(unittest.TestCase):
    _patched_subtensor = None

    @classmethod
    def setUpClass(cls) -> None:
        mock_delegate_info = {
            "hotkey_ss58": "",
            "total_stake": bittensor.Balance.from_rao(0),
            "nominators": [],
            "owner_ss58": "",
            "take": 0.18,
            "validator_permits": [],
            "registrations": [],
            "return_per_1000": bittensor.Balance.from_rao(0),
            "total_daily_return": bittensor.Balance.from_rao(0)
        }
        cls._patched_subtensor = patch('bittensor._subtensor.subtensor_mock.MockSubtensor.__new__', new=MagicMock(
            return_value=MagicMock(
                get_subnets=MagicMock(return_value=[1]), # Mock subnet 1 ONLY.
                block=10_000,
                get_delegates=MagicMock(return_value=[
                    bittensor.DelegateInfo( **mock_delegate_info )
                ]),
            )
        ))
        cls._patched_subtensor.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._patched_subtensor.stop()

    def setUp(self):
        self._config = TestCLINoNetwork.construct_config()

    @property
    def config(self):
        copy_ = deepcopy(self._config)
        return copy_

    @staticmethod
    def construct_config():
        defaults = bittensor.Config()

        defaults.netuid = 1
        bittensor.subtensor.add_defaults( defaults )
        defaults.subtensor.network = 'mock'
        defaults.no_version_checking = True
        bittensor.axon.add_defaults( defaults )
        bittensor.wallet.add_defaults( defaults )
        bittensor.dataset.add_defaults( defaults )

        return defaults

    def test_check_configs(self):
        config = self.config
        config.no_prompt = True
        config.model = "core_server"
        config.dest = "no_prompt"
        config.amount = 1
        config.mnemonic = "this is a mnemonic"
        config.seed = None
        config.uids = [1,2,3]
        config.weights = [0.25, 0.25, 0.25, 0.25]
        config.no_version_checking = True
        config.ss58_address = bittensor.Keypair.create_from_seed( b'0' * 32 ).ss58_address
        config.public_key_hex = None
        config.proposal_hash = ""

        cli = bittensor.cli

        # Get argparser
        parser = cli.__create_parser__()
        # Get all commands from argparser
        commands = [
            command for command in parser._actions[1].choices
        ]

        def ask_response(prompt: str) -> Any:
            if "delegate index" in prompt:
                return 0
            elif "wallet name" in prompt:
                return "mock"
            elif "hotkey" in prompt:
                return "mock"
        with patch('rich.prompt.Prompt.ask', ask_response):
            for cmd in commands:
                config.command = cmd
                cli.check_config(config)

    def test_new_coldkey( self ):
        config = self.config
        config.wallet.name = "new_coldkey_testwallet"

        config.command = "new_coldkey"
        config.amount = 1
        config.dest = "no_prompt"
        config.model = "core_server"
        config.n_words = 12
        config.use_password = False
        config.no_prompt = True
        config.overwrite_coldkey = True

        cli = bittensor.cli(config)
        cli.run()

    def test_new_hotkey( self ):
        config = self.config
        config.wallet.name = "new_hotkey_testwallet"
        config.command = "new_hotkey"
        config.amount = 1
        config.dest = "no_prompt"
        config.model = "core_server"
        config.n_words = 12
        config.use_password = False
        config.no_prompt = True
        config.overwrite_hotkey = True


        cli = bittensor.cli(config)
        cli.run()

    def test_regen_coldkey( self ):
        config = self.config
        config.wallet.name = "regen_coldkey_testwallet"
        config.command = "regen_coldkey"
        config.amount = 1
        config.dest = "no_prompt"
        config.model = "core_server"
        config.mnemonic = "faculty decade seven jelly gospel axis next radio grain radio remain gentle"
        config.seed = None
        config.n_words = 12
        config.use_password = False
        config.no_prompt = True
        config.overwrite_coldkey = True


        cli = bittensor.cli(config)
        cli.run()

    def test_regen_coldkeypub( self ):
        config = self.config
        config.wallet.name = "regen_coldkeypub_testwallet"
        config.command = "regen_coldkeypub"
        config.ss58_address = "5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
        config.public_key = None
        config.use_password = False
        config.no_prompt = True
        config.overwrite_coldkeypub = True


        cli = bittensor.cli(config)
        cli.run()

    def test_regen_hotkey( self ):
        config = self.config
        config.wallet.name = "regen_hotkey_testwallet"
        config.command = "regen_hotkey"
        config.amount = 1
        config.model = "core_server"
        config.mnemonic = "faculty decade seven jelly gospel axis next radio grain radio remain gentle"
        config.seed = None
        config.n_words = 12
        config.use_password = False
        config.no_prompt = True
        config.overwrite_hotkey = True


        cli = bittensor.cli(config)
        cli.run()

    def test_list( self ):
        # Mock IO for wallet
        with patch('bittensor.wallet', side_effect=[MagicMock(
            coldkeypub_file=MagicMock(
                exists_on_device=MagicMock(
                    return_value=True # Wallet exists
                ),
                is_encrypted=MagicMock(
                    return_value=False # Wallet is not encrypted
                ),
            ),
            coldkeypub=MagicMock(
                ss58_address=bittensor.Keypair.create_from_mnemonic(
                        bittensor.Keypair.generate_mnemonic()
                ).ss58_address
            )
        ),
        MagicMock(
            hotkey_file=MagicMock(
                exists_on_device=MagicMock(
                    return_value=True # Wallet exists
                ),
                is_encrypted=MagicMock(
                    return_value=False # Wallet is not encrypted
                ),
            ),
            hotkey=MagicMock(
                ss58_address=bittensor.Keypair.create_from_mnemonic(
                        bittensor.Keypair.generate_mnemonic()
                ).ss58_address
            )
        )]):
            config = self.config
            config.wallet.path = 'tmp/walletpath'
            config.wallet.name = 'mock_wallet'
            config.no_prompt = True
            config.command = "list"


            cli = bittensor.cli(config)
            with patch('os.walk', side_effect=[iter(
                    [('/tmp/walletpath', ['mock_wallet'], [])] # 1 wallet dir
            ),
            iter(
                [('/tmp/walletpath/mock_wallet/hotkeys', [], ['hk0'])] # 1 hotkey file
            )]):
                cli.run()

    def test_list_no_wallet( self ):
        # Mock IO for wallet
        with patch('bittensor.Wallet.coldkeypub_file', MagicMock(
            exists_on_device=MagicMock(
                return_value=False # Wallet doesn't exist
            )
        )):
            config = self.config
            config.wallet.path = '/tmp/test_cli_test_list_no_wallet'
            config.no_prompt = True
            config.command = "list"


            cli = bittensor.cli(config)
            # This shouldn't raise an error anymore
            cli.run()

    def test_btcli_help(self):
        """
        Verify the correct help text is output when the --help flag is passed
        """
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            with patch('argparse.ArgumentParser._print_message', return_value=None) as mock_print_message:
                args = [
                    '--help'
                ]
                bittensor.cli(args=args).run()

        # Should try to print help
        mock_print_message.assert_called_once()

        call_args = mock_print_message.call_args
        args, _ = call_args
        help_out = args[0]

        # Expected help output even if parser isn't working well
        ## py3.6-3.9 or py3.10+
        assert 'optional arguments' in help_out or 'options' in help_out
        # Expected help output if all commands are listed
        assert 'positional arguments' in help_out
        # Verify that cli is printing the help message for
        # Get argparser
        parser = bittensor.cli.__create_parser__()
        # Get all commands from argparser
        commands = [
            command for command in parser._actions[1].choices
        ]
        # Verify that all commands are listed in the help message, AND
        # Verify there are no duplicate commands
        ##  Listed twice. Once in the positional arguments and once in the optional arguments
        for command in commands:
            pat = re.compile(rf'\n\s+({command})[^\S\r\n]+\w')
            matches = pat.findall(help_out)
            self.assertGreaterEqual( len(matches), 1, f"Command {command} not found in help output")
            self.assertLess( len(matches), 2, f"Duplicate command {command} in help output")

    def test_register_cuda_use_cuda_flag(self):
            class ExitEarlyException(Exception):
                """Raised by mocked function to exit early"""
                pass

            base_args = [
                "register",
                "--wallet.path", "tmp/walletpath",
                "--wallet.name", "mock",
                "--wallet.hotkey", "hk0",
                "--no_prompt",
                "--cuda.dev_id", "0",
                "--network", "mock"
            ]
            bittensor.subtensor.check_config = MagicMock(return_value = True)
            with patch('torch.cuda.is_available', return_value=True):
                with patch('bittensor.Subtensor.get_subnets', return_value = [1]):
                    with patch('bittensor.Subtensor.subnet_exists', side_effect=lambda netuid: netuid == 1):
                        with patch('bittensor.Subtensor.register', side_effect=ExitEarlyException):
                            # Should be able to set true without argument
                            args = base_args + [
                                "--subtensor.register.cuda.use_cuda", # should be True without any arugment
                            ]
                            with pytest.raises(ExitEarlyException):
                                cli = bittensor.cli(args=args)
                                cli.run()

                            assert cli.config.subtensor.register.cuda.get('use_cuda') == True # should be None

                            # Should be able to set to false with no argument

                            args = base_args + [
                                "--subtensor.register.cuda.no_cuda",
                            ]
                            with pytest.raises(ExitEarlyException):
                                cli = bittensor.cli(args=args)
                                cli.run()

                            assert cli.config.subtensor.register.cuda.use_cuda == False

class MockException(Exception):
    pass


class TestEmptyArgs(unittest.TestCase):
    """
    Test that the CLI doesn't crash when no args are passed
    """
    _patched_subtensor = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._patched_subtensor = patch('bittensor._subtensor.subtensor_mock.MockSubtensor.__new__', new=MagicMock(
        ))
        cls._patched_subtensor.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._patched_subtensor.stop()
    
    @patch('rich.prompt.PromptBase.ask', side_effect=MockException)
    def test_command_no_args(self, patched_prompt_ask):
        # Get argparser
        parser = bittensor.cli.__create_parser__()
        # Get all commands from argparser
        commands = [
            command for command in parser._actions[1].choices
        ]

        # Test that each command can be run with no args
        for command in commands:
            try:
                bittensor.cli(args=[
                    command
                ]).run()
            except MockException:
                pass # Expected exception

            # Should not raise any other exceptions
        

class TestCLIDefaultsNoNetwork(unittest.TestCase):
    _patched_subtensor = None

    @classmethod
    def setUpClass(cls) -> None:
        mock_delegate_info = {
            "hotkey_ss58": "",
            "total_stake": bittensor.Balance.from_rao(0),
            "nominators": [],
            "owner_ss58": "",
            "take": 0.18, 
            "validator_permits": [],
            "registrations": [], 
            "return_per_1000": bittensor.Balance.from_rao(0), 
            "total_daily_return": bittensor.Balance.from_rao(0)
        }
        cls._patched_subtensor = patch('bittensor._subtensor.subtensor_mock.MockSubtensor.__new__', new=MagicMock(
            return_value=MagicMock(
                get_subnets=MagicMock(return_value=[1]), # Mock subnet 1 ONLY.
                block=10_000,
                get_delegates=MagicMock(return_value=[
                    bittensor.DelegateInfo( **mock_delegate_info )
                ]),
            )
        ))
        cls._patched_subtensor.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._patched_subtensor.stop()

    def test_inspect_prompt_wallet_name(self):
        # Patch command to exit early
        with patch('bittensor._cli.commands.inspect.InspectCommand.run', return_value=None):

            # Test prompt happens when no wallet name is passed
            with patch('rich.prompt.Prompt.ask') as mock_ask_prompt:
                cli = bittensor.cli(args=[
                        'inspect',
                        # '--wallet.name', 'mock',
                    ])
                cli.run()

                # Prompt happened
                mock_ask_prompt.assert_called_once()

            # Test NO prompt happens when wallet name is passed
            with patch('rich.prompt.Prompt.ask') as mock_ask_prompt:
                cli = bittensor.cli(args=[
                        'inspect',
                        '--wallet.name', 'coolwalletname',
                    ])
                cli.run()

                # NO prompt happened
                mock_ask_prompt.assert_not_called()

            # Test NO prompt happens when wallet name 'default' is passed
            with patch('rich.prompt.Prompt.ask') as mock_ask_prompt:
                cli = bittensor.cli(args=[
                        'inspect',
                        '--wallet.name', 'default',
                    ])
                cli.run()

                # NO prompt happened
                mock_ask_prompt.assert_not_called()

    def test_overview_prompt_wallet_name(self):
        # Patch command to exit early
        with patch('bittensor._cli.commands.overview.OverviewCommand.run', return_value=None):

            # Test prompt happens when no wallet name is passed
            with patch('rich.prompt.Prompt.ask') as mock_ask_prompt:
                cli = bittensor.cli(args=[
                        'overview',
                        # '--wallet.name', 'mock',
                        '--netuid', '1'
                    ])
                cli.run()

                # Prompt happened
                mock_ask_prompt.assert_called_once()

            # Test NO prompt happens when wallet name is passed
            with patch('rich.prompt.Prompt.ask') as mock_ask_prompt:
                cli = bittensor.cli(args=[
                        'overview',
                        '--wallet.name', 'coolwalletname',
                        '--netuid', '1',
                    ])
                cli.run()

                # NO prompt happened
                mock_ask_prompt.assert_not_called()

            # Test NO prompt happens when wallet name 'default' is passed
            with patch('rich.prompt.Prompt.ask') as mock_ask_prompt:
                cli = bittensor.cli(args=[
                        'overview',
                        '--wallet.name', 'default',
                        '--netuid', '1',
                    ])
                cli.run()

                # NO prompt happened
                mock_ask_prompt.assert_not_called()

    def test_stake_prompt_wallet_name_and_hotkey_name(self):
        base_args = [
            'stake',
            '--all',
        ]
        # Patch command to exit early
        with patch('bittensor._cli.commands.stake.StakeCommand.run', return_value=None):

            # Test prompt happens when 
            # - wallet name IS NOT passed, AND
            # - hotkey name IS NOT passed
            with patch('rich.prompt.Prompt.ask') as mock_ask_prompt:
                mock_ask_prompt.side_effect = ['mock', 'mock_hotkey']

                cli = bittensor.cli(args=base_args + [
                        # '--wallet.name', 'mock',
                        #'--wallet.hotkey', 'mock_hotkey', 
                    ])
                cli.run()

                # Prompt happened
                mock_ask_prompt.assert_called()
                self.assertEqual(mock_ask_prompt.call_count, 2, msg="Prompt should have been called twice")
                args0, kwargs0 = mock_ask_prompt.call_args_list[0]
                combined_args_kwargs0 = [arg for arg in args0] + [val for val in [val for val in kwargs0.values()]]
                # check that prompt was called for wallet name
                self.assertTrue(
                    any(filter(lambda x: 'wallet name' in x.lower(), combined_args_kwargs0)),
                    msg=f"Prompt should have been called for wallet name: {combined_args_kwargs0}"
                )

                args1, kwargs1 = mock_ask_prompt.call_args_list[1]
                combined_args_kwargs1 = [arg for arg in args1] + [val for val in kwargs1.values()]
                # check that prompt was called for hotkey

                self.assertTrue(
                    any(filter(lambda x: 'hotkey' in x.lower(), combined_args_kwargs1)),
                    msg=f"Prompt should have been called for hotkey: {combined_args_kwargs1}"
                )

            # Test prompt happens when 
            # - wallet name IS NOT passed, AND
            # - hotkey name IS passed
            with patch('rich.prompt.Prompt.ask') as mock_ask_prompt:
                mock_ask_prompt.side_effect = ['mock', 'mock_hotkey']

                cli = bittensor.cli(args=base_args + [
                        #'--wallet.name', 'mock',
                        '--wallet.hotkey', 'mock_hotkey', 
                    ])
                cli.run()

                # Prompt happened
                mock_ask_prompt.assert_called()
                self.assertEqual(mock_ask_prompt.call_count, 1, msg="Prompt should have been called ONCE")
                args0, kwargs0 = mock_ask_prompt.call_args_list[0]
                combined_args_kwargs0 = [arg for arg in args0] + [val for val in kwargs0.values()]
                # check that prompt was called for wallet name
                self.assertTrue(
                    any(filter(lambda x: 'wallet name' in x.lower(), combined_args_kwargs0)),
                    msg=f"Prompt should have been called for wallet name: {combined_args_kwargs0}"
                )

            # Test prompt happens when 
            # - wallet name IS passed, AND
            # - hotkey name IS NOT passed
            with patch('rich.prompt.Prompt.ask') as mock_ask_prompt:
                mock_ask_prompt.side_effect = ['mock', 'mock_hotkey']

                cli = bittensor.cli(args=base_args + [
                        '--wallet.name', 'mock',
                        #'--wallet.hotkey', 'mock_hotkey', 
                    ])
                cli.run()

                # Prompt happened
                mock_ask_prompt.assert_called()
                self.assertEqual(mock_ask_prompt.call_count, 1, msg="Prompt should have been called ONCE")
                args0, kwargs0 = mock_ask_prompt.call_args_list[0]
                combined_args_kwargs0 = [arg for arg in args0] + [val for val in kwargs0.values()]
                # check that prompt was called for hotkey
                self.assertTrue(
                    any(filter(lambda x: 'hotkey' in x.lower(), combined_args_kwargs0)),
                    msg=f"Prompt should have been called for hotkey {combined_args_kwargs0}"
                )


            # Test NO prompt happens when
            # - wallet name IS passed, AND
            # - hotkey name IS passed
            with patch('rich.prompt.Prompt.ask') as mock_ask_prompt:
                cli = bittensor.cli(args=base_args + [
                        '--wallet.name', 'coolwalletname',
                        '--wallet.hotkey', 'coolwalletname_hotkey',
                    ])
                cli.run()

                # NO prompt happened
                mock_ask_prompt.assert_not_called()

            # Test NO prompt happens when
            # - wallet name 'default' IS passed, AND
            # - hotkey name 'default' IS passed
            with patch('rich.prompt.Prompt.ask') as mock_ask_prompt:
                cli = bittensor.cli(args=base_args + [
                        '--wallet.name', 'default',
                        '--wallet.hotkey', 'default',
                    ])
                cli.run()

                # NO prompt happened
                mock_ask_prompt.assert_not_called()

    def test_unstake_prompt_wallet_name_and_hotkey_name(self):
        base_args = [
            'unstake',
            '--all',
        ]
        # Patch command to exit early
        with patch('bittensor._cli.commands.unstake.UnStakeCommand.run', return_value=None):

            # Test prompt happens when 
            # - wallet name IS NOT passed, AND
            # - hotkey name IS NOT passed
            with patch('rich.prompt.Prompt.ask') as mock_ask_prompt:
                mock_ask_prompt.side_effect = ['mock', 'mock_hotkey']

                cli = bittensor.cli(args=base_args + [
                        # '--wallet.name', 'mock',
                        #'--wallet.hotkey', 'mock_hotkey', 
                    ])
                cli.run()

                # Prompt happened
                mock_ask_prompt.assert_called()
                self.assertEqual(mock_ask_prompt.call_count, 2, msg="Prompt should have been called twice")
                args0, kwargs0 = mock_ask_prompt.call_args_list[0]
                combined_args_kwargs0 = [arg for arg in args0] + [val for val in kwargs0.values()]
                # check that prompt was called for wallet name
                self.assertTrue(
                    any(filter(lambda x: 'wallet name' in x.lower(), combined_args_kwargs0)),
                    msg=f"Prompt should have been called for wallet name: {combined_args_kwargs0}"
                )

                args1, kwargs1 = mock_ask_prompt.call_args_list[1]
                combined_args_kwargs1 = [arg for arg in args1] + [val for val in kwargs1.values()]
                # check that prompt was called for hotkey
                self.assertTrue(
                    any(filter(lambda x: 'hotkey' in x.lower(), combined_args_kwargs1)),
                    msg=f"Prompt should have been called for hotkey {combined_args_kwargs1}"
                )

            # Test prompt happens when 
            # - wallet name IS NOT passed, AND
            # - hotkey name IS passed
            with patch('rich.prompt.Prompt.ask') as mock_ask_prompt:
                mock_ask_prompt.side_effect = ['mock', 'mock_hotkey']

                cli = bittensor.cli(args=base_args + [
                        #'--wallet.name', 'mock',
                        '--wallet.hotkey', 'mock_hotkey', 
                    ])
                cli.run()

                # Prompt happened
                mock_ask_prompt.assert_called()
                self.assertEqual(mock_ask_prompt.call_count, 1, msg="Prompt should have been called ONCE")
                args0, kwargs0 = mock_ask_prompt.call_args_list[0]
                combined_args_kwargs0 = [arg for arg in args0] + [val for val in kwargs0.values()]
                # check that prompt was called for wallet name
                self.assertTrue(
                    any(filter(lambda x: 'wallet name' in x.lower(), combined_args_kwargs0)),
                    msg=f"Prompt should have been called for wallet name: {combined_args_kwargs0}"
                )

            # Test prompt happens when 
            # - wallet name IS passed, AND
            # - hotkey name IS NOT passed
            with patch('rich.prompt.Prompt.ask') as mock_ask_prompt:
                mock_ask_prompt.side_effect = ['mock', 'mock_hotkey']

                cli = bittensor.cli(args=base_args + [
                        '--wallet.name', 'mock',
                        #'--wallet.hotkey', 'mock_hotkey', 
                    ])
                cli.run()

                # Prompt happened
                mock_ask_prompt.assert_called()
                self.assertEqual(mock_ask_prompt.call_count, 1, msg="Prompt should have been called ONCE")
                args0, kwargs0 = mock_ask_prompt.call_args_list[0]
                combined_args_kwargs0 = [arg for arg in args0] + [val for val in kwargs0.values()]
                # check that prompt was called for hotkey
                self.assertTrue(
                    any(filter(lambda x: 'hotkey' in x.lower(), combined_args_kwargs0)),
                    msg=f"Prompt should have been called for hotkey {combined_args_kwargs0}"
                )


            # Test NO prompt happens when
            # - wallet name IS passed, AND
            # - hotkey name IS passed
            with patch('rich.prompt.Prompt.ask') as mock_ask_prompt:
                cli = bittensor.cli(args=base_args + [
                        '--wallet.name', 'coolwalletname',
                        '--wallet.hotkey', 'coolwalletname_hotkey',
                    ])
                cli.run()

                # NO prompt happened
                mock_ask_prompt.assert_not_called()

            # Test NO prompt happens when
            # - wallet name 'default' IS passed, AND
            # - hotkey name 'default' IS passed
            with patch('rich.prompt.Prompt.ask') as mock_ask_prompt:
                cli = bittensor.cli(args=base_args + [
                        '--wallet.name', 'default',
                        '--wallet.hotkey', 'default',
                    ])
                cli.run()

                # NO prompt happened
                mock_ask_prompt.assert_not_called()

    def test_delegate_prompt_wallet_name(self):
        base_args = [
            'delegate',
            '--all',
            '--delegate_ss58key', _get_mock_coldkey(0)
        ]
        # Patch command to exit early
        with patch('bittensor._cli.commands.delegates.DelegateStakeCommand.run', return_value=None):

            # Test prompt happens when 
            # - wallet name IS NOT passed
            with patch('rich.prompt.Prompt.ask') as mock_ask_prompt:
                mock_ask_prompt.side_effect = ['mock']

                cli = bittensor.cli(args=base_args + [
                        # '--wallet.name', 'mock',
                    ])
                cli.run()

                # Prompt happened
                mock_ask_prompt.assert_called()
                self.assertEqual(mock_ask_prompt.call_count, 1, msg="Prompt should have been called ONCE")
                args0, kwargs0 = mock_ask_prompt.call_args_list[0]
                combined_args_kwargs0 = [arg for arg in args0] + [val for val in kwargs0.values()]
                # check that prompt was called for wallet name
                self.assertTrue(
                    any(filter(lambda x: 'wallet name' in x.lower(), combined_args_kwargs0)),
                    msg=f"Prompt should have been called for wallet name: {combined_args_kwargs0}"
                )

            # Test NO prompt happens when
            # - wallet name IS passed
            with patch('rich.prompt.Prompt.ask') as mock_ask_prompt:
                cli = bittensor.cli(args=base_args + [
                        '--wallet.name', 'coolwalletname',
                    ])
                cli.run()

                # NO prompt happened
                mock_ask_prompt.assert_not_called()

    def test_undelegate_prompt_wallet_name(self):
        base_args = [
            'undelegate',
            '--all',
            '--delegate_ss58key', _get_mock_coldkey(0)
        ]
        # Patch command to exit early
        with patch('bittensor._cli.commands.delegates.DelegateUnstakeCommand.run', return_value=None):

            # Test prompt happens when 
            # - wallet name IS NOT passed
            with patch('rich.prompt.Prompt.ask') as mock_ask_prompt:
                mock_ask_prompt.side_effect = ['mock']

                cli = bittensor.cli(args=base_args + [
                        # '--wallet.name', 'mock',
                    ])
                cli.run()

                # Prompt happened
                mock_ask_prompt.assert_called()
                self.assertEqual(mock_ask_prompt.call_count, 1, msg="Prompt should have been called ONCE")
                args0, kwargs0 = mock_ask_prompt.call_args_list[0]
                combined_args_kwargs0 = [arg for arg in args0] + [val for val in kwargs0.values()]
                # check that prompt was called for wallet name
                self.assertTrue(
                    any(filter(lambda x: 'wallet name' in x.lower(), combined_args_kwargs0)),
                    msg=f"Prompt should have been called for wallet name: {combined_args_kwargs0}"
                )

            # Test NO prompt happens when
            # - wallet name IS passed
            with patch('rich.prompt.Prompt.ask') as mock_ask_prompt:
                cli = bittensor.cli(args=base_args + [
                        '--wallet.name', 'coolwalletname',
                    ])
                cli.run()

                # NO prompt happened
                mock_ask_prompt.assert_not_called()

    def test_delegate_prompt_hotkey(self):
        # Tests when
        # - wallet name IS passed, AND
        # - delegate hotkey IS NOT passed
        base_args = [
            'delegate',
            '--all',
            '--wallet.name', 'mock', 
        ]

        delegate_ss58 = _get_mock_coldkey(0)
        with patch('bittensor._cli.commands.delegates.show_delegates'):
            with patch('bittensor.Subtensor.get_delegates', return_value=[
                bittensor.DelegateInfo(
                    hotkey_ss58=delegate_ss58, # return delegate with mock coldkey
                    total_stake=bittensor.Balance.from_float(0.1),
                    nominators=[],
                    owner_ss58='',
                    take=0.18,
                    validator_permits=[],
                    registrations=[],
                    return_per_1000=bittensor.Balance.from_float(0.1),
                    total_daily_return=bittensor.Balance.from_float(0.1)
                )
            ]):
                # Patch command to exit early
                with patch('bittensor._cli.commands.delegates.DelegateStakeCommand.run', return_value=None):

                    # Test prompt happens when 
                    # - delegate hotkey IS NOT passed
                    with patch('rich.prompt.Prompt.ask') as mock_ask_prompt:
                        mock_ask_prompt.side_effect = ['0'] # select delegate with mock coldkey

                        cli = bittensor.cli(args=base_args + [
                                # '--delegate_ss58key', delegate_ss58,
                            ])
                        cli.run()

                        # Prompt happened
                        mock_ask_prompt.assert_called()
                        self.assertEqual(mock_ask_prompt.call_count, 1, msg="Prompt should have been called ONCE")
                        args0, kwargs0 = mock_ask_prompt.call_args_list[0]
                        combined_args_kwargs0 = [arg for arg in args0] + [val for val in kwargs0.values()]
                        # check that prompt was called for delegate hotkey 
                        self.assertTrue(
                            any(filter(lambda x: 'delegate' in x.lower(), combined_args_kwargs0)),
                            msg=f"Prompt should have been called for delegate: {combined_args_kwargs0}"
                        )

                    # Test NO prompt happens when
                    # - delegate hotkey IS passed
                    with patch('rich.prompt.Prompt.ask') as mock_ask_prompt:
                        cli = bittensor.cli(args=base_args + [
                                '--delegate_ss58key', delegate_ss58,
                            ])
                        cli.run()

                        # NO prompt happened
                        mock_ask_prompt.assert_not_called()

    def test_undelegate_prompt_hotkey(self):
        # Tests when
        # - wallet name IS passed, AND
        # - delegate hotkey IS NOT passed
        base_args = [
            'undelegate',
            '--all',
            '--wallet.name', 'mock', 
        ]

        delegate_ss58 = _get_mock_coldkey(0)
        with patch('bittensor._cli.commands.delegates.show_delegates'):
            with patch('bittensor.Subtensor.get_delegates', return_value=[
                bittensor.DelegateInfo(
                    hotkey_ss58=delegate_ss58, # return delegate with mock coldkey
                    total_stake=bittensor.Balance.from_float(0.1),
                    nominators=[],
                    owner_ss58='',
                    take=0.18,
                    validator_permits=[],
                    registrations=[],
                    return_per_1000=bittensor.Balance.from_float(0.1),
                    total_daily_return=bittensor.Balance.from_float(0.1)
                )
            ]):
                # Patch command to exit early
                with patch('bittensor._cli.commands.delegates.DelegateUnstakeCommand.run', return_value=None):

                    # Test prompt happens when 
                    # - delegate hotkey IS NOT passed
                    with patch('rich.prompt.Prompt.ask') as mock_ask_prompt:
                        mock_ask_prompt.side_effect = ['0'] # select delegate with mock coldkey

                        cli = bittensor.cli(args=base_args + [
                                # '--delegate_ss58key', delegate_ss58,
                            ])
                        cli.run()

                        # Prompt happened
                        mock_ask_prompt.assert_called()
                        self.assertEqual(mock_ask_prompt.call_count, 1, msg="Prompt should have been called ONCE")
                        args0, kwargs0 = mock_ask_prompt.call_args_list[0]
                        combined_args_kwargs0 = [arg for arg in args0] + [val for val in kwargs0.values()]
                        # check that prompt was called for delegate hotkey 
                        self.assertTrue(
                            any(filter(lambda x: 'delegate' in x.lower(), combined_args_kwargs0)),
                            msg=f"Prompt should have been called for delegate: {combined_args_kwargs0}"
                        )

                    # Test NO prompt happens when
                    # - delegate hotkey IS passed
                    with patch('rich.prompt.Prompt.ask') as mock_ask_prompt:
                        cli = bittensor.cli(args=base_args + [
                                '--delegate_ss58key', delegate_ss58,
                            ])
                        cli.run()

                        # NO prompt happened
                        mock_ask_prompt.assert_not_called()



if __name__ == "__main__":
    unittest.main()