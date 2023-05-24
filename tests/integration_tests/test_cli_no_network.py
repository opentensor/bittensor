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
        cls._patched_subtensor = patch('bittensor._subtensor.subtensor_mock.mock_subtensor.mock', new=MagicMock(
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
        bittensor.dendrite.add_defaults( defaults )
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
        # Verify that all commands are listed in the help message
        for command in commands:
            assert command in help_out
        
        # Verify there are no duplicate commands
        # Listed twice. Once in the positional arguments and once in the optional arguments
        for command in commands:
            pat = re.compile(rf'\n\s+({command})\s+\w')
            matches = pat.findall(help_out)
        
            self.assertEqual( len(matches), 1, f"Duplicate command {command} in help output")

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

if __name__ == "__main__":
    unittest.main()