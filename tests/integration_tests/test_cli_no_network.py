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
from typing import Any, Optional
import pytest
from copy import deepcopy
import re

from tests.helpers import _get_mock_coldkey, __mock_wallet_factory__, MockConsole

import bittensor
from bittensor import Balance
from rich.table import Table


class MockException(Exception):
    pass


mock_delegate_info = {
    "hotkey_ss58": "",
    "total_stake": bittensor.Balance.from_rao(0),
    "nominators": [],
    "owner_ss58": "",
    "take": 0.18,
    "validator_permits": [],
    "registrations": [],
    "return_per_1000": bittensor.Balance.from_rao(0),
    "total_daily_return": bittensor.Balance.from_rao(0),
}


def return_mock_sub_1(*args, **kwargs):
    return MagicMock(
        return_value=MagicMock(
            get_subnets=MagicMock(return_value=[1]),  # Mock subnet 1 ONLY.
            block=10_000,
            get_delegates=MagicMock(
                return_value=[bittensor.DelegateInfo(**mock_delegate_info)]
            ),
        )
    )


def return_mock_wallet_factory(*args, **kwargs):
    return MagicMock(
        return_value=__mock_wallet_factory__(*args, **kwargs),
        add_args=bittensor.wallet.add_args,
    )


@patch(
    "bittensor.subtensor",
    new_callable=return_mock_sub_1,
)
@patch("bittensor.wallet", new_callable=return_mock_wallet_factory)
class TestCLINoNetwork(unittest.TestCase):
    def setUp(self):
        self._config = TestCLINoNetwork.construct_config()

    def config(self):
        copy_ = deepcopy(self._config)
        return copy_

    @staticmethod
    def construct_config():
        parser = bittensor.cli.__create_parser__()
        defaults = bittensor.config(parser=parser, args=["subnets", "metagraph"])

        # Parse commands and subcommands
        for command in bittensor.ALL_COMMANDS:
            if (
                command in bittensor.ALL_COMMANDS
                and "commands" in bittensor.ALL_COMMANDS[command]
            ):
                for subcommand in bittensor.ALL_COMMANDS[command]["commands"]:
                    defaults.merge(
                        bittensor.config(parser=parser, args=[command, subcommand])
                    )
            else:
                defaults.merge(bittensor.config(parser=parser, args=[command]))

        defaults.netuid = 1
        defaults.subtensor.network = "mock"
        defaults.no_version_checking = True

        return defaults

    def test_check_configs(self, _, __):
        config = self.config()
        config.no_prompt = True
        config.model = "core_server"
        config.dest = "no_prompt"
        config.amount = 1
        config.mnemonic = "this is a mnemonic"
        config.seed = None
        config.uids = [1, 2, 3]
        config.weights = [0.25, 0.25, 0.25, 0.25]
        config.no_version_checking = True
        config.ss58_address = bittensor.Keypair.create_from_seed(b"0" * 32).ss58_address
        config.public_key_hex = None
        config.proposal_hash = ""

        cli_instance = bittensor.cli

        # Define the response function for rich.prompt.Prompt.ask
        def ask_response(prompt: str) -> Any:
            if "delegate index" in prompt:
                return 0
            elif "wallet name" in prompt:
                return "mock"
            elif "hotkey" in prompt:
                return "mock"

        # Patch the ask response
        with patch("rich.prompt.Prompt.ask", ask_response):
            # Loop through all commands and their subcommands
            for command, command_data in bittensor.ALL_COMMANDS.items():
                config.command = command
                if isinstance(command_data, dict):
                    for subcommand in command_data["commands"].keys():
                        config.subcommand = subcommand
                        cli_instance.check_config(config)
                else:
                    config.subcommand = None
                    cli_instance.check_config(config)

    def test_new_coldkey(self, _, __):
        config = self.config()
        config.wallet.name = "new_coldkey_testwallet"

        config.command = "wallet"
        config.subcommand = "new_coldkey"
        config.amount = 1
        config.dest = "no_prompt"
        config.model = "core_server"
        config.n_words = 12
        config.use_password = False
        config.no_prompt = True
        config.overwrite_coldkey = True

        cli = bittensor.cli(config)
        cli.run()

    def test_new_hotkey(self, _, __):
        config = self.config()
        config.wallet.name = "new_hotkey_testwallet"
        config.command = "wallet"
        config.subcommand = "new_hotkey"
        config.amount = 1
        config.dest = "no_prompt"
        config.model = "core_server"
        config.n_words = 12
        config.use_password = False
        config.no_prompt = True
        config.overwrite_hotkey = True

        cli = bittensor.cli(config)
        cli.run()

    def test_regen_coldkey(self, _, __):
        config = self.config()
        config.wallet.name = "regen_coldkey_testwallet"
        config.command = "wallet"
        config.subcommand = "regen_coldkey"
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

    def test_regen_coldkeypub(self, _, __):
        config = self.config()
        config.wallet.name = "regen_coldkeypub_testwallet"
        config.command = "wallet"
        config.subcommand = "regen_coldkeypub"
        config.ss58_address = "5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
        config.public_key = None
        config.use_password = False
        config.no_prompt = True
        config.overwrite_coldkeypub = True

        cli = bittensor.cli(config)
        cli.run()

    def test_regen_hotkey(self, _, __):
        config = self.config()
        config.wallet.name = "regen_hotkey_testwallet"
        config.command = "wallet"
        config.subcommand = "regen_hotkey"
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

    def test_list(self, _, __):
        # Mock IO for wallet
        with patch(
            "bittensor.wallet",
            side_effect=[
                MagicMock(
                    coldkeypub_file=MagicMock(
                        exists_on_device=MagicMock(return_value=True),  # Wallet exists
                        is_encrypted=MagicMock(
                            return_value=False  # Wallet is not encrypted
                        ),
                    ),
                    coldkeypub=MagicMock(
                        ss58_address=bittensor.Keypair.create_from_mnemonic(
                            bittensor.Keypair.generate_mnemonic()
                        ).ss58_address
                    ),
                ),
                MagicMock(
                    hotkey_file=MagicMock(
                        exists_on_device=MagicMock(return_value=True),  # Wallet exists
                        is_encrypted=MagicMock(
                            return_value=False  # Wallet is not encrypted
                        ),
                    ),
                    hotkey=MagicMock(
                        ss58_address=bittensor.Keypair.create_from_mnemonic(
                            bittensor.Keypair.generate_mnemonic()
                        ).ss58_address
                    ),
                ),
            ],
        ):
            config = self.config()
            config.wallet.path = "tmp/walletpath"
            config.wallet.name = "mock_wallet"
            config.no_prompt = True
            config.command = "wallet"
            config.subcommand = "list"

            cli = bittensor.cli(config)
            with patch(
                "os.walk",
                side_effect=[
                    iter([("/tmp/walletpath", ["mock_wallet"], [])]),  # 1 wallet dir
                    iter(
                        [
                            ("/tmp/walletpath/mock_wallet/hotkeys", [], ["hk0"])
                        ]  # 1 hotkey file
                    ),
                ],
            ):
                cli.run()

    def test_list_no_wallet(self, _, __):
        with patch(
            "bittensor.wallet",
            side_effect=[
                MagicMock(
                    coldkeypub_file=MagicMock(
                        exists_on_device=MagicMock(return_value=True)
                    )
                )
            ],
        ):
            config = self.config()
            config.wallet.path = "/tmp/test_cli_test_list_no_wallet"
            config.no_prompt = True
            config.command = "wallet"
            config.subcommand = "list"

            cli = bittensor.cli(config)
            # This shouldn't raise an error anymore
            cli.run()

    def test_btcli_help(self, _, __):
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            with patch(
                "argparse.ArgumentParser._print_message", return_value=None
            ) as mock_print_message:
                args = ["--help"]
                bittensor.cli(args=args).run()

        mock_print_message.assert_called_once()

        call_args = mock_print_message.call_args
        help_out = call_args[0][0]

        # Extract commands from the help text.
        commands_section = re.search(
            r"positional arguments:.*?{(.+?)}", help_out, re.DOTALL
        ).group(1)
        extracted_commands = [cmd.strip() for cmd in commands_section.split(",")]

        # Get expected commands
        parser = bittensor.cli.__create_parser__()
        expected_commands = [command for command in parser._actions[-1].choices]

        # Validate each expected command is in extracted commands
        for command in expected_commands:
            assert (
                command in extracted_commands
            ), f"Command {command} not found in help output"

        # Check for duplicates
        assert len(extracted_commands) == len(
            set(extracted_commands)
        ), "Duplicate commands found in help output"

    @patch("torch.cuda.is_available", return_value=True)
    def test_register_cuda_use_cuda_flag(self, _, __, patched_sub):
        base_args = [
            "subnets",
            "pow_register",
            "--wallet.path",
            "tmp/walletpath",
            "--wallet.name",
            "mock",
            "--wallet.hotkey",
            "hk0",
            "--no_prompt",
            "--cuda.dev_id",
            "0",
        ]

        patched_sub.return_value = MagicMock(
            get_subnets=MagicMock(return_value=[1]),
            subnet_exists=MagicMock(return_value=True),
            register=MagicMock(side_effect=MockException),
        )

        # Should be able to set true without argument
        args = base_args + [
            "--pow_register.cuda.use_cuda",  # should be True without any arugment
        ]
        with pytest.raises(MockException):
            cli = bittensor.cli(args=args)
            cli.run()

        self.assertEqual(cli.config.pow_register.cuda.get("use_cuda"), True)

        # Should be able to set to false with no argument

        args = base_args + [
            "--pow_register.cuda.no_cuda",
        ]
        with pytest.raises(MockException):
            cli = bittensor.cli(args=args)
            cli.run()

        self.assertEqual(cli.config.pow_register.cuda.get("use_cuda"), False)


def return_mock_sub_2(*args, **kwargs):
    return MagicMock(
        return_value=MagicMock(
            get_subnet_burn_cost=MagicMock(return_value=0.1),
            get_subnets=MagicMock(return_value=[1]),  # Need to pass check config
            get_delegates=MagicMock(
                return_value=[
                    bittensor.DelegateInfo(
                        hotkey_ss58="",
                        total_stake=Balance.from_rao(0),
                        nominators=[],
                        owner_ss58="",
                        take=0.18,
                        validator_permits=[],
                        registrations=[],
                        return_per_1000=Balance(0.0),
                        total_daily_return=Balance(0.0),
                    )
                ]
            ),
            block=10_000,
        ),
        add_args=bittensor.subtensor.add_args,
    )


@patch("bittensor.wallet", new_callable=return_mock_wallet_factory)
@patch("bittensor.subtensor", new_callable=return_mock_sub_2)
class TestEmptyArgs(unittest.TestCase):
    """
    Test that the CLI doesn't crash when no args are passed
    """

    @patch("rich.prompt.PromptBase.ask", side_effect=MockException)
    def test_command_no_args(self, _, __, patched_prompt_ask):
        # Get argparser
        parser = bittensor.cli.__create_parser__()
        # Get all commands from argparser
        commands = [
            command
            for command in parser._actions[-1].choices  # extract correct subparser keys
            if len(command) > 1  # Skip singleton aliases
            and command
            not in [
                "subnet",
                "sudos",
                "stakes",
                "roots",
                "wallets",
                "weight",
                "st",
                "wt",
                "su",
            ]  # Skip duplicate aliases
        ]
        # Test that each command and its subcommands can be run with no args
        for command in commands:
            command_data = bittensor.ALL_COMMANDS.get(command)

            # If command is dictionary, it means it has subcommands
            if isinstance(command_data, dict):
                for subcommand in command_data["commands"].keys():
                    try:
                        # Run each subcommand
                        bittensor.cli(args=[command, subcommand]).run()
                    except MockException:
                        pass  # Expected exception
            else:
                try:
                    # If no subcommands, just run the command
                    bittensor.cli(args=[command]).run()
                except MockException:
                    pass  # Expected exception

            # Should not raise any other exceptions


mock_delegate_info = {
    "hotkey_ss58": "",
    "total_stake": bittensor.Balance.from_rao(0),
    "nominators": [],
    "owner_ss58": "",
    "take": 0.18,
    "validator_permits": [],
    "registrations": [],
    "return_per_1000": bittensor.Balance.from_rao(0),
    "total_daily_return": bittensor.Balance.from_rao(0),
}


def return_mock_sub_3(*args, **kwargs):
    return MagicMock(
        return_value=MagicMock(
            get_subnets=MagicMock(return_value=[1]),  # Mock subnet 1 ONLY.
            block=10_000,
            get_delegates=MagicMock(
                return_value=[bittensor.DelegateInfo(**mock_delegate_info)]
            ),
        ),
        block=10_000,
    )


@patch("bittensor.subtensor", new_callable=return_mock_sub_3)
class TestCLIDefaultsNoNetwork(unittest.TestCase):
    def test_inspect_prompt_wallet_name(self, _):
        # Patch command to exit early
        with patch("bittensor.commands.inspect.InspectCommand.run", return_value=None):
            # Test prompt happens when no wallet name is passed
            with patch("rich.prompt.Prompt.ask") as mock_ask_prompt:
                cli = bittensor.cli(
                    args=[
                        "wallet",
                        "inspect",
                        # '--wallet.name', 'mock',
                    ]
                )
                cli.run()

                # Prompt happened
                mock_ask_prompt.assert_called_once()

            # Test NO prompt happens when wallet name is passed
            with patch("rich.prompt.Prompt.ask") as mock_ask_prompt:
                cli = bittensor.cli(
                    args=[
                        "wallet",
                        "inspect",
                        "--wallet.name",
                        "coolwalletname",
                    ]
                )
                cli.run()

                # NO prompt happened
                mock_ask_prompt.assert_not_called()

            # Test NO prompt happens when wallet name 'default' is passed
            with patch("rich.prompt.Prompt.ask") as mock_ask_prompt:
                cli = bittensor.cli(
                    args=[
                        "wallet",
                        "inspect",
                        "--wallet.name",
                        "default",
                    ]
                )
                cli.run()

                # NO prompt happened
                mock_ask_prompt.assert_not_called()

    def test_overview_prompt_wallet_name(self, _):
        # Patch command to exit early
        with patch(
            "bittensor.commands.overview.OverviewCommand.run", return_value=None
        ):
            # Test prompt happens when no wallet name is passed
            with patch("rich.prompt.Prompt.ask") as mock_ask_prompt:
                cli = bittensor.cli(
                    args=[
                        "wallet",
                        "overview",
                        # '--wallet.name', 'mock',
                        "--netuid",
                        "1",
                    ]
                )
                cli.run()

                # Prompt happened
                mock_ask_prompt.assert_called_once()

            # Test NO prompt happens when wallet name is passed
            with patch("rich.prompt.Prompt.ask") as mock_ask_prompt:
                cli = bittensor.cli(
                    args=[
                        "wallet",
                        "overview",
                        "--wallet.name",
                        "coolwalletname",
                        "--netuid",
                        "1",
                    ]
                )
                cli.run()

                # NO prompt happened
                mock_ask_prompt.assert_not_called()

            # Test NO prompt happens when wallet name 'default' is passed
            with patch("rich.prompt.Prompt.ask") as mock_ask_prompt:
                cli = bittensor.cli(
                    args=[
                        "wallet",
                        "overview",
                        "--wallet.name",
                        "default",
                        "--netuid",
                        "1",
                    ]
                )
                cli.run()

                # NO prompt happened
                mock_ask_prompt.assert_not_called()

    def test_stake_prompt_wallet_name_and_hotkey_name(self, _):
        base_args = [
            "stake",
            "add",
            "--all",
        ]
        # Patch command to exit early
        with patch("bittensor.commands.stake.StakeCommand.run", return_value=None):
            # Test prompt happens when
            # - wallet name IS NOT passed, AND
            # - hotkey name IS NOT passed
            with patch("rich.prompt.Prompt.ask") as mock_ask_prompt:
                mock_ask_prompt.side_effect = ["mock", "mock_hotkey"]

                cli = bittensor.cli(
                    args=base_args
                    + [
                        # '--wallet.name', 'mock',
                        #'--wallet.hotkey', 'mock_hotkey',
                    ]
                )
                cli.run()

                # Prompt happened
                mock_ask_prompt.assert_called()
                self.assertEqual(
                    mock_ask_prompt.call_count,
                    2,
                    msg="Prompt should have been called twice",
                )
                args0, kwargs0 = mock_ask_prompt.call_args_list[0]
                combined_args_kwargs0 = [arg for arg in args0] + [
                    val for val in [val for val in kwargs0.values()]
                ]
                # check that prompt was called for wallet name
                self.assertTrue(
                    any(
                        filter(
                            lambda x: "wallet name" in x.lower(), combined_args_kwargs0
                        )
                    ),
                    msg=f"Prompt should have been called for wallet name: {combined_args_kwargs0}",
                )

                args1, kwargs1 = mock_ask_prompt.call_args_list[1]
                combined_args_kwargs1 = [arg for arg in args1] + [
                    val for val in kwargs1.values()
                ]
                # check that prompt was called for hotkey

                self.assertTrue(
                    any(filter(lambda x: "hotkey" in x.lower(), combined_args_kwargs1)),
                    msg=f"Prompt should have been called for hotkey: {combined_args_kwargs1}",
                )

            # Test prompt happens when
            # - wallet name IS NOT passed, AND
            # - hotkey name IS passed
            with patch("rich.prompt.Prompt.ask") as mock_ask_prompt:
                mock_ask_prompt.side_effect = ["mock", "mock_hotkey"]

                cli = bittensor.cli(
                    args=base_args
                    + [
                        #'--wallet.name', 'mock',
                        "--wallet.hotkey",
                        "mock_hotkey",
                    ]
                )
                cli.run()

                # Prompt happened
                mock_ask_prompt.assert_called()
                self.assertEqual(
                    mock_ask_prompt.call_count,
                    1,
                    msg="Prompt should have been called ONCE",
                )
                args0, kwargs0 = mock_ask_prompt.call_args_list[0]
                combined_args_kwargs0 = [arg for arg in args0] + [
                    val for val in kwargs0.values()
                ]
                # check that prompt was called for wallet name
                self.assertTrue(
                    any(
                        filter(
                            lambda x: "wallet name" in x.lower(), combined_args_kwargs0
                        )
                    ),
                    msg=f"Prompt should have been called for wallet name: {combined_args_kwargs0}",
                )

            # Test prompt happens when
            # - wallet name IS passed, AND
            # - hotkey name IS NOT passed
            with patch("rich.prompt.Prompt.ask") as mock_ask_prompt:
                mock_ask_prompt.side_effect = ["mock", "mock_hotkey"]

                cli = bittensor.cli(
                    args=base_args
                    + [
                        "--wallet.name",
                        "mock",
                        #'--wallet.hotkey', 'mock_hotkey',
                    ]
                )
                cli.run()

                # Prompt happened
                mock_ask_prompt.assert_called()
                self.assertEqual(
                    mock_ask_prompt.call_count,
                    1,
                    msg="Prompt should have been called ONCE",
                )
                args0, kwargs0 = mock_ask_prompt.call_args_list[0]
                combined_args_kwargs0 = [arg for arg in args0] + [
                    val for val in kwargs0.values()
                ]
                # check that prompt was called for hotkey
                self.assertTrue(
                    any(filter(lambda x: "hotkey" in x.lower(), combined_args_kwargs0)),
                    msg=f"Prompt should have been called for hotkey {combined_args_kwargs0}",
                )

            # Test NO prompt happens when
            # - wallet name IS passed, AND
            # - hotkey name IS passed
            with patch("rich.prompt.Prompt.ask") as mock_ask_prompt:
                cli = bittensor.cli(
                    args=base_args
                    + [
                        "--wallet.name",
                        "coolwalletname",
                        "--wallet.hotkey",
                        "coolwalletname_hotkey",
                    ]
                )
                cli.run()

                # NO prompt happened
                mock_ask_prompt.assert_not_called()

            # Test NO prompt happens when
            # - wallet name 'default' IS passed, AND
            # - hotkey name 'default' IS passed
            with patch("rich.prompt.Prompt.ask") as mock_ask_prompt:
                cli = bittensor.cli(
                    args=base_args
                    + [
                        "--wallet.name",
                        "default",
                        "--wallet.hotkey",
                        "default",
                    ]
                )
                cli.run()

                # NO prompt happened
                mock_ask_prompt.assert_not_called()

    def test_unstake_prompt_wallet_name_and_hotkey_name(self, _):
        base_args = [
            "stake",
            "remove",
            "--all",
        ]
        # Patch command to exit early
        with patch("bittensor.commands.unstake.UnStakeCommand.run", return_value=None):
            # Test prompt happens when
            # - wallet name IS NOT passed, AND
            # - hotkey name IS NOT passed
            with patch("rich.prompt.Prompt.ask") as mock_ask_prompt:
                mock_ask_prompt.side_effect = ["mock", "mock_hotkey"]

                cli = bittensor.cli(
                    args=base_args
                    + [
                        # '--wallet.name', 'mock',
                        #'--wallet.hotkey', 'mock_hotkey',
                    ]
                )
                cli.run()

                # Prompt happened
                mock_ask_prompt.assert_called()
                self.assertEqual(
                    mock_ask_prompt.call_count,
                    2,
                    msg="Prompt should have been called twice",
                )
                args0, kwargs0 = mock_ask_prompt.call_args_list[0]
                combined_args_kwargs0 = [arg for arg in args0] + [
                    val for val in kwargs0.values()
                ]
                # check that prompt was called for wallet name
                self.assertTrue(
                    any(
                        filter(
                            lambda x: "wallet name" in x.lower(), combined_args_kwargs0
                        )
                    ),
                    msg=f"Prompt should have been called for wallet name: {combined_args_kwargs0}",
                )

                args1, kwargs1 = mock_ask_prompt.call_args_list[1]
                combined_args_kwargs1 = [arg for arg in args1] + [
                    val for val in kwargs1.values()
                ]
                # check that prompt was called for hotkey
                self.assertTrue(
                    any(filter(lambda x: "hotkey" in x.lower(), combined_args_kwargs1)),
                    msg=f"Prompt should have been called for hotkey {combined_args_kwargs1}",
                )

            # Test prompt happens when
            # - wallet name IS NOT passed, AND
            # - hotkey name IS passed
            with patch("rich.prompt.Prompt.ask") as mock_ask_prompt:
                mock_ask_prompt.side_effect = ["mock", "mock_hotkey"]

                cli = bittensor.cli(
                    args=base_args
                    + [
                        #'--wallet.name', 'mock',
                        "--wallet.hotkey",
                        "mock_hotkey",
                    ]
                )
                cli.run()

                # Prompt happened
                mock_ask_prompt.assert_called()
                self.assertEqual(
                    mock_ask_prompt.call_count,
                    1,
                    msg="Prompt should have been called ONCE",
                )
                args0, kwargs0 = mock_ask_prompt.call_args_list[0]
                combined_args_kwargs0 = [arg for arg in args0] + [
                    val for val in kwargs0.values()
                ]
                # check that prompt was called for wallet name
                self.assertTrue(
                    any(
                        filter(
                            lambda x: "wallet name" in x.lower(), combined_args_kwargs0
                        )
                    ),
                    msg=f"Prompt should have been called for wallet name: {combined_args_kwargs0}",
                )

            # Test prompt happens when
            # - wallet name IS passed, AND
            # - hotkey name IS NOT passed
            with patch("rich.prompt.Prompt.ask") as mock_ask_prompt:
                mock_ask_prompt.side_effect = ["mock", "mock_hotkey"]

                cli = bittensor.cli(
                    args=base_args
                    + [
                        "--wallet.name",
                        "mock",
                        #'--wallet.hotkey', 'mock_hotkey',
                    ]
                )
                cli.run()

                # Prompt happened
                mock_ask_prompt.assert_called()
                self.assertEqual(
                    mock_ask_prompt.call_count,
                    1,
                    msg="Prompt should have been called ONCE",
                )
                args0, kwargs0 = mock_ask_prompt.call_args_list[0]
                combined_args_kwargs0 = [arg for arg in args0] + [
                    val for val in kwargs0.values()
                ]
                # check that prompt was called for hotkey
                self.assertTrue(
                    any(filter(lambda x: "hotkey" in x.lower(), combined_args_kwargs0)),
                    msg=f"Prompt should have been called for hotkey {combined_args_kwargs0}",
                )

            # Test NO prompt happens when
            # - wallet name IS passed, AND
            # - hotkey name IS passed
            with patch("rich.prompt.Prompt.ask") as mock_ask_prompt:
                cli = bittensor.cli(
                    args=base_args
                    + [
                        "--wallet.name",
                        "coolwalletname",
                        "--wallet.hotkey",
                        "coolwalletname_hotkey",
                    ]
                )
                cli.run()

                # NO prompt happened
                mock_ask_prompt.assert_not_called()

            # Test NO prompt happens when
            # - wallet name 'default' IS passed, AND
            # - hotkey name 'default' IS passed
            with patch("rich.prompt.Prompt.ask") as mock_ask_prompt:
                cli = bittensor.cli(
                    args=base_args
                    + [
                        "--wallet.name",
                        "default",
                        "--wallet.hotkey",
                        "default",
                    ]
                )
                cli.run()

                # NO prompt happened
                mock_ask_prompt.assert_not_called()

    def test_delegate_prompt_wallet_name(self, _):
        base_args = [
            "root",
            "delegate",
            "--all",
            "--delegate_ss58key",
            _get_mock_coldkey(0),
        ]
        # Patch command to exit early
        with patch(
            "bittensor.commands.delegates.DelegateStakeCommand.run", return_value=None
        ):
            # Test prompt happens when
            # - wallet name IS NOT passed
            with patch("rich.prompt.Prompt.ask") as mock_ask_prompt:
                mock_ask_prompt.side_effect = ["mock"]

                cli = bittensor.cli(
                    args=base_args
                    + [
                        # '--wallet.name', 'mock',
                    ]
                )
                cli.run()

                # Prompt happened
                mock_ask_prompt.assert_called()
                self.assertEqual(
                    mock_ask_prompt.call_count,
                    1,
                    msg="Prompt should have been called ONCE",
                )
                args0, kwargs0 = mock_ask_prompt.call_args_list[0]
                combined_args_kwargs0 = [arg for arg in args0] + [
                    val for val in kwargs0.values()
                ]
                # check that prompt was called for wallet name
                self.assertTrue(
                    any(
                        filter(
                            lambda x: "wallet name" in x.lower(), combined_args_kwargs0
                        )
                    ),
                    msg=f"Prompt should have been called for wallet name: {combined_args_kwargs0}",
                )

            # Test NO prompt happens when
            # - wallet name IS passed
            with patch("rich.prompt.Prompt.ask") as mock_ask_prompt:
                cli = bittensor.cli(
                    args=base_args
                    + [
                        "--wallet.name",
                        "coolwalletname",
                    ]
                )
                cli.run()

                # NO prompt happened
                mock_ask_prompt.assert_not_called()

    def test_undelegate_prompt_wallet_name(self, _):
        base_args = [
            "root",
            "undelegate",
            "--all",
            "--delegate_ss58key",
            _get_mock_coldkey(0),
        ]
        # Patch command to exit early
        with patch(
            "bittensor.commands.delegates.DelegateUnstakeCommand.run", return_value=None
        ):
            # Test prompt happens when
            # - wallet name IS NOT passed
            with patch("rich.prompt.Prompt.ask") as mock_ask_prompt:
                mock_ask_prompt.side_effect = ["mock"]

                cli = bittensor.cli(
                    args=base_args
                    + [
                        # '--wallet.name', 'mock',
                    ]
                )
                cli.run()

                # Prompt happened
                mock_ask_prompt.assert_called()
                self.assertEqual(
                    mock_ask_prompt.call_count,
                    1,
                    msg="Prompt should have been called ONCE",
                )
                args0, kwargs0 = mock_ask_prompt.call_args_list[0]
                combined_args_kwargs0 = [arg for arg in args0] + [
                    val for val in kwargs0.values()
                ]
                # check that prompt was called for wallet name
                self.assertTrue(
                    any(
                        filter(
                            lambda x: "wallet name" in x.lower(), combined_args_kwargs0
                        )
                    ),
                    msg=f"Prompt should have been called for wallet name: {combined_args_kwargs0}",
                )

            # Test NO prompt happens when
            # - wallet name IS passed
            with patch("rich.prompt.Prompt.ask") as mock_ask_prompt:
                cli = bittensor.cli(
                    args=base_args
                    + [
                        "--wallet.name",
                        "coolwalletname",
                    ]
                )
                cli.run()

                # NO prompt happened
                mock_ask_prompt.assert_not_called()

    def test_history_prompt_wallet_name(self, _):
        base_args = [
            "wallet",
            "history",
        ]
        # Patch command to exit early
        with patch(
            "bittensor.commands.wallets.GetWalletHistoryCommand.run", return_value=None
        ):
            # Test prompt happens when
            # - wallet name IS NOT passed
            with patch("rich.prompt.Prompt.ask") as mock_ask_prompt:
                mock_ask_prompt.side_effect = ["mock"]

                cli = bittensor.cli(
                    args=base_args
                    + [
                        # '--wallet.name', 'mock',
                    ]
                )
                cli.run()

                # Prompt happened
                mock_ask_prompt.assert_called()
                self.assertEqual(
                    mock_ask_prompt.call_count,
                    1,
                    msg="Prompt should have been called ONCE",
                )
                args0, kwargs0 = mock_ask_prompt.call_args_list[0]
                combined_args_kwargs0 = [arg for arg in args0] + [
                    val for val in kwargs0.values()
                ]
                # check that prompt was called for wallet name
                self.assertTrue(
                    any(
                        filter(
                            lambda x: "wallet name" in x.lower(), combined_args_kwargs0
                        )
                    ),
                    msg=f"Prompt should have been called for wallet name: {combined_args_kwargs0}",
                )

            # Test NO prompt happens when
            # - wallet name IS passed
            with patch("rich.prompt.Prompt.ask") as mock_ask_prompt:
                cli = bittensor.cli(
                    args=base_args
                    + [
                        "--wallet.name",
                        "coolwalletname",
                    ]
                )
                cli.run()

                # NO prompt happened
                mock_ask_prompt.assert_not_called()

    def test_delegate_prompt_hotkey(self, _):
        # Tests when
        # - wallet name IS passed, AND
        # - delegate hotkey IS NOT passed
        base_args = [
            "root",
            "delegate",
            "--all",
            "--wallet.name",
            "mock",
        ]

        delegate_ss58 = _get_mock_coldkey(0)
        with patch("bittensor.commands.delegates.show_delegates"):
            with patch(
                "bittensor.subtensor.Subtensor.get_delegates",
                return_value=[
                    bittensor.DelegateInfo(
                        hotkey_ss58=delegate_ss58,  # return delegate with mock coldkey
                        total_stake=bittensor.Balance.from_float(0.1),
                        nominators=[],
                        owner_ss58="",
                        take=0.18,
                        validator_permits=[],
                        registrations=[],
                        return_per_1000=bittensor.Balance.from_float(0.1),
                        total_daily_return=bittensor.Balance.from_float(0.1),
                    )
                ],
            ):
                # Patch command to exit early
                with patch(
                    "bittensor.commands.delegates.DelegateStakeCommand.run",
                    return_value=None,
                ):
                    # Test prompt happens when
                    # - delegate hotkey IS NOT passed
                    with patch("rich.prompt.Prompt.ask") as mock_ask_prompt:
                        mock_ask_prompt.side_effect = [
                            "0"
                        ]  # select delegate with mock coldkey

                        cli = bittensor.cli(
                            args=base_args
                            + [
                                # '--delegate_ss58key', delegate_ss58,
                            ]
                        )
                        cli.run()

                        # Prompt happened
                        mock_ask_prompt.assert_called()
                        self.assertEqual(
                            mock_ask_prompt.call_count,
                            1,
                            msg="Prompt should have been called ONCE",
                        )
                        args0, kwargs0 = mock_ask_prompt.call_args_list[0]
                        combined_args_kwargs0 = [arg for arg in args0] + [
                            val for val in kwargs0.values()
                        ]
                        # check that prompt was called for delegate hotkey
                        self.assertTrue(
                            any(
                                filter(
                                    lambda x: "delegate" in x.lower(),
                                    combined_args_kwargs0,
                                )
                            ),
                            msg=f"Prompt should have been called for delegate: {combined_args_kwargs0}",
                        )

                    # Test NO prompt happens when
                    # - delegate hotkey IS passed
                    with patch("rich.prompt.Prompt.ask") as mock_ask_prompt:
                        cli = bittensor.cli(
                            args=base_args
                            + [
                                "--delegate_ss58key",
                                delegate_ss58,
                            ]
                        )
                        cli.run()

                        # NO prompt happened
                        mock_ask_prompt.assert_not_called()

    def test_undelegate_prompt_hotkey(self, _):
        # Tests when
        # - wallet name IS passed, AND
        # - delegate hotkey IS NOT passed
        base_args = [
            "root",
            "undelegate",
            "--all",
            "--wallet.name",
            "mock",
        ]

        delegate_ss58 = _get_mock_coldkey(0)
        with patch("bittensor.commands.delegates.show_delegates"):
            with patch(
                "bittensor.subtensor.Subtensor.get_delegates",
                return_value=[
                    bittensor.DelegateInfo(
                        hotkey_ss58=delegate_ss58,  # return delegate with mock coldkey
                        total_stake=bittensor.Balance.from_float(0.1),
                        nominators=[],
                        owner_ss58="",
                        take=0.18,
                        validator_permits=[],
                        registrations=[],
                        return_per_1000=bittensor.Balance.from_float(0.1),
                        total_daily_return=bittensor.Balance.from_float(0.1),
                    )
                ],
            ):
                # Patch command to exit early
                with patch(
                    "bittensor.commands.delegates.DelegateUnstakeCommand.run",
                    return_value=None,
                ):
                    # Test prompt happens when
                    # - delegate hotkey IS NOT passed
                    with patch("rich.prompt.Prompt.ask") as mock_ask_prompt:
                        mock_ask_prompt.side_effect = [
                            "0"
                        ]  # select delegate with mock coldkey

                        cli = bittensor.cli(
                            args=base_args
                            + [
                                # '--delegate_ss58key', delegate_ss58,
                            ]
                        )
                        cli.run()

                        # Prompt happened
                        mock_ask_prompt.assert_called()
                        self.assertEqual(
                            mock_ask_prompt.call_count,
                            1,
                            msg="Prompt should have been called ONCE",
                        )
                        args0, kwargs0 = mock_ask_prompt.call_args_list[0]
                        combined_args_kwargs0 = [arg for arg in args0] + [
                            val for val in kwargs0.values()
                        ]
                        # check that prompt was called for delegate hotkey
                        self.assertTrue(
                            any(
                                filter(
                                    lambda x: "delegate" in x.lower(),
                                    combined_args_kwargs0,
                                )
                            ),
                            msg=f"Prompt should have been called for delegate: {combined_args_kwargs0}",
                        )

                    # Test NO prompt happens when
                    # - delegate hotkey IS passed
                    with patch("rich.prompt.Prompt.ask") as mock_ask_prompt:
                        cli = bittensor.cli(
                            args=base_args
                            + [
                                "--delegate_ss58key",
                                delegate_ss58,
                            ]
                        )
                        cli.run()

                        # NO prompt happened
                        mock_ask_prompt.assert_not_called()

    def test_vote_command_prompt_proposal_hash(self, _):
        """Test that the vote command prompts for proposal_hash when it is not passed"""
        base_args = [
            "root",
            "senate_vote",
            "--wallet.name",
            "mock",
            "--wallet.hotkey",
            "mock_hotkey",
        ]

        mock_proposal_hash = "mock_proposal_hash"

        with patch("bittensor.subtensor.Subtensor.is_senate_member", return_value=True):
            with patch(
                "bittensor.subtensor.Subtensor.get_vote_data",
                return_value={"index": 1},
            ):
                # Patch command to exit early
                with patch(
                    "bittensor.commands.senate.VoteCommand.run",
                    return_value=None,
                ):
                    # Test prompt happens when
                    # - proposal_hash IS NOT passed
                    with patch("rich.prompt.Prompt.ask") as mock_ask_prompt:
                        mock_ask_prompt.side_effect = [
                            mock_proposal_hash  # Proposal hash
                        ]

                        cli = bittensor.cli(
                            args=base_args
                            # proposal_hash not added
                        )
                        cli.run()

                        # Prompt happened
                        mock_ask_prompt.assert_called()
                        self.assertEqual(
                            mock_ask_prompt.call_count,
                            1,
                            msg="Prompt should have been called once",
                        )
                        args0, kwargs0 = mock_ask_prompt.call_args_list[0]
                        combined_args_kwargs0 = [arg for arg in args0] + [
                            val for val in kwargs0.values()
                        ]
                        # check that prompt was called for proposal_hash
                        self.assertTrue(
                            any(
                                filter(
                                    lambda x: "proposal" in x.lower(),
                                    combined_args_kwargs0,
                                )
                            ),
                            msg=f"Prompt should have been called for proposal: {combined_args_kwargs0}",
                        )

                    # Test NO prompt happens when
                    # - proposal_hash IS passed
                    with patch("rich.prompt.Prompt.ask") as mock_ask_prompt:
                        cli = bittensor.cli(
                            args=base_args
                            + [
                                "--proposal_hash",
                                mock_proposal_hash,
                            ]
                        )
                        cli.run()

                        # NO prompt happened
                        mock_ask_prompt.assert_not_called()

    @patch("bittensor.wallet", new_callable=return_mock_wallet_factory)
    def test_commit_reveal_weights_enabled_parse_boolean_argument(self, mock_sub, __):
        param = "commit_reveal_weights_enabled"

        def _test_value_parsing(parsed_value: bool, modified: str):
            cli = bittensor.cli(
                args=[
                    "sudo",
                    "set",
                    "--netuid",
                    "1",
                    "--param",
                    param,
                    "--value",
                    modified,
                    "--wallet.name",
                    "mock",
                ]
            )
            cli.run()

            _, kwargs = mock_sub.call_args
            passed_config = kwargs["config"]
            self.assertEqual(passed_config.param, param, msg="Incorrect param")
            self.assertEqual(
                passed_config.value,
                parsed_value,
                msg=f"Boolean argument not correctly for {modified}",
            )

        for boolean_value in [True, False, 1, 0]:
            as_str = str(boolean_value)

            _test_value_parsing(boolean_value, as_str)
            _test_value_parsing(boolean_value, as_str.capitalize())
            _test_value_parsing(boolean_value, as_str.upper())
            _test_value_parsing(boolean_value, as_str.lower())

    @patch("bittensor.wallet", new_callable=return_mock_wallet_factory)
    def test_hyperparameter_allowed_values(
        self,
        mock_sub,
        __,
    ):
        params = ["alpha_values"]

        def _test_value_parsing(param: str, value: str):
            cli = bittensor.cli(
                args=[
                    "sudo",
                    "set",
                    "--netuid",
                    "1",
                    "--param",
                    param,
                    "--value",
                    value,
                    "--wallet.name",
                    "mock",
                ]
            )
            should_raise_error = False
            error_message = ""

            try:
                alpha_low_str, alpha_high_str = value.strip("[]").split(",")
                alpha_high = float(alpha_high_str)
                alpha_low = float(alpha_low_str)
                if alpha_high <= 52428 or alpha_high >= 65535:
                    should_raise_error = True
                    error_message = "between 52428 and 65535"
                elif alpha_low < 0 or alpha_low > 52428:
                    should_raise_error = True
                    error_message = "between 0 and 52428"
            except ValueError:
                should_raise_error = True
                error_message = "a number or a boolean"
            except TypeError:
                should_raise_error = True
                error_message = "a number or a boolean"

            if isinstance(value, bool):
                should_raise_error = True
                error_message = "a number or a boolean"

            if should_raise_error:
                with pytest.raises(ValueError) as exc_info:
                    cli.run()
                assert (
                    f"Hyperparameter {param} value is not within bounds. Value is {value} but must be {error_message}"
                    in str(exc_info.value)
                )
            else:
                cli.run()
                _, kwargs = mock_sub.call_args
                passed_config = kwargs["config"]
                self.assertEqual(passed_config.param, param, msg="Incorrect param")
                self.assertEqual(
                    passed_config.value,
                    value,
                    msg=f"Value argument not set correctly for {param}",
                )

        for param in params:
            for value in [
                [0.8, 11],
                [52429, 52428],
                [52427, 53083],
                [6553, 53083],
                [-123, None],
                [1, 0],
                [True, "Some string"],
            ]:
                as_str = str(value).strip("[]")
                _test_value_parsing(param, as_str)

    @patch("bittensor.wallet", new_callable=return_mock_wallet_factory)
    def test_network_registration_allowed_parse_boolean_argument(self, mock_sub, __):
        param = "network_registration_allowed"

        def _test_value_parsing(parsed_value: bool, modified: str):
            cli = bittensor.cli(
                args=[
                    "sudo",
                    "set",
                    "--netuid",
                    "1",
                    "--param",
                    param,
                    "--value",
                    modified,
                    "--wallet.name",
                    "mock",
                ]
            )
            cli.run()

            _, kwargs = mock_sub.call_args
            passed_config = kwargs["config"]
            self.assertEqual(passed_config.param, param, msg="Incorrect param")
            self.assertEqual(
                passed_config.value,
                parsed_value,
                msg=f"Boolean argument not correctly for {modified}",
            )

        for boolean_value in [True, False, 1, 0]:
            as_str = str(boolean_value)

            _test_value_parsing(boolean_value, as_str)
            _test_value_parsing(boolean_value, as_str.capitalize())
            _test_value_parsing(boolean_value, as_str.upper())
            _test_value_parsing(boolean_value, as_str.lower())

    @patch("bittensor.wallet", new_callable=return_mock_wallet_factory)
    def test_network_pow_registration_allowed_parse_boolean_argument(
        self, mock_sub, __
    ):
        param = "network_pow_registration_allowed"

        def _test_value_parsing(parsed_value: bool, modified: str):
            cli = bittensor.cli(
                args=[
                    "sudo",
                    "set",
                    "--netuid",
                    "1",
                    "--param",
                    param,
                    "--value",
                    modified,
                    "--wallet.name",
                    "mock",
                ]
            )
            cli.run()

            _, kwargs = mock_sub.call_args
            passed_config = kwargs["config"]
            self.assertEqual(passed_config.param, param, msg="Incorrect param")
            self.assertEqual(
                passed_config.value,
                parsed_value,
                msg=f"Boolean argument not correctly for {modified}",
            )

        for boolean_value in [True, False, 1, 0]:
            as_str = str(boolean_value)

            _test_value_parsing(boolean_value, as_str)
            _test_value_parsing(boolean_value, as_str.capitalize())
            _test_value_parsing(boolean_value, as_str.upper())
            _test_value_parsing(boolean_value, as_str.lower())


if __name__ == "__main__":
    unittest.main()
