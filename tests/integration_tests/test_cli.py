# The MIT License (MIT)
# Copyright © 2022 Yuma Rao
# Copyright © 2022-2023 Opentensor Foundation
# Copyright © 2023 Opentensor Technologies Inc

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
from copy import deepcopy
from types import SimpleNamespace
from typing import Dict
from unittest.mock import MagicMock, patch

import random

import pytest
from substrateinterface.base import Keypair

import bittensor
from bittensor.mock import MockSubtensor
from bittensor import Balance
from tests.helpers import (
    MockConsole,
    _get_mock_keypair,
    _get_mock_wallet as generate_wallet,
)


_subtensor_mock: MockSubtensor = MockSubtensor()


def setUpModule():
    _subtensor_mock.reset()

    _subtensor_mock.create_subnet(netuid=1)

    _subtensor_mock.create_subnet(netuid=2)

    _subtensor_mock.create_subnet(netuid=3)

    # Set diff 0
    _subtensor_mock.set_difficulty(netuid=1, difficulty=0)

    _subtensor_mock.set_difficulty(netuid=2, difficulty=0)

    _subtensor_mock.set_difficulty(netuid=3, difficulty=0)


def return_mock_sub(*args, **kwargs):
    return MockSubtensor


@patch("bittensor.subtensor", new_callable=return_mock_sub)
class TestCLIWithNetworkAndConfig(unittest.TestCase):
    def setUp(self):
        self._config = TestCLIWithNetworkAndConfig.construct_config()

    @property
    def config(self):
        copy_ = deepcopy(self._config)
        return copy_

    @staticmethod
    def construct_config():
        parser = bittensor.cli.__create_parser__()
        defaults = bittensor.config(parser=parser, args=[])
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
        # Always use mock subtensor.
        defaults.subtensor.network = "finney"
        # Skip version checking.
        defaults.no_version_checking = True

        return defaults

    def test_overview(self, _):
        config = self.config
        config.wallet.path = "/tmp/test_cli_test_overview"
        config.wallet.name = "mock_wallet"
        config.command = "wallet"
        config.subcommand = "overview"
        config.no_prompt = True
        config.all = False
        config.netuid = []  # Don't set, so it tries all networks.

        cli = bittensor.cli(config)

        mock_hotkeys = ["hk0", "hk1", "hk2", "hk3", "hk4"]

        mock_coldkey_kp = _get_mock_keypair(0, self.id())

        mock_wallets = [
            SimpleNamespace(
                name=config.wallet.name,
                coldkey=mock_coldkey_kp,
                coldkeypub=mock_coldkey_kp,
                hotkey_str=hk,
                hotkey=_get_mock_keypair(idx + 100, self.id()),
                coldkeypub_file=MagicMock(
                    exists_on_device=MagicMock(return_value=True)  # Wallet exists
                ),
            )
            for idx, hk in enumerate(mock_hotkeys)
        ]

        mock_registrations = [
            (1, mock_wallets[0]),
            (1, mock_wallets[1]),
            # (1, mock_wallets[2]), Not registered on netuid 1
            (2, mock_wallets[0]),
            # (2, mock_wallets[1]), Not registered on netuid 2
            (2, mock_wallets[2]),
            (3, mock_wallets[0]),
            (3, mock_wallets[1]),
            (3, mock_wallets[2]),  # All registered on netuid 3 (but hk3)
            (3, mock_wallets[4]),  # hk4 is only on netuid 3
        ]  # hk3 is not registered on any network

        # Register each wallet to it's subnet.
        print("Registering wallets to mock subtensor...")

        for netuid, wallet in mock_registrations:
            _ = _subtensor_mock.force_register_neuron(
                netuid=netuid,
                coldkey=wallet.coldkey.ss58_address,
                hotkey=wallet.hotkey.ss58_address,
            )

        def mock_get_wallet(*args, **kwargs):
            hk = kwargs.get("hotkey")
            name_ = kwargs.get("name")

            if not hk and kwargs.get("config"):
                hk = kwargs.get("config").wallet.hotkey
            if not name_ and kwargs.get("config"):
                name_ = kwargs.get("config").wallet.name

            for wallet in mock_wallets:
                if wallet.name == name_ and wallet.hotkey_str == hk:
                    return wallet
            else:
                for wallet in mock_wallets:
                    if wallet.name == name_:
                        return wallet
                else:
                    return mock_wallets[0]

        mock_console = MockConsole()
        with patch(
            "bittensor.commands.overview.get_hotkey_wallets_for_wallet"
        ) as mock_get_all_wallets:
            mock_get_all_wallets.return_value = mock_wallets
            with patch("bittensor.wallet") as mock_create_wallet:
                mock_create_wallet.side_effect = mock_get_wallet
                with patch("bittensor.__console__", mock_console):
                    cli.run()

                    # Check that the overview was printed.
                    self.assertIsNotNone(mock_console.captured_print)

                    output_no_syntax = mock_console.remove_rich_syntax(
                        mock_console.captured_print
                    )

                    # Check that each subnet was printed.
                    self.assertIn("Subnet: 1", output_no_syntax)
                    self.assertIn("Subnet: 2", output_no_syntax)
                    self.assertIn("Subnet: 3", output_no_syntax)

                    # Check that only registered hotkeys are printed once for each subnet.
                    for wallet in mock_wallets:
                        expected = [
                            wallet.hotkey_str for _, wallet in mock_registrations
                        ].count(wallet.hotkey_str)
                        occurrences = output_no_syntax.count(wallet.hotkey_str)
                        self.assertEqual(occurrences, expected)

                    # Check that unregistered hotkeys are not printed.
                    for wallet in mock_wallets:
                        if wallet not in [w for _, w in mock_registrations]:
                            self.assertNotIn(wallet.hotkey_str, output_no_syntax)

    def test_overview_not_in_first_subnet(self, _):
        config = self.config
        config.wallet.path = "/tmp/test_cli_test_overview"
        config.wallet.name = "mock_wallet"
        config.command = "wallet"
        config.subcommand = "overview"
        config.no_prompt = True
        config.all = False
        config.netuid = []  # Don't set, so it tries all networks.

        cli = bittensor.cli(config)

        mock_hotkeys = ["hk0", "hk1", "hk2", "hk3", "hk4"]

        mock_coldkey_kp = _get_mock_keypair(0, self.id())

        mock_wallets = [
            SimpleNamespace(
                name=config.wallet.name,
                coldkey=mock_coldkey_kp,
                coldkeypub=mock_coldkey_kp,
                hotkey_str=hk,
                hotkey=_get_mock_keypair(idx + 100, self.id()),
                coldkeypub_file=MagicMock(
                    exists_on_device=MagicMock(return_value=True)  # Wallet exists
                ),
            )
            for idx, hk in enumerate(mock_hotkeys)
        ]

        mock_registrations = [
            # No registrations in subnet 1 or 2
            (3, mock_wallets[4])  # hk4 is on netuid 3
        ]

        # Register each wallet to it's subnet
        print("Registering mock wallets to subnets...")

        for netuid, wallet in mock_registrations:
            print(
                "Registering wallet {} to subnet {}".format(wallet.hotkey_str, netuid)
            )
            _ = _subtensor_mock.force_register_neuron(
                netuid=netuid,
                coldkey=wallet.coldkey.ss58_address,
                hotkey=wallet.hotkey.ss58_address,
            )

        def mock_get_wallet(*args, **kwargs):
            hk = kwargs.get("hotkey")
            name_ = kwargs.get("name")

            if not hk and kwargs.get("config"):
                hk = kwargs.get("config").wallet.hotkey
            if not name_ and kwargs.get("config"):
                name_ = kwargs.get("config").wallet.name

            for wallet in mock_wallets:
                if wallet.name == name_ and wallet.hotkey_str == hk:
                    return wallet
            else:
                for wallet in mock_wallets:
                    if wallet.name == name_:
                        return wallet
                else:
                    return mock_wallets[0]

        mock_console = MockConsole()
        with patch(
            "bittensor.commands.overview.get_hotkey_wallets_for_wallet"
        ) as mock_get_all_wallets:
            mock_get_all_wallets.return_value = mock_wallets
            with patch("bittensor.wallet") as mock_create_wallet:
                mock_create_wallet.side_effect = mock_get_wallet
                with patch("bittensor.__console__", mock_console):
                    cli.run()

                    # Check that the overview was printed.
                    self.assertIsNotNone(mock_console.captured_print)

                    output_no_syntax = mock_console.remove_rich_syntax(
                        mock_console.captured_print
                    )

                    # Check that each subnet was printed except subnet 1 and 2.
                    # Subnet 1 and 2 are not printed because no wallet is registered to them.
                    self.assertNotIn("Subnet: 1", output_no_syntax)
                    self.assertNotIn("Subnet: 2", output_no_syntax)
                    self.assertIn("Subnet: 3", output_no_syntax)

                    # Check that only registered hotkeys are printed once for each subnet.
                    for wallet in mock_wallets:
                        expected = [
                            wallet.hotkey_str for _, wallet in mock_registrations
                        ].count(wallet.hotkey_str)
                        occurrences = output_no_syntax.count(wallet.hotkey_str)
                        self.assertEqual(occurrences, expected)

                    # Check that unregistered hotkeys are not printed.
                    for wallet in mock_wallets:
                        if wallet not in [w for _, w in mock_registrations]:
                            self.assertNotIn(wallet.hotkey_str, output_no_syntax)

    def test_overview_with_hotkeys_config(self, _):
        config = self.config
        config.command = "wallet"
        config.subcommand = "overview"
        config.no_prompt = True
        config.hotkeys = ["some_hotkey"]
        config.all = False
        config.netuid = []  # Don't set, so it tries all networks.

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_without_hotkeys_config(self, _):
        config = self.config
        config.command = "wallet"
        config.subcommand = "overview"
        config.no_prompt = True
        config.all = False
        config.netuid = []  # Don't set, so it tries all networks.

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_with_sort_by_config(self, _):
        config = self.config
        config.command = "wallet"
        config.subcommand = "overview"
        config.no_prompt = True
        config.wallet.sort_by = "rank"
        config.all = False
        config.netuid = []  # Don't set, so it tries all networks.

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_with_sort_by_bad_column_name(self, _):
        config = self.config
        config.command = "wallet"
        config.subcommand = "overview"
        config.no_prompt = True
        config.wallet.sort_by = "totallynotmatchingcolumnname"
        config.all = False
        config.netuid = []  # Don't set, so it tries all networks.

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_without_sort_by_config(self, _):
        config = self.config
        config.command = "wallet"
        config.subcommand = "overview"
        config.no_prompt = True
        config.all = False
        config.netuid = []  # Don't set, so it tries all networks.

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_with_sort_order_config(self, _):
        config = self.config
        config.command = "wallet"
        config.subcommand = "overview"
        config.wallet.sort_order = "desc"  # Set descending sort order
        config.no_prompt = True
        config.all = False
        config.netuid = []  # Don't set, so it tries all networks.

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_with_sort_order_config_bad_sort_type(self, _):
        config = self.config
        config.command = "wallet"
        config.subcommand = "overview"
        config.wallet.sort_order = "nowaythisshouldmatchanyorderingchoice"
        config.no_prompt = True
        config.all = False
        config.netuid = []  # Don't set, so it tries all networks.

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_without_sort_order_config(self, _):
        config = self.config
        config.command = "wallet"
        config.subcommand = "overview"
        # Don't specify sort_order in config
        config.no_prompt = True
        config.all = False
        config.netuid = []  # Don't set, so it tries all networks.

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_with_width_config(self, _):
        config = self.config
        config.command = "wallet"
        config.subcommand = "overview"
        config.width = 100
        config.no_prompt = True
        config.all = False
        config.netuid = []  # Don't set, so it tries all networks.

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_without_width_config(self, _):
        config = self.config
        config.command = "wallet"
        config.subcommand = "overview"
        # Don't specify width in config
        config.no_prompt = True
        config.all = False
        config.netuid = []  # Don't set, so it tries all networks.

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_all(self, _):
        config = self.config
        config.command = "wallet"
        config.subcommand = "overview"
        config.no_prompt = True
        config.netuid = []  # Don't set, so it tries all networks.

        config.all = True
        cli = bittensor.cli(config)
        cli.run()

    def test_unstake_with_specific_hotkeys(self, _):
        config = self.config
        config.command = "stake"
        config.subcommand = "remove"
        config.no_prompt = True
        config.amount = 5.0
        config.wallet.name = "fake_wallet"
        config.hotkeys = ["hk0", "hk1", "hk2"]
        config.all_hotkeys = False
        # Notice no max_stake specified

        mock_stakes: Dict[str, Balance] = {
            # All have more than 5.0 stake
            "hk0": Balance.from_float(10.0),
            "hk1": Balance.from_float(11.1),
            "hk2": Balance.from_float(12.2),
        }

        mock_coldkey_kp = _get_mock_keypair(0, self.id())

        mock_wallets = [
            SimpleNamespace(
                name=config.wallet.name,
                coldkey=mock_coldkey_kp,
                coldkeypub=mock_coldkey_kp,
                hotkey_str=hk,
                hotkey=_get_mock_keypair(idx + 100, self.id()),
            )
            for idx, hk in enumerate(config.hotkeys)
        ]

        # Register mock wallets and give them stakes

        for wallet in mock_wallets:
            _ = _subtensor_mock.force_register_neuron(
                netuid=1,
                hotkey=wallet.hotkey.ss58_address,
                coldkey=wallet.coldkey.ss58_address,
                stake=mock_stakes[wallet.hotkey_str].rao,
            )

        cli = bittensor.cli(config)

        def mock_get_wallet(*args, **kwargs):
            if kwargs.get("hotkey"):
                for wallet in mock_wallets:
                    if wallet.hotkey_str == kwargs.get("hotkey"):
                        return wallet
            else:
                return mock_wallets[0]

        with patch("bittensor.wallet") as mock_create_wallet:
            mock_create_wallet.side_effect = mock_get_wallet

            # Check stakes before unstaking
            for wallet in mock_wallets:
                stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                    hotkey_ss58=wallet.hotkey.ss58_address,
                    coldkey_ss58=wallet.coldkey.ss58_address,
                )
                self.assertEqual(stake.rao, mock_stakes[wallet.hotkey_str].rao)

            cli.run()

            # Check stakes after unstaking
            for wallet in mock_wallets:
                stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                    hotkey_ss58=wallet.hotkey.ss58_address,
                    coldkey_ss58=wallet.coldkey.ss58_address,
                )
                self.assertAlmostEqual(
                    stake.tao,
                    mock_stakes[wallet.hotkey_str].tao - config.amount,
                    places=4,
                )

    def test_unstake_with_all_hotkeys(self, _):
        config = self.config
        config.command = "stake"
        config.subcommand = "remove"
        config.no_prompt = True
        config.amount = 5.0
        config.wallet.name = "fake_wallet"
        # Notice wallet.hotkeys not specified
        config.all_hotkeys = True
        # Notice no max_stake specified

        mock_stakes: Dict[str, Balance] = {
            # All have more than 5.0 stake
            "hk0": Balance.from_float(10.0),
            "hk1": Balance.from_float(11.1),
            "hk2": Balance.from_float(12.2),
        }

        mock_coldkey_kp = _get_mock_keypair(0, self.id())

        mock_wallets = [
            SimpleNamespace(
                name=config.wallet.name,
                coldkey=mock_coldkey_kp,
                coldkeypub=mock_coldkey_kp,
                hotkey_str=hk,
                hotkey=_get_mock_keypair(idx + 100, self.id()),
            )
            for idx, hk in enumerate(list(mock_stakes.keys()))
        ]

        # Register mock wallets and give them stakes

        for wallet in mock_wallets:
            _ = _subtensor_mock.force_register_neuron(
                netuid=1,
                hotkey=wallet.hotkey.ss58_address,
                coldkey=wallet.coldkey.ss58_address,
                stake=mock_stakes[wallet.hotkey_str].rao,
            )

        cli = bittensor.cli(config)

        def mock_get_wallet(*args, **kwargs):
            if kwargs.get("hotkey"):
                for wallet in mock_wallets:
                    if wallet.hotkey_str == kwargs.get("hotkey"):
                        return wallet
            else:
                return mock_wallets[0]

        with patch(
            "bittensor.commands.unstake.get_hotkey_wallets_for_wallet"
        ) as mock_get_all_wallets:
            mock_get_all_wallets.return_value = mock_wallets
            with patch("bittensor.wallet") as mock_create_wallet:
                mock_create_wallet.side_effect = mock_get_wallet

                # Check stakes before unstaking
                for wallet in mock_wallets:
                    stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                        hotkey_ss58=wallet.hotkey.ss58_address,
                        coldkey_ss58=wallet.coldkey.ss58_address,
                    )
                    self.assertEqual(stake.rao, mock_stakes[wallet.hotkey_str].rao)

                cli.run()

                # Check stakes after unstaking
                for wallet in mock_wallets:
                    stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                        hotkey_ss58=wallet.hotkey.ss58_address,
                        coldkey_ss58=wallet.coldkey.ss58_address,
                    )
                    self.assertAlmostEqual(
                        stake.tao,
                        mock_stakes[wallet.hotkey_str].tao - config.amount,
                        places=4,
                    )

    def test_unstake_with_exclude_hotkeys_from_all(self, _):
        config = self.config
        config.command = "stake"
        config.subcommand = "remove"
        config.no_prompt = True
        config.amount = 5.0
        config.wallet.name = "fake_wallet"
        config.hotkeys = ["hk1"]  # Exclude hk1
        config.all_hotkeys = True

        mock_stakes: Dict[str, Balance] = {
            # All have more than 5.0 stake
            "hk0": Balance.from_float(10.0),
            "hk1": Balance.from_float(11.1),
            "hk2": Balance.from_float(12.2),
        }

        mock_coldkey_kp = _get_mock_keypair(0, self.id())

        mock_wallets = [
            SimpleNamespace(
                name=config.wallet.name,
                coldkey=mock_coldkey_kp,
                coldkeypub=mock_coldkey_kp,
                hotkey_str=hk,
                hotkey=_get_mock_keypair(idx + 100, self.id()),
            )
            for idx, hk in enumerate(list(mock_stakes.keys()))
        ]

        # Register mock wallets and give them stakes

        for wallet in mock_wallets:
            _ = _subtensor_mock.force_register_neuron(
                netuid=1,
                hotkey=wallet.hotkey.ss58_address,
                coldkey=wallet.coldkey.ss58_address,
                stake=mock_stakes[wallet.hotkey_str].rao,
            )

        cli = bittensor.cli(config)

        def mock_get_wallet(*args, **kwargs):
            if kwargs.get("hotkey"):
                for wallet in mock_wallets:
                    if wallet.hotkey_str == kwargs.get("hotkey"):
                        return wallet
            else:
                return mock_wallets[0]

        with patch(
            "bittensor.commands.unstake.get_hotkey_wallets_for_wallet"
        ) as mock_get_all_wallets:
            mock_get_all_wallets.return_value = mock_wallets
            with patch("bittensor.wallet") as mock_create_wallet:
                mock_create_wallet.side_effect = mock_get_wallet

                # Check stakes before unstaking
                for wallet in mock_wallets:
                    stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                        hotkey_ss58=wallet.hotkey.ss58_address,
                        coldkey_ss58=wallet.coldkey.ss58_address,
                    )
                    self.assertEqual(stake.rao, mock_stakes[wallet.hotkey_str].rao)

                cli.run()

                # Check stakes after unstaking
                for wallet in mock_wallets:
                    stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                        hotkey_ss58=wallet.hotkey.ss58_address,
                        coldkey_ss58=wallet.coldkey.ss58_address,
                    )
                    if wallet.hotkey_str == "hk1":
                        # hk1 should not have been unstaked
                        self.assertAlmostEqual(
                            stake.tao, mock_stakes[wallet.hotkey_str].tao, places=4
                        )
                    else:
                        self.assertAlmostEqual(
                            stake.tao,
                            mock_stakes[wallet.hotkey_str].tao - config.amount,
                            places=4,
                        )

    def test_unstake_with_multiple_hotkeys_max_stake(self, _):
        config = self.config
        config.command = "stake"
        config.subcommand = "remove"
        config.no_prompt = True
        # Notie amount is not specified
        config.max_stake = 5.0  # The keys should have at most 5.0 tao staked after
        config.wallet.name = "fake_wallet"
        config.hotkeys = ["hk0", "hk1", "hk2"]
        config.all_hotkeys = False

        mock_stakes: Dict[str, Balance] = {
            # All have more than 5.0 stake
            "hk0": Balance.from_float(10.0),
            "hk1": Balance.from_float(4.9),
            "hk2": Balance.from_float(12.2),
        }

        mock_coldkey_kp = _get_mock_keypair(0, self.id())

        mock_wallets = [
            SimpleNamespace(
                name=config.wallet.name,
                coldkey=mock_coldkey_kp,
                coldkeypub=mock_coldkey_kp,
                hotkey_str=hk,
                hotkey=_get_mock_keypair(idx + 100, self.id()),
            )
            for idx, hk in enumerate(list(mock_stakes.keys()))
        ]

        # Register mock wallets and give them stakes
        print("Registering mock wallets...")

        for wallet in mock_wallets:
            print("Registering mock wallet {}".format(wallet.hotkey_str))
            _ = _subtensor_mock.force_register_neuron(
                netuid=1,
                hotkey=wallet.hotkey.ss58_address,
                coldkey=wallet.coldkey.ss58_address,
                stake=mock_stakes[wallet.hotkey_str].rao,
            )

        cli = bittensor.cli(config)

        def mock_get_wallet(*args, **kwargs):
            if kwargs.get("hotkey"):
                for wallet in mock_wallets:
                    if wallet.hotkey_str == kwargs.get("hotkey"):
                        return wallet
            else:
                return mock_wallets[0]

        with patch(
            "bittensor.commands.unstake.get_hotkey_wallets_for_wallet"
        ) as mock_get_all_wallets:
            mock_get_all_wallets.return_value = mock_wallets
            with patch("bittensor.wallet") as mock_create_wallet:
                mock_create_wallet.side_effect = mock_get_wallet

                # Check stakes before unstaking
                for wallet in mock_wallets:
                    stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                        hotkey_ss58=wallet.hotkey.ss58_address,
                        coldkey_ss58=wallet.coldkey.ss58_address,
                    )
                    self.assertEqual(stake.rao, mock_stakes[wallet.hotkey_str].rao)

                cli.run()

                # Check stakes after unstaking
                for wallet in mock_wallets:
                    stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                        hotkey_ss58=wallet.hotkey.ss58_address,
                        coldkey_ss58=wallet.coldkey.ss58_address,
                    )
                    # All should have been unstaked below or equal to max_stake
                    self.assertLessEqual(
                        stake.tao, config.max_stake + 0.0001
                    )  # Add a small buffer for fp errors

                    if wallet.hotkey_str == "hk1":
                        # hk1 should not have been unstaked because it was already below max_stake
                        self.assertAlmostEqual(
                            stake.tao, mock_stakes[wallet.hotkey_str].tao, places=4
                        )

    def test_stake_with_specific_hotkeys(self, _):
        config = self.config
        config.command = "stake"
        config.subcommand = "add"
        config.no_prompt = True
        config.amount = 5.0
        config.wallet.name = "fake_wallet"
        config.hotkeys = ["hk0", "hk1", "hk2"]
        config.all_hotkeys = False
        # Notice no max_stake specified

        mock_balance = Balance.from_float(22.2)

        mock_coldkey_kp = _get_mock_keypair(0, self.id())

        mock_wallets = [
            SimpleNamespace(
                name=config.wallet.name,
                coldkey=mock_coldkey_kp,
                coldkeypub=mock_coldkey_kp,
                hotkey_str=hk,
                hotkey=_get_mock_keypair(idx + 100, self.id()),
            )
            for idx, hk in enumerate(config.hotkeys)
        ]

        # Register mock wallets and give them balances
        print("Registering mock wallets...")

        for wallet in mock_wallets:
            print("Registering mock wallet {}".format(wallet.hotkey_str))
            _ = _subtensor_mock.force_register_neuron(
                netuid=1,
                hotkey=wallet.hotkey.ss58_address,
                coldkey=wallet.coldkey.ss58_address,
            )

        success, err = _subtensor_mock.force_set_balance(
            ss58_address=mock_coldkey_kp.ss58_address, balance=mock_balance.rao
        )

        cli = bittensor.cli(config)

        def mock_get_wallet(*args, **kwargs):
            if kwargs.get("hotkey"):
                for wallet in mock_wallets:
                    if wallet.hotkey_str == kwargs.get("hotkey"):
                        return wallet
            else:
                return mock_wallets[0]

        with patch("bittensor.wallet") as mock_create_wallet:
            mock_create_wallet.side_effect = mock_get_wallet

            # Check stakes before staking
            for wallet in mock_wallets:
                stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                    hotkey_ss58=wallet.hotkey.ss58_address,
                    coldkey_ss58=wallet.coldkey.ss58_address,
                )
                self.assertEqual(stake.rao, 0)

            cli.run()

            # Check stakes after staking
            for wallet in mock_wallets:
                stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                    hotkey_ss58=wallet.hotkey.ss58_address,
                    coldkey_ss58=wallet.coldkey.ss58_address,
                )
                self.assertAlmostEqual(stake.tao, config.amount, places=4)

    def test_stake_with_all_hotkeys(self, _):
        config = self.config
        config.command = "stake"
        config.subcommand = "add"
        config.no_prompt = True
        config.amount = 5.0
        config.wallet.name = "fake_wallet"
        # Notice wallet.hotkeys is not specified
        config.all_hotkeys = True
        # Notice no max_stake specified

        mock_hotkeys = ["hk0", "hk1", "hk2"]

        mock_balance = Balance.from_float(22.0)

        mock_coldkey_kp = _get_mock_keypair(0, self.id())

        mock_wallets = [
            SimpleNamespace(
                name=config.wallet.name,
                coldkey=mock_coldkey_kp,
                coldkeypub=mock_coldkey_kp,
                hotkey_str=hk,
                hotkey=_get_mock_keypair(idx + 100, self.id()),
            )
            for idx, hk in enumerate(mock_hotkeys)
        ]

        # Register mock wallets and give them no stake
        print("Registering mock wallets...")

        for wallet in mock_wallets:
            print("Registering mock wallet {}".format(wallet.hotkey_str))
            _ = _subtensor_mock.force_register_neuron(
                netuid=1,
                hotkey=wallet.hotkey.ss58_address,
                coldkey=wallet.coldkeypub.ss58_address,
            )

        # Set the coldkey balance
        success, err = _subtensor_mock.force_set_balance(
            ss58_address=mock_coldkey_kp.ss58_address, balance=mock_balance.rao
        )

        cli = bittensor.cli(config)

        def mock_get_wallet(*args, **kwargs):
            if kwargs.get("hotkey"):
                for wallet in mock_wallets:
                    if wallet.hotkey_str == kwargs.get("hotkey"):
                        return wallet
            else:
                return mock_wallets[0]

        with patch("bittensor.wallet") as mock_create_wallet:
            mock_create_wallet.side_effect = mock_get_wallet
            with patch(
                "bittensor.commands.stake.get_hotkey_wallets_for_wallet"
            ) as mock_get_hotkey_wallets_for_wallet:
                mock_get_hotkey_wallets_for_wallet.return_value = mock_wallets

                # Check stakes before staking
                for wallet in mock_wallets:
                    stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                        hotkey_ss58=wallet.hotkey.ss58_address,
                        coldkey_ss58=wallet.coldkey.ss58_address,
                    )
                    # Check that all stakes are 0
                    self.assertEqual(stake.rao, 0)

                # Check that the balance is correct
                balance = _subtensor_mock.get_balance(
                    address=wallet.coldkeypub.ss58_address
                )

                self.assertAlmostEqual(balance.tao, mock_balance.tao, places=4)

                cli.run()

                # Check stakes after staking
                for wallet in mock_wallets:
                    stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                        hotkey_ss58=wallet.hotkey.ss58_address,
                        coldkey_ss58=wallet.coldkey.ss58_address,
                    )
                    # Check that all stakes are 5.0
                    self.assertAlmostEqual(stake.tao, config.amount, places=4)

                # Check that the balance is correct
                balance = _subtensor_mock.get_balance(
                    address=wallet.coldkeypub.ss58_address
                )
                self.assertAlmostEqual(
                    balance.tao,
                    mock_balance.tao - (config.amount * len(mock_wallets)),
                    places=4,
                )

    def test_stake_with_exclude_hotkeys_from_all(self, _):
        config = self.config
        config.command = "stake"
        config.subcommand = "add"
        config.no_prompt = True
        config.amount = 5.0
        config.wallet.name = "fake_wallet"
        config.hotkeys = ["hk1"]  # exclude hk1
        config.all_hotkeys = True
        # Notice no max_stake specified

        mock_hotkeys = ["hk0", "hk1", "hk2"]

        mock_balance = Balance.from_float(25.0)

        mock_coldkey_kp = _get_mock_keypair(0, self.id())

        mock_wallets = [
            SimpleNamespace(
                name=config.wallet.name,
                coldkey=mock_coldkey_kp,
                coldkeypub=mock_coldkey_kp,
                hotkey_str=hk,
                hotkey=_get_mock_keypair(idx + 100, self.id()),
            )
            for idx, hk in enumerate(mock_hotkeys)
        ]

        # Register mock wallets and give them balances
        print("Registering mock wallets...")

        for wallet in mock_wallets:
            print("Registering mock wallet {}".format(wallet.hotkey_str))
            _ = _subtensor_mock.force_register_neuron(
                netuid=1,
                hotkey=wallet.hotkey.ss58_address,
                coldkey=wallet.coldkeypub.ss58_address,
            )

        # Set the coldkey balance
        _subtensor_mock.force_set_balance(
            ss58_address=mock_coldkey_kp.ss58_address, balance=mock_balance.rao
        )

        cli = bittensor.cli(config)

        def mock_get_wallet(*args, **kwargs):
            if kwargs.get("hotkey"):
                for wallet in mock_wallets:
                    if wallet.hotkey_str == kwargs.get("hotkey"):
                        return wallet
            else:
                return mock_wallets[0]

        with patch(
            "bittensor.commands.stake.get_hotkey_wallets_for_wallet"
        ) as mock_get_all_wallets:
            mock_get_all_wallets.return_value = mock_wallets
            with patch("bittensor.wallet") as mock_create_wallet:
                mock_create_wallet.side_effect = mock_get_wallet

                # Check stakes before staking
                for wallet in mock_wallets:
                    stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                        hotkey_ss58=wallet.hotkey.ss58_address,
                        coldkey_ss58=wallet.coldkey.ss58_address,
                    )
                    # Check that all stakes are 0
                    self.assertEqual(stake.rao, 0)

                # Check that the balance is correct
                balance = _subtensor_mock.get_balance(
                    address=wallet.coldkeypub.ss58_address
                )

                self.assertAlmostEqual(balance.tao, mock_balance.tao, places=4)

                cli.run()

                # Check stakes after staking
                for wallet in mock_wallets:
                    stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                        hotkey_ss58=wallet.hotkey.ss58_address,
                        coldkey_ss58=wallet.coldkey.ss58_address,
                    )

                    if wallet.hotkey_str == "hk1":
                        # Check that hk1 stake is 0
                        # We excluded it from staking
                        self.assertEqual(stake.tao, 0)
                    else:
                        # Check that all stakes are 5.0
                        self.assertAlmostEqual(stake.tao, config.amount, places=4)

                # Check that the balance is correct
                balance = _subtensor_mock.get_balance(
                    address=wallet.coldkeypub.ss58_address
                )
                self.assertAlmostEqual(
                    balance.tao, mock_balance.tao - (config.amount * 2), places=4
                )

    def test_stake_with_multiple_hotkeys_max_stake(self, _):
        config = self.config
        config.command = "stake"
        config.subcommand = "add"
        config.no_prompt = True
        # Notie amount is not specified
        config.max_stake = 15.0  # The keys should have at most 15.0 tao staked after
        config.wallet.name = "fake_wallet"
        config.hotkeys = ["hk0", "hk1", "hk2"]
        config.all_hotkeys = False

        mock_balance = Balance.from_float(config.max_stake * 3)

        mock_stakes: Dict[str, Balance] = {
            "hk0": Balance.from_float(0.0),
            "hk1": Balance.from_float(config.max_stake * 2),
            "hk2": Balance.from_float(0.0),
        }

        mock_coldkey_kp = _get_mock_keypair(0, self.id())

        mock_wallets = [
            SimpleNamespace(
                name=config.wallet.name,
                coldkey=mock_coldkey_kp,
                coldkeypub=mock_coldkey_kp,
                hotkey_str=hk,
                hotkey=_get_mock_keypair(idx + 100, self.id()),
            )
            for idx, hk in enumerate(config.hotkeys)
        ]

        # Register mock wallets and give them balances
        print("Registering mock wallets...")

        for wallet in mock_wallets:
            print("Registering mock wallet {}".format(wallet.hotkey_str))
            if wallet.hotkey_str == "hk1":
                # Set the stake for hk1
                _ = _subtensor_mock.force_register_neuron(
                    netuid=1,
                    hotkey=wallet.hotkey.ss58_address,
                    coldkey=wallet.coldkeypub.ss58_address,
                    stake=mock_stakes[wallet.hotkey_str].rao,
                )
            else:
                _ = _subtensor_mock.force_register_neuron(
                    netuid=1,
                    hotkey=wallet.hotkey.ss58_address,
                    coldkey=wallet.coldkeypub.ss58_address,
                )

        _subtensor_mock.force_set_balance(
            ss58_address=mock_coldkey_kp.ss58_address, balance=mock_balance.rao
        )

        cli = bittensor.cli(config)

        def mock_get_wallet(*args, **kwargs):
            if kwargs.get("hotkey"):
                for wallet in mock_wallets:
                    if wallet.hotkey_str == kwargs.get("hotkey"):
                        return wallet
            else:
                return mock_wallets[0]

        with patch("bittensor.wallet") as mock_create_wallet:
            mock_create_wallet.side_effect = mock_get_wallet

            # Check stakes before staking
            for wallet in mock_wallets:
                stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                    hotkey_ss58=wallet.hotkey.ss58_address,
                    coldkey_ss58=wallet.coldkey.ss58_address,
                )
                # Check that all stakes are correct
                if wallet.hotkey_str == "hk1":
                    self.assertAlmostEqual(stake.tao, config.max_stake * 2, places=4)
                else:
                    self.assertEqual(stake.rao, 0)

            # Check that the balance is correct
            balance = _subtensor_mock.get_balance(
                address=wallet.coldkeypub.ss58_address
            )

            self.assertAlmostEqual(balance.tao, mock_balance.tao, places=4)

            cli.run()

            # Check stakes after staking
            for wallet in mock_wallets:
                stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                    hotkey_ss58=wallet.hotkey.ss58_address,
                    coldkey_ss58=wallet.coldkey.ss58_address,
                )

                # Check that all stakes at least 15.0
                self.assertGreaterEqual(stake.tao + 0.1, config.max_stake)

                if wallet.hotkey_str == "hk1":
                    # Check that hk1 stake was not changed
                    # It had more than max_stake already
                    self.assertAlmostEqual(
                        stake.tao, mock_stakes[wallet.hotkey_str].tao, places=4
                    )

            # Check that the balance decreased
            balance = _subtensor_mock.get_balance(
                address=wallet.coldkeypub.ss58_address
            )
            self.assertLessEqual(balance.tao, mock_balance.tao)

    def test_stake_with_multiple_hotkeys_max_stake_not_enough_balance(self, _):
        config = self.config
        config.command = "stake"
        config.subcommand = "add"
        config.no_prompt = True
        # Notie amount is not specified
        config.max_stake = 15.0  # The keys should have at most 15.0 tao staked after
        config.wallet.name = "fake_wallet"
        config.hotkeys = ["hk0", "hk1", "hk2"]
        config.all_hotkeys = False

        mock_balance = Balance.from_float(15.0 * 2)  # Not enough for all hotkeys

        mock_coldkey_kp = _get_mock_keypair(0, self.id())

        mock_wallets = [
            SimpleNamespace(
                name=config.wallet.name,
                coldkey=mock_coldkey_kp,
                coldkeypub=mock_coldkey_kp,
                hotkey_str=hk,
                hotkey=_get_mock_keypair(idx + 100, self.id()),
            )
            for idx, hk in enumerate(config.hotkeys)
        ]

        # Register mock wallets and give them balances
        print("Registering mock wallets...")

        for wallet in mock_wallets:
            print("Registering mock wallet {}".format(wallet.hotkey_str))
            _ = _subtensor_mock.force_register_neuron(
                netuid=1,
                hotkey=wallet.hotkey.ss58_address,
                coldkey=wallet.coldkeypub.ss58_address,
            )

        _subtensor_mock.force_set_balance(
            ss58_address=mock_coldkey_kp.ss58_address, balance=mock_balance.rao
        )

        cli = bittensor.cli(config)

        def mock_get_wallet(*args, **kwargs):
            if kwargs.get("hotkey"):
                for wallet in mock_wallets:
                    if wallet.hotkey_str == kwargs.get("hotkey"):
                        return wallet
            else:
                return mock_wallets[0]

        with patch("bittensor.wallet") as mock_create_wallet:
            mock_create_wallet.side_effect = mock_get_wallet

            # Check stakes before staking
            for wallet in mock_wallets:
                stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                    hotkey_ss58=wallet.hotkey.ss58_address,
                    coldkey_ss58=wallet.coldkey.ss58_address,
                )
                # Check that all stakes are 0
                self.assertEqual(stake.rao, 0)

            # Check that the balance is correct
            balance = _subtensor_mock.get_balance(
                address=wallet.coldkeypub.ss58_address
            )

            self.assertAlmostEqual(balance.tao, mock_balance.tao, places=4)

            cli.run()

            # Check stakes after staking
            for wallet in mock_wallets:
                stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                    hotkey_ss58=wallet.hotkey.ss58_address,
                    coldkey_ss58=wallet.coldkey.ss58_address,
                )

                if wallet.hotkey_str == "hk2":
                    # Check that the stake is still 0
                    self.assertEqual(stake.tao, 0)

                else:
                    # Check that all stakes are maximum of 15.0
                    self.assertLessEqual(stake.tao, config.max_stake)

                # Check that the balance is correct
                balance = _subtensor_mock.get_balance(
                    address=wallet.coldkeypub.ss58_address
                )
                self.assertLessEqual(balance.tao, mock_balance.tao)

    def test_stake_with_single_hotkey_max_stake(self, _):
        config = self.config
        config.command = "stake"
        config.subcommand = "add"
        config.no_prompt = True
        # Notie amount is not specified
        config.max_stake = 15.0  # The keys should have at most 15.0 tao staked after
        config.wallet.name = "fake_wallet"
        config.hotkeys = ["hk0"]
        config.all_hotkeys = False

        mock_balance = Balance.from_float(15.0 * 3)

        mock_coldkey_kp = _get_mock_keypair(0, self.id())

        mock_wallets = [
            SimpleNamespace(
                name=config.wallet.name,
                coldkey=mock_coldkey_kp,
                coldkeypub=mock_coldkey_kp,
                hotkey_str=hk,
                hotkey=_get_mock_keypair(idx + 100, self.id()),
            )
            for idx, hk in enumerate(config.hotkeys)
        ]

        # Register mock wallets and give them balances
        print("Registering mock wallets...")

        for wallet in mock_wallets:
            print("Registering mock wallet {}".format(wallet.hotkey_str))
            _ = _subtensor_mock.force_register_neuron(
                netuid=1,
                hotkey=wallet.hotkey.ss58_address,
                coldkey=wallet.coldkeypub.ss58_address,
            )

        _subtensor_mock.force_set_balance(
            ss58_address=mock_coldkey_kp.ss58_address, balance=mock_balance.rao
        )

        cli = bittensor.cli(config)

        def mock_get_wallet(*args, **kwargs):
            if kwargs.get("hotkey"):
                for wallet in mock_wallets:
                    if wallet.hotkey_str == kwargs.get("hotkey"):
                        return wallet
            else:
                return mock_wallets[0]

        with patch("bittensor.wallet") as mock_create_wallet:
            mock_create_wallet.side_effect = mock_get_wallet

            # Check stakes before staking
            for wallet in mock_wallets:
                stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                    hotkey_ss58=wallet.hotkey.ss58_address,
                    coldkey_ss58=wallet.coldkey.ss58_address,
                )
                # Check that all stakes are 0
                self.assertEqual(stake.rao, 0)

            # Check that the balance is correct
            balance = _subtensor_mock.get_balance(
                address=wallet.coldkeypub.ss58_address
            )

            self.assertAlmostEqual(balance.tao, mock_balance.tao, places=4)

            cli.run()

            # Check stakes after staking
            for wallet in mock_wallets:
                stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                    hotkey_ss58=wallet.hotkey.ss58_address,
                    coldkey_ss58=wallet.coldkey.ss58_address,
                )

                # Check that all stakes are maximum of 15.0
                self.assertLessEqual(stake.tao, config.max_stake)

            # Check that the balance is correct
            balance = _subtensor_mock.get_balance(
                address=wallet.coldkeypub.ss58_address
            )
            self.assertLessEqual(balance.tao, mock_balance.tao)

    def test_stake_with_single_hotkey_max_stake_not_enough_balance(self, _):
        config = self.config
        config.command = "stake"
        config.subcommand = "add"
        config.no_prompt = True
        # Notie amount is not specified
        config.max_stake = 15.0  # The keys should have at most 15.0 tao staked after
        config.wallet.name = "fake_wallet"
        config.hotkeys = ["hk0"]
        config.all_hotkeys = False

        mock_balance = Balance.from_float(1.0)  # Not enough balance to do max

        mock_coldkey_kp = _get_mock_keypair(0, self.id())

        mock_wallets = [
            SimpleNamespace(
                name=config.wallet.name,
                coldkey=mock_coldkey_kp,
                coldkeypub=mock_coldkey_kp,
                hotkey_str=hk,
                hotkey=_get_mock_keypair(idx + 100, self.id()),
            )
            for idx, hk in enumerate(config.hotkeys)
        ]

        # Register mock wallets and give them balances
        print("Registering mock wallets...")

        for wallet in mock_wallets:
            _ = _subtensor_mock.force_register_neuron(
                netuid=1,
                hotkey=wallet.hotkey.ss58_address,
                coldkey=wallet.coldkeypub.ss58_address,
            )

        _subtensor_mock.force_set_balance(
            ss58_address=mock_coldkey_kp.ss58_address, balance=mock_balance.rao
        )

        cli = bittensor.cli(config)

        def mock_get_wallet(*args, **kwargs):
            if kwargs.get("hotkey"):
                for wallet in mock_wallets:
                    if wallet.hotkey_str == kwargs.get("hotkey"):
                        return wallet
            else:
                return mock_wallets[0]

        with patch("bittensor.wallet") as mock_create_wallet:
            mock_create_wallet.side_effect = mock_get_wallet

            # Check stakes before staking
            for wallet in mock_wallets:
                stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                    hotkey_ss58=wallet.hotkey.ss58_address,
                    coldkey_ss58=wallet.coldkey.ss58_address,
                )
                # Check that all stakes are 0
                self.assertEqual(stake.rao, 0)

            # Check that the balance is correct
            balance = _subtensor_mock.get_balance(
                address=wallet.coldkeypub.ss58_address
            )

            self.assertAlmostEqual(balance.tao, mock_balance.tao, places=4)

            cli.run()

            wallet = mock_wallets[0]

            # Check did not stake
            stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                hotkey_ss58=wallet.hotkey.ss58_address,
                coldkey_ss58=wallet.coldkey.ss58_address,
            )

            # Check that stake is less than max_stake - 1.0
            self.assertLessEqual(stake.tao, config.max_stake - 1.0)

            # Check that the balance decreased by less than max_stake
            balance = _subtensor_mock.get_balance(
                address=wallet.coldkeypub.ss58_address
            )
            self.assertGreaterEqual(balance.tao, mock_balance.tao - config.max_stake)

    def test_stake_with_single_hotkey_max_stake_enough_stake(self, _):
        # tests max stake when stake >= max_stake already
        config = self.config
        config.command = "stake"
        config.subcommand = "add"
        config.no_prompt = True
        # Notie amount is not specified
        config.max_stake = 15.0  # The keys should have at most 15.0 tao staked after
        config.wallet.name = "fake_wallet"
        config.hotkeys = ["hk0"]
        config.all_hotkeys = False

        mock_balance = Balance.from_float(config.max_stake * 3)

        mock_stakes: Dict[str, Balance] = {  # has enough stake, more than max_stake
            "hk0": Balance.from_float(config.max_stake * 2)
        }

        mock_coldkey_kp = _get_mock_keypair(0, self.id())

        mock_wallets = [
            SimpleNamespace(
                name=config.wallet.name,
                coldkey=mock_coldkey_kp,
                coldkeypub=mock_coldkey_kp,
                hotkey_str=hk,
                hotkey=_get_mock_keypair(idx + 100, self.id()),
            )
            for idx, hk in enumerate(config.hotkeys)
        ]

        # Register mock wallets and give them balances
        print("Registering mock wallets...")

        for wallet in mock_wallets:
            _ = _subtensor_mock.force_register_neuron(
                netuid=1,
                hotkey=wallet.hotkey.ss58_address,
                coldkey=wallet.coldkeypub.ss58_address,
                stake=mock_stakes[wallet.hotkey_str].rao,  # More than max_stake
            )

        success, err = _subtensor_mock.force_set_balance(
            ss58_address=mock_coldkey_kp.ss58_address, balance=mock_balance.rao
        )

        cli = bittensor.cli(config)

        def mock_get_wallet(*args, **kwargs):
            if kwargs.get("hotkey"):
                for wallet in mock_wallets:
                    if wallet.hotkey_str == kwargs.get("hotkey"):
                        return wallet
            else:
                return mock_wallets[0]

        with patch("bittensor.wallet") as mock_create_wallet:
            mock_create_wallet.side_effect = mock_get_wallet

            # Check stakes before staking
            wallet = mock_wallets[0]

            stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                hotkey_ss58=wallet.hotkey.ss58_address,
                coldkey_ss58=wallet.coldkey.ss58_address,
            )
            # Check that stake is correct
            self.assertAlmostEqual(
                stake.tao, mock_stakes[wallet.hotkey_str].tao, places=4
            )
            # Check that the stake is greater than or equal to max_stake
            self.assertGreaterEqual(stake.tao, config.max_stake)

            # Check that the balance is correct
            balance = _subtensor_mock.get_balance(
                address=wallet.coldkeypub.ss58_address
            )

            self.assertAlmostEqual(balance.tao, mock_balance.tao, places=4)

            cli.run()

            wallet = mock_wallets[0]

            # Check did not stake, since stake >= max_stake
            stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                hotkey_ss58=wallet.hotkey.ss58_address,
                coldkey_ss58=wallet.coldkey.ss58_address,
            )

            # Check that all stake is unchanged
            self.assertAlmostEqual(
                stake.tao, mock_stakes[wallet.hotkey_str].tao, places=4
            )

            # Check that the balance is the same
            balance = _subtensor_mock.get_balance(
                address=wallet.coldkeypub.ss58_address
            )
            self.assertAlmostEqual(balance.tao, mock_balance.tao, places=4)

    def test_nominate(self, _):
        config = self.config
        config.command = "root"
        config.subcommand = "nominate"
        config.no_prompt = True
        config.wallet.name = "w0"
        config.hotkey = "hk0"

        mock_balance = Balance.from_float(100.0)

        mock_wallet = SimpleNamespace(
            name="w0",
            coldkey=_get_mock_keypair(0, self.id()),
            coldkeypub=_get_mock_keypair(0, self.id()),
            hotkey_str="hk0",
            hotkey=_get_mock_keypair(0 + 100, self.id()),
        )

        # Register mock wallet and give it a balance
        _ = _subtensor_mock.force_register_neuron(
            netuid=1,
            hotkey=mock_wallet.hotkey.ss58_address,
            coldkey=mock_wallet.coldkey.ss58_address,
            balance=mock_balance.rao,
        )

        cli = bittensor.cli(config)

        def mock_get_wallet(*args, **kwargs):
            hk = kwargs.get("hotkey")
            name_ = kwargs.get("name")

            if not hk and kwargs.get("config"):
                hk = kwargs.get("config").wallet.hotkey
            if not name_ and kwargs.get("config"):
                name_ = kwargs.get("config").wallet.name

            if mock_wallet.name == name_:
                return mock_wallet
            else:
                raise ValueError("Mock wallet not found")

        with patch("bittensor.wallet") as mock_create_wallet:
            mock_create_wallet.side_effect = mock_get_wallet

            cli.run()

            # Check the nomination
            is_delegate = _subtensor_mock.is_hotkey_delegate(
                hotkey_ss58=mock_wallet.hotkey.ss58_address
            )
            self.assertTrue(is_delegate)

    def test_delegate_stake(self, _):
        config = self.config
        config.command = "root"
        config.subcommand = "delegate"
        config.no_prompt = True
        config.amount = 5.0
        config.wallet.name = "w1"

        mock_balances: Dict[str, Balance] = {
            # All have more than 5.0 stake
            "w0": {
                "hk0": Balance.from_float(10.0),
            },
            "w1": {"hk1": Balance.from_float(11.1)},
        }

        mock_stake = Balance.from_float(5.0)

        mock_wallets = []
        for idx, wallet_name in enumerate(list(mock_balances.keys())):
            for idx_hk, hk in enumerate(list(mock_balances[wallet_name].keys())):
                wallet = SimpleNamespace(
                    name=wallet_name,
                    coldkey=_get_mock_keypair(idx, self.id()),
                    coldkeypub=_get_mock_keypair(idx, self.id()),
                    hotkey_str=hk,
                    hotkey=_get_mock_keypair(idx * 100 + idx_hk, self.id()),
                )
                mock_wallets.append(wallet)

        # Set hotkey to be the hotkey from the other wallet
        config.delegate_ss58key: str = mock_wallets[0].hotkey.ss58_address

        # Register mock wallets and give them balance
        _ = _subtensor_mock.force_register_neuron(
            netuid=1,
            hotkey=mock_wallets[0].hotkey.ss58_address,
            coldkey=mock_wallets[0].coldkey.ss58_address,
            balance=mock_balances["w0"]["hk0"].rao,
            stake=mock_stake.rao,  # Needs set stake to be a validator
        )

        # Give w1 some balance
        success, err = _subtensor_mock.force_set_balance(
            ss58_address=mock_wallets[1].coldkey.ss58_address,
            balance=mock_balances["w1"]["hk1"].rao,
        )

        # Make the first wallet a delegate
        success = _subtensor_mock.nominate(wallet=mock_wallets[0])
        self.assertTrue(success)

        cli = bittensor.cli(config)

        def mock_get_wallet(*args, **kwargs):
            hk = kwargs.get("hotkey")
            name_ = kwargs.get("name")

            if not hk and kwargs.get("config"):
                hk = kwargs.get("config").wallet.hotkey
            if not name_ and kwargs.get("config"):
                name_ = kwargs.get("config").wallet.name

            for wallet in mock_wallets:
                if wallet.name == name_ and wallet.hotkey_str == hk:
                    return wallet
            else:
                for wallet in mock_wallets:
                    if wallet.name == name_:
                        return wallet
                else:
                    return mock_wallets[0]

        with patch("bittensor.wallet") as mock_create_wallet:
            mock_create_wallet.side_effect = mock_get_wallet

            cli.run()

            # Check the stake
            stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                hotkey_ss58=mock_wallets[0].hotkey.ss58_address,
                coldkey_ss58=mock_wallets[1].coldkey.ss58_address,
            )
            self.assertAlmostEqual(stake.tao, config.amount, places=4)

    def test_undelegate_stake(self, _):
        config = self.config
        config.command = "root"
        config.subcommand = "undelegate"
        config.no_prompt = True
        config.amount = 5.0
        config.wallet.name = "w1"

        mock_balances: Dict[str, Balance] = {
            # All have more than 5.0 stake
            "w0": {
                "hk0": Balance.from_float(10.0),
            },
            "w1": {"hk1": Balance.from_float(11.1)},
        }

        mock_stake = Balance.from_float(5.0)
        mock_delegated = Balance.from_float(6.0)

        mock_wallets = []
        for idx, wallet_name in enumerate(list(mock_balances.keys())):
            for idx_hk, hk in enumerate(list(mock_balances[wallet_name].keys())):
                wallet = SimpleNamespace(
                    name=wallet_name,
                    coldkey=_get_mock_keypair(idx, self.id()),
                    coldkeypub=_get_mock_keypair(idx, self.id()),
                    hotkey_str=hk,
                    hotkey=_get_mock_keypair(idx * 100 + idx_hk, self.id()),
                )
                mock_wallets.append(wallet)

        # Set hotkey to be the hotkey from the other wallet
        config.delegate_ss58key: str = mock_wallets[0].hotkey.ss58_address

        # Register mock wallets and give them balance
        _ = _subtensor_mock.force_register_neuron(
            netuid=1,
            hotkey=mock_wallets[0].hotkey.ss58_address,
            coldkey=mock_wallets[0].coldkey.ss58_address,
            balance=mock_balances["w0"]["hk0"].rao,
            stake=mock_stake.rao,  # Needs set stake to be a validator
        )

        # Give w1 some balance
        success, err = _subtensor_mock.force_set_balance(
            ss58_address=mock_wallets[1].coldkey.ss58_address,
            balance=mock_balances["w1"]["hk1"].rao,
        )

        # Make the first wallet a delegate
        success = _subtensor_mock.nominate(wallet=mock_wallets[0])
        self.assertTrue(success)

        # Stake to the delegate
        success = _subtensor_mock.delegate(
            wallet=mock_wallets[1],
            delegate_ss58=mock_wallets[0].hotkey.ss58_address,
            amount=mock_delegated,
            prompt=False,
        )
        self.assertTrue(success)

        # Verify the stake
        stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
            hotkey_ss58=mock_wallets[0].hotkey.ss58_address,
            coldkey_ss58=mock_wallets[1].coldkey.ss58_address,
        )
        self.assertAlmostEqual(stake.tao, mock_delegated.tao, places=4)

        cli = bittensor.cli(config)

        def mock_get_wallet(*args, **kwargs):
            hk = kwargs.get("hotkey")
            name_ = kwargs.get("name")

            if not hk and kwargs.get("config"):
                hk = kwargs.get("config").wallet.hotkey
            if not name_ and kwargs.get("config"):
                name_ = kwargs.get("config").wallet.name

            for wallet in mock_wallets:
                if wallet.name == name_ and wallet.hotkey_str == hk:
                    return wallet
            else:
                for wallet in mock_wallets:
                    if wallet.name == name_:
                        return wallet
                else:
                    return mock_wallets[0]

        with patch("bittensor.wallet") as mock_create_wallet:
            mock_create_wallet.side_effect = mock_get_wallet

            cli.run()

            # Check the stake
            stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                hotkey_ss58=mock_wallets[0].hotkey.ss58_address,
                coldkey_ss58=mock_wallets[1].coldkey.ss58_address,
            )
            self.assertAlmostEqual(
                stake.tao, mock_delegated.tao - config.amount, places=4
            )

    def test_transfer(self, _):
        config = self.config
        config.command = "wallet"
        config.subcommand = "transfer"
        config.no_prompt = True
        config.amount = 3.2
        config.wallet.name = "w1"

        mock_balances: Dict[str, Balance] = {
            "w0": Balance.from_float(10.0),
            "w1": Balance.from_float(config.amount + 0.001),
        }

        mock_wallets = []
        for idx, wallet_name in enumerate(list(mock_balances.keys())):
            wallet = SimpleNamespace(
                name=wallet_name,
                coldkey=_get_mock_keypair(idx, self.id()),
                coldkeypub=_get_mock_keypair(idx, self.id()),
            )
            mock_wallets.append(wallet)

        # Set dest to w0
        config.dest = mock_wallets[0].coldkey.ss58_address

        # Give w0 and w1 balance

        for wallet in mock_wallets:
            success, err = _subtensor_mock.force_set_balance(
                ss58_address=wallet.coldkey.ss58_address,
                balance=mock_balances[wallet.name].rao,
            )

        cli = bittensor.cli(config)

        def mock_get_wallet(*args, **kwargs):
            name_ = kwargs.get("name")

            if not name_ and kwargs.get("config"):
                name_ = kwargs.get("config").wallet.name

            for wallet in mock_wallets:
                if wallet.name == name_:
                    return wallet
            else:
                raise ValueError(f"No mock wallet found with name: {name_}")

        with patch("bittensor.wallet") as mock_create_wallet:
            mock_create_wallet.side_effect = mock_get_wallet

            cli.run()

            # Check the balance of w0
            balance = _subtensor_mock.get_balance(
                address=mock_wallets[0].coldkey.ss58_address
            )
            self.assertAlmostEqual(
                balance.tao, mock_balances["w0"].tao + config.amount, places=4
            )

            # Check the balance of w1
            balance = _subtensor_mock.get_balance(
                address=mock_wallets[1].coldkey.ss58_address
            )
            self.assertAlmostEqual(
                balance.tao, mock_balances["w1"].tao - config.amount, places=4
            )  # no fees

    def test_transfer_not_enough_balance(self, _):
        config = self.config
        config.command = "wallet"
        config.subcommand = "transfer"
        config.no_prompt = True
        config.amount = 3.2
        config.wallet.name = "w1"

        mock_balances: Dict[str, Balance] = {
            "w0": Balance.from_float(10.0),
            "w1": Balance.from_float(config.amount - 0.1),  # not enough balance
        }

        mock_wallets = []
        for idx, wallet_name in enumerate(list(mock_balances.keys())):
            wallet = SimpleNamespace(
                name=wallet_name,
                coldkey=_get_mock_keypair(idx, self.id()),
                coldkeypub=_get_mock_keypair(idx, self.id()),
            )
            mock_wallets.append(wallet)

        # Set dest to w0
        config.dest = mock_wallets[0].coldkey.ss58_address

        # Give w0 and w1 balance

        for wallet in mock_wallets:
            success, err = _subtensor_mock.force_set_balance(
                ss58_address=wallet.coldkey.ss58_address,
                balance=mock_balances[wallet.name].rao,
            )

        cli = bittensor.cli(config)

        def mock_get_wallet(*args, **kwargs):
            name_ = kwargs.get("name")

            if not name_ and kwargs.get("config"):
                name_ = kwargs.get("config").wallet.name

            for wallet in mock_wallets:
                if wallet.name == name_:
                    return wallet
            else:
                raise ValueError(f"No mock wallet found with name: {name_}")

        mock_console = MockConsole()
        with patch("bittensor.wallet") as mock_create_wallet:
            mock_create_wallet.side_effect = mock_get_wallet

            with patch("bittensor.__console__", mock_console):
                cli.run()

            # Check that the overview was printed.
            self.assertIsNotNone(mock_console.captured_print)

            output_no_syntax = mock_console.remove_rich_syntax(
                mock_console.captured_print
            )

            self.assertIn("Not enough balance", output_no_syntax)

            # Check the balance of w0
            balance = _subtensor_mock.get_balance(
                address=mock_wallets[0].coldkey.ss58_address
            )
            self.assertAlmostEqual(
                balance.tao, mock_balances["w0"].tao, places=4
            )  # did not transfer

            # Check the balance of w1
            balance = _subtensor_mock.get_balance(
                address=mock_wallets[1].coldkey.ss58_address
            )
            self.assertAlmostEqual(
                balance.tao, mock_balances["w1"].tao, places=4
            )  # did not transfer

    def test_register(self, _):
        config = self.config
        config.command = "subnets"
        config.subcommand = "register"
        config.no_prompt = True

        mock_wallet = generate_wallet(hotkey=_get_mock_keypair(100, self.id()))

        # Give the wallet some balance for burning
        success, err = _subtensor_mock.force_set_balance(
            ss58_address=mock_wallet.coldkeypub.ss58_address,
            balance=Balance.from_float(200.0),
        )

        with patch("bittensor.wallet", return_value=mock_wallet) as mock_create_wallet:
            cli = bittensor.cli(config)
            cli.run()
            mock_create_wallet.assert_called_once()

            # Verify that the wallet was registered
            subtensor = bittensor.subtensor(config)
            registered = subtensor.is_hotkey_registered_on_subnet(
                hotkey_ss58=mock_wallet.hotkey.ss58_address, netuid=1
            )

            self.assertTrue(registered)

    def test_pow_register(self, _):
        config = self.config
        config.command = "subnets"
        config.subcommand = "pow_register"
        config.pow_register.num_processes = 1
        config.pow_register.update_interval = 50_000
        config.no_prompt = True

        mock_wallet = generate_wallet(hotkey=_get_mock_keypair(100, self.id()))

        class MockException(Exception):
            pass

        with patch("bittensor.wallet", return_value=mock_wallet) as mock_create_wallet:
            with patch(
                "bittensor.extrinsics.registration.POWSolution.is_stale",
                side_effect=MockException,
            ) as mock_is_stale:
                with pytest.raises(MockException):
                    cli = bittensor.cli(config)
                    cli.run()
                    mock_create_wallet.assert_called_once()

                self.assertEqual(mock_is_stale.call_count, 1)

    def test_stake(self, _):
        amount_to_stake: Balance = Balance.from_tao(0.5)
        config = self.config
        config.no_prompt = True
        config.command = "stake"
        config.subcommand = "add"
        config.amount = amount_to_stake.tao
        config.stake_all = False
        config.use_password = False
        config.model = "core_server"
        config.hotkey = "hk0"

        subtensor = bittensor.subtensor(config)

        mock_wallet = generate_wallet(hotkey=_get_mock_keypair(100, self.id()))

        # Register the hotkey and give it some balance
        _subtensor_mock.force_register_neuron(
            netuid=1,
            hotkey=mock_wallet.hotkey.ss58_address,
            coldkey=mock_wallet.coldkey.ss58_address,
            balance=(
                amount_to_stake + Balance.from_tao(1.0)
            ).rao,  # 1.0 tao extra for fees, etc
        )

        with patch("bittensor.wallet", return_value=mock_wallet) as mock_create_wallet:
            old_stake = subtensor.get_stake_for_coldkey_and_hotkey(
                hotkey_ss58=mock_wallet.hotkey.ss58_address,
                coldkey_ss58=mock_wallet.coldkey.ss58_address,
            )

            cli = bittensor.cli(config)
            cli.run()
            mock_create_wallet.assert_called()
            self.assertEqual(mock_create_wallet.call_count, 2)

            new_stake = subtensor.get_stake_for_coldkey_and_hotkey(
                hotkey_ss58=mock_wallet.hotkey.ss58_address,
                coldkey_ss58=mock_wallet.coldkey.ss58_address,
            )

            self.assertGreater(new_stake, old_stake)

    def test_metagraph(self, _):
        config = self.config
        config.wallet.name = "metagraph_testwallet"
        config.command = "subnets"
        config.subcommand = "metagraph"
        config.no_prompt = True

        # Add some neurons to the metagraph
        mock_nn = []

        def register_mock_neuron(i: int) -> int:
            mock_nn.append(
                SimpleNamespace(
                    hotkey=_get_mock_keypair(i + 100, self.id()).ss58_address,
                    coldkey=_get_mock_keypair(i, self.id()).ss58_address,
                    balance=Balance.from_rao(random.randint(0, 2**45)).rao,
                    stake=Balance.from_rao(random.randint(0, 2**45)).rao,
                )
            )
            uid = _subtensor_mock.force_register_neuron(
                netuid=config.netuid,
                hotkey=mock_nn[i].hotkey,
                coldkey=mock_nn[i].coldkey,
                balance=mock_nn[i].balance,
                stake=mock_nn[i].stake,
            )
            return uid

        for i in range(5):
            _ = register_mock_neuron(i)

        _subtensor_mock.neurons_lite(netuid=config.netuid)

        cli = bittensor.cli(config)

        mock_console = MockConsole()
        with patch("bittensor.__console__", mock_console):
            cli.run()

        # Check that the overview was printed.
        self.assertIsNotNone(mock_console.captured_print)

        output_no_syntax = mock_console.remove_rich_syntax(mock_console.captured_print)

        self.assertIn("Metagraph", output_no_syntax)
        nn = _subtensor_mock.neurons_lite(netuid=config.netuid)
        self.assertIn(
            str(len(nn) - 1), output_no_syntax
        )  # Check that the number of neurons is output
        # Check each uid is in the output
        for neuron in nn:
            self.assertIn(str(neuron.uid), output_no_syntax)

    def test_inspect(self, _):
        config = self.config
        config.wallet.name = "inspect_testwallet"
        config.no_prompt = True
        config.n_words = 12
        config.use_password = False
        config.overwrite_coldkey = True
        config.overwrite_hotkey = True

        # First create a new coldkey
        config.command = "wallet"
        config.subcommand = "new_coldkey"
        cli = bittensor.cli(config)
        cli.run()

        # Now let's give it a hotkey
        config.command = "wallet"
        config.subcommand = "new_hotkey"
        cli.config = config
        cli.run()

        # Now inspect it
        config.command = "wallet"
        cli.config.subcommand = "inspect"
        cli.config = config
        cli.run()

        config.command = "wallet"
        cli.config.subcommand = "list"
        cli.config = config
        cli.run()


@patch("bittensor.subtensor", new_callable=return_mock_sub)
class TestCLIWithNetworkUsingArgs(unittest.TestCase):
    """
    Test the CLI by passing args directly to the bittensor.cli factory
    """

    def test_list_delegates(self, _):
        cli = bittensor.cli(args=["root", "list_delegates"])
        cli.run()

    def test_list_subnets(self, _):
        cli = bittensor.cli(
            args=[
                "subnets",
                "list",
            ]
        )
        cli.run()

    def test_delegate(self, _):
        """
        Test delegate add command
        """
        mock_wallet = generate_wallet(hotkey=_get_mock_keypair(100, self.id()))
        delegate_wallet = generate_wallet(hotkey=_get_mock_keypair(100 + 1, self.id()))

        # register the wallet
        _ = _subtensor_mock.force_register_neuron(
            netuid=1,
            hotkey=mock_wallet.hotkey.ss58_address,
            coldkey=mock_wallet.coldkey.ss58_address,
        )

        # register the delegate
        _ = _subtensor_mock.force_register_neuron(
            netuid=1,
            hotkey=delegate_wallet.hotkey.ss58_address,
            coldkey=delegate_wallet.coldkey.ss58_address,
        )

        # make the delegate a delegate
        _subtensor_mock.nominate(delegate_wallet, wait_for_finalization=True)
        self.assertTrue(
            _subtensor_mock.is_hotkey_delegate(delegate_wallet.hotkey.ss58_address)
        )

        # Give the wallet some TAO
        _, err = _subtensor_mock.force_set_balance(
            ss58_address=mock_wallet.coldkey.ss58_address,
            balance=Balance.from_tao(20.0),
        )
        self.assertEqual(err, None)

        # Check balance
        old_balance = _subtensor_mock.get_balance(mock_wallet.coldkey.ss58_address)
        self.assertEqual(old_balance.tao, 20.0)

        # Check delegate stake
        old_delegate_stake = _subtensor_mock.get_total_stake_for_hotkey(
            delegate_wallet.hotkey.ss58_address
        )

        # Check wallet stake
        old_wallet_stake = _subtensor_mock.get_total_stake_for_coldkey(
            mock_wallet.coldkey.ss58_address
        )

        with patch(
            "bittensor.wallet", return_value=mock_wallet
        ):  # Mock wallet creation. SHOULD NOT BE REGISTERED
            cli = bittensor.cli(
                args=[
                    "root",
                    "delegate",
                    "--subtensor.network",
                    "mock",  # Mock network
                    "--wallet.name",
                    "mock",
                    "--delegate_ss58key",
                    delegate_wallet.hotkey.ss58_address,
                    "--amount",
                    "10.0",  # Delegate 10 TAO
                    "--no_prompt",
                ]
            )
            cli.run()

        # Check delegate stake
        new_delegate_stake = _subtensor_mock.get_total_stake_for_hotkey(
            delegate_wallet.hotkey.ss58_address
        )

        # Check wallet stake
        new_wallet_stake = _subtensor_mock.get_total_stake_for_coldkey(
            mock_wallet.coldkey.ss58_address
        )

        # Check that the delegate stake increased by 10 TAO
        self.assertAlmostEqual(
            new_delegate_stake.tao, old_delegate_stake.tao + 10.0, delta=1e-6
        )

        # Check that the wallet stake increased by 10 TAO
        self.assertAlmostEqual(
            new_wallet_stake.tao, old_wallet_stake.tao + 10.0, delta=1e-6
        )

        new_balance = _subtensor_mock.get_balance(mock_wallet.coldkey.ss58_address)
        self.assertAlmostEqual(new_balance.tao, old_balance.tao - 10.0, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
