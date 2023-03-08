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
from types import SimpleNamespace
from typing import Dict
from unittest.mock import ANY, MagicMock, call, patch
import pytest
from copy import deepcopy

import bittensor
import substrateinterface
from bittensor._subtensor.subtensor_mock import mock_subtensor, Mock_Subtensor
from bittensor.utils.balance import Balance
from substrateinterface.base import Keypair
from tests.helpers import CLOSE_IN_VALUE, get_mock_keypair


_subtensor_mock: Mock_Subtensor = None

def setUpModule():
    global _subtensor_mock
    # Start a mock instance of subtensor.
    _subtensor_mock = bittensor.subtensor( network = 'mock' )

    # create a mock subnet
    created_subnet, err = _subtensor_mock.sudo_add_network( netuid = 1, tempo = 99, modality = 0 )
    assert err == None

    # Make registration difficulty 0. Instant registration.
    set_diff, err = _subtensor_mock.sudo_set_difficulty( netuid = 1, difficulty = 0 )
    assert err == None

def tearDownModule() -> None:
    # Kill the mock instance of subtensor.
    _subtensor_mock.optionally_kill_owned_mock_instance()

def generate_wallet(coldkey : 'Keypair' = None, hotkey: 'Keypair' = None):
    wallet = bittensor.wallet(_mock=True).create()

    if not coldkey:
        coldkey = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    if not hotkey:
        hotkey = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())

    wallet.set_coldkey(coldkey, encrypt=False, overwrite=True)
    wallet.set_coldkeypub(coldkey, encrypt=False, overwrite=True)    
    wallet.set_hotkey(hotkey, encrypt=False, overwrite=True)

    return wallet

class TestCLIWithNetworkAndConfig(unittest.TestCase):
    def setUp(self):
        self._config = TestCLIWithNetworkAndConfig.construct_config()
    
    @property
    def config(self):
        copy_ = deepcopy(self._config)
        return copy_

    @staticmethod
    def construct_config():
        defaults = bittensor.Config()
        defaults.netuid = 1
        bittensor.subtensor.add_defaults( defaults )
        # Always use mock subtensor.
        defaults.subtensor.network = 'mock'
        # Skip version checking.
        defaults.no_version_checking = True
        bittensor.dendrite.add_defaults( defaults )
        bittensor.axon.add_defaults( defaults )
        bittensor.wallet.add_defaults( defaults )
        bittensor.dataset.add_defaults( defaults )
        
        return defaults

    def test_overview( self ):
        # Mock IO for wallet
        with patch('bittensor.Wallet.coldkeypub_file', MagicMock(
            exists_on_device=MagicMock(
                return_value=True # Wallet exists
            )
        )):
            config = self.config
            config.wallet.path = '/tmp/test_cli_test_overview'
            config.wallet.name = 'mock_wallet'
            config.command = "overview"
            config.no_cache = True  # Don't use neuron cache
            config.no_prompt = True
            config.all = False

            cli = bittensor.cli(config)
            with patch('os.walk', return_value=iter(
                    [('/tmp/test_cli_test_overview/mock_wallet/hotkeys', [], ['hk0', 'hk1', 'hk2'])] # no dirs, 3 files
                )):
                with patch('bittensor.Wallet.hotkey', ss58_address=bittensor.Keypair.create_from_mnemonic(
                        bittensor.Keypair.generate_mnemonic()
                ).ss58_address):
                    with patch('bittensor.Wallet.coldkeypub', ss58_address=bittensor.Keypair.create_from_mnemonic(
                        bittensor.Keypair.generate_mnemonic()
                    ).ss58_address):
                        cli.run()

    def test_overview_no_wallet( self ):
        # Mock IO for wallet
        with patch('bittensor.Wallet.coldkeypub_file', MagicMock(
            exists_on_device=MagicMock(
                return_value=False
            )
        )):
            bittensor.subtensor.register = MagicMock(return_value = True)  
            
            config = self.config
            config.command = "overview"
            config.no_prompt = True
            config.all = False
            

            cli = bittensor.cli(config)
            cli.run()

    def test_overview_with_hotkeys_config( self ):        
        config = self.config
        config.command = "overview"
        config.no_prompt = True
        config.hotkeys = ['some_hotkey']
        config.all = False
        

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_without_hotkeys_config( self ):        
        config = self.config
        config.command = "overview"
        config.no_prompt = True
        config.all = False
        

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_with_sort_by_config( self ):        
        config = self.config
        config.command = "overview"
        config.no_prompt = True
        config.wallet.sort_by = "rank"
        config.all = False
        

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_with_sort_by_bad_column_name( self ):        
        config = self.config
        config.command = "overview"
        config.no_prompt = True
        config.wallet.sort_by = "totallynotmatchingcolumnname"
        config.all = False
        

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_without_sort_by_config( self ):        
        config = self.config
        config.command = "overview"
        config.no_prompt = True
        config.all = False
        

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_with_sort_order_config( self ):        
        config = self.config
        config.command = "overview"
        config.wallet.sort_order = "desc" # Set descending sort order
        config.no_prompt = True
        config.all = False
        

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_with_sort_order_config_bad_sort_type( self ):        
        config = self.config
        config.command = "overview"
        config.wallet.sort_order = "nowaythisshouldmatchanyorderingchoice" 
        config.no_prompt = True
        config.all = False
        

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_without_sort_order_config( self ):        
        config = self.config
        config.command = "overview"
        # Don't specify sort_order in config
        config.no_prompt = True
        config.all = False
        

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_with_width_config( self ):        
        config = self.config
        config.command = "overview"
        config.width = 100
        config.no_prompt = True
        config.all = False
        

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_without_width_config( self ):        
        config = self.config
        config.command = "overview"
        # Don't specify width in config
        config.no_prompt = True
        config.all = False
        

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_all( self ):
        config = self.config
        config.command = "overview"
        config.no_prompt = True
        

        config.all = True
        cli = bittensor.cli(config)
        cli.run()

    def test_unstake_with_specific_hotkeys( self ):        
        config = self.config
        config.command = "unstake"
        config.no_prompt = True 
        config.amount = 5.0
        config.wallet.name = "fake_wallet"
        config.hotkeys = [
            'hk0', 'hk1', 'hk2'
        ]   
        config.all_hotkeys =False
        # Notice no max_stake specified

        mock_stakes: Dict[str, bittensor.Balance] = {
            # All have more than 5.0 stake
            'hk0': bittensor.Balance.from_float(10.0),
            'hk1': bittensor.Balance.from_float(11.1),
            'hk2': bittensor.Balance.from_float(12.2),
        }

        mock_coldkey_kp = get_mock_keypair(0, self.id())

        mock_wallets = [
            SimpleNamespace(
                name = config.wallet.name,
                coldkey = mock_coldkey_kp,
                coldkeypub = mock_coldkey_kp,
                hotkey_str = hk,
                hotkey = get_mock_keypair(idx + 100, self.id()),
            ) for idx, hk in enumerate(config.hotkeys)
        ]

        # Register mock wallets and give them stakes
        for wallet in mock_wallets:
            success, err = _subtensor_mock.sudo_register(
                netuid = 1,
                hotkey = wallet.hotkey.ss58_address,
                coldkey = wallet.coldkey.ss58_address,
                stake = mock_stakes[wallet.hotkey_str].rao,
            )
            self.assertTrue(success, err)

        cli = bittensor.cli(config)

        def mock_get_wallet(*args, **kwargs):
            if kwargs.get('hotkey'):
                for wallet in mock_wallets:
                    if wallet.hotkey_str == kwargs.get('hotkey'):
                        return wallet
            else:
                return mock_wallets[0]

        with patch('bittensor.wallet.__new__') as mock_create_wallet:
            mock_create_wallet.side_effect = mock_get_wallet

            # Check stakes before unstaking
            for wallet in mock_wallets:
                stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                    hotkey_ss58=wallet.hotkey.ss58_address,
                    coldkey_ss58=wallet.coldkey.ss58_address
                )
                self.assertEqual(stake.rao, mock_stakes[wallet.hotkey_str].rao)

            cli.run()

            # Check stakes after unstaking
            for wallet in mock_wallets:
                stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                    hotkey_ss58=wallet.hotkey.ss58_address,
                    coldkey_ss58=wallet.coldkey.ss58_address
                )
                self.assertAlmostEqual(stake.tao, mock_stakes[wallet.hotkey_str].tao - config.amount, places=4)

    def test_unstake_with_all_hotkeys( self ):
        config = self.config
        config.command = "unstake"
        config.no_prompt = True 
        config.amount = 5.0
        config.wallet.name = "fake_wallet"
        # Notice wallet.hotkeys not specified
        config.all_hotkeys =True
        # Notice no max_stake specified
        
        mock_stakes: Dict[str, bittensor.Balance] = {
            # All have more than 5.0 stake
            'hk0': bittensor.Balance.from_float(10.0),
            'hk1': bittensor.Balance.from_float(11.1),
            'hk2': bittensor.Balance.from_float(12.2),
        }

        mock_coldkey_kp = get_mock_keypair(0, self.id())

        mock_wallets = [
            SimpleNamespace(
                name = config.wallet.name,
                coldkey = mock_coldkey_kp,
                coldkeypub = mock_coldkey_kp,
                hotkey_str = hk,
                hotkey = get_mock_keypair(idx + 100, self.id()),
            ) for idx, hk in enumerate(list(mock_stakes.keys()))
        ]

        # Register mock wallets and give them stakes
        for wallet in mock_wallets:
            success, err = _subtensor_mock.sudo_register(
                netuid = 1,
                hotkey = wallet.hotkey.ss58_address,
                coldkey = wallet.coldkey.ss58_address,
                stake = mock_stakes[wallet.hotkey_str].rao,
            )
            self.assertTrue(success, err)

        cli = bittensor.cli(config)

        def mock_get_wallet(*args, **kwargs):
            if kwargs.get('hotkey'):
                for wallet in mock_wallets:
                    if wallet.hotkey_str == kwargs.get('hotkey'):
                        return wallet
            else:
                return mock_wallets[0]
        # TODO: Fix mocking here
        with patch('bittensor.cli_utils.get_hotkey_wallets_for_wallet') as mock_get_all_wallets:
            mock_get_all_wallets.side_effect = mock_get_wallet

            # Check stakes before unstaking
            for wallet in mock_wallets:
                stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                    hotkey_ss58=wallet.hotkey.ss58_address,
                    coldkey_ss58=wallet.coldkey.ss58_address
                )
                self.assertEqual(stake.rao, mock_stakes[wallet.hotkey_str].rao)

            cli.run()

            # Check stakes after unstaking
            for wallet in mock_wallets:
                stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                    hotkey_ss58=wallet.hotkey.ss58_address,
                    coldkey_ss58=wallet.coldkey.ss58_address
                )
                self.assertAlmostEqual(stake.tao, mock_stakes[wallet.hotkey_str].tao - config.amount, places=4)

    def test_unstake_with_exclude_hotkeys_from_all( self ):
        config = self.config
        config.command = "unstake"
        config.no_prompt = True 
        config.amount = 5.0
        config.wallet.name = "fake_wallet"
        config.hotkeys = ["hk1"] # Exclude hk1
        config.all_hotkeys =True

        mock_stakes: Dict[str, bittensor.Balance] = {
            # All have more than 5.0 stake
            'hk0': bittensor.Balance.from_float(10.0),
            'hk1': bittensor.Balance.from_float(11.1),
            'hk2': bittensor.Balance.from_float(12.2),
        }

        mock_coldkey_kp = get_mock_keypair(0, self.id())

        mock_wallets = [
            SimpleNamespace(
                name = config.wallet.name,
                coldkey = mock_coldkey_kp,
                coldkeypub = mock_coldkey_kp,
                hotkey_str = hk,
                hotkey = get_mock_keypair(idx + 100, self.id()),
            ) for idx, hk in enumerate(list(mock_stakes.keys()))
        ]

        # Register mock wallets and give them stakes
        for wallet in mock_wallets:
            success, err = _subtensor_mock.sudo_register(
                netuid = 1,
                hotkey = wallet.hotkey.ss58_address,
                coldkey = wallet.coldkey.ss58_address,
                stake = mock_stakes[wallet.hotkey_str].rao,
            )
            self.assertTrue(success, err)

        cli = bittensor.cli(config)

        def mock_get_wallet(*args, **kwargs):
            if kwargs.get('hotkey'):
                for wallet in mock_wallets:
                    if wallet.hotkey_str == kwargs.get('hotkey'):
                        return wallet
            else:
                return mock_wallets[0]
    
        with patch('bittensor.cli_utils.get_hotkey_wallets_for_wallet') as mock_get_all_wallets:
            mock_get_all_wallets.side_effect = mock_get_wallet

            # Check stakes before unstaking
            for wallet in mock_wallets:
                stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                    hotkey_ss58=wallet.hotkey.ss58_address,
                    coldkey_ss58=wallet.coldkey.ss58_address
                )
                self.assertEqual(stake.rao, mock_stakes[wallet.hotkey_str].rao)

            cli.run()

            # Check stakes after unstaking
            for wallet in mock_wallets:
                stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                    hotkey_ss58=wallet.hotkey.ss58_address,
                    coldkey_ss58=wallet.coldkey.ss58_address
                )
                if wallet.hotkey_str == 'hk1':
                    # hk1 should not have been unstaked
                    self.assertAlmostEqual(stake.tao, mock_stakes[wallet.hotkey_str].tao, places=4)
                else:
                    self.assertAlmostEqual(stake.tao, mock_stakes[wallet.hotkey_str].tao - config.amount, places=4)

    def test_unstake_with_multiple_hotkeys_max_stake( self ):        
        config = self.config
        config.command = "unstake"
        config.no_prompt = True 
        # Notie amount is not specified
        config.max_stake = 5.0 # The keys should have at most 5.0 tao staked after
        config.wallet.name = "fake_wallet"
        config.hotkeys = [
            'hk0', 'hk1', 'hk2'
        ]   
        config.all_hotkeys =False

        mock_stakes: Dict[str, bittensor.Balance] = {
            # All have more than 5.0 stake
            'hk0': bittensor.Balance.from_float(10.0),
            'hk1': bittensor.Balance.from_float(4.9),
            'hk2': bittensor.Balance.from_float(12.2),
        }

        mock_coldkey_kp = get_mock_keypair(0, self.id())

        mock_wallets = [
            SimpleNamespace(
                name = config.wallet.name,
                coldkey = mock_coldkey_kp,
                coldkeypub = mock_coldkey_kp,
                hotkey_str = hk,
                hotkey = get_mock_keypair(idx + 100, self.id()),
            ) for idx, hk in enumerate(list(mock_stakes.keys()))
        ]

        # Register mock wallets and give them stakes
        for wallet in mock_wallets:
            success, err = _subtensor_mock.sudo_register(
                netuid = 1,
                hotkey = wallet.hotkey.ss58_address,
                coldkey = wallet.coldkey.ss58_address,
                stake = mock_stakes[wallet.hotkey_str].rao,
            )
            self.assertTrue(success, err)

        cli = bittensor.cli(config)

        def mock_get_wallet(*args, **kwargs):
            if kwargs.get('hotkey'):
                for wallet in mock_wallets:
                    if wallet.hotkey_str == kwargs.get('hotkey'):
                        return wallet
            else:
                return mock_wallets[0]
    
        with patch('bittensor.cli_utils.get_hotkey_wallets_for_wallet') as mock_get_all_wallets:
            mock_get_all_wallets.side_effect = mock_get_wallet

            # Check stakes before unstaking
            for wallet in mock_wallets:
                stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                    hotkey_ss58=wallet.hotkey.ss58_address,
                    coldkey_ss58=wallet.coldkey.ss58_address
                )
                self.assertEqual(stake.rao, mock_stakes[wallet.hotkey_str].rao)

            cli.run()

            # Check stakes after unstaking
            for wallet in mock_wallets:
                stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                    hotkey_ss58=wallet.hotkey.ss58_address,
                    coldkey_ss58=wallet.coldkey.ss58_address
                )
                # All should have been unstaked below or equal to max_stake
                self.assertLessEqual(stake.tao, config.max_stake)

                if wallet.hotkey_str == 'hk1':
                    # hk1 should not have been unstaked because it was already below max_stake
                    self.assertAlmostEqual(stake.tao, mock_stakes[wallet.hotkey_str].tao, places=4)

    def test_stake_with_specific_hotkeys( self ):        
        config = self.config
        config.command = "stake"
        config.no_prompt = True 
        config.amount = 5.0
        config.wallet.name = "fake_wallet"
        config.hotkeys = [
            'hk0', 'hk1', 'hk2'
        ]   
        config.all_hotkeys =False
        # Notice no max_stake specified

        mock_balance = bittensor.Balance.from_float(22.2)

        mock_coldkey_kp = get_mock_keypair(0, self.id())

        mock_wallets = [
            SimpleNamespace(
                name = config.wallet.name,
                coldkey = mock_coldkey_kp,
                coldkeypub = mock_coldkey_kp,
                hotkey_str = hk,
                hotkey = get_mock_keypair(idx + 100, self.id()),
            ) for idx, hk in enumerate(config.hotkeys)
        ]

        # Register mock wallets and give them balances
        for wallet in mock_wallets:
            success, err = _subtensor_mock.sudo_register(
                netuid = 1,
                hotkey = wallet.hotkey.ss58_address,
                coldkey = wallet.coldkey.ss58_address
            )
            self.assertTrue(success, err)

        success, err = _subtensor_mock.sudo_force_set_balance(
            ss58_address=mock_coldkey_kp.ss58_address,
            balance=mock_balance.rao
        )
        self.assertTrue(success, err)

        cli = bittensor.cli(config)

        def mock_get_wallet(*args, **kwargs):
            if kwargs.get('hotkey'):
                for wallet in mock_wallets:
                    if wallet.hotkey_str == kwargs.get('hotkey'):
                        return wallet
            else:
                return mock_wallets[0]

        with patch('bittensor.wallet') as mock_create_wallet:
            mock_create_wallet.side_effect = mock_get_wallet

            # Check stakes before staking
            for wallet in mock_wallets:
                stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                    hotkey_ss58=wallet.hotkey.ss58_address,
                    coldkey_ss58=wallet.coldkey.ss58_address
                )
                self.assertEqual(stake.rao, 0)

            cli.run()

            # Check stakes after staking
            for wallet in mock_wallets:
                stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                    hotkey_ss58=wallet.hotkey.ss58_address,
                    coldkey_ss58=wallet.coldkey.ss58_address
                )
                self.assertAlmostEqual(stake.tao, config.amount, places=4)

    def test_stake_with_all_hotkeys( self ):        
        config = self.config
        config.command = "stake"
        config.no_prompt = True 
        config.amount = 5.0
        config.wallet.name = "fake_wallet"
        # Notice wallet.hotkeys is not specified
        config.all_hotkeys = True
        # Notice no max_stake specified
        
        mock_hotkeys = ['hk0', 'hk1', 'hk2']

        mock_balance = bittensor.Balance.from_float(22.0)

        mock_coldkey_kp = get_mock_keypair(0, self.id())

        mock_wallets = [
            SimpleNamespace(
                name = config.wallet.name,
                coldkey = mock_coldkey_kp,
                coldkeypub = mock_coldkey_kp,
                hotkey_str = hk,
                hotkey = get_mock_keypair(idx + 100, self.id()),
            ) for idx, hk in enumerate(mock_hotkeys)
        ]

        # Register mock wallets and give them no stake
        for wallet in mock_wallets:
            success, err = _subtensor_mock.sudo_register(
                netuid = 1,
                hotkey = wallet.hotkey.ss58_address,
                coldkey = wallet.coldkeypub.ss58_address
            )
            self.assertTrue(success, err)

        # Set the coldkey balance
        success, err = _subtensor_mock.sudo_force_set_balance(
            ss58_address=mock_coldkey_kp.ss58_address,
            balance=mock_balance.rao
        )
        self.assertTrue(success, err)

        cli = bittensor.cli(config)

        def mock_get_wallet(*args, **kwargs):
            if kwargs.get('hotkey'):
                for wallet in mock_wallets:
                    if wallet.hotkey_str == kwargs.get('hotkey'):
                        return wallet
            else:
                return mock_wallets[0]

        with patch('bittensor.wallet') as mock_create_wallet:
            mock_create_wallet.side_effect = mock_get_wallet
            with patch('bittensor.cli_utils.get_hotkey_wallets_for_wallet') as mock_get_hotkey_wallets_for_wallet:
                mock_get_hotkey_wallets_for_wallet.return_value = mock_wallets

                # Check stakes before staking
                for wallet in mock_wallets:
                    stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                        hotkey_ss58=wallet.hotkey.ss58_address,
                        coldkey_ss58=wallet.coldkey.ss58_address
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
                        coldkey_ss58=wallet.coldkey.ss58_address
                    )
                    # Check that all stakes are 5.0
                    self.assertAlmostEqual(stake.tao, config.amount, places=4)

                # Check that the balance is correct
                balance = _subtensor_mock.get_balance(
                    address=wallet.coldkeypub.ss58_address
                )
                self.assertAlmostEqual(balance.tao, mock_balance.tao - (config.amount * len(mock_wallets)), places=4)

    def test_stake_with_exclude_hotkeys_from_all( self ):        
        config = self.config
        config.command = "stake"
        config.no_prompt = True 
        config.amount = 5.0
        config.wallet.name = "fake_wallet"
        config.hotkeys = ['hk1'] # exclude hk1
        config.all_hotkeys =True
        # Notice no max_stake specified
        
        mock_hotkeys = ['hk0', 'hk1', 'hk2']

        mock_balance = bittensor.Balance.from_float(20.0)

        mock_coldkey_kp = get_mock_keypair(0, self.id())

        mock_wallets = [
            SimpleNamespace(
                name = config.wallet.name,
                coldkey = mock_coldkey_kp,
                coldkeypub = mock_coldkey_kp,
                hotkey_str = hk,
                hotkey = get_mock_keypair(idx + 100, self.id()),
            ) for idx, hk in enumerate(mock_hotkeys)
        ]

        # Register mock wallets and give them balances
        for wallet in mock_wallets:
            success, err = _subtensor_mock.sudo_register(
                netuid = 1,
                hotkey = wallet.hotkey.ss58_address,
                coldkey = wallet.coldkeypub.ss58_address
            )
            self.assertTrue(success, err)

        # Set the coldkey balance
        success, err = _subtensor_mock.sudo_force_set_balance(
            ss58_address=mock_coldkey_kp.ss58_address,
            balance=mock_balance.rao
        )
        self.assertTrue(success, err)

        cli = bittensor.cli(config)

        def mock_get_wallet(*args, **kwargs):
            if kwargs.get('hotkey'):
                for wallet in mock_wallets:
                    if wallet.hotkey_str == kwargs.get('hotkey'):
                        return wallet
            else:
                return mock_wallets[0]

        with patch('bittensor.wallet') as mock_create_wallet:
            mock_create_wallet.side_effect = mock_get_wallet

            # Check stakes before staking
            for wallet in mock_wallets:
                stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                    hotkey_ss58=wallet.hotkey.ss58_address,
                    coldkey_ss58=wallet.coldkey.ss58_address
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
                    coldkey_ss58=wallet.coldkey.ss58_address
                )

                if wallet.hotkey_str == 'hk1':
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
            self.assertAlmostEqual(balance.tao, mock_balance.tao - (config.amount * 2), places=4)

    def test_stake_with_multiple_hotkeys_max_stake( self ):        
        config = self.config
        config.command = "stake"
        config.no_prompt = True 
        # Notie amount is not specified
        config.max_stake = 15.0 # The keys should have at most 15.0 tao staked after
        config.wallet.name = "fake_wallet"
        config.hotkeys = [
            'hk0', 'hk1', 'hk2'
        ]   
        config.all_hotkeys =False

        mock_balance = bittensor.Balance.from_float(config.max_stake * 3)

        mock_stakes: Dict[str, bittensor.Balance] = { 
            'hk0': bittensor.Balance.from_float(0.0),
            'hk1': bittensor.Balance.from_float(config.max_stake * 2),
            'hk2': bittensor.Balance.from_float(0.0),
        }

        mock_coldkey_kp = get_mock_keypair(0, self.id())

        mock_wallets = [
            SimpleNamespace(
                name = config.wallet.name,
                coldkey = mock_coldkey_kp,
                coldkeypub = mock_coldkey_kp,
                hotkey_str = hk,
                hotkey = get_mock_keypair(idx + 100, self.id()),
            ) for idx, hk in enumerate(config.hotkeys)
        ]

        # Register mock wallets and give them balances
        for wallet in mock_wallets:
            success, err = _subtensor_mock.sudo_register(
                netuid = 1,
                hotkey = wallet.hotkey.ss58_address,
                coldkey = wallet.coldkeypub.ss58_address
            )
            self.assertTrue(success, err)

        success, err = _subtensor_mock.sudo_force_set_balance(
            ss58_address=mock_coldkey_kp.ss58_address,
            balance=mock_balance.rao
        )
        self.assertTrue(success, err)

        cli = bittensor.cli(config)

        def mock_get_wallet(*args, **kwargs):
            if kwargs.get('hotkey'):
                for wallet in mock_wallets:
                    if wallet.hotkey_str == kwargs.get('hotkey'):
                        return wallet
            else:
                return mock_wallets[0]

        with patch('bittensor.wallet') as mock_create_wallet:
            mock_create_wallet.side_effect = mock_get_wallet

            # Check stakes before staking
            for wallet in mock_wallets:
                stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                    hotkey_ss58=wallet.hotkey.ss58_address,
                    coldkey_ss58=wallet.coldkey.ss58_address
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
                    coldkey_ss58=wallet.coldkey.ss58_address
                )

                # Check that all stakes at least 15.0
                self.assertGreaterEqual(stake.tao + 0.1, config.max_stake)

                if wallet.hotkey_str == "hk1":
                    # Check that hk1 stake was not changed
                    # It had more than max_stake already
                    self.assertAlmostEqual(stake.tao, mock_stakes[wallet.hotkey_str].tao, places=4)

            # Check that the balance decreased
            balance = _subtensor_mock.get_balance(
                address=wallet.coldkeypub.ss58_address
            )
            self.assertLessEqual(balance.tao, mock_balance.tao)

    def test_stake_with_multiple_hotkeys_max_stake_not_enough_balance( self ):        
        config = self.config
        config.command = "stake"
        config.no_prompt = True 
        # Notie amount is not specified
        config.max_stake = 15.0 # The keys should have at most 15.0 tao staked after
        config.wallet.name = "fake_wallet"
        config.hotkeys = [
            'hk0', 'hk1', 'hk2'
        ]   
        config.all_hotkeys =False

        mock_balance = bittensor.Balance.from_float(15.0 * 2) # Not enough for all hotkeys

        mock_coldkey_kp = get_mock_keypair(0, self.id())

        mock_wallets = [
            SimpleNamespace(
                name = config.wallet.name,
                coldkey = mock_coldkey_kp,
                coldkeypub = mock_coldkey_kp,
                hotkey_str = hk,
                hotkey = get_mock_keypair(idx + 100, self.id()),
            ) for idx, hk in enumerate(config.hotkeys)
        ]

        # Register mock wallets and give them balances
        for wallet in mock_wallets:
            success, err = _subtensor_mock.sudo_register(
                netuid = 1,
                hotkey = wallet.hotkey.ss58_address,
                coldkey = wallet.coldkeypub.ss58_address
            )
            self.assertTrue(success, err)

        success, err = _subtensor_mock.sudo_force_set_balance(
            ss58_address=mock_coldkey_kp.ss58_address,
            balance=mock_balance.rao
        )
        self.assertTrue(success, err)

        cli = bittensor.cli(config)

        def mock_get_wallet(*args, **kwargs):
            if kwargs.get('hotkey'):
                for wallet in mock_wallets:
                    if wallet.hotkey_str == kwargs.get('hotkey'):
                        return wallet
            else:
                return mock_wallets[0]

        with patch('bittensor.wallet') as mock_create_wallet:
            mock_create_wallet.side_effect = mock_get_wallet

            # Check stakes before staking
            for wallet in mock_wallets:
                stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                    hotkey_ss58=wallet.hotkey.ss58_address,
                    coldkey_ss58=wallet.coldkey.ss58_address
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
                    coldkey_ss58=wallet.coldkey.ss58_address
                )

                if wallet.hotkey_str == 'hk2':
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

    def test_stake_with_single_hotkey_max_stake( self ):        
        config = self.config
        config.command = "stake"
        config.no_prompt = True 
        # Notie amount is not specified
        config.max_stake = 15.0 # The keys should have at most 15.0 tao staked after
        config.wallet.name = "fake_wallet"
        config.hotkeys = [
            'hk0'
        ]   
        config.all_hotkeys =False

        mock_balance = bittensor.Balance.from_float(15.0 * 3)

        mock_coldkey_kp = get_mock_keypair(0, self.id())

        mock_wallets = [
            SimpleNamespace(
                name = config.wallet.name,
                coldkey = mock_coldkey_kp,
                coldkeypub = mock_coldkey_kp,
                hotkey_str = hk,
                hotkey = get_mock_keypair(idx + 100, self.id()),
            ) for idx, hk in enumerate(config.hotkeys)
        ]

        # Register mock wallets and give them balances
        for wallet in mock_wallets:
            success, err = _subtensor_mock.sudo_register(
                netuid = 1,
                hotkey = wallet.hotkey.ss58_address,
                coldkey = wallet.coldkeypub.ss58_address
            )
            self.assertTrue(success, err)

        success, err = _subtensor_mock.sudo_force_set_balance(
            ss58_address=mock_coldkey_kp.ss58_address,
            balance=mock_balance.rao
        )
        self.assertTrue(success, err)

        cli = bittensor.cli(config)

        def mock_get_wallet(*args, **kwargs):
            if kwargs.get('hotkey'):
                for wallet in mock_wallets:
                    if wallet.hotkey_str == kwargs.get('hotkey'):
                        return wallet
            else:
                return mock_wallets[0]

        with patch('bittensor.wallet') as mock_create_wallet:
            mock_create_wallet.side_effect = mock_get_wallet

            # Check stakes before staking
            for wallet in mock_wallets:
                stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                    hotkey_ss58=wallet.hotkey.ss58_address,
                    coldkey_ss58=wallet.coldkey.ss58_address
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
                    coldkey_ss58=wallet.coldkey.ss58_address
                )

                # Check that all stakes are maximum of 15.0
                self.assertLessEqual(stake.tao, config.max_stake)

            # Check that the balance is correct
            balance = _subtensor_mock.get_balance(
                address=wallet.coldkeypub.ss58_address
            )
            self.assertLessEqual(balance.tao, mock_balance.tao)

    def test_stake_with_single_hotkey_max_stake_not_enough_balance( self ):        
        config = self.config
        config.command = "stake"
        config.no_prompt = True 
        # Notie amount is not specified
        config.max_stake = 15.0 # The keys should have at most 15.0 tao staked after
        config.wallet.name = "fake_wallet"
        config.hotkeys = [
            'hk0'
        ]   
        config.all_hotkeys =False

        mock_balance = bittensor.Balance.from_float(1.0) # Not enough balance to do max

        mock_coldkey_kp = get_mock_keypair(0, self.id())

        mock_wallets = [
            SimpleNamespace(
                name = config.wallet.name,
                coldkey = mock_coldkey_kp,
                coldkeypub = mock_coldkey_kp,
                hotkey_str = hk,
                hotkey = get_mock_keypair(idx + 100, self.id()),
            ) for idx, hk in enumerate(config.hotkeys)
        ]

        # Register mock wallets and give them balances
        for wallet in mock_wallets:
            success, err = _subtensor_mock.sudo_register(
                netuid = 1,
                hotkey = wallet.hotkey.ss58_address,
                coldkey = wallet.coldkeypub.ss58_address
            )
            self.assertTrue(success, err)

        success, err = _subtensor_mock.sudo_force_set_balance(
            ss58_address=mock_coldkey_kp.ss58_address,
            balance=mock_balance.rao
        )
        self.assertTrue(success, err)

        cli = bittensor.cli(config)

        def mock_get_wallet(*args, **kwargs):
            if kwargs.get('hotkey'):
                for wallet in mock_wallets:
                    if wallet.hotkey_str == kwargs.get('hotkey'):
                        return wallet
            else:
                return mock_wallets[0]

        with patch('bittensor.wallet') as mock_create_wallet:
            mock_create_wallet.side_effect = mock_get_wallet

            # Check stakes before staking
            for wallet in mock_wallets:
                stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                    hotkey_ss58=wallet.hotkey.ss58_address,
                    coldkey_ss58=wallet.coldkey.ss58_address
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
                coldkey_ss58=wallet.coldkey.ss58_address
            )

            # Check that stake is less than max_stake - 1.0
            self.assertLessEqual(stake.tao, config.max_stake - 1.0)

            # Check that the balance decreased by less than max_stake
            balance = _subtensor_mock.get_balance(
                address=wallet.coldkeypub.ss58_address
            )
            self.assertGreaterEqual(balance.tao, mock_balance.tao - config.max_stake)


    def test_stake_with_single_hotkey_max_stake_enough_stake( self ):
        # tests max stake when stake >= max_stake already
        config = self.config
        config.command = "stake"
        config.no_prompt = True 
        # Notie amount is not specified
        config.max_stake = 15.0 # The keys should have at most 15.0 tao staked after
        config.wallet.name = "fake_wallet"
        config.hotkeys = [
            'hk0'
        ]   
        config.all_hotkeys =False

        mock_balance = bittensor.Balance.from_float(config.max_stake * 3)

        mock_stakes: Dict[str, bittensor.Balance] = { # has enough stake, more than max_stake
            'hk0': bittensor.Balance.from_float(config.max_stake * 2)
        }

        mock_coldkey_kp = get_mock_keypair(0, self.id())

        mock_wallets = [
            SimpleNamespace(
                name = config.wallet.name,
                coldkey = mock_coldkey_kp,
                coldkeypub = mock_coldkey_kp,
                hotkey_str = hk,
                hotkey = get_mock_keypair(idx + 100, self.id()),
            ) for idx, hk in enumerate(config.hotkeys)
        ]

        # Register mock wallets and give them balances
        for wallet in mock_wallets:
            success, err = _subtensor_mock.sudo_register(
                netuid = 1,
                hotkey = wallet.hotkey.ss58_address,
                coldkey = wallet.coldkeypub.ss58_address
            )
            self.assertTrue(success, err)

        success, err = _subtensor_mock.sudo_force_set_balance(
            ss58_address=mock_coldkey_kp.ss58_address,
            balance=mock_balance.rao
        )
        self.assertTrue(success, err)

        cli = bittensor.cli(config)

        def mock_get_wallet(*args, **kwargs):
            if kwargs.get('hotkey'):
                for wallet in mock_wallets:
                    if wallet.hotkey_str == kwargs.get('hotkey'):
                        return wallet
            else:
                return mock_wallets[0]

        with patch('bittensor.wallet') as mock_create_wallet:
            mock_create_wallet.side_effect = mock_get_wallet

            # Check stakes before staking
            wallet = mock_wallets[0]
        

            stake = _subtensor_mock.get_stake_for_coldkey_and_hotkey(
                hotkey_ss58=wallet.hotkey.ss58_address,
                coldkey_ss58=wallet.coldkey.ss58_address
            )
            # Check that stake is correct
            self.assertAlmostEqual(stake.tao, mock_stakes[wallet.hotkey_str].tao, places=4)
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
                coldkey_ss58=wallet.coldkey.ss58_address
            )

            # Check that all stake is unchanged
            self.assertAlmostEqual(stake.tao, mock_stakes[wallet.hotkey_str].tao, places=4)

            # Check that the balance is the same
            balance = _subtensor_mock.get_balance(
                address=wallet.coldkeypub.ss58_address
            )
            self.assertAlmostEqual(balance.tao, mock_balance.tao, places=4)
                
    def test_register( self ):
        config = self.config
        config.command = "register"
        config.subtensor.register.num_processes = 1
        config.subtensor.register.update_interval = 50_000
        config.no_prompt = True

        mock_wallet = generate_wallet()
        
        with patch('bittensor.wallet', return_value=mock_wallet) as mock_create_wallet:
            cli = bittensor.cli(config)
            cli.run()
            mock_create_wallet.assert_called_once()
            
            subtensor = bittensor.subtensor(config)
            registered = subtensor.is_hotkey_registered_on_subnet( hotkey_ss58 = mock_wallet.hotkey.ss58_address, netuid = 1 )

            self.assertTrue( registered )
      
    def test_stake( self ):
        amount_to_stake: Balance = Balance.from_tao( 0.5 )
        config = self.config
        config.no_prompt = True
        config.command = "stake"
        config.amount = amount_to_stake.tao
        config.stake_all = False
        config.wallet._mock = True
        config.use_password = False
        config.model = "core_server"
        config.hotkey = "hk0"

        subtensor = bittensor.subtensor(config)

        mock_wallet = generate_wallet()

        # Register the hotkey and give it some balance
        _subtensor_mock.sudo_register(
            netuid = 1,
            hotkey = mock_wallet.hotkey.ss58_address,
            coldkey = mock_wallet.coldkey.ss58_address,
            balance = (amount_to_stake + Balance.from_tao( 1.0 )).rao # 1.0 tao extra for fees, etc
        )

        with patch('bittensor.wallet', return_value=mock_wallet) as mock_create_wallet:
            
            old_stake = subtensor.get_stake_for_coldkey_and_hotkey(
                hotkey_ss58 = mock_wallet.hotkey.ss58_address,
                coldkey_ss58 = mock_wallet.coldkey.ss58_address,
            )

            cli = bittensor.cli(config)
            cli.run()
            mock_create_wallet.assert_called()
            self.assertEqual(mock_create_wallet.call_count, 2)
            
            new_stake = subtensor.get_stake_for_coldkey_and_hotkey(
                hotkey_ss58 = mock_wallet.hotkey.ss58_address,
                coldkey_ss58 = mock_wallet.coldkey.ss58_address,
            )

            self.assertGreater( new_stake, old_stake )      
   
    def test_metagraph( self ):    
        config = self.config
        config.wallet.name = "metagraph_testwallet"
        config.command = "metagraph"
        config.no_prompt = True
        

        cli = bittensor.cli(config)
        cli.run()

    def test_set_weights( self ):

        config = self.config
        config.wallet.name = "set_weights_testwallet"
        config.no_prompt = True
        config.uids = [1, 2, 3, 4]
        config.weights = [0.25, 0.25, 0.25, 0.25]
        config.n_words = 12
        config.use_password = False
        


        config.overwrite_hotkey = True

        # First create a new hotkey
        config.command = "new_hotkey"
        cli = bittensor.cli(config)
        cli.run()
        
        # Now set the weights
        config.command = "set_weights"
        cli.config = config
        cli.run()

    def test_inspect( self ):
        config = self.config
        config.wallet.name = "inspect_testwallet"
        config.no_prompt = True
        config.n_words = 12
        config.use_password = False
        config.overwrite_coldkey = True
        config.overwrite_hotkey = True
        

        # First create a new coldkey
        config.command = "new_coldkey"
        cli = bittensor.cli(config)
        cli.run()

        # Now let's give it a hotkey
        config.command = "new_hotkey"
        cli.config = config
        cli.run()

        # Now inspect it
        cli.config.command = "inspect"
        cli.config = config
        cli.run()

        cli.config.command = "list"
        cli.config = config
        cli.run()

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
            ]
            bittensor.subtensor.check_config = MagicMock(return_value = True)  
            with patch('torch.cuda.is_available', return_value=True):
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

class TestCLIWithNetworkUsingArgs(unittest.TestCase):
    """
    Test the CLI by passing args directly to the bittensor.cli factory
    """
    def test_run_reregister_false(self):
        """
        Verify that the btcli run command does not reregister a not registered wallet
            if --wallet.reregister is False
        """

        # Mock wallet SHOULD NOT BE REGISTERED
        mock_wallet = bittensor.wallet(_mock = True)
        self.assertFalse(_subtensor_mock.is_hotkey_registered( 
            hotkey_ss58 = mock_wallet.hotkey.ss58_address,
            netuid = 1
        ))
        
        with patch('bittensor.Subtensor.register', MagicMock(side_effect=Exception("shouldn't register during test"))):
            with pytest.raises(SystemExit):
                cli = bittensor.cli(args=[
                    'run',
                    '--netuid', '1',
                    '--wallet.name', 'mock',
                    '--wallet.hotkey', 'mock_hotkey',
                    '--wallet._mock', 'True',
                    '--subtensor.network', 'mock', # Mock network
                    '--no_prompt',
                    '--wallet.reregister', 'False' # Don't reregister
                ])
                cli.run()

    def test_run_synapse_all(self):
        """
        Verify that setting --synapse All works
        """

        class MockException(Exception):
            """Raised by mocked function to exit early"""
            pass

        with patch('bittensor.neurons.core_server.neuron', MagicMock(side_effect=MockException("should exit early"))) as mock_neuron:
            with patch('bittensor.Wallet.is_registered', MagicMock(return_value=True)): # mock registered
                with pytest.raises(MockException):
                    cli = bittensor.cli(args=[
                        'run',
                        '--subtensor.network', 'mock', # Mock network
                        '--netuid', '1',
                        '--wallet.name', 'mock',
                        '--wallet.hotkey', 'mock_hotkey',
                        '--wallet._mock', 'True',
                        '--cuda.no_cuda',
                        '--no_prompt',
                        '--model', 'core_server',
                        '--synapse', 'All',
                    ])
                    cli.run()

                assert mock_neuron.call_count == 1
                args, kwargs = mock_neuron.call_args

                self.assertEqual(len(args), 0) # Should not have any args; indicates that "All" synapses are being used
                self.assertEqual(len(kwargs), 1) # should have one kwarg; netuid

    def test_list_delegates(self):
        cli = bittensor.cli(args=[
            'list_delegates',
            '--subtensor.network', 'mock', # Mock network
        ])
        cli.run()

    def test_list_subnets(self):
        cli = bittensor.cli(args=[
            'list_subnets',
            '--subtensor.network', 'mock', # Mock network
        ])
        cli.run()

    def test_delegate(self):
        """
        Test delegate add command
        """
        mock_wallet = generate_wallet()
        delegate_wallet = generate_wallet()

        # register the wallet
        _, err = _subtensor_mock.sudo_register(
            netuid = 1,
            hotkey = mock_wallet.hotkey.ss58_address,
            coldkey = mock_wallet.coldkey.ss58_address,
        )
        self.assertEqual(err, None)

        # register the delegate
        _, err = _subtensor_mock.sudo_register(
            netuid = 1,
            hotkey = delegate_wallet.hotkey.ss58_address,
            coldkey = delegate_wallet.coldkey.ss58_address,
        )
        self.assertEqual(err, None)

        # make the delegate a delegate
        _subtensor_mock.nominate(delegate_wallet, wait_for_finalization=True)
        self.assertTrue(_subtensor_mock.is_hotkey_delegate( delegate_wallet.hotkey.ss58_address ))

        # Give the wallet some TAO
        _, err = _subtensor_mock.sudo_force_set_balance(
            ss58_address=mock_wallet.coldkey.ss58_address,
            balance = bittensor.Balance.from_tao( 20.0 )
        )
        self.assertEqual(err, None)

        # Check balance
        old_balance = _subtensor_mock.get_balance( mock_wallet.coldkey.ss58_address )
        self.assertEqual(old_balance.tao, 20.0)

        # Check delegate stake
        old_delegate_stake = _subtensor_mock.get_total_stake_for_hotkey(delegate_wallet.hotkey.ss58_address)

        # Check wallet stake
        old_wallet_stake = _subtensor_mock.get_total_stake_for_coldkey(mock_wallet.coldkey.ss58_address)

        with patch('bittensor._wallet.wallet_mock.Wallet_mock', return_value=mock_wallet): # Mock wallet creation. SHOULD NOT BE REGISTERED
            cli = bittensor.cli(args=[
                'delegate',
                '--subtensor.network', 'mock', # Mock network
                '--wallet.name', 'mock',
                '--wallet._mock', 'True',
                '--delegate_ss58key', delegate_wallet.hotkey.ss58_address,
                '--amount', '10.0', # Delegate 10 TAO
                '--no_prompt',
            ])
            cli.run()

        # Check delegate stake
        new_delegate_stake = _subtensor_mock.get_total_stake_for_hotkey(delegate_wallet.hotkey.ss58_address)

        # Check wallet stake
        new_wallet_stake = _subtensor_mock.get_total_stake_for_coldkey(mock_wallet.coldkey.ss58_address)

        # Check that the delegate stake increased by 10 TAO
        self.assertAlmostEqual(new_delegate_stake.tao, old_delegate_stake.tao + 10.0, delta=1e-6)

        # Check that the wallet stake increased by 10 TAO
        self.assertAlmostEqual(new_wallet_stake.tao, old_wallet_stake.tao + 10.0, delta=1e-6)

        new_balance = _subtensor_mock.get_balance( mock_wallet.coldkey.ss58_address )
        self.assertAlmostEqual(new_balance.tao, old_balance.tao - 10.0, delta=1e-6)


if __name__ == '__main__':
    unittest.main()