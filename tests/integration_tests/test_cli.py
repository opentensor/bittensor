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
from tests.helpers import CLOSE_IN_VALUE, get_mock_hotkey


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
        config.all_hotkey = False
        # Notice no max_stake specified
        

        mock_coldkey = "" # Not None

        mock_stakes: Dict[str, float] = {
            # All have more than 5.0 stake
            'hk0': bittensor.Balance.from_float(10.0),
            'hk1': bittensor.Balance.from_float(11.1),
            'hk2': bittensor.Balance.from_float(12.2),
        }

        mock_wallets = [
            SimpleNamespace(
                name = config.wallet.name,
                hotkey_str = config.hotkeys[0],
                hotkey = get_mock_hotkey(0),
                get_stake = MagicMock(
                    return_value = mock_stakes[config.hotkeys[0]]
                ),
                is_registered = MagicMock(
                    return_value = True
                ),
                _coldkey = mock_coldkey,
                coldkey =  MagicMock(
                            return_value=mock_coldkey
                        )
            ) 
        ] + [
            SimpleNamespace(
                name = config.wallet.name,
                hotkey_str = hk,
                hotkey = get_mock_hotkey(idx),
                get_stake = MagicMock(
                    return_value = mock_stakes[hk]
                ),
                is_registered = MagicMock(
                    return_value = True
                )
            ) for idx, hk in enumerate(config.hotkeys)
        ]

        # The 0th wallet is created twice during unstake
        mock_wallets[1]._coldkey = mock_coldkey
        mock_wallets[1].coldkey = MagicMock(
                                    return_value=mock_coldkey
                                )
        
        cli = bittensor.cli(config)
        mock_coldkey = MagicMock(

        )

        with patch('bittensor.wallet') as mock_create_wallet:
            mock_create_wallet.side_effect = mock_wallets
            with patch('bittensor.Subtensor.unstake_multiple', return_value=True) as mock_unstake:
                cli.run()
                mock_create_wallet.assert_has_calls(
                    [
                        call(config=ANY, hotkey=hk) for hk in config.hotkeys
                    ],
                    any_order=True
                )
                mock_unstake.assert_has_calls(
                    [call(wallets=mock_wallets[1:], amounts=[5.0]*len(mock_wallets[1:]), wait_for_inclusion=True, prompt=False)],
                    any_order = True
                )

    def test_unstake_with_all_hotkeys( self ):
        config = self.config
        config.command = "unstake"
        config.no_prompt = True 
        config.amount = 5.0
        config.wallet.name = "fake_wallet"
        # Notice wallet.hotkeys not specified
        config.all_hotkey = True
        # Notice no max_stake specified
        

        mock_coldkey = "" # Not None

        mock_stakes: Dict[str, float] = {
            # All have more than 5.0 stake
            'hk0': bittensor.Balance.from_float(10.0),
            'hk1': bittensor.Balance.from_float(11.1),
            'hk2': bittensor.Balance.from_float(12.2),
        }

        mock_wallets = [
            SimpleNamespace(
                name = config.wallet.name,
                hotkey_str = hk,
                hotkey = get_mock_hotkey(idx),
                get_stake = MagicMock(
                    return_value = mock_stakes[hk]
                ),
                is_registered = MagicMock(
                    return_value = True
                )
            ) for idx, hk in enumerate(list(mock_stakes.keys()))
        ]

        mock_wallets[0]._coldkey = mock_coldkey
        mock_wallets[0].coldkey = MagicMock(
                                    return_value=mock_coldkey
                                )
        
        cli = bittensor.cli(config)
    
        with patch.object(cli, '_get_hotkey_wallets_for_wallet') as mock_get_all_wallets:
            mock_get_all_wallets.return_value = mock_wallets
            with patch('bittensor.Subtensor.unstake_multiple', return_value=True) as mock_unstake:
                cli.run()
                mock_get_all_wallets.assert_called_once()
                mock_unstake.assert_has_calls(
                    [call(wallets=mock_wallets, amounts=[5.0]*len(mock_wallets), wait_for_inclusion=True, prompt=False)],
                    any_order = True
                )

    def test_unstake_with_exclude_hotkeys_from_all( self ):
        config = self.config
        config.command = "unstake"
        config.no_prompt = True 
        config.amount = 5.0
        config.wallet.name = "fake_wallet"
        config.hotkeys = ["hk1"] # Exclude hk1
        config.all_hotkey = True
        # Notice no max_stake specified
        

        mock_coldkey = "" # Not None

        mock_stakes: Dict[str, float] = {
            # All have more than 5.0 stake
            'hk0': bittensor.Balance.from_float(10.0),
            'hk1': bittensor.Balance.from_float(11.1),
            'hk2': bittensor.Balance.from_float(12.2),
        }

        mock_wallets = [
            SimpleNamespace(
                name = config.wallet.name,
                hotkey_str = hk,
                hotkey = get_mock_hotkey(idx),
                get_stake = MagicMock(
                    return_value = mock_stakes[hk]
                ),
                is_registered = MagicMock(
                    return_value = True
                )
            ) for idx, hk in enumerate(list(mock_stakes.keys()))
        ]

        mock_wallets[0]._coldkey = mock_coldkey
        mock_wallets[0].coldkey = MagicMock(
                                    return_value=mock_coldkey
                                )
        
        cli = bittensor.cli(config)
    
        with patch.object(cli, '_get_hotkey_wallets_for_wallet') as mock_get_all_wallets:
            mock_get_all_wallets.return_value = mock_wallets
            with patch('bittensor.Subtensor.unstake_multiple', return_value=True) as mock_unstake:
                cli.run()
                mock_get_all_wallets.assert_called_once()
                mock_unstake.assert_has_calls(
                    [call(wallets=[mock_wallets[0], mock_wallets[2]], amounts=[5.0, 5.0], wait_for_inclusion=True, prompt=False)],
                    any_order = True
                )

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
        config.all_hotkey = False
        # Notice no max_stake specified
        

        mock_coldkey = "" # Not None

        mock_stakes: Dict[str, float] = {
            # All have more than 5.0 stake
            'hk0': bittensor.Balance.from_float(10.0),
            'hk1': bittensor.Balance.from_float(11.1),
            'hk2': bittensor.Balance.from_float(12.2),
        }

        mock_wallets = [
            SimpleNamespace(
                name = config.wallet.name,
                hotkey_str = config.hotkeys[0],
                hotkey = get_mock_hotkey(0),
                get_stake = MagicMock(
                    return_value = mock_stakes[config.hotkeys[0]]
                ),
                is_registered = MagicMock(
                    return_value = True
                ),

                _coldkey = mock_coldkey,
                coldkey =  MagicMock(
                            return_value=mock_coldkey
                        )
            ) 
        ] + [
            SimpleNamespace(
                name = config.wallet.name,
                hotkey_str = hk,
                hotkey = get_mock_hotkey(idx),
                get_stake = MagicMock(
                    return_value = mock_stakes[hk]
                ),
                is_registered = MagicMock(
                    return_value = True
                )
            ) for idx, hk in enumerate(config.hotkeys)
        ]

        # The 0th wallet is created twice during unstake
        mock_wallets[1]._coldkey = mock_coldkey
        mock_wallets[1].coldkey = MagicMock(
                                    return_value=mock_coldkey
                                )
        
        cli = bittensor.cli(config)

        with patch('bittensor.wallet') as mock_create_wallet:
            mock_create_wallet.side_effect = mock_wallets
            with patch('bittensor.Subtensor.unstake_multiple', return_value=True) as mock_unstake:
                cli.run()
                mock_create_wallet.assert_has_calls(
                    [
                        call(config=ANY, hotkey=hk) for hk in config.hotkeys
                    ],
                    any_order=True
                )
                mock_unstake.assert_has_calls(
                    [call(wallets=mock_wallets[1:], amounts=[CLOSE_IN_VALUE((mock_stakes[mock_wallet.hotkey_str].tao - config.max_stake), 0.001) for mock_wallet in mock_wallets[1:]], wait_for_inclusion=True, prompt=False)],
                    any_order = True
                )

    def test_unstake_with_multiple_hotkeys_max_stake_not_enough_stake( self ):        
        config = self.config
        config.command = "unstake"
        config.no_prompt = True 
        # Notie amount is not specified
        config.max_stake = 5.0 # The keys should have at most 5.0 tao staked after
        config.wallet.name = "fake_wallet"
        config.hotkeys = [
            'hk0', 'hk1', 'hk2'
        ]   
        config.all_hotkey = False
        # Notice no max_stake specified
        

        mock_coldkey = "" # Not None

        mock_stakes: Dict[str, float] = {
            # hk1 has less than 5.0 stake
            'hk0': bittensor.Balance.from_float(10.0),
            'hk1': bittensor.Balance.from_float(4.9),
            'hk2': bittensor.Balance.from_float(12.2),
        }

        mock_wallets = [
            SimpleNamespace(
                name = config.wallet.name,
                hotkey_str = config.hotkeys[0],
                hotkey = get_mock_hotkey(0),
                get_stake = MagicMock(
                    return_value = mock_stakes[config.hotkeys[0]]
                ),
                is_registered = MagicMock(
                    return_value = True
                ),

                _coldkey = mock_coldkey,
                coldkey =  MagicMock(
                            return_value=mock_coldkey
                        )
            ) 
        ] + [
            SimpleNamespace(
                name = config.wallet.name,
                hotkey_str = hk,
                hotkey = get_mock_hotkey(idx),
                get_stake = MagicMock(
                    return_value = mock_stakes[hk]
                ),
                is_registered = MagicMock(
                    return_value = True
                )
            ) for idx, hk in enumerate(config.hotkeys)
        ]

        # The 0th wallet is created twice during unstake
        mock_wallets[1]._coldkey = mock_coldkey
        mock_wallets[1].coldkey = MagicMock(
                                    return_value=mock_coldkey
                                )
        
        cli = bittensor.cli(config)

        with patch('bittensor.wallet') as mock_create_wallet:
            mock_create_wallet.side_effect = mock_wallets
            with patch('bittensor.Subtensor.unstake_multiple', return_value=True) as mock_unstake:
                cli.run()
                mock_create_wallet.assert_has_calls(
                    [
                        call(config=ANY, hotkey=hk) for hk in config.hotkeys
                    ],
                    any_order=True
                )
                mock_unstake.assert_called()

                # Python 3.7 
                ## https://docs.python.org/3.7/library/unittest.mock.html#call
                ## Uses the 1st index as args list
                ## call.args only works in Python 3.8+
                args, kwargs = mock_unstake.call_args
                mock_wallets_ = kwargs['wallets']
                

                # We shouldn't unstake from hk1 as it has less than max_stake staked
                assert all(mock_wallet.hotkey_str != 'hk1' for mock_wallet in mock_wallets_)

    def test_stake_with_specific_hotkeys( self ):        
        config = self.config
        config.command = "stake"
        config.no_prompt = True 
        config.amount = 5.0
        config.wallet.name = "fake_wallet"
        config.hotkeys = [
            'hk0', 'hk1', 'hk2'
        ]   
        config.all_hotkey = False
        # Notice no max_stake specified
        

        mock_coldkey = "" # Not None

        # enough to stake 5.0 to all 3 hotkeys
        mock_balance: Balance = bittensor.Balance.from_float(5.0 * 3)

        mock_wallets = [
            SimpleNamespace(
                name = config.wallet.name,
                hotkey_str = config.hotkeys[0],
                hotkey = get_mock_hotkey(0),
                is_registered = MagicMock(
                    return_value = True
                ),

                _coldkey = mock_coldkey,
                coldkey =  MagicMock(
                            return_value=mock_coldkey
                        )
            ) 
        ] + [
            SimpleNamespace(
                name = config.wallet.name,
                hotkey_str = hk,
                hotkey = get_mock_hotkey(idx),
                is_registered = MagicMock(
                    return_value = True
                )
            ) for idx, hk in enumerate(config.hotkeys)
        ]

        # The 0th wallet is created twice during unstake
        mock_wallets[1]._coldkey = mock_coldkey
        mock_wallets[1].coldkey = MagicMock(
                                    return_value=mock_coldkey
                                )
        mock_wallets[1].get_balance = MagicMock(
                                    return_value=mock_balance
                                )
        
        cli = bittensor.cli(config)

        with patch('bittensor.wallet') as mock_create_wallet:
            mock_create_wallet.side_effect = mock_wallets
            with patch('bittensor.Subtensor.add_stake_multiple', return_value=True) as mock_add_stake:
                cli.run()
                mock_create_wallet.assert_has_calls(
                    [
                        call(config=ANY, hotkey=hk) for hk in config.hotkeys
                    ],
                    any_order=True
                )
                mock_add_stake.assert_has_calls(
                    [call(wallets=mock_wallets[1:], amounts=[5.0] * len(mock_wallets[1:]), wait_for_inclusion=True, prompt=False)],
                    any_order = True
                )

    def test_stake_with_all_hotkeys( self ):        
        config = self.config
        config.command = "stake"
        config.no_prompt = True 
        config.amount = 5.0
        config.wallet.name = "fake_wallet"
        # Notice wallet.hotkeys is not specified
        config.all_hotkey = True
        # Notice no max_stake specified
        

        mock_hotkeys = ['hk0', 'hk1', 'hk2']

        mock_coldkey = "" # Not None

        # enough to stake 5.0 to all 3 hotkeys
        mock_balance: Balance = bittensor.Balance.from_float(5.0 * 3)

        mock_wallets = [
            SimpleNamespace(
                name = config.wallet.name,
                hotkey_str = hk,
                hotkey = get_mock_hotkey(idx),
                is_registered = MagicMock(
                    return_value = True
                )
            ) for idx, hk in enumerate(mock_hotkeys)
        ]

        mock_wallets[0]._coldkey = mock_coldkey
        mock_wallets[0].coldkey = MagicMock(
                                    return_value=mock_coldkey
                                )
        mock_wallets[0].get_balance = MagicMock(
                                    return_value=mock_balance
                                )
        
        cli = bittensor.cli(config)

        with patch.object(cli, '_get_hotkey_wallets_for_wallet') as mock_get_all_wallets:
            mock_get_all_wallets.return_value = mock_wallets
            with patch('bittensor.Subtensor.add_stake_multiple', return_value=True) as mock_add_stake:
                cli.run()
                mock_get_all_wallets.assert_called_once()
                mock_add_stake.assert_has_calls(
                    [call(wallets=mock_wallets, amounts=[5.0] * len(mock_wallets), wait_for_inclusion=True, prompt=False)],
                    any_order = True
                )

    def test_stake_with_exclude_hotkeys_from_all( self ):        
        config = self.config
        config.command = "stake"
        config.no_prompt = True 
        config.amount = 5.0
        config.wallet.name = "fake_wallet"
        config.hotkeys = ['hk1'] # exclude hk1
        config.all_hotkey = True
        

        # Notice no max_stake specified

        mock_hotkeys = ['hk0', 'hk1', 'hk2']

        mock_coldkey = "" # Not None

        # enough to stake 5.0 to all 3 hotkeys
        mock_balance: Balance = bittensor.Balance.from_float(5.0 * 3)

        mock_wallets = [
            SimpleNamespace(
                name = config.wallet.name,
                hotkey_str = hk,
                hotkey = get_mock_hotkey(idx),
                is_registered = MagicMock(
                    return_value = True
                )
            ) for idx, hk in enumerate(mock_hotkeys)
        ]

        # The 0th wallet is created twice during unstake
        mock_wallets[0]._coldkey = mock_coldkey
        mock_wallets[0].coldkey = MagicMock(
                                    return_value=mock_coldkey
                                )
        mock_wallets[0].get_balance = MagicMock(
                                    return_value=mock_balance
                                )
        
        cli = bittensor.cli(config)

        with patch.object(cli, '_get_hotkey_wallets_for_wallet') as mock_get_all_wallets:
            mock_get_all_wallets.return_value = mock_wallets
            with patch('bittensor.Subtensor.add_stake_multiple', return_value=True) as mock_add_stake:
                cli.run()
                mock_get_all_wallets.assert_called_once()
                mock_add_stake.assert_has_calls(
                    [call(wallets=[mock_wallets[0], mock_wallets[2]], amounts=[5.0, 5.0], wait_for_inclusion=True, prompt=False)],
                    any_order = True
                )

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
        config.all_hotkey = False
        # Notice no max_stake specified
        

        mock_balance = bittensor.Balance(15.0 * 3) # Enough to stake 15.0 on each hotkey

        mock_coldkey = "" # Not None

        mock_stakes: Dict[str, float] = {
            # All have more than 5.0 stake
            'hk0': bittensor.Balance.from_float(10.0),
            'hk1': bittensor.Balance.from_float(11.1),
            'hk2': bittensor.Balance.from_float(12.2),
        }

        mock_wallets = [
            SimpleNamespace(
                name = config.wallet.name,
                hotkey_str = config.hotkeys[0],
                hotkey = get_mock_hotkey(0),
                get_stake = MagicMock(
                    return_value = mock_stakes[config.hotkeys[0]]
                ),
                is_registered = MagicMock(
                    return_value = True
                ),

                _coldkey = mock_coldkey,
                coldkey =  MagicMock(
                            return_value=mock_coldkey
                        )
            ) 
        ] + [
            SimpleNamespace(
                name = config.wallet.name,
                hotkey_str = hk,
                hotkey = get_mock_hotkey(idx),
                get_stake = MagicMock(
                    return_value = mock_stakes[hk]
                ),
                is_registered = MagicMock(
                    return_value = True
                )
            ) for idx, hk in enumerate(config.hotkeys)
        ]

        # The 0th wallet is created twice during unstake
        mock_wallets[1]._coldkey = mock_coldkey
        mock_wallets[1].coldkey = MagicMock(
                                    return_value=mock_coldkey
                                )
        mock_wallets[1].get_balance = MagicMock(
            return_value = mock_balance
        )
        
        cli = bittensor.cli(config)

        with patch('bittensor.wallet') as mock_create_wallet:
            mock_create_wallet.side_effect = mock_wallets
            with patch('bittensor.Subtensor.add_stake_multiple', return_value=True) as mock_add_stake:
                cli.run()
                mock_create_wallet.assert_has_calls(
                    [
                        call(config=ANY, hotkey=hk) for hk in config.hotkeys
                    ],
                    any_order=True
                )
                mock_add_stake.assert_has_calls(
                    [call(wallets=mock_wallets[1:], amounts=[CLOSE_IN_VALUE((config.max_stake - mock_stakes[mock_wallet.hotkey_str].tao), 0.001) for mock_wallet in mock_wallets[1:]], wait_for_inclusion=True, prompt=False)],
                    any_order = True
                )

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
        config.all_hotkey = False
        

        # Notice no max_stake specified

        mock_balance = bittensor.Balance(1.0) # Not enough to stake 15.0 on each hotkey

        mock_coldkey = "" # Not None

        mock_stakes: Dict[str, float] = {
            # All have more than 5.0 stake
            'hk0': bittensor.Balance.from_float(5.0),
            'hk1': bittensor.Balance.from_float(6.1),
            'hk2': bittensor.Balance.from_float(7.2),
        }

        mock_wallets = [
            SimpleNamespace(
                name = config.wallet.name,
                hotkey_str = config.hotkeys[0],
                hotkey = get_mock_hotkey(0),
                get_stake = MagicMock(
                    return_value = mock_stakes[config.hotkeys[0]]
                ),
                is_registered = MagicMock(
                    return_value = True
                ),

                _coldkey = mock_coldkey,
                coldkey =  MagicMock(
                            return_value=mock_coldkey
                        )
            ) 
        ] + [
            SimpleNamespace(
                name = config.wallet.name,
                hotkey_str = hk,
                hotkey = get_mock_hotkey(idx),
                get_stake = MagicMock(
                    return_value = mock_stakes[hk]
                ),
                is_registered = MagicMock(
                    return_value = True
                )
            ) for idx, hk in enumerate(config.hotkeys)
        ]

        # The 0th wallet is created twice during unstake
        mock_wallets[1]._coldkey = mock_coldkey
        mock_wallets[1].coldkey = MagicMock(
                                    return_value=mock_coldkey
                                )
        mock_wallets[1].get_balance = MagicMock(
            return_value = mock_balance
        )
        
        cli = bittensor.cli(config)

        with patch('bittensor.wallet') as mock_create_wallet:
            mock_create_wallet.side_effect = mock_wallets
            with patch('bittensor.Subtensor.add_stake', return_value=True) as mock_add_stake:
                cli.run()
                mock_create_wallet.assert_has_calls(
                    [
                        call(config=ANY, hotkey=hk) for hk in config.hotkeys
                    ],
                    any_order=True
                )
                # We should stake what we have in the balance
                mock_add_stake.assert_called_once()
                
                total_staked = 0.0

                args, kwargs = mock_add_stake.call_args
                total_staked = kwargs['amount']
                
                # We should not try to stake more than the mock_balance
                self.assertAlmostEqual(total_staked, mock_balance.tao, delta=0.001)

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
        config.all_hotkey = False
        # Notice no max_stake specified
        

        mock_balance = bittensor.Balance(15.0) # Enough to stake 15.0 on one hotkey

        mock_coldkey = "" # Not None

        mock_stakes: Dict[str, float] = {
            # All have more than 5.0 stake
            'hk0': bittensor.Balance.from_float(10.0),
        }

        mock_wallets = [
            SimpleNamespace(
                name = config.wallet.name,
                hotkey_str = config.hotkeys[0],
                hotkey = get_mock_hotkey(0),
                get_stake = MagicMock(
                    return_value = mock_stakes[config.hotkeys[0]]
                ),
                is_registered = MagicMock(
                    return_value = True
                ),

                _coldkey = mock_coldkey,
                coldkey =  MagicMock(
                            return_value=mock_coldkey
                        )
            ) 
        ] + [
            SimpleNamespace(
                name = config.wallet.name,
                hotkey_str = hk,
                hotkey = get_mock_hotkey(idx),
                get_stake = MagicMock(
                    return_value = mock_stakes[hk]
                ),
                is_registered = MagicMock(
                    return_value = True
                )
            ) for idx, hk in enumerate(config.hotkeys)
        ]

        # The 0th wallet is created twice during unstake
        mock_wallets[1]._coldkey = mock_coldkey
        mock_wallets[1].coldkey = MagicMock(
                                    return_value=mock_coldkey
                                )
        mock_wallets[1].get_balance = MagicMock(
            return_value = mock_balance
        )
        
        cli = bittensor.cli(config)

        with patch('bittensor.wallet') as mock_create_wallet:
            mock_create_wallet.side_effect = mock_wallets
            with patch('bittensor.Subtensor.add_stake', return_value=True) as mock_add_stake:
                cli.run()
                mock_create_wallet.assert_has_calls(
                    [
                        call(config=ANY, hotkey=hk) for hk in config.hotkeys
                    ],
                    any_order=True
                )
                mock_add_stake.assert_has_calls(
                    [call(wallet=mock_wallets[1], amount=CLOSE_IN_VALUE((config.max_stake - mock_stakes[mock_wallets[1].hotkey_str].tao), 0.001), wait_for_inclusion=True, prompt=False)],
                    any_order = True
                )

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
        config.all_hotkey = False
        

        # Notice no max_stake specified

        mock_balance = bittensor.Balance(1.0) # Not enough to stake 15.0 on the hotkey

        mock_coldkey = "" # Not None

        mock_stakes: Dict[str, float] = {
            # has 5.0 stake
            'hk0': bittensor.Balance.from_float(5.0)
        }

        mock_wallets = [
            SimpleNamespace(
                name = config.wallet.name,
                hotkey_str = config.hotkeys[0],
                hotkey = get_mock_hotkey(0),
                get_stake = MagicMock(
                    return_value = mock_stakes[config.hotkeys[0]]
                ),
                is_registered = MagicMock(
                    return_value = True
                ),

                _coldkey = mock_coldkey,
                coldkey =  MagicMock(
                            return_value=mock_coldkey
                        )
            ) 
        ] + [
            SimpleNamespace(
                name = config.wallet.name,
                hotkey_str = hk,
                hotkey = get_mock_hotkey(idx),
                get_stake = MagicMock(
                    return_value = mock_stakes[hk]
                ),
                is_registered = MagicMock(
                    return_value = True
                )
            ) for idx, hk in enumerate(config.hotkeys)
        ]

        # The 0th wallet is created twice during unstake
        mock_wallets[1]._coldkey = mock_coldkey
        mock_wallets[1].coldkey = MagicMock(
                                    return_value=mock_coldkey
                                )
        mock_wallets[1].get_balance = MagicMock(
            return_value = mock_balance
        )
        
        cli = bittensor.cli(config)

        with patch('bittensor.wallet') as mock_create_wallet:
            mock_create_wallet.side_effect = mock_wallets
            with patch('bittensor.Subtensor.add_stake', return_value=True) as mock_add_stake:
                cli.run()
                mock_create_wallet.assert_has_calls(
                    [
                        call(config=ANY, hotkey=hk) for hk in config.hotkeys
                    ],
                    any_order=True
                )
                # We should stake what we have in the balance
                mock_add_stake.assert_called_once()
                
                total_staked = 0.0

                args, kwargs = mock_add_stake.call_args
                total_staked = kwargs['amount']
                
                # We should not try to stake more than the mock_balance
                self.assertAlmostEqual(total_staked, mock_balance.tao, delta=0.001)

    def test_stake_with_single_hotkey_max_stake_enough_stake( self ):
        # tests max stake when stake >= max_stake already
        bittensor.subtensor.register = MagicMock(return_value = True)
        
        config = self.config
        config.command = "stake"
        config.no_prompt = True 
        # Notie amount is not specified
        config.max_stake = 15.0 # The keys should have at most 15.0 tao staked after
        config.wallet.name = "fake_wallet"
        config.hotkeys = [
            'hk0'
        ]   
        config.all_hotkey = False
        

        # Notice no max_stake specified

        mock_balance = bittensor.Balance(30.0) # enough to stake 15.0 on the hotkey

        mock_coldkey = "" # Not None

        mock_stakes: Dict[str, float] = {
            # already has 15.0 stake
            'hk0': bittensor.Balance.from_float(15.0)
        }

        mock_wallets = [
            SimpleNamespace(
                name = config.wallet.name,
                hotkey_str = config.hotkeys[0],
                hotkey = get_mock_hotkey(config.hotkeys[0]),
                get_stake = MagicMock(
                    return_value = mock_stakes[config.hotkeys[0]]
                ),
                is_registered = MagicMock(
                    return_value = True
                ),

                _coldkey = mock_coldkey,
                coldkey =  MagicMock(
                            return_value=mock_coldkey
                        )
            ) 
        ] + [
            SimpleNamespace(
                name = config.wallet.name,
                hotkey_str = hk,
                hotkey = get_mock_hotkey(idx),
                get_stake = MagicMock(
                    return_value = mock_stakes[hk]
                ),
                is_registered = MagicMock(
                    return_value = True
                )
            ) for idx, hk in enumerate(config.hotkeys)
        ]

        # The 0th wallet is created twice during unstake
        mock_wallets[1]._coldkey = mock_coldkey
        mock_wallets[1].coldkey = MagicMock(
                                    return_value=mock_coldkey
                                )
        mock_wallets[1].get_balance = MagicMock(
            return_value = mock_balance
        )
        
        cli = bittensor.cli(config)

        with patch('bittensor.wallet') as mock_create_wallet:
            mock_create_wallet.side_effect = mock_wallets
            with patch('bittensor.Subtensor.add_stake', return_value=True) as mock_add_stake:
                cli.run()
                mock_create_wallet.assert_has_calls(
                    [
                        call(config=ANY, hotkey=hk) for hk in config.hotkeys
                    ],
                    any_order=True
                )
                # We should stake what we have in the balance
                mock_add_stake.assert_not_called()
                
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
        config = self.config
        config.no_prompt = True
        config.command = "stake"
        config.amount = 0.5
        config.stake_all = False
        config.wallet._mock = True
        config.use_password = False
        config.model = "core_server"

        subtensor = bittensor.subtensor(config)

        mock_wallet = generate_wallet()
        with patch('bittensor.wallet', return_value=mock_wallet) as mock_create_wallet:
            
            old_stake = subtensor.get_stake_for_coldkey_and_hotkey(
                hotkey_ss58 = mock_wallet.hotkey.ss58_address,
                coldkey_ss58 = mock_wallet.coldkey.ss58_address,
            )

            cli = bittensor.cli(config)
            cli.run()
            mock_create_wallet.assert_called_once()
            
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