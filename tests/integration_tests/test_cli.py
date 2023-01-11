# The MIT License (MIT)
# Copyright © 2022 Yuma Rao
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


import unittest
from types import SimpleNamespace
from typing import Dict
from unittest.mock import ANY, MagicMock, call, patch
import pytest

import bittensor
import substrateinterface
from bittensor._subtensor.subtensor_mock import mock_subtensor
from bittensor.utils.balance import Balance
from substrateinterface.base import Keypair
from tests.helpers import CLOSE_IN_VALUE


class TestCli(unittest.TestCase):

    def setUp(self):
        class success():
            def __init__(self):
                self.is_success = True
                self.value = 1
            def process_events(self):
                return True

        self.config = TestCli.construct_config()
        # Mocked objects
        self.mock_neuron = TestCli._neuron_dict_to_namespace(
            dict({
                "version":1,
                "ip":0,
                "port":0,
                "ip_type":0,
                "uid":1,
                "modality":0,
                "hotkey":'some_hotkey',
                "coldkey":'some_coldkey',
                "active":0,
                "last_update":0,
                "priority":0,
                "stake":1000000000000.0,
                "rank":0.0,
                "trust":0.0,
                "consensus":0.0,
                "incentive":0.0,
                "dividends":0.0,
                "emission":0.0,
                "bonds":[],
                "weights":[],
                "is_null":False
            })
        )
        bittensor.Subtensor.register = MagicMock(return_value = True) 
        bittensor.Subtensor.neuron_for_pubkey = MagicMock(return_value=self.mock_neuron)
        bittensor.Subtensor.neuron_for_uid = MagicMock(return_value=self.mock_neuron)
        substrateinterface.SubstrateInterface.submit_extrinsic = MagicMock(return_value = success()) 
        substrateinterface.SubstrateInterface.query = MagicMock(return_value=success())
        substrateinterface.SubstrateInterface.get_block_hash = MagicMock(return_value='0x')
        bittensor.Subtensor.get_balance = MagicMock(return_value = Balance.from_tao(0))

    @staticmethod
    def construct_config():
        defaults = bittensor.Config()
        bittensor.subtensor.add_defaults( defaults )
        bittensor.dendrite.add_defaults( defaults )
        bittensor.axon.add_defaults( defaults )
        bittensor.wallet.add_defaults( defaults )
        bittensor.dataset.add_defaults( defaults )
        
        return defaults

    @staticmethod
    def _neuron_dict_to_namespace(neuron_dict) -> SimpleNamespace:
            if neuron_dict['hotkey'] == '5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM':
                return bittensor.subtensor._null_neuron()
            else:
                RAOPERTAO = 1000000000
                U64MAX = 18446744073709551615
                neuron = SimpleNamespace( **neuron_dict )
                neuron.total_stake = Balance.from_rao(neuron.stake)
                neuron.rank = neuron.rank / U64MAX
                neuron.trust = neuron.trust / U64MAX
                neuron.consensus = neuron.consensus / U64MAX
                neuron.incentive = neuron.incentive / U64MAX
                neuron.dividends = neuron.dividends / U64MAX
                neuron.emission = neuron.emission / RAOPERTAO
                neuron.is_null = False
                return neuron

    @staticmethod
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

    def test_check_configs(self):
        commands = ["run", "transfer", "register", "unstake", 
        "stake", "overview", "new_coldkey", "new_hotkey", 
        "regen_coldkey", "regen_hotkey", "metagraph", "weights", 
        "set_weights", "inspect"]
        config = self.config
        config.no_prompt = True
        config.model = "core_server"
        config.dest = "no_prompt"
        config.amount = 1
        config.mnemonic = "this is a mnemonic"
        config.seed = None
        config.uids = [1,2,3]
        config.weights = [0.25, 0.25, 0.25, 0.25]
        config.no_version_checking = False

        cli = bittensor.cli
        
        for cmd in commands:
            config.command = cmd
            cli.check_config(config)

    def test_overview( self ):
        # Mock IO for wallet
        with patch('bittensor.Wallet.coldkeypub_file', MagicMock(
            exists_on_device=MagicMock(
                return_value=True # Wallet exists
            )
        )):
            bittensor.subtensor.register = MagicMock(return_value = True)  
            
            config = self.config
            config.wallet.path = '/tmp/test_cli_test_overview'
            config.wallet.name = 'mock_wallet'
            config.command = "overview"
            config.no_cache = True  # Don't use neuron cache
            config.no_prompt = True
            config.all = False
            config.no_version_checking = False

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
            config.no_version_checking = False

            cli = bittensor.cli(config)
            cli.run()

    def test_overview_with_cache( self ):
        config = self.config
        config.command = "overview"
        config.no_cache = False # Use neuron cache
        config.no_prompt = True
        config.all = False
        config.no_version_checking = False

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_with_cache_cache_fails( self ):
        config = self.config
        config.command = "overview"
        config.no_cache = False # Use neuron cache
        config.no_prompt = True
        config.all = False
        config.no_version_checking = False

        with patch('bittensor.Metagraph.retrieve_cached_neurons') as mock_retrieve_cached_neurons:
            # Mock the cache retrieval to fail
            mock_retrieve_cached_neurons.side_effect = Exception("Cache failed")

            # Should not raise an exception
            cli = bittensor.cli(config)
            cli.run()

    def test_overview_without_no_cache_confg( self ):        
        config = self.config
        config.command = "overview"
        # Don't specify no_cache in config
        config.no_prompt = True
        config.all = False
        config.no_version_checking = False

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_with_hotkeys_config( self ):        
        config = self.config
        config.command = "overview"
        config.no_prompt = True
        config.wallet.hotkeys = ['some_hotkey']
        config.all = False
        config.no_version_checking = False

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_without_hotkeys_config( self ):        
        config = self.config
        config.command = "overview"
        config.no_prompt = True
        config.all = False
        config.no_version_checking = False

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_with_sort_by_config( self ):        
        config = self.config
        config.command = "overview"
        config.no_prompt = True
        config.wallet.sort_by = "rank"
        config.all = False
        config.no_version_checking = False

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_with_sort_by_bad_column_name( self ):        
        config = self.config
        config.command = "overview"
        config.no_prompt = True
        config.wallet.sort_by = "totallynotmatchingcolumnname"
        config.all = False
        config.no_version_checking = False

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_without_sort_by_config( self ):        
        config = self.config
        config.command = "overview"
        config.no_prompt = True
        config.all = False
        config.no_version_checking = False

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_with_sort_order_config( self ):        
        config = self.config
        config.command = "overview"
        config.wallet.sort_order = "desc" # Set descending sort order
        config.no_prompt = True
        config.all = False
        config.no_version_checking = False

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_with_sort_order_config_bad_sort_type( self ):        
        config = self.config
        config.command = "overview"
        config.wallet.sort_order = "nowaythisshouldmatchanyorderingchoice" 
        config.no_prompt = True
        config.all = False
        config.no_version_checking = False

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_without_sort_order_config( self ):        
        config = self.config
        config.command = "overview"
        # Don't specify sort_order in config
        config.no_prompt = True
        config.all = False
        config.no_version_checking = False

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_with_width_config( self ):        
        config = self.config
        config.command = "overview"
        config.width = 100
        config.no_prompt = True
        config.all = False
        config.no_version_checking = False

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_without_width_config( self ):        
        config = self.config
        config.command = "overview"
        # Don't specify width in config
        config.no_prompt = True
        config.all = False
        config.no_version_checking = False

        cli = bittensor.cli(config)
        cli.run()

    def test_overview_all( self ):
        config = self.config
        config.command = "overview"
        config.no_prompt = True
        config.no_version_checking = False

        config.all = True
        cli = bittensor.cli(config)
        cli.run()

    def test_unstake_with_specific_hotkeys( self ):        
        config = self.config
        config.command = "unstake"
        config.no_prompt = True 
        config.amount = 5.0
        config.wallet.name = "fake_wallet"
        config.wallet.hotkeys = [
            'hk0', 'hk1', 'hk2'
        ]   
        config.wallet.all_hotkeys = False
        # Notice no max_stake specified
        config.no_version_checking = False

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
                hotkey_str = config.wallet.hotkeys[0],
                get_stake = MagicMock(
                    return_value = mock_stakes[config.wallet.hotkeys[0]]
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
                get_stake = MagicMock(
                    return_value = mock_stakes[hk]
                ),
                is_registered = MagicMock(
                    return_value = True
                )
            ) for hk in config.wallet.hotkeys
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
                        call(config=ANY, hotkey=hk) for hk in config.wallet.hotkeys
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
        config.wallet.all_hotkeys = True
        # Notice no max_stake specified
        config.no_version_checking = False

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
                get_stake = MagicMock(
                    return_value = mock_stakes[hk]
                ),
                is_registered = MagicMock(
                    return_value = True
                )
            ) for hk in list(mock_stakes.keys())
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
        config.wallet.hotkeys = ["hk1"] # Exclude hk1
        config.wallet.all_hotkeys = True
        # Notice no max_stake specified
        config.no_version_checking = False

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
                get_stake = MagicMock(
                    return_value = mock_stakes[hk]
                ),
                is_registered = MagicMock(
                    return_value = True
                )
            ) for hk in list(mock_stakes.keys())
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
        config.wallet.hotkeys = [
            'hk0', 'hk1', 'hk2'
        ]   
        config.wallet.all_hotkeys = False
        # Notice no max_stake specified
        config.no_version_checking = False

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
                hotkey_str = config.wallet.hotkeys[0],
                get_stake = MagicMock(
                    return_value = mock_stakes[config.wallet.hotkeys[0]]
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
                get_stake = MagicMock(
                    return_value = mock_stakes[hk]
                ),
                is_registered = MagicMock(
                    return_value = True
                )
            ) for hk in config.wallet.hotkeys
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
                        call(config=ANY, hotkey=hk) for hk in config.wallet.hotkeys
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
        config.wallet.hotkeys = [
            'hk0', 'hk1', 'hk2'
        ]   
        config.wallet.all_hotkeys = False
        # Notice no max_stake specified
        config.no_version_checking = False

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
                hotkey_str = config.wallet.hotkeys[0],
                get_stake = MagicMock(
                    return_value = mock_stakes[config.wallet.hotkeys[0]]
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
                get_stake = MagicMock(
                    return_value = mock_stakes[hk]
                ),
                is_registered = MagicMock(
                    return_value = True
                )
            ) for hk in config.wallet.hotkeys
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
                        call(config=ANY, hotkey=hk) for hk in config.wallet.hotkeys
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
        config.wallet.hotkeys = [
            'hk0', 'hk1', 'hk2'
        ]   
        config.wallet.all_hotkeys = False
        # Notice no max_stake specified
        config.no_version_checking = False

        mock_coldkey = "" # Not None

        # enough to stake 5.0 to all 3 hotkeys
        mock_balance: Balance = bittensor.Balance.from_float(5.0 * 3)

        mock_wallets = [
            SimpleNamespace(
                name = config.wallet.name,
                hotkey_str = config.wallet.hotkeys[0],
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
                is_registered = MagicMock(
                    return_value = True
                )
            ) for hk in config.wallet.hotkeys
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
                        call(config=ANY, hotkey=hk) for hk in config.wallet.hotkeys
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
        config.wallet.all_hotkeys = True
        # Notice no max_stake specified
        config.no_version_checking = False

        mock_hotkeys = ['hk0', 'hk1', 'hk2']

        mock_coldkey = "" # Not None

        # enough to stake 5.0 to all 3 hotkeys
        mock_balance: Balance = bittensor.Balance.from_float(5.0 * 3)

        mock_wallets = [
            SimpleNamespace(
                name = config.wallet.name,
                hotkey_str = hk,
                is_registered = MagicMock(
                    return_value = True
                )
            ) for hk in mock_hotkeys
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
        config.wallet.hotkeys = ['hk1'] # exclude hk1
        config.wallet.all_hotkeys = True
        config.no_version_checking = False

        # Notice no max_stake specified

        mock_hotkeys = ['hk0', 'hk1', 'hk2']

        mock_coldkey = "" # Not None

        # enough to stake 5.0 to all 3 hotkeys
        mock_balance: Balance = bittensor.Balance.from_float(5.0 * 3)

        mock_wallets = [
            SimpleNamespace(
                name = config.wallet.name,
                hotkey_str = hk,
                is_registered = MagicMock(
                    return_value = True
                )
            ) for hk in mock_hotkeys
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
        config.wallet.hotkeys = [
            'hk0', 'hk1', 'hk2'
        ]   
        config.wallet.all_hotkeys = False
        # Notice no max_stake specified
        config.no_version_checking = False

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
                hotkey_str = config.wallet.hotkeys[0],
                get_stake = MagicMock(
                    return_value = mock_stakes[config.wallet.hotkeys[0]]
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
                get_stake = MagicMock(
                    return_value = mock_stakes[hk]
                ),
                is_registered = MagicMock(
                    return_value = True
                )
            ) for hk in config.wallet.hotkeys
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
                        call(config=ANY, hotkey=hk) for hk in config.wallet.hotkeys
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
        config.wallet.hotkeys = [
            'hk0', 'hk1', 'hk2'
        ]   
        config.wallet.all_hotkeys = False
        config.no_version_checking = False

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
                hotkey_str = config.wallet.hotkeys[0],
                get_stake = MagicMock(
                    return_value = mock_stakes[config.wallet.hotkeys[0]]
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
                get_stake = MagicMock(
                    return_value = mock_stakes[hk]
                ),
                is_registered = MagicMock(
                    return_value = True
                )
            ) for hk in config.wallet.hotkeys
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
                        call(config=ANY, hotkey=hk) for hk in config.wallet.hotkeys
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
        config.wallet.hotkeys = [
            'hk0'
        ]   
        config.wallet.all_hotkeys = False
        # Notice no max_stake specified
        config.no_version_checking = False

        mock_balance = bittensor.Balance(15.0) # Enough to stake 15.0 on one hotkey

        mock_coldkey = "" # Not None

        mock_stakes: Dict[str, float] = {
            # All have more than 5.0 stake
            'hk0': bittensor.Balance.from_float(10.0),
        }

        mock_wallets = [
            SimpleNamespace(
                name = config.wallet.name,
                hotkey_str = config.wallet.hotkeys[0],
                get_stake = MagicMock(
                    return_value = mock_stakes[config.wallet.hotkeys[0]]
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
                get_stake = MagicMock(
                    return_value = mock_stakes[hk]
                ),
                is_registered = MagicMock(
                    return_value = True
                )
            ) for hk in config.wallet.hotkeys
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
                        call(config=ANY, hotkey=hk) for hk in config.wallet.hotkeys
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
        config.wallet.hotkeys = [
            'hk0'
        ]   
        config.wallet.all_hotkeys = False
        config.no_version_checking = False

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
                hotkey_str = config.wallet.hotkeys[0],
                get_stake = MagicMock(
                    return_value = mock_stakes[config.wallet.hotkeys[0]]
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
                get_stake = MagicMock(
                    return_value = mock_stakes[hk]
                ),
                is_registered = MagicMock(
                    return_value = True
                )
            ) for hk in config.wallet.hotkeys
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
                        call(config=ANY, hotkey=hk) for hk in config.wallet.hotkeys
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
        config.wallet.hotkeys = [
            'hk0'
        ]   
        config.wallet.all_hotkeys = False
        config.no_version_checking = False

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
                hotkey_str = config.wallet.hotkeys[0],
                get_stake = MagicMock(
                    return_value = mock_stakes[config.wallet.hotkeys[0]]
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
                get_stake = MagicMock(
                    return_value = mock_stakes[hk]
                ),
                is_registered = MagicMock(
                    return_value = True
                )
            ) for hk in config.wallet.hotkeys
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
                        call(config=ANY, hotkey=hk) for hk in config.wallet.hotkeys
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
        config.no_version_checking = False

        with patch('bittensor.Subtensor.register', return_value=True):
            cli = bittensor.cli(config)
            cli.run()
            
    def test_stake( self ):
        config = self.config
        config.no_prompt = True
        config.command = "stake"
        config.amount = 0.5
        config.stake_all = False
        config.wallet._mock = True
        config.use_password = False
        config.no_version_checking = False

        config.model = "core_server"

        cli = bittensor.cli(config)
        cli.run()

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
        config.no_version_checking = False

        cli = bittensor.cli(config)
        cli.run()

    def test_new_hotkey( self ):
        
        #subtensor.register(wallet=wallet)    
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
        config.no_version_checking = False

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
        config.no_version_checking = False

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
        config.no_version_checking = False

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
        config.no_version_checking = False

        cli = bittensor.cli(config)
        cli.run()

    def test_metagraph( self ):    
        config = self.config
        config.wallet.name = "metagraph_testwallet"
        config.command = "metagraph"
        config.no_prompt = True
        config.no_version_checking = False

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
        config.no_version_checking = False


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
        config.no_version_checking = False

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
            config.no_version_checking = False

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
            config.no_version_checking = False

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
        assert 'overview' in help_out
        assert 'run' in help_out


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
class TestCLIUsingArgs(unittest.TestCase):
    """
    Test the CLI by passing args directly to the bittensor.cli factory
    """
    def test_run_reregister_false(self):
        """
        Verify that the btcli run command does not reregister a not registered wallet
            if --wallet.reregister is False
        """

        with patch('bittensor.Wallet.is_registered', MagicMock(return_value=False)) as mock_wallet_is_reg: # Wallet is not registered
            with patch('bittensor.Subtensor.register', MagicMock(side_effect=Exception("shouldn't register during test"))):
                with pytest.raises(SystemExit):
                    cli = bittensor.cli(args=[
                        'run',
                        '--wallet.name', 'mock',
                        '--wallet.hotkey', 'mock_hotkey',
                        '--wallet._mock', 'True',
                        '--no_prompt',
                        '--wallet.reregister', 'False' # Don't reregister
                    ])
                    cli.run()

                    args, kwargs = mock_wallet_is_reg.call_args
                    # args[0] should be self => the wallet
                    assert args[0].config.wallet.reregister == False

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

                assert len(args) == 0 and len(kwargs) == 0 # should not have any args; indicates that "All" synapses are being used


if __name__ == '__main__':
    unittest.main()