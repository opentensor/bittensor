# The MIT License (MIT)
# Copyright © 2021 Opentensor.ai

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
import argparse
import random

from munch import Munch
from loguru import logger

import bittensor
import bittensor.utils.networking as net
from bittensor.substrate.base import SubstrateInterface, Keypair
from bittensor.utils.neurons import Neuron, Neurons
from bittensor.utils.balance import Balance

class Subtensor:
    custom_type_registry = {
        "runtime_id": 2,
        "types": {
            "NeuronMetadataOf": {
                "type": "struct",
                "type_mapping": [["ip", "u128"], ["port", "u16"], ["ip_type", "u8"], ["uid", "u64"], ["modality", "u8"], ["hotkey", "AccountId"], ["coldkey", "AccountId"]]
            }
        }
    }

    def __init__(self, config: 'Munch' = None, wallet: 'bittensor.wallet.Wallet' = None):
        r""" Initializes a new Metagraph chain interface.
            Args:
                config (:obj:`Munch`, `optional`): 
                    metagraph.Metagraph.config()
                wallet (:obj:`bittensor.wallet.Wallet`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
        """
        if config == None:
            config = Subtensor.build_config()
        self.config = config

        if wallet == None:
            wallet = bittensor.wallet.Wallet( self.config )
        self.wallet = wallet

        self.substrate = SubstrateInterface(
            url = self.config.subtensor.chain_endpoint,
            address_type = 42,
            type_registry_preset='substrate-node-template',
            type_registry=self.custom_type_registry,
        )

    @staticmethod
    def build_config() -> Munch:
        # Parses and returns a config Munch for this object.
        parser = argparse.ArgumentParser(); 
        Subtensor.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        Subtensor.check_config(config)
        return config

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        bittensor.wallet.Wallet.add_args( parser )
        try:
            parser.add_argument('--subtensor.chain_endpoint', default=None, type=str, 
                                help='''The subtensor chain endpoint. The likely choices are:
                                        -- localhost:9944 -- (your locally running node)
                                        -- feynman.akira.bittensor.com:9944 (testnet)
                                        -- feynman.kusanagi.bittensor.com:12345 (mainnet)
                                    If subtensor.network is set it is overloaded by subtensor.network.
                                    ''')
            parser.add_argument('--subtensor.network', default=None, type=str, 
                                help='''The subtensor network flag. The likely choices are:
                                        -- akira (testing network)
                                        -- kusanagi (main network)
                                    If this option is set it overloads subtensor.chain_endpoint with 
                                    an entry point node from that network.
                                    ''')
        except:
            pass
        
    @staticmethod   
    def check_config(config: Munch):
        bittensor.wallet.Wallet.check_config( config )

        # Neither are set, default to akira.
        if config.subtensor.network == None and config.subtensor.chain_endpoint == None:
            logger.info('Defaulting to network: akira')
            config.subtensor.network = 'akira'

        # Switch based on network config item. 
        if config.subtensor.network != None:
            all_networks = ['akira', 'boltzmann', 'kusanagi']
            assert config.subtensor.network in all_networks, 'metagraph.network == {} not one of {}'.format(config.subtensor.network, all_networks)
            if config.subtensor.network == "akira":
                config.subtensor.chain_endpoint = random.choice(bittensor.__akira_entrypoints__)
            elif config.subtensor.network == "boltzmann":
                config.subtensor.chain_endpoint = random.choice(bittensor.__boltzmann_entrypoints__)
            elif config.subtensor.network == "kusanagi":
                config.subtensor.chain_endpoint = random.choice(bittensor.__kusanagi_entrypoints__)
            else:
                raise ValueError('metagraph.network == {} not one of {}'.format(config.subtensor.network, all_networks))

        # The chain endpoint it set.
        elif config.subtensor.chain_endpoint != None:
            all_entrypoints = bittensor.__akira_entrypoints__ + bittensor.__boltzmann_entrypoints__ + bittensor.__kusanagi_entrypoints__
            if not config.subtensor.chain_endpoint in all_entrypoints:
                logger.info('metagraph.chain_endpoint == {}, NOTE: not one of {}', config.subtensor.chain_endpoint, all_entrypoints)

    def connect(self):
        logger.trace("connect() C")
        self.substrate.connect()

    def is_connected(self):
        return self.substrate.is_connected()

    def subscribe(self, ip: str, port: int, modality: int, coldkey: str):
        ip_as_int  = net.ip_to_int(ip)
        params = {
            'ip': ip_as_int,
            'port': port, 
            'ip_type': 4,
            'modality': modality,
            'coldkey': coldkey
        }

        call = self.substrate.compose_call(
            call_module='SubtensorModule',
            call_function='subscribe',
            call_params=params
        )

        extrinsic = self.substrate.create_signed_extrinsic(call=call, keypair=self.wallet.hotkey.public_key)
        self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)  # Waiting for inclusion and other does not work

    def get_balance(self, address):
        logger.debug("Getting balance for: {}", address)
        result = self.substrate.get_runtime_state(
            module='System',
            storage_function='Account',
            params=[address],
            block_hash=None
        )
        print (result)

        balance_info = result.get('result')
        if not balance_info:
            logger.debug("{} has no balance", address)
            return Balance(0)

        balance = balance_info['data']['free']
        logger.debug("{} has {} rao", address, balance)
        return Balance(balance)

    def add_stake(self, amount : Balance, hotkey_id):
        logger.info("Adding stake of {} rao from coldkey {} to hotkey {}", amount.rao, self.wallet.coldkey.public_key, hotkey_id)
        call = self.substrate.compose_call(
            call_module='SubtensorModule',
            call_function='add_stake',
            call_params={
                'hotkey': hotkey_id,
                'ammount_staked': amount.rao
            }
        )
        extrinsic = self.substrate.create_signed_extrinsic(call=call, keypair=self.wallet.coldkey)
        result = self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)
        return result

    def transfer(self, dest:str, amount: Balance):
        logger.debug("Requesting transfer of {}, from coldkey: {} to dest: {}", amount.rao, self.wallet.coldkey.public_key, dest)
        call = self.substrate.compose_call(
            call_module='Balances',
            call_function='transfer',
            call_params={
                'dest': dest, 
                'value': amount.rao
            }
        )
        extrinsic = self.substrate.create_signed_extrinsic(call=call, keypair = self.wallet.coldkey)
        result = self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)
        return result

    def unstake(self, amount : Balance, hotkey_id):
        logger.debug("Requesting unstake of {} rao for hotkey: {} to coldkey: {}", amount.rao, hotkey_id, self.wallet.coldkey.public_key)
        call = self.substrate.compose_call(
            call_module='SubtensorModule',
            call_function='remove_stake',
            call_params={'ammount_unstaked': amount.rao, 'hotkey': hotkey_id}
        )

        extrinsic = self.substrate.create_signed_extrinsic(call=call, keypair=self.wallet.coldkey)
        result = self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)
        return result

    def set_weights(self, destinations, values, wait_for_inclusion=False):
        call = self.substrate.compose_call(
            call_module='SubtensorModule',
            call_function='set_weights',
            call_params = {'dests': destinations, 'weights': values}
        )
        extrinsic = self.substrate.create_signed_extrinsic(call=call, keypair = self.wallet.hotkey)
        result = self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=wait_for_inclusion)
        return result

    def get_current_block(self):
        return self.substrate.get_block_number(None)

    def get_active(self) -> int:
        result =self.substrate.iterate_map(
            module='SubtensorModule',
            storage_function='Active',
        )
        return result

    def get_uid_for_pubkey(self, pubkey = str) -> int:
        result = self.substrate.get_runtime_state(
            module='SubtensorModule',
            storage_function='Active',
            params=[pubkey]
        )
        if result['result'] is None:
            return None
        return int(result['result'])

    def get_neuron_for_uid(self, uid:int):
        result = self.substrate.get_runtime_state(
                module='SubtensorModule',
                storage_function='Neurons',
                params=[uid]
        )
        return result['result']
    
    def neurons(self, uid=None, decorator=False):

        # Todo (parall4x, 23-12-2020) Get rid of this decorator flag. This should be refactored into something that returns Neuron objects only
        if uid:
            result = self.substrate.get_runtime_state(
                module='SubtensorModule',
                storage_function='Neurons',
                params=[uid]
            )
            return Neurons.from_list(result['result']) if decorator else result['result']

        else:
            neurons = self.substrate.iterate_map(
                module='SubtensorModule',
                storage_function='Neurons'
            )
            return Neurons.from_list(neurons) if decorator else neurons

    def get_stake_for_uid(self, uid) -> Balance:
        stake = self.substrate.get_runtime_state(
            module='SubtensorModule',
            storage_function='Stake',
            params = [uid]
        )
        result = stake['result']
        if not result:
            return Balance(0)
        return Balance(result)

    def weight_uids_for_uid(self, uid):
        result = self.substrate.get_runtime_state(
            module='SubtensorModule',
            storage_function='WeightUids',
            params = [uid]
        )
        return result['result']

    def weight_vals_for_uid(self, uid):
        result = self.substrate.get_runtime_state(
            module='SubtensorModule',
            storage_function='WeightVals',
            params = [uid]
        )
        return result['result']

    def get_last_emit_data_for_uid(self, uid):
        result = self.substrate.get_runtime_state(
            module='SubtensorModule',
            storage_function='LastEmit',
            params = [uid]
        )
        return result['result']

    def get_last_emit_data(self):
        result = self.substrate.iterate_map(
            module='SubtensorModule',
            storage_function='LastEmit'
        )
        return result