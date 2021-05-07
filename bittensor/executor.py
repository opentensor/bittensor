# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

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
import copy
import sys
import os
import pandas as pd

from munch import Munch
from termcolor import colored
from prettytable import PrettyTable

import bittensor
from bittensor.utils.neurons import NeuronEndpoint, NeuronEndpoints
from bittensor.utils.balance import Balance

from loguru import logger
logger = logger.opt(colors=True)

class Executor ( bittensor.neuron.Neuron ):

    def __init__(   
            self, 
            config: 'Munch' = None, 
            **kwargs,
        ):
        r""" Initializes a new Metagraph chain interface.
            Args:
                config (:obj:`Munch`, `optional`): 
                    neuron.Neuron.config()
        """
        # config for the wallet, metagraph sub-objects.
        if config == None:
            config = Executor.default_config()
        config = copy.deepcopy(config); bittensor.config.Config.update_with_kwargs(config, kwargs )
        Executor.check_config( config )
        self.config = config
        super(Executor, self).__init__( self.config, **kwargs )

    @staticmethod
    def default_config () -> Munch:
        parser = argparse.ArgumentParser(); 
        Executor.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        return config

    @staticmethod   
    def add_args (parser: argparse.ArgumentParser):
        bittensor.neuron.Neuron.add_args(parser)
        
    @staticmethod   
    def check_config (config: Munch):
        bittensor.neuron.Neuron.check_config( config )
            
    def regenerate_coldkey ( self, mnemonic: str, use_password: bool, overwrite: bool = False ):
        r""" Regenerates a colkey under this wallet.
        """
        self.wallet.regenerate_coldkey( mnemonic = mnemonic, use_password = use_password, overwrite = overwrite )

    def regenerate_hotkey ( self, mnemonic: str, use_password: bool, overwrite: bool = False):
        r""" Regenerates a hotkey under this wallet.
        """
        self.wallet.regenerate_hotkey( mnemonic = mnemonic, use_password = use_password, overwrite = overwrite )

    def create_new_coldkey ( self, n_words: int, use_password: bool, overwrite: bool = False ):
        r""" Creates a new coldkey under this wallet.
        """
        self.wallet.create_new_coldkey( n_words = n_words, use_password = use_password, overwrite = overwrite)   

    def create_new_hotkey ( self, n_words: int, use_password: bool, overwrite: bool = False ):  
        r""" Creates a new hotkey under this wallet.
        """
        self.wallet.create_new_hotkey( n_words = n_words, use_password = use_password, overwrite = overwrite )  

    def overview ( self ): 
        r""" Prints an overview for the wallet's colkey.
        """
        self.wallet.assert_coldkey()
        self.wallet.assert_coldkeypub()
        self.connect_to_chain()
        self.load_metagraph()
        self.sync_metagraph()
        self.save_metagraph()
        balance = self.subtensor.get_balance( self.wallet.coldkey.ss58_address )
        
        owned_neurons = [] 
        neuron_endpoints = self.metagraph.neuron_endpoints
        for uid, cold in enumerate(self.metagraph.coldkeys):
            if cold == self.wallet.coldkey.public_key:
                owned_neurons.append( neuron_endpoints[uid] )
                
        logger.opt(raw=True).info("--===[[ Neurons ]]===--\n")
        t = PrettyTable(["UID", "IP", "STAKE (\u03C4) ", "RANK  (\u03C4)", "INCENTIVE  (\u03C4/block) ", "LastEmit (blocks)", "HOTKEY"])
        t.align = 'l'
        total_stake = 0.0
        for neuron in owned_neurons:
            uid = neuron.uid
            stake = self.metagraph.S[ uid ].item()
            rank = self.metagraph.R[ uid ].item()
            incentive = self.metagraph.I[ uid ].item()
            lastemit = int(self.metagraph.block - self.metagraph.lastemit[ uid ])
            t.add_row([neuron.uid, neuron.ip, stake, rank, incentive, lastemit, neuron.hotkey])
            total_stake += stake
        logger.opt(raw=True).info(t.get_string() + '\n')
        logger.success( "Total staked: \u03C4{}", total_stake)
        logger.success( "Total balance: \u03C4{}", balance.tao)
        
    def unstake_all ( self ):
        r""" Unstaked from all hotkeys associated with this wallet's coldkey.
        """
        self.wallet.assert_coldkey()
        self.wallet.assert_coldkeypub()
        self.connect_to_chain()
        neurons = self._associated_neurons()
        for neuron in neurons:
            neuron.stake = self.subtensor.get_stake_for_uid( neuron.uid )
            result = self.subtensor.unstake( 
                wallet = self.wallet, 
                amount = neuron.stake, 
                hotkey_id = neuron.hotkey, 
                wait_for_finalization = True, 
                timeout = bittensor.__blocktime__ * 5 
            )
            if result:
                logger.success( "Unstaked: \u03C4{} from uid: {} to coldkey.pub: {}".format( neuron.stake, neuron.uid, self.wallet.coldkey.public_key ))
            else:
                logger.critical("Unstaking transaction failed")

    def unstake( self, amount_tao: int, uid: int ):
        r""" Unstaked token of amount to from uid.
        """
        self.wallet.assert_coldkey()
        self.wallet.assert_coldkeypub()
        self.connect_to_chain()
        unstaking_balance = Balance.from_float( amount_tao )
        neurons = self._associated_neurons()
        neuron = neurons.get_by_uid( uid )
        if not neuron:
            logger.critical("Neuron with uid: {} is not associated with coldkey.pub: {}".format( uid, self.wallet.coldkey.public_key))
            quit()

        neuron.stake = self.subtensor.get_stake_for_uid(neuron.uid)
        if unstaking_balance > neuron.stake:
            logger.critical("Neuron with uid: {} does not have enough stake ({}) to be able to unstake {}".format( uid, neuron.stake, unstaking_balance))
            quit()

        logger.info("Requesting unstake of \u03C4{} from hotkey: {} to coldkey: {}".format(unstaking_balance.tao, neuron.hotkey, self.wallet.coldkey.public_key))
        logger.info("Waiting for finalization...")
        result = self.subtensor.unstake (
            wallet = self.wallet, 
            amount = unstaking_balance, 
            hotkey_id = neuron.hotkey, 
            wait_for_finalization = True, 
            timeout = bittensor.__blocktime__ * 5
        )
        if result:
            logger.success("Unstaked: \u03C4{} from uid:{} to coldkey.pub:{}".format(unstaking_balance.tao, neuron.uid, self.wallet.coldkey.public_key))
        else:
            logger.critical("Unstaking transaction failed")

    def stake( self, amount_tao: int, uid: int ):
        r""" Stakes token of amount to hotkey uid.
        """
        self.wallet.assert_coldkey()
        self.wallet.assert_coldkeypub()
        self.subtensor.connect()
        staking_balance = Balance.from_float( amount_tao )
        account_balance = self.subtensor.get_balance( self.wallet.coldkey.ss58_address )
        if account_balance < staking_balance:
            logger.critical("Not enough balance (\u03C4{}) to stake \u03C4{}".format(account_balance, staking_balance))
            quit()

        neurons = self._associated_neurons()
        neuron = neurons.get_by_uid( uid )
        if not neuron:
            logger.critical("Neuron with uid: {} is not associated with coldkey.pub: {}".format( uid, self.wallet.coldkey.public_key ))
            quit()

        logger.info("Adding stake of \u03C4{} from coldkey {} to hotkey {}".format( staking_balance.tao, self.wallet.coldkey.public_key, neuron.hotkey))
        logger.info("Waiting for finalization...")
        result = self.subtensor.add_stake ( 
            wallet = self.wallet, 
            amount = staking_balance, 
            hotkey_id = neuron.hotkey, 
            wait_for_finalization = True, 
            timeout = bittensor.__blocktime__ * 5
        )
        if result: 
            logger.success("Staked: \u03C4{} to uid: {} from coldkey.pub: {}".format( staking_balance.tao, uid, self.wallet.coldkey.public_key ))
        else:
            logger.critical("Stake transaction failed")

    def transfer( self, amount_tao: int, destination: str):
        r""" Transfers token of amount to dest.
            
        """
        self.wallet.assert_coldkey()
        self.wallet.assert_coldkeypub()
        self.subtensor.connect()
        transfer_balance = Balance.from_float( amount_tao )
        acount_balance = self.subtensor.get_balance(self.wallet.coldkey.ss58_address)
        if acount_balance < transfer_balance:
            logger.critical("Not enough balance (\u03C4{}) to transfer \u03C4{}".format(acount_balance, transfer_balance))
            quit()

        logger.info("Requesting transfer of \u03C4{}, from coldkey: {} to destination: {}".format(transfer_balance.tao, self.wallet.coldkey.public_key, destination))
        logger.info("Waiting for finalization...")
        result = self.subtensor.transfer( 
            wallet = self.wallet, 
            dest = destination, 
            amount = transfer_balance,  
            wait_for_finalization = True, 
            timeout = bittensor.__blocktime__ * 5 
        )
        if result:
            logger.success("Transfer finalized with amount: \u03C4{} to destination: {} from coldkey.pub: {}".format(transfer_balance.tao, destination, self.wallet.coldkey.public_key))
        else:
            logger.critical("Transfer failed")
 
