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
import sys
import os
import pandas as pd

from munch import Munch
from loguru import logger
from termcolor import colored
from prettytable import PrettyTable

import bittensor
from bittensor.utils.neurons import Neuron, Neurons
from bittensor.utils.balance import Balance

class Executor:

    def __init__(   
            self, 
            config: 'Munch' = None, 
            wallet: 'bittensor.wallet.Wallet' = None,
            subtensor: 'bittensor.subtensor.Subtensor' = None,
            metagraph: 'bittensor.metagraph.Metagraph' = None,
            **kwargs,
        ):
        r""" Initializes a new Metagraph chain interface.
            Args:
                config (:obj:`Munch`, `optional`): 
                    neuron.Neuron.config()
                wallet (:obj:`bittensor.wallet.Wallet`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
                subtensor (:obj:`bittensor.subtensor.Subtensor`, `optional`):
                    subtensor interface utility.
                metagraph (:obj:`bittensor.metagraph.Metagraph`, `optional`):
                    bittensor metagraph object.
        """
        # config for the wallet, metagraph sub-objects.
        if config == None:
            config = Executor.default_config()
        Executor.check_config( config )
        self.config = config

        if wallet == None:
            wallet = bittensor.wallet.Wallet( config = self.config )
        self.config.wallet = wallet.config.wallet
        self.wallet = wallet

        if subtensor == None:
            subtensor = bittensor.subtensor.Subtensor( config = self.config, wallet = self.wallet )
        self.config.subtensor = subtensor.config.subtensor
        self.subtensor = subtensor

        if metagraph == None:
            metagraph = bittensor.metagraph.Metagraph( subtensor = self.subtensor )
        self.metagraph = metagraph
        self.metagraph.subtensor = subtensor     

    @staticmethod
    def default_config () -> Munch:
        parser = argparse.ArgumentParser(); 
        Executor.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        return config

    @staticmethod   
    def add_args (parser: argparse.ArgumentParser):
        bittensor.wallet.Wallet.add_args( parser )
        bittensor.subtensor.Subtensor.add_args( parser )
        
    @staticmethod   
    def check_config (config: Munch):
        bittensor.wallet.Wallet.check_config(config)
        bittensor.subtensor.Subtensor.check_config(config)
            
    def regenerate_coldkey ( self, mnemonic: str, use_password: bool ):
        r""" Regenerates a colkey under this wallet.
        """
        self.wallet.regenerate_coldkey( mnemonic = mnemonic, use_password = use_password )

    def regenerate_hotkey ( self, mnemonic: str, use_password: bool ):
        r""" Regenerates a hotkey under this wallet.
        """
        self.wallet.regenerate_hotkey( mnemonic = mnemonic, use_password = use_password )

    def create_new_coldkey ( self, n_words: int, use_password: bool ):
        r""" Creates a new coldkey under this wallet.
        """
        self.wallet.create_new_coldkey( n_words = n_words, use_password = use_password )   

    def create_new_hotkey ( self, n_words: int, use_password: bool ):  
        r""" Creates a new hotkey under this wallet.
        """
        self.wallet.create_new_hotkey( n_words = n_words, use_password = use_password  )  

    def _associated_neurons( self ) -> Neurons:
        r""" Returns a list of neurons associate with this wallet's coldkey.
        """
        bittensor.__logger__.log('USER-ACTION', "Retrieving all nodes associated with cold key : {}".format( self.wallet.coldkeypub ))
        neurons = self.subtensor.neurons()
        neurons = Neurons.from_list( neurons )
        result = filter(lambda x : x.coldkey == self.wallet.coldkey.public_key, neurons )# These are the neurons associated with the provided cold key
        associated_neurons = Neurons(result)
        # Load stakes
        for neuron in associated_neurons:
            neuron.stake = self.subtensor.get_stake_for_uid(neuron.uid)
        return associated_neurons

    def overview ( self ): 
        r""" Prints an overview for the wallet's colkey.
        """
        self.wallet.assert_coldkey()
        self.wallet.assert_coldkeypub()
        self.subtensor.connect()
        self.metagraph.sync()
        balance = self.subtensor.get_balance( self.wallet.coldkey.ss58_address )
        neurons = self._associated_neurons()

        bittensor.__logger__.log('USER-SUCCESS', "BALANCE: %s : [\u03C4%s]" % ( self.wallet.coldkey.ss58_address, balance.tao ))
        bittensor.__logger__.log('USER-INFO', "")
        bittensor.__logger__.log('USER-INFO', "--===[[ Neurons ]]===--")
        t = PrettyTable(["UID", "IP", "STAKE (\u03C4) ", "RANK  (\u03C4)", "INCENTIVE  (\u03C4/block) ", "LastEmit (blocks)", "HOTKEY"])
        t.align = 'l'
        total_stake = 0.0
        for neuron in neurons:
            stake = float(self.metagraph.S[neuron.uid])
            rank = float(self.metagraph.R[neuron.uid])
            incentive = float(self.metagraph.I[neuron.uid])
            lastemit = int(self.metagraph.block - self.metagraph.lastemit[neuron.uid])
            t.add_row([neuron.uid, neuron.ip, stake, rank, incentive, lastemit, neuron.hotkey])
            total_stake += neuron.stake.__float__()
        bittensor.__logger__.log('USER-INFO', t.get_string())
        bittensor.__logger__.log('USER-SUCCESS', "Total stake: {}", total_stake)

    def unstake_all ( self ):
        r""" Unstaked from all hotkeys associated with this wallet's coldkey.
        """
        self.wallet.assert_coldkey()
        self.wallet.assert_coldkeypub()
        self.subtensor.connect()
        neurons = self._associated_neurons()
        for neuron in neurons:
            neuron.stake = self.subtensor.get_stake_for_uid( neuron.uid )
            result = self.subtensor.unstake( neuron.stake, neuron.hotkey, wait_for_finalization = True, timeout = bittensor.__blocktime__ * 5)
            if result:
                bittensor.__logger__.log('USER-SUCCESS', "Unstaked: \u03C4{} from uid: {} to coldkey.pub: {}".format( neuron.stake, neuron.uid, self.wallet.coldkey.public_key ))
            else:
                bittensor.__logger__.log('USER-CRITICAL', "Unstaking transaction failed")

    def unstake( self, amount_tao: int, uid: int ):
        r""" Unstaked token of amount to from uid.
        """
        self.wallet.assert_coldkey()
        self.wallet.assert_coldkeypub()
        self.subtensor.connect()
        unstaking_balance = Balance.from_float( amount_tao )
        neurons = self._associated_neurons()
        neuron = neurons.get_by_uid( uid )
        if not neuron:
            bittensor.__logger__.log('USER-CRITICAL', "Neuron with uid: {} is not associated with coldkey.pub: {}".format( uid, self.wallet.coldkey.public_key))
            quit()

        neuron.stake = self.subtensor.get_stake_for_uid(neuron.uid)
        if unstaking_balance > neuron.stake:
            bittensor.__logger__.log('USER-CRITICAL', "Neuron with uid: {} does not have enough stake ({}) to be able to unstake {}".format( uid, neuron.stake, unstaking_balance))
            quit()

        bittensor.__logger__.log('USER-ACTION', "Requesting unstake of \u03C4{} from hotkey: {} to coldkey: {}".format(unstaking_balance.tao, neuron.hotkey, self.wallet.coldkey.public_key))
        bittensor.__logger__.log('USER-INFO', "Waiting for finalization...")
        result = self.subtensor.unstake(unstaking_balance, neuron.hotkey, wait_for_finalization = True, timeout = bittensor.__blocktime__ * 5)
        if result:
            bittensor.__logger__.log('USER-SUCCESS', "Unstaked: \u03C4{} from uid:{} to coldkey.pub:{}".format(unstaking_balance.tao, neuron.uid, self.wallet.coldkey.public_key))
        else:
            bittensor.__logger__.log('USER-CRITICAL', "Unstaking transaction failed")

    def stake( self, amount_tao: int, uid: int ):
        r""" Stakes token of amount to hotkey uid.
        """
        self.wallet.assert_coldkey()
        self.wallet.assert_coldkeypub()
        self.subtensor.connect()
        staking_balance = Balance.from_float( amount_tao )
        account_balance = self.subtensor.get_balance( self.wallet.coldkey.ss58_address )
        if account_balance < staking_balance:
            bittensor.__logger__.log('USER-CRITICAL', "Not enough balance (\u03C4{}) to stake \u03C4{}".format(account_balance, staking_balance))
            quit()

        neurons = self._associated_neurons()
        neuron = neurons.get_by_uid( uid )
        if not neuron:
            bittensor.__logger__.log('USER-CRITICAL', "Neuron with uid: {} is not associated with coldkey.pub: {}".format( uid, self.wallet.coldkey.public_key ))
            quit()

        bittensor.__logger__.log('USER-ACTION', "Adding stake of \u03C4{} from coldkey {} to hotkey {}".format( staking_balance.tao, self.wallet.coldkey.public_key, neuron.hotkey))
        bittensor.__logger__.log('USER-INFO', "Waiting for finalization...")
        result = self.subtensor.add_stake( staking_balance, neuron.hotkey, wait_for_finalization = True, timeout = bittensor.__blocktime__ * 5)
        if result: 
            bittensor.__logger__.log('USER-SUCCESS', "Staked: \u03C4{} to uid: {} from coldkey.pub: {}".format( staking_balance.tao, uid, self.wallet.coldkey.public_key ))
        else:
            bittensor.__logger__.log('USER-CRITICAL', "Stake transaction failed")

    def transfer( self, amount_tao: int, destination: str):
        r""" Transfers token of amount to dest.
            
        """
        self.wallet.assert_coldkey()
        self.wallet.assert_coldkeypub()
        self.subtensor.connect()
        transfer_balance = Balance.from_float( amount_tao )
        acount_balance = self.subtensor.get_balance(self.wallet.coldkey.ss58_address)
        if acount_balance < transfer_balance:
            bittensor.__logger__.log('USER-CRITICAL', "Not enough balance (\u03C4{}) to transfer \u03C4{}".format(acount_balance, transfer_balance))
            quit()

        bittensor.__logger__.log('USER-ACTION', "Requesting transfer of \u03C4{}, from coldkey: {} to destination: {}".format(transfer_balance.tao, self.wallet.coldkey.public_key, destination))
        bittensor.__logger__.log('USER-INFO', "Waiting for finalization...")
        result = self.subtensor.transfer( destination, transfer_balance,  wait_for_finalization = True, timeout = bittensor.__blocktime__ * 5 )
        if result:
            bittensor.__logger__.log('USER-SUCCESS', "Transfer finalized with amount: \u03C4{} to destination: {} from coldkey.pub: {}".format(transfer_balance.tao, destination, self.wallet.coldkey.public_key))
        else:
            bittensor.__logger__.log('USER-CRITICAL', "Transfer failed")
 
