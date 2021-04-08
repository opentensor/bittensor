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
            config: Munch = None,
            wallet: 'bittensor.Wallet' = None,
            subtensor: 'bittensor.Subtensor' = None,
            metagraph: 'bittensor.Metagraph' = None,
        ):
        r""" Initializes a new Executor.
        """
        # config for the wallet, metagraph sub-objects.
        if config == None:
            config = Executor.default_config()
        Executor.check_config( config )
        self.config = config

        if wallet == None:
            wallet = bittensor.Wallet( config = self.config )
        self.config.wallet = wallet.config.wallet
        self.wallet = wallet

        if subtensor == None:
            subtensor = bittensor.Subtensor( config = self.config, wallet = self.wallet )
        self.config.subtensor = subtensor.config.subtensor
        self.subtensor = subtensor

        if metagraph == None:
            metagraph = bittensor.Metagraph( subtensor = self.subtensor )
        self.metagraph = metagraph
        self.metagraph.subtensor = subtensor
    
    @staticmethod
    def default_config () -> Munch:
        parser = argparse.ArgumentParser(); 
        Executor.add_args(parser) 
        config = bittensor.Config.to_config(parser); 
        return config

    @staticmethod   
    def add_args (parser: argparse.ArgumentParser):
        bittensor.Wallet.add_args( parser )
        bittensor.Subtensor.add_args( parser )
        
    @staticmethod   
    def check_config (config: Munch):
        bittensor.Wallet.check_config(config)
        bittensor.Subtensor.check_config(config)
            
    def regenerate_coldkey ( self, mnemonic: str, use_password:bool ):
        r""" Regenerates a colkey under this wallet.
        """
        self.wallet.regenerate_coldkey( mnemonic = mnemonic, use_password = use_password )

    def regenerate_hotkey ( self, mnemonic: str, use_password:bool ):
        r""" Regenerates a hotkey under this wallet.
        """
        self.wallet.regenerate_hotkey( mnemonic = mnemonic, use_password = use_password)

    def create_new_coldkey ( self, n_words:int, use_password:bool ):
        r""" Creates a new coldkey under this wallet.
        """
        self.wallet.create_new_coldkey( n_words = n_words, use_password = use_password )   

    def create_new_hotkey ( self, n_words:int, use_password:bool ):  
        r""" Creates a new hotkey under this wallet.
        """
        self.wallet.create_new_hotkey( n_words = n_words, use_password = use_password )  

    def _associated_neurons( self ) -> Neurons:
        r""" Returns a list of neurons associate with this wallet's coldkey.
        """
        bittensor.__cli_logger__.log("USER-ACTION", 'Retrieving all nodes associated with coldkey: {}'.format( self.wallet.coldkeypub ))
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
        self.subtensor.connect()
        self.metagraph.sync()
        balance = self.subtensor.get_balance( self.wallet.coldkey.ss58_address )
        neurons = self._associated_neurons()

        print("BALANCE: %s : [%s]" % ( self.wallet.coldkey.ss58_address, "\u03C4" + str(balance.tao) ))
        print()
        print("--===[[ STAKES ]]===--")
        t = PrettyTable(["UID", "IP", "STAKE (\u03C4)", "RANK (\u03C4)", "INCENTIVE (\u03C4)"])
        t.align = 'l'
        total_stake = 0.0
        S = self.metagraph.S()
        R = self.metagraph.R()
        I = self.metagraph.I()
        for neuron in neurons:
            stake = float(S[neuron.uid])
            rank = float(R[neuron.uid])
            incentive = float(I[neuron.uid])
            t.add_row([neuron.uid, neuron.ip, "\u03C4" + str(neuron.stake.__float__()) , "\u03C4" + str(rank / pow(10, 9)), "\u03C4" + str(incentive)])
            total_stake += neuron.stake.__float__()
        print(t.get_string())
        print("Total stake: ", "\u03C4" + str(total_stake))

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
                bittensor.__cli_logger__.success("Unstaked: {} Tao from uid: {} to coldkey.pub: {}".format( neuron.stake, neuron.uid, self.wallet.coldkey.public_key ))
            else:
                bittensor.__cli_logger__.critical("Unstaking transaction failed")

    def unstake( self, amount_int:int, uid:int ):
        r""" Unstaked token of amount to from uid.
        """
        self.wallet.assert_coldkey()
        self.wallet.assert_coldkeypub()
        self.subtensor.connect()
        amount_balance = Balance.from_float( amount_int )
        neurons = self._associated_neurons()
        neuron = neurons.get_by_uid( uid )
        if not neuron:
            bittensor.__cli_logger__.critical("Neuron with uid: {} is not associated with coldkey.pub: {}".format( uid, self.wallet.coldkey.public_key))
            quit()

        neuron.stake = self.subtensor.get_stake_for_uid(neuron.uid)
        if amount_balance > neuron.stake:
            bittensor.__cli_logger__.critical("Neuron with uid: {} does not have enough stake ({}) to be able to unstake {}".format( uid, neuron.stake, amount_balance ))
            quit()

        bittensor.__cli_logger__.log('USER-ACTION', "Requesting unstake of {} rao for hotkey: {} to coldkey: {}".format(amount_balance.rao, neuron.hotkey, self.wallet.coldkey.public_key))
        bittensor.__cli_logger__.log('USER', "Waiting for finalization...")
        result = self.subtensor.unstake(amount_balance, neuron.hotkey, wait_for_finalization = True, timeout = bittensor.__blocktime__ * 5)
        if result:
            bittensor.__cli_logger__.success("Unstaked:{} from uid:{} to coldkey.pub:{}".format(amount_balance.tao, neuron.uid, self.wallet.coldkey.public_key))
        else:
            bittensor.__cli_logger__.critical("<Unstaking transaction failed")

    def stake( self, amount_int: int, uid: int ):
        r""" Stakes token of amount to hotkey uid.
        """
        self.wallet.assert_coldkey()
        self.wallet.assert_coldkeypub()
        self.subtensor.connect()
        amount_balance = Balance.from_float( amount_int )
        balance = self.subtensor.get_balance( self.wallet.coldkey.ss58_address )
        if balance < amount_balance:
            bittensor.__cli_logger__.critical("Not enough balance ({}) to stake {}".format(balance, amount_balance))
            quit()

        neurons = self._associated_neurons()
        neuron = neurons.get_by_uid( uid )
        if not neuron:
            bittensor.__cli_logger__.critical("Neuron with uid: {} is not associated with coldkey.pub: {}".format( uid, self.wallet.coldkey.public_key))
            quit()

        bittensor.__cli_logger__.log('USER-ACTION', "Adding stake of {} rao from coldkey {} to hotkey {}".format(amount_balance.rao, self.wallet.coldkey.public_key, neuron.hotkey))
        bittensor.__cli_logger__.log('USER', "Waiting for finalization...")
        result = self.subtensor.add_stake( amount_balance, neuron.hotkey, wait_for_finalization = True, timeout = bittensor.__blocktime__ * 5)
        if result:
            bittensor.__cli_logger__.success("Staked: {} Tao to uid: {} from coldkey.pub: {}".format(amount_balance.tao, uid, self.wallet.coldkey.public_key))
        else:
            bittensor.__cli_logger__.critical("Stake transaction failed")

    def transfer( self, ammount_int: int, destination: str ):
        r""" Transfers token of amount to dest.
            
        """
        self.wallet.assert_coldkey()
        self.wallet.assert_coldkeypub()
        self.subtensor.connect()
        amount_balance = Balance.from_float( ammount_int )
        balance = self.subtensor.get_balance(self.wallet.coldkey.ss58_address)
        if balance < amount_balance:
            bittensor.__cli_logger__.critical("Not enough balance ({}) to transfer {}".format(balance, amount_balance))
            quit()

        bittensor.__cli_logger__.log('USER-ACTION', 'Requesting transfer of {}, from coldkey: {} to dest: {}'.format(amount_balance.rao, self.wallet.coldkey.public_key, destination))
        bittensor.__cli_logger__.log('USER', 'Waiting for finalization...')
        result = self.subtensor.transfer( destination, amount_balance,  wait_for_finalization = True, timeout = bittensor.__blocktime__ * 5 )
        if result:
            bittensor.__cli_logger__.success("Transfer finalized with amount: {} Tao to dest: {} from coldkey.pub: {}".format(amount_balance.tao, destination, self.wallet.coldkey.public_key))
        else:
            bittensor.__cli_logger__.critical("Transfer failed")
 
 
if __name__ == "__main__":
    # ---- Build and Run ----
    config = Executor.default_config(); 
    bittensor.__cli_logger__.log('USER', bittensor.Config.toString(config))
    executor = Executor( config )
    executor.run_command()
