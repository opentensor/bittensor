
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
import json
import os
import re
import stat
import traceback as tb

from io import StringIO
from munch import Munch
from termcolor import colored
from loguru import logger
logger = logger.opt(ansi=True)
from cryptography.exceptions import InvalidSignature, InvalidKey
from cryptography.fernet import InvalidToken

import bittensor

class FailedConnectToChain(Exception):
    pass

class FailedSubscribeToChain(Exception):
    pass

class FailedToEnterNeuron(Exception):
    pass

class FailedToPollChain(Exception):
    pass

class Neuron:
    def __init__(self, 
                config: Munch = None, 
                wallet: 'bittensor.wallet.Wallet' = None, 
                subtensor: 'bittensor.subtensor.Subtensor' = None,
                metagraph: 'bittensor.metagraph.Metagraph' = None,
                nucleus: 'bittensor.nucleus.Nucleus' = None,
                axon: 'bittensor.axon.Axon' = None,
                dendrite: 'bittensor.dendrite.Dendrite' = None,
                **kwargs,
            ):
        r""" Initializes a new full Neuron object.
            
            Args:
                config (:obj:`Munch`, `optional`): 
                    neuron.Neuron.default_config()
                wallet (:obj:`bittensor.wallet.Wallet`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
                subtensor (:obj:`bittensor.subtensor.Subtensor`, `optional`):
                    subtensor interface utility.
                metagraph (:obj:`bittensor.metagraph.Metagraph`, `optional`):
                    bittensor network metagraph.
                nucleus (:obj:`bittensor.nucleus.Nucleus`, `optional`):
                    backend processing nucleus.
                axon (:obj:`bittensor.axon.Axon`, `optional`):
                    synapse serving endpoint.
                dendrite (:obj:`bittensor.dendrite.Dendrite`, `optional`):
                    synapse connecting object. 
                modality (default=0, type=int)
                    Neuron network modality. TEXT=0, IMAGE=1. Currently only allowed TEXT
        """
        # Config: Config items for all subobjects: wallet, metagraph, nucleus, axon, dendrite.
        # This object can be instantiated by calling Neuron.default_config()
        if config == None:
            config = Neuron.default_config()
        bittensor.config.Config.update_with_kwargs(config.neuron, kwargs) 
        Neuron.check_config(config)
        self.config = config
    #     # Wallet: Holds the hotkey keypair and coldkey pub which are user to sign messages 
    #     # and subscribe to the chain.
    #     if wallet == None:
    #         wallet = bittensor.wallet.Wallet (self.config )
    #     self.wallet = wallet
    #     # Subtensor: provides an interface to the subtensor chain given a wallet.
    #     if subtensor == None:
    #         subtensor = bittensor.subtensor.Subtensor( self.config )
    #     self.subtensor = subtensor
    #     # Metagraph: Maintains a connection to the subtensor chain and hold chain state.
    #     if metagraph == None:
    #         metagraph = bittensor.metagraph.Metagraph(config = self.config, wallet = self.wallet, subtensor = self.subtensor)
    #     self.metagraph = metagraph
    #     # Nucleus: Processes requests passed to this neuron on its axon endpoint.
    #     if nucleus == None:
    #         nucleus = bittensor.nucleus.Nucleus(config = self.config, wallet = self.wallet )
    #     self.nucleus = nucleus
    #     # Axon: RPC server endpoint which serves your synapse. Responds to Forward and Backward requests.
    #     if axon == None:
    #         axon = bittensor.axon.Axon(config = self.config, wallet = self.wallet, nucleus = self.nucleus )
    #     self.axon = axon
    #     # Dendrite: RPC client makes Forward and Backward requests to downstream peers.
    #     if dendrite == None:
    #         dendrite = bittensor.dendrite.Dendrite(config = self.config, wallet = self.wallet )
    #     self.dendrite = dendrite

    # @staticmethod   
    # def default_config() -> Munch:
    #     # Parses and returns a config Munch for this object.
    #     parser = argparse.ArgumentParser(); 
    #     Neuron.add_args(parser) 
    #     config = bittensor.config.Config.to_config(parser); 
    #     return config

    # @staticmethod   
    # def add_args(parser: argparse.ArgumentParser):
    #     bittensor.wallet.Wallet.add_args( parser )
    #     bittensor.subtensor.Subtensor.add_args( parser )
    #     bittensor.metagraph.Metagraph.add_args( parser )
    #     bittensor.nucleus.Nucleus.add_args( parser )
    #     bittensor.axon.Axon.add_args(parser)
    #     bittensor.dendrite.Dendrite.add_args( parser )
    #     bittensor.synapse.Synapse.add_args( parser )
    #     try:
    #         parser.add_argument('--neuron.modality', default=0, type=int, 
    #                             help='''Neuron network modality. TEXT=0, IMAGE=1. Currently only allowed TEXT''')
    #     except:
    #         pass

    # @staticmethod   
    # def check_config(config: Munch):
    #     assert config.neuron.modality == bittensor.proto.Modality.TEXT, 'Only TEXT modalities are allowed at this time.'

    # def start(self):
        
    #     # ---- Start the axon ----
    #     self.axon.start()

    #     # ---- Subscribe to chain ----
    #     logger.log('USER-ACTION', '\nConnecting to network: {}'.format(self.config.subtensor.network))
    #     self.subtensor.connect()

    #     logger.log('USER-ACTION', '\nSubscribing:')
    #     subscribe_success = self.subtensor.subscribe(
    #             wallet = self.wallet,
    #             ip = self.config.axon.external_ip, 
    #             port = self.config.axon.external_port,
    #             modality = self.config.neuron.modality,
    #             wait_for_finalization = True,
    #             timeout = 4 * bittensor.__blocktime__,
    #     )
    #     if not subscribe_success:
    #         self.stop()
    #         raise RuntimeError('Failed to subscribe neuron.')
        
    #     # ---- Sync graph ----
    #     self.metagraph.sync()
    #     print(self.metagraph)

    #     self.run()

    # def run(self):


    # def stop(self):
    #     logger.info('Shutting down the Axon server ...')
    #     try:
    #         self.axon.stop()
    #         logger.info('Axon server stopped')
    #     except Exception as e:
    #         logger.error('Neuron: Error while stopping axon server: {} ', e)

    # def __del__(self):
    #     self.stop()

