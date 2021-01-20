'''
The MIT License (MIT)
Copyright © 2021 Opentensor.ai

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.
'''

import argparse
import json
import os
import re
import stat
import traceback as tb

from io import StringIO
from munch import Munch
from loguru import logger
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
                metagraph: 'bittensor.metagraph.Metagraph' = None,
                nucleus: 'bittensor.nucleus.Nucleus' = None,
                axon: 'bittensor.axon.Axon' = None,
                dendrite: 'bittensor.dendrite.Dendrite' = None,
            ):
        r""" Initializes a new full Neuron object.
            Args:
                config (:obj:`Munch`, `optional`): 
                    neuron.Neuron.config()
                wallet (:obj:`bittensor.nucleus.Nucleus`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
                metagraph (:obj:`bittensor.metagraph.Metagraph`, `optional`):
                    bittensor network metagraph.
                nucleus (:obj:`bittensor.nucleus.Nucleus`, `optional`):
                    backend processing nucleus.
                axon (:obj:`bittensor.axon.Axon`, `optional`):
                    synapse serving endpoint.
                dendrite (:obj:`bittensor.dendrite.Dendrite`, `optional`):
                    synapse connecting object. 
        """
        # Config: Config items for all subobjects: wallet, metagraph, nucleus, axon, dendrite.
        # This object can be instantiated by calling Neuron.build_config()
        if config == None:
            config = Neuron.build_config()
        self.config = config
        # Wallet: Holds the hotkey keypair and coldkey pub which are user to sign messages 
        # and subscribe to the chain.
        if wallet == None:
            wallet = bittensor.wallet.Wallet(self.config)
        # Metagraph: Maintains a connection to the subtensor chain and hold chain state.
        if metagraph == None:
            metagraph = bittensor.metagraph.Metagraph(config = self.config, wallet = wallet)
        self.metagraph = metagraph
        # Nucleus: Processes requests passed to this neuron on its axon endpoint.
        if nucleus == None:
            nucleus = bittensor.nucleus.Nucleus(config = self.config, wallet = wallet, metagraph = self.metagraph)
        self.nucleus = nucleus
        # Axon: RPC server endpoint which serves your synapse. Responde to Forward and Backward requests.
        if axon == None:
            axon = bittensor.axon.Axon(config = self.config, wallet = wallet, nucleus = self.nucleus, metagraph = self.metagraph)
        self.axon = axon
        # Dendrite: RPC client makes Forward and Backward requests to downstream peers.
        if dendrite == None:
            dendrite = bittensor.dendrite.Dendrite(config = self.config, wallet = wallet, metagraph = self.metagraph)
        self.dendrite = dendrite

    def serve(self, synapse: 'bittensor.synapse.Synapse'):
        r""" Serves a synapse.Synapse to the axon server replacing the previous synapse.Synapse if exists.

            Args:
                synapse (:obj:`bittensor.synapse.Synapse`, `required`): 
                    synapse object to serve on the axon server.
        """
        self.axon.serve(synapse)

    def start(self):
        # Stop background grpc threads for serving the synapse object.
        try:
            self.axon.start()
            logger.info('Started Axon server')
        except Exception as e:
            logger.error('Neuron: Failed to start axon server with error: {}', e)
            raise FailedToEnterNeuron

        try:
            code, message = self.metagraph.connect(timeout=12)
            if code != bittensor.metagraph.Metagraph.ConnectSuccess:
                logger.error('Neuron: Timeout while subscribing to the chain endpoint with message {}', message)
                logger.error('Check that your internet connection is working and the chain endpoint {} is available', self.config.metagraph.chain_endpoint)
                failed_connect_msg_help = ''' The subtensor chain endpoint should likely be one of the following choices:
                                        -- localhost:9944 -- (your locally running node)
                                        -- feynman.akira.bittensor.com:9944 (testnet)
                                        -- feynman.kusanagi.bittensor.com:9944 (mainnet)
                                    To connect to your local node you will need to run the subtensor locally: (See: docs/running_a_validator.md)
                                '''
                logger.error(failed_connect_msg_help)
                raise FailedConnectToChain
        except Exception as e:
            logger.error('Neuron: Error while connecting to the chain endpoint: {}', e)
            raise FailedToEnterNeuron

        try:
            code, message = self.metagraph.subscribe(timeout=12)
            if code != bittensor.metagraph.Metagraph.SubscribeSuccess:
                logger.error('Neuron: Error while subscribing to the chain endpoint with message: {}', message)
                raise FailedToEnterNeuron

        except Exception as e:
            logger.error('Neuron: Error while subscribing to the chain endpoint: {}', e)
            raise FailedToEnterNeuron

        try:
            self.metagraph.sync()
            logger.info(self.metagraph)
        except Exception as e:
            logger.error('Neuron: Error while syncing chain state with error {}', e)
            raise FailedToEnterNeuron

    def stop(self):

        logger.info('Shutting down the Axon server ...')
        try:
            self.axon.stop()
            logger.info('Axon server stopped')
        except Exception as e:
            logger.error('Neuron: Error while stopping axon server: {} ', e)

    def __enter__(self):
        bittensor.exceptions.handlers.rollbar.init() # If a bittensor.exceptions.handlers.rollbar token is present, this will enable error reporting to bittensor.exceptions.handlers.rollbar
        logger.trace('Neuron enter')
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """ Defines the exit protocol from asyncio task.

        Args:
            exc_type (Type): The type of the exception.
            exc_value (RuntimeError): The value of the exception, typically RuntimeError. 
            exc_traceback (traceback): The traceback that can be printed for this exception, detailing where error actually happend.

        Returns:
            Neuron: present instance of Neuron.
        """        
        self.stop()
        if exc_value:

            top_stack = StringIO()
            tb.print_stack(file=top_stack)
            top_lines = top_stack.getvalue().strip('\n').split('\n')[:-4]
            top_stack.close()

            full_stack = StringIO()
            full_stack.write('Traceback (most recent call last):\n')
            full_stack.write('\n'.join(top_lines))
            full_stack.write('\n')
            tb.print_tb(exc_traceback, file=full_stack)
            full_stack.write('{}: {}'.format(exc_type.__name__, str(exc_value)))
            sinfo = full_stack.getvalue()
            full_stack.close()
            # Log the combined stack
            logger.error('Exception:{}'.format(sinfo))

            if bittensor.exceptions.handlers.rollbar.is_enabled():
                bittensor.exceptions.handlers.rollbar.send_exception()

        return self

    @staticmethod   
    def build_config() -> Munch:
        # Parses and returns a config Munch for this object.
        parser = argparse.ArgumentParser(); 
        Neuron.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        Neuron.check_config(config)
        return config

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        bittensor.wallet.Wallet.add_args( parser )
        bittensor.metagraph.Metagraph.add_args( parser )
        bittensor.nucleus.Nucleus.add_args( parser )
        bittensor.axon.Axon.add_args(parser)
        bittensor.dendrite.Dendrite.add_args( parser )

    @staticmethod   
    def check_config(config: Munch):
        bittensor.wallet.Wallet.check_config( config )
        bittensor.metagraph.Metagraph.check_config( config )
        bittensor.nucleus.Nucleus.check_config( config )
        bittensor.axon.Axon.check_config( config )
        bittensor.dendrite.Dendrite.check_config( config )

    def __del__(self):
        self.stop()

