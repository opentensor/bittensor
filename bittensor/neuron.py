
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
import bittensor.utils.networking as net
import bittensor

from munch import Munch
from loguru import logger

logger = logger.opt(colors=True)


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
        # Wallet: Holds the hotkey keypair and coldkey pub which are user to sign messages 
        # and subscribe to the chain.
        if wallet == None:
            wallet = bittensor.wallet.Wallet (self.config )
        self.wallet = wallet
        # Subtensor: provides an interface to the subtensor chain given a wallet.
        if subtensor == None:
            subtensor = bittensor.subtensor.Subtensor( self.config )
        self.subtensor = subtensor
        # Metagraph: Maintains a connection to the subtensor chain and hold chain state.
        if metagraph == None:
            metagraph = bittensor.metagraph.Metagraph(config = self.config, wallet = self.wallet, subtensor = self.subtensor)
        self.metagraph = metagraph
        # Axon: RPC server endpoint which serves your synapse. Responds to Forward and Backward requests.
        if axon == None:
            axon = bittensor.axon.Axon(config = self.config, wallet = self.wallet )
        self.axon = axon
        # Dendrite: RPC client makes Forward and Backward requests to downstream peers.
        if dendrite == None:
            dendrite = bittensor.dendrite.Dendrite(config = self.config, wallet = self.wallet )
        self.dendrite = dendrite

    @staticmethod   
    def default_config() -> Munch:
        # Parses and returns a config Munch for this object.
        parser = argparse.ArgumentParser(); 
        Neuron.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        return config

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        bittensor.wallet.Wallet.add_args( parser )
        bittensor.subtensor.Subtensor.add_args( parser )
        bittensor.metagraph.Metagraph.add_args( parser )
        bittensor.axon.Axon.add_args(parser)
        bittensor.dendrite.Dendrite.add_args( parser )
        bittensor.synapse.Synapse.add_args( parser )
        try:
            parser.add_argument('--neuron.modality', default=0, type=int, 
                                help='''Neuron network modality. TEXT=0, IMAGE=1. Currently only allowed TEXT''')
        except:
            pass
        try:
            parser.add_argument(
                '--use_upnpc', 
                dest='use_upnpc', 
                action='store_true', 
                help='''Turns on port forwarding on your router using upnpc.'''
            )
            parser.set_defaults ( 
                use_upnpc=False 
            )
        except argparse.ArgumentError:
            pass      
        try:
            parser.add_argument(
                '--record_log', 
                dest='record_log', 
                action='store_true', 
                help='''Turns on logging to file.'''
            )
            parser.set_defaults ( 
                record_log=True 
            )
        except argparse.ArgumentError:
            pass
        try:
            parser.add_argument (
                '--debug', 
                dest='debug', 
                action='store_true', 
                help='''Turn on bittensor debugging information'''
            )
            parser.set_defaults ( 
                debug=False 
            )
        except argparse.ArgumentError:
            pass
        try:
            parser.add_argument(
                '--config', 
                type=str, 
                help='If set, arguments are overridden by passed file. '
            )
        except argparse.ArgumentError:
            pass
        
    @staticmethod   
    def check_config(config: Munch):
        assert config.neuron.modality == bittensor.proto.Modality.TEXT, 'Only TEXT modalities are allowed at this time.'

    def __exit__ ( self, exc_type, exc_value, exc_traceback ): 
        self.shutdown()

    def __del__ ( self ):
        self.shutdown()

    def __enter__ ( self ):
        self.startup()

    def startup ( self ):
        self.init_logging()
        self.init_debugging()
        self.init_external_ports_and_addresses()
        self.init_wallet()
        self.connect_to_chain()
        self.subscribe_to_chain()
        self.init_axon()
        self.sync_metagraph()

    def shutdown ( self ):
        self.teardown_axon()

    def teardown_axon(self):
        logger.info('\nTearing down axon...')
        self.axon.stop()

    def init_debugging( self ):
        # ---- Set debugging ----
        if self.config.debug: bittensor.__debug_on__ = True; logger.info('DEBUG is <green>ON</green>')
        else: logger.info('DEBUG is <red>OFF</red>')

    def init_logging ( self ):
        if self.config.record_log == True:
            filepath = "~/.bittensor/bittensor_output.log"
            logger.add (
                filepath,
                format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
                rotation="25 MB",
                retention="10 days"
            )
            logger.info('LOGGING is <green>ON</green> with sink: <cyan>{}</cyan>', "~/.bittensor/bittensor_output.log")
        else: 
            logger.info('LOGGING is <red>OFF</red>')

    def init_external_ports_and_addresses ( self ):
        # ---- Punch holes for UPNPC ----
        if self.config.use_upnpc: 
            logger.info('UPNPC is <green>ON</green>')
            try:
                self.external_port = net.upnpc_create_port_map( local_port = self.config.axon.local_port )
            except net.UPNPCException as upnpc_exception:
                logger.critical('Failed to hole-punch with upnpc')
                quit()
        else: 
            logger.info('UPNPC is <red>OFF</red>')
            self.external_port = self.config.axon.local_port

        # ---- Get external ip ----
        logger.info('\nFinding external ip...')
        try:
            self.external_ip = net.get_external_ip()
        except net.ExternalIPNotFound as external_port_exception:
            logger.critical('Unable to attain your external ip. Check your internet connection.')
            quit()
        logger.success('Found external ip: <cyan>{}</cyan>', self.external_ip)

    def init_wallet( self ):
        # ---- Load Wallets ----
        logger.info('\nLoading wallet...')
        if not self.wallet.has_coldkeypub:
            self.wallet.create_new_coldkey( n_words = 12, use_password = True )
        if not self.wallet.has_hotkey:
            self.wallet.create_new_hotkey( n_words = 12, use_password = False )

    def init_axon( self ):
        # ---- Starting axon ----
        logger.info('\nStarting Axon...')
        self.axon.start()
        
    def sync_metagraph( self ):
        # ---- Sync metagraph ----
        logger.info('\nSyncing Metagraph...')
        self.metagraph.sync()
        logger.info( self.metagraph )

    def connect_to_chain ( self ):
        # ---- Connect to chain ----
        logger.info('\nConnecting to network...')
        self.subtensor.connect()
        if not self.subtensor.is_connected():
            logger.critical('Failed to connect subtensor to network:<cyan>{}</cyan>', self.subtensor.config.subtensor.network)
            quit()

    def subscribe_to_chain( self ):
        # ---- Subscribe to chain ----
        logger.info('\nSubscribing to chain...')
        subscribe_success = self.subtensor.subscribe(
                wallet = self.wallet,
                ip = self.external_ip, 
                port = self.external_port,
                modality = bittensor.proto.Modality.TEXT,
                wait_for_finalization = True,
                timeout = 4 * bittensor.__blocktime__,
        )
        if not subscribe_success:
            logger.critical('Failed to subscribe neuron.')
            quit()
