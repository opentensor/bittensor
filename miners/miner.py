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

import bittensor
import argparse
import copy
import os
import bittensor.utils.networking as net

from munch import Munch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tensorboard import program
from loguru import logger
logger = logger.opt(colors=True)

class AbstractMiner ():

    def __init__( self, config: Munch = None, **kwargs):
        r""" Initializes a new base miner object.
            
            Args:
                config (:obj:`Munch`, `optional`): 
                    miner.Miner.default_config()
        """
        if config == None:
            config = AbstractMiner.default_config()
        config = copy.deepcopy( config ); bittensor.config.Config.update_with_kwargs( config, kwargs )
        AbstractMiner.check_config( config )
        self.config = config
        self.wallet = bittensor.wallet.Wallet ( config = self.config )
        self.subtensor = bittensor.subtensor.Subtensor( config = self.config )
        self.metagraph = bittensor.metagraph.Metagraph()
        self.axon = bittensor.axon.Axon( config = self.config, wallet = self.wallet )
        self.dendrite = bittensor.dendrite.Dendrite( config = self.config, wallet = self.wallet )

    @staticmethod   
    def default_config() -> Munch:
        # Parses and returns a config Munch for this object.
        parser = argparse.ArgumentParser(); 
        AbstractMiner.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        return config

    @staticmethod
    def check_config(config: Munch):
        assert 'name' in config.miner, 'miners must specify a name argument.'
        bittensor.wallet.Wallet.check_config( config )
        bittensor.subtensor.Subtensor.check_config( config )
        bittensor.axon.Axon.check_config( config )
        bittensor.dendrite.Dendrite.check_config( config )
        bittensor.nucleus.Nucleus.check_config( config )
        full_path = os.path.expanduser('{}/{}/{}'.format( config.miner.root_dir, config.wallet.name + "-" + config.wallet.hotkey, config.miner.name ))
        config.miner.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.miner.full_path):
            os.makedirs(config.miner.full_path)

    @staticmethod   
    def add_args( parser: argparse.ArgumentParser ):
        bittensor.wallet.Wallet.add_args( parser )
        bittensor.subtensor.Subtensor.add_args( parser )
        bittensor.axon.Axon.add_args( parser )
        bittensor.dendrite.Dendrite.add_args( parser )
        bittensor.nucleus.Nucleus.add_args( parser )
        parser.add_argument('--debug', default=False, dest='debug', action='store_true', help='''Turn on bittensor debugging information''')
        parser.add_argument('--config', type=str, help='If set, arguments are overridden by passed file.')
        parser.add_argument('--miner.modality', default=0, type=int, help='''Miner network modality. TEXT=0, IMAGE=1. Currently only allowed TEXT''')
        parser.add_argument('--miner.use_upnpc', default=False, dest='use_upnpc', action='store_true', help='''Turns on port forwarding on your router using upnpc.''')
        parser.add_argument('--miner.record_log', default=False, dest='record_log', action='store_true', help='''Turns on logging to file.''')   
        parser.add_argument('--miner.root_dir', default='~/.bittensor/miners/', type=str, help='Root path to load and save data associated with each miner')
        parser.add_argument('--miner.use_tensorboard', default=True, dest='use_tensorboard', action='store_true', help='Turn on bittensor logging to tensorboard')
        parser.add_argument('--miner.no_tensorboard', dest='use_tensorboard', action='store_false', help='Turn off bittensor logging to tensorboard')
        parser.set_defaults ( use_tensorboard=True )

    def startup( self ):
        self.init_logging()
        self.init_tensorboad()
        self.init_debugging()
        self.init_external_ports_and_addresses()
        self.init_wallet()
        self.connect_to_chain()
        self.subscribe_to_chain()
        self.init_axon()
        self.load_metagraph()
        self.sync_metagraph()
        self.save_metagraph()

    def shutdown ( self ):
        self.teardown_axon()

    def __exit__ ( self, exc_type, exc_value, exc_traceback ): 
        self.shutdown()

    def __del__ ( self ):
        self.shutdown()

    def __enter__ ( self ):
        self.startup()
    
    def init_logging ( self ):
        # Override Neuron init_logging.
        if self.config.record_log == True:
            filepath = self.config.miner.full_path + "/bittensor_output.log"
            logger.add (
                filepath,
                format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
                rotation="25 MB",
                retention="10 days"
            )
            logger.info('LOGGING is <green>ON</green> with sink: <cyan>{}</cyan>', filepath)
        else: 
            logger.info('LOGGING is <red>OFF</red>')

    def init_tensorboad( self ):
        if self.config.use_tensorboard == True:
            event_file_dir = self.config.miner.full_path + '/tensorboard-' + '-'.join(str(datetime.now()).split())
            self.tensorboard = SummaryWriter( log_dir = event_file_dir )
            self._tensorboard_program = program.TensorBoard()
            self._tensorboard_program.configure(argv=[None, '--logdir', event_file_dir ])
            self._tensorbaord_url = self._tensorboard_program.launch()
            logger.info('TENSORBOARD is <green>ON</green> with entrypoint: <cyan>http://localhost:6006/</cyan>', )
        else: 
            logger.info('TENSORBOARD is <red>OFF</red>')

    def init_debugging( self ):
        # ---- Set debugging ----
        if self.config.debug: bittensor.__debug_on__ = True; logger.info('DEBUG is <green>ON</green>')
        else: logger.info('DEBUG is <red>OFF</red>')

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

    def load_metagraph( self ):
        # ---- Sync metagraph ----
        path = os.path.expanduser('~/.bittensor/' + str(self.config.subtensor.network) + '.pt')
        if os.path.isfile(path):
            logger.info('\nLoading Metagraph...')
            try:
                path = '~/.bittensor/' + str(self.config.subtensor.network) + '.pt'
                self.metagraph.load_from_path( path = path )
                logger.success('Loaded metagraph from: <cyan>{}</cyan>', path)
            except:
                logger.error('Failed to load metagraph from path: <cyan>{}</cyan>', path)    
        
    def sync_metagraph( self ):
        # ---- Sync metagraph ----
        logger.info('\nSyncing Metagraph...')
        self.metagraph.sync( subtensor = self.subtensor )
        logger.info( self.metagraph )

    def save_metagraph( self ):
        # ---- Sync metagraph ----
        logger.info('\nSaving Metagraph...')
        try:
            path = '~/.bittensor/' + str(self.config.subtensor.network) + '.pt'
            self.metagraph.save_to_path( path = path )
            logger.success('Saved metagraph to: <cyan>{}</cyan>', path)  
        except:
            logger.error('Failed to save metagraph to path: <cyan>{}</cyan>', path)    

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