import argparse
import copy
import math
import torch
import time
import sys
import os
import traceback

import threading
import multiprocessing as mp
import bittensor.utils.networking as net

from tqdm import tqdm
from munch import Munch
from termcolor import colored
from types import SimpleNamespace
from qqdm import qqdm, format_str
from typing import Tuple, List, Optional
from torch.utils.tensorboard import SummaryWriter
import bittensor

from loguru import logger
logger = logger.opt(colors=True)

class Miner( bittensor.neuron.Neuron ):

    def __init__( 
            self, 
            config: Munch = None,
            root_dir: str = None,
            uid: str = None,
            record_log: bool = None,
            use_upnpc: bool = None,
            debug: bool = None,
            **kwargs
        ):
        r""" Initializes a new base miner object.
            
            Args:
                config (:obj:`Munch`, `optional`): 
                    miner.Miner.default_config()
                root_dir (str, default '~/.bittensor/miners/'):
                    Root path to load and save data associated with each miner
                uid (str, default=str(time.time()).split('.')[0]):
                    Saved state goes into miner.root_dir / (wallet.name  + wallet.hotkey) / miner.uid
                record_log (bool, default=True):
                    Record all logs when running this miner
                use_upnpc (bool, default=False):
                    Turns on port forwarding on your router using upnpc.
                debug (bool, default=False):
                    Turn on bittensor debugging information.
        """
        if config == None:
            config = Miner.default_config()
        config = copy.deepcopy(config); bittensor.config.Config.update_with_kwargs(config, kwargs )
        Miner.check_config(config)
        self.config = config
        self.tensorboard = SummaryWriter( log_dir = self.config.miner.full_path )
        if self.config.miner.record_log == True:
            filepath = self.config.miner.full_path + "/bittensor_output.log"
            logger.add (
                filepath,
                format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
                rotation="25 MB",
                retention="10 days"
            )
        super(Miner, self).__init__( self.config )

    @staticmethod   
    def default_config() -> Munch:
        # Parses and returns a config Munch for this object.
        parser = argparse.ArgumentParser(); 
        Miner.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        return config

    @staticmethod
    def check_config(config: Munch):
        bittensor.neuron.Neuron.check_config( config )
        os.listdir(".")
        miner_path = os.path.expanduser('{}/{}'.format(config.miner.root_dir, config.wallet.name + "-" + config.wallet.hotkey))
        max_trial_uid = 0
        for name in os.listdir(miner_path):
            try:
                max_trial_uid = max( int(name.split('-')[1]), max_trial_uid )
            except:
                pass
        config.miner.uid = str(max_trial_uid + 1)
        full_path = '{}/{}'.format(miner_path, "trial-" + config.miner.uid)
        config.miner.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.miner.full_path):
            os.makedirs(config.miner.full_path)

    @staticmethod   
    def add_args( parser: argparse.ArgumentParser ):
        bittensor.neuron.Neuron.add_args( parser )
        try:
            parser.add_argument (
                '--miner.root_dir',
                default='~/.bittensor/miners/',
                type=str,
                help='Root path to load and save data associated with each miner'
            )
            parser.add_argument (
                '--miner.uid',
                default=-1,
                type=str,
                help='if miner.uid < 0, defaults to next larget trial-uid value. Saved state goes into miner.root_dir / (wallet.name  + wallet.hotkey) / miner.uid'
            )
            parser.add_argument (
                '--miner.record_log',
                default=True,
                type=bool,
                help='Record all logs when running this miner'
            )
            parser.add_argument(
                '--use_upnpc', 
                dest='use_upnpc', 
                action='store_true', 
                help='''Turns on port forwarding on your router using upnpc.'''
            )
            parser.set_defaults ( 
                use_upnpc=False 
            )
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

    def __exit__ ( self, exc_type, exc_value, exc_traceback ): 
        self.shutdown()

    def __del__ ( self ):
        self.shutdown()

    def __enter__ ( self ):
        self.startup()

    def startup ( self ):
        self._init_logging()
        self._init_debugging()
        self._init_external_ports_and_addresses()
        self._init_wallet()
        self._connect_to_chain()
        self._subscribe_to_chain()
        self._init_axon()
        self._init_metagraph()

    def shutdown ( self ):
        self._teardown_axon()

    def run ( self ):
        raise NotImplementedError
    
    def _init_logging( self ):
        if self.config.miner.record_log: logger.info('logging is <green>ON</green> with sink: <cyan>{}</cyan>', self.config.miner.full_path + "/bittensor_output.log")
        else: logger.info('logging is <red>OFF</red>')

    def _init_debugging( self ):
        # ---- Set debugging ----
        if self.config.debug: bittensor.__debug_on__ = True; logger.info('debug is <green>ON</green>')
        else: logger.info('debug is <red>OFF</red>')

    def _init_external_ports_and_addresses ( self ):
        # ---- Punch holes for UPNPC ----
        if self.config.use_upnpc: 
            logger.info('upnpc is <green>ON</green>')
            try:
                self.external_port = net.upnpc_create_port_map( local_port = self.config.axon.local_port )
            except net.UPNPCException as upnpc_exception:
                logger.critical('Failed to hole-punch with upnpc')
                quit()
        else: 
            logger.info('upnpc is <red>OFF</red>')
            self.external_port = self.config.axon.local_port

        # ---- Get external ip ----
        logger.info('\nFinding external ip...')
        try:
            self.external_ip = net.get_external_ip()
        except net.ExternalIPNotFound as external_port_exception:
            logger.critical('Unable to attain your external ip. Check your internet connection.')
            quit()
        logger.success('Found external ip: <cyan>{}</cyan>', self.external_ip)

    def _init_wallet( self ):
        # ---- Load Wallets ----
        logger.info('\nLoading wallet...')
        if not self.wallet.has_coldkeypub:
            self.wallet.create_new_coldkey( n_words = 12, use_password = True )
        if not self.wallet.has_hotkey:
            self.wallet.create_new_hotkey( n_words = 12, use_password = False )

    def _connect_to_chain ( self ):
        # ---- Connect to chain ----
        logger.info('\nConnecting to network...')
        self.subtensor.connect()
        if not self.subtensor.is_connected():
            logger.critical('Failed to connect subtensor to network:<cyan>{}</cyan>', self.subtensor.config.subtensor.network)
            quit()

    def _subscribe_to_chain( self ):
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
            logger.critical('Failed to subscribe miner.')
            quit()

    def _teardown_axon(self):
        logger.info('\nTearing down axon...')
        self.axon.stop()

    def _init_axon( self ):
        # ---- Starting axon ----
        logger.info('\nStarting Axon...')
        self.axon.start()
        
    def _init_metagraph( self ):
        # ---- Sync metagraph ----
        logger.info('\nSyncing Metagraph...')
        self.metagraph.sync()
        logger.info( self.metagraph )

     # ---- Forward loop -----
    def forward_loop ( self ): 
        # ---- Loop until event is set -----
        logger.success('Forward thread started.')
        while not self.quit_forward.is_set():
            with self.get_forward_lock():
                self.run_next_forward_call()

    # ---- Start up forward loop -----
    def start_forward_loop( self ):
        if not hasattr(self, 'forward_thread'):
            self.forward_thread = threading.Thread( target = self.forward_loop, name = 'forward', daemon=True )
        if not hasattr(self, 'quit_forward'):
            self.quit_forward = mp.Event()
        if not hasattr(self, 'lock_forward'):
            self.lock_forward = mp.Lock()
        if self.quit_forward.is_set():
            self.quit_forward.clear()
        if not self.forward_thread.is_alive():
            self.forward_thread.start()

    # ---- Get Backward lock ----
    def get_forward_lock( self ):
        if not hasattr(self, 'forward_lock'):
            self.forward_lock = mp.Lock()
        return self.forward_lock

    # ---- Stop forward loop -----
    def stop_forward_loop( self ):
        if hasattr(self, 'quit_forward'):
            self.quit_forward.set()
        if hasattr(self, 'forward_thread'):
            if self.forward_thread.is_alive():
                self.forward_thread.join( timeout = 10 )
                if not self.forward_thread.is_alive():
                    logger.success("Forward thread joined.")
                else:
                    logger.error('Failed join forward thread.')
    
    # ---- Runs the backward call -----
    def run_next_backward_call( self ):
        try:
            # ---- Pull request ----
            pong, pubkey, inputs_x, grads_dy, modality = self.axon.next_backward_item( timeout = 1.0 )

            # ---- Process Backward request -----
            if None not in [ pong, pubkey, inputs_x, grads_dy, modality ]:
                logger.debug(' <white>Axon Backward Request</white> ---> <white>from</white>:<cyan>{}</cyan>, <white>inputs</white>:<cyan>{}</cyan>', pubkey, inputs.shape)
                outputs = self.backward_call ( 
                    pubkey = pubkey,
                    inputs_x = inputs,
                    grads_dy = grads_dy,
                    modality = modality
                )
                pong.send( outputs.detach() )
                logger.debug('<white>Axon Backward Response</white> <--- <white>to</white>:<cyan>{}</cyan>, <white>outputs</white>:<cyan>{}</cyan>', pubkey, outputs.shape)

        except Exception as e:
            logger.exception('Error in backward thread with error {}', e)

    # ---- Backward loop -----
    def backward_loop ( self ): 
        # ---- Loop until event is set -----
        logger.success('Backward thread started.')
        while not self.quit_forward.is_set():
            with self.get_backward_lock():
                self.run_next_backward_call()

    # ---- Start up backward loop -----
    def start_backward_loop( self ):
        if not hasattr(self, 'backward_thread'):
            self.backward_thread = threading.Thread( target = self.backward_loop, name = 'backward', daemon=True )
        if not hasattr(self, 'quit_backward'):
            self.quit_backward = mp.Event()
        if not hasattr(self, 'lock_backward'):
            self.lock_backward = mp.Lock()
        if not self.quit_backward.is_set():
            self.quit_backward.clear()
        if not self.backward_thread.is_alive():
            self.backward_thread.start()

    # ---- Get Backward lock ----
    def get_backward_lock( self ):
        if not hasattr(self, 'backward_lock'):
            self.backward_lock = mp.Lock()
        return self.backward_lock

    # ---- Stop backward loop -----
    def stop_backward_loop( self ):
        if hasattr(self, 'quit_backward'):
            self.quit_backward.set()
        if hasattr(self, 'backward_thread'):
            if self.backward_thread.is_alive():
                self.backward_thread.join( timeout = 10 )
                if not self.backward_thread.is_alive():
                    logger.success("Backward thread joined.")
                else:
                    logger.error('Failed join backward thread.') 