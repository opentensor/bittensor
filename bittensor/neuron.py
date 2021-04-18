import argparse
import math
import torch
import time
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
logger = logger.opt(ansi=True)

class Neuron():

    def __init__( self, config: Munch = None ):
        r""" Initializes a new full Neuron object.
            
            Args:
                config (:obj:`Munch`, `optional`): 
                    neuron.Neuron.default_config()
        """
        # Config: Config items for all subobjects: wallet, metagraph, nucleus, axon, dendrite.
        # This object can be instantiated by calling Neuron.default_config()
        if config == None:
            config = Neuron.default_config()
        Neuron.check_config(config)
        self.config = config
        
        # --- Bittensor components ----
        # Wallet: Holds the hotkey keypair and coldkey pub which are user to sign messages 
        # and subscribe to the chain.
        self.wallet = bittensor.wallet.Wallet( self.config )
        
        # Subtensor: provides an interface to the subtensor chain given a wallet.
        self.subtensor = bittensor.subtensor.Subtensor( self.config )
        
        # Metagraph: Maintains a connection to the subtensor chain and hold chain state.
        self.metagraph = bittensor.metagraph.Metagraph( config = self.config, wallet = self.wallet, subtensor = self.subtensor )
                
        # Axon: RPC server endpoint which serves your nucleus. Responds to Forward and Backward requests.
        self.axon = bittensor.axon.Axon( config = self.config, wallet = self.wallet  )
        
        # Dendrite: RPC client makes Forward and Backward requests to downstream peers.
        self.dendrite = bittensor.dendrite.Dendrite( config = self.config, wallet = self.wallet )

        # ---- Thread locks ----
        self._training_lock = mp.Lock()
        self._forward_lock = mp.Lock()
        self._backward_lock = mp.Lock()

        # ---- Stop events ----
        self.quit_forward = mp.Event()
        self.quit_backward = mp.Event()
        self.quit_training = mp.Event()

        # ---- Processing Threads ----
        self.forward_thread = threading.Thread( target = self.forward_loop, name = 'forward', daemon=True )
        self.backward_thread = threading.Thread( target = self.backward_loop, name = 'backward', daemon=True )
        self.training_thread = threading.Thread( target = self.training_loop, name = 'training', daemon=True )

        # ---- Running state ----
        self.global_step = 0
        self.epoch = 0
        self.best_train_loss = math.inf

        # ---- Logging ----
        self.tensorboard = SummaryWriter(log_dir = self.config.neuron.full_path)
        if self.config.neuron.record_log == True:
            filepath = self.config.neuron.full_path + "/{}_{}.log".format(self.config.neuron.name, self.config.neuron.uid),
            logger.add (
                filepath,
                format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
                rotation="250 MB",
                retention="10 days"
            )

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
        bittensor.nucleus.Nucleus.add_args( parser )
        bittensor.axon.Axon.add_args( parser )
        bittensor.nucleus.Nucleus.add_args( parser )
        bittensor.dendrite.Dendrite.add_args( parser )
        parser.add_argument (
            '--neuron.root_dir',
            default='~/.bittensor/miners/',
            type=str,
            help='Root path to load and save data associated with each neuron'
        )
        parser.add_argument (
            '--neuron.name',
            default='gpt2-genesis',
            type=str,
            help='Trials for this neuron go in neuron.root / neuron.name'
        )
        parser.add_argument (
            '--neuron.uid',
            default=str(time.time()).split('.')[0],
            type=str,
            help='Saved models go in neuron.root_dir / neuron.name / neuron.uid'
        )
        parser.add_argument (
            '--neuron.record_log',
            default=False,
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

    @staticmethod
    def check_config(config: Munch):
        full_path = '{}/{}/{}'.format(config.neuron.root_dir, config.neuron.name, config.neuron.uid)
        config.neuron.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.neuron.full_path):
            os.makedirs(config.neuron.full_path)

    @property
    def _model(self):
        if self.model == None:
            raise RuntimeError('Subclasses must set a model class')

    def __del__( self ):
        self.shutdown()

    def __enter__(self):
        self.startup()
        return self

    def __exit__( self, exc_type, exc_value, exc_traceback ):   
        self.shutdown()

    def startup( self ):
        # ---- Set debugging ----
        if self.config.debug: bittensor.__debug_on__ = True; logger.info('DEBUG is <green>ON</green>')
        else: logger.info('DEBUG is <red>OFF</red>')

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
        logger.success('Found external ip <cyan>{}</cyan>', self.external_ip)

        # ---- Load Wallets ----
        logger.info('\nLoading wallet...')
        if not self.wallet.has_coldkeypub:
            self.wallet.create_new_coldkey( n_words = 12, use_password = True )
        if not self.wallet.has_hotkey:
            self.wallet.create_new_hotkey( n_words = 12, use_password = False )

        # ---- Connect to chain ----
        logger.info('\nConnecting to network...')
        self.subtensor.connect()
        if not self.subtensor.is_connected():
            logger.critical('Failed to connect subtensor')
            quit()
        
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

        # ---- Starting axon ----
        logger.info('\nStarting Axon...')
        self.axon.start()
        
        # ---- Sync metagraph ----
        logger.info('\nSyncing Metagraph...')
        self.metagraph.sync()
        print(self.metagraph)

        # --- Start threads ----
        logger.info('\nStarting threads...')
        self.forward_thread.start()
        self.backward_thread.start()
        self.training_thread.start()

    def shutdown( self ):
        logger.info("Stopping Axon on: <cyan>{}:{}</cyan>", self.axon.config.axon.local_ip, self.axon.config.axon.local_port)
        try:
            self.axon.stop()
            logger.success("Axon stopped")
        except Exception as e:
            logger.exception('Neuron: Error while stopping axon server: {} ', e)
        
        # ---- Set join events ----
        self.quit_backward.set()
        self.quit_forward.set()
        self.quit_training.set()

        # ---- Join threads ----
        self.forward_thread.join( timeout = 10 )
        if not self.forward_thread.is_alive():
            logger.success("Forward thread joined.")
        else:
            logger.error('Failed join forward thread.')

        self.backward_thread.join( timeout = 10 )
        if not self.backward_thread.is_alive():
            logger.success("Backward thread joined.")
        else:
            logger.error('Failed join backward thread.')

        self.training_thread.join( timeout = 10 )
        if not self.training_thread.is_alive():
            logger.success("Training thread joined.")
        else:
            logger.error('Failed join training thread.')

    def run( self ):

        # --- Run startup ----
        with self:

            # --- Run Forever ----
            while True:

                # ---- Aquire lock ----
                # Training lock is re-aquired after each training epoch.
                with self._training_lock:

                    # ---- Sync metagraph ----
                    # Metagraph must be updated with training stopped or else, undefined
                    # behaviour can occur.
                    self.metagraph.sync() # Pulls the latest metagraph state.

                # ---- Set weights ----
                self.metagraph.set_weights(
                    weights = self.get_row_weights(), 
                    wait_for_inclusion = True
                )

                # ---- Update Tensorboard ----
                self.dendrite.__to_tensorboard__( self.tensorboard, self.global_step )
                self.metagraph.__to_tensorboard__( self.tensorboard, self.global_step )
                self.axon.__to_tensorboard__( self.tensorboard, self.global_step )
                self.tensorboard.add_scalar('Neuron/Train_loss', self.training_loss, self.global_step )
            

    # ---- Subclass row weights ----
    def get_row_weights( self ) -> torch.FloatTensor:
        raise NotImplementedError()

    # ---- Subclass epoch batches ----
    def get_epoch_batches( self ) -> List[dict]:
        raise NotImplementedError()

    # ---- Subclass Training call ----
    def training_call( self, batch: dict ) -> SimpleNamespace:
        raise NotImplementedError()

    def run_training_epoch( self ):
        training_batches = self.get_epoch_batches( epoch = self.epoch )
        progress_bar = qqdm(enumerate(training_batches), total=len(training_batches), desc=format_str('blue', f'Epoch Progress'))
        for iteration, (training_batch) in progress_bar:
            if self.quit_training.is_set():
                break
            output = self.training_call( batch = training_batch )
            self.training_logs( progress_bar, iteration = iteration, output = output )
            self.global_step += 1

    # ---- Training loop ----
    def training_loop(self):
         # ---- Loop until event is set -----
        logger.success('Training thread started. ')
        while not self.quit_training.is_set():
            with self._training_lock:
                self.run_training_epoch()

    # ---- Subclass Forward call ----
    def forward_call( self, pubkey:str, inputs:torch.FloatTensor, modality:int ) -> torch.FloatTensor:
        raise NotImplementedError()

    # ---- Forward loop -----
    def forward_loop ( self ): 
        # ---- Loop until event is set -----
        logger.success('Forward thread started. ')
        while not self.quit_forward.is_set():
            with self._forward_lock:
                try:
                    # ---- Pull request ----
                    logger.debug('<white>Forward</white>: waiting for query ... ', self.axon)
                    pong, pubkey, inputs, modality = self.axon.next_forward_item( timeout = 1.0 )
                    if None not in [ pong, pubkey, inputs, modality]:
                        logger.debug('Recieved Forward Query: <white>from</white>:<cyan>{}</cyan>, <white>inputs</white>:<cyan>{}</cyan>', pubkey, inputs.shape)
                        outputs = self.forward_call ( 
                            pubkey = pubkey,
                            inputs = inputs,
                            modality = modality
                        )
                        pong.send( outputs.detach() )
                        logger.debug('Sent forward response: to:<cyan>{}</cyan>, outputs.shape:<cyan>{}</cyan>', pubkey, outputs.shape)
                except Exception as e:
                    logger.exception('Error in forward thread with error {}', e)
                    continue

    # ---- Subclass Backward call ----
    def backward_call( self, pubkey:str, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, modality:int ) -> torch.FloatTensor:
        raise NotImplementedError()
            
    # ---- Backward loop -----
    def backward_loop ( self ): 
        # ---- Loop until event is set -----
        logger.success('Backward thread started. ')
        while not self.quit_forward.is_set():
            with self._backward_lock:
                try:
                    # ---- Pull request ----
                    logger.debug('<white>Backward</white>: waiting for query ... ',)
                    pong, pubkey, inputs_x, grads_dy, modality = self.axon.next_backward_item( timeout = 1.0 )

                    # ---- Process Backward request -----
                    if None not in [ pong, pubkey, inputs_x, grads_dy, modality ]:
                        logger.debug('Recieved Backward Query: <white>from</white>:<cyan>{}</cyan>, <white>inputs</white>:<cyan>{}</cyan>', pubkey, inputs.shape)
                        outputs = self.backward_call ( 
                            pubkey = pubkey,
                            inputs_x = inputs,
                            grads_dy = grads_dy,
                            modality = modality
                        )
                        pong.send( outputs.detach() )
                        logger.debug('Sent backward response: <white>to</white>:<cyan>{}</cyan>, <white>outputs</white>:<cyan>{}</cyan>', pubkey, outputs.shape)

                except Exception as e:
                    logger.exception('Error in backward thread with error {}', e)
                    continue

    # ---- Training logs ----
    def training_logs( self, progress_bar, iteration:int, output: SimpleNamespace ):
        index = self.metagraph.state.index_for_uid[self.metagraph.uid]
        progress_bar.set_infos({
            'GS': colored('{}'.format(self.global_step), 'red'),
            'LS': colored('{}'.format(iteration), 'blue'),
            'Epoch': colored('{}'.format(self.epoch+1), 'green'),
            'L-loss': colored('{:.5f}'.format(output.local_target_loss.item()), 'yellow'),
            'R-loss': colored('{:.5f}'.format(output.remote_target_loss.item()), 'red'),
            'D-loss': colored('{:.5f}'.format(output.distillation_loss.item()), 'green'),
            'nPeers': colored(self.metagraph.n, 'blue'),
            'Stake(\u03C4)': colored('{:.3f}'.format(self.metagraph.S[index]), 'green'),
            'Rank(\u03C4)': colored('{:.3f}'.format(self.metagraph.R[index]), 'yellow'),
            'Incentive(\u03C4/block)': colored('{:.6f}'.format(self.metagraph.I[index]), 'red'),
            'Axon': self.axon.__str__(),
            'Dendrite': self.dendrite.__str__(),
        })
        self.tensorboard.add_scalar('R-loss', output.remote_target_loss.item(), self.global_step)
        self.tensorboard.add_scalar('L-loss', output.local_target_loss.item(), self.global_step)
        self.tensorboard.add_scalar('D-loss', output.distillation_loss.item(), self.global_step)
