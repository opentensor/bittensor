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

class Neuron():

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
        r""" Initializes a new Neuron object.
            
            Args:
                config (:obj:`Munch`, `optional`): 
                    neuron.Neuron.default_config()
                root_dir (str, default '~/.bittensor/miners/'):
                    Root path to load and save data associated with each neuron
                uid (str, default=str(time.time()).split('.')[0]):
                    Saved models go in neuron.root_dir / (wallet.name  + wallet.hotkey) / neuron.uid
                record_log (bool, default=True):
                    Record all logs when running this miner
                use_upnpc (bool, default=False):
                    Turns on port forwarding on your router using upnpc.
                debug (bool, default=False):
                    Turn on bittensor debugging information.
        """
        # Config: Config items for all subobjects: wallet, metagraph, nucleus, axon, dendrite.
        # This object can be instantiated by calling Neuron.default_config()
        if config == None:
            config = Neuron.default_config()
        config = copy.deepcopy(config) ; bittensor.config.Config.update_with_kwargs( copy.deepcopy(config), kwargs )
        Neuron.check_config(config)
        self.config = config
        
        # --- Bittensor components ----
        # Wallet: Holds the hotkey keypair and coldkey pub which are user to sign messages 
        # and subscribe to the chain.
        self.wallet = bittensor.wallet.Wallet( config = self.config )
        
        # Subtensor: provides an interface to the subtensor chain given a wallet.
        self.subtensor = bittensor.subtensor.Subtensor( config = self.config )
        
        # Metagraph: Maintains a connection to the subtensor chain and hold chain state.
        self.metagraph = bittensor.metagraph.Metagraph( config = self.config, wallet = self.wallet, subtensor = self.subtensor  )
                
        # Axon: RPC server endpoint which serves your nucleus. Responds to Forward and Backward requests.
        self.axon = bittensor.axon.Axon( config = self.config, wallet = self.wallet )
        
        # Dendrite: RPC client makes Forward and Backward requests to downstream peers.
        self.dendrite = bittensor.dendrite.Dendrite( config = self.config, wallet = self.wallet )

        # ---- Logging ----
        self.tensorboard = SummaryWriter(log_dir = self.config.neuron.full_path)
        if self.config.neuron.record_log == True:
            filepath = self.config.neuron.full_path + "/bittensor_output.log"
            print ( filepath )
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
    def add_args( parser: argparse.ArgumentParser ):
        bittensor.wallet.Wallet.add_args( parser )
        bittensor.subtensor.Subtensor.add_args( parser )
        bittensor.metagraph.Metagraph.add_args( parser )
        bittensor.nucleus.Nucleus.add_args( parser )
        bittensor.axon.Axon.add_args( parser )
        bittensor.nucleus.Nucleus.add_args( parser )
        bittensor.dendrite.Dendrite.add_args( parser )
        try:
            parser.add_argument (
                '--neuron.root_dir',
                default='~/.bittensor/miners/',
                type=str,
                help='Root path to load and save data associated with each neuron'
            )
            parser.add_argument (
                '--neuron.uid',
                default=str(time.time()).split('.')[0],
                type=str,
                help='Saved models go in neuron.root_dir / (wallet.name  + wallet.hotkey) / neuron.uid'
            )
            parser.add_argument (
                '--neuron.record_log',
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

    @staticmethod
    def check_config(config: Munch):
        assert 'n_epochs' in config.miner
        full_path = '{}/{}/{}'.format(config.neuron.root_dir, config.wallet.name + "-" + config.wallet.hotkey, config.neuron.uid)
        config.neuron.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.neuron.full_path):
            os.makedirs(config.neuron.full_path)

    def __exit__( self, exc_type, exc_value, exc_traceback ): 
        self.shutdown()

    def __del__( self ):
        self.shutdown()

    def __enter__( self ):
        self.startup()

    def startup( self ):
        # ---- Set debugging ----
        if self.config.debug: bittensor.__debug_on__ = True; logger.info('debug is <green>ON</green>')
        else: logger.info('debug is <red>OFF</red>')

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

        # ---- Logging sink ----
        if self.config.neuron.record_log: logger.info('logging is <green>ON</green> with sink: <cyan>{}</cyan>', self.config.neuron.full_path + "/bittensor_output.log")
        else: logger.info('logging is <red>OFF</red>')

        # ---- Get external ip ----
        logger.info('\nFinding external ip...')
        try:
            self.external_ip = net.get_external_ip()
        except net.ExternalIPNotFound as external_port_exception:
            logger.critical('Unable to attain your external ip. Check your internet connection.')
            quit()
        logger.success('Found external ip: <cyan>{}</cyan>', self.external_ip)

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
            logger.critical('Failed to connect subtensor to network:<cyan>{}</cyan>', self.subtensor.config.subtensor.network)
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
        total_peers = torch.max( self.metagraph.uids )
        peers_online = torch.numel(torch.where( self.metagraph.block - self.metagraph.lastemit < 1000 )[0])
        logger.info('Chain block:<green>{}</green>\n'.format(self.metagraph.block))
        logger.info('Tao staked:<green>\u03C4{}</green>\n'.format(torch.sum(self.metagraph.S)))
        logger.info('Subscribed peers:<green>{}</green>\n'.format(total_peers))
        logger.info('Active peers:<green>{}</green>\n'.format(peers_online))

    def shutdown(self):
        logger.info('\nTearing down axon...')
        self.axon.stop()
        
    def run(self):
        raise NotImplementedError


class BasicNeuron( Neuron ):
        
    def __init__(
        self,
        config: Munch = None,
        **kwargs,
    ):
        if config == None:
            config = BasicNeuron.default_config()
        config = copy.deepcopy( config ); bittensor.config.Config.update_with_kwargs( copy.deepcopy(config), kwargs )
        BasicNeuron.check_config( config )
        self.config = config
        super(BasicNeuron, self).__init__( self.config )

    @staticmethod   
    def default_config() -> Munch:
        parser = argparse.ArgumentParser(); 
        BasicNeuron.add_args(parser) 
        config = bittensor.config.Config.to_config( parser ); 
        return config

    @staticmethod   
    def add_args( parser: argparse.ArgumentParser ):
        Neuron.add_args( parser )
        
    @staticmethod
    def check_config(config: Munch):
        Neuron.check_config( config )

    def startup(self):
        super().startup()
        self.start_forward_loop()
        self.start_forward_loop()
    
    def shutdown(self):
        super().shutdown()
        self.stop_forward_loop()
        self.stop_backward_loop()
        
    def save_model( self ):
        r""" Saves a state dictionary to neuron.full_path
        """
        try:
            state_dict = self.get_state_dict()
            logger.info( 'Saving model to: <cyan>{}/model.torch</cyan>'.format( self.config.neuron.full_path ))
            torch.save( state_dict, "{}/model.torch".format( self.config.neuron.full_path ))
            logger.success('Saved model' )
        except Exception as e:
             logger.error('Failed to save model with error:{}', e)

    def reload_model( self ):
        r""" Reloads a state dictionary from neuron.full_path
        """
        try:
            state_dict = torch.load("{}/model.torch".format( self.config.neuron.full_path ))
            reload_from_state_dict( state_dict )
        except Exception as e:
            logger.error('Failed to reload model with error: {}', e)

    # ---- Subclass Forward call ----
    def forward_call( self, pubkey:str, inputs:torch.FloatTensor, modality:int ) -> torch.FloatTensor:
        raise NotImplementedError()

    # ---- Subclass Backward call ----
    def backward_call( self, pubkey:str, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, modality:int ) -> torch.FloatTensor:
        raise NotImplementedError()

    # ---- Runs the forward call -----
    def run_next_forward_call( self ):
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

    # ---- Forward loop -----
    def forward_loop ( self ): 
        # ---- Loop until event is set -----
        logger.success('Forward thread started.\n')
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

    # ---- Backward loop -----
    def backward_loop ( self ): 
        # ---- Loop until event is set -----
        logger.success('Backward thread started.\n')
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

    # # --- Run Neuron ----
    # def run( self ):
    #     r""" Neuron main loop.
    #     """
    #     # --- Run startup ----
    #     with self:

    #         # --- Run state ----
    #         self.epoch = 0
    #         self.global_step = 0
            
    #         # ---- Save model ----
    #         self.save_model()

    #         # --- Run forever, or n_epochs ----
    #         for self.epoch in range( self.epoch, sys.maxsize if self.config.neuron.n_epochs < 0 else self.config.neuron.n_epochs ):

    #             # ---- Train ----
    #             self.run_next_training_epoch()

    #             # ---- Save or Reload model ----
    #             if self.should_save():
    #                 self.save_model()

    #             if self.should_reload():
    #                 self.reload_model()

    #             # ---- Metagraph ----
    #             self.metagraph.sync() # Pulls the latest metagraph state.

    #             # ---- Set weights ----
    #             self.metagraph.set_weights(
    #                 weights = self.get_row_weights(), 
    #                 wait_for_inclusion = True
    #             )

    #             # ---- Update Tensorboard ----
    #             self.epoch_to_tensorboard()
                
    # def epoch_to_tensorboard(self):
    #     r""" This function is called by neuron.run() after each epoch.
    #         The subclass may override this function to send custom data to tensorboard after every epoch.
    #     """
    #     self.axon.__to_tensorboard__( self.tensorboard, self.global_step )
    #     self.dendrite.__to_tensorboard__( self.tensorboard, self.global_step )
    #     self.metagraph.__to_tensorboard__( self.tensorboard, self.global_step )

    # # ---- Training logs ----
    # def training_logs( self, progress_bar, iteration:int, output: SimpleNamespace ):
    #     r""" This function is called by neuron.run_training_epoch() after each training step.
    #         The function must populate the passed progress bar with training step state.
    #     """
    #     index = self.metagraph.state.index_for_uid[self.metagraph.uid]
    #     progress_bar.set_infos({
    #         'GS': colored('{}'.format(self.global_step), 'red'),
    #         'LS': colored('{}'.format(iteration), 'blue'),
    #         'Epoch': colored('{}'.format(self.epoch+1), 'green'),
    #         'L-loss': colored('{:.5f}'.format(output.local_target_loss.item()), 'red'),
    #         'R-loss': colored('{:.5f}'.format(output.remote_target_loss.item()), 'green'),
    #         'D-loss': colored('{:.5f}'.format(output.distillation_loss.item()), 'blue'),
    #         'nPeers': colored(self.metagraph.n, 'yellow'),
    #         'Stake(\u03C4)': colored('{:.3f}'.format(self.metagraph.S[index]), 'red'),
    #         'Rank(\u03C4)': colored('{:.3f}'.format(self.metagraph.R[index]), 'green'),
    #         'Incentive(\u03C4/block)': colored('{:.6f}'.format(self.metagraph.I[index]), 'yellow'),
    #         'Axon': self.axon.__str__(),
    #         'Dendrite': self.dendrite.__str__(),
    #     })
    #     self.tensorboard.add_scalar('R-loss', output.remote_target_loss.item(), self.global_step)
    #     self.tensorboard.add_scalar('L-loss', output.local_target_loss.item(), self.global_step)
    #     self.tensorboard.add_scalar('D-loss', output.distillation_loss.item(), self.global_step)

    # def get_state_dict( self ) -> dict:
    #     r""" This function is called by neuron.save_model() on save.
    #         The function must return a state dict which will be passed to neuron.reload_from_state_dict.       
    #         Returns:
    #             state_dict (:obj:`dict`): 
    #                 Dictionary containing run state information such as the model parameters.
    #     """
    #     raise NotImplementedError

    # def reload_from_state_dict( self, state_dict: dict):
    #     r""" This function is called by neuron.reload_model() on reload.
    #         The function must reload the training state from the passed state_dict. 
    #         Args:
    #             state_dict (:obj:`dict`): 
    #                 Dictionary containing run state information such as the model parameters. Output 
    #                 of get_state_dict.
    #     """
    #     raise NotImplementedError()

    # def should_save( self ) -> bool:
    #     r""" This function is called by neuron.run() after every epoch.
    #         If this function returns True, the model is saved to disk and can be reloaded late.
    #         Returns:
    #             should_save (bool):
    #                 True by default. Saves model after each epoch.
    #     """
    #     return True

    # def should_reload( self ) -> bool:
    #     r""" This function is called by neuron.run() after every epoch.
    #         If the function returns True the model state is saved to neuron.full_path.
    #         Returns:
    #             should_reload (bool):
    #                 False by default. Does not reload the model after each epoch.
    #     """
    #     return False
      
    # def save_model( self ):
    #     r""" This function is called by neuron.run() if neuron.should_save() returns True.
    #     """
    #     try:
    #         state_dict = self.get_state_dict()
    #         logger.info( 'Saving model to: <cyan>{}/model.torch</cyan>'.format( self.config.neuron.full_path ))
    #         torch.save( state_dict, "{}/model.torch".format( self.config.neuron.full_path ))
    #         logger.success('Saved model' )
    #     except Exception as e:
    #          logger.error('Failed to save model with error:{}', e)

    # def reload_model( self ):
    #     r""" This function is called by neuron.run() if neuron.should_reload() returns True.
    #     """
    #     try:
    #         state_dict = torch.load("{}/model.torch".format( self.config.neuron.full_path ))
    #         reload_from_state_dict( state_dict )
    #     except Exception as e:
    #         logger.error('Failed to reload model with error: {}', e)

    # # ---- Subclass Training call ----
    # def training_call( self, batch: dict ) -> SimpleNamespace:
    #     r""" This function is called by neuron.run_next_training_epoch() for each batch 
    #         retrieved by neuron.get_epoch_batches(). It should run a single training batch 
    #         through the model and apply a gradient update and return a SimpleNamespace object.
    #         This object will be passed to neuron.training_logs()
    #         Args:
    #             batch ( dict, `required`): 
    #                 training batch dictionary as returned from get_epoch_batches            
    #         Returns:
    #             outputs ( SimpleNamespace ): 
    #                 SimpleNamespace output as returned by a nucleus forward call.
    #                 Must include fields local_loss, remote_loss, distillation_loss
    #     """
    #     raise NotImplementedError

    # # --- Run Epoch ----
    # def run_next_training_epoch( self ):
    #     r""" Called by neuron.run(), runs a training epoch of length self.config.miner.epoch_length
    #     """
    #     training_batches = self.get_epoch_batches( epoch = self.epoch )
    #     progress_bar = qqdm(enumerate(training_batches), total=len(training_batches), desc=format_str('blue', f'Epoch Progress'))
    #     for iteration, (training_batch) in progress_bar:
    #         output = self.training_call( batch = training_batch )
    #         self.training_logs( progress_bar, iteration = iteration, output = output )
    #         self.global_step += 1

    # # ---- Subclass Get Row Weights ----
    # def get_row_weights( self ) -> torch.FloatTensor:
    #     r""" This function is called after each training epoch to recieve row_weights.
    #         The passed row_weights are then submitted to chain.
    #         Returns:
    #             row_weights ( torch.FloatTensor, shape=(self.metagraph.n) ): 
    #                 torch row_weights matching the metagraph size.
    #                 weight values should be normalized and be in range [0,1].
    #     """
    #     raise NotImplementedError

    # # ---- Subclass Get epoch batches ----
    # def get_epoch_batches( self, epoch:int ) -> List[ dict ]:
    #     r""" Returns training batches for an epoch.
    #         Returns:
    #             batches ( List[dict], shape=(self.config.miner.epoch_length) ): 
    #                 List of batches as dictionary containing tokenized sentences
    #                 'inputs' = torch.LongTensor.
    #     """
    #     raise NotImplementedError


