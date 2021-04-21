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
                rotation="50 MB",
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
        full_path = '{}/{}/{}'.format(config.miner.root_dir, config.wallet.name + "-" + config.wallet.hotkey, config.miner.uid)
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
                default=str(time.time()).split('.')[0],
                type=str,
                help='Saved state goes into miner.root_dir / (wallet.name  + wallet.hotkey) / miner.uid'
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

    def start ( self ):
        with self:
            self.run()
    
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


class BasicMiner( Miner ):

    def __init__( 
            self, 
            config: Munch = None,
            **kwargs
        ):
        r""" Initializes a new basic miner object.
            
            Args:
                config (:obj:`Munch`, `optional`): 
                    miner.BasicMiner.default_config()
        """
        if config == None:
            config = BasicMiner.default_config()
        config = copy.deepcopy(config); bittensor.config.Config.update_with_kwargs(config, kwargs )
        BasicMiner.check_config(config)
        self.config = config
        super(Miner, self).__init__( self.config )

    @staticmethod   
    def default_config() -> Munch:
        parser = argparse.ArgumentParser(); 
        BasicMiner.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        return config
    
    @staticmethod
    def check_config(config: Munch):
        Miner.check_config( config )

    @staticmethod   
    def add_args( parser: argparse.ArgumentParser ):
        Miner.add_args( parser )
        try:
            parser.add_argument('--miner.n_epochs', default=-1, type=int, help='Miner runs for this many epochs, or forever if < 0')
        except argparse.ArgumentError:
            pass

    def startup( self ):
        super().startup()
        self.start_forward_loop()
        self.start_backward_loop()
    
    def shutdown(self):
        super().shutdown()
        self.stop_forward_loop()
        self.stop_backward_loop()

    # ---- Training call ----
    def training_call( self, batch: dict ) -> SimpleNamespace:
        r""" Runs a single training batch through the nucleus and applies a gradient update.
            Args:
                batch ( dict, `required`): 
                    training batch dictionary as returned from get_epoch_batches            
            Returns:
                outputs ( SimpleNamespace ): 
                    SimpleNamespace output as returned by a nucleus forward call.
                    Must include fields local_loss, remote_loss, distillation_loss
        """
        raise NotImplementedError()

    # ---- Get epoch batches ----
    def get_epoch_batches( self, epoch:int ) -> List[ dict ]:
        r""" Returns training batches for each epoch.
            Returns:
                batches ( List[dict], shape=(self.config.miner.epoch_length) ): 
                    List of batches as dictionary containing tokenized sentences
                    'inputs' = torch.LongTensor.
        """
        raise NotImplementedError()

    # ---- Get Row Weights ----
    def get_row_weights( self ) -> torch.FloatTensor:
        r""" Called after each training epoch. Returns row_weights to be set on chain.
            Returns:
                row_weights ( torch.FloatTensor, shape=(self.metagraph.n) ): 
                    torch row_weights matching the metagraph size.
                    weight values should be normalized and be in range [0,1].
        """
        raise NotImplementedError()

    def get_state_dict( self ) -> dict:
        r""" Called by miner.save_state().
            Returns a state dict which can be passed to miner.reload_from_state_dict on reload.
            Returns:
                state_dict (:obj:`dict`): 
                    Dictionary containing run state information such as the nucleus parameters.
        """
        raise NotImplementedError()

    def reload_from_state_dict( self, state_dict: dict):
        r""" Called by miner.reload_state().
            Reloads the training state from the passed state_dict. 
            Args:
                state_dict (:obj:`dict`): 
                    Dictionary containing run state information such as the nucleus parameters. Output 
                    of get_state_dict.
        """
        raise NotImplementedError()
        
     # ---- Subclass Forward call ----
    def forward_call( self, pubkey:str, inputs:torch.FloatTensor, modality:int ) -> torch.FloatTensor:
        r""" Called by neuron.forward_loop which can be overridden by the child class.
            The arguments reflect an RPC request from another neuron in the network, the response tensor
            should be the hidden units of the local nucleus of shape [batch_size, sequence_len, __network_dim__].
            
            Args:
                pubkey ( str, `required`): 
                    The public key of the caller.
                inputs ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.
                modality ( bittensor.proto.Modality, `required`):
                    modality of inputs e.g. bittensor.proto.Modality.TEXT.
            
            Returns:
                outputs (:obj:`torch.FloatTensor`): 
                    The nucleus's outputs as a torch tensor of shape [batch_size, sequence_len, __network_dim__]
        """
        raise NotImplementedError()

    # ---- Subclass Backward call ----
    def backward_call( self, pubkey:str, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, modality:int ) -> torch.FloatTensor:
        r""" Called by neuron.backward_loop which can be overridden in the child class.
            Arguments reflect an RPC backward request from another neuron in the network, the response tensor
            should be the gradients of the miner's nucleus w.r.t to the inputs and the passed output grads.
            
            Args:
                pubkey ( str, `required`): 
                    The public key of the caller.
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs from previous forward call.
                grads_dy ( :obj:`torch.Tensor`, `required`):
                    torch grads of forward output.
                modality ( bittensor.proto.Modality, `required`):
                    modality of inputs e.g. bittensor.proto.Modality.TEXT.
            
            Returns:
                outputs (:obj:`torch.FloatTensor`): 
                    The gradients w.r.t to the inputs [batch_size, sequence_len, __network_dim__]
        """
        raise NotImplementedError()

    def get_nucleus() -> 'bittensor.nucleus.Nucleus':
        r""" Called by miner.should_reload().
            Should return a bittensor.nucleus object.
            Returns:
                nucleus (bittensor.nucleus.Nucleus):
                    Mine nucleus object.
        """
        raise NotImplementedError()

    # ---- Runs the forward call -----
    def run_next_forward_call( self ):
        try:
            # ---- Pull request ----
            pong, pubkey, inputs, modality = self.axon.next_forward_item( timeout = 1.0 )
            if None not in [ pong, pubkey, inputs, modality]:
                logger.debug('<white>Axon Forward Request</white> ---> <white>from</white>:<cyan>{}</cyan>, <white>inputs</white>:<cyan>{}</cyan>', pubkey, inputs.shape)
                outputs = self.forward_call ( 
                    pubkey = pubkey,
                    inputs = inputs,
                    modality = modality
                )
                pong.send( outputs.detach() )
                logger.debug('<white>Axon Forward Response</white> ---> <white>to</white>:<cyan>{}</cyan>, <white>outputs</white>:<cyan>{}</cyan>', pubkey, outputs.shape)
        except Exception as e:
            logger.exception('Error in forward thread with error {}', e)

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

    def should_run( self, epoch: int ) -> bool:
        r""" Called by miner.run() every epoch, if the response is false, training stops.
        """
        if self.config.miner.n_epochs < 0:
            return True
        elif epoch < self.config.miner.n_epochs:
            return True
        else:
            return False

    def should_save( self ) -> bool:
        r""" Called by miner.run() after every epoch.
            If this function returns True, the miner state is saved to disk and can be reloaded later.
            Returns:
                should_save (bool):
                    True by default. Saves miner state after each epoch.
        """
        if self.epoch_loss < self.last_saved_loss:
            return True
        else:
            return False

    def save_state( self ):
        r""" This function is called by neuron.run() if neuron.should_save() returns True.
        """
        try:
            state_dict = self.get_state_dict()
            logger.info( 'Saving model to: <cyan>{}/model.torch</cyan>'.format( self.config.neuron.full_path ))
            torch.save( state_dict, "{}/model.torch".format( self.config.neuron.full_path, self.epoch_loss ))
            self.last_saved_loss = self.epoch_loss
            logger.success('Saved model' )
        except Exception as e:
             logger.error('Failed to save model with error:{}', e)

    def reload_state( self ):
        r""" Called by neuron.run() if neuron.should_reload() returns True.
        """
        try:
            state_dict = torch.load("{}/model.torch".format( self.config.neuron.full_path ))
            reload_from_state_dict( state_dict )
        except Exception as e:
            logger.error('Failed to reload model with error: {}', e)

    def should_reload(self) -> bool:
        r""" Called by miner.run() after every epoch.
            If the function returns True the miner state dict is saved to miner.full_path.
            Returns:
                should_reload (bool):
                    False by default. Does not reload the miner state after each epoch.
        """
        if torch.any(torch.isnan(torch.cat([param.view(-1) for param in self.get_nucleus().parameters()]))):
            return True

    def epoch_to_tensorboard(self):
        r""" Called by miner.run() after each epoch.
            Sends neuron state to tensorboard.
        """
        self.axon.__to_tensorboard__( self.tensorboard, self.global_step )
        self.dendrite.__to_tensorboard__( self.tensorboard, self.global_step )
        self.metagraph.__to_tensorboard__( self.tensorboard, self.global_step )

    # ---- Training logs ----
    def training_logs( self, progress_bar, iteration:int, output: SimpleNamespace ):
        r""" Called by miner.run_training_epoch() after each training step.
            The function populates and displays the passed progress bar.
        """
        index = self.metagraph.state.index_for_uid[self.metagraph.uid]
        progress_bar.set_infos({
            'GS': colored('{}'.format(self.global_step), 'red'),
            'LS': colored('{}'.format(iteration), 'blue'),
            'Epoch': colored('{}'.format(self.epoch+1), 'green'),
            'Epoch-loss': colored('{:.4f}'.format(self.epoch_loss), 'yellow'),
            'Saved-loss': colored('{:.4f}'.format(self.last_saved_loss), 'red'),
            'L-loss': colored('{:.4f}'.format(output.local_target_loss.item()), 'blue'),
            'R-loss': colored('{:.4f}'.format(output.remote_target_loss.item()), 'green'),
            'D-loss': colored('{:.4f}'.format(output.distillation_loss.item()), 'yellow'),
            'nPeers': colored(self.metagraph.n, 'red'),
            'Stake(\u03C4)': colored('{:.3f}'.format(self.metagraph.S[index]), 'green'),
            'Rank(\u03C4)': colored('{:.3f}'.format(self.metagraph.R[index]), 'blue'),
            'Incentive(\u03C4/block)': colored('{:.6f}'.format(self.metagraph.I[index]), 'yellow'),
            'Axon': self.axon.__str__(),
            'Dendrite': self.dendrite.__str__(),
        })
        self.tensorboard.add_scalar('R-loss', output.remote_target_loss.item(), self.global_step)
        self.tensorboard.add_scalar('L-loss', output.local_target_loss.item(), self.global_step)
        self.tensorboard.add_scalar('D-loss', output.distillation_loss.item(), self.global_step)

    # --- Run Epoch ----
    def run_next_training_epoch( self, training_batches: List[dict] ) -> float:
        r""" Called by miner.run(), calls training_call for passed batches.
            Args:
                training_batches (List[dict]):
                    Training batches as returned by get_epoch_batches.
        """
        total_epoch_loss = 0.0
        progress_bar = qqdm(enumerate(training_batches), total=len(training_batches), desc=format_str('blue', f'Epoch Progress'))
        for iteration, (training_batch) in progress_bar:
            output = self.training_call( batch = training_batch )
            total_epoch_loss += output.local_target_loss.item()
            self.epoch_loss = total_epoch_loss / (iteration + 1) 
            self.global_step += 1
            self.training_logs( progress_bar, iteration = iteration, output = output )
        self.scheduler.step()

    # --- Run Neuron ----
    def run( self ):
        r""" Neuron main loop.
        """
        # --- Run state ----
        self.epoch = -1
        self.epoch_loss = math.inf
        self.global_step = 0
        
        # ---- Save miner state ----
        self.save_state()

        # --- Run until ----
        while self.should_run( self.epoch ):
            self.epoch += 1

            # ---- Train ----
            self.run_next_training_epoch( 
                training_batches = self.get_epoch_batches( self.epoch ) 
            )

            # ---- Save or Reload state ----
            if self.should_save():
                self.save_state()
            elif self.should_reload():
                self.reload_state()

            # ---- Set weights ----
            self.metagraph.set_weights(
                weights = self.get_row_weights(), 
                wait_for_inclusion = True
            )

            # ---- Metagraph ----
            self.metagraph.sync() # Pulls the latest metagraph state.

            # ---- Update Tensorboard ----
            self.epoch_to_tensorboard()
