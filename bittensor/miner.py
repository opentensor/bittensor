import argparse
import copy
import math
import torch
import random
import json
import time
import sys
import os
import traceback

import threading
import concurrent.futures
import multiprocessing as mp
import bittensor.utils.networking as net

from tqdm import tqdm
from munch import Munch
from termcolor import colored
from types import SimpleNamespace
from qqdm import qqdm, format_str
from typing import Tuple, List, Optional
from tqdm import tqdm

# Rich imports.
from rich.live import Live
from rich.table import Table
from rich import print
from rich.console import RenderGroup
from rich.panel import Panel
from rich import print
from rich.columns import Columns
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn

from torch.utils.tensorboard import SummaryWriter
import bittensor

from datetime import datetime
from tensorboard import program

from loguru import logger
logger = logger.opt(colors=True)

class Miner( bittensor.neuron.Neuron ):

    def __init__( 
            self, 
            config: Munch = None,
            root_dir: str = None,
            **kwargs
        ):
        r""" Initializes a new base miner object.
            
            Args:
                config (:obj:`Munch`, `optional`): 
                    miner.Miner.default_config()
                root_dir (str, default '~/.bittensor/miners/'):
                    Root path to load and save data associated with each miner
        """
        if config == None:
            config = Miner.default_config()
        config = copy.deepcopy( config ); bittensor.config.Config.update_with_kwargs( config, kwargs )
        Miner.check_config( config )
        self.config = config
        super( Miner, self ).__init__( self.config, **kwargs )

    @staticmethod   
    def default_config() -> Munch:
        # Parses and returns a config Munch for this object.
        parser = argparse.ArgumentParser(); 
        Miner.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        return config

    @staticmethod
    def check_config(config: Munch):
        assert 'name' in config.miner, 'miners must specify a name argument.'
        bittensor.neuron.Neuron.check_config( config )
        full_path = os.path.expanduser('{}/{}/{}'.format( config.miner.root_dir, config.wallet.name + "-" + config.wallet.hotkey, config.miner.name ))
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
        except argparse.ArgumentError:
            logger.warning('argument miner.root_dir was parsed twice')
            pass
        try:
            parser.add_argument (
                '--miner.use_tensorboard',
                dest='use_tensorboard',
                action='store_true',
                help='Turn on bittensor logging to tensorboard'
            )
            parser.add_argument (
                '--miner.no_tensorboard',
                dest='use_tensorboard',
                action='store_false',
                help='Turn off bittensor logging to tensorboard',
                default=True
            )
        except argparse.ArgumentError:
            logger.warning('argument miner.root_dir was parsed twice')
            pass
    
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

class BaseMiner( Miner ):

    def __init__( 
            self, 
            config: Munch = None,
            **kwargs
        ):
        r""" Initializes a new basic miner object.
            
            Args:
                config (:obj:`Munch`, `optional`): 
                    miner.BaseMiner.default_config()
        """
        if config == None:
            config = BaseMiner.default_config()
        config = copy.deepcopy( config ); bittensor.config.Config.update_with_kwargs( config, kwargs )
        BaseMiner.check_config( config )
        self.config = config
        super( BaseMiner, self ).__init__( self.config, **kwargs )

    @staticmethod   
    def default_config() -> Munch:
        parser = argparse.ArgumentParser(); 
        BaseMiner.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        return config
    
    @staticmethod
    def check_config(config: Munch):
        Miner.check_config( config )

    @staticmethod   
    def add_args( parser: argparse.ArgumentParser ):
        Miner.add_args( parser )
        parser.add_argument (
                '--miner.resume', 
                dest='resume', 
                action='store_true', 
                help='''Resume training from last save state.''',
                default=False,
            )
        parser.add_argument (
                '--miner.rich', 
                dest='rich', 
                action='store_true', 
                help='''Rich text display''',
                default=False
            )
        parser.add_argument (
                '--miner.restart_on_failure', 
                dest='restart_on_failure', 
                action='store_true', 
                help='''Restart miner on unknown error.''',
                default=False 
            )

        parser.add_argument('--miner.max_backward_workers', default='10', type=int, help='Maximum number of concurrent backward processing threads.')
        parser.add_argument('--miner.max_forward_workers', default='10', type=int, help='Maximum number of concurrent forward processing threads.')


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

    def set_mechanism_weights( self ):
        r""" Called after every training epoch, sets the row_weights into the incentive mechanism on chain.
        """
        row_weights = self.get_row_weights()
        uids = self.metagraph.uids
        did_set = self.subtensor.set_weights(
            wallet = self.wallet,
            uids = uids,
            weights = row_weights, 
            wait_for_inclusion = True
        )
        if did_set:
            logger.success('Successfully set weights with row:\n {}', row_weights.tolist())
        else:
            logger.warning('Failed to set weights on chain.')

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
        r""" Called by miner.forward_loop which can be overridden by the child class.
            The arguments reflect an RPC request from another miner in the network, the response tensor
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
        r""" Called by miner.backward_loop which can be overridden in the child class.
            Arguments reflect an RPC backward request from another miner in the network, the response tensor
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

    # ---- Runs the forward call -----
    def run_next_forward_call( self, pong: mp.Pipe, pubkey: str, inputs: torch.Tensor, modality: int ):
        r""" 
            Calls the backward call on the miner given inputs. This call multi-threaded using a threadpool executor.
                        
            Returns:
                pong (:obj:`mp.Pipe, `optional`): 
                    multiprocessing pipe tunnel for the response.
                public_key (str, `optional`):
                    public key of caller.
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.
                modality ( bittensor.proto.Modality, `required`):
                    modality of inputs.

        """
        try:
            outputs = self.forward_call ( 
                pubkey = pubkey,
                inputs = inputs,
                modality = modality
            )
            outputs = outputs.to('cpu')
            pong.send( outputs.detach() )
        except BrokenPipeError:
            logger.info('Failed to process forward request before timeout')
            pass

    # ---- Runs the backard call -----
    def run_next_backward_call( self, pong: mp.Pipe, pubkey: str, inputs_x: torch.Tensor, grads_dy: torch.float32, modality: int ):
        r""" 
            Calls the backward call on the miner given inputs. This call multi-threaded using a threadpool executor.
            
            Returns:
                pong (:obj:`mp.Pipe, `optional`): 
                    multiprocessing pipe tunnel for the response.
                pubkey (str, `optional`):
                    public key of caller.
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.
                grads_dy ( :obj:`torch.Tensor`, `required`):
                    torch gradient inputs to be backward processed with inputs.
                modality ( bittensor.proto.Modality, `required`):
                    modality of inputs.
        """

        try:
            outputs = self.backward_call ( 
                pubkey = pubkey,
                grads_dy = grads_dy, 
                inputs_x = inputs_x,
                modality = modality
            )
            pong.send( outputs.detach() )
        except BrokenPipeError:
            logger.info('Failed to process forward request before timeout')
            pass

    # ---- Forward loop -----
    def forward_loop ( self ): 
        r""" 
            Uses a threadpool executor to make concurrent calls to the miner.forward_call given inputs.
        """
        # ---- Loop until event is set -----
        logger.success('<white>Forward loop:</white> Started.')
        with concurrent.futures.ThreadPoolExecutor( max_workers = self.config.miner.max_forward_workers ) as executor:
            while not self.quit_forward.is_set():
                with self.get_forward_lock():
                    try:
                        # Submit next call.
                        pong, pubkey, inputs, modality = self.axon.next_forward_item( timeout = 1.0 )
                        if None not in [ pong, pubkey, inputs, modality]:
                            executor.submit( self.run_next_forward_call, pong, pubkey, inputs, modality )
                    except Exception as e:
                        logger.exception('Error in forward thread with error {}', e)
                        traceback.print_exc()
                        sys.exit()

    # ---- Backward loop -----
    def backward_loop ( self ): 
        r""" 
            Uses a threadpool executor to make concurrent calls to the miner.backward_call given inputs.
        """
        logger.success('<white>Backward loop:</white> Started')
        with concurrent.futures.ThreadPoolExecutor( max_workers = self.config.miner.max_backward_workers ) as executor:
            while not self.quit_backward.is_set():
                with self.get_backward_lock():
                    try:
                        # Submit backward call.
                        pong, pubkey, inputs_x, grads_dy, modality = self.axon.next_backward_item( timeout = 1.0 )
                        if None not in [ pong, pubkey, inputs_x, grads_dy, modality]:
                            executor.submit( self.run_next_backward_call, pong, pubkey, inputs_x, grads_dy, modality )
                    except Exception as e:
                        logger.exception('Error in backward thread with error {}', e)
                        traceback.print_exc()
                        sys.exit()

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
    def get_forward_lock( self ):
        if not hasattr(self, 'forward_lock'):
            self.forward_lock = mp.Lock()
        return self.forward_lock

    # ---- Get Backward lock ----
    def get_backward_lock( self ):
        if not hasattr(self, 'backward_lock'):
            self.backward_lock = mp.Lock()
        return self.backward_lock

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
        r""" This function is called by miner.run() if miner.should_save() returns True.
        """
        try:
            state_dict = self.get_state_dict()
            state_dict['epoch'] = self.epoch
            state_dict['epoch_loss'] = self.epoch_loss
            state_dict['global_step'] = self.global_step
            torch.save( state_dict, "{}/model.torch".format( self.config.miner.full_path, self.epoch_loss ))
            logger.success( 'Saved model to: <cyan>{}/model.torch</cyan>\n'.format( self.config.miner.full_path ))
        except Exception as e:
             logger.exception('Failed to save model with error:{}', e)

    def reload_state( self ):
        r""" Called by miner.run() if miner.should_reload() returns True.
        """
        try:
            state_dict = torch.load("{}/model.torch".format( self.config.miner.full_path ))
            self.reload_from_state_dict( state_dict )
            self.epoch = state_dict['epoch']
            self.epoch_loss = state_dict['epoch_loss']
            self.global_step = state_dict['global_step']
            logger.success( 'Reloaded model from: <cyan>{}/model.torch</cyan>\n'.format( self.config.miner.full_path ))
        except Exception as e:
            logger.exception('Failed to reload model with error: {}', e)

    def should_reload(self) -> bool:
        r""" Called by miner.run() after every epoch.
            If the function returns True the miner state dict is saved to miner.full_path.
            Returns:
                should_reload (bool):
                    False by default. Does not reload the miner state after each epoch.
        """
        if torch.any(torch.isnan(torch.cat([param.view(-1) for param in self.get_nucleus().parameters()]))):
            return True

    def sync_chain_state( self ):
        r""" Called after each training epoch. Miner should update chain state and resize objects.
        """
        try:
            self.metagraph.sync()
        except RuntimeError:
            logger.info('Failed to sync metagraph. Reloading from last saved state.')
            self.metagraph.load()
        self.metagraph.save()

    def epoch_logs(self):
        r""" Called by miner.run() after each epoch.
            Sends miner state to tensorboard.
        """
        self.axon.__to_tensorboard__( self.tensorboard, self.global_step )
        self.dendrite.__to_tensorboard__( self.tensorboard, self.global_step )
        self.metagraph.__to_tensorboard__( self.tensorboard, self.global_step )

    # --- Run Epoch ----
    def run_epoch( self ):
        r""" Called by miner.run(), calls training_call for passed batches.
        """
        # --- Init Epoch ----
        total_epoch_loss = 0.0
        training_batches = self.dataset.dataloader( self.config.miner.epoch_length )
        
        if self.config.rich and not self.config.debug: 
            # Rich display.
            progress = Progress(
                "[progress.description]{task.description}",
                BarColumn(),
                "{task.completed} of {task.total}",
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeRemainingColumn(),
                TimeElapsedColumn(),
                auto_refresh=False, expand=True
            )
            epoch_task = progress.add_task('[underline bold white]Epoch', total=len(training_batches))
            prev_row_weights = self.get_row_weights().tolist()
            with Live(self.update_rich_logs(progress, epoch_task, 0, None, prev_row_weights, prev_row_weights), refresh_per_second=4, screen=True) as live:
                for iteration, (inputs) in enumerate(training_batches):

                    # ---- Forward / Backward ----
                    prev_row_weights = self.get_row_weights().tolist()
                    input = copy.deepcopy(inputs)
                    del inputs
                    output = self.training_call( batch = { 'inputs': input } )
                    next_row_weights = self.get_row_weights().tolist()
                    total_epoch_loss += output.local_target_loss.item()

                    live.update(self.update_rich_logs(progress, epoch_task, iteration, output, prev_row_weights, next_row_weights))
                    self.global_step += 1
        else:

            # QQDM display.
            progress_bar = qqdm(enumerate(training_batches), total=len(training_batches), desc=format_str('blue', f'Epoch Progress'))
            for iteration, (inputs) in progress_bar:
                # ---- Forward / Backward ----
                prev_row_weights = self.get_row_weights().tolist()
                input = copy.deepcopy(inputs)
                del inputs
                output = self.training_call( batch = { 'inputs': input } )
                next_row_weights = self.get_row_weights().tolist()
                total_epoch_loss += output.local_target_loss.item()

                # ---- Update training state ----
                self.qqdm_logs( progress_bar, iteration = iteration, output = output, prev_row_weights = prev_row_weights, next_row_weights = next_row_weights )
                self.global_step += 1

        self.epoch_loss = total_epoch_loss / (iteration + 1) 
        self.epoch += 1

    # --- Run Miner ----
    def run( self ):
        r""" Miner main loop.
        """
        # ---- Setup ----
        with self:  

            # --- Setup run state ----
            self.epoch = 0
            self.global_step = 0 
            self.epoch_loss = math.inf
            self.last_saved_loss = math.inf

            # ---- Optionally reload ----
            if self.config.resume: 
                try:  
                    self.reload_state()
                except:
                    logger.warning("Failed to reload state. Starting from new model.")
                    self.save_state()
            else:
                self.save_state()  
        
            # --- Run until ----
            while self.should_run( self.epoch ):
                try:
                    
                    # ---- Synchronize with chain ----
                    self.sync_chain_state()

                    # ---- Train ----
                    self.run_epoch()

                    # ---- Save or Reload state ----
                    if self.should_save():
                        self.save_state()
                    elif self.should_reload():
                        self.reload_state()

                    # ---- Set weights on chain ----
                    self.set_mechanism_weights()

                    # ---- Update Tensorboard ----
                    self.epoch_logs() 
                
                except KeyboardInterrupt:
                    # User ended.
                    break

                except Exception as e:
                    logger.exception('Unknown exception: {} with traceback {}', e, traceback.format_exc())
                    if self.config.restart_on_failure == True:
                        logger.info('Restarting from last saved state.')
                        self.reload_state()
                        continue
                    else:
                        break

    # ---- QQDM Training logs ----
    def qqdm_logs( self, progress_bar, iteration:int, output: SimpleNamespace, prev_row_weights: List[float], next_row_weights: List[float] ):
        r""" Called by miner.run_training_epoch() after each training step.
            The function populates and displays the passed progress bar.
        """
        try:
            self_uid = self.metagraph.hotkeys.index( self.wallet.hotkey.public_key )
            stake = self.metagraph.S[ self_uid ].item()
            rank = self.metagraph.R[ self_uid ].item()
            incentive = self.metagraph.I[ self_uid ].item()
        except:
            stake = 0.0
            rank = 0.0
            incentive = 0.0
            pass
        info = {
            'GS': colored('{}'.format(self.global_step), 'red'),
            'LS': colored('{}'.format(iteration), 'blue'),
            'Epoch': colored('{}'.format(self.epoch+1), 'green'),
            'Epoch-loss': colored('{:.4f}'.format(self.epoch_loss), 'yellow'),
            'Saved-loss': colored('{:.4f}'.format(self.last_saved_loss), 'red'),
            'L-loss': colored('{:.4f}'.format(output.local_target_loss.item()), 'blue'),
            'R-loss': colored('{:.4f}'.format(output.remote_target_loss.item()), 'green'),
            'D-loss': colored('{:.4f}'.format(output.distillation_loss.item()), 'yellow'),
            'nPeers': colored(self.metagraph.n.item(), 'red'),
            'Stake(\u03C4)': colored('{:.3f}'.format(stake), 'green'),
            'Rank(\u03C4)': colored('{:.3f}'.format(rank), 'blue'),
            'Incentive(\u03C4/block)': colored('{:.6f}'.format(incentive), 'yellow'),
            'Axon': self.axon.__str__(),
            'Dendrite': self.dendrite.__str__(),
        } 
        for uid in self.metagraph.uids.tolist():
            if next_row_weights[uid] != 0:
                weight_dif = next_row_weights[uid] - prev_row_weights[uid]
                if weight_dif > 0:
                    info[colored(str(uid), 'green')] = colored('{:.4f}'.format(next_row_weights[uid]), 'green')
                elif weight_dif == 0:
                    info[str(uid)] = colored('{:.4f}'.format(next_row_weights[uid]), 'white')
                else:
                    info[colored(str(uid), 'red')] = colored('{:.4f}'.format(next_row_weights[uid]), 'red')

        progress_bar.set_infos( info )
        self.tensorboard.add_scalar('R-loss', output.remote_target_loss.item(), self.global_step)
        self.tensorboard.add_scalar('L-loss', output.local_target_loss.item(), self.global_step)
        self.tensorboard.add_scalar('D-loss', output.distillation_loss.item(), self.global_step)

    def update_rich_logs(self, progress, epoch_task, iteration, output, prev, next) -> Table:
            try:
                self_uid = self.metagraph.hotkeys.index( self.wallet.hotkey.public_key )
                stake = self.metagraph.S[ self_uid ].item()
                rank = self.metagraph.R[ self_uid ].item()
                incentive = self.metagraph.I[ self_uid ].item()
            except:
                self_uid = None
                stake = 0.0
                rank = 0.0
                incentive = 0.0
                pass
            if output == None:
                lloss = 0.0
                rloss = 0.0
                dloss = 0.0
            else:
                lloss = output.local_target_loss.item()
                rloss = output.remote_target_loss.item()
                dloss = output.distillation_loss.item()

            # Training State.
            Cols = [
                '[white]Epoch: [green]{}'.format(self.epoch+1),
                '[white]Global step: [bold red]{}'.format(self.global_step),
                '[white]Local step: [bold blue]{}'.format(iteration),
                '[white]Epoch-loss: [green]{:.4f}'.format(self.epoch_loss),
                '[white]Saved-loss: [green]{:.4f}'.format(self.last_saved_loss),
                '[white]Local-loss: [blue]{:.4f}'.format(lloss),
                '[white]Remote-loss: [green]{:.4f}'.format(rloss),
                '[white]Distillation-loss: [yellow]{:.4f}'.format(dloss),
                '[white]Stake: [bold white]\u03C4[green]{:.3f}'.format(stake),
                '[white]Rank: [bold white]\u03C4[blue]{:.3f}'.format(rank),
                '[white]Incentive: [bold white]\u03C4[yellow]{:.6f}[bold white]/block'.format(incentive),
                '[white]Axon: ' + self.axon.__rich__(),
                '[white]Dendrite: ' + self.dendrite.__rich__()
            ]
            columns = Columns( Cols, equal=True, expand=True)

            # Metagraph.
            if self.metagraph.n != 0:
                peers_online = torch.numel(torch.where( self.metagraph.block - self.metagraph.lastemit < 1000 )[0])
            else:
                peers_online = 0
            MetaCols = [
                '[white]block[/white]: [blue]{}[/blue]'.format(self.metagraph.block.item()),
                '[white]inflation_rate[/white]: [blue]{}[/blue]'.format(self.metagraph.tau.item()),
                '[white]staked[/white]: [bold white]\u03C4[green]{}[/green]/[bold white]\u03C4[blue]{}[/blue]'.format(torch.sum(self.metagraph.S), self.metagraph.block.item()/2),
                '[white]active[/white]: [green]{}[/green]/[blue]{}[/blue]'.format(peers_online, self.metagraph.n.item())
            ]
            meta_columns = Columns( MetaCols, equal=True, expand=True)

            # Add response codes column.
            code_cols = []
            if output != None:
                for (uid, req_size, code) in list(zip( output.router.uids.tolist(), output.router.request_sizes.tolist(), output.router.return_codes.tolist())):
                    code_string = bittensor.utils.codes.code_to_string(code)
                    code_color = bittensor.utils.codes.code_to_color(code)
                    code_cols.append( '[white]' + str(uid) + ' [' + code_color + ']' + code_string)
            code_col = Columns( code_cols, equal=True, expand=True)

            # Add weights column.
            weight_cols_vals = []
            for uid in range( self.metagraph.n.item() ):
                if next[uid] == 0:
                    continue
                if next[uid] > prev[uid]:
                    weight_cols_vals.append( '[bold white frame]' + str(uid) + ' [bold green frame]' + '{:.3}'.format(next[uid]))
                elif next[uid] == prev[uid]:
                    weight_cols_vals.append( '[dim white frame]' + str(uid) + ' [dim white frame]' + '{:.3}'.format(next[uid]))
                else:
                    weight_cols_vals.append( '[bold white frame]' + str(uid) + ' [dim red frame]' + '{:.3}'.format(next[uid]))
            weight_columns = Columns( weight_cols_vals, equal=True, expand=True)

            # Chain weights column.
            chain_weight_cols_vals = []
            if self_uid != None:
                for uid, weight in enumerate( self.metagraph.W[self_uid, :].tolist() ):
                    if weight != 0:
                        chain_weight_cols_vals.append( '[bold white frame]{} [dim green frame]{:.3}'.format(uid, weight) )
            chain_weight_columns = Columns( chain_weight_cols_vals, equal=True, expand=True)

            # Add progress bar.
            progress.update(epoch_task, advance=1, refresh=True)
            
            if self.config.use_tensorboard == True:
                tensorboard_str = '[white]tensorboard[/white]:[green]ON[/green] endpoint: [blue]http://localhost:6006/[/blue]'
            else:
                tensorboard_str = ''
            
            if self.config.record_log == True:
                filepath = self.config.miner.full_path + "/bittensor_output.log"
                logging_str = '[white]logging[/white]:[green]ON[/green] sink: [blue]{}[/blue]'.format(filepath)
            else: 
                logging_str = ''

            group = RenderGroup( progress, '\n[underline bold white]Training:\n', columns, "\n[underline bold white]Response Codes:\n", code_col, "\n[underline bold white]Row Weights:\n", weight_columns, "\n[underline bold white]Chain Weights:\n", chain_weight_columns, '\n[underline bold white]Metagraph:\n', meta_columns, "\n[underline bold white]Extra\n", tensorboard_str, logging_str)
            
            return group


