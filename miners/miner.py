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
import threading
import math
import torch
import traceback
import concurrent.futures
import multiprocessing as mp
import sys

from qqdm import qqdm, format_str
from rich.live import Live
from rich.table import Table
from rich.console import RenderGroup
from rich.columns import Columns
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TimeElapsedColumn

from termcolor import colored
from types import SimpleNamespace
from typing import List
from munch import Munch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tensorboard import program
from loguru import logger
logger = logger.opt(colors=True)

class AbstractMiner ():

    def __init__( self, config: 'bittensor.Config' = None, **kwargs):
        r""" Initializes a new base miner object.
            
            Args:
                config (:obj:`bittensor.Config`, `optional`): 
                    miner.Miner.default_config()
        """
        if config == None:
            config = AbstractMiner.default_config()
        config = copy.deepcopy( config )
        AbstractMiner.check_config( config )
        self.config = config
        self.wallet = bittensor.wallet ( config = self.config )
        self.subtensor = bittensor.subtensor( config = self.config )
        self.metagraph = bittensor.metagraph( config = config )
        self.axon = bittensor.axon( config = self.config, wallet = self.wallet )
        self.dendrite = bittensor.dendrite( config = self.config, wallet = self.wallet )

    @staticmethod   
    def default_config() -> 'bittensor.Config':
        # Parses and returns a config Munch for this object.
        parser = argparse.ArgumentParser(); 
        AbstractMiner.add_args(parser) 
        config = bittensor.config( parser ); 
        return config

    @staticmethod
    def check_config(config: 'bittensor.Config'):
        assert 'name' in config.miner, 'miners must specify a name argument.'
        bittensor.wallet.check_config( config )
        bittensor.subtensor.check_config( config )
        bittensor.axon.check_config( config )
        bittensor.dendrite.check_config( config )
        bittensor.metagraph.check_config( config )
        full_path = os.path.expanduser('{}/{}/{}'.format( config.miner.root_dir, config.wallet.name + "-" + config.wallet.hotkey, config.miner.name ))
        config.miner.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.miner.full_path):
            os.makedirs(config.miner.full_path)

    @staticmethod   
    def add_args( parser: argparse.ArgumentParser ):
        bittensor.wallet.add_args( parser )
        bittensor.subtensor.add_args( parser )
        bittensor.axon.add_args( parser )
        bittensor.dendrite.add_args( parser )
        bittensor.metagraph.add_args( parser )
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
        self.axon.stop()

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

    def reconnect_to_chain( self ):
        self.subtensor = bittensor.subtensor( config = self.config ) # Re-create subtensor object.
        self.connect_to_chain()

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


class BasicMiner( AbstractMiner ):

    def __init__( self, config: 'bittensor.Config' = None, **kwargs ):
        r""" Initializes a new basic miner object.
            
            Args:
                config (:obj:`bittensor.Config`, `optional`): 
                    miner.BaseMiner.default_config()
        """
        if config == None:
            config = BasicMiner.default_config()
        config = copy.deepcopy( config );
        BasicMiner.check_config( config )
        self.config = config
        super( BasicMiner, self ).__init__( self.config, **kwargs )

    @staticmethod   
    def default_config() -> 'bittensor.Config':
        parser = argparse.ArgumentParser(); 
        BasicMiner.add_args(parser) 
        config = bittensor.config( parser ); 
        return config
    
    @staticmethod
    def check_config(config: 'bittensor.Config'):
        AbstractMiner.check_config( config )

    @staticmethod   
    def add_args( parser: argparse.ArgumentParser ):
        AbstractMiner.add_args( parser )
        parser.add_argument ('--miner.resume', dest='resume', action='store_true', help='''Resume training from last save state.''', default=False )
        parser.add_argument ('--miner.rich', dest='rich', action='store_true', help='''Rich text display''', default=False)
        parser.add_argument ('--miner.restart_on_failure', dest='restart_on_failure', action='store_true', help='''Restart miner on unknown error.''', default=False)
        parser.add_argument ('--miner.max_backward_workers', default='10', type=int, help='Maximum number of concurrent backward processing threads.')
        parser.add_argument ('--miner.max_forward_workers', default='10', type=int, help='Maximum number of concurrent forward processing threads.')

    def init_axon( self ):
        # ---- Starting axon ----
        logger.info('\nStarting Axon...')
        self.axon.attach( self )
        self.axon.start()

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

    # ---- Training call ----
    def train ( self, batch: dict ) -> SimpleNamespace:
        r""" Runs a single training batch through the nucleus and applies a gradient update.
            Args:
                batch ( dict, `required`): 
                    training batch dictionary as returned from get_epoch_batches            
            Returns:
                output = SimpleNamespace ( 
                    local_context (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Hidden layer context.

                    local_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Hidden layer encoding produced using local_context.

                    local_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__vocab_size__)`, `optional`):
                        GPT MLM Target predictions produced using local_context. 

                    local_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        GPT MLM loss using local_context.

                    remote_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `optional`): 
                        Hidden layer encoding produced using the remote_context.

                    remote_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,  bittensor.__vocab_size__)`, `optional`):
                        GPT MLM Target predictions using the remote_context.

                    remote_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`):
                        GPT MLM loss using the remote_context.

                    distillation_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        Distillation loss between local_context and remote_context.

                    router (:obj:`SimpleNamespace`, `required`): 
                        Output simplenamespace from routing call.
            )
        """
        raise NotImplementedError()

    # ---- Subclass Forward call ----
    def forward ( self, pubkey:str, inputs:torch.FloatTensor, modality:int ) -> torch.FloatTensor:
        r""" Recieves and processes forward calls for a bittensor.axon.
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
    def backward ( self, pubkey:str, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, modality:int ) -> torch.FloatTensor:
        r""" Recieves and processes backward calls for a bittensor.axon.
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
    def get_row_weights( self ) -> torch.float32:
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

    def set_mechanism_weights( self ):
        r""" Called after every training epoch, sets the row_weights into the incentive mechanism on chain.
        """
        try:
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
                self.reconnect_to_chain()

        except Exception as e:
            logger.error('Failure setting weights on chain with error: {}', e)
        
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
        except Exception as e:
            logger.info('Failed to sync metagraph. Reloading from last saved state. With error: {}', e)
            self.metagraph.load()
            self.reconnect_to_chain()
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
        r""" Called by miner.run(), calls train for passed batches.
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
                    output = self.train ( batch = { 'inputs': inputs } )
                    next_row_weights = self.get_row_weights().tolist()
                    total_epoch_loss += output.local_target_loss.item()

                    live.update(self.update_rich_logs(progress, epoch_task, iteration, output, prev_row_weights, next_row_weights))
                    self.global_step += 1
        else:

            # QQDM display.ss
            progress_bar = qqdm(enumerate(training_batches), total=len(training_batches), desc=format_str('blue', f'Epoch Progress'))
            for iteration, (inputs) in progress_bar:
                # ---- Forward / Backward ----
                prev_row_weights = self.get_row_weights().tolist()
                output = self.train ( batch = { 'inputs': inputs } )
                next_row_weights = self.get_row_weights().tolist()
                total_epoch_loss += output.local_target_loss.item()

                # ---- Update training state ----
                self.qqdm_logs( progress_bar, iteration = iteration, output = output, prev_row_weights = prev_row_weights, next_row_weights = next_row_weights )
                self.global_step += 1

        self.epoch_loss = total_epoch_loss / (iteration + 1) 
        self.epoch += 1

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
