#!/bin/python3
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
""" Base neuron version 1.

Example:
    $ import neurons
    $ neurons.text.base_neuron_v1().run()

"""

from re import I
import pandas
from pandas.core.frame import DataFrame
import bittensor
import math
import torch
import traceback
import sys
import wandb
from termcolor import colored
from qqdm import qqdm, format_str
from loguru import logger

from bittensor._metagraph import metagraph
logger = logger.opt(colors=True)

from types import SimpleNamespace
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
from functools import partial

import torch.nn.functional as F

class BaseServer():

    def __init__(self):
        pass

    def __enter__(self):
        self.axon_start()

    def __exit__(self):
        self.axon_stop() 
    
    def axon_start(self):
        self.axon = bittensor.axon (
            config = self.config,
            wallet = self.wallet,
            forward_text = self.forward_text,
            backward_text = self.backward_text,
            blacklist = self.blacklist,
        )

        self.axon.start().serve (
            use_upnpc = self.config.neuron.use_upnpc, 
            subtensor = self.subtensor
        )

    def axon_stop(self):
        self.axon.stop()

    def init_check(self):
        raise NotImplementedError

    # ---- Axon Forward call ----
    def forward_text ( self, inputs_x: torch.FloatTensor) -> torch.FloatTensor:
        r""" Subscribed to an axon servicing endpoint: processes forward messages from the wire.
            The arguments reflect an RPC request from another miner in the network, the response tensor
            should be the hidden units computed using the local context and with shape: [batch_size, sequence_len, __network_dim__].

            Args:
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.

            Returns:
                outputs (:obj:`torch.FloatTensor`):
                    The nucleus's outputs as a torch tensor of shape [batch_size, sequence_len, __network_dim__]
        """
        output = self.nucleus.forward(
            forward_type = 'local',
            inputs = inputs_x.to( self.device )
        )
        return output.local_hidden

    # ---- Axon Backward call ----
    def backward_text ( self, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor ):
        r""" Subscribed to an axon servicing endpoint: Processes backward messages from the wire.
            Arguments reflect an RPC backward request from another miner in the network. No response
            needed for tokenized text inputs (uint64s have no gradient).

            Args:
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs from previous forward call.
                grads_dy ( :obj:`torch.Tensor`, `required`):
                    torch grads of forward output.                    
        """
        if self.config.neuron.accumulate_remote_gradients:
            with torch.enable_grad():
                # ---- Set up inputs for gradient computations.
                outputs_y = self.nucleus.local_forward( inputs = inputs_x.to( self.device ) ).local_context.to( self.device )
                # ---- The backward call will accumulate gradients on our parameters.
                torch.autograd.backward (
                    tensors = [outputs_y],
                    grad_tensors = [grads_dy.to( self.device )]
                )
    
    def priority(self, pubkey:str, request_type:bittensor.proto.RequestType, inputs_x: torch.FloatTensor) -> float:
        r"""Return the request priority based on stake and size of input. 
            Used by the Axon to order requests.
            Args:
                pubkey ( str, `required`):
                    The public ss58 address of the caller.
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.
                request_type ( bittensor.proto.RequestType, `required`):
                    the request type ('FORWARD' or 'BACKWARD').
        """        
        # Priority = stake / request_size 
        priority = self.metagraph.S[ self.metagraph.hotkeys.index(pubkey) ] / sys.getsizeof(inputs_x)
        return priority

    def blacklist(self, pubkey:str, request_type:bittensor.proto.RequestType) -> bool:
        r"""Axon security blacklisting, used to blacklist message from low stake members
            Currently, this is not turned on.
            Args:
                pubkey ( str, `required`):
                    The public key of the caller.
                request_type ( bittensor.proto.RequestType, `required`):
                    the request type ('FORWARD' or 'BACKWARD').
        """
        # Blacklist requests from peers who are not subscribed or have stake less that black_list
        is_registered = pubkey in self.metagraph.hotkeys

        # If we allow non-registered requests return False = not blacklisted.
        if not is_registered:
            if self.config.neuron.blacklist_allow_non_registered:
                return False
            else:
                return True
        else:
            # Else, get stake and check is above blacklist stake min.
            uid = self.metagraph.hotkeys.index( pubkey )
            if self.metagraph.S[uid].item() >= self.config.neuron.blacklist:
                return False
            else:
                return True

class BaseNeuron():
    def __init__( self, config: 'bittensor.config', nucleus: 'Nucleus'):

        self.config = config
        self.wallet = bittensor.wallet ( config = self.config )
        self.subtensor = bittensor.subtensor ( config = self.config )
        self.metagraph = bittensor.metagraph ( config = self.config, subtensor = self.subtensor )
        self.dendrite = bittensor.dendrite ( config = self.config, wallet = self.wallet )
        self.dataset = bittensor.dataset ( config = self.config )
        self.device = torch.device( device = self.config.neuron.device )

        self.nucleus.metagraph = lambda : self.metagraph
        self.nucleus.dendrite = self.dendrite

    def save( self ):
        r""" Saves the training state to disk.
        """
        try:
            state_dict = {
                'epoch': self.epoch,
                'epoch_loss': self.stats.local_target_epoch_loss,
                'global_step': self.stats.global_step,
                'nucleus_state': self.nucleus.state_dict(), # Save nucleus state.
                'optimizer_state': self.optimizer.state_dict(), # Save optimizer.
                'network': self.subtensor.network # Save Network
            }
            torch.save( state_dict, "{}/model.torch".format( self.config.neuron.full_path ) )
            bittensor.logging.success(prefix='Saved model', sufix='<blue>{}/model.torch</blue>'.format( self.config.neuron.full_path ) )
        except Exception as e:
            logger.exception('Failed to save model with error:{}', e)

    def get_saved_state( self ):
        r""" Returns a saved state dict or none.
        """
        try:
            return torch.load("{}/model.torch".format( self.config.neuron.full_path ))
        except Exception as e:
            logger.warning('No saved model found with error: {}', e)
            return None

    def reload_stat(self, state_dict):
        self.epoch = state_dict['epoch']
        self.stats.local_target_epoch_loss = state_dict['epoch_loss']
        self.stats.best_epoch_loss = state_dict['epoch_loss']
        self.stats.global_step = state_dict['global_step']

    def reload_nucleus(self, state_dict):

        # ---- Model parameters
        try:
            self.nucleus.load_state_dict( state_dict['nucleus_state'], strict=False )
        except Exception as e:
            logger.exception('Failed to load nucleus state with error, updating the current state')
            state_dict['nucleus_state'] = self.nucleus.state_dict()
            torch.save( state_dict, "{}/model.torch".format( self.config.neuron.full_path ) )

        # ---- Optimizer
        self.optimizer = torch.optim.SGD(
            [{"params": self.nucleus.parameters()}],
            lr = state_dict['optimizer_state']['param_groups'][0]['lr'],
            momentum = state_dict['optimizer_state']['param_groups'][0]['momentum'],
        )

        # ---- Peer weight
        self.nucleus.peer_weights = nn.Parameter(
            torch.ones(
                list(state_dict['nucleus_state']['peer_weights'].shape),
                requires_grad=True
            ).to(self.device)
        )
        chain_growth = max(0, self.metagraph.n.item() - state_dict['nucleus_state']['peer_weights'].shape[0])
        self.nucleus.peer_weights = nn.Parameter(torch.cat([self.nucleus.peer_weights, torch.ones([chain_growth],dtype=torch.float32,requires_grad=True, device = self.device)]))

        # ---- device, dendrite, metagraph
        self.nucleus.to( self.device ) # Load nucleus
        self.nucleus.dendrite = self.dendrite # Set local dendrite.
        self.nucleus.metagraph = self.metagraph_callback # Set local metagraph.
        bittensor.logging.success( prefix = 'Reloaded model', sufix = '<blue>{}/model.torch</blue>'.format( self.config.neuron.full_path ))        

    def reload( self ):
        r""" Reloads/updates the training state from the disk.
        """        
        # --- Loads and syncs metagraph.
        self.meta_sync()

        # ---- Load training state.
        state_dict = self.get_saved_state()
        self.reload_stat(state_dict)

    def checkpoint( self ):
        r""" Optionally Saves, updates and then reloads the miner training state.
        """
        last_saved = self.get_saved_state()
        if last_saved == None or last_saved['epoch_loss'] >= self.stats.local_target_epoch_loss:
            self.stats.best_epoch_loss = self.stats.local_target_epoch_loss
            self.save()

        # Checks if epochs managed to diverage
        if not math.isfinite(self.stats.local_target_epoch_loss):
            logger.error('Incorrect epoch loss detected, reloading to previous saved state')
            self.reload()

    def meta_sync (self, current_block ):
        """ Miner sync with metagraph and update chain weight
        """
        # ---- Sync with metagraph ----
        self.metagraph.sync()
        self.stats.last_sync_block= self.subtensor.get_current_block()
        chain_growth = max(self.metagraph.n.item()- self.nucleus.peer_weights.shape[0], 0)
        self.nucleus.peer_weights = nn.Parameter(torch.cat([self.nucleus.peer_weights, torch.ones([chain_growth],dtype=torch.float32,requires_grad=True).to(self.device)]))
        self.stats.scores = torch.nn.Parameter(torch.cat( [self.stats.scores, torch.zeros([chain_growth], dtype=torch.float32, requires_grad=False).to(self.device)]))
        self.stats.ema_scores = torch.nn.Parameter(torch.cat( [self.stats.ema_scores, torch.zeros([chain_growth], dtype=torch.float32, requires_grad=False).to(self.device)]))
        bittensor.logging.success( 'Synced metagraph:', 'Block: {}'.format(current_block))

    def set_peer_weights( self ):
        r""" Sets the fisher ema score to peers.
        """

        try:
            k = min( self.config.neuron.n_topk_peer_weights, self.metagraph.n.item() )
            inactive_uids = torch.where(self.metagraph.active == 0)[0]
            self.stats.ema_scores[inactive_uids] = 0
            topk_scores, topk_uids = bittensor.unbiased_topk( self.stats.ema_scores , k = k )
            topk_uids = topk_uids.detach().to('cpu')
            topk_scores = topk_scores.detach().to('cpu')
            self.subtensor.set_weights(
                uids = topk_uids,
                weights = topk_scores,
                wait_for_inclusion = False,
                wallet = self.wallet,
            )

        except Exception as e:
            logger.error('Failure setting weights on chain with error: {}', e)

class BaseNeuronLog():
    def __init__(self):
        self.stats = SimpleNamespace(
            global_step = 0,
            last_sync_block = 0,
            epoch_data_size = 0,
            epoch_sync_count = 0,
            local_target_epoch_loss = math.inf,
            distillation_epoch_loss = math.inf,
            remote_target_epoch_loss = math.inf,
            local_epoch_acc = 0,
            best_epoch_loss = math.inf,
            scores = torch.nn.Parameter(torch.zeros(0), requires_grad = False).to(self.device),
            ema_scores = torch.nn.Parameter(torch.zeros(0), requires_grad = False).to(self.device)
        )
        
    # ---- Training logs ----
    def logs( self, progress_bar, iteration:int, output: SimpleNamespace ):
        r""" Called after every training step. Displays miner state to screen.
        """
        self_neuron = self.subtensor.neuron_for_pubkey( self.wallet.hotkey.ss58_address )
        self_uid = self_neuron.uid
        stake = self_neuron.stake
        rank = self_neuron.rank
        incentive = self_neuron.incentive
        normalized_peer_weights = F.softmax (self.nucleus.peer_weights.detach(), dim=0)
        current_block = self.subtensor.get_current_block()

        # ---- Progress bar log
        info = {
            'Step': colored('{}'.format(self.stats.global_step), 'red'),
            'Epoch': colored('{}'.format(self.epoch+1), 'yellow'),
            'Best-loss': colored('{:.4f}'.format(self.stats.best_epoch_loss), 'green'),          
            'L-loss': colored('{:.4f}'.format(output.local_target_loss.item()), 'blue'),
            'R-loss': colored('{:.4f}'.format(output.remote_target_loss.item()), 'red'),
            'D-loss': colored('{:.4f}'.format(output.distillation_loss.item()), 'yellow'),
            'L-acc': colored('{:.4f}'.format(output.local_accuracy), 'green'),
            'nPeers': colored(self.metagraph.n.item(), 'blue'),
            'Stake(\u03C4)': colored('{:.3f}'.format(stake), 'red'),
            'Rank(\u03C4)': colored('{:.3f}'.format(rank), 'yellow'),
            'Incentive(\u03C4/block)': colored('{:.6f}'.format(incentive), 'green'),
            'Current Block': colored('{}'.format(current_block), 'blue'),
            'Synced Block': colored('{}'.format(self.stats.last_sync_block), 'yellow'),
        }
        # ---- Miner summary per peer for progress bar
        k = min( self.config.neuron.n_topk_peer_weights, self.metagraph.n.item() )
        topk_scores, topk_uids = bittensor.unbiased_topk( self.stats.ema_scores, k, dim=0 )
        for uid, ema_score in zip( topk_uids, topk_scores ) :
            color =  'green' if self.stats.scores[uid] - ema_score > 0 else 'red'
            info[f'uid_{uid.item()}'] = colored('{:.4f}'.format(ema_score), color)

        progress_bar.set_infos( info )

        # ---- wandb log if it is the end of epoch 
        if self.config.neuron.use_wandb and ((iteration + 1) % (self.config.neuron.epoch_length ) == 0):
            # ---- Miner summary for wandb
            wandb_info = {
                'neuron/stake':stake,
                'neuron/rank':rank,
                'neuron/incentive':incentive,
                'neuron/num_peers':self.metagraph.n.item(),
                'nucleus/remote_target_epoch_loss': self.stats.remote_target_epoch_loss,
                'nucleus/distillation_epoch_loss': self.stats.distillation_epoch_loss,
                'nucleus/local_target_epoch_loss': self.stats.local_target_epoch_loss,
                'nucleus/local_epoch_acc': self.stats.local_epoch_acc,
                'neuron/num_sync_metagraph': self.stats.epoch_sync_count,
                'neuron/data_size': self.stats.epoch_data_size,
            }

            # Build stats dataframe.
            df = pandas.concat( [
                bittensor.utils.indexed_values_to_dataframe( prefix = 'fisher_ema_score', index = topk_uids, values = self.stats.ema_scores, filter_zeros = True),
                bittensor.utils.indexed_values_to_dataframe( prefix = 'raw_peer_weight', index = topk_uids, values = self.nucleus.peer_weights, filter_zeros = True),
                bittensor.utils.indexed_values_to_dataframe( prefix = 'normalized_peer_weight', index = topk_uids, values = normalized_peer_weights, filter_zeros = True),
                bittensor.utils.indexed_values_to_dataframe( prefix = 'w_{}_i'.format(self_uid), index = topk_uids, values = self.metagraph.W[ self_uid, : ], filter_zeros = True),
                bittensor.utils.indexed_values_to_dataframe( prefix = 'w_i_{}'.format(self_uid), index = topk_uids, values = self.metagraph.W[ :, self_uid ], filter_zeros = True),
                self.axon.to_dataframe( metagraph = self.metagraph ),
                self.dendrite.to_dataframe( metagraph = self.metagraph )
            ], axis = 1)
            df['uid'] = df.index
            stats_data_table = wandb.Table( dataframe = df)

            wandb_info_axon = self.axon.to_wandb()
            wandb_info_dend = self.dendrite.to_wandb()
            wandb.log( { **wandb_info, **wandb_info_axon, **wandb_info_dend }, step = current_block)
            wandb.log( { 'stats': stats_data_table}, step = current_block)
            wandb.log( { 'axon_query_times': wandb.plot.scatter( stats_data_table, "uid", "axon_query_time", title="Axon Query time vs UID") } )
            wandb.log( { 'dendrite_query_times': wandb.plot.scatter( stats_data_table, "uid", "dendrite_query_time", title="Dendrite Query time vs UID") } )
