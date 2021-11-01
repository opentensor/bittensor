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

import bittensor
import math
import torch
import traceback
import sys
import wandb

from termcolor import colored
from qqdm import qqdm, format_str
from loguru import logger; logger = logger.opt(colors=True)
from types import SimpleNamespace
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
from functools import partial

import torch.nn.functional as F

class Neuron:

    def __init__( self, config: 'bittensor.config', nucleus: 'Nucleus'):
        r""" Initializes the neuron with the passed config.
        """
        self.config = config
        self.wallet = bittensor.wallet ( config = self.config )
        self.subtensor = bittensor.subtensor ( config = self.config )
        self.metagraph = bittensor.metagraph ( config = self.config, subtensor = self.subtensor )
        self.dendrite = bittensor.dendrite ( config = self.config, wallet = self.wallet )
        self.dataset = bittensor.dataset ( config = self.config )
        self.axon = bittensor.axon (
            config = self.config,
            wallet = self.wallet,
            forward_text = self.forward_text,
            backward_text = self.backward_text,
            blacklist = self.blacklist,
        )
        self.device = torch.device( device = self.config.neuron.device )
        self.nucleus = nucleus
        self.nucleus.metagraph = self.metagraph
        self.nucleus.dendrite = self.dendrite
        self.optimizer = torch.optim.SGD(
            [ {'params': self.nucleus.peer_weights, 'lr': self.config.neuron.learning_rate_chain} ],
            lr = self.config.neuron.learning_rate,
            momentum = self.config.neuron.momentum,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
            step_size = 1.0,
            gamma = 0.95
        )
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
            ema_scores = torch.nn.Parameter(torch.ones(0), requires_grad = False).to(self.device)
        )
        # ---- Decay factor for fisher ema score 
        self.fisher_ema_decay = 0.995

    def __enter__(self):
        self.wallet.create()
        self.metagraph.sync().save()
        self.axon.start().serve (
            use_upnpc = self.config.neuron.use_upnpc, 
            subtensor = self.subtensor
        )

    def __exit__ ( self, exc_type, exc_value, exc_traceback ):
        self.axon.stop()   
        print(exc_type, exc_value, exc_traceback)
    
    def run( self ):
        r""" Miner main loop.
        """
        # ---- Build Bittensor neuron ----
        self.wallet.register()
        with self:
            if self.config.neuron.use_wandb:
                bittensor.wandb(
                    config = self.config,
                    cold_pubkey = self.wallet.coldkeypub.ss58_address,
                    hot_pubkey = self.wallet.hotkey.ss58_address,
                    root_dir = self.config.neuron.full_path
                )

            # ---- Init run state ----
            self.epoch = 0            
            self.stats.ema_scores = torch.ones( self.metagraph.n.item()).to(self.device) * (1 / self.metagraph.n.item())

            # ---- reloads previous run if not restart ----
            if self.config.neuron.restart:
                self.save()

            try:
                self.reload()
                self.axon.check()
            except Exception as e:
                logger.error("Error when trying to reload model: {}".format(e))
                self.save()
                self.reload()
                self.axon.check()
            
            # --- Run until n_epochs ----
            while self.epoch < self.config.neuron.n_epochs:
                try:

                    # --- Init epoch stat----
                    self.stats.epoch_data_size = 0
                    self.stats.epoch_sync_count = 0
                    total_local_target_epoch_loss = 0
                    total_distillation_epoch_loss = 0
                    total_remote_target_epoch_loss = 0
                    total_local_epoch_acc = 0
                    batches_count = 0

                    # ---- Run epoch ----
                    start_block = self.subtensor.get_current_block() + 1
                    end_block = start_block + self.config.neuron.epoch_length
                    block_steps = [ block_delta for block_delta in range(start_block, end_block)]
                    progress_bar = qqdm( block_steps, total=len(block_steps), desc=format_str('blue', f'Epoch:'))
                    progress_bar.set_bar = partial(progress_bar.set_bar,  element='#')
                    for block in progress_bar:

                        # --- Iterate over batches until the end of the block.
                        current_block = self.subtensor.get_current_block()
                        while block >= current_block:
                            
                            # ---- Forward pass ----
                            inputs = next( self.dataset )
                            output = self.nucleus.remote_forward (
                                inputs = inputs.to( self.device ),
                                training = True,
                            )
                            
                            # ---- Backward pass ----
                            output.loss = output.local_target_loss + output.distillation_loss + output.remote_target_loss
                            scores = torch.nn.functional.normalize ( torch.relu( self.nucleus.compute_scores(output.remote_target_loss) ), p=1, dim = 0 )
                            output.loss.backward() # Accumulates gradients on the nucleus.
                            clip_grad_norm_(self.nucleus.parameters(), self.config.neuron.clip_gradients)
                            
                            # ---- Apply and zero accumulated gradients.
                            self.optimizer.step() 
                            self.optimizer.zero_grad()
                            current_block = self.subtensor.get_current_block()
                            
                            # ---- Aggrigate outputs and losses 
                            total_local_target_epoch_loss += output.local_target_loss.item()
                            total_distillation_epoch_loss += output.distillation_loss.item()
                            total_remote_target_epoch_loss += output.remote_target_loss.item()
                            total_local_epoch_acc += output.local_accuracy
                            self.stats.epoch_data_size += inputs.nelement()
                            batches_count += 1
                            
                            # ---- Expand ema_scores tensor if the chain grew and aggrigate the score
                            chain_growth = scores.shape[0] - self.stats.ema_scores.shape[0]
                            if chain_growth > 0:
                                self.stats.ema_scores = torch.nn.Parameter(torch.cat( [self.stats.ema_scores, torch.zeros([chain_growth], dtype=torch.float32, requires_grad=True)]))
                            self.stats.ema_scores = self.fisher_ema_decay * self.stats.ema_scores + (1 - self.fisher_ema_decay) * scores
                            self.stats.scores = scores


                        # ---- Sync with metagraph if the current block >= last synced block + sync block time 
                        current_block = self.subtensor.get_current_block()
                        block_diff = current_block - self.stats.last_sync_block
                        if block_diff >= self.config.neuron.sync_block_time:
                            self.sync(current_block)                                                                                                                
                            self.stats.last_sync_block = current_block
                            self.stats.epoch_sync_count += 1
                            
                        # ---- Update the epoch loss if it is the last iteration within epoch
                        if block+1 == end_block :
                            self.stats.local_target_epoch_loss = total_local_target_epoch_loss / batches_count
                            self.stats.distillation_epoch_loss = total_distillation_epoch_loss / batches_count
                            self.stats.remote_target_epoch_loss = total_remote_target_epoch_loss / batches_count
                            self.stats.local_epoch_acc = total_local_epoch_acc / batches_count

                        # ---- Block logs.
                        self.logs (
                            progress_bar,
                            iteration = block-start_block,
                            output = output,
                        )
                        self.stats.global_step += 1

                    # ---- Update params ----
                    self.epoch += 1

                    # ---- Checkpoint state ----
                    self.checkpoint()

                except KeyboardInterrupt:
                    # --- User ended session ----
                    break

                except Exception as e:
                    # --- Unknown error ----
                    logger.exception('Unknown exception: {} with traceback {}', e, traceback.format_exc())
                    if self.config.neuron.restart_on_failure == True:
                        logger.info('Restarting from last saved state.')
                        self.reload()
                    else:
                        break

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
        output = self.nucleus.local_forward (
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
    
    def priority(self, pubkey:str, request_type:str, inputs_x: torch.FloatTensor) -> float:
        r"""Return the request priority based on stake and size of input. 
            Used by the Axon to order requests.
            Args:
                pubkey ( str, `required`):
                    The public ss58 address of the caller.
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.
                request_type ( str, `required`):
                    the request type ('forward' or 'backward').
        """        
        # Priority = stake / request_size 
        priority = self.metagraph.S[ self.metagraph.hotkeys.index(pubkey) ] / sys.getsizeof(inputs_x)
        return priority

    def blacklist(self, pubkey:str, request_type:str) -> bool:
        r"""Axon security blacklisting, used to blacklist message from low stake members
            Currently, this is not turned on.
            Args:
                pubkey ( str, `required`):
                    The public key of the caller.
                request_type ( str, `required`):
                    the request type ('forward' or 'backward').
        """
        # Blacklist requests from peers who are not subscribed or have stake less that black_list
        uid = self.metagraph.hotkeys.index(pubkey)
        if self.metagraph.S[uid].item() < self.config.neuron.blacklist:
            return True
        else:
            return False

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

    def get_saved_state( self ):
        r""" Returns a saved state dict or none.
        """
        try:
            return torch.load("{}/model.torch".format( self.config.neuron.full_path ))
        except Exception as e:
            logger.warning('No saved model found with error: {}', e)
            logger.info('Initalizing with new model')
            return None

    def reload( self ):
        r""" Reloads/updates the training state from the disk.
        """
        state_dict = self.get_saved_state()
        self.metagraph.sync().save()

        # ---- Load training state.
        self.epoch = state_dict['epoch']
        self.stats.local_target_epoch_loss = state_dict['epoch_loss']
        self.stats.best_epoch_loss = state_dict['epoch_loss']
        self.stats.global_step = state_dict['global_step']

        # --- Updates the shape of nucleus chain weights
        chain_growth = self.metagraph.n.item() - state_dict['nucleus_state']['peer_weights'].shape[0]
        self.nucleus.peer_weights = nn.Parameter(
            torch.ones(
                list(state_dict['nucleus_state']['peer_weights'].shape),
                requires_grad=True
            ).to(self.device)
        )

        self.nucleus.load_state_dict( state_dict['nucleus_state'], strict=False )
        self.nucleus.peer_weights = nn.Parameter(torch.cat([self.nucleus.peer_weights, torch.ones([chain_growth],dtype=torch.float32,requires_grad=True).to(self.device)]))
        self.nucleus.to( self.device ) # Load nucleus
        self.nucleus.dendrite = self.dendrite # Set local dendrite.
        self.nucleus.metagraph = self.metagraph # Set local metagraph.
        self.optimizer = torch.optim.SGD(
            [{"params": self.nucleus.parameters()}],
            lr = state_dict['optimizer_state']['param_groups'][0]['lr'],
            momentum = state_dict['optimizer_state']['param_groups'][0]['momentum'],
        )
        bittensor.logging.success( prefix = 'Reloaded model', sufix = '<blue>{}/model.torch</blue>'.format( self.config.neuron.full_path ))


    def sync (self, current_block ):
        """ Miner sync with metagraph and update chain weight
        """
        # ---- Set weights on chain ----
        self.set_peer_weights()

        # ---- Sync with metagraph ----
        self.metagraph.sync().save()
        chain_growth = self.metagraph.n.item()- self.nucleus.peer_weights.shape[0]
        self.nucleus.peer_weights = nn.Parameter(torch.cat([self.nucleus.peer_weights, torch.ones([chain_growth],dtype=torch.float32,requires_grad=True).to(self.device)]))
        self.stats.ema_scores = torch.nn.Parameter(torch.cat( [self.stats.ema_scores, torch.ones([chain_growth], dtype=torch.float32, requires_grad=True).to(self.device)]))
        bittensor.logging.success( 'Synced metagraph:', 'Block: {}'.format(current_block))

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

    def set_peer_weights( self ):
        r""" Sets the fisher ema score to peers.
        """

        try:
            k = min(self.config.neuron.n_topk_peer_weights, self.metagraph.n.item())
            topk_scores, topk_uids = torch.topk( self.stats.ema_scores.detach(), k = k )
            did_set = self.subtensor.timeout_set_weights(
                timeout=100,
                uids = topk_uids,
                weights = topk_scores,
                wait_for_inclusion = True,
                wallet = self.wallet,
            )
            if did_set:
                bittensor.logging.success(prefix='Set weights:', sufix='{}'.format(list(zip(topk_scores, topk_uids))))
            else:
                logger.error('Failed to set weights on chain. (Timeout)')

        except Exception as e:
            logger.error('Failure setting weights on chain with error: {}', e)

    # ---- Training logs ----
    def logs( self, progress_bar, iteration:int, output: SimpleNamespace ):
        r""" Called after every training step. Displays miner state to screen.
        """
        self_uid = self.metagraph.hotkey_to_uid(self.wallet.hotkey.ss58_address)
        stake = self.metagraph.S[ self_uid ].item()
        rank = self.metagraph.R[ self_uid ].item()
        incentive = self.metagraph.I[ self_uid ].item()     
        normalized_peer_weights =  F.softmax (self.nucleus.peer_weights.detach())
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
        topk_scores, topk_idx = torch.topk(self.stats.ema_scores, 5, dim=0)

        for uid, ema_score in zip(topk_idx, topk_scores) :
            color =  'green' if self.stats.scores[uid] - ema_score > 0 else 'red'
            info[f'uid_{uid.item()}'] = colored('{:.4f}'.format(ema_score), color)

        progress_bar.set_infos( info )

        # ---- wandb log if it is the end of epoch 
        if  self.config.neuron.use_wandb and ((iteration + 1) % (self.config.neuron.epoch_length ) == 0):
            # ---- Miner summary for wandb
            wandb_info = {
                'stake':stake,
                'rank':rank,
                'incentive':incentive,
                'num_peers':self.metagraph.n.item(),
                'remote_target_epoch_loss': self.stats.remote_target_epoch_loss,
                'distillation_epoch_loss': self.stats.distillation_epoch_loss,
                'local_target_epoch_loss': self.stats.local_target_epoch_loss,
                'local_epoch_acc': self.stats.local_epoch_acc,
                'num_sync_metagraph': self.stats.epoch_sync_count,
                'data_size': self.stats.epoch_data_size,
                }
            # ---- Miner summary per peer
            for uid in self.metagraph.uids.tolist():
                uid_str = str(uid).zfill(3)
                wandb_info[f'peers_norm_weight uid: {uid_str}']= normalized_peer_weights[uid]
                wandb_info[f'peers_wo_norm_weight uid: {uid_str}']= self.nucleus.peer_weights[uid]
                wandb_info[f'fisher_ema uid: {uid_str}'] = self.stats.ema_scores[uid]

            wandb_info_axon = self.axon.to_wandb()
            wandb_info_dend = self.dendrite.to_wandb()
                
            try:
                wandb.log({**wandb_info, **wandb_info_axon, **wandb_info_dend})
            except Exception as e:
                logger.warning('Failed to update weights and biases with error:{}', e)

