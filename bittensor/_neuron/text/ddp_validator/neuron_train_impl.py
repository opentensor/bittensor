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
import os
import pandas
from pandas.core.frame import DataFrame
import bittensor
import math
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
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
from torch.multiprocessing import Manager
import pickle
from os import path
import random
import time

class DDPNeuronTrain:
    def __init__( self, config: 'bittensor.config', nucleus: 'Nucleus', wallet: 'bittensor.wallet'):
        r""" Initializes the neuron with the passed config.
        """
        self.config = config
        self.wallet = wallet
        self.world_size = config.neuron.world_size
        self.nucleus = nucleus
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
            scores = {},
            ema_scores = torch.nn.Parameter(torch.zeros(self.config.nucleus.max_n), requires_grad = False)
        )
        # ---- Decay factor for fisher ema score 
        self.fisher_ema_decay = 0.995

    def stop( self ):
        r""" Stop the dendrite and dataset
        """
        del self.dendrite
        self.dataset.close()
    
    def init_process(self, rank):
        r""" For each process, anchor them to the process group 
        so that they know how to communication with each other.

        Args:
            rank (int):
                rank (id) of the process.
        """
        os.environ['MASTER_ADDR'] = self.config.neuron.address
        os.environ['MASTER_PORT'] = self.config.neuron.port
        if 'cuda' in self.config.neuron.device:
            backend = 'nccl'
        else:
            backend = 'gloo'

        dist.init_process_group(backend, rank=rank, world_size=self.world_size)
    
    def init_bit(self, rank = 0):
        r""" Init bittensor modules .
        
        Args:
            rank (int):
                rank (id) of the process.
        """

        if self.config.neuron.multiprocessing and self.config.neuron.device == 'cuda':
            self.device = torch.device( device = f'cuda:{rank}' )
        else:
            self.device = torch.device( device = self.config.neuron.device )
        
        self.subtensor = bittensor.subtensor ( config = self.config )
        self.metagraph = bittensor.metagraph ( config = self.config, subtensor = self.subtensor )
        self.dendrite = bittensor.dendrite ( config = self.config, wallet = self.wallet )
        self.dataset = bittensor.dataset ( config = self.config, world_size = self.world_size, rank = rank)
        self.stats.ema_scores.to(self.device)
        self.optimizer = torch.optim.SGD(
            # [ {'params': nucleus_ddp.peer_weights, 'lr': self.config.neuron.learning_rate_chain} ],
            [ {'params': self.nucleus.parameters(), 'lr': self.config.neuron.learning_rate_chain} ],
            lr = self.config.neuron.learning_rate,
            momentum = self.config.neuron.momentum,
        )
        
        if rank == 0 :
            self.subtensor.register( self.wallet )
    
    def cleanup(self):
        r""" Kill the process.
        """
        dist.destroy_process_group()

    def run_parallel( self ):
        r""" Spawn multiple processes.
        """
        with Manager() as manager:
            self.stats.scores = manager.dict()
            mp.spawn(self.run,
                args=(self.world_size,),
                nprocs=self.world_size,
                join=True
            )

    def clip_grad_hook( self, process_group: dist.ProcessGroup, bucket: dist.GradBucket ): # -> torch.futures.Future[torch.Tensor]:
        r""" Norm the gradient before all reduce.
        """
        if len(bucket.parameters()) > 70:
            total_norm = clip_grad_norm_(bucket.parameters(), 1)
            self.total_norm = total_norm
            bittensor.logging.success("clip grad norm, large bucket", sufix = f'{self.id} total_norm {self.total_norm}')
        else:
            bittensor.logging.success("clip grad norm, small bucket", sufix = f'{self.id} total_norm {self.total_norm}')
            clip_coef = 1 / (self.total_norm + 1e-6)
            bittensor.logging.success("clip grad norm, small bucket", sufix = f'{self.id} clip coef {clip_coef}')
            clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
            bittensor.logging.success("clip grad norm, small bucket", sufix = f'{self.id} clip coef clamped {clip_coef_clamped}')

            for p in bucket.parameters():
                p.grad.detach().mul_(clip_coef_clamped)
                bittensor.logging.success("clip grad norm, small bucket", sufix = f'p {p[:20]}')
            
        flat_grads = [ torch.reshape(p.grad, (-1,) ) for p in bucket.parameters()]
        tensor = torch.cat(flat_grads)
        bucket.set_buffer(tensor)
        group_to_use = process_group if process_group is not None else dist.group.WORLD

        # Apply the division first to avoid overflow, especially for FP16.
        tensor.div_(group_to_use.size())

        return (dist.all_reduce(tensor, group=group_to_use, async_op=True).get_future().then(lambda fut: fut.value()[0]))

    def run( self, rank = 0, world_size = 0):
        r""" Miner main loop.
        """
        # ---- Build Bittensor neuron ----
        self.init_bit(rank)
        self.init_process(rank)
        self.meta_sync()

        if rank == 0 and self.config.neuron.use_wandb:
            bittensor.wandb(
                config = self.config,
                cold_pubkey = self.wallet.coldkeypub.ss58_address,
                hot_pubkey = self.wallet.hotkey.ss58_address,
                root_dir = self.config.neuron.full_path
            )

        # ---- Init run state ----
        self.epoch = 0   
        self.stats.ema_scores = torch.nn.Parameter(torch.ones(self.config.nucleus.max_n).to(self.device) * (1 / self.metagraph.n.item()), requires_grad = False)

        # ---- Reloads nucleus if not restart----
        if not self.config.neuron.no_restart:
            self.save_nucleus()

        try:
            self.reload_nucleus(rank)
        except Exception as e:
            logger.error("Error when trying to reload model: {}".format(e))
            self.save_nucleus()
            self.reload_nucleus(rank)


        # --- Run until n_epochs ----
        while self.epoch < self.config.neuron.n_epochs:
            try:
                # --- Init epoch stat----
                self.stats.epoch_data_size = 0
                self.stats.epoch_sync_count = 0
                epoch_stats = SimpleNamespace(
                    total_local_target_epoch_loss = 0,
                    total_distillation_epoch_loss = 0,
                    total_remote_target_epoch_loss = 0,
                    total_local_epoch_acc = 0,
                    batches_count = 0,
                )

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
                        self.id = random.randint(1000, 9999)
                        inputs = next( self.dataset )
                        output = self.nucleus.forward( inputs = inputs.to( self.device ), training = True)

                        # ---- Backward pass ----
                        output.loss = output.local_target_loss + output.distillation_loss + output.remote_target_loss
                        output.loss.backward(retain_graph = True) # Accumulates gradients on the nucleus.
                        
                        # ---- Apply and zero accumulated gradients.
                        self.optimizer.step() 
                        self.optimizer.zero_grad()

                        # ---- Update stats and scores. 
                        epoch_stats = self.agg_epoch_stats(epoch_stats, output)
                        self.stats.epoch_data_size += inputs.nelement()
                        current_block = self.subtensor.get_current_block()
                        
                        output.scores = output.scores.detach()
                        output.scores.requires_grad = False
                        self.stats.scores[rank] = output.scores
                        scores_avg =  sum(self.stats.scores.values())/len(self.stats.scores)
                        self.stats.ema_scores[:len(scores_avg)] = self.fisher_ema_decay * self.stats.ema_scores[:len(scores_avg)] + (1 - self.fisher_ema_decay) * scores_avg
                        
                    # ---- Update the epoch loss if it is the last iteration within epoch
                    if block+1 == end_block :
                        self.update_epoch_loss(epoch_stats)

                    # ---- Sync with metagraph if the current block >= last synced block + sync block time 
                    block_diff = self.subtensor.get_current_block() - self.stats.last_sync_block
                    if block_diff >= self.config.neuron.sync_block_time:
                        self.meta_sync(current_block)                                                                                                                
                        self.stats.epoch_sync_count += 1
                        if rank == 0:
                            self.set_peer_weights()

                    # ---- Logs for block.
                    self.stats.global_step += 1
                    if rank == 0:
                        self.logs ( progress_bar, iteration = block-start_block, output = output)

                if rank == 0:
                    self.checkpoint()

                self.epoch += 1
            
            except KeyboardInterrupt:
                # --- User ended session ----
                self.stop()
                self.cleanup()
                break

            except Exception as e:
                # --- Unknown error ----
                logger.exception('Unknown exception: {} with traceback {}', e, traceback.format_exc())
                if self.config.neuron.restart_on_failure == True:
                    logger.info('Restarting from last saved state.')
                    self.reload_nucleus()
                else:
                    break

    def agg_epoch_stats(self, epoch_stats, output):
        r""" Append the losses from output to the epoch stats.
        Args:
            epoch_stats: SimpleNamespace
                Aggregate all the losses (local_target_loss/distillation_loss/remote_target_loss) as float

            output: SimpleNamespace
                output from the nucleus forward call. Which has all the hidden layers/losses.
        """
        epoch_stats.total_local_target_epoch_loss += output.local_target_loss.item()
        epoch_stats.total_distillation_epoch_loss += output.distillation_loss.item()
        epoch_stats.total_remote_target_epoch_loss += output.remote_target_loss.item()
        epoch_stats.total_local_epoch_acc += output.local_accuracy
        epoch_stats.batches_count += 1

        # ---- Temp debug logging.
        bittensor.logging.success( prefix = f'adding to local target epoch loss', sufix = f' {output.local_target_loss.item(), epoch_stats.total_local_target_epoch_loss}, Rank ')
        bittensor.logging.success( prefix = f'adding to distillation epoch loss', sufix = f' {output.distillation_loss.item(), epoch_stats.total_distillation_epoch_loss}, Rank ')
        bittensor.logging.success( prefix = f'adding to remote target epoch loss', sufix = f' {output.remote_target_loss.item(), epoch_stats.total_remote_target_epoch_loss}, Rank ')
        bittensor.logging.success( prefix = f'adding to local accuracy', sufix = f' {output.local_accuracy, epoch_stats.total_local_epoch_acc}, Rank ')

        if torch.any(torch.isnan(output.local_target_loss)).item():
            bittensor.logging.success( prefix = f'got nan accuracy', sufix = f'')
            print(output, '\n\n\n\n\n\n\n\n\n\n\n\n')
            out_dict = {}

            for k,v in output.__dict__:
                out_dict[k] = str(v)

            a_file = open("output_error.pkl", "wb")
            pickle.dump(out_dict, a_file)
            a_file.close()

            self.cleanup()
        return epoch_stats

    def update_epoch_loss(self, epoch_stats):
        r""" Update self.stats with the averaged losses.
        Args:
            epoch_stats: SimpleNamespace
                Aggregate all the losses (local_target_loss/distillation_loss/remote_target_loss) as float
        """
        batches_count = epoch_stats.batches_count
        self.stats.local_target_epoch_loss = epoch_stats.total_local_target_epoch_loss / (batches_count) 
        self.stats.distillation_epoch_loss = epoch_stats.total_distillation_epoch_loss / (batches_count)
        self.stats.remote_target_epoch_loss = epoch_stats.total_remote_target_epoch_loss / (batches_count)
        self.stats.local_epoch_acc = epoch_stats.total_local_epoch_acc / (batches_count)

    def save_nucleus( self ):
        r""" Saves the training state to disk.
        """
        try:
            state_dict = {
                'epoch': self.epoch,
                'epoch_loss': self.stats.local_target_epoch_loss,
                'global_step': self.stats.global_step,
                'nucleus_state': self.nucleus.state_dict(), # Save nucleus state.
                'optimizer_state': self.optimizer.state_dict(),
                'network': self.subtensor.network, # Save Network
            }
            torch.save( state_dict, "{}/model.torch".format( self.config.neuron.full_path ) )
            bittensor.logging.success(prefix='Saved model', sufix='<blue>{}/model.torch</blue>'.format( self.config.neuron.full_path ) )
        except Exception as e:
            logger.exception('Failed to save model with error:{}', e)

    def reload_nucleus( self, rank = 0 ):
        r""" Reloads/updates the training state from the disk.
        """
        # --- Load prev state.
        try:
            state_dict = torch.load("{}/model.torch".format( self.config.neuron.full_path ))
        except Exception as e:
            logger.warning('No saved model found with error: {}', e)
            logger.warning('Saving the current model and stats: {}', e)
            self.save_nucleus()
            state_dict = torch.load("{}/model.torch".format( self.config.neuron.full_path ))
        
        # ---- Load statistics.
        self.epoch = state_dict['epoch']
        self.stats.local_target_epoch_loss = state_dict['epoch_loss']
        self.stats.best_epoch_loss = state_dict['epoch_loss']
        self.stats.global_step = state_dict['global_step']

        # ---- Load nucleus.
        self.nucleus = self.nucleus.to(self.device)
        self.nucleus.device = self.device 
        self.nucleus.dendrite = self.dendrite # Set local dendrite.
        self.nucleus.metagraph = lambda: self.metagraph # Set local metagraph.

        # ---- Load nucleus parameters.
        try:
            self.nucleus.load_state_dict( state_dict['nucleus_state'], strict=False )
        except Exception as e:
            logger.exception('Failed to load nucleus state with error, updating the current state')
            state_dict['nucleus_state'] = self.nucleus.state_dict()
            torch.save( state_dict, "{}/model.torch".format( self.config.neuron.full_path ) )
        
        # ---- Register nucleus to DDP
        if 'cuda' in self.config.neuron.device:
            self.nucleus.to(rank)
            self.nucleus = DDP(self.nucleus,  device_ids=[rank])
        else:
            self.nucleus = DDP(self.nucleus, bucket_cap_mb = 10000000)
            self.nucleus.register_comm_hook(state=None, hook=self.clip_grad_hook)

        # ---- Load optimizer.
        self.optimizer = torch.optim.SGD(
            [{"params": self.nucleus.parameters()}],
            lr = state_dict['optimizer_state']['param_groups'][0]['lr'],
            momentum = state_dict['optimizer_state']['param_groups'][0]['momentum'],
        )

        bittensor.logging.success( prefix = 'Reloaded model', sufix = '<blue>{}/model.torch</blue>'.format( self.config.neuron.full_path ))

    def meta_sync (self, current_block = None ):
        """ Miner sync with metagraph and update chain weight
        """
        self.metagraph.sync().save()
        self.stats.last_sync_block= self.subtensor.get_current_block()
        bittensor.logging.success( 'Synced metagraph', f'Block: {current_block} ---> Block: {self.stats.last_sync_block}')

    def set_peer_weights( self ):
        r""" Sets the fisher ema score to peers.
        """

        try:
            # ---- Avoid setting weight to inactive uids. 
            inactive_uids = torch.where(self.metagraph.active == 0)[0]
            self.stats.ema_scores[inactive_uids] = 0
            
            # ---- Set weight to topk uids.
            k = min( self.config.neuron.n_topk_peer_weights, self.metagraph.n.item() )
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

    def checkpoint( self ):
        r""" Optionally Saves, updates and then reloads the miner training state.
        """
        # ---- Get the old states.
        try:
            last_saved = torch.load("{}/model.torch".format( self.config.neuron.full_path ))
        except Exception as e:
            last_saved = None

        # ---- Get the best epoch loss.
        if last_saved == None or last_saved['epoch_loss'] >= self.stats.local_target_epoch_loss:
            self.stats.best_epoch_loss = self.stats.local_target_epoch_loss
            self.save_nucleus()

        # ---- Checks if epochs managed to diverage.
        if not math.isfinite(self.stats.local_target_epoch_loss):
            logger.error('Incorrect epoch loss detected, reloading to previous saved state')
            self.reload_nucleus()

    # ---- Training logs ----
    def logs( self, progress_bar, iteration:int, output: SimpleNamespace ):
        r""" Called after every training step. Displays miner state to screen.
        """
        self_neuron = self.subtensor.neuron_for_pubkey( self.wallet.hotkey.ss58_address )
        self_uid = self_neuron.uid
        stake = self_neuron.stake
        rank = self_neuron.rank
        incentive = self_neuron.incentive
        # normalized_peer_weights = F.softmax (self.nucleus.state_dict()['module.peer_weights'], dim=0)
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
                # bittensor.utils.indexed_values_to_dataframe( prefix = 'raw_peer_weight', index = topk_uids, values = self.nucleus.state_dict()['module.peer_weights'], filter_zeros = True),
                # bittensor.utils.indexed_values_to_dataframe( prefix = 'normalized_peer_weight', index = topk_uids, values = normalized_peer_weights, filter_zeros = True),
                bittensor.utils.indexed_values_to_dataframe( prefix = 'w_{}_i'.format(self_uid), index = topk_uids, values = self.metagraph.W[ self_uid, : ], filter_zeros = True),
                bittensor.utils.indexed_values_to_dataframe( prefix = 'w_i_{}'.format(self_uid), index = topk_uids, values = self.metagraph.W[ :, self_uid ], filter_zeros = True),
                self.dendrite.to_dataframe( metagraph = self.metagraph )
            ], axis = 1)
            df['uid'] = df.index
            stats_data_table = wandb.Table( dataframe = df)

            wandb_info_dend = self.dendrite.to_wandb()
            wandb.log( { **wandb_info, **wandb_info_dend }, step = current_block)
            wandb.log( { 'stats': stats_data_table}, step = current_block)
            wandb.log( { 'axon_query_times': wandb.plot.scatter( stats_data_table, "uid", "axon_query_time", title="Axon Query time vs UID") } )
            wandb.log( { 'dendrite_query_times': wandb.plot.scatter( stats_data_table, "uid", "dendrite_query_time", title="Dendrite Query time vs UID") } )