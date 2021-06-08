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
"""GPT2 Language Modelling miner

The genesis miner.

Example:
    $ python miners/gpt2_genesis.py

To run with a config file:
    $ python miners/gpt2_genesis.py --config <path to config file>

"""

import argparse
from bittensor._dataloader import dataloader
from concurrent.futures import thread
import os
import copy
import math
from re import L
import traceback
from tensorboard import data
import torch
from torch.autograd import backward
import torch.nn.functional as F
import bittensor

from qqdm import qqdm, format_str
from termcolor import colored
from transformers import AdamW
from types import SimpleNamespace
from nuclei.gpt2 import GPT2Nucleus
from routers.sgmoe import SGMOERouter
from typing import Tuple, List, Optional
from torch.nn.utils import clip_grad_norm_
from pytorch_transformers import WarmupCosineWithHardRestartsSchedule

from miners import miner

from loguru import logger
logger = logger.opt(colors=True)

class Miner:
    def __init__( self, hparams ):
        
        self.wallet = bittensor.wallet (
            name = hparams.wallet.name,
            path = hparams.wallet.path,
            hotkey = hparams.wallet.hotkey
        )

        self.subtensor = bittensor.subtensor (
            network = hparams.subtensor.network
        )

        self.metagraph = bittensor.metagraph ( 
            subtensor = self.subtensor
        )

        self.axon = bittensor.axon ( 
            wallet = self.wallet,
            local_ip = hparams.local_ip,
            local_port = hparams.local_port,
            forward_callback = self.forward,
            backward_callback = self.backward,
        )

        self.dendrite = bittensor.dendrite ( 
            wallet = self.wallet,
            max_active_receptors = hparams.dendrite.max_active_receptors,
            max_worker_threads = hparams.dendrite.max_worker_threads
        )

        self.dataset = bittensor.dataloader ( 
            batch_size = self.config.miner.batch_size_train, 
            block_size = self.nucleus.get_block_size(),
            max_corpus_size = hparams.dataloader.max_corpus_size,
        )

        self.router = SGMOERouter ( 
            query_dim = bittensor.__network_dim__ 
        )

        self.nucleus = GPT2Nucleus ( 
            routing_callback = self.route 
        )

        self.optimizer = self.configure_optimizers()


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.router.to( self.device )
        self.nucleus.to( self.device )
        self.nucleus.attach( self ) # Assign the routing function.
        self.axon.attach_forward_callback( self.forward )
        self.axon.attach_backward_callback( self.backward )

        # ---- Row Weights ----
        self.row_weights = torch.ones([0]).to( self.nucleus.device )

        # ---- Optimizer ----
        self.optimizer = self.configure_optimizers()
        self.lr = self.config.miner.learning_rate

        # ---- Dataset ----
        # The Genesis Dataset:
        self.dataset = bittensor.dataloader( 
            batch_size = self.config.miner.batch_size_train, 
            block_size = self.nucleus.get_block_size(),
            max_corpus_size = hparams.dataloader.max_corpus_size,
        )
        self.tokens = 0

        self.dataset.dataloader( self.config.miner.epoch_length )

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

    # ---- Epoch -----
    def run_epoch(self):
        epoch_examples = self.dataset.dataloader( self.hparams.epoch_length )

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


    # ---- Axon Forward call ----
    def forward ( self, pubkey:str, inputs: torch.FloatTensor, modality:int ) -> torch.FloatTensor:
        r""" Subscribed to an axon servicing endpoint.
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
        inputs = inputs.to( self.nucleus.device )
        output = self.nucleus.local_forward (
            inputs = inputs        
        )
        return output.local_hidden

    # ---- Axon Backward call ----
    def backward ( self, pubkey:str, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, modality:int ) -> torch.FloatTensor:
        r""" Subscribed to an axon servicing endpoint.
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
                    The gradients w.r.t to the inputs [batch_size, sequence_len, -1]
        """
        # TODO(const): add backward processing.
        # Not processing backward requests
        return None

    def route ( self, inputs: torch.LongTensor, query: torch.FloatTensor ) -> torch.FloatTensor:
        r""" Routing function for a bittensor nucleus. Accepts tokenized text inputs and a query. Routes text inputs to neurons
            based on that query. This function must be overridden by a miner class and assigned to the nucleus.

            Args:
                inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_dim)`, `required`): 
                    Tensor of tokenized sentences.
                
                query (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, query_dim)`, `required`): 
                    Context tensor used to select which neurons to query for each example.
            
            Returns:
                remote_context (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`): 
                    Joined responses from network call.
        """
        # ---- Forward messages through network ---- 
        outputs = self.router.forward_text( self.metagraph, self.dendrite, inputs, query )

        # ---- Train row weights ----
        self.row_weights = (1 - 0.1) * self.row_weights + 0.1 * outputs.weights # Moving avg update.

        # ---- Return responses -----
        return outputs.responses

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

                    remote_context (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`): 
                        Joined responses from network call.

                    distillation_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        Distillation loss between local_context and remote_context.

            )
        """
        # ---- Forward pass ----
        inputs = batch['inputs']
        output = self.nucleus.remote_forward(
            inputs = inputs,
            training = True,
        )

        # ---- Backward pass ----
        output.loss = output.local_target_loss + output.distillation_loss + output.remote_target_loss
        output.loss.backward() # Accumulates gradients on the nucleus.
        clip_grad_norm_(self.nucleus.parameters(), self.config.miner.clip_gradients)
        clip_grad_norm_(self.router.parameters(), self.config.miner.clip_gradients)
        self.optimizer.step() # Applies accumulated gradients.
        self.optimizer.zero_grad() # Zeros out gradients for next accummulation
        self.decay_learning_rate( inputs )

        # ---- Update global loss ----
        return output

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
            If this function returns True, the nucleus is saved to disk and can be reloaded later.
            Returns:
                should_save (bool):
                    True by default. Saves nucleus after each epoch.
        """
        # Save if the epoch loss has decreased.
        if self.epoch_loss < self.last_saved_loss:
            return True
        else:
            return False

    def should_reload(self) -> bool:
        r""" Called by miner.run() after every epoch.
            If the function returns True the nucleus state dict is reloaded from miner.full_path.
            Returns:
                should_reload (bool):
                    False by default -> does not reload the nucleus this epoch.
        """
        # Only reload if the nucleus or router have seen Nans.
        nans_in_nucleus = torch.any(torch.isnan(torch.cat([param.view(-1) for param in self.nucleus.parameters()])))
        nans_in_router = torch.any(torch.isnan(torch.cat([param.view(-1) for param in self.router.parameters()])))
        if nans_in_nucleus or nans_in_router:
            return True
        else:
            return False
            

    def get_state_dict( self ) -> dict:
        r""" Called by miner.save_state().
            Returns a state dict which can be passed to miner.reload_from_state_dict on reload.
            Returns:
                state_dict (:obj:`dict`): 
                    Dictionary containing run state information such as the nucleus parameters.
        """
        return {
            'row_weights': self.row_weights, # Save row.
            'router_state': self.router.state_dict(), # Save router state.
            'nucleus_state': self.nucleus.state_dict(), # Save nucleus state.
            'optimizer_state': self.optimizer.state_dict(), # Save optimizer.
        }

    def reload_from_state_dict( self, state_dict: dict):
        r""" Called by miner.reload_state().
            Reloads the training state from the passed state_dict. 
            Args:
                state_dict (:obj:`dict`): 
                    Dictionary containing run state information such as the nucleus parameters. Output 
                    of get_state_dict.
        """
        self.row_weights = state_dict['row_weights'] # Load row weights
        self.nucleus.load_state_dict( state_dict['nucleus_state'] ) # Load nucleus
        self.router.load_state_dict( state_dict['router_state']) # Load router
        self.nucleus.attach( self )# Re-assign the routing function.
        self.router.sync_with_chain_state( self.metagraph ) # Resize the router.
        self.optimizer.load_state_dict( state_dict['optimizer_state'] ) # Load optimizer.
        self.optimizer = self.configure_optimizers( self.optimizer )

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


    def sync_chain_state( self ):
        r""" Called after each training epoch. Miner should update chain-state and resize objects.
        """
        super().sync_chain_state() # Syncs metagraph and saves to file.
        self.row_weights = torch.nn.functional.pad( self.row_weights, pad = [0, self.metagraph.n - self.row_weights.numel()], value=0) # Pad row weights.
        self.router.sync_with_chain_state( self.metagraph ) # Resize the router.
        self.optimizer = self.configure_optimizers( self.optimizer )

    # ---- Get Row Weights ----
    def get_row_weights( self ) -> torch.FloatTensor:
        r""" Called after each training epoch. Returns row_weights to be set on chain.
            Returns:
                row_weights ( torch.FloatTensor, shape=(self.metagraph.n) ): 
                    Torch row_weights matching the metagraph size to be eventually set on chain.
                    weight values should be normalized and be in range [0,1].
        """
        return F.normalize(self.row_weights, p = 1, dim = 0)

    def configure_optimizers( self, previous_optimizer = None):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the nucleus into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.

        Args:
            previous_optimizer:
                optimizer from previous configure or None if not existent

        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.Tanh)
        for mn, m in self.nucleus.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.nucleus.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # Adds new router params for extended size.
        if previous_optimizer != None:
            # extract the state dict from your old optimizer
            old_state_params = previous_optimizer.state_dict()

            newly_added_router_params = [ p for p in self.router.parameters() if not p in old_state_params["param_groups"][0]["params"] ]
            next_router_params = newly_added_router_params + old_state_params["param_groups"][0]["params"]

            optim_groups = [
                {"params": next_router_params }, # The router may have been extended.
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.config.miner.weight_decay, "betas": [0.9, 0.95]},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0, "betas": [0.9, 0.95]},
            ]

        else:
            optim_groups = [
                {"params": self.router.parameters() },
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.config.miner.weight_decay, "betas": [0.9, 0.95]},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0, "betas": [0.9, 0.95]},
            ]

        # Build optimizer.
        optimizer = torch.optim.AdamW( optim_groups, lr = self.config.miner.learning_rate, betas = (0.9, 0.95) )
        return optimizer

    def decay_learning_rate(self, batch):
        """Decay the learning rate based on the progress thus far.
        Adjusts the self.config.miner.learning_rate according to the
        tokens processed so far, returns number of tokens.

        Args:
            tokens (int): Number of tokens processed so far.
        """

        if self.config.miner.lr_decay:
            # number of tokens processed this step
            self.tokens += (batch >= 0).sum()
            if self.tokens < self.config.miner.warmup_tokens:
                # linear warmup
                lr_mult = float(self.tokens) / float(max(1, self.config.miner.warmup_tokens))
            else:
                # cosine learning rate decay
                progress = float(self.tokens - self.config.miner.warmup_tokens) / float(max(1, self.config.miner.final_tokens - self.config.miner.warmup_tokens))
                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

            self.lr = self.config.miner.learning_rate * lr_mult

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
        else:
            self.lr = self.config.miner.learning_rate

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']


class miner:

    def __new__(
        self, 
        config: 'bittensor.Config' = None, 
        namespace: str = ''
    ):
        # ---- Load Config ----
        if config == None:
            config = bittensor.config.cut_namespace( miner.config( namespace ), namespace ).miner
        config = copy.deepcopy(config)
        miner.check_config( config )
        router = SGMOERouter( config = config.router, query_dim = bittensor.__network_dim__ )
        nucleus = GPT2Nucleus( config = config.nucleus )
        wallet = bittensor.wallet ( config = config.wallet )
        subtensor = bittensor.subtensor( config = config.subtensor )
        metagraph = bittensor.metagraph( config = config.metagraph )
        axon = bittensor.axon( config = config.axon, wallet = wallet )
        dendrite = bittensor.dendrite( config = config.dendrite, wallet = wallet )
        return Miner( config, wallet, subtensor, metagraph, axon, dendrite, router, nucleus )

    @staticmethod
    def config( namespace: str = '' ) -> 'bittensor.Config':
        parser = argparse.ArgumentParser()
        miner.add_args(parser)
        config = bittensor.config( parser )
        return bittensor.config.cut_namespace( config, namespace )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, namespace: str = ''):
        parser.add_argument('--learning_rate', default=3e-2, type=float, help='Training initial learning rate.')
        parser.add_argument('--weight_decay', default=0.25, type=float, help='nucleus parameter weight decay.')
        parser.add_argument('--lr_decay', default=True, type=bool, help='learning rate decay params: linear warmup followed by cosine decay to 10%% of original.')
        parser.add_argument('--warmup_tokens', default=375e6, type=float, help='A linear LR warmup over the first miner.warmup_tokens tokens (default is 365 million)')
        parser.add_argument('--final_tokens', default=260e9, type=float, help='At what point we reach 10%% of original LR')
        parser.add_argument('--clip_gradients', default=1.0, type=float, help='Implement gradient clipping to avoid exploding loss on smaller architectures.')
        parser.add_argument('--n_epochs', default=-1, type=int, help='Number of training epochs.')
        parser.add_argument('--epoch_length', default=500, type=int, help='Iterations of training per epoch')
        parser.add_argument('--batch_size_train', default=2, type=int, help='Training batch size.')
        parser.add_argument('--name', default='gpt2_genesis', type=str, help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ')
        parser.add_argument('--debug', default=False, dest='debug', action='store_true', help='''Turn on bittensor debugging information''')
        parser.add_argument('--config', type=str, help='If set, arguments are overridden by passed file.')
        parser.add_argument('--modality', default=0, type=int, help='''Miner network modality. TEXT=0, IMAGE=1. Currently only allowed TEXT''')
        parser.add_argument('--use_upnpc', default=False, dest='use_upnpc', action='store_true', help='''Turns on port forwarding on your router using upnpc.''')
        parser.add_argument('--record_log', default=False, dest='record_log', action='store_true', help='''Turns on logging to file.''')   
        parser.add_argument('--root_dir', default='~/.bittensor/miners/', type=str, help='Root path to load and save data associated with each miner')
        parser.add_argument('--use_tensorboard', default=True, dest='use_tensorboard', action='store_true', help='Turn on bittensor logging to tensorboard')
        bittensor.dataloader.add_args( parser )
        GPT2Nucleus.add_args( parser )
        SGMOERouter.add_args( parser )
        bittensor.wallet.add_args( parser, namespace )
        bittensor.subtensor.add_args( parser, namespace )
        bittensor.axon.add_args( parser, namespace )
        bittensor.dendrite.add_args( parser, namespace )
        bittensor.metagraph.add_args( parser, namespace )

    @staticmethod
    def check_config(config: 'bittensor.Config'):
        assert config.miner.batch_size_train > 0, "batch_size_train must a positive value"
        assert config.miner.learning_rate > 0, "learning_rate must be a positive value."
        GPT2Nucleus.check_config( config )
        SGMOERouter.check_config( config )
        bittensor.dataloader.check_config( config )
        bittensor.wallet.check_config( config.wallet )
        bittensor.subtensor.check_config( config.subtensor )
        bittensor.axon.check_config( config.axon )
        bittensor.dendrite.check_config( config.dendrite )
        bittensor.metagraph.check_config( config.metagraph )
        full_path = os.path.expanduser('{}/{}/{}'.format( config.miner.root_dir, config.wallet.name + "-" + config.wallet.hotkey, config.miner.name ))
        config.miner.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.miner.full_path):
            os.makedirs(config.miner.full_path)

if __name__ == "__main__":
    # ---- Build and Run ----
    miner = Miner()
    print (miner.config)
    miner.run()
