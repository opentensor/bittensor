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
""" The Exodus miner.

Example:
    $ python miners/text/template_miner.py

"""

import argparse
import bittensor
import math
import torch
import traceback
import os
import sys
import yaml
import wandb
import signal

from termcolor import colored
from typing import List
from qqdm import qqdm, format_str
from loguru import logger; logger = logger.opt(colors=True)
from types import SimpleNamespace
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn

import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from substrateinterface.utils.ss58 import ss58_encode

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Nucleus(nn.Module):

    def __init__(self, config ):
        super(Nucleus, self).__init__()
        self.config = config

        # Embedding Layer.
        self.embedding = nn.Embedding( bittensor.__vocab_size__,  bittensor.__network_dim__ )

        # Local Model
        local_layers = TransformerEncoderLayer( bittensor.__network_dim__, self.config.nucleus.nhead, self.config.nucleus.nhid, self.config.nucleus.dropout )
        local_hidden_layers = TransformerEncoderLayer( bittensor.__network_dim__, self.config.nucleus.nhead, self.config.nucleus.nhid, self.config.nucleus.dropout )
        self.local_pos_encoder = PositionalEncoding(bittensor.__network_dim__, self.config.nucleus.dropout)
        self.local_encoder = TransformerEncoder( local_layers, self.config.nucleus.nlayers )
        self.local_hidden = TransformerEncoder( local_hidden_layers, self.config.nucleus.nlayers )
        self.local_decoder = nn.Linear( bittensor.__network_dim__, bittensor.__vocab_size__ , bias=False)

        # Remote Model
        remote_context_layers = TransformerEncoderLayer( bittensor.__network_dim__, self.config.nucleus.nhead, self.config.nucleus.nhid, self.config.nucleus.dropout )
        self.remote_hidden = TransformerEncoder( remote_context_layers, self.config.nucleus.nlayers )
        self.remote_decoder = nn.Linear( bittensor.__network_dim__, bittensor.__vocab_size__ , bias=False)

        self.loss_fct = nn.CrossEntropyLoss()
        self.chain_weights = nn.Parameter(torch.ones( [0] , requires_grad=True))
        self.init_weights()

    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        r""" Add custom params to the parser.
        """
        parser.add_argument('--nucleus.nhid', type=int, help='the dimension of the feedforward network model in nn.TransformerEncoder', default=200)
        parser.add_argument('--nucleus.nhead', type=int, help='the number of heads in the multiheadattention models', default=2)
        parser.add_argument('--nucleus.nlayers', type=int, help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder', default=2)
        parser.add_argument('--nucleus.dropout', type=float, help='the dropout value', default=0.2)
        parser.add_argument('--nucleus.topk', type=int, help='the number of peers queried during each remote forward call', default=20)
        parser.add_argument('--nucleus.punishment', type=float, help='The punishment on the chain weights that do not respond ', default=0.001 )

    def init_weights(self):
        initrange = 0.1
        self.remote_decoder.weight.data.uniform_(-initrange, initrange)
        self.local_decoder.weight.data.uniform_(-initrange, initrange)

    def local_forward(self, inputs: torch.int64, training : bool = True) -> SimpleNamespace:
        """ Forward pass through GPT2 nucleus.
            Args:
                inputs (:obj:`torch.int64` of shape :obj:`(batch_size, block_size)`, `required`):
                    Batch_size length x list of text sentences.
                training (:obj:`bool')`, `optional`, defaults to True):
                    Switch to True if this forward pass computes a CLM loss.

            Returns:
                SimpleNamespace {
                    local_context (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Hidden layer context.
                    local_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__vocab_size__)`, `optional`):
                        MLM Target predictions produced using local_context.
                    local_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`):
                        MLM loss using local_context.
                }
        """
        # To be filled.
        output = SimpleNamespace()

        # local_context: hidden layer encoding of sequence with local_context.
        # local_context.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        output.local_context = self.local_encoder( self.embedding( inputs ) )* math.sqrt(bittensor.__network_dim__)

        # local_context: adding positional encoding to local_context.
        # local_context.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        output.local_context = self.local_pos_encoder(output.local_context)

        if training :
            # local_hidden: local model which learns a new projection from the local_context
            # local_hidden.shape = [batch_size, sequence_len, bittensor.__vocab_size__]
            output.local_hidden = self.local_hidden( output.local_context.detach())

            # local_target: projection of local_hidden onto target dimension.
            # local_target.shape = [batch_size, sequence_len, bittensor.__vocab_size__]
            output.local_target = self.local_decoder( output.local_hidden )

            # local_target_loss: MLM loss between local_target and passed targets.
            # local_target_loss.shape = [1]
            shift_logits = output.local_target[..., :-1, :].contiguous()
            shift_labels = inputs[..., 1:].contiguous()
            output.local_target_loss = self.loss_fct( shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1) )

            predictions=shift_logits.detach().max(2).indices
            output.local_accuracy = (predictions==shift_labels).sum().item()/predictions.nelement()
        return output

    def remote_forward(self, inputs: torch.int64, training: bool) -> SimpleNamespace:
        """ Forward pass inputs and labels through the GPT2 module and into the remote network.
        Args:
            inputs (:obj:`torch.int64` of shape :obj:`(batch_size, sequence_len)`, `required`):
                Tokenized sentences using bittensor.tokenizer()
            training (:obj:`bool')`, `optional`, defaults to True):
                Switch to True if this forward pass computes an MLM loss.
        Returns:
            self.local_forward() + SimpleNamespace (
                remote_context (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                    Joined responses from the network.
                remote_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,  bittensor.__vocab_size__)`, `optional`):
                    Target predictions using the remote_context layer.
                remote_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`):
                    MLM loss using remote_target.
                distillation_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`):
                    Distillation loss between local_context and remote_context.
            )
        """
        # Run local model
        output = self.local_forward( inputs, training )

        # remote_context: joined responses from a dendrite.forward_text call.
        # remote_context.shape = [batch_size, sequence_len (or block_size), bittensor.__network_dim__]
        output.remote_context, output.quested_peers, output.responded_peers = self.remote( inputs )

        # remote_hidden: projects from the remote_context
        # remote_hidden.shape = [batch_size, sequence_len, bittensor.__vocab_size__]
        output.remote_hidden = self.remote_hidden( output.remote_context )

        # distillation_loss : distillation loss between local_context and remote_context
        # distillation_loss.shape = [1]
        # This trains the local_context (student) to emulate the network context.
        output.distillation_loss = F.mse_loss( output.local_context, output.remote_hidden.detach() )

        if training :
            # remote_target: projection of remote_hidden onto target dimension.
            # remote_target.shape = [batch_size, sequence_len, bittensor.__vocab_size__]
            output.remote_target = self.remote_decoder( output.remote_hidden )

            # remote_target_loss: MLM loss between remote_target and passed targets.
            # remote_target_loss.shape = [1]
            shift_logits = output.remote_target[..., :-1, :].contiguous()

            shift_labels = inputs[..., 1:].contiguous()
            output.remote_target_loss = self.loss_fct( shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1) )

        return output

    def remote(self, inputs: torch.int64 ) -> torch.float32:
        """ Forwards the inputs through the network, selects the topk peers based on self.chain_weights.
        Args:
            inputs (:obj:`torch.int64` of shape :obj:`(batch_size, sequence_len)`, `required`):
                Batch_size length list of text sentences.
        Returns:
            outputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `optional`):
                Joined hidden layer responses from peers.
        """

        # ---- Get active peers and their weights ---- 
        active_uids = torch.where(bittensor.neuron.metagraph.active > 0)[0]
        active_chain_weights = self.chain_weights[active_uids]

        # ---- Topk Weights ---- (TODO: check if the gaussians are enough disrupt the chain weights)
        real_topk = min( self.config.nucleus.topk, bittensor.neuron.metagraph.n.item(), len(active_uids))
        noise = torch.normal( 0, torch.std(active_chain_weights).item()+0.0000001, size=( active_chain_weights.size())).to( self.config.miner.device )
        topk_weights, topk_idx = torch.topk(active_chain_weights + noise , real_topk, dim=0)
        topk_uids = active_uids[topk_idx]

        # ---- Filter endpoints ----
        endpoints = bittensor.neuron.metagraph.endpoints[ topk_uids ]

        # ---- Query network ----
        responses, return_ops = bittensor.neuron.dendrite.forward_text (
            endpoints = endpoints,
            inputs = inputs
        )

        # ---- Join based on weights ----
        joining_uids= torch.where(return_ops==0)[0]
        joining_weights = F.softmax( topk_weights[(return_ops == 0)], dim = 0 ) 
        output_context = torch.zeros( (inputs.shape[0], inputs.shape[1], bittensor.__network_dim__)).to( self.config.miner.device )
        for index, joining_weight in enumerate( joining_weights ):
            output_context += responses[joining_uids[index]].detach().to( self.config.miner.device ) * joining_weight

        # ---- Punish peers with non-successful return ops ----
        with torch.no_grad():
            self.chain_weights[topk_uids[(return_ops != 0)]] -=  self.config.nucleus.punishment
            self.chain_weights[self.chain_weights < -1] = -1 #lower bound for chain weights

        # ---- Return response -----
        output_quested_peers = torch.zeros(bittensor.neuron.metagraph.n.item())
        output_quested_peers[topk_uids] = 1
        
        output_responded_peers = torch.zeros(bittensor.neuron.metagraph.n.item())
        output_responded_peers[topk_uids[joining_uids]] = 1

        return output_context, output_quested_peers, output_responded_peers

class Miner:

    def __init__( self, config: 'bittensor.config' = None ):
        r""" Initializes a miner with the passed config.
        """
        if config == None: config = Miner.config()
        self.config = config; Miner.check_config( self.config ); print ( self.config )

        # Miner training device.
        self.device = torch.device(
            device = self.config.miner.device
        )

        # Dataset of text.
        self.dataset = bittensor.dataloader (
            config = self.config
        )

        # Trainable machine learning model.
        self.nucleus = Nucleus(
            config = self.config,
        ).to( self.device )

        # Torch optimizer.
        self.optimizer = torch.optim.SGD(
            [ {"params": self.nucleus.parameters()}],
            lr = self.config.miner.learning_rate,
            momentum = self.config.miner.momentum,
        )

        #Torch scheduler
        self.scheduler= torch.optim.lr_scheduler.StepLR(self.optimizer,
            step_size= 1.0,
            gamma=0.9
        )

        # Bittensor backend
        self.neuron = bittensor.init (
            config = self.config,
            root_dir = self.config.miner.full_path,
            forward_text = self.forward_text,
            backward_text = self.backward_text,
            blacklist = self.blacklist
        ) 

        #bittensor priority thread pool 
        self.thread_pool = bittensor.prioritythreadpool(
            config = self.config
        )

        # ---- Init how often to sync with metagraph, value will be overriden by config----  
        self.last_sync_block = 0

    @staticmethod
    def config() -> 'bittensor.Config':
        r""" Fills a config namespace object with defaults or information from the command line.
        """
        # ---- Add miner args.
        parser = argparse.ArgumentParser()
        parser.add_argument('--miner.config', type=str, help='If set, defaults are overridden by passed file.')
        parser.add_argument('--miner.learning_rate', type=float, help='Training initial learning rate.', default=1)
        parser.add_argument('--miner.weight_decay', type=float, help='nucleus parameter weight decay.', default=0.25)
        parser.add_argument('--miner.momentum', type=float, help='optimizer momentum.', default=0.8)
        parser.add_argument('--miner.clip_gradients', type=float, help='Implement gradient clipping to avoid exploding loss on smaller architectures.', default=1.0)
        parser.add_argument('--miner.n_epochs', type=int, help='Number of training epochs.', default=sys.maxsize )
        parser.add_argument('--miner.epoch_length', type=int, help='Iterations of training per epoch', default=100)
        parser.add_argument('--miner.batch_size_train', type=int, help='Training batch size.', default=2)
        parser.add_argument('--miner.restart_on_failure',  action='store_true', help='''Restart miner on unknown error.''', default=False)
        parser.add_argument('--miner.compute_remote_gradients', action='store_true', help='''Does the miner compute and return gradients from backward queries.''', default=False)
        parser.add_argument('--miner.accumulate_remote_gradients', action='store_true', help='''Does the miner accumulate remote gradients from backward queries.''', default=False)
        parser.add_argument('--miner.n_topk_chain_weights', type=int, help='Maximum number of weights to submit to chain', default=100 )
        parser.add_argument('--miner.name', type=str, help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default='template_miner')
        parser.add_argument('--miner.device', type=str, help='miner default training device cpu/cuda', default=("cuda" if torch.cuda.is_available() else "cpu"))
        parser.add_argument('--miner.timeout', type=int, help='Number of seconds to wait for axon request', default=1)
        parser.add_argument('--miner.blacklist', type=float, help='Amount of stake (tao) in order not to get blacklisted', default=0)
        parser.add_argument('--miner.sync_block_time', type=int, help='How often the sync the miner with metagraph, in terms of block time', default=15)


        bittensor.add_args( parser )
        Nucleus.add_args( parser ) 
        bittensor.prioritythreadpool.add_args( parser )
 
        # ---- Loads config_file and updates defaults
        config_file_path = vars(parser.parse_known_args()[0])['miner.config']
        if config_file_path:
            config_file_path = os.path.expanduser(config_file_path)
            try:
                with open(config_file_path) as f:
                    params_config = yaml.safe_load(f)
                    print('Config File Detected at {} updating defaults'.format(config_file_path))
                    parser.set_defaults(**params_config)
            except Exception as e:
                print('Error in loading: {} using default parser settings'.format(e))

        return bittensor.config( parser )

    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        r""" Checks/validates the config namespace object.
        """
        assert config.miner.batch_size_train > 0, "batch_size_train must be a positive value"
        assert config.miner.learning_rate > 0, "learning_rate must be a positive value."
        bittensor.check_config( config )
        full_path = os.path.expanduser('{}/{}/{}/{}'.format( config.logging.logging_dir, config.wallet.name, config.wallet.hotkey, config.miner.name ))
        config.miner.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.miner.full_path):
            os.makedirs(config.miner.full_path)

    def sync (self, current_block ):
        """ Miner sync with metagraph and update chain weight
        """
        # ---- Set weights on chain ----
        self.set_chain_weights()

        # ---- Sync with metagraph ----
        bittensor.neuron.metagraph.load().sync().save()
        chain_growth = bittensor.neuron.metagraph.n.item()- self.nucleus.chain_weights.shape[0]
        self.nucleus.chain_weights = nn.Parameter(torch.cat([self.nucleus.chain_weights, torch.ones([chain_growth],dtype=torch.float32,requires_grad=True)]))
        
        bittensor.logging.success( 'Synced metagraph:', 'Block: {}'.format(current_block))
        
    
    def run( self ):
        r""" Miner main loop.
        """
        # ---- Build Bittensor neuron ----
        with self.neuron:
            if self.config.neuron.use_wandb:
                bittensor.wandb(
                    config = self.config,
                    cold_pubkey = self.neuron.wallet.coldkeypub,
                    hot_pubkey = self.neuron.wallet.hotkey.public_key,
                    root_dir = self.neuron.root_dir
                )


            # ---- Init run state ----
            self.epoch = 0
            self.global_step = 0
            self.epoch_loss = math.inf/2
            self.best_epoch_loss = math.inf

            # ---- reloads previous run ----
            try:
                self.reload()
                self.neuron.axon.check()
            except:
                self.save()
                self.reload()
                self.neuron.axon.check()

            # --- Run until n_epochs ----
            while self.epoch < self.config.miner.n_epochs:
                try:
                    # ---- Train state ----
                    self.run_epoch()

                    # ---- Checkpoint state ----
                    self.checkpoint()

                except KeyboardInterrupt:
                    # --- User ended session ----
                    break

                except Exception as e:
                    # --- Unknown error ----
                    logger.exception('Unknown exception: {} with traceback {}', e, traceback.format_exc())
                    if self.config.miner.restart_on_failure == True:
                        logger.info('Restarting from last saved state.')
                        self.reload()
                    else:
                        break

    # --- Run Epoch ----
    def run_epoch( self ):
        r""" Runs a single training epoch pulled from the dataloader.
        """
        # --- Init Epoch ----
        self.quested_peers_count = torch.zeros(0)
        self.responded_peers_count = torch.zeros(0)
        self.epoch_data_size = 0
        total_epoch_loss = 0.0
        total_local_target_epoch_loss = 0
        total_distillation_epoch_loss = 0
        total_remote_target_epoch_loss = 0
        
        self.sync_count = 0
        epoch_batches = self.dataset.dataloader( self.config.miner.epoch_length )
        progress_bar = qqdm(enumerate(epoch_batches), total=len(epoch_batches), desc=format_str('blue', f'Epoch Progress'))
        
        for iteration, (inputs) in progress_bar:

            # ---- Forward / Backward ----
            output = self.train ( batch = { 'inputs': inputs } )

            total_epoch_loss += output.local_target_loss.item()
            total_local_target_epoch_loss += output.local_target_loss.item()
            total_distillation_epoch_loss += output.distillation_loss.item()
            total_remote_target_epoch_loss += output.remote_target_loss.item()

            if len(self.quested_peers_count) < len(output.quested_peers):
                fill = torch.zeros(len(output.quested_peers) - len(self.quested_peers_count))
                self.quested_peers_count = torch.cat((self.quested_peers_count, fill))
                self.responded_peers_count = torch.cat((self.responded_peers_count, fill))
            
            self.quested_peers_count += output.quested_peers
            self.responded_peers_count += output.responded_peers
            self.epoch_data_size += inputs.nelement()

            # ---- Sync with metagraph ----
            current_block = self.neuron.subtensor.get_current_block()
            block_diff = current_block - self.last_sync_block
            if block_diff >= self.config.miner.sync_block_time:
                self.sync(current_block)
                self.last_sync_block = current_block
                self.sync_count += 1
            
            # ---- Update the epoch loss if it is the last iteration
            if iteration == self.config.miner.epoch_length - 1:
                self.epoch_loss = total_epoch_loss / self.config.miner.epoch_length
                self.local_target_epoch_loss = total_local_target_epoch_loss / self.config.miner.epoch_length
                self.distillation_epoch_loss = total_distillation_epoch_loss / self.config.miner.epoch_length
                self.remote_target_epoch_loss = total_remote_target_epoch_loss / self.config.miner.epoch_length
            
            # ---- Logs ----
            self.logs (
                progress_bar,
                iteration = iteration,
                output = output,
            )

            self.global_step += 1

        
        self.epoch += 1

    # ---- Training call ----
    def train ( self, batch: dict ) -> SimpleNamespace:
        r""" Runs a single training batch through the nucleus and applies a gradient update.
            Args:
                batch ( dict, `required`):
                    training batch dictionary.
            Returns:
                output = SimpleNamespace (
                    local_context (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Representations produced by the nucleus's distillation-model prior to producing the hidden units.
                    local_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Hidden layer representations produced using the local_context.
                    local_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__vocab_size__)`, `required`):
                        GPT2 MLM target predictions produced using local_hidden.
                    local_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `required`):
                        GPT2 MLM loss computed from the local_target.
                    remote_context (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Representations returned from the nucleus.remote function after querying the network.
                    remote_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Hidden layer representations produced using the remote_context.
                    remote_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,  bittensor.__vocab_size__)`, `required`):
                        GPT MLM Target predictions produced using remote_hidden.
                    remote_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `required`):
                        GPT2 MLM loss computed from the remote_target.
                    distillation_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `required`):
                        Distillation loss between local_context and remote_context.
            )
        """
        # Zeros out gradients for next accummulation
        self.optimizer.zero_grad()

        # ---- Forward pass ----
        
        inputs = batch['inputs']
        output = self.nucleus.remote_forward (
            inputs = inputs.to( self.device ),
            training = True,
        )

        # ---- Backward pass ----
        
        output.loss = output.local_target_loss + output.distillation_loss + output.remote_target_loss
        output.loss.backward() # Accumulates gradients on the nucleus.
        clip_grad_norm_(self.nucleus.parameters(), self.config.miner.clip_gradients)
        self.optimizer.step() # Applies accumulated gradients.
        
        # ---- Update global loss ----
        return output

    # ---- Axon Forward call ----
    def forward_text ( self, pubkey:str, inputs_x: torch.FloatTensor) -> torch.FloatTensor:
        r""" Subscribed to an axon servicing endpoint: processes forward messages from the wire.
            The arguments reflect an RPC request from another miner in the network, the response tensor
            should be the hidden units computed using the local context and with shape: [batch_size, sequence_len, __network_dim__].

            Args:
                pubkey ( str, `required`):
                    The public key of the caller.
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.
                modality ( bittensor.proto.Modality, `required`):
                    modality of inputs e.g. bittensor.proto.Modality.TEXT.

            Returns:
                outputs (:obj:`torch.FloatTensor`):
                    The nucleus's outputs as a torch tensor of shape [batch_size, sequence_len, __network_dim__]
        """
        def call(inputs):
            inputs_x = inputs.to( self.device )
            output = self.nucleus.local_forward (
                inputs = inputs_x
            )
            return output.local_hidden

        priority = self.neuron.metagraph.S[ self.neuron.metagraph.hotkeys.index(pubkey) ] / sys.getsizeof(inputs_x)
        future = self.thread_pool.submit( call,inputs = inputs_x, priority = priority )
        return future.result(timeout = self.config.miner.timeout)

    # ---- Axon Backward call ----
    def backward_text ( self, pubkey:str, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor ) -> torch.FloatTensor:
        r""" Subscribed to an axon servicing endpoint: Processes backward messages from the wire.
            Arguments reflect an RPC backward request from another miner in the network, the response tensor
            should be the gradients of the miner's nucleus w.r.t to the inputs_x and the passed output grads_dy.

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
                outputs (:obj:`torch.FloatTensor`, `optional`):
                    The gradients w.r.t to the inputs [batch_size, sequence_len, -1]
        """
        if self.config.miner.accumulate_remote_gradients:
            def call(input,grad):
                with torch.enable_grad():
                    # ---- Set up inputs for gradient computations.
                    outputs_y = self.nucleus.local_forward( inputs = input ).local_context.to( self.device )
                    # ---- The backward call will accumulate gradients on our parameters.
                
                    torch.autograd.backward (
                        tensors = [outputs_y],
                        grad_tensors = [grad]
                    )
                    return inputs_x.grad if inputs_x.grad != None else None                    

            priority = self.neuron.metagraph.S[ self.neuron.metagraph.hotkeys.index(pubkey) ] / sys.getsizeof(inputs_x)
            self.thread_pool.submit(call, input=inputs_x.to( self.device ), grad=grads_dy.to( self.device ), priority=priority)

    def checkpoint( self ):
        r""" Optionally Saves, updates and then reloads the miner training state.
        """
        last_saved = self.get_saved_state()
        if last_saved == None or last_saved['epoch_loss'] >= self.epoch_loss:
            self.save()

        # Checks if epochs managed to diverage
        if not math.isfinite(self.epoch_loss):
            logger.error('Incorrect epoch loss detected, reloading to previous saved state')
            self.reload()

    def get_saved_state( self ):
        r""" Returns a saved state dict or none.
        """
        try:
            return torch.load("{}/model.torch".format( self.config.miner.full_path ))
        except Exception as e:
            logger.warning('No saved model found with error: {}', e)
            logger.info('Initalizing with new model')
            return None

    def reload( self ):
        r""" Reloads/updates the training state from the disk.
        """
        state_dict = self.get_saved_state()

        # --- loads and syncs metagraph
        try:
            bittensor.neuron.metagraph.load().sync().save()

        except:
            bittensor.neuron.metagraph.sync().save()

        # ---- Load training state.
        self.epoch = state_dict['epoch']
        self.epoch_loss = state_dict['epoch_loss']
        self.global_step = state_dict['global_step']
        if 'network' in state_dict.keys() and bittensor.neuron.subtensor.network == state_dict['network']: # checks if you are loading into the same network
            chain_growth = bittensor.neuron.metagraph.n.item()- state_dict['nucleus_state']['chain_weights'].shape[0]
            #updates the shape of nucleus chain weights
            self.nucleus.chain_weights = nn.Parameter(
                torch.ones(
                    list(state_dict['nucleus_state']['chain_weights'].shape),
                    requires_grad=True
                )
            )
        else:
            logger.exception('Incorrect Network setting between miner input and saved state. Please use the same network')
            raise Exception('Network does not match saved state')

        self.nucleus.load_state_dict( state_dict['nucleus_state'], strict=False )
        self.nucleus.chain_weights = nn.Parameter(torch.cat([self.nucleus.chain_weights, torch.ones([chain_growth],dtype=torch.float32,requires_grad=True)]))
        self.nucleus.to( self.device ) # Load nucleus

        # --- Load optimizer.
        self.optimizer = torch.optim.SGD(
            [{"params": self.nucleus.parameters()}],
            lr = state_dict['optimizer_state']['param_groups'][0]['lr'],
            weight_decay = state_dict['optimizer_state']['param_groups'][0]['weight_decay'],
        )
        bittensor.logging.success( prefix = 'Reloaded model', sufix = '<blue>{}/model.torch</blue>'.format( self.config.miner.full_path ))

    def save( self ):
        r""" Saves the training state to disk.
        """
        try:
            state_dict = {
                'epoch': self.epoch,
                'epoch_loss': self.epoch_loss,
                'global_step': self.global_step,
                'nucleus_state': self.nucleus.state_dict(), # Save nucleus state.
                'optimizer_state': self.optimizer.state_dict(), # Save optimizer.
                'network': bittensor.neuron.subtensor.network # Save Network
            }
            torch.save( state_dict, "{}/model.torch".format( self.config.miner.full_path ) )
            bittensor.logging.success(prefix='Saved model', sufix='<blue>{}/model.torch</blue>'.format( self.config.miner.full_path ) )
        except Exception as e:
            logger.exception('Failed to save model with error:{}', e)

    def set_chain_weights( self ):
        r""" Sets the chain weights.
        """
        try:
            real_topk = min( self.config.miner.n_topk_chain_weights , bittensor.neuron.metagraph.n.item() )
            topk_weights, topk_uids = torch.topk( self.nucleus.chain_weights.detach(), k = real_topk )
            normalized_topk_weights = torch.nn.functional.normalize( topk_weights - torch.min( topk_weights ), p = 1, dim = 0)
            did_set = bittensor.neuron.subtensor.timeout_set_weights(
                timeout=5,
                uids = topk_uids,
                weights = normalized_topk_weights,
                wait_for_inclusion = True,
                wallet = bittensor.neuron.wallet,
            )
            if did_set:
                bittensor.logging.success(prefix='Set weights:', sufix='{}'.format(self.nucleus.chain_weights.tolist()))
            else:
                logger.warning('Failed to set weights on chain.')

        except Exception as e:
            logger.error('Failure setting weights on chain with error: {}', e)

    # ---- Training logs ----
    def logs( self, progress_bar, iteration:int, output: SimpleNamespace ):
        r""" Called after every training step. Displays miner state to screen.
        """
        self_uid = bittensor.neuron.metagraph.hotkeys.index(bittensor.neuron.wallet.hotkey.ss58_address)
        stake = bittensor.neuron.metagraph.S[ self_uid ].item()
        rank = bittensor.neuron.metagraph.R[ self_uid ].item()
        incentive = bittensor.neuron.metagraph.I[ self_uid ].item()
        info = {
            'GS': colored('{}'.format(self.global_step), 'red'),
            'LS': colored('{}'.format(iteration), 'blue'),
            'Epoch': colored('{}'.format(self.epoch+1), 'green'),
            'Loss': colored('{:.4f}'.format(self.epoch_loss), 'yellow'),
            'Best': colored('{:.4f}'.format(self.best_epoch_loss), 'red'),
            'L-loss': colored('{:.4f}'.format(output.local_target_loss.item()), 'blue'),
            'R-loss': colored('{:.4f}'.format(output.remote_target_loss.item()), 'green'),
            'D-loss': colored('{:.4f}'.format(output.distillation_loss.item()), 'yellow'),
            'nPeers': colored(bittensor.neuron.metagraph.n.item(), 'red'),
            'Stake(\u03C4)': colored('{:.3f}'.format(stake), 'green'),
            'Rank(\u03C4)': colored('{:.3f}'.format(rank), 'blue'),
            'Incentive(\u03C4/block)': colored('{:.6f}'.format(incentive), 'yellow'),
            'L-accuracy': colored('{}'.format(output.local_accuracy), 'red'),
        }
        
        if self.config.neuron.use_wandb and (iteration + 1) % (self.config.miner.epoch_length ) == 0:
            if self.config.neuron.use_wandb:
                wandb_info = {
                    'remote_target_loss':output.remote_target_loss.item(),
                    'distillation_loss':output.distillation_loss.item(),
                    'local_target_loss': output.local_target_loss.item(),
                    'remote_target_epoch_loss': self.remote_target_epoch_loss,
                    'distillation_epoch_loss': self.distillation_epoch_loss,
                    'local_target_epoch_loss': self.local_target_epoch_loss,
                    'local_accuracy':output.local_accuracy,
                    'Number of Peers':bittensor.neuron.metagraph.n.item(),
                    'Stake':stake,
                    'Rank':rank,
                    'Incentive':incentive,
                    'Axon QPS':bittensor.neuron.axon.stats.qps.value,
                    'Axon in bytes (total)':bittensor.neuron.axon.stats.total_in_bytes.value,
                    'Axon out bytes (total)':bittensor.neuron.axon.stats.total_out_bytes.value,
                    'Sync with metagraph': self.sync_count,
                    'Data size': self.epoch_data_size,
                    }
                #removing normalization of chain weights for display
                normalized_chain_weights =  F.softmax (self.nucleus.chain_weights.detach())
                respond_rate = self.responded_peers_count / self.quested_peers_count
                endpoints = bittensor.neuron.metagraph.endpoint_objs
                for uid in bittensor.neuron.metagraph.uids.tolist():
                    pubkey = endpoints[uid].hotkey
                    if normalized_chain_weights[uid].item() > 0:
                        if self.nucleus.chain_weights.grad != None:
                            weight_dif = -self.nucleus.chain_weights.grad[uid].item()
                        else:
                            weight_dif = 0

                        if weight_dif > 0:
                            info[str(uid)] = colored('{:.4f}'.format(normalized_chain_weights[uid]), 'green')
                        elif weight_dif == 0:
                            info[str(uid)] = colored('{:.4f}'.format(normalized_chain_weights[uid]), 'white')
                        else:
                            info[str(uid)] = colored('{:.4f}'.format(normalized_chain_weights[uid]), 'red')
                        if self.config.neuron.use_wandb:
                            wandb_info[f'Chain weights (norm) uid: {str(uid)}']= normalized_chain_weights[uid]
                            wandb_info[f'Chain weights (w/o norm) uid: {str(uid)}']= self.nucleus.chain_weights[uid]

                            wandb_info[f'Quested uid: {str(uid)}']= self.quested_peers_count[uid]
                            wandb_info[f'Responded uid: {str(uid)}']= self.responded_peers_count[uid]
                            wandb_info[f'Respond rate uid: {str(uid)}']= respond_rate[uid]
                            # wandb_info[f'Axon in byte uid: {str(uid)}']= bittensor.neuron.axon.stats.in_bytes_per_pubkey[pubkey]
                            # wandb_info[f'Axon out byte uid: {str(uid)}']= bittensor.neuron.axon.stats.out_bytes_per_pubkey[pubkey]
            try:
                wandb.log(wandb_info)
            except Exception as e:
                logger.warning('Failed to update weights and biases with error:{}', e)

        progress_bar.set_infos( info )

    def blacklist(self,pubkey:str) -> bool:
        r"""Axon security blacklisting, used to blacklist message from low stake members
        Currently, this is not turned on.
        """
        uid =self.neuron.metagraph.hotkeys.index(pubkey)
        if self.neuron.metagraph.S[uid] < self.config.miner.blacklist:
            return True
        else:
            return False

if __name__ == "__main__":
    Miner().run()
