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
import wandb

from termcolor import colored
from qqdm import qqdm, format_str
from loguru import logger; logger = logger.opt(colors=True)
from types import SimpleNamespace
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn

import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Globals 
global_dendrite = None
global_metagraph = None

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
        self.peer_weights = nn.Parameter(torch.ones( [0] , requires_grad=True))
        self.noise_offset = 0.0000001
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

    def compute_scores ( self, loss ):
        """Computes salience scores for each peer in the network w.r.t the loss. 
        We use a simplified fishers information score. score_i = hessian_ii * peer_weight_i^2
        """
        peer_weights_d1 = torch.autograd.grad(loss, self.peer_weights, create_graph=True, retain_graph=True, allow_unused=True)[0]
        if peer_weights_d1 == None: return torch.ones_like( self.peer_weights ) * (1 / global_metagraph.n.item()) # None if no grad w.r.t the chain weights.
        peer_weights_d2 = torch.autograd.grad(peer_weights_d1.sum(), self.peer_weights, retain_graph=True, allow_unused=True )[0]
        validator_scores =  peer_weights_d2 * (self.peer_weights**2)/2  
        return validator_scores

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
        output.remote_context = self.remote( inputs )

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
        """ Forwards the inputs through the network, selects the topk peers based on self.peer_weights.
        Args:
            inputs (:obj:`torch.int64` of shape :obj:`(batch_size, sequence_len)`, `required`):
                Batch_size length list of text sentences.
        Returns:
            outputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `optional`):
                Joined hidden layer responses from peers.
        """

        # ---- Get active peers and their weights ---- 
        active_uids = torch.where(global_metagraph.active > 0)[0]
        active_peer_weights = self.peer_weights[active_uids]

        # ---- Topk Weights ---- (TODO: check if the gaussians are enough disrupt the chain weights)
        real_topk = min( self.config.nucleus.topk, global_metagraph.n.item(), len(active_uids))
        std = torch.std(active_peer_weights).item() if torch.std(active_peer_weights).item() else self.noise_offset
        noise = torch.normal( 0, std, size=( active_peer_weights.size())).to( self.config.miner.device )
        topk_weights, topk_idx = torch.topk(active_peer_weights + noise , real_topk, dim=0)
        topk_uids = active_uids[topk_idx]

        # ---- Filter endpoints ----
        endpoints = global_metagraph.endpoints[ topk_uids ]

        # ---- Query network ----
        responses, return_ops, query_times = global_dendrite.forward_text (
            endpoints = endpoints,
            inputs = inputs
        )

        # ---- Join based on weights ----
        joining_uids= torch.where( return_ops == bittensor.proto.ReturnCode.Success )[0]
        joining_weights = F.softmax( topk_weights[(return_ops == bittensor.proto.ReturnCode.Success)], dim = 0 ) 
        output = torch.zeros( (inputs.shape[0], inputs.shape[1], bittensor.__network_dim__)).to( self.config.miner.device )
        for index, joining_weight in enumerate( joining_weights ):
            output += responses[joining_uids[index]].to( self.config.miner.device ) * joining_weight

        # ---- Punish peers with non-successful return ops ----
        with torch.no_grad():
            self.peer_weights[topk_uids[(return_ops != bittensor.proto.ReturnCode.Success)]] -=  self.config.nucleus.punishment
            self.peer_weights[self.peer_weights < -1] = -1 #lower bound for chain weights
        
        return output

class Miner:

    def __init__( self, config: 'bittensor.config' = None ):
        r""" Initializes a miner with the passed config.
        """
        if config == None: config = Miner.config()
        self.config = config; Miner.check_config( self.config ); print ( self.config )
        bittensor.logging (
            config = self.config,
            logging_dir = self.config.miner.full_path,
        )
        # Bittensor backend
        self.wallet = bittensor.wallet(
            config = self.config
        )
        self.subtensor = bittensor.subtensor(
            config = self.config
        )
        self.metagraph = bittensor.metagraph(
            config = self.config,
            subtensor = self.subtensor
        )
        global global_metagraph; global_metagraph = self.metagraph
        self.dendrite = bittensor.dendrite(
            config = self.config,
            wallet = self.wallet
        )
        global global_dendrite; global_dendrite = self.dendrite
        self.axon = bittensor.axon (
            config = self.config,
            wallet = self.wallet,
            forward_text = self.forward_text,
            backward_text = self.backward_text,
            blacklist = self.blacklist,
        )
        # Miner training device.
        self.device = torch.device(
            device = self.config.miner.device
        )
        # Dataset of text.
        self.dataset = bittensor.dataset (
            config = self.config
        )
        # Trainable machine learning model.
        self.nucleus = Nucleus(
            config = self.config,
        ).to( self.device )
        # Torch optimizer.
        # the peer_weights layer has it own learning weight, other layers follow the default
        self.optimizer = torch.optim.SGD(
            [ {'params': self.nucleus.peer_weights, 'lr': self.config.miner.learning_rate_chain} ],
            lr = self.config.miner.learning_rate,
            momentum = self.config.miner.momentum,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
            step_size = 1.0,
            gamma = 0.95
        )

        # ---- Init of when was the block that we sync to 
        self.last_sync_block = 0
        # ---- Store all the stats
        self.stats = SimpleNamespace(
            global_step = 0,
            epoch_data_size = 0,
            epoch_sync_count = 0,
            local_target_epoch_loss = math.inf,
            distillation_epoch_loss = math.inf,
            remote_target_epoch_loss = math.inf,
            local_epoch_acc = 0,
            best_epoch_loss = math.inf,
            ema_scores = torch.ones(0).to(self.device)
        )
        # ---- Decay factor for fisher ema score 
        self.fisher_ema_decay = 0.995

    @staticmethod
    def config() -> 'bittensor.Config':
        r""" Fills a config namespace object with defaults or information from the command line.
        """
        # ---- Add miner args.
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, help='If set, defaults are overridden by passed file.')
        parser.add_argument('--miner.learning_rate', type=float, help='Training initial learning rate.', default=1)
        parser.add_argument('--miner.learning_rate_chain', type=float, help='Training initial learning rate.', default=1)
        parser.add_argument('--miner.weight_decay', type=float, help='nucleus parameter weight decay.', default=0.25)
        parser.add_argument('--miner.momentum', type=float, help='optimizer momentum.', default=0.8)
        parser.add_argument('--miner.clip_gradients', type=float, help='Implement gradient clipping to avoid exploding loss on smaller architectures.', default=1.0)
        parser.add_argument('--miner.n_epochs', type=int, help='Number of training epochs.', default=sys.maxsize )
        parser.add_argument('--miner.epoch_length', type=int, help='Iterations of training per epoch', default=100)
        parser.add_argument('--miner.batch_size_train', type=int, help='Training batch size.', default=2)
        parser.add_argument('--miner.restart_on_failure',  action='store_true', help='''Restart miner on unknown error.''', default=False)
        parser.add_argument('--miner.compute_remote_gradients', action='store_true', help='''Does the miner compute and return gradients from backward queries.''', default=False)
        parser.add_argument('--miner.accumulate_remote_gradients', action='store_true', help='''Does the miner accumulate remote gradients from backward queries.''', default=False)
        parser.add_argument('--miner.n_topk_peer_weights', type=int, help='Maximum number of weights to submit to chain', default=100 )
        parser.add_argument('--miner.name', type=str, help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default='template_miner')
        parser.add_argument('--miner.device', type=str, help='miner default training device cpu/cuda', default=("cuda" if torch.cuda.is_available() else "cpu"))
        parser.add_argument('--miner.timeout', type=int, help='Number of seconds to wait for axon request', default=10)
        parser.add_argument('--miner.blacklist', type=float, help='Amount of stake (tao) in order not to get blacklisted', default=0)
        parser.add_argument('--miner.sync_block_time', type=int, help='How often the sync the miner with metagraph, in terms of block time', default=15)
        parser.add_argument('--miner.restart', type=bool, help='If True, train the miner from the beginning', default=False)
        parser.add_argument('--miner.use_wandb', action='store_true', help='''Miner activates its weights and biases powers''', default=False)
        parser.add_argument('--miner.use_upnpc', action='store_true', help='''Miner attempts to port forward axon using upnpc.''', default=False)

        bittensor.logging.add_args( parser )
        bittensor.wallet.add_args( parser )
        bittensor.subtensor.add_args( parser )
        bittensor.metagraph.add_args( parser )
        bittensor.dataset.add_args( parser )
        bittensor.dendrite.add_args( parser )
        bittensor.axon.add_args( parser )
        bittensor.wandb.add_args( parser )
        Nucleus.add_args( parser ) 

        return bittensor.config( parser )

    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        r""" Checks/validates the config namespace object.
        """
        assert config.miner.batch_size_train > 0, "batch_size_train must be a positive value"
        assert config.miner.learning_rate > 0, "learning_rate must be a positive value."
        bittensor.logging.check_config( config )
        bittensor.wallet.check_config( config )
        bittensor.subtensor.check_config( config )
        bittensor.metagraph.check_config( config )
        bittensor.dataset.check_config( config )
        bittensor.dendrite.check_config( config )
        bittensor.axon.check_config( config )
        bittensor.wandb.check_config( config )
        full_path = os.path.expanduser('{}/{}/{}/{}'.format( config.logging.logging_dir, config.wallet.name, config.wallet.hotkey, config.miner.name ))
        config.miner.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.miner.full_path):
            os.makedirs(config.miner.full_path)

    def __enter__(self):
        self.wallet.create().register()
        self.metagraph.sync().save()
        self.axon.start().serve (
            use_upnpc = self.config.miner.use_upnpc, 
            subtensor = self.subtensor
        )

    def __exit__ ( self, exc_type, exc_value, exc_traceback ):
        self.axon.stop()   
        print(exc_type, exc_value, exc_traceback)
    
    def run( self ):
        r""" Miner main loop.
        """
        # ---- Build Bittensor neuron ----
        with self:
            if self.config.miner.use_wandb:
                bittensor.wandb(
                    config = self.config,
                    cold_pubkey = self.wallet.coldkeypub.ss58_address,
                    hot_pubkey = self.wallet.hotkey.ss58_address,
                    root_dir = self.config.miner.full_path
                )

            # ---- Init run state ----
            self.epoch = 0            
            self.stats.ema_scores = torch.ones( self.metagraph.n.item()).to(self.device) * (1 / self.metagraph.n.item())

            # ---- reloads previous run if not restart ----
            if self.config.miner.restart:
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
            while self.epoch < self.config.miner.n_epochs:
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
                    end_block = start_block + self.config.miner.epoch_length
                    block_steps = [ block_delta for block_delta in range(start_block, end_block)]
                    progress_bar = qqdm( block_steps, total=len(block_steps), desc=format_str('white', f'Epoch:'))
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
                            clip_grad_norm_(self.nucleus.parameters(), self.config.miner.clip_gradients)
                            
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

                        # ---- Sync with metagraph if the current block >= last synced block + sync block time 
                        current_block = self.subtensor.get_current_block()
                        block_diff = current_block - self.last_sync_block
                        if block_diff >= self.config.miner.sync_block_time:
                            self.sync(current_block)                                                                                                                
                            self.last_sync_block = current_block
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
                    if self.config.miner.restart_on_failure == True:
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
        if self.config.miner.accumulate_remote_gradients:
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
        uid = self.metagraph.hotkeys.index(pubkey)
        if self.metagraph.S[uid].item() < self.config.miner.blacklist:
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
            return torch.load("{}/model.torch".format( self.config.miner.full_path ))
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
        self.optimizer = torch.optim.SGD(
            [{"params": self.nucleus.parameters()}],
            lr = state_dict['optimizer_state']['param_groups'][0]['lr'],
            momentum = state_dict['optimizer_state']['param_groups'][0]['momentum'],
        )
        bittensor.logging.success( prefix = 'Reloaded model', sufix = '<blue>{}/model.torch</blue>'.format( self.config.miner.full_path ))


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
            torch.save( state_dict, "{}/model.torch".format( self.config.miner.full_path ) )
            bittensor.logging.success(prefix='Saved model', sufix='<blue>{}/model.torch</xblue>'.format( self.config.miner.full_path ) )
        except Exception as e:
            logger.exception('Failed to save model with error:{}', e)

    def set_peer_weights( self ):
        r""" Sets the fisher ema score to peers.
        """

        try:
            k = min(self.config.miner.n_topk_peer_weights, self.metagraph.n.item())
            topk_scores, topk_uids = torch.topk( self.stats.ema_scores.detach(), k = k )
            did_set = self.subtensor.timeout_set_weights(
                timeout=10,
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

        # ---- Progress bar log
        info = {
            'GS': colored('{}'.format(self.stats.global_step), 'red'),
            'LS': colored('{}'.format(iteration), 'blue'),
            'Epoch': colored('{}'.format(self.epoch+1), 'green'),
            'Best': colored('{:.4f}'.format(self.stats.best_epoch_loss), 'red'),            
            'L-loss': colored('{:.4f}'.format(output.local_target_loss.item()), 'blue'),
            'R-loss': colored('{:.4f}'.format(output.remote_target_loss.item()), 'green'),
            'D-loss': colored('{:.4f}'.format(output.distillation_loss.item()), 'yellow'),
            'nPeers': colored(self.metagraph.n.item(), 'red'),
            'Stake(\u03C4)': colored('{:.3f}'.format(stake), 'green'),
            'Rank(\u03C4)': colored('{:.3f}'.format(rank), 'blue'),
            'Incentive(\u03C4/block)': colored('{:.6f}'.format(incentive), 'yellow'),
            'L-accuracy': colored('{}'.format(output.local_accuracy), 'red'),
        }
        # ---- Miner summary per peer for progress bar
        for uid in self.metagraph.uids.tolist():
            if normalized_peer_weights[uid].item() > 0:
                if self.nucleus.peer_weights.grad != None:
                    weight_diff = -self.nucleus.peer_weights.grad[uid].item()
                else:
                    weight_diff = 0

                color = ('green' if weight_diff > 0 else ('white' if weight_diff == 0 else 'red'))
                info[str(uid)] = colored('{:.4f}'.format(normalized_peer_weights[uid]), color)

        progress_bar.set_infos( info )

        # ---- wandb log if it is the end of epoch 
        if  self.config.miner.use_wandb and ((iteration + 1) % (self.config.miner.epoch_length ) == 0):
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



if __name__ == "__main__":
    Miner().run()
