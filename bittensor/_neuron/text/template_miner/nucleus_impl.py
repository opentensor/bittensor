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

import argparse
import bittensor
import math
import torch

from loguru import logger; logger = logger.opt(colors=True)
from types import SimpleNamespace
import torch.nn as nn

import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

def jacobian(y, x, create_graph=False,hessian =False):                                                               
    jac = []                                                                                          
    flat_y = y.reshape(-1)                                                                            
    grad_y = torch.zeros_like(flat_y)                                                                 
    for i in range(len(flat_y)): 
        if hessian ==True and flat_y[i].item() == 0:
            grad_x = torch.zeros_like(x)
            jac.append(grad_x.reshape(x.shape)) 
            print('skipped')
            pass
        else:
            print(flat_y[i].item())                                                                         
            grad_y[i] = 1.
            print(grad_y)                                                                             
            grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
            print(grad_x)
            jac.append(grad_x.reshape(x.shape))                                                           
            grad_y[i] = 0.                                                                                
    return torch.stack(jac).reshape(y.shape + x.shape)       
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
        local_layers = TransformerEncoderLayer( bittensor.__network_dim__, self.config.nucleus.nhead, self.config.nucleus.nhid, self.config.nucleus.dropout, batch_first=True)
        local_hidden_layers = TransformerEncoderLayer( bittensor.__network_dim__, self.config.nucleus.nhead, self.config.nucleus.nhid, self.config.nucleus.dropout, batch_first=True )
        self.local_pos_encoder = PositionalEncoding(bittensor.__network_dim__, self.config.nucleus.dropout)
        self.local_encoder = TransformerEncoder( local_layers, self.config.nucleus.nlayers )
        self.local_hidden = TransformerEncoder( local_hidden_layers, self.config.nucleus.nlayers )
        self.local_decoder = nn.Linear( bittensor.__network_dim__, bittensor.__vocab_size__ , bias=False)

        # Remote Model
        remote_context_layers = TransformerEncoderLayer( bittensor.__network_dim__, self.config.nucleus.nhead, self.config.nucleus.nhid, self.config.nucleus.dropout, batch_first=True )
        self.remote_hidden = TransformerEncoder( remote_context_layers, self.config.nucleus.nlayers )
        self.remote_decoder = nn.Linear( bittensor.__network_dim__, bittensor.__vocab_size__ , bias=False)

        self.loss_fct = nn.CrossEntropyLoss()
        self.noise_multiplier = self.config.nucleus.noise_multiplier
        self.peer_weights = nn.Parameter(torch.ones( [0] , requires_grad=True))
        self.init_weights()
        self.metagraph = None
        self.dendrite = None

    def init_weights(self):
        initrange = 0.1
        self.remote_decoder.weight.data.uniform_(-initrange, initrange)
        self.local_decoder.weight.data.uniform_(-initrange, initrange)

    def compute_scores ( self, loss ):
        """Computes salience scores for each peer in the network w.r.t the loss. 
        We use a simplified fishers information score. score_i = hessian_ii * peer_weight_i^2
        """
        peer_weights_d1 = jacobian(loss, self.peer_weights, create_graph=True)
        if peer_weights_d1 == None: return torch.ones_like( self.peer_weights ) * (1 / self.metagraph().n.item()) # None if no grad w.r.t the chain weights.
        peer_weights_d2 = jacobian(peer_weights_d1, self.peer_weights, hessian=True)
        second_order = (peer_weights_d2.detach() * (torch.outer(-self.peer_weights.detach(),-self.peer_weights.detach()))/2 ).sum(dim=1)
        first_order = (peer_weights_d1.detach()* -self.peer_weights.detach())
        validator_scores =  second_order + first_order
        return validator_scores

    def local_forward(self, inputs: torch.LongTensor, training: bool = True) -> SimpleNamespace:
        """ Forward pass through local transformer model of nucleus.
            Args:
                inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, block_size)`, `required`):
                    Input batch of batch_size token sequences each of length block_size, where
                    each token is :obj:`torch.int64` in range [0, bittensor.__vocab_size__ - 1]
                training (:obj:`bool')`, `optional`, defaults to True):
                    Switch to True if this forward pass computes a CLM loss.

            Returns:
                SimpleNamespace {
                    local_context (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Transformer encoding produced using embedded inputs.
                    local_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__vocab_size__)`, `optional`):
                        Transformer encoding produced using local_context.
                    local_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__vocab_size__)`, `optional`):
                        Next token prediction logits produced using local_hidden.
                    local_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`):
                        Next token prediction loss using local_hidden.
                    local_accuracy (:obj:`float`, `optional`):
                        Next token prediction accuracy using local_hidden.
                }
        """
        # To be filled.
        output = SimpleNamespace()

        # https://pytorch.org/docs/1.8.1/generated/torch.nn.Transformer.html#torch.nn.Transformer.forward
        # src: (S, N, E) the sequence to the encoder (required).
        # src_mask: (S, S) the mask for the src sequence (optional).
        # where S is the source sequence length, N is the batch size, E is the feature number

        # inputs.shape = [batch_size, sequence_len]
        sequence_len = inputs.shape[1]

        # src_mask: attention mask adds -inf to positions not allowed to attend, preventing forward-looking when
        #           predicting each token in the sequence.
        # src_mask.shape = [sequence_len, sequence_len]
        src_mask = torch.triu(torch.ones(sequence_len, sequence_len) * float('-inf'), diagonal=1)
        src_mask = src_mask.to(self.config.neuron.device)

        # embedding: retrieve learned representation vectors for input vocabulary tokens.
        # inputs.shape = [batch_size, sequence_len]
        # embedding.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        embedding = self.embedding(inputs)

        # local_context: hidden layer encoding of sequence with local_context.
        # local_context.shape = [sequence_len, batch_size, bittensor.__network_dim__]
        local_context = self.local_encoder(embedding, mask=src_mask) * math.sqrt(bittensor.__network_dim__)

        # local_context: adding positional encoding to local_context.
        # local_context.shape = [sequence_len, batch_size, bittensor.__network_dim__]
        local_context = self.local_pos_encoder(local_context)

        # external expects output.local_context shape = [batch_size, sequence_len, bittensor.__network_dim__]
        output.local_context = local_context

        if training:
            # local_hidden: local model which learns a new projection from the local_context
            # local_hidden.shape = [sequence_len, batch_size, bittensor.__vocab_size__]
            local_hidden = self.local_hidden(local_context.detach(), mask=src_mask)

            # local_target: projection of local_hidden onto target dimension.
            # local_target.shape = [sequence_len, batch_size, bittensor.__vocab_size__]
            local_target = self.local_decoder(local_hidden)

            # external expects output shape = [batch_size, sequence_len, bittensor.__network_dim__]
            output.local_hidden = local_hidden
            output.local_target = local_target

            # local_target_loss: MLM loss between local_target and passed targets.
            # local_target_loss.shape = [1]
            shift_logits = output.local_target[..., :-1, :].contiguous()
            shift_labels = inputs[..., 1:].contiguous()
            output.local_target_loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            predictions = shift_logits.detach().max(2).indices
            output.local_accuracy = (predictions == shift_labels).sum().item() / predictions.nelement()
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
        output.remote_context, output.query_uids, output.responses, output.total_uids = self.remote( inputs )

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
        active_uids = torch.where(self.metagraph().active > 0)[0]
        active_peer_weights = self.peer_weights[active_uids]

        # ---- Topk Weights ---- (TODO: check if the gaussians are enough disrupt the chain weights)
        real_topk = min( self.config.nucleus.topk, self.metagraph().n.item(), len(active_uids))
        std = torch.std(active_peer_weights).item() if torch.std(active_peer_weights).item() else self.config.nucleus.noise_offset
        noise = torch.normal( 0, std, size=( active_peer_weights.size())).to( self.config.neuron.device ) * self.noise_multiplier
        topk_weights, topk_idx = bittensor.unbiased_topk(active_peer_weights + noise , real_topk, dim=0)
        topk_uids = active_uids[topk_idx]

        # ---- Filter endpoints ----
        endpoints = self.metagraph().endpoints[ topk_uids ]

        # ---- Query network ----
        responses, return_ops, query_times = self.dendrite.forward_text (
            endpoints = endpoints.to('cpu'),
            inputs = inputs
        )

        # ---- Join based on weights ----
        joining_uids= torch.where( return_ops == bittensor.proto.ReturnCode.Success )[0]
        joining_weights = F.softmax( topk_weights[(return_ops == bittensor.proto.ReturnCode.Success)], dim = 0 ) 
        output = torch.zeros( (inputs.shape[0], inputs.shape[1], bittensor.__network_dim__)).to( self.config.neuron.device )
        for index, joining_weight in enumerate( joining_weights ):
            output += responses[joining_uids[index]].to( self.config.neuron.device ) * joining_weight

        # ---- Punish peers with non-successful return ops ----
        with torch.no_grad():
            self.peer_weights[topk_uids[(return_ops != bittensor.proto.ReturnCode.Success)]] -=  self.config.nucleus.punishment
            self.peer_weights[self.peer_weights < -1] = -1 #lower bound for chain weights
        
        return output, topk_uids[joining_uids], responses, topk_uids

