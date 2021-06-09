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
    $ python miners/text/gpt2_exodus.py

"""

import argparse
import bittensor
import math
import torch

from loguru import logger
from types import SimpleNamespace
from nuclei.gpt2 import GPT2Nucleus
from routers.sgmoe import SGMOERouter
from torch.nn.utils import clip_grad_norm_

class miner:

    def __init__( self, config: 'bittensor.config' = None ):
        if config == None: config = miner.config().miner
        self.config = config; print ( self.config )
        self.router = SGMOERouter( config = self.config.router )
        self.nucleus = GPT2Nucleus( config = self.config.nucleus )
        self.axon = bittensor.axon( config = self.config.axon )
        self.wallet = bittensor.wallet( config = self.config.wallet )
        self.dendrite = bittensor.dendrite( config = self.config.dendrite )
        self.subtensor = bittensor.subtensor( config = self.config.subtensor )
        self.metagraph = bittensor.metagraph( config = self.config.metagraph )
        self.row_weights = torch.ones( [0] )
        self.epoch = 0
        self.epoch_loss = math.inf
        self.global_step = 0
        self.refresh()

    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        assert config.miner.batch_size_train > 0, "batch_size_train must a positive value"
        assert config.miner.learning_rate > 0, "learning_rate must be a positive value."
        bittensor.wallet.check_config( config.wallet )
        bittensor.subtensor.check_config( config.subtensor )
        bittensor.metagraph.check_config( config.metagraph )
        bittensor.dataloader.check_config( config.dataloader )
        bittensor.dendrite.check_config( config.dendrite )
        bittensor.axon.check_config( config.axon )
        GPT2Nucleus.check_config( config.nucleus )
        SGMOERouter.check_config( config.router )

    @staticmethod   
    def config( config: 'bittensor.Config' = None, namespace: str = 'miner' ) -> 'bittensor.Config':
        if config == None: config = bittensor.config()
        miner_config = bittensor.config()
        config[ namespace ] = miner_config
        parser = argparse.ArgumentParser()
        if namespace != '': namespace += '.'
        parser.add_argument('--' + namespace + 'debug', dest='debug', default=False, action='store_true', help='''Turn on bittensor debugging information''')
        parser.add_argument('--' + namespace + 'config', dest ='config', type=str, help='If set, arguments are overridden by passed file.')
        parser.add_argument('--' + namespace + 'modality', dest ='modality', default=0, type=int, help='''Miner network modality. TEXT=0, IMAGE=1. Currently only allowed TEXT''')
        parser.add_argument('--' + namespace + 'use_upnpc', dest ='use_upnpc', default=False, action='store_true', help='''Turns on port forwarding on your router using upnpc.''')
        parser.add_argument('--' + namespace + 'record_log', dest='record_log',  default=False, action='store_true', help='''Turns on logging to file.''')   
        parser.add_argument('--' + namespace + 'root_dir', dest='root_dir', default='~/.bittensor/miners/', type=str, help='Root path to load and save data associated with each miner')
        parser.add_argument('--' + namespace + 'use_tensorboard', dest ='use_tensorboard', default=True, action='store_true', help='Turn on bittensor logging to tensorboard')
        parser.add_argument('--' + namespace + 'learning_rate', dest ='learning_rate', default=3e-2, type=float, help='Training initial learning rate.')
        parser.add_argument('--' + namespace + 'weight_decay', dest ='weight_decay', default=0.25, type=float, help='nucleus parameter weight decay.') 
        parser.add_argument('--' + namespace + 'lr_decay', dest ='lr_decay', default=True, type=bool, help='learning rate decay params: linear warmup followed by cosine decay to 10%% of original.' )
        parser.add_argument('--' + namespace + 'warmup_tokens', dest ='warmup_tokens', default=375e6, type=float, help='A linear LR warmup over the first miner.warmup_tokens tokens (default is 365 million)')
        parser.add_argument('--' + namespace + 'final_tokens', dest ='final_tokens', default=260e9, type=float, help='At what point we reach 10%% of original LR')
        parser.add_argument('--' + namespace + 'clip_gradients', dest='clip_gradients', default=1.0, type=float, help='Implement gradient clipping to avoid exploding loss on smaller architectures.')
        parser.add_argument('--' + namespace + 'n_epochs', dest='n_epochs', default=-1, type=int, help='Number of training epochs.')
        parser.add_argument('--' + namespace + 'epoch_length', dest='epoch_length', default=500, type=int, help='Iterations of training per epoch')
        parser.add_argument('--' + namespace + 'batch_size_train', dest='batch_size_train',  default=2, type=int, help='Training batch size.')
        parser.add_argument('--' + namespace + 'name', dest ='name', default='gpt2_genesis', type=str, help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ')
        parser.parse_known_args( namespace = miner_config )
        bittensor.wallet.config( miner_config )
        bittensor.subtensor.config( miner_config )
        bittensor.metagraph.config( miner_config )
        bittensor.dataloader.config( miner_config )
        bittensor.dendrite.config( miner_config )
        bittensor.axon.config( miner_config )
        GPT2Nucleus.config( miner_config )
        SGMOERouter.config( miner_config )
        return config

    def __enter__(self):
        pass

    def __exit_(self):
        pass

    def run ( self ):
        with self:
            pass

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

    def refresh( self ):
        self.save_state()
        self.reload_sate()
        self.metagraph.sync()
        optim_groups = [
            {"params": self.router.parameters() },
            {"params": self.nucleus.parameters() },
        ]
        self.optimizer = torch.optim.AdamW( optim_groups, lr = self.config.miner.learning_rate, betas = (0.9, 0.95) )

    def reload_sate( self ):
        try:
            state_dict = torch.load("{}/model.torch".format( self.config.miner.full_path ))
            self.epoch = state_dict['epoch']
            self.epoch_loss = state_dict['epoch_loss']
            self.global_step = state_dict['global_step']
            self.row_weights = state_dict['row_weights'] # Load row weights
            self.nucleus.load_state_dict( state_dict['nucleus_state'] ) # Load nucleus
            self.router.load_state_dict( state_dict['router_state']) # Load router
            self.nucleus.attach( self )# Re-assign the routing function.
            self.router.sync_with_chain_state( self.metagraph ) # Resize the router.
            self.optimizer.load_state_dict( state_dict['optimizer_state'] ) # Load optimizer.
            self.optimizer = self.configure_optimizers( self.optimizer )
            logger.success( 'Reloaded model from: <cyan>{}/model.torch</cyan>\n'.format( self.config.miner.full_path ))
        except Exception as e:
            logger.exception('Failed to reload model with error: {}', e)

    def save_state( self ):
        try:
            state_dict = {
                'epoch': self.epoch,
                'epoch_loss': self.epoch_loss,
                'global_step': self.global_step,
                'row_weights': self.row_weights, # Save row.
                'router_state': self.router.state_dict(), # Save router state.
                'nucleus_state': self.nucleus.state_dict(), # Save nucleus state.
                'optimizer_state': self.optimizer.state_dict(), # Save optimizer.
            }
            torch.save( state_dict, "{}/model.torch".format( self.config.miner.full_path, self.epoch_loss ))
            logger.success( 'Saved model to: <cyan>{}/model.torch</cyan>\n'.format( self.config.miner.full_path ))
        except Exception as e:
             logger.exception('Failed to save model with error:{}', e)

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


if __name__ == "__main__":
    miner().run()
