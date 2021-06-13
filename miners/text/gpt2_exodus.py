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
import traceback
import os
import sys

from termcolor import colored
from typing import List
from qqdm import qqdm, format_str
from datetime import datetime
from loguru import logger; logger = logger.opt(colors=True)
from types import SimpleNamespace
from nuclei.gpt2 import GPT2Nucleus
from routers.sgmoe import SGMOERouter
from torch.nn.utils import clip_grad_norm_
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter
import bittensor.utils.networking as net

class miner:

    def __init__( self, config: 'bittensor.config' = None ):
        r""" Initializes a miner with the passed config.
        """
        if config == None: config = miner.config()
        self.config = config; miner.check_config( self.config ); print ( self.config )
        self.wallet = bittensor.wallet( 
            config = self.config.wallet 
        )
        self.dendrite = bittensor.dendrite( 
            config = self.config.dendrite, 
            wallet = self.wallet 
        )
        self.subtensor = bittensor.subtensor( 
            config = self.config.subtensor 
        )
        self.metagraph = bittensor.metagraph( 
            config = self.config.metagraph 
        )
        self.axon = bittensor.axon( 
            config = self.config.axon, 
            wallet = self.wallet, 
            forward_callback = self.forward,
            backward_callback = self.backward
        )
        self.dataset = bittensor.dataloader(
            config = self.config.dataloader
        )
        self.device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
        self.router = SGMOERouter( 
            config = self.config.router 
        ).to( self.device )
        self.nucleus = GPT2Nucleus( 
            config = self.config.nucleus,
            routing_callback = self.route
        ).to( self.device )
        self.optimizer = torch.optim.AdamW( 
            [
                {"params": self.router.parameters()}, 
                {"params": self.nucleus.parameters()}
            ], 
            lr = self.config.learning_rate, betas = (0.9, 0.95) 
        )
        self.mechanism_weights = torch.ones( [0] )
        self.epoch = 0
        self.global_step = 0
        self.epoch_loss = math.inf/2
        self.best_epoch_loss = math.inf

    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        r""" Checks/validates the config namspace object.
        """
        assert config.batch_size_train > 0, "batch_size_train must be a positive value"
        assert config.learning_rate > 0, "learning_rate must be a positive value."
        bittensor.wallet.check_config( config.wallet )
        bittensor.subtensor.check_config( config.subtensor )
        bittensor.metagraph.check_config( config.metagraph )
        bittensor.dataloader.check_config( config.dataloader )
        bittensor.dendrite.check_config( config.dendrite )
        bittensor.axon.check_config( config.axon )
        GPT2Nucleus.check_config( config.nucleus )
        SGMOERouter.check_config( config.router )
        full_path = os.path.expanduser('{}/{}/{}'.format( config.root_dir, config.wallet.name + "-" + config.wallet.hotkey, config.name ))
        config.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.full_path):
            os.makedirs(config.full_path)

    @staticmethod   
    def config() -> 'bittensor.Config':
        r""" Fills a config namespace object with information from the command line.
        """
        config = bittensor.config()
        parser = argparse.ArgumentParser()
        parser.add_argument('--miner.debug', dest='debug', action='store_true', help='''Turn on bittensor debugging information''', default=False)
        parser.add_argument('--miner.config', dest ='config', type=str, help='If set, arguments are overridden by passed file.')
        parser.add_argument('--miner.modality', dest ='modality', type=int, help='''Miner network modality. TEXT=0, IMAGE=1. Currently only allowed TEXT''', default=0)
        parser.add_argument('--miner.use_upnpc', dest ='use_upnpc', action='store_true', help='''Turns on port forwarding on your router using upnpc.''', default=False)
        parser.add_argument('--miner.record_log', dest='record_log', action='store_true', help='''Turns on logging to file.''', default=True)   
        parser.add_argument('--miner.root_dir', dest='root_dir', type=str, help='Root path to load and save data associated with each miner', default='~/.bittensor/miners/')
        parser.add_argument('--miner.use_tensorboard', dest ='use_tensorboard', action='store_true', help='Turn on bittensor logging to tensorboard', default=True)
        parser.add_argument('--miner.learning_rate', dest ='learning_rate', type=float, help='Training initial learning rate.', default=3e-2)
        parser.add_argument('--miner.weight_decay', dest ='weight_decay', type=float, help='nucleus parameter weight decay.', default=0.25) 
        parser.add_argument('--miner.lr_decay', dest ='lr_decay', type=bool, help='learning rate decay params: linear warmup followed by cosine decay to 10%% of original.', default=True)
        parser.add_argument('--miner.warmup_tokens', dest ='warmup_tokens', type=float, help='A linear LR warmup over the first miner.warmup_tokens tokens (default is 365 million)', default=375e6)
        parser.add_argument('--miner.final_tokens', dest ='final_tokens', type=float, help='At what point we reach 10%% of original LR', default=260e9)
        parser.add_argument('--miner.clip_gradients', dest='clip_gradients', type=float, help='Implement gradient clipping to avoid exploding loss on smaller architectures.', default=1.0)
        parser.add_argument('--miner.n_epochs', dest='n_epochs', type=int, help='Number of training epochs.', default=sys.maxsize )
        parser.add_argument('--miner.epoch_length', dest='epoch_length', type=int, help='Iterations of training per epoch', default=500)
        parser.add_argument('--miner.batch_size_train', dest='batch_size_train', type=int, help='Training batch size.', default=2)
        parser.add_argument('--miner.reload', dest='reload',action='store_true', help='''Reload training from previous trial run.''', default=False )
        parser.add_argument('--miner.restart_on_failure', dest='restart_on_failure', action='store_true', help='''Restart miner on unknown error.''', default=False)
        parser.add_argument('--miner.name', dest ='name', type=str, help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default='gpt2_exodus')
        parser.parse_known_args( namespace = config )
        bittensor.wallet.config( config )
        bittensor.subtensor.config( config )
        bittensor.metagraph.config( config )
        bittensor.dataloader.config( config )
        bittensor.dendrite.config( config )
        bittensor.axon.config( config )
        GPT2Nucleus.config( config )
        SGMOERouter.config( config )
        return config

    def __enter__(self):
        self.startup()

    def __exit__ ( self, exc_type, exc_value, exc_traceback ): 
        self.shutdown()

    def run( self ):
        r""" Miner main loop.
        """
        # ---- Setup ----
        with self:  

            # ---- Optionally reload previous run ----
            if self.config.reload:
                self.reload()
            else:
                self.save()

            # --- Run until n_epochs ----
            while self.epoch < self.config.n_epochs:
                try:
                    # ---- Checkpoint state ----
                    self.checkpoint()

                    # ---- Train state ----
                    self.run_epoch()

                    # ---- Set weights on chain ----
                    self.set_mechanism_weights()
                
                except KeyboardInterrupt:
                    # --- User ended ----
                    break

                except Exception as e:
                    # --- Unknown error ----
                    logger.exception('Unknown exception: {} with traceback {}', e, traceback.format_exc())
                    if self.config.restart_on_failure == True:
                        logger.info('Restarting from last saved state.')
                        self.reload()
                        continue
                    else:
                        break

    # --- Run Epoch ----
    def run_epoch( self ):
        r""" Runs through a single training epoch pulled from the dataloader.
        """
        # --- Init Epoch ----
        total_epoch_loss = 0.0
        epoch_batches = self.dataset.dataloader( self.config.epoch_length )
        progress_bar = qqdm(enumerate(epoch_batches), total=len(epoch_batches), desc=format_str('blue', f'Epoch Progress'))
        for iteration, (inputs) in progress_bar:

            # ---- Forward / Backward ----
            prev_mechanism_weights = self.mechanism_weights.tolist()
            output = self.train ( batch = { 'inputs': inputs } )
            next_mechanism_weights = self.mechanism_weights.tolist()
            total_epoch_loss += output.local_target_loss.item()

            # ---- Logs ----
            self.epoch_logs ( 
                progress_bar, 
                iteration = iteration, 
                output = output, 
                prev_mechanism_weights = prev_mechanism_weights, 
                next_mechanism_weights = next_mechanism_weights 
            )
            self.global_step += 1

        self.epoch_loss = total_epoch_loss / (iteration + 1) 
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
            inputs = inputs.to( self.device ),
            training = True,
        )

        # ---- Backward pass ----
        output.loss = output.local_target_loss + output.distillation_loss + output.remote_target_loss
        output.loss.backward() # Accumulates gradients on the nucleus.
        clip_grad_norm_(self.nucleus.parameters(), self.config.clip_gradients)
        clip_grad_norm_(self.router.parameters(), self.config.clip_gradients)
        self.optimizer.step() # Applies accumulated gradients.
        self.optimizer.zero_grad() # Zeros out gradients for next accummulation

        # ---- Update global loss ----
        return output

    # ---- Axon Forward call ----
    def forward ( self, pubkey:str, inputs: torch.FloatTensor, modality:int ) -> torch.FloatTensor:
        r""" Subscribed to an axon servicing endpoint, processes forward messages from the wire.
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
        inputs = inputs.to( self.device )
        output = self.nucleus.local_forward (
            inputs = inputs        
        )
        return output.local_hidden

    # ---- Axon Backward call ----
    def backward ( self, pubkey:str, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, modality:int ) -> torch.FloatTensor:
        r""" Subscribed to an axon servicing endpoint. Processes backward messages from the wire.
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
        r""" Subscribed to the nucleus as a callback which is made during remote training. 
            Accepts tokenized text inputs and a query. Routes text inputs to neurons
            based on that query. 

            Args:
                inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_dim)`, `required`): 
                    Tensor of tokenized sentences.
                
                query (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, query_dim)`, `required`): 
                    Context tensor used to select which neurons to query for each example.
            
            Returns:
                response (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`): 
                    Joined responses from the network call.
        """
        # ---- Forward messages through network ---- 
        outputs = self.router.forward_text( self.metagraph, self.dendrite, inputs, query )

        # ---- Train mechanism weights ----
        self.mechanism_weights = (1 - 0.1) * self.mechanism_weights + 0.1 * outputs.weights # Moving avg update.

        # ---- Return response -----
        return outputs.response

    def checkpoint( self ):
        r""" Saves, updates and then reload the miner training state.
        """
        last_saved = self.get_saved_state()
        if last_saved == None or last_saved['epoch_loss'] >= self.epoch_loss:
            self.save()
        self.metagraph.sync()
        self.metagraph.save()
        self.reload()

    def get_saved_state( self ):
        try:
            return torch.load("{}/model.torch".format( self.config.full_path ))
        except Exception as e:
            logger.exception('Failed to reload model with error: {}', e)
        
    def reload( self ):
        r""" Reloads the training state from the disk.
        """
        state_dict = self.get_saved_state()

        # ---- Load training state.
        self.epoch = state_dict['epoch']
        self.epoch_loss = state_dict['epoch_loss']
        self.global_step = state_dict['global_step']

        # ---- Load router and resize to the metagraph size.
        self.router.load_state_dict( state_dict['router_state'] ) # Load router
        self.router.sync_with_chain_state( self.metagraph ) # Resize the router.

        # ---- Load nucleus and attach the routing function.
        self.nucleus.load_state_dict( state_dict['nucleus_state'] ) # Load nucleus
        self.nucleus.attach( self )# Re-assign the routing function.

        # --- Load optimizer.
        optim_groups = [
            {"params": self.router.parameters() },
            {"params": self.nucleus.parameters() },
        ]
        self.optimizer = torch.optim.AdamW( optim_groups, lr = self.config.learning_rate, betas = (0.9, 0.95) )

        # ---- Load mechanism weights and pad to size.
        self.mechanism_weights = state_dict['mechanism_weights']
        self.mechanism_weights = torch.nn.functional.pad( 
            self.mechanism_weights, 
            pad = [0, self.metagraph.n - self.mechanism_weights.numel()], 
            value=0 
        ) 
        logger.success('Reloaded model from: <cyan>{}/model.torch</cyan>'.format( self.config.full_path ))

    def save( self ):
        r""" Saves the training state to disk.
        """
        try:
            state_dict = {
                'epoch': self.epoch,
                'epoch_loss': self.epoch_loss,
                'global_step': self.global_step,
                'mechanism_weights': self.mechanism_weights, # Save row.
                'router_state': self.router.state_dict(), # Save router state.
                'nucleus_state': self.nucleus.state_dict(), # Save nucleus state.
                'optimizer_state': self.optimizer.state_dict(), # Save optimizer.
            }
            torch.save( state_dict, "{}/model.torch".format( self.config.full_path, self.epoch_loss ))
            logger.success('Saved model to: <cyan>{}/model.torch</cyan>'.format( self.config.full_path ))
        except Exception as e:
             logger.exception('Failed to save model with error:{}', e)

    def set_mechanism_weights( self ):
        r""" Called after every training epoch, sets the mechanism weights on chain.
        """
        try:
            uids = self.metagraph.uids
            did_set = self.subtensor.set_weights(
                wallet = self.wallet,
                uids = uids,
                weights = self.mechanism_weights, 
                wait_for_inclusion = True
            )
            if did_set:
                logger.success('Successfully set weights with row:\n {}', self.mechanism_weights.tolist())
            else:
                logger.warning('Failed to set weights on chain.')
                self.subtensor = bittensor.subtensor( config = self.config.subtensor )
                self.subtensor.connect()

        except Exception as e:
            logger.error('Failure setting weights on chain with error: {}', e)

    def startup( self ):
        r""" Starts and subscribes the miner.
        """

        # ---- Setup logging ----
        if self.config.record_log == True:
            filepath = self.config.full_path + "/bittensor_output.log"
            logger.add (
                filepath,
                format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
                rotation="25 MB",
                retention="10 days"
            )
            logger.info('LOGGING is <green>ON</green> with sink: <cyan>{}</cyan>', filepath)
        else: 
            logger.info('LOGGING is <red>OFF</red>')

        # ---- Setup tensorboard ----
        if self.config.use_tensorboard == True:
            event_file_dir = self.config.full_path + '/tensorboard-' + '-'.join(str(datetime.now()).split())
            self.tensorboard = SummaryWriter( log_dir = event_file_dir )
            self._tensorboard_program = program.TensorBoard()
            self._tensorboard_program.configure(argv=[None, '--logdir', event_file_dir, '--load_fast=true'])
            self._tensorbaord_url = self._tensorboard_program.launch()
            logger.info('TENSORBOARD is <green>ON</green> with entrypoint: <cyan>http://localhost:6006/</cyan>', )
        else: 
            logger.info('TENSORBOARD is <red>OFF</red>')

        # ---- Setup debugging ----
        if self.config.debug: bittensor.__debug_on__ = True; logger.info('DEBUG is <green>ON</green>')
        else: logger.info('DEBUG is <red>OFF</red>')

        # ---- Setup UPNPC ----
        if self.config.use_upnpc: 
            logger.info('UPNPC is <green>ON</green>')
            try:
                self.external_port = net.upnpc_create_port_map( local_port = self.axon.local_port )
            except net.UPNPCException as upnpc_exception:
                logger.critical('Failed to hole-punch with upnpc')
                raise RuntimeError('Failed to hole-punch with upnpc')
        else: 
            logger.info('UPNPC is <red>OFF</red>')
            self.external_port = self.config.axon.local_port

        # ---- Get external ip ----
        logger.info('\nFinding external ip...')
        try:
            self.external_ip = net.get_external_ip()
        except net.ExternalIPNotFound as external_port_exception:
            logger.critical('Unable to attain your external ip. Check your internet connection.')
            raise RuntimeError('Unable to attain your external ip. Check your internet connection.')
        logger.success('Found external ip: <cyan>{}</cyan>', self.external_ip)

        # ---- Setup Wallet. ----
        logger.info('\nLoading wallet...')
        if not self.wallet.has_coldkeypub:
            self.wallet.create_new_coldkey( n_words = 12, use_password = True )
        assert self.wallet.has_coldkeypub
        if not self.wallet.has_hotkey:
            self.wallet.create_new_hotkey( n_words = 12, use_password = False )
        assert self.wallet.has_hotkey

        # ---- Setup metagraph ----
        self.metagraph.load()
        self.metagraph.sync()
        self.metagraph.save()

        # ---- Connect to chain ----
        logger.info('\nConnecting to network...')
        self.subtensor.connect()
        if not self.subtensor.is_connected():
            logger.critical('Failed to connect subtensor to network:<cyan>{}</cyan>', self.subtensor.network)
            raise RuntimeError('Failed to connect subtensor to network:{}'.format(self.subtensor.network)) 

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
            raise RuntimeError('Failed to subscribe neuron.')

        # ---- Starting axon ----
        logger.info('\nStarting Axon...')
        self.axon.start()

    def shutdown ( self ): 
        r""" Shutsdown the miner and it's dependencies.
        """
        # ---- Stop axon ----
        logger.info('\nStopping Axon...')
        self.axon.stop()

    # ---- QQDM Training logs ----
    def epoch_logs( self, progress_bar, iteration:int, output: SimpleNamespace, prev_mechanism_weights: List[float], next_mechanism_weights: List[float] ):
        r""" Called after every training step. Displays miner state to screen.
        """
        self_uid = self.metagraph.hotkeys.index( self.wallet.hotkey.public_key )
        stake = self.metagraph.S[ self_uid ].item()
        rank = self.metagraph.R[ self_uid ].item()
        incentive = self.metagraph.I[ self_uid ].item()
        info = {
            'GS': colored('{}'.format(self.global_step), 'red'),
            'LS': colored('{}'.format(iteration), 'blue'),
            'Epoch': colored('{}'.format(self.epoch+1), 'green'),
            'Loss': colored('{:.4f}'.format(self.epoch_loss), 'yellow'),
            'Best': colored('{:.4f}'.format(self.best_epoch_loss), 'red'),
            'L-loss': colored('{:.4f}'.format(output.local_target_loss.item()), 'blue'),
            'R-loss': colored('{:.4f}'.format(output.remote_target_loss.item()), 'green'),
            'D-loss': colored('{:.4f}'.format(output.distillation_loss.item()), 'yellow'),
            'nPeers': colored(self.metagraph.n.item(), 'red'),
            'Stake(\u03C4)': colored('{:.3f}'.format(stake), 'green'),
            'Rank(\u03C4)': colored('{:.3f}'.format(rank), 'blue'),
            'Incentive(\u03C4/block)': colored('{:.6f}'.format(incentive), 'yellow'),
        } 
        for uid in self.metagraph.uids.tolist():
            if next_mechanism_weights[uid] != 0:
                weight_dif = next_mechanism_weights[uid] - prev_mechanism_weights[uid]
                if weight_dif > 0:
                    info[colored(str(uid), 'green')] = colored('{:.4f}'.format(next_mechanism_weights[uid]), 'green')
                elif weight_dif == 0:
                    info[str(uid)] = colored('{:.4f}'.format(next_mechanism_weights[uid]), 'white')
                else:
                    info[colored(str(uid), 'red')] = colored('{:.4f}'.format(next_mechanism_weights[uid]), 'red')

        progress_bar.set_infos( info )

        if self.config.use_tensorboard:
            self.tensorboard.add_scalar('R-loss', output.remote_target_loss.item(), self.global_step)
            self.tensorboard.add_scalar('L-loss', output.local_target_loss.item(), self.global_step)
            self.tensorboard.add_scalar('D-loss', output.distillation_loss.item(), self.global_step)


if __name__ == "__main__":
    miner().run()
