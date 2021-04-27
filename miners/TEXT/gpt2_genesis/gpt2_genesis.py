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

This file demonstrates training the GPT2 neuron with language modelling.

Example:
        $ python miners/TEXT/gpt2_genesis/gpt2_genesis.py

Look at the yaml config file to tweak the parameters of the model. To run with those
default configurations, run:
        $ cd miners/TEXT
        $ python gpt2_genesis/gpt2_genesis.py --session.config_file gpt2_genesis/gpt2_genesis_config.yaml


"""
import argparse
import math
import os
import sys
import time
import torch
import time
import bittensor
import torch.nn.functional as F


from termcolor import colored
from munch import Munch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from bittensor.utils.model_utils import ModelToolbox
from synapses.gpt2 import GPT2Synapse
from torch.nn.utils import clip_grad_norm_
from transformers import AdamW
from qqdm import qqdm, format_str
from bittensor.dataloaders.text_dataloader import GenesisTextDataloader

class Miner():

    def __init__(self, config: Munch = None, **kwargs):
        if config == None:
            config = Miner.default_config()
        bittensor.config.Config.update_with_kwargs(config.miner, kwargs)
        Miner.check_config(config)
        self.config = config

        # ---- Neuron ----
        self.neuron = bittensor.neuron.Neuron(self.config)

        # ---- Model ----
        self.model = GPT2Synapse( self.config )

        # ---- Model Load/Save tools ----
        self.model_toolbox = ModelToolbox(GPT2Synapse, AdamW)

        # ---- Optimizer ----
        self.optimizer = self.configure_optimizers()
        self.lr = self.config.miner.learning_rate
        self.training_loss = math.inf
        self.best_train_loss = math.inf

        # ---- Dataset ----
        # The Genesis Dataset:
        # The dataset used to train Adam and his first 100 children.
        # Here block size = sequence length.
        self.dataset = GenesisTextDataloader(self.config.miner.batch_size_train, self.model.get_block_size())

        # Set up the dataloader
        self.dataloader = dataset.dataloader(self.config.miner.epoch_length)
        logger.info("LENGTH: {}".format(len(self.dataloader)))

        self.tokens = 0

        # ---- Logging ----
        self.tensorboard = SummaryWriter(log_dir = self.config.miner.full_path)
        if self.config.miner.record_log == True:
            filepath = self.config.miner.full_path + "/{}_{}.log".format(self.config.miner.name, self.config.miner.trial_uid),
            logger.add (
                filepath,
                format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
                rotation="250 MB",
                retention="10 days"
            )
               
    @staticmethod
    def default_config() -> Munch:
        parser = argparse.ArgumentParser()
        Miner.add_args(parser)
        config = bittensor.config.Config.to_config(parser)
        return config

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            '--miner.learning_rate', 
            default=3e-2, 
            type=float, 
            help='Training initial learning rate.'
        )
        parser.add_argument(
            '--miner.weight_decay', 
            default=0.25, 
            type=float, 
            help='Model parameter weight decay.'
        )
        parser.add_argument(
            '--miner.lr_decay',
            default=True,
            type=bool,
            help='learning rate decay params: linear warmup followed by cosine decay to 10%% of original.'
        )
        parser.add_argument(
            '--miner.warmup_tokens',
            default=375e6,
            type=float,
            help='A linear LR warmup over the first miner.warmup_tokens tokens (default is 365 million)'
        )
        parser.add_argument(
            '--miner.final_tokens',
            default=260e9,
            type=float,
            help='At what point we reach 10%% of original LR'
        )

        parser.add_argument(
            '--miner.clip_gradients',
            default=1.0,
            type=float,
            help='Implement gradient clipping to avoid exploding loss on smaller architectures.'
        )
        parser.add_argument(
            '--miner.n_epochs', 
            default=int(sys.maxsize), 
            type=int, 
            help='Number of training epochs.'
        )
        parser.add_argument(
            '--miner.epoch_length', 
            default=500, 
            type=int, 
            help='Iterations of training per epoch'
        )
        parser.add_argument(
            '--miner.batch_size_train', 
            default=2, 
            type=int, 
            help='Training batch size.'
        )
        parser.add_argument (
            '--miner.root_dir',
            default='~/.bittensor/miners/',
            type=str,
            help='Root path to load and save data associated with each miner'
        )
        parser.add_argument (
            '--miner.name',
            default='gpt2-genesis',
            type=str,
            help='Trials for this miner go in miner.root / miner.name'
        )
        parser.add_argument (
            '--miner.trial_uid',
            default=str(time.time()).split('.')[0],
            type=str,
            help='Saved models go in miner.root_dir / miner.name / miner.uid'
        )
        parser.add_argument (
            '--miner.record_log',
            default=False,
            type=bool,
            help='Record all logs when running this miner')

        parser.add_argument (
            '--miner.config_file',
            type=str,
            help='config file to run this neuron, if not using cmd line arguments.'
        )
        parser.add_argument (
            '--debug', 
            dest='debug', 
            action='store_true', 
            help='''Turn on bittensor debugging information'''
        )
        parser.set_defaults ( 
            debug=False 
        )

        GPT2Synapse.add_args(parser)
        bittensor.neuron.Neuron.add_args(parser)

    @staticmethod
    def check_config(config: Munch):
        if config.debug:  bittensor.__log_level__ = 'TRACE'; logger.debug('DEBUG is ON')
        else: logger.info('DEBUG is OFF') 
        assert config.miner.batch_size_train > 0, "batch_size_train must a positive value"
        assert config.miner.learning_rate > 0, "learning_rate must be a positive value."
        full_path = '{}/{}/{}'.format(config.miner.root_dir, config.miner.name, config.miner.trial_uid)
        config.miner.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.miner.full_path):
            os.makedirs(config.miner.full_path)


    def configure_optimizers(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.

        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.Tanh)
        for mn, m in self.model.named_modules():
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
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.config.miner.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.config.miner.learning_rate, betas=(0.9, 0.95))
        return optimizer

    # --- Main loop ----
    def run (self):

        # ---- Subscribe ----
        with self.neuron:

            # ---- Weights ----
            self.row = self.neuron.metagraph.row.to(self.model.device)

            # --- Run state ---
            self.global_step = 0

            # --- Loop for epochs ---
            for self.epoch in range(self.config.miner.n_epochs):

                # ---- Serve ----
                self.neuron.axon.serve( self.model )

                # ---- Train Model ----
                self.train()

                # If model has borked for some reason, we need to make sure it doesn't emit weights
                # Instead, reload into previous version of model
                if torch.any(torch.isnan(torch.cat([param.view(-1) for param in self.model.parameters()]))):
                    self.model, self.optimizer = self.model_toolbox.load_model(self.config)
                    continue

                # ---- Emitting weights ----
                self.neuron.metagraph.set_weights(self.row, wait_for_inclusion = True) # Sets my row-weights on the chain.

                # ---- Sync metagraph ----
                self.neuron.metagraph.sync() # Pulls the latest metagraph state (with my update.)
                self.row = self.neuron.metagraph.row.to(self.model.device)

                # ---- Update Tensorboard ----
                self.neuron.dendrite.__to_tensorboard__(self.tensorboard, self.global_step)
                self.neuron.metagraph.__to_tensorboard__(self.tensorboard, self.global_step)
                self.neuron.axon.__to_tensorboard__(self.tensorboard, self.global_step)

                # ---- Save best loss and model ----
                if self.training_loss < self.best_train_loss: #self.epoch % 10 == 0:
                        self.best_train_loss = self.training_loss  # update best train loss
                        self.model_toolbox.save_model(
                            self.config.miner.full_path,
                            {
                                'epoch': self.epoch,
                                'model_state_dict': self.model.state_dict(),
                                'loss': self.best_train_loss,
                                'optimizer_state_dict': self.optimizer.state_dict(),
                            }
                        )
                        self.tensorboard.add_scalar('Neuron/Train_loss', self.training_loss, self.global_step)
                logger.info("This epoch's training loss: {}...Current best training loss: {}".format(self.training_loss, self.best_train_loss))


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

    # ---- Train Epoch ----
    def train(self):

        def run_epoch():
            self.model.train(True)
            losses = []

            # we train for an epoch.
            logger.info("Preparing dataset batch...")
            # Set up the dataloader
            dataloader = self.dataset.dataloader(self.config.miner.epoch_length)
            pbar = qqdm(enumerate(dataloader), total=len(dataloader), desc=format_str('blue', f'Epoch Progress'))
            for it, (batch) in pbar:
                # ---- Forward pass ----
                batch = batch.to(self.model.device)
                output = self.model.remote_forward(self.neuron, batch, training=True)

                # ---- Backward pass ----
                loss = output.local_target_loss + output.distillation_loss + output.remote_target_loss
                loss.backward()

                # ---- Gradient Step ----
                clip_grad_norm_(self.model.parameters(), self.config.miner.clip_gradients)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.decay_learning_rate(batch)
                losses.append(loss.item())

                # ---- Train row weights ----
                batch_weights = torch.mean(output.router.weights, axis = 0).to(self.model.device) # Average over batch.
                self.row = (1 - 0.03) * self.row + 0.03 * batch_weights # Moving avg update.
                self.row = F.normalize(self.row, p = 1, dim = 0) # Ensure normalization.

                # ---- Logging ----
                index = self.neuron.metagraph.state.index_for_uid[self.neuron.metagraph.uid]
                pbar.set_infos({
                    'GS': colored('{}'.format(self.global_step), 'red'),
                    'LS': colored('{}'.format(it), 'blue'),
                    'Epoch': colored('{}'.format(self.epoch+1), 'green'),
                    'L-loss': colored('{:.5f}'.format(output.local_target_loss.item()), 'red'),
                    'R-loss': colored('{:.5f}'.format(output.remote_target_loss.item()), 'blue'),
                    'D-loss': colored('{:.5f}'.format(output.distillation_loss.item()), 'green'),
                    'lr:': colored('{:e}'.format(self.lr), 'white'),
                    'nPeers': self.neuron.metagraph.n,
                    'Stake(\u03C4)': float(self.neuron.metagraph.S[index]),
                    'Rank(\u03C4)': float(self.neuron.metagraph.R[index]),
                    'Incentive(\u03C4/block)': float(self.neuron.metagraph.I[index]),
                    'Axon': self.neuron.axon.__str__(),
                    'Dendrite': self.neuron.dendrite.__str__(),
                })
                self.tensorboard.add_scalar('Neuron/Rloss', output.remote_target_loss.item(), self.global_step)
                self.tensorboard.add_scalar('Neuron/Lloss', output.local_target_loss.item(), self.global_step)
                self.tensorboard.add_scalar('Neuron/Dloss', output.distillation_loss.item(), self.global_step)
                self.global_step += 1


            avg_loss = sum(losses) / len(losses)
            self.training_loss = avg_loss

        run_epoch()


if __name__ == "__main__":
    # ---- Build and Run ----
    miner = Miner()
    logger.info(bittensor.config.Config.toString(miner.config))
    miner.run()
