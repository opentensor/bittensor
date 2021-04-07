#!/bin/python3
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
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from datasets import load_dataset



class AdamCorpus():

    def __init__(self, block_size: int, tokenizer=bittensor.__tokenizer__()):
        self.block_size = block_size
        self.tokenizer = tokenizer

        self.lines = load_dataset('wikitext', 'wikitext-103-raw-v1')['train']

    def __len__(self):
        return len(self.lines) - self.block_size

    def __getitem__(self, idx):
        """ Returns a batch of sentences from text dataset.

            Args:
                idx: index of data input

            Returns:
                x
        """

        chunk = self.lines[idx:idx + self.block_size]['text']

        dix = []
        block_num=0
        while block_num < self.block_size:
            tokenized = self.tokenizer(chunk[block_num], padding=True, truncation=True)['input_ids']
            for t in tokenized:
                if block_num < self.block_size:
                    dix.append(t)
                    block_num += 1


        x = torch.tensor(dix, dtype=torch.long)
        return x

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

        # ---- Dataset ----
        # The Genesis Dataset:
        # The dataset used to train Adam and his first 100 children.
        # Here block size = sequence length.
        self.dataset = AdamCorpus(self.model.get_block_size())
        self.tokens = 0

        # ---- Logging ----
        self.tensorboard = SummaryWriter(log_dir = self.config.miner.full_path)
        if self.config.miner.record_log:
            logger.add(self.config.miner.full_path + "/{}_{}.log".format(self.config.miner.name, self.config.miner.trial_uid),format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")
        
        logger.info("Model contains {} parameters".format(self.model.num_parameters))

    @staticmethod
    def default_config() -> Munch:
        parser = argparse.ArgumentParser()
        Miner.add_args(parser)
        config = bittensor.config.Config.to_config(parser)
        return config

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument('--miner.learning_rate', default=3e-2, type=float, help='Training initial learning rate.')
        parser.add_argument('--miner.weight_decay', default=0.25, help='Model parameter weight decay.')
        parser.add_argument('--miner.lr_decay', default=True, help='learning rate decay params: linear warmup followed by cosine decay to 10% of original.')
        parser.add_argument('--miner.warmup_tokens', default=375e6, help='A linear LR warmup over the first miner.warmup_tokens tokens (default is 365 million)')
        parser.add_argument('--miner.final_tokens', default=260e9, help='At what point we reach 10% of original LR')
        parser.add_argument('--miner.num_workers', default=1, help='Number of workers for data loader.')

        parser.add_argument('--miner.clip_gradients', default=1.0, type=float, help='Implement gradient clipping to avoid exploding loss on smaller architectures.')
        parser.add_argument('--miner.n_epochs', default=int(sys.maxsize), type=int, help='Number of training epochs.')
        parser.add_argument('--miner.epoch_length', default=500, type=int, help='Iterations of training per epoch')
        parser.add_argument('--miner.batch_size_train', default=2, type=int, help='Training batch size.')
        parser.add_argument('--miner.sync_interval', default=100, type=int, help='Batches before we sync with chain and emit new weights.')
        parser.add_argument('--miner.log_interval', default=10, type=int, help='Batches before we log miner info.')
        parser.add_argument('--miner.accumulation_interval', default=1, type=int, help='Batches before we apply acummulated gradients.')
        parser.add_argument('--miner.apply_remote_gradients', default=True, type=bool, help='If true, neuron applies gradients which accumulate from remotes calls.')
        parser.add_argument('--miner.root_dir', default='~/.bittensor/miners/', type=str,  help='Root path to load and save data associated with each miner')
        parser.add_argument('--miner.name', default='gpt2-genesis', type=str, help='Trials for this miner go in miner.root / miner.name')
        parser.add_argument('--miner.trial_uid', default=str(time.time()).split('.')[0], type=str, help='Saved models go in miner.root_dir / miner.name / miner.uid')
        parser.add_argument('--miner.record_log', default=False, help='Record all logs when running this miner')
        parser.add_argument('--miner.custom_dataset', default="~/.bittensor/bittensor/miners/TEXT/gpt2_genesis/genesis_dataset/", type=str, help='Custom datasets to train on.')
        parser.add_argument('--miner.config_file', type=str, help='config file to run this neuron, if not using cmd line arguments.')
        GPT2Synapse.add_args(parser)
        bittensor.neuron.Neuron.add_args(parser)

    @staticmethod
    def check_config(config: Munch):
        assert config.miner.batch_size_train > 0, "batch_size_train must a positive value"
        assert config.miner.learning_rate > 0, "learning_rate must be a positive value."
        config.miner.custom_dataset = os.path.expanduser(config.miner.custom_dataset)
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

                # --- Epoch logs ----
                #print(self.neuron.axon.__full_str__())
                #print(self.neuron.dendrite.__full_str__())
                #print(self.neuron.metagraph)

                # ---- Update Tensorboard ----
                self.neuron.dendrite.__to_tensorboard__(self.tensorboard, self.global_step)
                self.neuron.metagraph.__to_tensorboard__(self.tensorboard, self.global_step)
                self.neuron.axon.__to_tensorboard__(self.tensorboard, self.global_step)

                # ---- Save best loss and model ----
                if self.training_loss and self.training_loss < self.best_train_loss: #self.epoch % 10 == 0:
                    if self.training_loss < self.best_train_loss:
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


    def reset_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def shuffle_dataset_epoch_length(self):
        """Shuffles the miner's dataset so we get a shuffled, randomized dataset
        of length miner.epoch_length

        Returns:
            [list] : shuffled dataset of length miner.epoch_length
        """

        shuffled_dataset = []
        loader = DataLoader(self.dataset, shuffle=True,
                        batch_size=self.config.miner.batch_size_train,
                        num_workers=self.config.miner.num_workers)


        for it, batch in enumerate(loader):
            shuffled_dataset.append(batch)
            if it == self.config.miner.epoch_length:
                break

        return shuffled_dataset

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    # ---- Train Epoch ----
    def train(self):
        self.training_loss = 0.0

        def run_epoch():
            self.model.train(True)
            self.best_train_loss = math.inf
            losses = []

            # Re-create dataloader every time we call train
            # This way, since epoch_length < len(dataset), we can
            # make sure that the dataset is randomly shuffled each time
            # we train for an epoch.
            logger.info("Preparing dataset batch...")
            dataset = self.shuffle_dataset_epoch_length()
            pbar = tqdm(enumerate(dataset), total=len(dataset))


            #self.reset_learning_rate(self.config.miner.learning_rate)
            for it, (batch) in pbar:
                batch = batch.to(self.model.device)
                output = self.model.remote_forward(self.neuron, batch, training=True)
                loss = output.local_target_loss + output.distillation_loss + output.remote_target_loss
                loss.backward()

                clip_grad_norm_(self.model.parameters(), self.config.miner.clip_gradients)
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.decay_learning_rate(batch)

                losses.append(loss.item())

                 # ---- Train row weights ----
                batch_weights = torch.mean(output.router.weights, axis = 0).to(self.model.device) # Average over batch.
                self.row = (1 - 0.03) * self.row + 0.03 * batch_weights # Moving avg update.
                self.row = F.normalize(self.row, p = 1, dim = 0) # Ensure normalization.

                pbar.set_infos({
                    'GS': colored('{}'.format(self.global_step), 'red'),
                    'LS': colored('{}'.format(it), 'blue'),
                    'Epoch': colored('{}'.format(self.epoch+1), 'green'),
                    'Local loss': colored('{:.5f}'.format(output.local_target_loss.item()), 'red'),
                    'Remote loss': colored('{:.5f}'.format(output.remote_target_loss.item()), 'blue'),
                    'Distillation loss': colored('{:.5f}'.format(output.distillation_loss.item()), 'green'),
                    'Learning Rate:': colored('{:e}'.format(self.lr), 'white'),
                    'Axon': self.neuron.axon.__str__(),
                    'Dendrite': self.neuron.dendrite.__str__(),
                })

                self.tensorboard.add_scalar('Neuron/Rloss', output.remote_target_loss.item(), self.global_step)
                self.tensorboard.add_scalar('Neuron/Lloss', output.local_target_loss.item(), self.global_step)
                self.tensorboard.add_scalar('Neuron/Dloss', output.distillation_loss.item(), self.global_step)
                self.global_step += 1


            avg_loss = sum(losses) / len(losses)
            self.training_loss = avg_loss
            if avg_loss < self.best_train_loss:
                self.best_train_loss = avg_loss
                self.model_toolbox.save_model(self.config.miner.full_path,{'epoch': self.epoch, 'model_state_dict': self.model.state_dict(), 'loss': loss.item(), 'optimizer_state_dict': self.optimizer.state_dict()})


        run_epoch()


if __name__ == "__main__":
    # ---- Build and Run ----
    miner = Miner()
    logger.info(bittensor.config.Config.toString(miner.config))
    miner.run()
