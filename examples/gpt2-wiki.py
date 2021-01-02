#!/bin/python3
"""GPT2 Language Modelling miner

This file demonstrates training the GPT2 neuron with language modelling.

Example:
        $ python neurons/gpt2-wiki.py

"""
import argparse
import math
import os
import time
import torch
import torch.nn.functional as F
import traceback
import time

from termcolor import colored
from munch import Munch
from datasets import load_dataset
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

import bittensor
from bittensor.neuron import Neuron
from bittensor.utils.logging import log_all
from bittensor.config import Config
from bittensor.synapses.gpt2 import GPT2LMSynapse, nextbatch
from pytorch_transformers import WarmupCosineWithHardRestartsSchedule

def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--session.learning_rate', default=0.01, type=float, help='Training initial learning rate.')
    parser.add_argument('--session.momentum', default=0.98, type=float, help='Training initial momentum for SGD.')
    parser.add_argument('--session.epoch_length', default=10, type=int, help='Iterations of training per epoch')
    parser.add_argument('--session.batch_size_train', default=1, type=int, help='Training batch size.')
    parser.add_argument('--session.sync_interval', default=100, type=int, help='Batches before we sync with chain and emit new weights.')
    parser.add_argument('--session.log_interval', default=10, type=int, help='Batches before we log session info.')
    parser.add_argument('--session.accumulation_interval', default=1, type=int, help='Batches before we apply acummulated gradients.')
    parser.add_argument('--session.apply_remote_gradients', default=False, type=bool, help='If true, neuron applies gradients which accumulate from remotes calls.')
    parser.add_argument('--session.root_dir', default='data/', type=str,  help='Root path to load and save data associated with each session')
    parser.add_argument('--session.name', default='gpt-wiki', type=str, help='Trials for this session go in session.root / session.name')
    parser.add_argument('--session.trial_uid', default=str(time.time()).split('.')[0], type=str, help='Saved models go in session.root_dir / session.name / session.uid')
    parser.add_argument('--session.record_log', default=True, help='Record all logs when running this session')
    GPT2LMSynapse.add_args(parser)
    Neuron.add_args(parser)

def check_config(config: Munch):
    assert config.session.momentum > 0 and config.session.momentum < 1, "momentum must be a value between 0 and 1"
    assert config.session.batch_size_train > 0, "batch_size_train must a positive value"
    assert config.session.learning_rate > 0, "learning_rate must be a positive value."
    full_path = '{}/{}/{}'.format(config.session.root_dir, config.session.name, config.session.trial_uid)
    config.session.full_path = full_path
    if not os.path.exists(config.session.full_path):
        os.makedirs(config.session.full_path)
    GPT2LMSynapse.check_config(config)
    Neuron.check_config(config)

# Neuron main.
def main(config: Munch, neuron: Neuron):

    # ---- Model ----
    model = GPT2LMSynapse(config, neuron)

    # ---- Optimizer ----
    optimizer = torch.optim.SGD(model.parameters(), lr = config.session.learning_rate, momentum=config.session.momentum)
    scheduler = WarmupCosineWithHardRestartsSchedule(optimizer, 50, 300)

    # ---- Dataset ----
    # 74 million sentences pulled from books.
    dataset = load_dataset('ag_news')['train']

    # ---- Logging ----
    tensorboard = SummaryWriter(log_dir = config.session.full_path)
    if config.session.record_log:
        logger.add("{}_{}.log".format(config.session.name, config.session.trial_uid),format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")

    # ---- Train Epoch ----
    def train(epoch: int, global_step: int, row_weights: torch.Tensor):
        # ----- Init training state ---
        model.train()
        training_loss = 0.0
        for local_step in range(config.session.epoch_length):
            try:

                # ---- Forward pass ----
                inputs = nextbatch(dataset, config.session.batch_size_train, bittensor.__tokenizer__())
                output = model(
                    inputs,
                    training = True,
                    remote = True # WITH rpc-queries made to the network
                )

                # ---- Backward pass ----
                output.remote_target_loss.backward() # Accumulates gradients on the model.
                optimizer.step() # Applies accumulated gradients.
                optimizer.zero_grad() # Zeros out gradients for next accummulation

                # ---- Train row weights ----
                batch_weights = torch.mean(output.weights, axis = 0) # Average over batch.
                row_weights = (1 - 0.03) * row_weights + 0.03 * batch_weights # Moving avg update.
                row_weights = F.normalize(row_weights, p = 1, dim = 0) # Ensure normalization.

                # ---- Step logs ----
                logger.info('GS: {} LS: {} Epoch: {} \t Local Target Loss: {}\tRemote Target Loss: {}\tDistillation Loss: {}\t Dendrite: {}\t Axon: {}',
                        colored('{}'.format(global_step), 'red'),
                        colored('{}'.format(local_step), 'blue'),
                        colored('{}'.format(epoch), 'green'),
                        colored('{:.4f}'.format(output.local_target_loss.item()), 'green'),
                        colored('{:.4f}'.format(output.remote_target_loss.item()), 'blue'),
                        colored('{:.4f}'.format(output.distillation_loss.item()), 'red'),
                        neuron.dendrite,
                        neuron.axon)
                tensorboard.add_scalar('Rloss', output.remote_target_loss.item(), global_step)
                tensorboard.add_scalar('Lloss', output.local_target_loss.item(), global_step)
                tensorboard.add_scalar('Dloss', output.distillation_loss.item(), global_step)

                # ---- Step increments ----
                global_step += 1
                training_loss += output.local_target_loss.item()
                torch.cuda.empty_cache()
                del output

            # --- Catch Errors during training ----
            except Exception as e:
                logger.error('Exception in training script with error: {}', e)
                logger.info(traceback.print_exc())
                logger.info('Continuing to train.')

        return training_loss, global_step, row_weights

    epoch = -1
    global_step = 0
    best_train_loss = math.inf
    while True:
        epoch += 1

        # ---- Train Model ----
        training_loss, global_step, trained_weights = train(epoch, global_step, neuron.metagraph.row_weights)
        scheduler.step()

        # ---- Emitting weights to chain. ----
        neuron.metagraph.emit( trained_weights, wait_for_inclusion = True ) # Sets my row-weights on the chain.

        # ---- Sync metagraph ----
        neuron.metagraph.sync() # Pulls the latest metagraph state (with my update.)
    
        # ---- Update Axon Priority ----
        neuron.axon.set_priority( neuron.metagraph.neurons, neuron.metagraph.col_weights ) # Sets the nucleus-backend request priority.

        # ---- Save best loss and model ----
        if training_loss and epoch % 10 == 0:
            if training_loss < best_train_loss:
                best_train_loss = training_loss # update best train loss
                logger.info( 'Saving/Serving model: epoch: {}, loss: {}, path: {}/model.torch'.format(epoch, best_train_loss, config.session.full_path))
                torch.save( {'epoch': epoch, 'model': model.state_dict(), 'loss': best_train_loss},"{}/model.torch".format(config.session.full_path))
                tensorboard.add_scalar('Train loss', training_loss, global_step)


if __name__ == "__main__":
    # ---- Load command line args ----
    parser = argparse.ArgumentParser(); add_args(parser) 
    config = Config.to_config(parser); check_config(config)
    logger.info(Config.toString(config))
   
    # ---- Build Neuron ----
    neuron = Neuron(config)

    # ---- Start Neuron ----
    with neuron:
        main(config, neuron)

