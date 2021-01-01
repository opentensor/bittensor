"""GPT2 Language Modelling

This file demonstrates training the GPT2 neuron with language modelling.

Example:
        $ python neurons/gpt2-wiki.py

"""
import argparse
import math
import os
import pathlib
import time
import torch
import torch.nn.functional as F

from termcolor import colored
from munch import Munch
from datasets import load_dataset
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

import bittensor
from bittensor.neuron import Neuron
from bittensor.config import Config
from bittensor.utils.logging import log_all
from bittensor.synapses.gpt2 import GPT2LMSynapse, nextbatch

def add_args(parser: argparse.ArgumentParser):    
    parser.add_argument('--session.learning_rate', default=0.01, type=float, help='Training initial learning rate.')
    parser.add_argument('--session.momentum', default=0.98, type=float, help='Training initial momentum for SGD.')
    parser.add_argument('--session.batch_size_train', default=20, type=int, help='Training batch size.')
    parser.add_argument('--session.sync_interval', default=100, type=int, help='Batches before we sync with chain and emit new weights.')
    parser.add_argument('--session.log_interval', default=10, type=int, help='Batches before we log session info.')
    parser.add_argument('--session.accumulation_interval', default=1, type=int, help='Batches before we apply acummulated gradients.')
    parser.add_argument('--session.apply_remote_gradients', default=False, type=bool, help='If true, neuron applies gradients which accumulate from remotes calls.')
    parser.add_argument('--session.root_dir', default='data/', type=str,  help='Root path to load and save data associated with each session')
    parser.add_argument('--session.name', default='gpt-wiki', type=str, help='Trials for this session go in session.root / session.name')
    parser.add_argument('--session.uid', default=str(time.time()).split('.')[0], type=str, help='Saved models go in session.root_dir / session.name / session.uid')
    GPT2LMSynapse.add_args(parser)
    Neuron.add_args(parser)

def check_config(config: Munch):
    assert config.session.momentum > 0 and config.session.momentum < 1, "momentum must be a value between 0 and 1"
    assert config.session.batch_size_train > 0, "batch_size_train must be a positive value"
    assert config.session.learning_rate > 0, "learning rate must be be a positive value."
    full_path = '{}/{}/{}/'.format(config.session.root_dir, config.session.name, config.session.uid)
    config.session.full_path = full_path
    if not os.path.exists(config.session.full_path):
        os.makedirs(config.session.full_path)
    GPT2LMSynapse.check_config(config)
    Neuron.check_config(config)
    
# Neuron main.
def main(config, neuron):

    # ---- Build Model ----
    model = GPT2LMSynapse(config, neuron)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    neuron.axon.serve( model )

    # ---- Dataset ----
    # 74 million sentences pulled from books.
    dataset = load_dataset('bookcorpus')['train']

    # ---- Optimizer ----
    optimizer = torch.optim.SGD(model.parameters(), lr = config.session.learning_rate, momentum=config.session.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # ---- Tensorboard ----
    tensorboard = SummaryWriter(log_dir = config.session.full_path)
        
    # ---- Init training state ----
    neuron.metagraph.sync() # Sync with the chain.
    row_weights = neuron.metagraph.row_weights # Weight to others (zeros initially)
    col_weights = neuron.metagraph.col_weights # Other to me.

    # ---- Train forever ----
    model.train()
    step = -1; history = []; best_loss = math.inf; 
    while True:
        try:
            step += 1
            # ---- Next Batch ----
            inputs = nextbatch(dataset, config.session.batch_size_train, bittensor.__tokenizer__())

            # ---- Forward Pass ----
            output = model(inputs.to(model.device), training = True, remote = True)
            history.append(output)

            # ---- Accumulate Local Gradients ----
            loss = output.loss / config.session.accumulation_interval # Need to average accross accumulation steps.
            loss.backward() # Accumulates gradients on model via sum.

            # ---- Accumulate Remote Gradients  ----
            if config.session.apply_remote_gradients:
                # TODO (const): batch normalization over the gradients for consistency.
                n_grads = neuron.axon.gradients.qsize
                for _, (_, input_x, grads_dy, modality) in list(neuron.axon.gradients.queue):
                    grads_dy = grads_dy / (config.session.accumulation_interval * n_grads)
                    model.backward(input_x, grads_dy, modality)

            # ---- Apply Gradients ----
            if (step+1) % config.session.accumulation_interval == 0:
                optimizer.step() # Apply accumulated gradients.
                optimizer.zero_grad() # Zero grads for next accummulation 
                scheduler.step() # Update learning rate etc.
                neuron.axon.serve( model ) # Serve the newest model.

            # ---- Step Logs + Tensorboard ----
            logger.info('Step: {} \t Remote Loss: {:.6f}\t Local Loss: {:.6f}\t Distilation Loss: {:.6f}'.format(step, output.remote_target_loss.item(), output.local_target_loss.item(), output.distillation_loss.item()))
            tensorboard.add_scalar('Rloss', output.remote_target_loss.item(), step)
            tensorboard.add_scalar('Lloss', output.local_target_loss.item(), step)
            tensorboard.add_scalar('Dloss', output.distillation_loss.item(), step)
            if (step+1) % config.session.log_interval == 0:
                log_all(neuron, history); history = [] # Log batch history.


            # ---- Update Weights ----
            batch_weights = torch.mean(output.weights, axis = 0)
            row_weights = (1 - 0.05) * row_weights + 0.05 * batch_weights # Moving Avg weights.
            row_weights = F.normalize(row_weights, p = 1, dim = 0)  

            # ---- Sync State ----
            if (step+1) % config.session.sync_interval == 0:
                # ---- Emit weights and sync from chain ----
                logger.info('Emitting with weights {}', row_weights.tolist())
                neuron.metagraph.emit( row_weights, wait_for_inclusion = True ) # Set weights on chain.
                neuron.metagraph.sync() # Sync with the chain.
                
                # ---- Get row and col Weights.
                row_weights = neuron.metagraph.row_weights # Weight to others.
                col_weights = neuron.metagraph.col_weights # Other to me.

                # ---- Update Axon Priority ----
                col_weights = neuron.metagraph.col_weights # weights to me.
                neuron.axon.set_priority( neuron.metagraph.neurons, col_weights ) # Sets the nucleus-backend request priority.

            # --- Save Model ----
            if output.loss.item() < best_loss:
                best_loss = output.loss
                logger.info( 'Saving model: epoch: {}, loss: {}, path: {}/model.torch', step, output.loss, config.session.full_path)
                torch.save( {'epoch': step, 'model': model.state_dict(), 'loss': output.loss},"{}/model.torch".format(config.session.full_path))

        # --- Catch Errors during training ----
        except Exception as e:
            logger.error('Exection in training script with error: {}', e)
            logger.info('Continuing to train.')
    
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


