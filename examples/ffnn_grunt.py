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

from munch import Munch
from loguru import logger
from termcolor import colored
from datasets import load_dataset

import bittensor
from bittensor.neuron import Neuron
from bittensor.config import Config
from bittensor.synapses.ffnn import FFNNSynapse

def add_args(parser: argparse.ArgumentParser):    
    parser.add_argument('--session.learning_rate', default=0.01, type=float, help='Training initial learning rate.')
    parser.add_argument('--session.momentum', default=0.9, type=float, help='Training initial momentum for SGD.')
    parser.add_argument('--session.batch_size_train', default=64, type=int, help='Training batch size.')
    parser.add_argument('--session.batch_size_test', default=64, type=int, help='Testing batch size.')
    parser.add_argument('--session.log_interval', default=150, type=int, help='Batches until session prints log statements.')
    parser.add_argument('--session.sync_interval', default=150, type=int, help='Batches before we we sync with chain and emit new weights.')
    parser.add_argument('--neuron.apply_remote_gradients', default=False, type=bool, help='If true, neuron applies gradients which accumulate from remotes calls.')
    parser.add_argument('--session.root_dir', default='data/', type=str,  help='Root path to load and save data associated with each session')
    parser.add_argument('--session.name', default='ffnn-grunt', type=str, help='Trials for this session go in session.root / session.name')
    parser.add_argument('--session.uid', default=str(time.time()).split('.')[0], type=str, help='Saved models go in session.root_dir / session.name / session.uid')
    Neuron.add_args(parser)
    FFNNSynapse.add_args(parser)

def check_config(config: Munch):
    assert config.session.log_interval > 0, "log_interval dimension must be positive"
    assert config.session.momentum > 0 and config.session.momentum < 1, "momentum must be a value between 0 and 1"
    assert config.session.batch_size_train > 0, "batch_size_train must be a positive value"
    assert config.session.batch_size_test > 0, "batch_size_test must be a positive value"
    assert config.session.learning_rate > 0, "learning rate must be be a positive value."
    full_path = '{}/{}/{}/'.format(config.session.root_dir, config.session.name, config.session.uid)
    config.session.full_path = full_path
    if not os.path.exists(config.session.full_path):
        os.makedirs(config.session.full_path)
    FFNNSynapse.check_config(config)
    Neuron.check_config(config)
    
# Neuron main.
def main(config, neuron):

    # ---- Build FFNN Model ----
    model = FFNNSynapse(config, neuron)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    neuron.axon.serve( model )

    # ---- Optimizer ----
    optimizer = torch.optim.SGD(model.parameters(), lr = config.session.learning_rate, momentum=config.session.momentum)
        
    # ---- Init training state ----
    neuron.metagraph.sync() # Sync with the chain.

    # ---- Train forever ----
    model.train()
    step = -1; 
    while True:
        step += 1

        # ---- Poll until gradients ----
        public_key, inputs_x, grads_dy, modality_x = neuron.axon.gradients.get(block = True)

        # ---- Backward Gradients ----
        # TODO (const): batch normalization over the gradients for consistency.
        grads_dy = torch.where(torch.isnan(grads_dy), torch.zeros_like(grads_dy), grads_dy)
        model.backward(inputs_x, grads_dy, modality_x)

        # ---- Apply Gradients ----
        optimizer.step() # Apply accumulated gradients.
        optimizer.zero_grad() # Clear any lingering gradients

        # ---- Serve latest model ----
        neuron.axon.serve( model ) # Serve the newest model.
        logger.info('Step: {} \t Key: {} \t sum(W[:,0])', step, public_key, torch.sum(neuron.metagraph.col_weights).item())
    
        # ---- Sync State ----
        if (step + 1) % config.session.sync_interval == 0:
            # ---- Sync metagrapn from chain ----
            neuron.metagraph.sync() # Sync with the chain.
            
            # ---- Update Axon Priority ----
            neuron.axon.set_priority( neuron.metagraph.neurons, neuron.metagraph.col_weights ) # Sets the nucleus-backend request priority.

            # --- Save Model ----
            logger.info( 'Saving model: epoch: {}, sum(W[:,0]): {}, path: {}/{}/{}/model.torch', step, torch.sum(neuron.metagraph.col_weights).item(), config.session.full_path)
            torch.save( {'epoch': step, 'model': model.state_dict(), 'loss': torch.sum(neuron.metagraph.col_weights).item()},"{}//model.torch".format(config.session.full_path))                
        
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
