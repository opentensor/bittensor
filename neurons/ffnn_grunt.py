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

import bittensor
from bittensor.utils.logging import log_col_weights, log_incentive, log_ranks
from bittensor.subtensor import Keypair
from bittensor.config import Config
from bittensor.synapses.ffnn import FFNNSynapse

def add_args(parser: argparse.ArgumentParser):    
    parser.add_argument('--neuron.datapath', default='data', type=str,help='Path to load and save data.')
    parser.add_argument('--neuron.learning_rate', default=0.005, type=float, help='Training initial learning rate.')
    parser.add_argument('--neuron.momentum', default=0.99, type=float, help='Training initial learning rate.')
    parser.add_argument('--neuron.sync_interval', default=100, type=int, help='Batches before we sync with chain and emit new weights.')
    parser.add_argument('--neuron.log_interval', default=10, type=int, help='Batches before we log session info.')
    parser.add_argument('--neuron.accumulation_interval', default=1, type=int, help='Batches before we apply acummulated gradients.')
    parser.add_argument('--neuron.apply_remote_gradients', default=False, type=bool, help='If true, neuron applies gradients which accumulate from remotes calls.')
    parser.add_argument('--neuron.name', default='ffnn-grunt', type=str, help='Trials for this neuron go in neuron.datapath / neuron.name')
    parser.add_argument('--neuron.trial_id', default=str(time.time()).split('.')[0], type=str, help='Saved models go in neuron.datapath / neuron.name / neuron.trial_id')
    FFNNSynapse.add_args(parser)

def check_config(config: Munch):
    assert config.neuron.momentum > 0 and config.neuron.momentum < 1, "momentum must be a value between 0 and 1"
    assert config.neuron.learning_rate > 0, "learning_rate must be a positive value."
    trial_path = '{}/{}/{}'.format(config.neuron.datapath, config.neuron.name, config.neuron.trial_id)
    config.neuron.trial_path = trial_path
    if not os.path.exists(config.neuron.trial_path):
        os.makedirs(config.neuron.trial_path)
    FFNNSynapse.check_config(config)
    
# Neuron main.
def main(config, session):

    # ---- Build FFNN Model ----
    model = FFNNSynapse(config, session)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    session.serve( model )

    # ---- Optimizer ----
    optimizer = torch.optim.SGD(model.parameters(), lr = config.neuron.learning_rate, momentum=config.neuron.momentum)
        
    # ---- Init training state ----
    session.metagraph.sync() # Sync with the chain.
    col_weights = session.metagraph.col_weights # Weights from others to me.
    priority_map = dict(zip(session.metagraph.public_keys, col_weights.tolist()))
    session.axon.set_priority( priority_map )

    # ---- Train forever ----
    model.train()
    step = -1; 
    while True:
        step += 1

        # ---- Poll until gradients ----
        public_key, inputs_x, grads_dy, modality_x = session.axon.gradients.get(block = True)

        # ---- Backward Gradients ----
        # TODO (const): batch normalization over the gradients for consistency.
        grads_dy = torch.where(torch.isnan(grads_dy), torch.zeros_like(grads_dy), grads_dy)
        model.backward(inputs_x, grads_dy, modality_x)

        # ---- Apply Gradients ----
        optimizer.step() # Apply accumulated gradients.
        optimizer.zero_grad() # Clear any lingering gradients

        # ---- Serve latest model ----
        session.serve( model ) # Serve the newest model.
        logger.info('Step: {} \t Key: {} \t sum(W[:,0])', step, public_key, torch.sum(col_weights).item())
    
        # ---- Sync State ----
        if (step + 1) % config.neuron.sync_interval == 0:
            # ---- Sync metagrapn from chain ----
            session.metagraph.sync() # Sync with the chain.
            
            # ---- Get col Weights.
            col_weights = session.metagraph.col_weights # Other to me.

            # ---- Update Axon Priority ----
            session.axon.set_priority( session.metagraph.neurons, col_weights ) # Sets the nucleus-backend request priority.

            # --- Save Model ----
            logger.info( 'Saving model: epoch: {}, sum(W[:,0]): {}, path: {}/{}/{}/model.torch', step, torch.sum(col_weights).item(), config.neuron.datapath, config.neuron.name, config.neuron.trial_id)
            torch.save( {'epoch': step, 'model': model.state_dict(), 'loss': torch.sum(col_weights).item()},"{}/{}/{}/model.torch".format(config.neuron.datapath , config.neuron.name, config.neuron.trial_id))

        # ---- Session Logs ----
        if (step+1) % config.neuron.log_interval == 0:
            log_col_weights(session)
            log_incentive(session)
            log_ranks(session)
                
        
if __name__ == "__main__":
    # ---- Load Bittensor config ----
    parser = argparse.ArgumentParser()
    add_args(parser)
    config = Config.load(parser)
    check_config(config)
    logger.info(Config.toString(config))

    # ---- Load Keypair ----
    mnemonic = Keypair.generate_mnemonic()
    keypair = Keypair.create_from_mnemonic(mnemonic)
   
    # ---- Build Session ----
    session = bittensor.init(config, keypair)

    # ---- Start Neuron ----
    with session:
        main(config, session)

