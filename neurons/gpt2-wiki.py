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
from bittensor.subtensor import Keypair
from bittensor.utils.logging import log_all
from bittensor.config import Config
from bittensor.synapses.gpt2 import GPT2LMSynapse, nextbatch

def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:    
    parser.add_argument('--neuron.datapath', default='data', type=str,help='Path to load and save data.')
    parser.add_argument('--neuron.learning_rate', default=0.01, type=float, help='Training initial learning rate.')
    parser.add_argument('--neuron.momentum', default=0.98, type=float, help='Training initial momentum for SGD.')
    parser.add_argument('--neuron.batch_size_train', default=20, type=int, help='Training batch size.')
    parser.add_argument('--neuron.sync_interval', default=100, type=int, help='Batches before we sync with chain and emit new weights.')
    parser.add_argument('--neuron.log_interval', default=10, type=int, help='Batches before we log session info.')
    parser.add_argument('--neuron.accumulation_interval', default=1, type=int, help='Batches before we apply acummulated gradients.')
    parser.add_argument('--neuron.apply_remote_gradients', default=False, type=bool, help='If true, neuron applies gradients which accumulate from remotes calls.')
    parser.add_argument('--neuron.name', default='gpt-wiki', type=str, help='Trials for this neuron go in neuron.datapath / neuron.name')
    parser.add_argument('--neuron.trial_id', default=str(time.time()).split('.')[0], type=str, help='Saved models go in neuron.datapath / neuron.name / neuron.trial_id')
    parser = GPT2LMSynapse.add_args(parser)
    return parser

def check_config(config: Munch) -> Munch:
    assert config.neuron.momentum > 0 and config.neuron.momentum < 1, "momentum must be a value between 0 and 1"
    assert config.neuron.batch_size_train > 0, "batch_size_train must a positive value"
    assert config.neuron.learning_rate > 0, "learning_rate must be a positive value."
    try:
        trial_path = str(config.neuron.datapath + '/' + config.neuron.name + '/' + config.neuron.trial_id)
        pathlib.Path(trial_path).mkdir(parents=True, exist_ok=True)
    except Exception as _:
        logger.error("No permission to trial path: {}", trial_path)
        raise ValueError
    config = GPT2LMSynapse.check_config(config)
    return config
    
# Neuron main.
def main(config, session):

    # ---- Build Model ----
    model = GPT2LMSynapse(config, session)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    session.serve( model )

    # ---- Dataset ----
    # 74 million sentences pulled from books.
    dataset = load_dataset('bookcorpus')['train']

    # ---- Optimizer ----
    optimizer = torch.optim.SGD(model.parameters(), lr = config.neuron.learning_rate, momentum=config.neuron.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
    # ---- Init training state ----
    session.metagraph.sync() # Sync with the chain.
    row_weights = session.metagraph.W[0, :] # Weight to others (zeros initially)
    col_weights = session.metagraph.W[:, 0] # Other to me.
    priority_map = dict(zip(session.metagraph.public_keys, col_weights.tolist()))
    session.axon.set_priority( priority_map )

    # ---- Train forever ----
    model.train()
    step = -1; history = []; best_loss = math.inf; 
    while True:
        step += 1
        # ---- Next Batch ----
        inputs = nextbatch(dataset, config.neuron.batch_size_train, bittensor.__tokenizer__)

        # ---- Forward Pass ----
        output = model(inputs.to(model.device), training = True, remote = True)
        history.append(output)

        # ---- Update Weights ----
        batch_weights = torch.mean(output.weights, axis = 0)
        row_weights = (1 - 0.05) * row_weights + 0.05 * batch_weights # Moving Avg weights.
        row_weights = F.normalize(row_weights, p = 1, dim = 0)

        # ---- Accumulate Local Gradients ----
        loss = output.loss / config.neuron.accumulation_interval # Need to average accross accumulation steps.
        loss.backward() # Accumulates gradients on model via sum.

        # ---- Accumulate Remote Gradients  ----
        if config.neuron.apply_remote_gradients:
            # TODO (const): batch normalization over the gradients for consistency.
            n_grads = session.axon.gradients.qsize
            for _, (_, input_x, grads_dy, modality) in list(session.axon.gradients.queue):
                grads_dy = grads_dy / (config.neuron.accumulation_interval * n_grads)
                model.backward(input_x, grads_dy, modality)

        # ---- Apply Gradients ----
        logger.info('Step: {} \t Remote Loss: {:.6f}\t Local Loss: {:.6f}\t Distilation Loss: {:.6f}'.format(step, output.loss.item(), output.remote_target_loss.item(), output.distillation_loss.item()))
        if (step+1) % config.neuron.accumulation_interval == 0:
            optimizer.step() # Apply accumulated gradients.
            optimizer.zero_grad() # Zero grads for next accummulation 
            scheduler.step() # Update learning rate etc.
            session.serve( model ) # Serve the newest model.

        # ---- Sync State ----
        if (step+1) % config.neuron.sync_interval == 0:
            # ---- Emit weights and sync from chain ----
            session.metagraph.emit( row_weights ) # Set weights on chain.
            session.metagraph.sync() # Sync with the chain.
            
            # ---- Get row and col Weights.
            row_weights = session.metagraph.W[0, :] # Weight to others.
            col_weights = session.metagraph.W[:, 0] # Other to me.

            # ---- Update Axon Priority ----
            priority_map = dict(zip(session.metagraph.public_keys, col_weights.tolist()))
            session.axon.set_priority( priority_map )

        # ---- Session Logs ----
        if (step+1) % config.neuron.log_interval == 0:
            log_all(session, history); history = [] # Log batch history.
        
        # --- Save Model ----
        if output.loss.item() < best_loss:
            best_loss = output.loss
            logger.info( 'Saving model: epoch: {}, loss: {}, path: {}/{}/{}/model.torch', step, output.loss, config.neuron.datapath, config.neuron.name, config.neuron.trial_id)
            torch.save( {'epoch': step, 'model': model.state_dict(), 'loss': output.loss},"{}/{}/{}/model.torch".format(config.neuron.datapath , config.neuron.name, config.neuron.trial_id))
            
    
if __name__ == "__main__":
    # ---- Load Bittensor config ----
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    config = Config.load(parser)
    config = check_config(config)
    logger.info(Config.toString(config))

    # ---- Load Keypair ----
    mnemonic = Keypair.generate_mnemonic()
    keypair = Keypair.create_from_mnemonic(mnemonic)
   
    # ---- Build Session ----
    session = bittensor.init(config, keypair)

    # ---- Start Neuron ----
    with session:
        main(config, session)

