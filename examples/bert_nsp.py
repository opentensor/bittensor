"""BERT Next Sentence Prediction Neuron.

This file demonstrates training the BERT neuron with next sentence prediction.

Example:
        $ python neurons/bert_nsp.py

"""
import bittensor
import argparse
from bittensor.config import Config
from bittensor import Session
from bittensor.subtensor import Keypair
from bittensor.synapses.bert import BertNSPSynapse

import numpy as np
from termcolor import colored
from loguru import logger
from datasets import load_dataset
import replicate
import random
import torch
import torch.nn.functional as F
from munch import Munch
import math

def nsp_batch(data, batch_size, tokenizer):
    """ Returns a random batch from text dataset with 50 percent NSP.

        Args:
            data: (List[dict{'text': str}]): Dataset of text inputs.
            batch_size: size of batch to create.
        
        Returns:
            input_ids List[str]: List of sentences.
            batch_labels torch.Tensor(batch_size): 1 if random next sentence, otherwise 0.
    """

    batch_inputs = []
    batch_next = []
    batch_labels = []
    for _ in range(batch_size):
        if random.random() > 0.5:
            pos = random.randint(0, len(data))
            batch_inputs.append(data[pos]['text'])
            batch_next.append(data[pos + 1]['text'])
            batch_labels.append(0)
        else:
            while True:
                pos_1 = random.randint(0, len(data))
                pos_2 = random.randint(0, len(data))
                batch_inputs.append(data[pos_1]['text'])
                batch_next.append(data[pos_2]['text'])
                batch_labels.append(1)
                if (pos_1 != pos_2) and (pos_1 != pos_2 - 1):
                    break

    tokenized = tokenizer(batch_inputs, text_pair = batch_next, return_tensors='pt', padding=True)
    return tokenized, torch.tensor(batch_labels, dtype=torch.long)

def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:    
    parser.add_argument('--neuron.datapath', default='data/', type=str, 
                        help='Path to load and save data.')
    parser.add_argument('--neuron.learning_rate', default=0.01, type=float, 
                        help='Training initial learning rate.')
    parser.add_argument('--neuron.momentum', default=0.98, type=float, 
                        help='Training initial momentum for SGD.')
    parser.add_argument('--neuron.batch_size_train', default=20, type=int, 
                        help='Training batch size.')
    parser.add_argument('--neuron.batch_size_test', default=20, type=int, 
                        help='Testing batch size.')
    parser = BertNSPSynapse.add_args(parser)
    return parser

def check_config(config: Munch) -> Munch:
    assert config.neuron.momentum > 0 and config.neuron.momentum < 1, "momentum must be a value between 0 and 1"
    assert config.neuron.batch_size_train > 0, "batch_size_train must a positive value"
    assert config.neuron.batch_size_test > 0, "batch_size_test must a positive value"
    assert config.neuron.learning_rate > 0, "learning_rate must be a positive value."
    Config.validate_path_create('neuron.datapath', config.neuron.datapath)
    config = BertNSPSynapse.check_config(config)
    return config

def train(model, config, session, optimizer, scheduler, dataset):
    step = 0
    best_loss = math.inf
    model.train()  # Turn on the train mode.
    while True:
        optimizer.zero_grad() # Clear gradients.

        # Emit and sync.
        if (session.metagraph.block() - session.metagraph.state.block) > 15:
            session.metagraph.emit()
            session.metagraph.sync()

         # Next batch.
        inputs, targets = nsp_batch(dataset['train'], config.neuron.batch_size_train, bittensor.__tokenizer__)

        # Forward pass.
        output = model (inputs = inputs['input_ids'].to(model.device), 
                        targets = targets.to(model.device),
                        remote = True )

        # Backprop.
        output.loss.backward()
        optimizer.step()
        scheduler.step()

        # Update weights.
        state_weights = session.metagraph.state.weights
        learned_weights = F.softmax(torch.mean(output.weights, axis=0))
        state_weights = (1 - 0.05) * state_weights + 0.05 * learned_weights
        norm_state_weights = F.softmax(state_weights)
        session.metagraph.state.set_weights( norm_state_weights )
        
        step += 1
        logger.info('Step: {} \t Remote Loss: {:.6f}\t Local Loss: {:.6f}\t Distilation Loss: {:.6f}'.format(
            step, output.loss.item(), output.remote_target_loss.item(), output.distillation_loss.item()))

    # After each epoch, checkpoint the losses and re-serve the network.
    if output.loss.item() < best_loss:
        best_loss = output.loss
        logger.info( 'Saving/Serving model: epoch: {}, loss: {}, path: {}/{}/model.torch', epoch, output.loss, config.neuron.datapath, config.neuron.neuron_name)
        torch.save( {'epoch': epoch, 'model': model.state_dict(), 'loss': output.loss},"{}/{}/model.torch".format(config.neuron.datapath , config.neuron.neuron_name))
        
        # Save experiment metrics
        session.replicate_util.checkpoint_experiment(epoch, loss=best_loss, remote_target_loss=output.remote_target_loss.item(), distillation_loss=output.distillation_loss.item())
        session.serve( model.deepcopy() )


def main(config, session):
    # Build Synapse
    model = BertNSPSynapse(config, session)
    if config.session.checkout_experiment:
        try:            
            model = session.replicate_util.checkout_experiment(model, best=False)
        except Exception as e:
            logger.warning("Something happened checking out the model. {}".format(e))
            logger.info("Using new model")

    # Set deivce and serve to the axon endpoint.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    session.serve( model )

    # Dataset: 74 million sentences pulled from books.
    dataset = load_dataset('bookcorpus')

    # Optimizer.
    optimizer = torch.optim.SGD(model.parameters(), lr = config.neuron.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # train forever.
    train(model, config, session, optimizer, scheduler, dataset)
    

if __name__ == "__main__":
    # 1. Load bittensor config.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    config = Config.load(parser)
    config = check_config(config)

    # 2. Load Keypair.
    mnemonic = Keypair.generate_mnemonic()
    keypair = Keypair.create_from_mnemonic(mnemonic)
   
    # 3. Load Session.
    session = bittensor.init(config, keypair)

    # 4. Start Neuron.
    with session:
        bittensor.run(main, config, session)
            