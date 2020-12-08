"""BERT Masked Language Modelling.

This file demonstrates training the BERT neuron with masked language modelling.

Example:
        $ python examples/bert_mlm.py

"""
import bittensor
import argparse
from bittensor.config import Config
from bittensor import Session
from bittensor.subtensor import Keypair
from bittensor.synapses.bert import BertMLMSynapse

import numpy as np
from termcolor import colored
from loguru import logger
from datasets import load_dataset
import replicate
import random
import torch
import torch.nn.functional as F
from transformers import DataCollatorForLanguageModeling
from munch import Munch
import math

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
    parser = BertMLMSynapse.add_args(parser)
    return parser

def check_config(config: Munch) -> Munch:
    assert config.neuron.momentum > 0 and config.neuron.momentum < 1, "momentum must be a value between 0 and 1"
    assert config.neuron.batch_size_train > 0, "batch_size_train must a positive value"
    assert config.neuron.batch_size_test > 0, "batch_size_test must a positive value"
    assert config.neuron.learning_rate > 0, "learning_rate must be a positive value."
    config = BertMLMSynapse.check_config(config)
    return config

def mlm_batch(data, batch_size, tokenizer, collator):
    """ Returns a random batch from text dataset with 50 percent NSP.

        Args:
            data: (List[dict{'text': str}]): Dataset of text inputs.
            batch_size: size of batch to create.
        
        Returns:
            tensor_batch torch.Tensor (batch_size, sequence_length): List of tokenized sentences.
            labels torch.Tensor (batch_size, sequence_length)
    """
    batch_text = []
    for _ in range(batch_size):
        batch_text.append(data[random.randint(0, len(data))]['text'])

    # Tokenizer returns a dict { 'input_ids': list[], 'attention': list[] }
    # but we need to convert to List [ dict ['input_ids': ..., 'attention': ... ]]
    # annoying hack...
    tokenized = tokenizer(batch_text)
    tokenized = [dict(zip(tokenized,t)) for t in zip(*tokenized.values())]

    # Produces the masked language model inputs aw dictionary dict {'inputs': tensor_batch, 'labels': tensor_batch}
    # which can be used with the Bert Language model. 
    collated_batch =  collator(tokenized)
    return collated_batch['input_ids'], collated_batch['labels']

def train(model, config, session, optimizer, scheduler, dataset, collator):
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
        inputs, labels = mlm_batch(dataset, config.neuron.batch_size_train, bittensor.__tokenizer__, collator)
                
        # Forward pass.
        output = model( inputs.to(model.device), labels.to(model.device), remote = True)

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
    model = BertMLMSynapse(config, session)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    session.serve( model )

    # Dataset: 74 million sentences pulled from books.
    # The collator accepts a list [ dict{'input_ids, ...; } ] where the internal dict 
    # is produced by the tokenizer.
    dataset = load_dataset('bookcorpus')['train']
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=bittensor.__tokenizer__, mlm=True, mlm_probability=0.15
    )

    # Optimizer.
    optimizer = torch.optim.SGD(model.parameters(), lr = config.neuron.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # train forever.
    train(model, config, session, optimizer, scheduler, dataset, data_collator)
    

if __name__ == "__main__":
    # 1. Load bittensor config.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    config = Config.load(parser)
    config = check_config(config)
    logger.info(Config.toString(config))

    # 2. Load Keypair.
    mnemonic = Keypair.generate_mnemonic()
    keypair = Keypair.create_from_mnemonic(mnemonic)
   
    # 3. Load Session.
    session = bittensor.init(config, keypair)

    # 4. Start Neuron.
    with session:
        main(config, session)
            