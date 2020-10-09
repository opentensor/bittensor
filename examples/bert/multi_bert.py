"""BERT Next Sentence Prediction Synapse

This file demonstrates a bittensor.Synapse trained for Next Sentence Prediction.

Example:
        $ python examples/bert/main.py

"""

from bittensor import bittensor_pb2
from bittensor.examples.bert.model import BertNSPSynapse
import bittensor

import argparse
from datasets import load_dataset, list_metrics, load_metric
from loguru import logger
import os, sys
import math
import random
import time
import transformers
import torch

def nsp_batch(data, batch_size):
    """ Returns a random batch from text dataset with 50 percent NSP likelihood.

        Args:
            data: (List[dict{'text': str}]): Dataset of text inputs.
            batch_size: size of batch to create.
        
        Returns:
            batch_inputs List[str]: List of sentences.
            batch_next List[str]: List of (potential) next sentences 
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
    return batch_inputs, batch_next, torch.tensor(batch_labels, dtype=torch.long)
            
def main(hparams):
    # Args
    learning_rate = 0.01 
    n_synapses = 2
    batch_size = 500
    epoch_size = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset: 74 million sentence pulled from books.
    dataset = load_dataset('bookcorpus')['train']

    # Build Synapses
    model_config = transformers.modeling_bert.BertConfig(hidden_size=bittensor.__network_dim__, num_hidden_layers=2, num_attention_heads=2, intermediate_size=512, is_decoder=False)
    models = []
    for _ in range(n_synapses):
        model = BertNSPSynapse(model_config)
        model.to(device)
        model.train()
        models.append(model)

    # Setup Bittensor.
    # Create background objects.
    # Connect the metagraph.
    # Start the axon server.
    config = bittensor.Config( hparams )
    bittensor.init( config )
    bittensor.start()
    for model in models: 
        bittensor.serve( model )
  
    # Optimizer.
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    def step(model):
        # Single training step.
        optimizer.zero_grad()
        sentences, next_sentences, next_sentence_labels = nsp_batch(dataset, batch_size)
        output = model(sentences, next_sentences, next_sentence_labels, query=True)
        loss = output['loss']
        loss.backward()
        optimizer.step()
        return loss.item()

    def train(epoch):
        for i in range(epoch_size):
            losses = [step(model) for model in models]
            logger.info('Train Step: {}: {}', i, losses) 
            scheduler.step()
        
    epoch = 0
    try:
        while True:
            train(epoch)
            epoch += 1
    except Exception as e:
        logger.exception(e)
        bittensor.stop()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    hparams = bittensor.Config.add_args(parser)
    hparams = parser.parse_args()
    main(hparams)