"""BERT Next Sentence Prediction Neuron.

This file demonstrates training the BERT neuron with next sentence prediction.

Example:
        $ python examples/bert/main.py

"""
import bittensor
from bittensor.config import Config
from bittensor import BTSession
from bittensor.neuron import Neuron
from bittensor.synapse import Synapse
from bittensor.synapses.bert import BertNSPSynapse

from datasets import load_dataset, list_metrics, load_metric
from loguru import logger
import os, sys
import math
import random
import time
import transformers
import torch

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

class Neuron (NeuronBase):
    def __init__(self, config):
        self.config = config


    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:    
        parser.add_argument('--neuron.datapath', default='data/', type=str, 
                            help='Path to load and save data.')
        parser.add_argument('--neuron.learning_rate', default=0.01, type=float, 
                            help='Training initial learning rate.')
        parser.add_argument('--neuron.momentum', default=0.98, type=float, 
                            help='Training initial momentum for SGD.')
        parser.add_argument('--neuron.batch_size', default=20, type=int, 
                            help='Training batch size.')
        parser.add_argument('--neuron.epoch_size', default=50, type=int, 
                            help='Testing batch size.')
        parser = BertNSPSynapse.add_args(parser)
        return parser

    @staticmethod   
    def check_config(config: Munch) -> Munch:
        assert config.neuron.momentum > 0 and config.neuron.momentum < 1, "momentum must be a value between 0 and 1"
        assert config.neuron.batch_size > 0, "batch_size must a positive value"
        assert config.neuron.epoch_size > 0, "epoch_size must a positive value"
        assert config.neuron.learning_rate > 0, "learning_rate must be a positive value."
        Config.validate_path_create('neuron.datapath', config.neuron.datapath)
        config = BertNSPSynapse.check_config(config)
        return config

    def start(self, session: BTSession): 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build Synapse
        model = BertNSPSynapse(self.config, session)
        model.to(device)
        session.serve( model )

        # Dataset: 74 million sentences pulled from books.
        dataset = load_dataset('bookcorpus')
    
        # Optimizer.
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.neuron.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        def train(dataset, model, epoch):
            model.train()  # Turn on the train mode.
            optimizer.zero_grad() # Zero out lingering gradients.

            step = 0
            while step < self.config.neuron.epoch_size:
                # Next batch.
                inputs, targets = nsp_batch(dataset['train'], self.config.neuron.batch_size, bittensor.__tokenizer__)
                
                # Compute full pass and get loss with a network query.
                output = model (inputs = inputs['input_ids'], 
                                attention_mask = inputs ['attention_mask'],
                                targets = targets,
                                remote = True )
                
                loss = output['loss']
                loss.backward()
                optimizer.step()
                scheduler.step()

                step += 1
                logger.info('Train Step: {} [{}/{} ({:.1f}%)]\t Network Loss: {:.6f}\t Local Loss: {:.6f}\t Distilation Loss: {:.6f}'.format(
                    epoch, step, self.config.neuron.epoch_size, float(step * 100)/float(self.config.neuron.epoch_size), output.network_target_loss.item(), output.local_target_loss.item(), output.distillation_loss.item()))
        
        epoch = 0
        while True:
            train(dataset, model, epoch)
            epoch += 1
   
            