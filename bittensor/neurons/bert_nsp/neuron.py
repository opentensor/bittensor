"""BERT Next Sentence Prediction Neuron.

This file demonstrates training the BERT neuron with next sentence prediction.

Example:
        $ python examples/bert/main.py

"""
import bittensor
import bittensor
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

class Neuron (Neuron):
    def __init__(self, config, session: BTSession):
        self.config = config
        self.session = session

    def stop(self):
        pass

    def start(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build Synapse
        model_config = transformers.modeling_bert.BertConfig(vocab_size=bittensor.__vocab_size__, hidden_size=bittensor.__network_dim__, num_hidden_layers=2, num_attention_heads=2, intermediate_size=512, is_decoder=False)
        model = BertNSPSynapse(model_config, self.session)
        model.to(device)
        self.session.serve( model )

        # Dataset: 74 million sentences pulled from books.
        dataset = load_dataset('bookcorpus')
    
        # Optimizer.
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.training.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        def train(dataset, model, epoch):
            model.train()  # Turn on the train mode.
            optimizer.zero_grad() # Zero out lingering gradients.

            step = 0
            while step < epoch_size:
                # Next batch.
                inputs, labels = nsp_batch(dataset['train'], self.config.training.batch_size, bittensor.__tokenizer__)
                
                # Compute full pass and get loss with a network query.
                output = model (inputs = inputs['input_ids'], 
                                attention_mask = inputs ['attention_mask'],
                                labels = labels,
                                query=True )
                
                loss = output['loss']
                loss.backward()
                optimizer.step()
                scheduler.step()

                step += 1
                logger.info('Train Step: {} [{}/{} ({:.1f}%)]\t Network Loss: {:.6f}\t Local Loss: {:.6f}\t Distilation Loss: {:.6f}'.format(
                    epoch, step, self.config.training.epoch_size, float(step * 100)/float(self.config.training.epoch_size), output['network_target_loss'].item(), output['local_target_loss'].item(), output['distillation_loss'].item()))
        
        epoch = 0
        while True:
            train(dataset, model, epoch)
            epoch += 1
   
            