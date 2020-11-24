"""BERT Next Sentence Prediction Neuron.

This file demonstrates training the BERT neuron with next sentence prediction.

Example:
        $ python examples/bert/main.py

"""
import bittensor
from bittensor.config import Config
from bittensor import BTSession
from bittensor.neuron import Neuron
from bittensor.synapses.bert import BertSynapseConfig, BertMLMSynapse, mlm_batch

from datasets import load_dataset
from loguru import logger
import random
import torch
import transformers
from transformers import DataCollatorForLanguageModeling

class Neuron (Neuron):
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
        parser = BertMLMSynapse.add_args(parser)
        return parser

    @staticmethod   
    def check_config(config: Munch) -> Munch:
        assert config.neuron.momentum > 0 and config.neuron.momentum < 1, "momentum must be a value between 0 and 1"
        assert config.neuron.batch_size > 0, "batch_size must a positive value"
        assert config.neuron.epoch_size > 0, "epoch_size must a positive value"
        assert config.neuron.learning_rate > 0, "learning_rate must be a positive value."
        Config.validate_path_create('neuron.datapath', config.neuron.datapath)
        config = BertMLMSynapse.check_config(config)
        return config

    def start(self, session: BTSession): 
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build Synapse
        model = BertMLMSynapse(self.config, session)
        model.to(device)
        session.serve( model )

        # Dataset: 74 million sentences pulled from books.
        # The collator accepts a list [ dict{'input_ids, ...; } ] where the internal dict 
        # is produced by the tokenizer.
        dataset = load_dataset('bookcorpus')
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=bittensor.__tokenizer__, mlm=True, mlm_probability=0.15
        )

        # Optimizer.
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.neuron.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        def train(dataset, model, epoch):
            model.train()  # Turn on the train mode.
            optimizer.zero_grad() # Zero out lingering gradients.

            step = 0
            while step < self.config.neuron.epoch_size:
                # Next batch.
                inputs, labels = mlm_batch(dataset['train'], self.config.neuron.batch_size, bittensor.__tokenizer__, data_collator)
                
                # Compute full pass and get loss with a network query.
                output = model( inputs.to(device), labels.to(device), remote = True)
                
                output.loss.backward()
                optimizer.step()
                scheduler.step()

                step += 1
                logger.info('Train Step: {} [{}/{} ({:.1f}%)]\t Remote Loss: {:.6f}\t Local Loss: {:.6f}\t Distilation Loss: {:.6f}'.format(
                    epoch, step, self.config.neuron.epoch_size, float(step * 100)/float(self.config.neuron.epoch_size), output.remote_target_loss.item(), output.local_target_loss.item(), output.distillation_loss.item()))
        
        epoch = 0
        while True:
            train(dataset, model, epoch)
            epoch += 1