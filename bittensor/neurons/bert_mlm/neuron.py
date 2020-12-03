"""BERT Next Sentence Prediction Neuron.

This file demonstrates training the BERT neuron with next sentence prediction.

Example:
        $ python examples/bert/main.py

"""
import bittensor
import argparse
from bittensor.config import Config
from bittensor import BTSession
from bittensor.neuron import NeuronBase
from bittensor.synapses.bert import BertMLMSynapse, mlm_batch

import numpy as np
from termcolor import colored
from datasets import load_dataset
from loguru import logger
import torch
import torch.nn.functional as F
from transformers import DataCollatorForLanguageModeling
import replicate
from munch import Munch
import math

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
        parser.add_argument('--neuron.batch_size_train', default=20, type=int, 
                            help='Training batch size.')
        parser.add_argument('--neuron.batch_size_test', default=20, type=int, 
                            help='Testing batch size.')
        parser.add_argument('--neuron.epoch_size', default=50, type=int, 
                            help='Testing batch size.')
        parser.add_argument('--neuron.log_interval', default=10, type=int, 
                            help='Batches until neuron prints log statements.')
        parser = BertMLMSynapse.add_args(parser)
        return parser

    @staticmethod   
    def check_config(config: Munch) -> Munch:
        assert config.neuron.momentum > 0 and config.neuron.momentum < 1, "momentum must be a value between 0 and 1"
        assert config.neuron.batch_size_train > 0, "batch_size must a positive value"
        assert config.neuron.batch_size_test > 0, "batch_size must a positive value"
        assert config.neuron.epoch_size > 0, "epoch_size must a positive value"
        assert config.neuron.learning_rate > 0, "learning_rate must be a positive value."
        Config.validate_path_create('neuron.datapath', config.neuron.datapath)
        config = BertMLMSynapse.check_config(config)
        return config

    def start(self, session: BTSession): 
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build Synapse
        model = BertMLMSynapse(self.config, session)

        try:
            if self.config.session.checkout_experiment:
                model = session.replicate_util.checkout_experiment(model, best=False)
        except Exception as e:
            logger.warning("Something happened checking out the model. {}".format(e))
            logger.info("Using new model")

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

            step = 0
            best_loss = math.inf
            history = []
            while step < self.config.neuron.epoch_size:
                # Zero grads.
                optimizer.zero_grad() # Zero out lingering gradients.

                # Emit and sync.
                if (session.metagraph.block() - session.metagraph.state.block) > 15:
                    session.metagraph.emit()
                    session.metagraph.sync()

                # Next batch.
                inputs, labels = mlm_batch(dataset['train'], self.config.neuron.batch_size_train, bittensor.__tokenizer__, data_collator)
                
                # Forward pass.
                output = model( inputs.to(device), labels.to(device), remote = True)
                history.append(output)
                
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

                # Log history.
                step += 1
                if (step + 1) % self.config.neuron.log_interval == 0:
                    log_training_output_history(
                        session = session, 
                        epoch = epoch, 
                        batch_idx = step, 
                        batch_size = self.config.neuron.batch_size_train, 
                        total_examples = self.config.neuron.batch_size_train * self.config.neuron.epoch_size, 
                        history = history)
                    history = [] # Clear history.


            # After each epoch, checkpoint the losses and re-serve the network.
            if output.loss.item() < best_loss:
                best_loss = output.loss.item()
                logger.info( 'Saving/Serving model: epoch: {}, loss: {}, path: {}/{}/model.torch', epoch, output.loss, self.config.neuron.datapath, self.config.neuron.neuron_name)
                torch.save( {'epoch': epoch, 'model': model.state_dict(), 'loss': output.loss},"{}/{}/model.torch".format(self.config.neuron.datapath , self.config.neuron.neuron_name))
                
                # Save experiment metrics
                session.replicate_util.checkpoint_experiment(epoch, loss=best_loss, remote_target_loss=output.remote_target_loss.item(), distillation_loss=output.distillation_loss.item())
                session.serve( model.deepcopy() )
                
        epoch = 0
        while True:
            train(dataset, model, epoch)
            epoch += 1