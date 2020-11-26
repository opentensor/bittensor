"""GPT2 Language Modelling 

This file demonstrates training the GPT2 neuron with language modelling.

Example:
        $ python examples/gpt/main.py

"""
import bittensor
from bittensor.session import BTSession
from bittensor.config import Config
from bittensor.neuron import NeuronBase
from bittensor.synapses.gpt2 import GPT2LMSynapse, nextbatch

import argparse
from munch import Munch
from datasets import load_dataset
from loguru import logger
import torch
import replicate
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
        parser = GPT2LMSynapse.add_args(parser)
        return parser

    @staticmethod   
    def check_config(config: Munch) -> Munch:
        assert config.neuron.momentum > 0 and config.neuron.momentum < 1, "momentum must be a value between 0 and 1"
        assert config.neuron.batch_size_train > 0, "batch_size_train must a positive value"
        assert config.neuron.batch_size_test > 0, "batch_size_test must a positive value"
        assert config.neuron.epoch_size > 0, "epoch_size must a positive value"
        assert config.neuron.learning_rate > 0, "learning_rate must be a positive value."
        Config.validate_path_create('neuron.datapath', config.neuron.datapath)
        config = GPT2LMSynapse.check_config(config)
        return config

    def start(self, session: BTSession): 

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build Synapse
        model = GPT2LMSynapse(self.config, session)

        try:
            if self.config.session.checkout_experiment:
                experiment = replicate.experiments.get(self.config.session.checkout_experiment)
                # This point can be changed by user. 
                # experiment.latest() returns the latest model checkpointed. 
                # experiment.best() returns the best performing model checkpointed.
                latest_experiment = experiment.latest()
                logger.info("Checking out experiment {} to {}".format(
                    self.config.session.checkout_experiment, 
                    self.config.neuron.datapath + self.config.neuron.neuron_name))
                
                model_file = latest_experiment.open(self.config.neuron.datapath + self.config.neuron.neuron_name + "/model.torch")
                checkpt = torch.load(model_file)
                model.load_state_dict(checkpt['model'])
        except Exception as e:
            logger.warning("Something happened checking out the model. {}".format(e))
            logger.info("Using new model")

        model.to(device)
        session.serve( model )

        # Dataset: 74 million sentences pulled from books.
        dataset = load_dataset('bookcorpus')['train']
    
        # Optimizer.
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.neuron.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        def train(dataset, model, epoch):
            model.train()  # Turn on the train mode.
            optimizer.zero_grad() # Zero out lingering gradients.

            step = 0
            best_loss = math.inf
            while step < self.config.neuron.epoch_size:
                # Next batch.
                inputs = nextbatch(dataset, self.config.neuron.batch_size_train, bittensor.__tokenizer__)
                
                # Compute full pass and get loss with a network query.
                output = model(inputs.to(device), training = True, remote = True)

                output.loss.backward()
                optimizer.step()
                scheduler.step()
                
                step += 1
                logger.info('Train Step: {} [{}/{} ({:.1f}%)]\t Remote Loss: {:.6f}\t Local Loss: {:.6f}\t Distilation Loss: {:.6f}'.format(
                    epoch, step, self.config.neuron.epoch_size, float(step * 100)/float(self.config.neuron.epoch_size), output.loss.item(), output.remote_target_loss.item(), output.distillation_loss.item()))


            # After each epoch, checkpoint the losses and re-serve the network.
            if output.loss.item() < best_loss:
                best_loss = output.loss
                logger.info( 'Saving/Serving model: epoch: {}, loss: {}, path: {}/{}/model.torch', epoch, output.loss, self.config.neuron.datapath, self.config.neuron.neuron_name)
                torch.save( {'epoch': epoch, 'model': model.state_dict(), 'loss': output.loss},"{}/{}/model.torch".format(self.config.neuron.datapath , self.config.neuron.neuron_name))
                
                # Save experiment metrics
                session.checkpoint_experiment(epoch, loss=best_loss, remote_target_loss=output.remote_target_loss.item(), distillation_loss=output.distillation_loss.item())
                session.serve( model.deepcopy() )

        epoch = 0
        while True:
            train(dataset, model, epoch)
            epoch += 1            