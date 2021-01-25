import argparse
import bittensor
import torch
import os
import sys
import time

from bittensor.synapse import Synapse
from munch import Munch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

class InvalidModelException(Exception):
    pass

class ModelInformationNotFoundException(Exception):
    pass

class Miner:
    def __init__(self, model_type: Synapse, config: Munch = None):
        """ 
        Miner class that encapsulates model handling and blockchain interactions
        """

        # Model type is the model's class. This is required to pass along any configurations during setup and for instantiating the model.
        self.model_type = model_type
        
        # Build config
        if not config:
            config = self.build_config()
        self.config = config

        # Instantiate model
        self.model = self.model_type( self.config )

        # ---- Model device ----
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # send model to device
        self.model.to(self.device)

        # ---- Neuron ----
        self.neuron = bittensor.neuron.Neuron(self.config)

        self.tensorboard = SummaryWriter(log_dir = self.config.miner.full_path)
        if self.config.miner.record_log:
            logger.add(self.config.miner.full_path + "/{}_{}.log".format(self.config.miner.name, self.config.miner.trial_uid),format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")
        
        # Initialize row weights
        self.row = None
    
    def load_model(self):
        """ Loads a model saved by save_model() and returns it. 

        Returns:
            model (:obj:`torch.nn.Module`) : Model that was saved earlier, loaded back up using the state dict and optimizer. 
            optimizer (:obj:`torch.optim`) : Model optimizer that was saved with the model.
        """
        model = self.model_type( self.config )
        optimizer = self.optimizer_class(model.parameters(), lr = self.config.miner.learning_rate, momentum=self.config.miner.momentum)
        
        try:
            checkpoint = torch.load("{}/model.torch".format(self.config.miner.full_path))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']

            logger.info( 'Reloaded model: epoch: {}, loss: {}, path: {}/model.torch'.format(epoch, loss, self.config.miner.full_path))
        except Exception as e:
            logger.warning ( 'Exception {}. Could not find model in path: {}/model.torch', e, self.config.miner.full_path )

        return model, optimizer

    def save_model(self, model_info):
        """Saves the model locally. 

        Args:
            model_info (:obj:`dict`, `required`): Dictionary containing the epoch we are saving at, the loss, and the PyTorch model object.

        Raises:
            :obj:`ModelInformationNotFoundException`: Raised whenever the loss, epoch, or PyTorch model object is missing from the input dictionary.
        """
        try:
            if 'epoch' not in model_info.keys():
                raise ModelInformationNotFoundException("Missing 'epoch' in torch save dict")

            if 'loss' not in model_info.keys():
                raise ModelInformationNotFoundException("Missing 'loss' in torch save dict")
            
            if 'model_state_dict' not in model_info.keys():
                raise ModelInformationNotFoundException("Missing 'model' in torch save dict")

            if 'optimizer_state_dict' not in model_info.keys():
                raise ModelInformationNotFoundException("Missing 'optimizer' in torch save dict")
            
            logger.info( 'Saving/Serving model: epoch: {}, loss: {}, path: {}/model.torch'.format(model_info['epoch'], model_info['loss'], self.config.miner.full_path))
            torch.save(model_info,"{}/model.torch".format(self.config.miner.full_path))

        except ModelInformationNotFoundException as e:
            logger.error("Encountered exception trying to save model: {}", e)

        
    def display_epoch(self):
        print(self.neuron.axon.__full_str__())
        print(self.neuron.dendrite.__full_str__())
        print(self.neuron.metagraph)

    def update_tensorboard(self, remote_target_loss = None, local_target_loss = None, distillation_loss = None):
        self.neuron.dendrite.__to_tensorboard__(self.tensorboard, self.global_step)
        self.neuron.metagraph.__to_tensorboard__(self.tensorboard, self.global_step)
        self.neuron.axon.__to_tensorboard__(self.tensorboard, self.global_step)

        if remote_target_loss:
            self.tensorboard.add_scalar('Rloss', remote_target_loss, self.global_step)
        
        if local_target_loss:
            self.tensorboard.add_scalar('Lloss', local_target_loss, self.global_step)
        
        if distillation_loss:
            self.tensorboard.add_scalar('Dloss', distillation_loss, self.global_step)

    ##########################################################################################################################
    #                                       Blockchain updates
    ##########################################################################################################################
    
    def update_row_weights(self):
        self.row = self.neuron.metagraph.row.to(self.device)
    
    def train_row_weights(self, router_weights: torch.FloatTensor):
        # Average over batch.
        batch_weights = torch.mean(router_weights, axis = 0).to(self.model.device) 
        # Moving avg update.
        self.row = 0.97 * self.row + 0.03 * batch_weights 
        # Ensure normalization.
        self.row = F.normalize(self.row, p = 1, dim = 0)
    
    def set_metagraph_weights_and_sync(self):
        # Sets my row-weights on the chain.
        self.neuron.metagraph.set_weights(self.row, wait_for_inclusion = True) 
        # Pulls the latest metagraph state
        self.neuron.metagraph.sync()
        # Update the row weights
        self.update_row_weights()

    ##########################################################################################################################
    #                                       Model utility functions
    ##########################################################################################################################
        
    def add_args(self, parser: argparse.ArgumentParser):    
        parser.add_argument('--miner.learning_rate', default=0.01, type=float, help='Training initial learning rate.')
        parser.add_argument('--miner.momentum', default=0.9, type=float, help='Training initial momentum for SGD.')
        parser.add_argument('--miner.n_epochs', default=int(sys.maxsize), type=int, help='Number of training epochs.')
        parser.add_argument('--miner.epoch_length', default=int(sys.maxsize), type=int, help='Iterations of training per epoch (or dataset EOF)')
        parser.add_argument('--miner.batch_size_train', default=64, type=int, help='Training batch size.')
        parser.add_argument('--miner.batch_size_test', default=64, type=int, help='Testing batch size.')
        parser.add_argument('--miner.log_interval', default=150, type=int, help='Batches until session prints log statements.')
        parser.add_argument('--miner.sync_interval', default=150, type=int, help='Batches before we we sync with chain and emit new weights.')
        parser.add_argument('--miner.root_dir', default='~/.bittensor/sessions/', type=str,  help='Root path to load and save data associated with each session')
        parser.add_argument('--miner.name', default='cifar', type=str, help='Trials for this session go in miner.root / miner.name')
        parser.add_argument('--miner.trial_uid', default=str(time.time()).split('.')[0], type=str, help='Saved models go in miner.root_dir / miner.name / miner.trial_uid')
        parser.add_argument('--miner.record_log', default=True, help='Record all logs when running this session')
        parser.add_argument('--miner.config_file', type=str, help='config file to run this neuron, if not using cmd line arguments.')
        bittensor.neuron.Neuron.add_args(parser)
        self.model_type.add_args(parser)

    def build_config(self) -> Munch:
        parser = argparse.ArgumentParser(); 
        self.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        Miner.check_config(config)
        return config

    @staticmethod
    def check_config(config: Munch):
        assert config.miner.log_interval > 0, "log_interval dimension must be positive"
        assert config.miner.momentum > 0 and config.miner.momentum < 1, "momentum must be a value between 0 and 1"
        assert config.miner.batch_size_train > 0, "batch_size_train must be a positive value"
        assert config.miner.batch_size_test > 0, "batch_size_test must be a positive value"
        assert config.miner.learning_rate > 0, "learning rate must be be a positive value."
        full_path = '{}/{}/{}/'.format(config.miner.root_dir, config.miner.name, config.miner.trial_uid)
        config.miner.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.miner.full_path):
            os.makedirs(config.miner.full_path)
        bittensor.neuron.Neuron.check_config(config)