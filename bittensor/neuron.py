import argparse
import math
import torch
import time
import os

from tqdm import tqdm
from qqdm import qqdm, format_str
from munch import Munch
from loguru import logger
from termcolor import colored
from types import SimpleNamespace
from typing import Tuple, List, Optional

from torch.utils.tensorboard import SummaryWriter

import bittensor

class Neuron():

    def __init__( self, config: Munch = None ):
        r""" Initializes a new full Neuron object.
            
            Args:
                config (:obj:`Munch`, `optional`): 
                    neuron.Neuron.default_config()
        """
        # Config: Config items for all subobjects: wallet, metagraph, nucleus, axon, dendrite.
        # This object can be instantiated by calling Neuron.default_config()
        if config == None:
            config = Neuron.default_config()
        Neuron.check_config(config)
        self.config = config
        
        # --- Bittensor components ----
        # Wallet: Holds the hotkey keypair and coldkey pub which are user to sign messages 
        # and subscribe to the chain.
        self.wallet = bittensor.wallet.Wallet( self.config )
        if not self.wallet.has_coldkeypub:
            self.wallet.create_new_coldkey( n_words = 12, use_password = True )
        if not self.wallet.has_hotkey:
            self.wallet.create_new_hotkey( n_words = 12, use_password = False )
        
        # Subtensor: provides an interface to the subtensor chain given a wallet.
        self.subtensor = bittensor.subtensor.Subtensor( self.config )
        
        # Metagraph: Maintains a connection to the subtensor chain and hold chain state.
        self.metagraph = bittensor.metagraph.Metagraph(config = self.config, wallet = self.wallet, subtensor = self.subtensor)
        
        # Nucleus: Processes requests passed to this neuron on its axon endpoint.
        self.nucleus = bittensor.nucleus.Nucleus(config = self.config, wallet = self.wallet )
        
        # Axon: RPC server endpoint which serves your synapse. Responds to Forward and Backward requests.
        self.axon = bittensor.axon.Axon(config = self.config, wallet = self.wallet, nucleus = self.nucleus )
        
        # Dendrite: RPC client makes Forward and Backward requests to downstream peers.
        self.dendrite = bittensor.dendrite.Dendrite(config = self.config, wallet = self.wallet )

        # ---- Running state ----
        self.global_step = 0
        self.epoch = 0
        self.best_train_loss = math.inf

        # ---- Logging ----
        self.tensorboard = SummaryWriter(log_dir = self.config.neuron.full_path)
        if self.config.neuron.record_log == True:
            filepath = self.config.neuron.full_path + "/{}_{}.log".format(self.config.neuron.name, self.config.neuron.uid),
            logger.add (
                filepath,
                format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
                rotation="250 MB",
                retention="10 days"
            )

    @staticmethod   
    def default_config() -> Munch:
        # Parses and returns a config Munch for this object.
        parser = argparse.ArgumentParser(); 
        Neuron.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        return config

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        bittensor.wallet.Wallet.add_args( parser )
        bittensor.subtensor.Subtensor.add_args( parser )
        bittensor.metagraph.Metagraph.add_args( parser )
        bittensor.nucleus.Nucleus.add_args( parser )
        bittensor.axon.Axon.add_args( parser )
        bittensor.synapse.Synapse.add_args( parser )
        bittensor.dendrite.Dendrite.add_args( parser )
        parser.add_argument (
            '--neuron.root_dir',
            default='~/.bittensor/miners/',
            type=str,
            help='Root path to load and save data associated with each neuron'
        )
        parser.add_argument (
            '--neuron.name',
            default='gpt2-genesis',
            type=str,
            help='Trials for this neuron go in neuron.root / neuron.name'
        )
        parser.add_argument (
            '--neuron.uid',
            default=str(time.time()).split('.')[0],
            type=str,
            help='Saved models go in neuron.root_dir / neuron.name / neuron.uid'
        )
        parser.add_argument (
            '--neuron.record_log',
            default=False,
            type=bool,
            help='Record all logs when running this miner')
        parser.add_argument (
            '--debug', 
            dest='debug', 
            action='store_true', 
            help='''Turn on bittensor debugging information'''
        )
        parser.set_defaults ( 
            debug=False 
        )

    @staticmethod
    def check_config(config: Munch):
        if config.debug:  bittensor.__log_level__ = 'TRACE'; logger.debug('DEBUG is ON')
        else: logger.info('DEBUG is OFF') 
        full_path = '{}/{}/{}'.format(config.neuron.root_dir, config.neuron.name, config.neuron.uid)
        config.neuron.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.neuron.full_path):
            os.makedirs(config.neuron.full_path)

    @property
    def _model(self):
        if self.model == None:
            raise RuntimeError('Subclasses must set model class')

    def get_row_weights( self ) -> torch.FloatTensor:
        raise NotImplementedError()

    def training_forward( self ) -> SimpleNamespace:
        raise NotImplementedError()

    def next_training_batches( self ) -> List[dict]:
        raise NotImplementedError()
    
    def run( self ):

        # ---- Connect to chain ----
        logger.info("Connecting to network: {}", self.subtensor.config.subtensor.network)
        self.subtensor.connect()
        if not self.subtensor.is_connected():
            raise RuntimeError('Failed to connect subtensor')
        
        # ---- Subscribe to chain ----
        logger.info("Subscribing to chain")
        subscribe_success = self.subtensor.subscribe(
                wallet = self.wallet,
                ip = self.config.axon.external_ip, 
                port = self.config.axon.external_port,
                modality = bittensor.proto.Modality.TEXT,
                wait_for_finalization = True,
                timeout = 4 * bittensor.__blocktime__,
        )
        if not subscribe_success:
            raise RuntimeError('Failed to subscribe neuron.')

        # ---- Starting axon ----
        logger.info("Serving Axon on: {}:{}", self.axon.config.axon.local_ip, self.axon.config.axon.local_port)
        self.axon.start()
        
        # ---- Sync graph ----
        logger.info("Syncing Metagraph")
        self.metagraph.sync()
        print(self.metagraph)
        
        # --- Run Forever ----
        while True:

            # ---- Serve ----
            self.axon.serve( self._model )

            # ---- Train Model ----
            self.training_loss = self.train()
            self.epoch += 1

            # ---- Set weights ----
            self.metagraph.set_weights(
                weights = self.get_row(), 
                wait_for_inclusion = True
            )

            # ---- Sync metagraph ----
            self.metagraph.sync() # Pulls the latest metagraph state (with my update.)

            # ---- Update Tensorboard ----
            self.dendrite.__to_tensorboard__(self.tensorboard, self.global_step)
            self.metagraph.__to_tensorboard__(self.tensorboard, self.global_step)
            self.axon.__to_tensorboard__(self.tensorboard, self.global_step)
            self.tensorboard.add_scalar('Neuron/Train_loss', self.training_loss, self.global_step)
            logger.info("This epoch's training loss: {}...Current best training loss: {}".format(self.training_loss, self.best_train_loss))

    # ---- Training Step logs ----
    def training_logs( self, progress_bar, iteration:int, output: SimpleNamespace ):
        index = self.metagraph.state.index_for_uid[self.metagraph.uid]
        progress_bar.set_infos({
            'GS': colored('{}'.format(self.global_step), 'red'),
            'LS': colored('{}'.format(iteration), 'blue'),
            'Epoch': colored('{}'.format(self.epoch+1), 'green'),
            'L-loss': colored('{:.5f}'.format(output.local_target_loss.item()), 'yellow'),
            'R-loss': colored('{:.5f}'.format(output.remote_target_loss.item()), 'red'),
            'D-loss': colored('{:.5f}'.format(output.distillation_loss.item()), 'green'),
            'lr:': colored('{:e}'.format(self.lr), 'red'),
            'nPeers': colored(self.metagraph.n, 'blue'),
            'Stake(\u03C4)': colored('{:.3f}'.format(self.metagraph.S[index]), 'green'),
            'Rank(\u03C4)': colored('{:.6f}'.format(self.metagraph.R[index]), 'yellow'),
            'Incentive(\u03C4/block)': colored('{:.3f}'.format(self.metagraph.I[index]), 'red'),
            'Axon': self.axon.__str__(),
            'Dendrite': self.dendrite.__str__(),
        })
        self.tensorboard.add_scalar('R-loss', output.remote_target_loss.item(), self.global_step)
        self.tensorboard.add_scalar('L-loss', output.local_target_loss.item(), self.global_step)
        self.tensorboard.add_scalar('D-loss', output.distillation_loss.item(), self.global_step)

    # ---- Train Epoch ----
    def train(self):
        def run_epoch():
            training_loss = 0.0
            training_batches = self.next_training_batches( epoch = self.epoch )
            progress_bar = qqdm(enumerate(training_batches), total=len(training_batches), desc=format_str('blue', f'Epoch Progress'))
            for iteration, (training_batch) in progress_bar:
                output = self.training_forward( training_batch )
                training_loss += output.loss.item()
                self.training_logs( progress_bar, iteration = iteration, output = output )
                self.global_step += 1
            training_loss /= (iteration + 1)
        run_epoch()
        return training_loss

    def stop(self):
        logger.info('Shutting down the Axon server ...')
        try:
            self.axon.stop()
            logger.info('Axon server stopped')
        except Exception as e:
            logger.error('Neuron: Error while stopping axon server: {} ', e)

    def __del__(self):
        self.stop()


