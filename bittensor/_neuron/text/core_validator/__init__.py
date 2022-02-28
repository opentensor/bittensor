# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.

""" Validator objects.

Example:
    >> import neurons
    >> neuron = neurons.text.validator.neuron()
    >> nucleus = neurons.text.validator.nucleus()
    >> neuron.run()
"""

import argparse
import bittensor
import os
import torch

from .validator_impl import Neuron, Nucleus

class nucleus:
    """ Validator Nucleus.
    
    Creates the validator nucleus.

    Example:
        $ import bittensor
        $ nucleus = bittensor.neurons.text.validator.nucleus()
    """
    def __new__(
        cls, 
        config: 'bittensor.config',
        device
    ):
        cls.check_config( config )
        return Nucleus( config = config, device = device)

    @classmethod
    def add_args( cls, parser ):
        parser.add_argument('--nucleus.topk', type=int, help='the number of peers queried during each remote forward call', default = 50 )
        parser.add_argument('--nucleus.nhid', type=int, help='the dimension of the feedforward network model in nn.TransformerEncoder', default=200 )
        parser.add_argument('--nucleus.nhead', type=int, help='the number of heads in the multiheadattention models', default = 2 )
        parser.add_argument('--nucleus.nlayers', type=int, help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder', default=2 )
        parser.add_argument('--nucleus.dropout', type=float, help='the dropout value', default=0.2)
        parser.add_argument('--nucleus.importance', type=float, help='hyperparameter for the importance loss', default=0.1)

    @classmethod
    def config ( cls ):
        parser = argparse.ArgumentParser()    
        cls.add_args( parser )
        nucleus.add_args( parser )        
        return bittensor.config( parser )

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        pass


class neuron:
    """ Neuron factory class.
    
    Creates validator objects from config for passing to run function.

    Example:
        $ import neurons
        $ neurons.text.validator.neuron()
    """
    def __new__(
        cls, 
        config: 'bittensor.config' = None
    ):
        if config == None: config = cls.config()
        cls.check_config( config )
        config.to_defaults()
        _mock = True if config.neuron._mock == True else None
        if config.neuron._mock == True:
            config.subtensor._mock = True
            config.wallet._mock = True
            config.dataset._mock = True
            config.dendrite._mock = True
            config.metagraph._mock = True
            config.subtensor._mock = True
        print (config)
        bittensor.logging( config = config, logging_dir = config.neuron.full_path )
        wallet = bittensor.wallet ( config = config )
        subtensor = bittensor.subtensor ( config = config)
        metagraph = bittensor.metagraph ( config = config, subtensor = subtensor )        
        dendrite = bittensor.dendrite ( config = config, wallet = wallet)
        dataset = bittensor.dataset ( config = config )
        device = torch.device ( device = config.neuron.device )    
        nucleus_model = nucleus ( config = config, device = device ).to( device )
        return Neuron ( 
            config = config,
            wallet = wallet,
            subtensor = subtensor,
            metagraph = metagraph,
            dendrite = dendrite,
            dataset = dataset,
            nucleus = nucleus_model,
            device = device
        )

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        r""" Checks/validates the config namespace object.
        """
        nucleus.check_config( config )
        bittensor.logging.check_config( config )
        bittensor.wallet.check_config( config )
        bittensor.subtensor.check_config( config )
        bittensor.metagraph.check_config( config )
        bittensor.dataset.check_config( config )
        bittensor.dendrite.check_config( config )
        bittensor.wandb.check_config( config )
        full_path = os.path.expanduser('{}/{}/{}/{}'.format( config.logging.logging_dir, config.wallet.name, config.wallet.hotkey, config.neuron.name ))
        config.neuron.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.neuron.full_path):
            os.makedirs(config.neuron.full_path)

    @classmethod
    def add_args( cls, parser ):
        parser.add_argument('--config', type=str, help='If set, defaults are overridden by passed file.')
        parser.add_argument('--neuron.name', type=str, help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default='core_validator')
        parser.add_argument('--neuron.no_restart', action='store_true', help='resume previous trial.', default=False )
        parser.add_argument('--neuron.learning_rate', type=float, help='Training initial learning rate.', default=0.1 )
        parser.add_argument('--neuron.momentum', type=float, help='optimizer momentum.', default=0.8 )
        parser.add_argument('--neuron.blocks_per_epoch', type=int, help='Blocks per epoch', default = 1000 )
        parser.add_argument('--neuron.epochs_until_reset', type=int, help='Number of epochs before weights are reset.', default = 10 )
        parser.add_argument('--neuron.device', type=str, help='miner default training device cpu/cuda', default=("cuda" if torch.cuda.is_available() else "cpu"))
        parser.add_argument('--neuron.clip_gradients', type=float, help='Implement gradient clipping to avoid exploding loss on smaller architectures.', default=1.0 )
        parser.add_argument('--neuron.restart_on_failure',  action='store_true', help='''Restart neuron on unknown error.''', default=True )
        parser.add_argument('--neuron._mock', action='store_true', help='To turn on neuron mocking for testing purposes.', default=False )

    @classmethod
    def config ( cls ):
        parser = argparse.ArgumentParser()    
        cls.add_args( parser )
        nucleus.add_args( parser )        
        bittensor.wallet.add_args( parser )
        bittensor.dendrite.add_args( parser )
        bittensor.subtensor.add_args( parser )
        bittensor.metagraph.add_args( parser )
        bittensor.logging.add_args( parser )
        bittensor.dataset.add_args( parser )
        bittensor.wandb.add_args(parser)
        return bittensor.config( parser )