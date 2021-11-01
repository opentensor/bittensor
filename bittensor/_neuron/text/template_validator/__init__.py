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

""" Template validator.

Example:
    $ import neurons
    $ neurons.text.template_validator.neuron().run()
"""

import argparse
import bittensor
import os
import torch

from .nucleus_impl import Validator
from .run import run


class neuron:

    def __init__(
        self, 
        config: 'bittensor.config' = None
    ):
        if config == None: config = neuron.config()
        config = config; 
        self.check_config( config )
        bittensor.logging (
            config = config,
            logging_dir = config.neuron.full_path,
        )
        self.config = config

        # Load/Create our bittensor wallet.
        self.wallet = bittensor.wallet ( config = config ).create().register()

        # Connect to the chain.
        self.subtensor = bittensor.subtensor ( config = config )
    
        # Load/Sync/Save our metagraph.
        self.metagraph = bittensor.metagraph ( subtensor = self.subtensor ).load().sync().save()
        
        self.uid = self.metagraph.hotkeys.index ( self.wallet.hotkey.ss58_address )

        # Create Dendrite.
        self.dendrite = bittensor.dendrite ( config = config )

        # Load genesis dataset.
        self.dataset = bittensor.dataset ( config = config )

        # Build Device.
        self.device = torch.device ( device = config.neuron.device )    

        self.nucleus = Validator(config=config, metagraph = self.metagraph, dendrite = self.dendrite, device = self.device)


    def run(self):
        run(self.config,
            validator = self.nucleus,
            subtensor = self.subtensor,
            wallet = self.wallet,
            metagraph = self.nucleus.metagraph,
            dataset = self.dataset,
            device = self.device,
            uid = self.uid,
            dendrite = self.nucleus.dendrite)

    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        r""" Checks/validates the config namespace object.
        """
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

    def config ():
        parser = argparse.ArgumentParser()    
        parser.add_argument('--config', type=str, help='If set, defaults are overridden by passed file.')
        parser.add_argument('--neuron.name', type=str, help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default='template_miner')
        parser.add_argument('--neuron.resume', action='store_true', help='resume previous trial.', default=False)
        parser.add_argument('--neuron.topk', type=int, help='the number of peers queried during each remote forward call', default=20)
        parser.add_argument('--neuron.learning_rate', type=float, help='Training initial learning rate.', default=1)
        parser.add_argument('--neuron.learning_rate_chain', type=float, help='Training initial learning rate.', default=1)
        parser.add_argument('--neuron.momentum', type=float, help='optimizer momentum.', default=0.8)
        parser.add_argument('--neuron.blocks_per_epoch', type=int, help='Blocks per epoch', default=30)
        parser.add_argument('--neuron.n_topk_peer_weights', type=int, help='Maximum number of weights to submit to chain', default=100 )
        parser.add_argument('--neuron.device', type=str, help='miner default training device cpu/cuda', default=("cuda" if torch.cuda.is_available() else "cpu"))
        parser.add_argument('--neuron.clip_gradients', type=float, help='Implement gradient clipping to avoid exploding loss on smaller architectures.', default=1.0)
        parser.add_argument('--nucleus.topk', type=int, help='the number of peers queried during each remote forward call', default=20)
        parser.add_argument('--nucleus.noise_multiplier', type=float, help='Noise standard deviation multiplier. Increases query exploration.', default=1.0)
        parser.add_argument('--nucleus.nhid', type=int, help='the dimension of the feedforward network model in nn.TransformerEncoder', default=200)
        parser.add_argument('--nucleus.nhead', type=int, help='the number of heads in the multiheadattention models', default=2)
        parser.add_argument('--nucleus.nlayers', type=int, help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder', default=2)
        parser.add_argument('--nucleus.dropout', type=float, help='the dropout value', default=0.2)
        parser.add_argument('--nucleus.punishment', type=float, help='the punishment for those not responding', default=0)

        
        bittensor.wallet.add_args( parser )
        bittensor.dendrite.add_args( parser )
        bittensor.subtensor.add_args( parser )
        bittensor.logging.add_args( parser )
        bittensor.dataset.add_args( parser )
        bittensor.wandb.add_args(parser)
        return bittensor.config( parser )