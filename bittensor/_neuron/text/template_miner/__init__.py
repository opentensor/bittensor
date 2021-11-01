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

""" Template neuron.

Example:
    $ import neurons
    $ neurons.text.template_miner.neuron().run()
"""

import argparse
import bittensor
import os
import sys
import torch

from .neuron_impl import Neuron
from .nucleus_impl import Nucleus


class nucleus():

    def __new__(self, config: 'bittensor.Config' ):
        return Nucleus( config )

    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        r""" Add custom params to the parser.
        """
        parser.add_argument('--nucleus.nhid', type=int, help='the dimension of the feedforward network model in nn.TransformerEncoder', default=200)
        parser.add_argument('--nucleus.nhead', type=int, help='the number of heads in the multiheadattention models', default=2)
        parser.add_argument('--nucleus.nlayers', type=int, help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder', default=2)
        parser.add_argument('--nucleus.dropout', type=float, help='the dropout value', default=0.2)
        parser.add_argument('--nucleus.topk', type=int, help='the number of peers queried during each remote forward call', default=20)
        parser.add_argument('--nucleus.punishment', type=float, help='The punishment on the chain weights that do not respond ', default=0.001 )

class neuron:

    def __new__(
        cls, 
        config: 'bittensor.config' = None
    ):
        if config == None: config = neuron.config()
        config = config; print(config)
        cls.check_config( config )

        neuron_nucleus = nucleus( config )
        bittensor.logging (
            config = config,
            logging_dir = config.neuron.full_path,
        )
        return Neuron( config, neuron_nucleus )

    @staticmethod
    def config() -> 'bittensor.Config':
        r""" Fills a config namespace object with defaults or information from the command line.
        """
        # ---- Add neuron args.
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, help='If set, defaults are overridden by passed file.')
        parser.add_argument('--neuron.learning_rate', type=float, help='Training initial learning rate.', default=1)
        parser.add_argument('--neuron.learning_rate_chain', type=float, help='Training initial learning rate.', default=1)
        parser.add_argument('--neuron.weight_decay', type=float, help='nucleus parameter weight decay.', default=0.25)
        parser.add_argument('--neuron.momentum', type=float, help='optimizer momentum.', default=0.8)
        parser.add_argument('--neuron.clip_gradients', type=float, help='Implement gradient clipping to avoid exploding loss on smaller architectures.', default=1.0)
        parser.add_argument('--neuron.n_epochs', type=int, help='Number of training epochs.', default=sys.maxsize )
        parser.add_argument('--neuron.epoch_length', type=int, help='Iterations of training per epoch', default=100)
        parser.add_argument('--neuron.batch_size_train', type=int, help='Training batch size.', default=2)
        parser.add_argument('--neuron.restart_on_failure',  action='store_true', help='''Restart neuron on unknown error.''', default=False)
        parser.add_argument('--neuron.compute_remote_gradients', action='store_true', help='''Does the neuron compute and return gradients from backward queries.''', default=False)
        parser.add_argument('--neuron.accumulate_remote_gradients', action='store_true', help='''Does the neuron accumulate remote gradients from backward queries.''', default=False)
        parser.add_argument('--neuron.n_topk_peer_weights', type=int, help='Maximum number of weights to submit to chain', default=100 )
        parser.add_argument('--neuron.name', type=str, help='Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name ', default='template_neuron')
        parser.add_argument('--neuron.device', type=str, help='neuron default training device cpu/cuda', default=("cuda" if torch.cuda.is_available() else "cpu"))
        parser.add_argument('--neuron.timeout', type=int, help='Number of seconds to wait for axon request', default=10)
        parser.add_argument('--neuron.blacklist', type=float, help='Amount of stake (tao) in order not to get blacklisted', default=0)
        parser.add_argument('--neuron.sync_block_time', type=int, help='How often the sync the neuron with metagraph, in terms of block time', default=15)
        parser.add_argument('--neuron.restart', type=bool, help='If True, train the neuron from the beginning', default=False)
        parser.add_argument('--neuron.use_wandb', action='store_true', help='''neuron activates its weights and biases powers''', default=False)
        parser.add_argument('--neuron.use_upnpc', action='store_true', help='''neuron attempts to port forward axon using upnpc.''', default=False)

        bittensor.logging.add_args( parser )
        bittensor.wallet.add_args( parser )
        bittensor.subtensor.add_args( parser )
        bittensor.metagraph.add_args( parser )
        bittensor.dataset.add_args( parser )
        bittensor.dendrite.add_args( parser )
        bittensor.axon.add_args( parser )
        bittensor.wandb.add_args( parser )
        nucleus.add_args( parser ) 
        return bittensor.config( parser )

    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        r""" Checks/validates the config namespace object.
        """
        assert config.neuron.batch_size_train > 0, "batch_size_train must be a positive value"
        assert config.neuron.learning_rate > 0, "learning_rate must be a positive value."
        bittensor.logging.check_config( config )
        bittensor.wallet.check_config( config )
        bittensor.subtensor.check_config( config )
        bittensor.metagraph.check_config( config )
        bittensor.dataset.check_config( config )
        bittensor.dendrite.check_config( config )
        bittensor.axon.check_config( config )
        bittensor.wandb.check_config( config )
        full_path = os.path.expanduser('{}/{}/{}/{}'.format( config.logging.logging_dir, config.wallet.name, config.wallet.hotkey, config.neuron.name ))
        config.neuron.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.neuron.full_path):
            os.makedirs(config.neuron.full_path)
