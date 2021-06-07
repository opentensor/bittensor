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

import bittensor
import argparse
import copy

from . import executor_impl

class executor:

    def __new__(
            cls,
            config: 'bittensor.Config' = None, 
            wallet: 'bittensor.Wallet' = None,
            subtensor: 'bittensor.Subtensor' = None,
            metagraph: 'bittensor.Metagraph' = None,
            axon: 'bittensor.Axon' = None,
            dendrite: 'bittensor.Dendrite' = None
        ) -> 'bittensor.Executor':
        r""" Creates a new Executor object from passed arguments.
            Args:
                config (:obj:`bittensor.Config`, `optional`): 
                    bittensor.executor.config()
                wallet (:obj:`bittensor.Wallet`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
                subtensor (:obj:`bittensor.Subtensor`, `optional`):
                    Bittensor subtensor chain connection.
                metagraph (:obj:`bittensor.Metagraph`, `optional`):
                    Bittensor metagraph chain state.
                axon (:obj:`bittensor.Axon`, `optional`):
                    Bittensor axon server.
                dendrite (:obj:`bittensor.Dendrite`, `optional`):
                    Bittensor dendrite client.
        """
        if config == None:
            config = executor.config()
        config = copy.deepcopy(config)
        if wallet == None:
            wallet = bittensor.wallet ( config = config.wallet )
        config.wallet = copy.deepcopy(wallet.config)
        if subtensor == None:
            subtensor = bittensor.subtensor( config = config.subtensor )
        config.subtensor = copy.deepcopy(subtensor.config)
        if metagraph == None:
            metagraph = bittensor.metagraph( config = config.metagraph )
        config.metagraph = copy.deepcopy(metagraph.config)
        if axon == None:
            axon = bittensor.axon( config = config.axon, wallet = wallet )
        config.axon = copy.deepcopy(axon.config)
        if dendrite == None:
            dendrite = bittensor.dendrite( config = config.dendrite, wallet = wallet )
        config.dendrite = copy.deepcopy(dendrite.config)
        executor.check_config( config )
        return executor_impl.Executor ( 
            config = config, 
            wallet = wallet, 
            subtensor = subtensor, 
            metagraph = metagraph, 
            axon = axon, 
            dendrite = dendrite 
        )
    
    @staticmethod
    def config() -> 'bittensor.Config':
        parser = argparse.ArgumentParser(); 
        executor.add_args(parser) 
        config = bittensor.config( parser ); 
        return config

    @staticmethod   
    def add_args (parser: argparse.ArgumentParser):
        bittensor.wallet.add_args( parser )
        bittensor.subtensor.add_args( parser )
        bittensor.axon.add_args( parser )
        bittensor.dendrite.add_args( parser )
        bittensor.metagraph.add_args( parser )
        
    @staticmethod   
    def check_config (config: 'bittensor.Config'):
        bittensor.wallet.check_config( config.wallet )
        bittensor.subtensor.check_config( config.subtensor )
        bittensor.axon.check_config( config.axon )
        bittensor.dendrite.check_config( config.dendrite )
        bittensor.metagraph.check_config( config.metagraph )
