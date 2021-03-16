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

import argparse
from munch import Munch
from loguru import logger

import multiprocessing.managers
from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager, NamespaceProxy, BaseProxy, AutoProxy

import bittensor

def AutoProxy(token, serializer, manager=None, authkey=None,
              exposed=None, incref=True, manager_owned=False):
    '''
    Return an auto-proxy for `token`
    '''
    _Client = multiprocessing.managers.listener_client[serializer][1]

    if exposed is None:
        conn = _Client(token.address, authkey=authkey)
        try:
            exposed = dispatch(conn, None, 'get_methods', (token,))
        finally:
            conn.close()

    if authkey is None and manager is not None:
        authkey = manager._authkey
    if authkey is None:
        authkey = multiprocessing.process.current_process().authkey

    ProxyType = multiprocessing.managers.MakeProxyType('AutoProxy[%s]' % token.typeid, exposed)
    proxy = ProxyType(token, serializer, manager=manager, authkey=authkey,
                      incref=incref, manager_owned=manager_owned)
    proxy._isauto = True
    return proxy
multiprocessing.managers.AutoProxy = AutoProxy

class Neuron:
    r"""
        Encapsulates bittensor objects as a single object.
    """
    def __init__( self, config: Munch = None, wallet: 'bittensor.Wallet' = None, **kwargs ):
        if config == None:
            config = Neuron.default_config()
        bittensor.Config.update_with_kwargs(config.neuron, kwargs) 
        Neuron.check_config(config)
        self.config = config

        if wallet == None:
            wallet = bittensor.Wallet ( config )
        else:
            config.wallet = wallet.config.wallet
        self.wallet = wallet
        
        if self.config.neuron.multiprocessing:
            BaseManager.register('Subtensor', bittensor.Subtensor)
            BaseManager.register('Metagraph', bittensor.Metagraph)
            BaseManager.register('Dendrite', bittensor.Dendrite)
            BaseManager.register('Axon', bittensor.Axon)
            manager = BaseManager()
            manager.start()

            self.subtensor = manager.Subtensor( config = self.config, wallet = self.wallet )
            self.metagraph = manager.Metagraph( config = self.config, wallet = self.wallet )
            self.dendrite = manager.Dendrite( config = self.config, walelt = self.wallet )
            self.axon = manager.Axon( config = self.config, wallet = self.wallet )
        else:
            self.subtensor = bittensor.Subtensor( config = self.config, wallet = self.wallet )
            self.metagraph = bittensor.Metagraph( config = self.config, wallet = self.wallet )
            self.dendrite = bittensor.Dendrite( config = self.config, walelt = self.wallet )
            self.axon = bittensor.Axon( config = self.config, wallet = self.wallet )

    @staticmethod       
    def default_config() -> Munch:
        parser = argparse.ArgumentParser(); 
        Neuron.add_args(parser) 
        config = bittensor.Config.to_config(parser); 
        return config

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        bittensor.Wallet.add_args( parser )
        bittensor.Subtensor.add_args( parser )
        bittensor.Metagraph.add_args( parser )
        bittensor.Axon.add_args(parser)
        bittensor.Dendrite.add_args( parser )
        try:
            parser.add_argument('--neuron.modality', default=0, type=int, 
                                help='''Neuron network modality. TEXT=0, IMAGE=1. Currently only allowed TEXT''')
            parser.add_argument('--neuron.multiprocessing', default=False, type=bool, 
                                help='''Are bittensor components process safe objects or run from a single thread.''')
            parser.add_argument('--neuron.debug', default=False, type=bool, 
                                help='''True if forward and backward calls print response messages to the screen''')
        except:
            pass

    @staticmethod   
    def check_config(config: Munch):
        bittensor.Axon.check_config( config )
        bittensor.Subtensor.check_config( config )
        bittensor.Metagraph.check_config( config )
        bittensor.Dendrite.check_config( config )
        assert config.neuron.modality == bittensor.proto.Modality.TEXT, 'Only TEXT modalities are allowed at this time.'
