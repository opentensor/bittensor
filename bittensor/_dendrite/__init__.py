""" Create and init class dendrite, which quries endpoints with tensors.
"""
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
import os
import copy
import bittensor
from . import dendrite_impl
from . import dendrite_mock
from .manager_server import ManagerServer
from multiprocessing.managers import BaseManager
from loguru import logger

class dendrite:
    r""" This is the factory class for a bittensor.dendrite() or the mocked dendrite class.
    
    The dendrite class operates as a normal torch autograd friendly operation which accepts a list of bittensor.endpoints and a list of torch tensors. 
    The passed endpoints are queried with the passed inputs and either return results or zeros. The operation is fully differentiable with a torch 
    computation graph such that calls to loss.backward() produce Backward calls on the passed endpoints.
    
    """

    def __new__(
            cls, 
            config: 'bittensor.config' = None,
            wallet: 'bittensor.Wallet' = None,
            timeout: int = None,
            requires_grad: bool = None,
            max_active_receptors: int = None,
            receptor_pool: 'bittensor.ReceptorPool' = None,
            multiprocess: bool = None,
            compression: str = None,
            _mock:bool=None
        ) -> 'bittensor.Dendrite':
        r""" Creates a new Dendrite object from passed arguments.
            Args:
                config (:obj:`bittensor.Config`, `optional`): 
                    Config namespace object created by calling bittensor.dendrite.config()
                wallet (:obj:`bittensor.Wallet`, `optional`):
                    A bittensor wallet object containing a pair of cryptographic keys, the hot and coldkey, used for signing messages
                    on the wire.
                timeout (:type:`int`, `optional`, default: bittensor.dendrite.config().dendrite.timeout ):
                    Default request timeout.
                requires_grad (:type:`bool`, `optional`, default: bittensor.dendrite.config().dendrite.requires_grad):
                    If true, the dendrite passes gradients on the wire by default.
                max_active_receptors (:type:`int`, `optional`, default: bittensor.dendrite.config().dendrite.max_active_receptors):
                    Maximum allowed active allocated TCP connections. Does not override the
                    optionally passed receptor pool.
                receptor_pool (:obj:`bittensor.ReceptorPool`, `optional`):
                    A bittensor receptor pool object which maintains a set of connections to other peers in the network and operates as
                    a normal torch.nn.Module. By default this object is created with the dendrite config.
                _mock (:obj:`bool`, `optional`):
                    For testing, if true the dendrite returns mocked outputs.
        """
        if config == None: 
            config = dendrite.config()
        config = copy.deepcopy(config)
        config.dendrite.timeout = timeout if timeout != None else config.dendrite.timeout
        config.dendrite.requires_grad = requires_grad if requires_grad != None else config.dendrite.requires_grad
        config.dendrite.max_active_receptors = max_active_receptors if max_active_receptors != None else config.dendrite.max_active_receptors
        config.dendrite.compression = compression if compression != None else config.dendrite.compression
        config.dendrite._mock = _mock if _mock != None else config.dendrite._mock
        dendrite.check_config( config )

        if wallet == None:
            wallet = bittensor.wallet( config = config )
            
        if receptor_pool == None:
            receptor_pool = bittensor.receptor_pool( 
                wallet = wallet,
                max_active_receptors = config.dendrite.max_active_receptors,
                compression = config.dendrite.compression,
            )
        if config.dendrite._mock:
            return dendrite_mock.DendriteMock ( 
                config = config,
                wallet = wallet
            )
        else:
            return dendrite_impl.Dendrite ( 
                config = config,
                wallet = wallet, 
                receptor_pool = receptor_pool,
            )

    @classmethod   
    def config(cls) -> 'bittensor.Config':
        """ Get config from the argument parser
        """
        parser = argparse.ArgumentParser()
        dendrite.add_args( parser )
        return bittensor.config( parser )

    @classmethod   
    def help(cls):
        """ Print help to stdout
        """
        parser = argparse.ArgumentParser()
        cls.add_args( parser )
        print (cls.__new__.__doc__)
        parser.print_help()

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser, prefix: str = None ):
        """ Accept specific arguments from parser
        """
        prefix_str = '' if prefix == None else prefix + '.'
        try:
            parser.add_argument('--' + prefix_str + 'dendrite.max_active_receptors', type=int, help='''Max number of concurrently active receptors / tcp-connections''',  default = bittensor.defaults.dendrite.max_active_receptors) 
            parser.add_argument('--' + prefix_str + 'dendrite.timeout', type=int, help='''Default request timeout.''', default = bittensor.defaults.dendrite.timeout)
            parser.add_argument('--' + prefix_str + 'dendrite.requires_grad', action='store_true', help='''If true, the dendrite passes gradients on the wire.''', default = bittensor.defaults.dendrite.requires_grad)
            parser.add_argument('--' + prefix_str + 'dendrite.no_requires_grad', dest = prefix_str + 'dendrite.requires_grad', action='store_false', help='''If set, the dendrite will not passes gradients on the wire.''')
            parser.add_argument('--' + prefix_str + 'dendrite.compression', type=str, help='''Which compression algorithm to use for compression (gzip, deflate, NoCompression) ''', default = bittensor.defaults.dendrite.compression)
            parser.add_argument('--' + prefix_str + 'dendrite._mock', action='store_true', help='To turn on dendrite mocking for testing purposes.', default=False)
            parser.add_argument('--' + prefix_str + 'dendrite.prometheus.level', 
                required = False, 
                type = str, 
                choices = [l.name for l in list(bittensor.prometheus.level)], 
                default = bittensor.defaults.dendrite.prometheus.level, 
                help = '''Prometheus logging level for dendrite. <OFF | INFO | DEBUG>''')
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass
        bittensor.wallet.add_args( parser, prefix = prefix)

    @classmethod   
    def add_defaults(cls, defaults):
        """ Adds parser defaults to object from enviroment variables.
        """
        defaults.dendrite = bittensor.Config()
        defaults.dendrite.max_active_receptors = os.getenv('BT_DENDRITE_MAX_ACTIVE_RECEPTORS') if os.getenv('BT_DENDRITE_MAX_ACTIVE_RECEPTORS') != None else 4096
        defaults.dendrite.timeout = os.getenv('BT_DENDRITE_TIMEOUT') if os.getenv('BT_DENDRITE_TIMEOUT') != None else bittensor.__blocktime__ + 2
        defaults.dendrite.requires_grad = os.getenv('BT_DENDRITE_REQUIRES_GRAD') if os.getenv('BT_DENDRITE_REQUIRES_GRAD') != None else True
        defaults.dendrite.compression = os.getenv('BT_DENDRITE_COMPRESSION') if os.getenv('BT_DENDRITE_COMPRESSION') != None else 'NoCompression'
        # Prometheus
        defaults.dendrite.prometheus = bittensor.config()
        defaults.dendrite.prometheus.level = os.getenv('BT_DENDRITE_PROMETHEUS_LEVEL') if os.getenv('BT_DENDRITE_PROMETHEUS_LEVEL') != None else bittensor.prometheus.level.DEBUG.name


    @classmethod   
    def check_config( cls, config: 'bittensor.Config' ):
        """ Check config for dendrite worker and receptors
        """
        assert config.dendrite
        assert 'timeout' in config.dendrite
        assert 'requires_grad' in config.dendrite
        assert config.dendrite.max_active_receptors >= 0, 'max_active_receptors must be larger or eq to 0'
        assert config.dendrite.prometheus.level in [l.name for l in list(bittensor.prometheus.level)], "dendrite.prometheus.level must be in: {}".format([l.name for l in list(bittensor.prometheus.level)])        
        bittensor.wallet.check_config( config )
