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
import grpc
import bittensor.utils.networking as net
from concurrent.futures import ThreadPoolExecutor

from . import receptor_impl

class receptor:

    def __new__(
            cls, 
            endpoint: 'bittensor.Endpoint', 
            config: 'bittensor.Config' = None,
            wallet: 'bittensor.Wallet' = None,
            timeout: int = None,
            do_backoff: bool = None,
            max_backoff:int = None,
            namespace: str = ''
        ) -> 'bittensor.Receptor':
        r""" Initializes a receptor grpc connection.
            Args:
                endpoint (:obj:`bittensor.Endpoint`, `required`):
                    neuron endpoint descriptor.
                wallet (:obj:`bittensor.Wallet`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
                timeout (default=0.5, type=float):
                    The per request RPC timeout. a.k.a the maximum request time.
                do_backoff (default=True, type=bool)
                    Neurons who return non successful return codes are
                        periodically not called with a multiplicative backoff.
                        The backoff doubles until max_backoff and then halves on ever sequential successful request.
                max_backoff (default=100, type=int)
                    The backoff doubles until this saturation point.
                config (:obj:`bittensor.Config`, `optional`): 
                    bittensor.receptor.config()
                namespace (:obj:`str, `optional`): 
                    config namespace.
        """        
        if config == None:
            config = bittensor.config.cut_namespace( receptor.config( namespace ), namespace ).receptor
        config.timeout = timeout if timeout != None else config.timeout
        config.do_backoff = do_backoff if do_backoff != None else config.do_backoff
        config.max_backoff = max_backoff if max_backoff != None else config.max_backoff
        config = copy.deepcopy(config) # Configuration information.

        if wallet == None:
            wallet = bittensor.wallet( config )
        config.wallet = copy.deepcopy(wallet.config)

        # Get remote IP.
        try:
            external_ip = config.axon.external_ip
        except:
            pass
        try:
            external_ip = str(net.get_external_ip())
        except:
            pass
        finally:
            external_ip = None

        # Get endpoint string.
        if endpoint.ip == external_ip:
            ip = "localhost:"
            endpoint_str = ip + str(endpoint.port)
        else:
            endpoint_str = endpoint.ip + ':' + str(endpoint.port)
        
        # Make channel and stub.
        channel = grpc.insecure_channel(
            endpoint_str,
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1)])
        stub = bittensor.grpc.BittensorStub( channel )

        receptor.check_config( config )
        return receptor_impl.Receptor( 
            config = config, 
            wallet = wallet, 
            endpoint = endpoint, 
            channel = channel, 
            stub = stub
        )

    @staticmethod   
    def config() -> 'bittensor.Config':
        parser = argparse.ArgumentParser()
        receptor.add_args(parser) 
        config = bittensor.config( parser ); 
        return config

    @staticmethod   
    def check_config(config: 'bittensor.Config'):
        bittensor.wallet.check_config( config )
        assert config.timeout >= 0, 'timeout must be positive value, got {}'.format(config.timeout)

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser, namespace: str = ''):
        if namespace != '':
            namespace = namespace + 'receptor.'
        else:
            namespace = 'receptor.'
        bittensor.wallet.add_args( parser, namespace = namespace )
        parser.add_argument('--' + namespace + 'timeout', default=3, type=float, 
            help='''The per request RPC timeout. a.k.a the maximum request time.''')
        parser.add_argument('--' + namespace + 'do_backoff', default=True, type=bool, 
            help='''Neurons who return non successful return codes are
                    periodically not called with a multiplicative backoff.
                    The backoff doubles until max_backoff and then halves on ever sequential successful request.''')
        parser.add_argument('--' + namespace + 'max_backoff', default=100, type=int, 
            help='''The backoff doubles until this saturation point.''')

    @staticmethod   
    def print_help(namespace: str = ''):
        parser = argparse.ArgumentParser(); 
        receptor.add_args( parser, namespace ) 
        parser.print_help()

class receptor_pool:

    def __new__(
            cls, 
            config: 'bittensor.Config' = None,
            wallet: 'bittensor.Wallet' = None,
            thread_pool: ThreadPoolExecutor = None,
            max_worker_threads: int = None,
            max_active_receptors: int = None,
            pass_gradients: bool = None,
            timeout: int = None,
            do_backoff : bool =  None,
            max_backoff: int = None,
            namespace: str = ''
        ) -> 'bittensor.ReceptorPool':
        r""" Initializes a receptor grpc connection.
            Args:
                endpoint (:obj:`bittensor.Endpoint`, `required`):
                    neuron endpoint descriptor.
                config (:obj:`bittensor.Config`, `optional`): 
                    bittensor.metagraph.config()
                wallet (:obj:`bittensor.Wallet`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
                thread_pool (:obj:`ThreadPoolExecutor`, `optional`):
                    thread pool executor passed the receptor pool unless defined.s
                max_worker_threads (:type:`int`, `optional`):
                    Maximum number of active client threads. Does not override passed 
                    Threadpool.
                max_active_receptors (:type:`int`, `optional`):
                    Maximum allowed active allocated TCP connections.
                pass_gradients (:type:`bool`, `optional`):
                    If true, the dendrite passes gradients on the wire by default.
                timeout (:type:`int`, `optional`):
                    Default request timeout.
                do_backoff (:type:`bool`, `optional`):
                    If true, non-successful requests cause a backoff from the error'd server.
                max_backoff (:type:`int`, `optional`):
                    If do_backoff, max_backoff is the number of maximum number of backed off requests
                    before another test query is sent.
                                config (:obj:`bittensor.Config`, `optional`): 
                    receptor_pool.config()
                namespace (:obj:`str, `optional`): 
                    config namespace.
        """        
        if config == None:
            config = bittensor.config.cut_namespace( receptor_pool.config( namespace ), namespace ).receptor_pool
        config.max_worker_threads = max_worker_threads if max_worker_threads != None else config.max_worker_threads
        config.max_active_receptors = max_active_receptors if max_active_receptors != None else config.max_active_receptors
        config.receptor.timeout = timeout if timeout != None else config.receptor.timeout
        config.receptor.do_backoff = do_backoff if do_backoff != None else config.receptor.do_backoff
        config.receptor.max_backoff = max_backoff if max_backoff != None else config.receptor.max_backoff
        config = copy.deepcopy(config) # Configuration information.

        if thread_pool == None:
            thread_pool = ThreadPoolExecutor( max_workers = config.max_worker_threads )

        receptor_pool.check_config( config )
        return bittensor.ReceptorPool( config = config, wallet = wallet, thread_pool = thread_pool )

    @staticmethod   
    def config( namespace: str = '' ) -> 'bittensor.Config':
        parser = argparse.ArgumentParser()
        receptor_pool.add_args( parser, namespace ) 
        config = bittensor.config( parser ) 
        return bittensor.config.cut_namespace( config, namespace )

    @staticmethod   
    def check_config( config: 'bittensor.Config' ):
        assert config.max_worker_threads >= 0, 'max worker threads must be positive {}'.format(config.receptor_pool.max_worker_threads)
        assert config.max_active_receptors >= 0, 'max active receptors must be positive {}'.format(config.receptor_pool.max_active_receptors)

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser, namespace: str = ''):
        if namespace != '':
            namespace = namespace + 'receptor_pool.'
        else:
            namespace = 'receptor_pool.'
        bittensor.receptor.add_args( parser, namespace = namespace)
        parser.add_argument('--' + namespace + 'max_worker_threads', default=20, type=int, 
                help='''Max number of concurrent threads used for sending RPC requests.''')
        parser.add_argument('--' + namespace + 'max_active_receptors', default=150, type=int, 
                help='''Max number of concurrently active receptors / tcp-connections''')
        return parser

    @staticmethod   
    def print_help(namespace: str = ''):
        parser = argparse.ArgumentParser(); 
        receptor_pool.add_args( parser, namespace ) 
        parser.print_help()