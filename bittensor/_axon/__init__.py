""" Create and init Axon, whcih services Forward and Backward requests from other neurons.
"""
# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2022 Opentensor Foundation

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
import inspect
import time
from concurrent import futures
from typing import Dict, List, Callable, Optional, Tuple, Union
from bittensor._threadpool import prioritythreadpool

import torch
import grpc
from substrateinterface import Keypair

import bittensor
from . import axon_impl

class axon:
    """ The factory class for bittensor.Axon object
    The Axon is a grpc server for the bittensor network which opens up communication between it and other neurons.
    The server protocol is defined in bittensor.proto and describes the manner in which forward and backwards requests
    are transported / encoded between validators and servers
    
    Examples:: 
            >>> config = bittensor.axon.config()
            >>> axon = bittensor.axon( config = config )
            >>> subtensor = bittensor.subtensor( network = 'nakamoto' )
            >>> axon.serve( subtensor = subtensor )
    """

    def __new__(
            cls,
            netuid: int,
            config: Optional['bittensor.config'] = None,
            wallet: Optional['bittensor.Wallet'] = None,
            forward_text: Optional['Callable'] = None,
            backward_text:Optional['Callable'] = None,
            synapse_last_hidden: Optional['Callable'] = None,
            synapse_causal_lm: Optional['Callable'] = None,
            synapse_causal_lm_next: Optional['Callable'] = None,
            synapse_seq_2_seq: Optional['Callable'] = None,
            synapse_lasthidden_timeout: Optional[int] = None,
            synapse_causallm_timeout: Optional[int] = None,
            synapse_causallmnext_timeout: Optional[int] = None,
            synapse_seq2seq_timeout: Optional[int] = None,

            synapse_checks: Optional['Callable'] = None,
            thread_pool: Optional['futures.ThreadPoolExecutor'] = None,
            priority_threadpool: Optional['bittensor.prioritythreadpool'] = None,
            server: Optional['grpc._Server'] = None,
            port: Optional[int] = None,
            ip: Optional[str] = None,
            external_ip: Optional[str] = None,
            external_port: Optional[int] = None,
            protocol: Optional[int] = None,
            max_workers: Optional[int] = None, 
            maximum_concurrent_rpcs: Optional[int] = None,
            blacklist: Optional['Callable'] = None,
            priority: Optional['Callable'] = None,
            forward_timeout: Optional[int] = None,
            backward_timeout: Optional[int] = None,
            compression:Optional[str] = None,
        ) -> 'bittensor.Axon':
        r""" Creates a new bittensor.Axon object from passed arguments.
            Args:
                netuid (:obj:`int`, `required`):
                    The netuid of the subent this axon is serving on.
                config (:obj:`Optional[bittensor.Config]`, `optional`): 
                    bittensor.axon.config()
                wallet (:obj:`Optional[bittensor.Wallet]`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
                forward_text (:obj:`Optional[callable]`, `optional`):
                    function which is called on forward text requests.
                backward_text (:obj:`Optional[callable]`, `optional`):
                    function which is called on backward text requests.
                synapse_last_hidden (:obj:`Optional[callable]`, `optional`):
                    function which is called by the last hidden synapse
                synapse_causal_lm (:obj:`Optional[callable]`, `optional`):
                    function which is called by the causal lm synapse
                synapse_causal_lm_next (:obj:`Optional[callable]`, `optional`):
                    function which is called by the TextCausalLMNext synapse
                synapse_seq_2_seq (:obj:`Optional[callable]`, `optional`):
                    function which is called by the seq2seq synapse   
                synapse_checks (:obj:`Optional[callable]`, 'optional'):
                    function which is called before each synapse to check for stake        
                thread_pool (:obj:`Optional[ThreadPoolExecutor]`, `optional`):
                    Threadpool used for processing server queries.
                server (:obj:`Optional[grpc._Server]`, `required`):
                    Grpc server endpoint, overrides passed threadpool.
                port (:type:`Optional[int]`, `optional`):
                    Binding port.
                ip (:type:`Optional[str]`, `optional`):
                    Binding ip.
                external_ip (:type:`Optional[str]`, `optional`):
                    The external ip of the server to broadcast to the network.
                external_port (:type:`Optional[int]`, `optional`):
                    The external port of the server to broadcast to the network.
                protocol (:type:`Optional[int]`, `optional`):
                    The protocol of the server to broadcast to the network.
                max_workers (:type:`Optional[int]`, `optional`):
                    Used to create the threadpool if not passed, specifies the number of active threads servicing requests.
                maximum_concurrent_rpcs (:type:`Optional[int]`, `optional`):
                    Maximum allowed concurrently processed RPCs.
                blacklist (:obj:`Optional[callable]`, `optional`):
                    function to blacklist requests.
                priority (:obj:`Optional[callable]`, `optional`):
                    function to assign priority on requests.
                forward_timeout (:type:`Optional[int]`, `optional`):
                    timeout on the forward requests. 
                backward_timeout (:type:`Optional[int]`, `optional`):
                    timeout on the backward requests.              
        """   
        if config == None: 
            config = axon.config()
        config = copy.deepcopy(config)
        config.axon.port = port if port != None else config.axon.port
        config.axon.ip = ip if ip != None else config.axon.ip
        config.axon.external_ip = external_ip if external_ip != None else config.axon.external_ip
        config.axon.external_port = external_port if external_port != None else config.axon.external_port
        config.axon.protocol = protocol if protocol != None else config.axon.protocol
        config.axon.max_workers = max_workers if max_workers != None else config.axon.max_workers
        config.axon.maximum_concurrent_rpcs = maximum_concurrent_rpcs if maximum_concurrent_rpcs != None else config.axon.maximum_concurrent_rpcs
        config.axon.forward_timeout = forward_timeout if forward_timeout != None else config.axon.forward_timeout
        config.axon.backward_timeout = backward_timeout if backward_timeout != None else config.axon.backward_timeout
        config.axon.compression = compression if compression != None else config.axon.compression
        config.axon.lasthidden_timeout = synapse_lasthidden_timeout if synapse_lasthidden_timeout != None else config.axon.lasthidden_timeout
        config.axon.causallm_timeout = synapse_causallm_timeout if synapse_causallm_timeout != None else config.axon.causallm_timeout
        config.axon.causallmnext_timeout = synapse_causallmnext_timeout if synapse_causallmnext_timeout is not None else config.axon.causallmnext_timeout
        config.axon.seq2seq_timeout = synapse_seq2seq_timeout if synapse_seq2seq_timeout != None else config.axon.seq2seq_timeout
        axon.check_config( config )

        # Determine the grpc compression algorithm
        if config.axon.compression == 'gzip':
            compress_alg = grpc.Compression.Gzip
        elif config.axon.compression == 'deflate':
            compress_alg = grpc.Compression.Deflate
        else:
            compress_alg = grpc.Compression.NoCompression

        if wallet == None:
            wallet = bittensor.wallet( config = config )
        if thread_pool == None:
            thread_pool = futures.ThreadPoolExecutor( max_workers = config.axon.max_workers )
        if server == None:
            receiver_hotkey = wallet.hotkey.ss58_address
            server = grpc.server( thread_pool,
                                  interceptors=(AuthInterceptor(receiver_hotkey=receiver_hotkey, blacklist=blacklist),),
                                  maximum_concurrent_rpcs = config.axon.maximum_concurrent_rpcs,
                                  options = [('grpc.keepalive_time_ms', 100000),
                                             ('grpc.keepalive_timeout_ms', 500000),
                                             ('grpc.max_receive_message_length', config.axon.maximum_message_length)
                                             ]
                                )

        synapses = {}
        synapses[bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE] = synapse_last_hidden
        synapses[bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM] = synapse_causal_lm
        synapses[bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT] = synapse_causal_lm_next
        synapses[bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ] = synapse_seq_2_seq

        synapse_timeouts = {
            bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE: config.axon.lasthidden_timeout,
            bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM: config.axon.causallm_timeout,
            bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT: config.axon.causallmnext_timeout,
            bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ: config.axon.seq2seq_timeout
        }
        
        synapse_check_function = synapse_checks if synapse_checks != None else axon.default_synapse_check

        if priority != None and priority_threadpool == None:
            priority_threadpool = bittensor.prioritythreadpool(config=config)

        axon_instance = axon_impl.Axon(
            wallet = wallet, 
            server = server,
            ip = config.axon.ip,
            port = config.axon.port,
            external_ip=config.axon.external_ip, # don't use internal ip if it is None, we will try to find it later
            external_port=config.axon.external_port or config.axon.port, # default to internal port if external port is not set
            protocol = config.axon.protocol,
            forward = forward_text,
            backward = backward_text,
            synapses = synapses,
            synapse_checks = synapse_check_function,
            synapse_timeouts = synapse_timeouts,
            priority = priority,
            priority_threadpool = priority_threadpool,
            forward_timeout = config.axon.forward_timeout,
            backward_timeout = config.axon.backward_timeout,
            prometheus_level = config.axon.prometheus.level,
            netuid = netuid,
        )
        bittensor.grpc.add_BittensorServicer_to_server( axon_instance, server )
        full_address = str( config.axon.ip ) + ":" + str( config.axon.port )
        server.add_insecure_port( full_address )
        return axon_instance 

    @classmethod   
    def config(cls) -> 'bittensor.Config':
        """ Get config from the argument parser
        Return: bittensor.config object
        """
        parser = argparse.ArgumentParser()
        axon.add_args( parser )
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
    def add_args( cls, parser: argparse.ArgumentParser, prefix: str = None  ):
        """ Accept specific arguments from parser
        """
        prefix_str = '' if prefix == None else prefix + '.'
        try:
            parser.add_argument('--' + prefix_str + 'axon.port', type=int, 
                    help='''The local port this axon endpoint is bound to. i.e. 8091''', default = bittensor.defaults.axon.port)
            parser.add_argument('--' + prefix_str + 'axon.ip', type=str, 
                help='''The local ip this axon binds to. ie. [::]''', default = bittensor.defaults.axon.ip)
            parser.add_argument('--' + prefix_str + 'axon.external_port', type=int, required=False,
                    help='''The public port this axon broadcasts to the network. i.e. 8091''', default = bittensor.defaults.axon.external_port)
            parser.add_argument('--' + prefix_str + 'axon.external_ip', type=str, required=False,
                help='''The external ip this axon broadcasts to the network to. ie. [::]''', default = bittensor.defaults.axon.external_ip)
            parser.add_argument('--' + prefix_str + 'axon.protocol', type=int, required=False,
                    help='''The protocol this axon broadcasts to the network. i.e. 0''', default = bittensor.defaults.axon.protocol)
            parser.add_argument('--' + prefix_str + 'axon.max_workers', type=int, 
                help='''The maximum number connection handler threads working simultaneously on this endpoint. 
                        The grpc server distributes new worker threads to service requests up to this number.''', default = bittensor.defaults.axon.max_workers)
            parser.add_argument('--' + prefix_str + 'axon.maximum_concurrent_rpcs', type=int, 
                help='''Maximum number of allowed active connections''',  default = bittensor.defaults.axon.maximum_concurrent_rpcs)
            parser.add_argument('--' + prefix_str + 'axon.backward_timeout', type=int,
                help='Number of seconds to wait for backward axon request', default=2*bittensor.__blocktime__)
            parser.add_argument('--' + prefix_str + 'axon.forward_timeout', type=int,
                help='Number of seconds to wait for forward axon request', default=5*bittensor.__blocktime__)
            parser.add_argument('--' + prefix_str + 'axon.priority.max_workers', type = int,
                help='''maximum number of threads in thread pool''', default = bittensor.defaults.axon.priority.max_workers)
            parser.add_argument('--' + prefix_str + 'axon.priority.maxsize', type=int, 
                help='''maximum size of tasks in priority queue''', default = bittensor.defaults.axon.priority.maxsize)
            parser.add_argument('--' + prefix_str + 'axon.compression', type=str, 
                help='''Which compression algorithm to use for compression (gzip, deflate, NoCompression) ''', default = bittensor.defaults.axon.compression)
            parser.add_argument('--' +  prefix_str + 'axon.lasthidden_timeout', type = int, 
            help='Timeout for last hidden synapse', default= bittensor.__blocktime__)
            parser.add_argument('--' +  prefix_str + 'axon.causallm_timeout', type = int, 
            help='Timeout for causallm synapse', default= bittensor.__blocktime__)
            parser.add_argument('--' +  prefix_str + 'axon.causallmnext_timeout', type = int, 
            help='Timeout for causallmnext synapse', default= bittensor.__blocktime__)
            parser.add_argument('--' +  prefix_str + 'axon.seq2seq_timeout', type = int, 
            help='Timeout for seq2seq synapse', default= 3*bittensor.__blocktime__)
            parser.add_argument('--' +  prefix_str + 'axon.maximum_message_length', type = int, 
            help='Maximum message length for requestion', default= 4*1024*1024)
            parser.add_argument('--' + prefix_str + 'axon.prometheus.level', 
                required = False, 
                type = str, 
                choices = [l.name for l in list(bittensor.prometheus.level)], 
                default = bittensor.defaults.axon.prometheus.level, 
                help = '''Prometheus logging level axon. <OFF | INFO | DEBUG>''')
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass

        bittensor.wallet.add_args( parser, prefix = prefix )

    @classmethod   
    def add_defaults(cls, defaults):
        """ Adds parser defaults to object from enviroment variables.
        """
        defaults.axon = bittensor.Config()
        defaults.axon.port = os.getenv('BT_AXON_PORT') if os.getenv('BT_AXON_PORT') != None else 8091
        defaults.axon.ip = os.getenv('BT_AXON_IP') if os.getenv('BT_AXON_IP') != None else '[::]'
        defaults.axon.protocol = os.getenv('BT_AXON_PROTOCOL') if os.getenv('BT_AXON_PROTOCOL') != None else 0
        defaults.axon.external_port = os.getenv('BT_AXON_EXTERNAL_PORT') if os.getenv('BT_AXON_EXTERNAL_PORT') != None else None
        defaults.axon.external_ip = os.getenv('BT_AXON_EXTERNAL_IP') if os.getenv('BT_AXON_EXTERNAL_IP') != None else None
        defaults.axon.max_workers = os.getenv('BT_AXON_MAX_WORERS') if os.getenv('BT_AXON_MAX_WORERS') != None else 10
        defaults.axon.maximum_concurrent_rpcs = os.getenv('BT_AXON_MAXIMUM_CONCURRENT_RPCS') if os.getenv('BT_AXON_MAXIMUM_CONCURRENT_RPCS') != None else 400
        
        defaults.axon.priority = bittensor.Config()
        defaults.axon.priority.max_workers = os.getenv('BT_AXON_PRIORITY_MAX_WORKERS') if os.getenv('BT_AXON_PRIORITY_MAX_WORKERS') != None else 10
        defaults.axon.priority.maxsize = os.getenv('BT_AXON_PRIORITY_MAXSIZE') if os.getenv('BT_AXON_PRIORITY_MAXSIZE') != None else -1

        defaults.axon.compression = 'NoCompression'

        # Prometheus
        defaults.axon.prometheus = bittensor.config()
        defaults.axon.prometheus.level = os.getenv('BT_AXON_PROMETHEUS_LEVEL') if os.getenv('BT_AXON_PROMETHEUS_LEVEL') != None else bittensor.prometheus.level.DEBUG.name

    @classmethod   
    def check_config(cls, config: 'bittensor.Config' ):
        """ Check config for axon port and wallet
        """
        assert config.axon.port > 1024 and config.axon.port < 65535, 'port must be in range [1024, 65535]'
        assert config.axon.external_port is None or (config.axon.external_port > 1024 and config.axon.external_port < 65535), 'external port must be in range [1024, 65535]'
        assert config.axon.prometheus.level in [l.name for l in list(bittensor.prometheus.level)], "axon.prometheus.level must be in: {}".format([l.name for l in list(bittensor.prometheus.level)])
        bittensor.wallet.check_config( config )

    @classmethod   
    def default_synapse_check(cls, synapse, hotkey, inputs ):
        """ default synapse check function
        """
        if len(hotkey) == bittensor.__ss58_address_length__:
            return True
        
        return False

    @staticmethod
    def check_backward_callback( backward_callback:Callable, pubkey:str = '_' ):
        """ Check and test axon backward callback function
        """
        if not inspect.ismethod(backward_callback) and not inspect.isfunction(backward_callback):
            raise ValueError('The axon backward callback must be a function with signature Callable[inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor ) -> torch.FloatTensor:, got {}'.format(backward_callback))        
        if len( inspect.signature(backward_callback).parameters) != 3:
            raise ValueError('The axon backward callback must have signature Callable[ inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, synapses ) -> torch.FloatTensor:, got {}'.format(inspect.signature(backward_callback)))
        if 'inputs_x' not in inspect.signature(backward_callback).parameters:
            raise ValueError('The axon backward callback must have signature Callable[inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor ) -> torch.FloatTensor:, got {}'.format(inspect.signature(backward_callback)))
        if 'grads_dy' not in inspect.signature(backward_callback).parameters:
            raise ValueError('The axon backward callback must have signature Callable[inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor ) -> torch.FloatTensor:, got {}'.format(inspect.signature(backward_callback)))
 

    @staticmethod
    def check_forward_callback( forward_callback:Callable, synapses:list = []):
        """ Check and test axon forward callback function
        """
        if not inspect.ismethod(forward_callback) and not inspect.isfunction(forward_callback):
            raise ValueError('The axon forward callback must be a function with signature Callable[inputs_x: torch.Tensor] -> torch.FloatTensor:, got {}'.format(forward_callback))   
        if len( inspect.signature(forward_callback).parameters) != 3:
            raise ValueError('The axon forward callback must have signature Callable[ inputs_x: torch.Tensor, synapses, hotkey] -> torch.FloatTensor:, got {}'.format(inspect.signature(forward_callback)))
        if 'inputs_x' not in inspect.signature(forward_callback).parameters:
            raise ValueError('The axon forward callback must have signature Callable[ inputs_x: torch.Tensor] -> torch.FloatTensor:, got {}'.format(inspect.signature(forward_callback)))
        
        sample_input = torch.randint(0,1,(3, 3))
        forward_callback([sample_input], synapses, hotkey='')

class AuthInterceptor(grpc.ServerInterceptor):
    """Creates a new server interceptor that authenticates incoming messages from passed arguments."""

    def __init__(
        self,
        receiver_hotkey: str,
        blacklist: Callable = None,
    ):
        r"""Creates a new server interceptor that authenticates incoming messages from passed arguments.
        Args:
            receiver_hotkey(str):
                the SS58 address of the hotkey which should be targeted by RPCs
            black_list (Function, `optional`):
                black list function that prevents certain pubkeys from sending messages
        """
        super().__init__()
        self.nonces = {}
        self.blacklist = blacklist
        self.receiver_hotkey = receiver_hotkey


    def parse_signature_v2(
        self, signature: str
    ) -> Union[Tuple[int, str, str, str, int], None]:
        r"""Attempts to parse a signature using the v2 format"""
        parts = signature.split(".")
        if len(parts) != 4:
            return None
        try:
            nonce = int(parts[0])
        except ValueError:
            return None
        sender_hotkey = parts[1]
        signature = parts[2]
        receptor_uuid = parts[3]
        return (nonce, sender_hotkey, signature, receptor_uuid)

    def parse_signature(
        self, metadata: Dict[str, str]
    ) -> Tuple[int, str, str, str, int]:
        r"""Attempts to parse a signature from the metadata"""
        signature = metadata.get("bittensor-signature")
        version = metadata.get('bittensor-version')
        if signature is None:
            raise Exception("Request signature missing")
        if int(version) < 370:
            raise Exception("Incorrect Version")
        
        parts = self.parse_signature_v2(signature)
        if parts is not None:
            return parts
        raise Exception("Unknown signature format")

    def check_signature(
        self,
        nonce: int,
        sender_hotkey: str,
        signature: str,
        receptor_uuid: str,
    ):
        r"""verification of signature in metadata. Uses the pubkey and nonce"""
        keypair = Keypair(ss58_address=sender_hotkey)
        # Build the expected message which was used to build the signature.
        message = f"{nonce}.{sender_hotkey}.{self.receiver_hotkey}.{receptor_uuid}"

        # Build the key which uniquely identifies the endpoint that has signed
        # the message.
        endpoint_key = f"{sender_hotkey}:{receptor_uuid}"

        if endpoint_key in self.nonces.keys():
            previous_nonce = self.nonces[endpoint_key]
            # Nonces must be strictly monotonic over time.
            if nonce <= previous_nonce:
                raise Exception("Nonce is too small")

        if not keypair.verify(message, signature):
            raise Exception("Signature mismatch")
        self.nonces[endpoint_key] = nonce

    def black_list_checking(self, hotkey: str, method: str):
        r"""Tries to call to blacklist function in the miner and checks if it should blacklist the pubkey"""
        if self.blacklist == None:
            return

        request_type = {
            "/Bittensor/Forward": bittensor.proto.RequestType.FORWARD,
            "/Bittensor/Backward": bittensor.proto.RequestType.BACKWARD,
        }.get(method)
        if request_type is None:
            raise Exception("Unknown request type")

        failed, error_message =  self.blacklist(hotkey, request_type)
        if failed:
            raise Exception(str(error_message))


    def intercept_service(self, continuation, handler_call_details):
        r"""Authentication between bittensor nodes. Intercepts messages and checks them"""
        method = handler_call_details.method
        metadata = dict(handler_call_details.invocation_metadata)

        try:
            (
                nonce,
                sender_hotkey,
                signature,
                receptor_uuid,
            ) = self.parse_signature(metadata)

            # signature checking
            self.check_signature(
                nonce, sender_hotkey, signature, receptor_uuid
            )

            # blacklist checking
            self.black_list_checking(sender_hotkey, method)

            return continuation(handler_call_details)

        except Exception as e:
            message = str(e)
            abort = lambda _, ctx: ctx.abort(grpc.StatusCode.UNAUTHENTICATED, message)
            return grpc.unary_unary_rpc_method_handler(abort)
