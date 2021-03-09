import sys
import random
from loguru import logger

import bittensor.bittensor_pb2 as proto
import bittensor.bittensor_pb2_grpc as grpc

# Bittensor code and protocol version.
__version__ = '1.0.4'

# Tensor dimension.
# NOTE (const): if/when this increases peers must be responsible for trimming or expanding output to this size.
__network_dim__ = 512 # All network responses have shape = [ __batch_size__, __sequence_dim__, __network_dim__ ]

# Substrate chain block time (seconds).
__blocktime__ = 6

# Load components.
import bittensor.axon
import bittensor.config 
import bittensor.executor
import bittensor.dendrite
import bittensor.metagraph
import bittensor.nucleus
import bittensor.receptor
import bittensor.substrate
import bittensor.subtensor
import bittensor.wallet

# Default logger
logger_config = {
    "handlers": [{
        "sink":
            sys.stdout,
        "format":
            "<level>{level: <8}</level>|<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    }]
}
logger.configure(**logger_config)

# Tokenizer
# NOTE (const): tokenizers are guaranteed to improve and expand as time progresses. We version the tokenizer here.
# neurons must be aware that versions will increase and be ready to convert between tokenizers.
# TODO (const): Add functionality to allow tokenizer conversion. i.e. for input token conversion.
__vocab_size__ = (50278 + 100)  # Must match the __tokenizer__() vocab size.
def __tokenizer__(  version = __version__ ):
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=False)
    tokenizer.padding_side = "left"
    tokenizer.add_prefix_space = False
    tokenizer.add_special_tokens({'bos_token': "[BOS]"}) # A special token representing the beginning of a sentence.
    tokenizer.add_special_tokens({'eos_token': "[EOS]"}) # A special token representing the end of a sentence.
    tokenizer.add_special_tokens({'unk_token': "[UNK]"}) # A special token representing an out-of-vocabulary token.
    tokenizer.add_special_tokens({'sep_token': "[SEP]"}) # A special token separating two different sentences in the same input (used by BERT for instance)
    tokenizer.add_special_tokens({'pad_token': "[PAD]"}) # A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by attention mechanisms or loss computation.
    tokenizer.add_special_tokens({'cls_token': "[CLS]"}) # A special token representing the class of the input (used by BERT for instance).
    tokenizer.add_special_tokens({'mask_token': "[MASK]"}) # A special token representing a masked token (used by masked-language modeling pretraining objectives, like BERT).
    additional_special_tokens = [
        "<s>NOTUSED",  # Used by BARThez
        "</s>NOTUSED", # Used by BARThez
        "<eop>", # Used by MarianMT
        "<eod>", # Used by MarianMT
        "<formula>", # Used by Transformer XL
        "<mask_1>" # Used by Pegasus
        "<special0>", # Used by XLM
        "<special1>", # Used by XLM
        "<special2>", # Used by XLM
        "<special3>", # Used by XLM
        "<special4>", # Used by XLM
        "<special5>", # Used by XLM
        "<special6>", # Used by XLM
        "<special7>", # Used by XLM
        "<special8>", # Used by XLM
        "<special9>", # Used by XLM
    ]
    tokenizer.additional_special_tokens = additional_special_tokens
    global __vocab_size__
    __vocab_size__ = len(tokenizer) + len(additional_special_tokens) + 100 # Plus 100 for eventual token size increase.

    return tokenizer

# Hardcoded entry point nodes. 
__akira_entrypoints__ = [
    '104.248.52.148:9944',
    '142.93.194.110:9944',
    '162.243.175.73:9944',
    '165.227.92.237:9944',
    '167.172.141.223:9944',
    '174.138.32.166:9944',
    '206.189.194.236:9944',
    '68.183.130.145:9944',
    '68.183.140.221:9944',
    '68.183.140.251:9944'
]
__kusanagi_entrypoints__ = [
    '142.93.203.149:9944',
    '157.230.11.1:9944',
    '157.230.11.116:9944',
    '157.230.11.31:9944',
    '157.230.11.36:9944',
    '157.230.11.53:9944',
    '157.230.3.108:9944',
    '159.65.236.189:9944',
    '165.227.81.42:9944',
    '206.189.207.173:9944'
]
__boltzmann_entrypoints__ = [
    'feynman.boltzmann.bittensor.com:9944',
    '157.230.223.68:9944'
]
__local_entrypoints__ = [
    '127.0.0.1:9944'
]

from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager, NamespaceProxy

import argparse
import json
import os
import re
import stat
import types
import traceback as tb

from io import StringIO
from munch import Munch
from termcolor import colored
from loguru import logger

class _DendriteProxy(NamespaceProxy):
    _exposed_ = tuple(dir(bittensor.dendrite.Dendrite))

    def __getattr__(self, name):
        result = super().__getattr__(name)
        if isinstance(result, types.MethodType):
            def wrapper(*args, **kwargs):
                self._callmethod(name, args)
            return wrapper
        return result


class _SubtensorProxy(NamespaceProxy):
    _exposed_ = tuple(dir(bittensor.subtensor.Subtensor))

    def __getattr__(self, name):
        result = super().__getattr__(name)
        if isinstance(result, types.MethodType):
            def wrapper(*args, **kwargs):
                self._callmethod(name, args)
            return wrapper
        return result


class _MetagraphProxy(NamespaceProxy):
    _exposed_ = tuple(dir(bittensor.metagraph.Metagraph))

    def __getattr__(self, name):
        result = super().__getattr__(name)
        if isinstance(result, types.MethodType):
            def wrapper(*args, **kwargs):
                self._callmethod(name, args)
            return wrapper
        return result


class _AxonProxy(NamespaceProxy):
    _exposed_ = tuple(dir(bittensor.axon.Axon))

    def __getattr__(self, name):
        result = super().__getattr__(name)
        if isinstance(result, types.MethodType):
            def wrapper(*args, **kwargs):
                self._callmethod(name, args)
            return wrapper
        return result


neuron = None
class Neuron:

    def __init__( self, config: Munch = None,  wallet: 'bittensor.wallet.Wallet' = None, **kwargs ):
        if config == None:
            config = Neuron.default_config()
        bittensor.config.Config.update_with_kwargs(config.neuron, kwargs) 
        Neuron.check_config(config)
        self.config = config
        print ( bittensor.config.Config.toString(config) )


        # Wallet: Holds the hotkey keypair and coldkey pub which are user to sign messages 
        # and subscribe to the chain.
        if wallet == None:
            wallet = bittensor.wallet.Wallet ( config )
        self.wallet = wallet

        BaseManager.register('Subtensor', bittensor.subtensor.Subtensor, _SubtensorProxy)
        BaseManager.register('Metagraph', bittensor.metagraph.Metagraph, _MetagraphProxy)
        BaseManager.register('Dendrite', bittensor.dendrite.Dendrite, _DendriteProxy)
        BaseManager.register('Axon', bittensor.axon.Axon, _AxonProxy)

        manager = BaseManager()
        manager.start()
        
        self.subtensor = manager.Subtensor( config = self.config, wallet = self.wallet )
        self.metagraph = manager.Metagraph( config = self.config, wallet = self.wallet )
        self.dendrite = manager.Dendrite( config = self.config, walelt = self.wallet )
        self.axon = manager.Axon( config = self.config, wallet = self.wallet )

    @staticmethod       
    def default_config() -> Munch:
        parser = argparse.ArgumentParser(); 
        Neuron.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        return config

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        bittensor.wallet.Wallet.add_args( parser )
        bittensor.subtensor.Subtensor.add_args( parser )
        bittensor.metagraph.Metagraph.add_args( parser )
        bittensor.axon.Axon.add_args(parser)
        bittensor.dendrite.Dendrite.add_args( parser )
        try:
            parser.add_argument('--neuron.modality', default=0, type=int, 
                                help='''Neuron network modality. TEXT=0, IMAGE=1. Currently only allowed TEXT''')
        except:
            pass

    @staticmethod   
    def check_config(config: Munch):
        bittensor.axon.Axon.check_config( config )
        bittensor.subtensor.Subtensor.check_config( config )
        bittensor.metagraph.Metagraph.check_config( config )
        bittensor.dendrite.Dendrite.check_config( config )
        assert config.neuron.modality == bittensor.proto.Modality.TEXT, 'Only TEXT modalities are allowed at this time.'

    def start(self):
        print(colored('', 'white'))
        # ---- Check hotkey ----
        print(colored('Loading wallet with path: {} name: {} hotkey: {}'.format(self.config.wallet.path, self.config.wallet.name, self.config.wallet.hotkey), 'white'))
        try:
            self.wallet.hotkey # Check loaded hotkey
        except:
            logger.info('Failed to load hotkey under path:{} wallet name:{} hotkey:{}', self.config.wallet.path, self.config.wallet.name, self.config.wallet.hotkey)
            choice = input("Would you like to create a new hotkey ? (y/N) ")
            if choice == "y":
                self.wallet.create_new_hotkey()
            else:
                raise RuntimeError('The neuron requires a loaded hotkey')

        # ---- Check coldkeypub ----
        try:
            self.wallet.coldkeypub
        except:
            logger.info('Failed to load coldkeypub under path:{} wallet name:{}', self.config.wallet.path, self.config.wallet.name)
            choice = input("Would you like to create a new coldkey ? (y/N) ")
            if choice == "y":
                self.wallet.create_new_coldkey()
            else:
                raise RuntimeError('The neuron requires a loaded coldkeypub')

        # ---- Start the axon ----
        self.axon.start()

        # ---- Subscribe to chain ----
        print(colored('\nConnecting to network: {}'.format(self.config.subtensor.network), 'white'))
        self.subtensor._callmethod('connect')

        print(colored('\nSubscribing:', 'white'))
        subscribe_success = self.subtensor.subscribe(
                self.config.axon.external_ip, 
                self.config.axon.external_port,
                self.config.neuron.modality,
                self.wallet.coldkeypub,
                wait_for_finalization = True,
                timeout = 4 * bittensor.__blocktime__,
        )
        if not subscribe_success:
            raise RuntimeError('Failed to subscribe neuron.')
        

import asyncio
def init( config: Munch = None,  wallet: 'bittensor.wallet.Wallet' = None, **kwargs ):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    global neuron
    neuron = Neuron(config, wallet)


# def start(self):
#     print(colored('', 'white'))
#     # ---- Check hotkey ----
#     print(colored('Loading wallet with path: {} name: {} hotkey: {}'.format( config.wallet.path, config.wallet.name, config.wallet.hotkey), 'white'))
#     try:
#         wallet.hotkey # Check loaded hotkey
#     except:
#         logger.info('Failed to load hotkey under path:{} wallet name:{} hotkey:{}', config.wallet.path, config.wallet.name, config.wallet.hotkey)
#         choice = input("Would you like to create a new hotkey ? (y/N) ")
#         if choice == "y":
#             wallet.create_new_hotkey()
#         else:
#             raise RuntimeError('The neuron requires a loaded hotkey')

#     # ---- Check coldkeypub ----
#     try:
#         wallet.coldkeypub
#     except:
#         logger.info('Failed to load coldkeypub under path:{} wallet name:{}', config.wallet.path, config.wallet.name)
#         choice = input("Would you like to create a new coldkey ? (y/N) ")
#         if choice == "y":
#             wallet.create_new_coldkey()
#         else:
#             raise RuntimeError('The neuron requires a loaded coldkeypub')

#     # ---- Start the axon ----
#     axon.start()

#     # ---- Subscribe to chain ----
#     print(colored('\nConnecting to network: {}'.format(config.subtensor.network), 'white'))
#     subtensor.connect()

#     print(colored('\nSubscribing:', 'white'))
#     subscribe_success = subtensor.subscribe(
#             config.axon.external_ip, 
#             config.axon.external_port,
#             config.neuron.modality,
#             wallet.coldkeypub,
#             wait_for_finalization = True,
#             timeout = 4 * bittensor.__blocktime__,
#     )
#     if not subscribe_success:
#         raise RuntimeError('Failed to subscribe neuron.')
    
#     # ---- Sync graph ----
#     metagraph.sync()
#     print( metagraph )

    # def stop(self):

    #     logger.info('Shutting down the Axon server ...')
    #     try:
    #         self.axon.stop()
    #         logger.info('Axon server stopped')
    #     except Exception as e:
    #         logger.error('Neuron: Error while stopping axon server: {} ', e)

    # def __enter__(self):
    #     bittensor.exceptions.handlers.rollbar.init() # If a bittensor.exceptions.handlers.rollbar token is present, this will enable error reporting to bittensor.exceptions.handlers.rollbar
    #     logger.trace('Neuron enter')
    #     self.start()
    #     return self

    # def __exit__(self, exc_type, exc_value, exc_traceback):
    #     """ Defines the exit protocol from asyncio task.
    #     Args:
    #         exc_type (Type): The type of the exception.
    #         exc_value (RuntimeError): The value of the exception, typically RuntimeError. 
    #         exc_traceback (traceback): The traceback that can be printed for this exception, detailing where error actually happend.
    #     Returns:
    #         Neuron: present instance of Neuron.
    #     """        
    #     self.stop()
    #     if exc_value:

    #         top_stack = StringIO()
    #         tb.print_stack(file=top_stack)
    #         top_lines = top_stack.getvalue().strip('\n').split('\n')[:-4]
    #         top_stack.close()

    #         full_stack = StringIO()
    #         full_stack.write('Traceback (most recent call last):\n')
    #         full_stack.write('\n'.join(top_lines))
    #         full_stack.write('\n')
    #         tb.print_tb(exc_traceback, file=full_stack)
    #         full_stack.write('{}: {}'.format(exc_type.__name__, str(exc_value)))
    #         sinfo = full_stack.getvalue()
    #         full_stack.close()
    #         # Log the combined stack
    #         logger.error('Exception:{}'.format(sinfo))

    #         if bittensor.exceptions.handlers.rollbar.is_enabled():
    #             bittensor.exceptions.handlers.rollbar.send_exception()

    #     return self

    # def __del__(self):
    #     self.stop()
