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
import wandb
import os

# Bittensor code and protocol version.
__version__ = '1.1.3'
__version_as_int__ = (100 * 2) + (10 * 0) + (1 * 0)  # Integer representation

# Vocabulary dimension.
#__vocab_size__ = len( tokenizer ) + len( tokenizer.additional_special_tokens) + 100 # Plus 100 for eventual token size increase.
__vocab_size__ = 50378

# Tensor dimension.
# NOTE (const): if/when this increases peers must be responsible for trimming or expanding output to this size.
__network_dim__ = 512 # All network responses have shape = [ __batch_size__, __sequence_dim__, __network_dim__ ]

# Substrate chain block time (seconds).
__blocktime__ = 6

# ---- Config ----
from bittensor._config import config as config

# ---- LOGGING ----
from bittensor._logging import logging as logging

# ---- Protos ----
import bittensor._proto.bittensor_pb2 as proto
import bittensor._proto.bittensor_pb2_grpc as grpc

# ---- Factories -----
from bittensor._cli import cli as cli
from bittensor._axon import axon as axon
from bittensor._wallet import wallet as wallet
from bittensor._receptor import receptor as receptor
from bittensor._endpoint import endpoint as endpoint
from bittensor._dendrite import dendrite as dendrite
from bittensor._executor import executor as executor
from bittensor._metagraph import metagraph as metagraph
from bittensor._subtensor import subtensor as subtensor
from bittensor._tokenizer import tokenizer as tokenizer
from bittensor._serializer import serializer as serializer
from bittensor._dataloader import dataloader as dataloader
from bittensor._receptor import receptor_pool as receptor_pool
from bittensor._wandb import wandb as wandb

# ---- Classes -----
from bittensor._cli.cli_impl import CLI as CLI
from bittensor._axon.axon_impl import Axon as Axon
from bittensor._config.config_impl import Config as Config
from bittensor._wallet.wallet_impl import Wallet as Wallet
from bittensor._receptor.receptor_impl import Receptor as Receptor
from bittensor._endpoint.endpoint_impl import Endpoint as Endpoint
from bittensor._executor.executor_impl import Executor as Executor
from bittensor._dendrite.dendrite_impl import Dendrite as Dendrite
from bittensor._metagraph.metagraph_impl import Metagraph as Metagraph
from bittensor._subtensor.subtensor_impl import Subtensor as Subtensor
from bittensor._serializer.serializer_impl import Serializer as Serializer
from bittensor._dataloader.dataloader_impl import Dataloader as Dataloader
from bittensor._receptor.receptor_pool_impl import ReceptorPool as ReceptorPool

import bittensor.utils.networking as net

# Singluar Neuron instance useful for creating simple miners.
neuron = None

def add_args( parser: argparse.ArgumentParser ):
    parser.add_argument('--neuron.use_upnpc', action='store_true', help='''Neuron punches a hole in your router using upnpc''', default=False)
    parser.add_argument('--neuron.use_wandb', action='store_true', help='''Neuron activates its weights and biases powers''', default=False)
    logging.add_args( parser )
    wallet.add_args( parser )
    subtensor.add_args( parser )
    metagraph.add_args( parser )
    dataloader.add_args( parser )
    dendrite.add_args( parser )
    axon.add_args( parser )
    wandb.add_args( parser )

def check_config( config ):
    logging.check_config( config )
    wallet.check_config( config )
    subtensor.check_config( config )
    metagraph.check_config( config )
    dataloader.check_config( config )
    dendrite.check_config( config )
    axon.check_config( config )

def default_config() -> 'Config':
    parser = argparse.ArgumentParser()
    add_args( parser )
    bittensor_config = config( parser )
    check_config(bittensor_config)
    return bittensor_config

class Neuron():

    def __init__(self, 
            config: 'Config',
            root_dir: str = '',
            axon_forward_callback: 'Callable' = None,
            axon_backward_callback: 'Callable' = None,
        ):
        if config == None: config = default_config()
        self.config = config
        self.root_dir = root_dir
        logging (
            config = self.config,
            logging_dir = root_dir,
        )
        self.wallet = wallet(
            config = self.config
        )
        self.dendrite = dendrite(
            config = self.config,
            wallet = self.wallet
        )
        self.subtensor = subtensor(
            config = self.config
        )
        self.metagraph = metagraph(
            config = self.config,
            subtensor = self.subtensor
        )
        self.axon = axon (
            config = self.config,
            wallet = self.wallet,
            forward_callback=axon_forward_callback,
            backward_callback=axon_backward_callback,
        )

    def __enter__(self):
        # ---- Setup UPNPC ----
        if self.config.neuron.use_upnpc:
            logging.success(prefix = 'Set upnpc', sufix = '<green>ON</green>')
            try:
                self.external_port = net.upnpc_create_port_map( port = self.axon.port )
            except net.UPNPCException as upnpc_exception:
                raise RuntimeError('Failed to hole-punch with upnpc')
        else:
            logging.success(prefix = 'Set upnpc', sufix = '<red>OFF</red>')
            self.external_port = self.config.axon.port

        # ---- Get external ip ----
        try:
            self.external_ip = net.get_external_ip()
            logging.success(prefix = 'External IP', sufix = '<blue>{}</blue>'.format(self.external_ip))
        except Exception as E:
            raise RuntimeError('Unable to attain your external ip. Check your internet connection. error:{}', E)

        # ---- Setup Wallet. ----
        if not self.wallet.has_coldkeypub:
            self.wallet.create_new_coldkey( n_words = 12, use_password = True )
        if not self.wallet.has_coldkeypub:
            raise RuntimeError('Miner must have access to a decrypted coldkeypub')
        if not self.wallet.has_hotkey:
            self.wallet.create_new_hotkey( n_words = 12, use_password = False )
        if not self.wallet.has_hotkey:
            raise RuntimeError('Miner must have access to a decrypted hotkey')

        # ---- Subscribe to chain ----
        subscribe_success = self.subtensor.subscribe(
                wallet = self.wallet,
                ip = self.external_ip,
                port = self.external_port,
                modality = proto.Modality.TEXT,
                wait_for_finalization = True,
                timeout = 4 * __blocktime__,
        )
        
        if not subscribe_success:
            raise RuntimeError('Failed to subscribe neuron.')

        # --- Update metagraph ----
        self.metagraph.load()
        self.metagraph.sync()

        # ---- Starting axon ----
        self.axon.start()

        # --- Init wandb ----
        if self.config.neuron.use_wandb:
            self.wandb = wandb(
                config = self.config,
                cold_pubkey = self.wallet.coldkeypub,
                hot_pubkey = self.wallet.hotkey.public_key,
                root_dir = self.root_dir
            )
            
    def __exit__ ( self, exc_type, exc_value, exc_traceback ):
        self.axon.stop()
        if self.config.neuron.use_wandb:
            self.wandb.finish()
            
        print(exc_type, exc_value, exc_traceback)

def init( 
        config: 'Config' = None,
        root_dir: str = None,
        axon_forward_callback: 'Callable' = None,
        axon_backward_callback: 'Callable' = None,
    ) -> Neuron:

    global neuron
    neuron = Neuron( 
        config = config,
        root_dir = root_dir,
        axon_forward_callback = axon_forward_callback,
        axon_backward_callback = axon_backward_callback,
    )
    return neuron

# Hardcoded entry point nodes. 
__kusanagi_entrypoints__ = [
    "test.kusanagi.bittensor.com:9944" 
]

__akatsuki_entrypoints__ = [
    "main.akatsuki.bittensor.com:9944"
]

__local_entrypoints__ = [
    '127.0.0.1:9944'
]
