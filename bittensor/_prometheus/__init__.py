"""
Create and init the config class, which manages the config of different bittensor modules.
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

import os
import argparse
import bittensor
from typing import List, Callable, Union
from prometheus_client import start_http_server
from enum import Enum

from loguru import logger
logger = logger.opt(colors=True)


class prometheus:
    """ Namespace for prometheus tooling.
    """

    # Prometheus global logging levels.
    class level ( Enum ):
        OFF = "OFF"
        INFO = "INFO"
        DEBUG = "DEBUG"
        def __str__(self):
            return self.value

    # Prometheus Global state.
    port: int = None
    started: bool = False

    def __new__( 
        cls,
        wallet: 'bittensor.wallet',
        netuid: int,
        config: 'bittensor.config' = None,
        port: int = None,
        level: Union[str, "prometheus.level"] = None,
        network: str = None,
        chain_endpoint: str = None,
        subtensor: 'bittensor.subtensor' = None,
    ):
        """ Instantiates a global prometheus DB which can be accessed by other processes.
            Each prometheus DB is designated by a port.
            Args:
                wallet (:obj: `bittensor.wallet`, `required`):
                    bittensor wallet object.
                netuid (:obj: `int`, `required`):
                    network uid to serve on.
                config (:obj:`bittensor.Config`, `optional`, defaults to bittensor.prometheus.config()):
                    A config namespace object created by calling bittensor.prometheus.config()
                port (:obj:`int`, `optional`, defaults to bittensor.defaults.prometheus.port ):
                    The port to run the prometheus DB on, this uniquely identifies the prometheus DB.
                level (:obj:`prometheus.level`, `optional`, defaults to bittensor.defaults.prometheus.level ):
                    Prometheus logging level. If OFF, the prometheus DB is not initialized.
                subtensor (:obj:`bittensor.Subtensor`, `optional`): 
                    Chain connection through which to serve.
                network (default='local', type=str)
                    If subtensor is not set, uses this network flag to create the subtensor connection.
                chain_endpoint (default=None, type=str)
                    Overrides the network argument if not set.
        """
        if config == None:
            config = prometheus.config()

        if isinstance(level, prometheus.level):
            level = level.name # Convert ENUM to str.

        if subtensor == None: subtensor = bittensor.subtensor( network = network, chain_endpoint = chain_endpoint) 
        
        config.prometheus.port = port if port != None else config.prometheus.port
        config.prometheus.level = level if level != None else config.prometheus.level

        if isinstance(config.prometheus.level, str):
            config.prometheus.level = config.prometheus.level.upper() # Convert str to upper case.
        
        cls.check_config( config )

        return cls.serve(
            cls,
            wallet = wallet,
            netuid = netuid,
            subtensor = subtensor,
            port = config.prometheus.port,
            level = config.prometheus.level,
        )
        
    def serve(cls, wallet, subtensor, netuid, port, level) -> bool:
        if level == prometheus.level.OFF.name: # If prometheus is off, return true.
            logger.success('Prometheus:'.ljust(20) + '<red>OFF</red>')
            return True
        else:
            # Serve prometheus. Not OFF
            serve_success = subtensor.serve_prometheus(
                wallet = wallet,
                port = port,
                netuid = netuid,
            )
            if serve_success:
                try:
                    start_http_server( port )
                except OSError:
                    # The singleton process is likely already running.
                    logger.error( "Prometheus:".ljust(20) + "<blue>{}</blue>  <red>already in use</red> ".format( port ) )
                prometheus.started = True
                prometheus.port = port
                logger.success( "Prometheus:".ljust(20) + "<green>ON</green>".ljust(20) + "using: <blue>[::]:{}</blue>".format( port ))
                return True
            else:
                logger.error('Prometheus:'.ljust(20) + '<red>OFF</red>')
                raise RuntimeError('Failed to serve neuron.')

    @classmethod
    def config(cls) -> 'bittensor.Config':
        """ Get config from the argument parser
        Return: bittensor.config object
        """
        parser = argparse.ArgumentParser()
        cls.add_args(parser=parser)
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
    def add_args(cls, parser: argparse.ArgumentParser, prefix: str = None ):
        """ Accept specific arguments from parser
        """
        prefix_str = '' if prefix == None else prefix + '.'
        try:
            parser.add_argument('--' + prefix_str + 'prometheus.port',  type=int, required=False, default = bittensor.defaults.prometheus.port, 
                help='''Prometheus serving port.''')
            parser.add_argument(
                '--' + prefix_str + 'prometheus.level', 
                required = False,
                type = str, 
                choices = [l.name for l in list(prometheus.level)],
                default = bittensor.defaults.prometheus.level, 
                help = '''Prometheus logging level. <OFF | INFO | DEBUG>''')
        except argparse.ArgumentError as e:
            pass

    @classmethod
    def add_defaults(cls, defaults):
        """ Adds parser defaults to object from enviroment variables.
        """
        defaults.prometheus = bittensor.Config()
        # Default the prometheus port to axon.port - 1000
        defaults.prometheus.port = os.getenv('BT_PROMETHEUS_PORT') if os.getenv('BT_PROMETHEUS_PORT') != None else 7091
        defaults.prometheus.level = os.getenv('BT_PROMETHEUS_LEVEL') if os.getenv('BT_PROMETHEUS_LEVEL') != None else bittensor.prometheus.level.INFO.value

    @classmethod
    def check_config(cls, config: 'bittensor.Config' ):
        """ Check config for wallet name/hotkey/path/hotkeys/sort_by
        """
        assert 'prometheus' in config
        assert config.prometheus.level in [l.name for l in list(prometheus.level)], "config.prometheus.level must be in: {}".format([l.name for l in list(prometheus.level)])
        assert config.prometheus.port > 1024 and config.prometheus.port < 65535, 'config.prometheus.port must be in range [1024, 65535]'
        if "axon" in config and "port" in config.axon:
            assert config.prometheus.port != config.axon.port, 'config.prometheus.port != config.axon.port'
