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
from prometheus_client import start_http_server

from loguru import logger
logger = logger.opt(colors=True)

class prometheus:
    """ Namespace for prometheus tooling.
    """

    # Prometheus Global state.
    port: int = None
    started: bool = False

    def __new__( 
        cls,
        config: 'bittensor.config' = None,
        port: int = None,
        off: bool = None
    ):
        """ Intantiates a global prometheus DB which can be accessed by other processes.
            Each prometheus DB is designated by a port.
            Args:
                config (:obj:`bittensor.Config`, `optional`, defaults to bittensor.prometheus.config()):
                    A config namespace object created by calling bittensor.prometheus.config()
                port (:obj:`int`, `optional`, defaults to bittensor.defaults.prometheus.port ):
                    The port to run the prometheus DB on, this uniquely identifies the prometheus DB.
                off (:obj:`bool`, `optional`, defaults to bittensor.defaults.prometheus.off ):
                    If true, turns of global prometheus logging.
        """
        if config == None:
            config = prometheus.config()
        config.prometheus.port = port if port != None else config.prometheus.port
        config.prometheus.off = off if off != None else config.prometheus.off
        if not config.prometheus.off:
            try:
                start_http_server( config.prometheus.port )
            except OSError:
                # The singleton process is likely already running.
                logger.error( "Prometheus:".ljust(20) + "<blue>{}</blue>  <red>already in use</red> ".format( config.prometheus.port ) )
                return
            prometheus.started = True
            prometheus.port = config.prometheus.port
            logger.success( "Prometheus:".ljust(20) + "<green>ON</green>".ljust(20) + "using: <blue>[::]:{}</blue>".format( config.prometheus.port ))
        else:
            logger.success('Prometheus:'.ljust(20) + '<red>OFF</red>')


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
            parser.add_argument('--' + prefix_str + 'prometheus.port', required=False, default=bittensor.defaults.prometheus.port, help='''Prometheus serving port.''')
            parser.add_argument('--' + prefix_str + 'prometheus.off', action='store_true', required=False, default=bittensor.defaults.prometheus.off, help='''If true, the prometheus server will not start''')
        except argparse.ArgumentError as e:
            pass

    @classmethod   
    def add_defaults(cls, defaults):
        """ Adds parser defaults to object from enviroment variables.
        """
        defaults.prometheus = bittensor.Config()
        # Default the prometheus port to axon.port - 1000
        defaults.prometheus.port = os.getenv('BT_PROMETHEUS_PORT') if os.getenv('BT_PROMETHEUS_PORT') != None else 7091
        defaults.prometheus.off = os.getenv('BT_PROMETHEUS_OFF') if os.getenv('BT_PROMETHEUS_OFF') != None else False

    @classmethod   
    def check_config(cls, config: 'bittensor.Config' ):
        """ Check config for wallet name/hotkey/path/hotkeys/sort_by
        """
        assert 'prometheus' in config
        assert isinstance(config.prometheus.port, int), 'config.prometheus.port must be an integer'
        assert isinstance(config.prometheus.off, bool), 'config.prometheus.off must be an boolean'
        assert config.prometheus.port > 1024 and config.prometheus.port < 65535, 'config.prometheus.port must be in range [1024, 65535]'
        if "axon" in config and "port" in config.axon:
            assert config.prometheus.port != config.axon.port, 'config.prometheus.port != config.axon.port'


