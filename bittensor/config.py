from loguru import logger

import argparse
import requests
import random

import bittensor
from bittensor.crypto import Crypto

class Config(object):
    r"""Base config for all neuron objects.
    Handles a parameters common to all bittensor neuron configurations.

    Args:
        chain_endpoint (:obj:`str`, `optional`, defaults to :obj:`None`):
            Bittensor network chain endpoint.
        axon_port (:obj:`int`, `optional`, defaults to :obj:`8091`):
            Axon terminal bind port.
        metagraph_port (:obj:`int`, `optional`, defaults to :obj:`8092`):
            Metagraph bind port (for gossip)
        metagraph_size (:obj:`int`, `optional`, defaults to :obj:`10000`):
            Size of metagraph synapse cache. Prunes network over this size.
        bootstrap (:obj:`str`, `optional`, defaults to :obj:`None`):
            Optional bootpeer for the metagraph object.
        neuron_key (:obj:`ed25519 key`, `optional`, defaults to :obj:`ed25519`):
            Cryptographic keys used by this neuron. Defaults to randomly generated ed25519 key.
        remote_ip (:obj:`str`, `optional`, defaults to :obj:`None`):
            Default serving endpoing (IP Address) advertised on the network. Defaults to returned IP on: https://api.ipify.org
        datapath (:obj:`str`, `optional`, defaults to :obj:`data/`)
            Location of save models.
    """
    __chainendpoint_default__ = ""
    __axon_port_default__ = "8091"
    __metagraph_port_default__ = "8092"
    __metagraph_size_default__ = 10000
    __bootstrap_default__ = ""
    __neuron_key_default__ = Crypto.public_key_to_string(Crypto.generate_private_ed25519().public_key())
    __remote_ip_default__ = requests.get('https://api.ipify.org').text
    __datapath_default__ = "data/"

    def __init__(self, **kwargs):
        # Bittensor chain endpoint.
        self.chain_endpoint = kwargs.pop("chain_endpoint", Config.__chainendpoint_default__)
        # Axon terminal bind port
        self.axon_port = kwargs.pop("axon_port", Config.__axon_port_default__)
        #  Metagraph bind port.
        self.metagraph_port = kwargs.pop("metagraph_port", Config.__metagraph_port_default__)
        # Metagraph cache size.
        self.metagraph_size = kwargs.pop("metagraph_size", Config.__metagraph_size_default__)  # Not used by all models
        # Peer to bootstrap 
        self.bootstrap = kwargs.pop("bootstrap", Config.__bootstrap_default__)  
        # Unique Ed255 key
        self.neuron_key = kwargs.pop("neuron_key", Config.__neuron_key_default__)
        # Default serving endpoing (IP Address) advertised on the network.
        self.remote_ip = kwargs.pop("remote_ip", Config.__remote_ip_default__)
        # Path to a saved torch model.
        self.datapath= kwargs.pop("datapath", Config.__datapath_default__)  

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err

        self.run_type_checks()

    def run_type_checks(self):
        assert  isinstance(self.chain_endpoint, type(bittensor.Config.__chainendpoint_default__))
        assert  isinstance(self.axon_port, type(bittensor.Config.__axon_port_default__))
        assert  isinstance(self.metagraph_port, type(bittensor.Config.__metagraph_port_default__))
        assert  isinstance(self.metagraph_size, type(bittensor.Config.__metagraph_size_default__))
        assert  isinstance(self.neuron_key, type(bittensor.Config.__neuron_key_default__))
        assert  isinstance(self.remote_ip, type(bittensor.Config.__remote_ip_default__))
        assert  isinstance(self.datapath, type(bittensor.Config.__datapath_default__))

    def __str__(self):
        return "\n Neuron key: {} \n Axon port: {} \n Metagraph port: {} \n Metagraph Size: {} \n bootpeer: {} \n remote_ip: {} \n".format(self.neuron_key, self.axon_port, self.metagraph_port, self.metagraph_size, self.bootstrap, self.remote_ip)

    def from_hparams(hparams):
        config = Config()
        config.set_from_hparams(hparams)
        return config

    def set_from_hparams(self, hparams):
        for key, value in hparams.__dict__.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err
        self.run_type_checks()

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument('--chain_endpoint',
                        default=Config.__chainendpoint_default__,
                        type=str,
                        help="bittensor chain endpoint.")
        parser.add_argument('--axon_port',
                        default=Config.__axon_port_default__,
                        type=str,
                        help="Axon terminal bind port")
        parser.add_argument('--metagraph_port',
                        default=Config.__metagraph_port_default__,
                        type=str,
                        help='Metagraph bind port.')
        parser.add_argument('--metagraph_size',
                            default=Config.__metagraph_size_default__,
                            type=int,
                            help='Metagraph cache size.')
        parser.add_argument('--bootstrap',
                            default=Config.__bootstrap_default__,
                            type=str,
                            help='Metagraph bootpeer')
        parser.add_argument('--neuron_key',
                            default=Config.__neuron_key_default__,
                            type=str,
                            help='Neuron key')
        parser.add_argument('--remote_ip',
                            default=Config.__remote_ip_default__,
                            type=str,
                            help='Remote serving ip.')
        parser.add_argument('--datapath',
                            default=Config.__datapath_default__,
                            type=str,
                            help='Path to a saved torch model.')
        return parser

