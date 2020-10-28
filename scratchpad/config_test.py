#!/bin/python3

from scratchpad.config import Config
import argparse
from loguru import logger

parser = argparse.ArgumentParser(description='BitTensor - Crypto P2P Neural Networks')
parser.add_argument('--chain_endpoint', dest=Config.CHAIN_ENDPOINT, type=str, help="bittensor chain endpoint")
parser.add_argument('--axon_port', dest=Config.AXON_PORT, type=int, help="TCP port that will be used to receive axon connections")
parser.add_argument('--metagraph_port', dest=Config.METAGRAPH_PORT, type=int, help='TCP port that will be used to receive metagraph connections')
parser.add_argument('--metagraph_size', dest=Config.METAGRAPH_SIZE, type=int, help='Metagraph cache size')
parser.add_argument('--bp_host', dest=Config.BOOTPEER_HOST, type=str, help='Hostname or IP of the first peer this neuron should connect to when signing onto the network. <IPv4:port>')
parser.add_argument('--np_port', dest=Config.BOOTPEER_PORT, type=int, help='TCP Port the bootpeer is listening on')
parser.add_argument('--neuron_key', dest=Config.NEURON_ID, type=str, help='Key of the neuron')
parser.add_argument('--ip', dest=Config.IP, type=str, help='The IP address of this neuron that will be published to the network')
parser.add_argument('--datapath', dest=Config.DATAPATH, type=str, help='Path to datasets')
parser.add_argument('--logdir', dest=Config.LOGPATH, type=str, help='Path to logs and saved models')

args = parser.parse_args()

config = Config(args)
if not config.isValid():
    logger.error("Invalid configuration. Aborting")

config.log_config()

