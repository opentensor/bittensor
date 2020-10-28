#!/bin/python3

from bittensor.config import Config
from bittensor.config import ConfigProcessor
import argparse
from loguru import logger

parser = argparse.ArgumentParser(description='BitTensor - Crypto P2P Neural Networks')

config = ConfigProcessor().create("../bittensor/config.ini", parser)
if not config:
    logger.error("Invalid configuration. Aborting")

config.log_config()



#-----

args = parser.parse_args()
print(args)