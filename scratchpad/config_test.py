#!/bin/python3

from config import Config
import argparse
from loguru import logger

parser = argparse.ArgumentParser(description='BitTensor - Crypto P2P Neural Networks')



config = Config("../bittensor/config.ini", parser)
if not config.isValid():
    logger.error("Invalid configuration. Aborting")

config.log_config()



#-----

args = parser.parse_args()
print(args)