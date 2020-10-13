import argparse
import yaml
import bittensor
from bittensor.metagraph import Metagraph
from loguru import logger
import time

metagraph = None

def start():
    logger.info("Starting metagraph on port {}".format(config.metagraph_port))
    metagraph.start()

def stop():
    metagraph.stop()

def main(config):
    global metagraph
    metagraph = bittensor.Metagraph( config )
    start()
    
    while (True):
        logger.info("nP: {}".format(len(metagraph.peers())))
        if len(metagraph.peers()):
            logger.info("Synapse found: {}".format(metagraph.synapses()))
        time.sleep(1)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = bittensor.Config.add_args(parser)
    config = parser.parse_args()
    main(config)
    
    