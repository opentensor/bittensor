from loguru import logger
import bittensor
import time
import argparse

class MetagraphLauncher():
    meta = None
    axon = None
    def __init__(self, arguments):
        logger.info('Loading Metagraph on port {} and axon port {}...'.format(arguments.metagraph_port, arguments.axon_port))
        config = bittensor.Config(  axon_port = arguments.axon_port,
                                    metagraph_port = arguments.metagraph_port)
        logger.info('config: {}', config)
        self.meta = bittensor.Metagraph(config)
        self.axon = bittensor.Axon(config)

    def start(self):
        self.meta.start()
        self.axon.start()

    def stop(self):
        self.meta.stop()
        self.axon.stop()
    
    def is_running(self):
        return self.meta._running

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metagraph_port',
                            default="8092",
                            type=str,
                            help='Metagraph bind port.')
    parser.add_argument('--axon_port',
                            default="8091",
                            type=str,
                            help='Axon bind port.')

    mgl = MetagraphLauncher(parser.parse_args())
    mgl.start()

    while mgl.is_running():
        logger.info("Peers: {}".format(mgl.meta.peers()))
        
