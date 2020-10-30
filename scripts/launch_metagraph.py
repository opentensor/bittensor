from loguru import logger
import bittensor
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metagraph_port',
                            default="8120",
                            type=str,
                            help='Metagraph bind port.')
    parser.add_argument('--axon_port',
                            default="8122",
                            type=str,
                            help='Metagraph bind port.')
                            

    mgl = MetagraphLauncher(parser.parse_args())
    mgl.start()

    while True:
        continue
        logger.info("I see {} connections. Performing Gossip...".format(len(mgl.meta.peers())))
        
        if len(mgl.meta.peers()) > 0:
            logger.info("Connected neurons: {}".format(mgl.meta.peers()))
        
        #time.sleep(5)