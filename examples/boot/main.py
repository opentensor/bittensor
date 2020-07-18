from loguru import logger
import sys
import argparse
import requests
import time
from torch.utils.tensorboard import SummaryWriter

from opentensor import opentensor_pb2
import opentensor


def main(hparams):
    # Null identity
    identity = opentensor.Identity()

    # Build object summary writer.
    writer = SummaryWriter(log_dir='./runs/' + identity.public_key())
    
    remote_ip = requests.get('https://api.ipify.org').text

    # Metagraph: maintains a cache of synapses on the network.
    metagraph = opentensor.Metagraph(identity,
                                    max_size = hparams.size,
                                    port = hparams.port,
                                    remote_ip = remote_ip,
                                    bootstrap = hparams.bootstrap)
    metagraph.start()
    logger.info('Serving on {}:{}', remote_ip, hparams.port)
    
    while True:
        writer.add_scalar('n_peers', len(metagraph.peers), time.time())
        writer.add_scalar('n_synapses', len(metagraph.synapses), time.time())
        logger.info('n_peers {}', len(metagraph.peers))
        logger.info('n_synapses {}', len(metagraph.synapses))
        try:
            time.sleep(30)
        except (KeyboardInterrupt, SystemExit):
            metagraph.stop()
    del metagraph


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap',
                        default='',
                        type=str,
                        help="peer to bootstrap")
    parser.add_argument('--port',
                        default='8080',
                        type=str,
                        help="port to bind on.")
    parser.add_argument('--size',
                        default=1000000,
                        type=int,
                        help="cache size")
    hparams = parser.parse_args()
    main(hparams)
