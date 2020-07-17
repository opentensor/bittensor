from loguru import logger
import sys
import argparse
import time
import torch
from torch.utils.tensorboard import SummaryWriter

from opentensor import opentensor_pb2
import opentensor


def main(hparams):
    # Null identity
    identity = opentensor.Identity()

    # Build object summary writer.
    writer = SummaryWriter(log_dir='./runs/' + identity.public_key())

    # Build the neuron object.
    neuron = opentensor.Neuron(identity=identity, bootstrap=hparams.bootstrap)
    neuron.start()
    while True:
        writer.add_scalar('n_peers', len(neuron.metagraph.synapses), time.time())
        writer.add_scalar('n_synapses', len(neuron.metagraph.peers), time.time())
        try:
            time.sleep(10)
        except (KeyboardInterrupt, SystemExit):
            neuron.stop()
    del neuron


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap',
                        default='',
                        type=str,
                        help="peer to bootstrap")
    hparams = parser.parse_args()
    main(hparams)
