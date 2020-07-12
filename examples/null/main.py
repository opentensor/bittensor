from loguru import logger
import argparse
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
    neuron = opentensor.Neuron(identity=identity,
                               remote_ip=hparams.remote_ip,
                               bootstrap=hparams.bootstrap,
                               writer=writer)
    try:
        neuron.start()
    finally:
        del neuron


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--remote_ip',
                        default='localhost',
                        type=str,
                        help="IP to advertise")
    parser.add_argument('--bootstrap',
                        default='',
                        type=str,
                        help="peer to bootstrap")
    hparams = parser.parse_args()
    main(hparams)
