from dataclasses import dataclass

from bittensor.utils import Certificate


# Dataclasses for chain data.
@dataclass
class NeuronCertificate:  # TODO Info?
    """
    Dataclass for neuron certificate.
    """

    certificate: Certificate
