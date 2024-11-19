from dataclasses import dataclass
from typing import List

from bittensor.core.chain_data.utils import from_scale_encoding, ChainDataType
from bittensor.utils import Certificate


# Dataclasses for chain data.
@dataclass
class NeuronCertificate:
    r"""
    Dataclass for neuron certificate.
    """

    certificate: Certificate

    @classmethod
    def from_vec_u8(cls, vec_u8: List[int]) -> "NeuronCertificate":
        r"""Returns a NeuronCertificate object from a ``vec_u8``."""
        return from_scale_encoding(vec_u8, ChainDataType.NeuronCertificate)
