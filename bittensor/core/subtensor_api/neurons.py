from typing import Union
from bittensor.core.subtensor import Subtensor as _Subtensor
from bittensor.core.async_subtensor import AsyncSubtensor as _AsyncSubtensor


class Neurons:
    """Class for managing neuron operations."""

    def __init__(self, subtensor: Union["_Subtensor", "_AsyncSubtensor"]):
        self.get_all_neuron_certificates = subtensor.get_all_neuron_certificates
        self.get_neuron_certificate = subtensor.get_neuron_certificate
        self.neuron_for_uid = subtensor.neuron_for_uid
        self.neurons = subtensor.neurons
        self.neurons_lite = subtensor.neurons_lite
        self.query_identity = subtensor.query_identity
