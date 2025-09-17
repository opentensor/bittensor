from typing import Union
from bittensor.core.subtensor import Subtensor as _Subtensor
from bittensor.core.async_subtensor import AsyncSubtensor as _AsyncSubtensor


class Metagraphs:
    """Class for managing metagraph operations."""

    def __init__(self, subtensor: Union["_Subtensor", "_AsyncSubtensor"]):
        self.get_metagraph_info = subtensor.get_metagraph_info
        self.get_all_metagraphs_info = subtensor.get_all_metagraphs_info
        self.get_all_mechagraphs_info = subtensor.get_all_mechagraphs_info
        self.get_mechagraph_info = subtensor.get_mechagraph_info
        self.get_selective_mechagraph_info = subtensor.get_selective_mechagraph_info
        self.metagraph = subtensor.metagraph
