from typing import Union
from bittensor.core.subtensor import Subtensor as _Subtensor
from bittensor.core.async_subtensor import AsyncSubtensor as _AsyncSubtensor


class Queries:
    """Class for managing subtensor query operations."""

    def __init__(self, subtensor: Union["_Subtensor", "_AsyncSubtensor"]):
        self.query_constant = subtensor.query_constant
        self.query_map = subtensor.query_map
        self.query_map_subtensor = subtensor.query_map_subtensor
        self.query_module = subtensor.query_module
        self.query_runtime_api = subtensor.query_runtime_api
        self.query_subtensor = subtensor.query_subtensor
