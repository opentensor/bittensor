
from typing import Union, TYPE_CHECKING


if TYPE_CHECKING:
    from bittensor.core.subtensor import Subtensor
    from bittensor.core.async_subtensor import AsyncSubtensor


class SubtensorModule:
    def __init__(self, subtensor: Union["Subtensor", "AsyncSubtensor"]):
        self._subtensor = subtensor

    def tempo(self, netuid: int):
        return self._subtensor.substrate.query(
            module="SubtensorModule",
            storage_function="Tempo",
            params=[netuid]
        )

    def tempos(self):
        return self._subtensor.substrate.query_map(
            module="SubtensorModule",
            storage_function="Tempo",
        )
