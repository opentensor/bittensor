from typing import Union, TYPE_CHECKING
from bittensor.core.subtensor_plugins import Plugin

if TYPE_CHECKING:
    from bittensor.core.subtensor import Subtensor
    from bittensor.core.async_subtensor import AsyncSubtensor


class SubtensorModule(Plugin):
    def __init__(self, subtensor: Union["Subtensor", "AsyncSubtensor"]):
        super().__init__(subtensor, "SubtensorModule")
        self._subtensor = subtensor

    def tempo(self, netuid: int):
        return self.query(
            storage_function="Tempo",
            params=[netuid]
        )

    def tempos(self):
        return self.query_map(
            storage_function="Tempo"
        )
