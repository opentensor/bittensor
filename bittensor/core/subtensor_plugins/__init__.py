from typing import Union, TYPE_CHECKING


if TYPE_CHECKING:
    from bittensor.core.subtensor import Subtensor
    from bittensor.core.async_subtensor import AsyncSubtensor


class Plugin:
    def __init__(self, subtensor: Union["Subtensor", "AsyncSubtensor"], module_name: str):
        self._subtensor = subtensor
        self.module = module_name

    def query(self, *args, **kwargs):
        return self._subtensor.substrate.query(
            module=self.module,
            *args, **kwargs
        )

    def query_map(self, *args, **kwargs):
        return self._subtensor.substrate.query_map(
            module=self.module,
            *args, **kwargs
        )
