from abc import ABC, abstractmethod
from typing import Any, Union, Optional, TYPE_CHECKING

from bittensor.core.axon import Axon
from bittensor.core.dendrite import Dendrite
from bittensor.utils.btlogging import logging

# For annotation purposes
if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.synapse import Synapse


# Community uses this class
class SubnetsAPI(ABC):
    """This class is not used within the bittensor package, but is actively used by the community."""

    def __init__(self, wallet: "Wallet"):
        self.wallet = wallet
        self.dendrite = Dendrite(wallet=wallet)

    async def __call__(self, *args, **kwargs):
        return await self.query_api(*args, **kwargs)

    @abstractmethod
    def prepare_synapse(self, *args, **kwargs) -> Any:
        """Prepare the synapse-specific payload."""

    @abstractmethod
    def process_responses(self, responses: list[Union["Synapse", Any]]) -> Any:
        """Process the responses from the network."""

    async def query_api(
        self,
        axons: Union["Axon", list["Axon"]],
        deserialize: Optional[bool] = False,
        timeout: Optional[int] = 12,
        **kwargs,
    ) -> Any:
        """
        Queries the API nodes of a subnet using the given synapse and bespoke query function.

        Parameters:
            axons: The list of axon(s) to query.
            deserialize: Whether to deserialize the responses.
            timeout: The timeout in seconds for the query.
            **kwargs: Keyword arguments for the prepare_synapse_fn.

        Returns:
            Any: The result of the process_responses_fn.
        """
        synapse = self.prepare_synapse(**kwargs)
        logging.debug(f"Querying validator axons with synapse {synapse.name}...")
        responses = await self.dendrite(
            axons=axons,
            synapse=synapse,
            deserialize=deserialize,
            timeout=timeout,
        )
        return self.process_responses(responses)
