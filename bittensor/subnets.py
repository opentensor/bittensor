# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2023 Opentensor Foundation
# Copyright © 2023 Opentensor Technologies Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import bittensor as bt
from abc import ABC, abstractmethod
from typing import Any, List, Union, Optional


class SubnetsAPI(ABC):
    def __init__(self, wallet: "bt.wallet"):
        self.wallet = wallet
        self.dendrite = bt.dendrite(wallet=wallet)

    async def __call__(self, *args, **kwargs):
        return await self.query_api(*args, **kwargs)

    @abstractmethod
    def prepare_synapse(self, *args, **kwargs) -> Any:
        """
        Prepare the synapse-specific payload.
        """
        ...

    @abstractmethod
    def process_responses(self, responses: List[Union["bt.Synapse", Any]]) -> Any:
        """
        Process the responses from the network.
        """
        ...

    async def query_api(
        self,
        axons: Union[bt.axon, List[bt.axon]],
        deserialize: Optional[bool] = False,
        timeout: Optional[int] = 12,
        **kwargs: Optional[Any],
    ) -> Any:
        """
        Queries the API nodes of a subnet using the given synapse and bespoke query function.

        Args:
            axons (Union[bt.axon, List[bt.axon]]): The list of axon(s) to query.
            deserialize (bool, optional): Whether to deserialize the responses. Defaults to False.
            timeout (int, optional): The timeout in seconds for the query. Defaults to 12.
            **kwargs: Keyword arguments for the prepare_synapse_fn.

        Returns:
            Any: The result of the process_responses_fn.
        """
        synapse = self.prepare_synapse(**kwargs)
        bt.logging.debug(f"Querying validator axons with synapse {synapse.name}...")
        responses = await self.dendrite(
            axons=axons,
            synapse=synapse,
            deserialize=deserialize,
            timeout=timeout,
        )
        return self.process_responses(responses)
