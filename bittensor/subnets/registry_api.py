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

import json
import torch
import random
import bittensor as bt
from abc import ABC, abstractmethod
from typing import Any, List, Union


class RegistryMeta(type):
    def __str__(cls):
        return f"APIRegistry with handlers: {list(cls._apis)}"

    def __repr__(cls):
        return f"APIRegistry with handlers: {list(cls._apis)}"


class APIRegistry(metaclass=RegistryMeta):
    _apis = {}

    @classmethod
    def register_api_handler(cls, key, handler):
        cls._apis[key] = handler

    @classmethod
    def get_api_handler(cls, key, *args, **kwargs):
        handler = cls._apis.get(key)
        if not handler:
            raise ValueError(f"No handler registered for key: {key}")
        return handler(*args, **kwargs)

    @classmethod
    def __call__(cls, *args, **kwargs):
        return cls.get_api_handler(*args, **kwargs)


def register_handler(key):
    def decorator(cls):
        APIRegistry.register_api_handler(key, cls)
        return cls

    return decorator


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

    async def ping_uids(self, metagraph, uids, timeout=3):
        """
        Pings a list of UIDs to check their availability on the Bittensor network.

        Args:
            metagraph (bittensor.metagraph): The metagraph instance containing network information.
            uids (list): A list of UIDs (unique identifiers) to ping.
            timeout (int, optional): The timeout in seconds for each ping. Defaults to 3.

        Returns:
            tuple: A tuple containing two lists:
                - The first list contains UIDs that were successfully pinged.
                - The second list contains UIDs that failed to respond.
        """
        axons = [metagraph.axons[uid] for uid in uids]
        try:
            responses = await self.dendrite(
                axons,
                bt.Synapse(),  # TODO: potentially get the synapses available back?
                deserialize=False,
                timeout=timeout,
            )
            successful_uids = [
                uid
                for uid, response in zip(uids, responses)
                if response.dendrite.status_code == 200
            ]
            failed_uids = [
                uid
                for uid, response in zip(uids, responses)
                if response.dendrite.status_code != 200
            ]
        except Exception as e:
            bt.logging.error(f"Dendrite ping failed: {e}")
            successful_uids = []
            failed_uids = uids
        bt.logging.debug("ping() successful uids:", successful_uids)
        bt.logging.debug("ping() failed uids    :", failed_uids)
        return successful_uids, failed_uids

    async def get_query_api_nodes(self, metagraph, n=0.1, timeout=3):
        """
        Fetches the available API nodes to query for the particular subnet.

        Args:
            metagraph (bittensor.metagraph): The metagraph instance containing network information.
            n (float, optional): The fraction of top nodes to consider based on stake. Defaults to 0.1.
            timeout (int, optional): The timeout in seconds for pinging nodes. Defaults to 3.

        Returns:
            list: A list of UIDs representing the available API nodes.
        """
        bt.logging.debug(f"Fetching available API nodes for subnet {metagraph.netuid}")
        vtrust_uids = [
            uid.item() for uid in metagraph.uids if metagraph.validator_trust[uid] > 0
        ]
        top_uids = torch.where(metagraph.S > torch.quantile(metagraph.S, 1 - n))
        top_uids = top_uids[0].tolist()
        init_query_uids = set(top_uids).intersection(set(vtrust_uids))
        query_uids, _ = await self.ping_uids(
            metagraph, init_query_uids, timeout=timeout
        )
        bt.logging.debug(
            f"Available API node UIDs for subnet {metagraph.netuid}: {query_uids}"
        )
        if len(query_uids) > 3:
            query_uids = random.sample(query_uids, 3)
        return query_uids

    async def get_query_api_axons(self, metagraph, n=0.1, timeout=3, uid=None):
        """
        Retrieves the axons of query API nodes based on their availability and stake.

        Args:
            metagraph (bittensor.metagraph): The metagraph instance containing network information.
            n (float, optional): The fraction of top nodes to consider based on stake. Defaults to 0.1.
            timeout (int, optional): The timeout in seconds for pinging nodes. Defaults to 3.
            uid (int, optional): The specific UID of the API node to query. Defaults to None.

        Returns:
            list: A list of axon objects for the available API nodes.
        """
        if uid is not None:
            query_uids = [uid]
        else:
            query_uids = await self.get_query_api_nodes(metagraph, n=n, timeout=timeout)
        return [metagraph.axons[uid] for uid in query_uids]

    async def query_api(
        self,
        metagraph: bt.metagraph,
        deserialize: bool = False,
        timeout: int = 12,
        n: float = 0.1,
        uid: int = None,
        **kwargs: Any,
    ) -> Any:
        """
        Queries the API nodes of a subnet using the given synapse and bespoke query function.

        Args:
            metagraph (bittensor.metagraph): The metagraph instance containing network information.
            deserialize (bool, optional): Whether to deserialize the responses. Defaults to False.
            timeout (int, optional): The timeout in seconds for the query. Defaults to 12.
            n (float, optional): The fraction of top nodes to consider based on stake. Defaults to 0.1.
            uid (int, optional): The specific UID of the API node to query. Defaults to None.
            **kwargs: Keyword arguments for the prepare_synapse_fn.

        Returns:
            Any: The result of the process_responses_fn.
        """
        synapse = self.prepare_synapse(**kwargs)
        axons = await self.get_query_api_axons(
            metagraph=metagraph, n=n, timeout=timeout, uid=uid
        )
        bt.logging.debug(
            f"Quering valdidator axons with synapse {synapse.name} for subnet {metagraph.netuid}..."
        )
        responses = await self.dendrite(
            axons=axons,
            synapse=synapse,
            deserialize=deserialize,
            timeout=timeout,
        )
        return self.process_responses(responses)
