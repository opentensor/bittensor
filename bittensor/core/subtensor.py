"""
Implementation of the Subtensor class for interacting with the Bittensor blockchain.
"""

import asyncio
import logging
from typing import Any, Optional, TYPE_CHECKING, Union

from bittensor.core.errors import NetworkQueryError
from bittensor.core.settings import version_as_int
from bittensor.utils.btlogging import logging as logger
from bittensor.utils.registration import POWSolution

if TYPE_CHECKING:
    from bittensor.core.chain_data import (
        NeuronInfo,
        NeuronInfoLite,
        PrometheusInfo,
        SubnetInfo,
        SubnetHyperparameters,
        IPInfo,
        StakeInfo,
        DelegateInfo,
        ScheduledColdkeySwapInfo,
    )


class Subtensor:
    """
    The Subtensor class provides an interface for interacting with the Bittensor blockchain.
    
    This class handles all substrate interactions and provides methods for querying
    blockchain state and submitting extrinsics.
    """

    def __init__(
        self,
        network: Optional[str] = None,
        config: Optional[dict] = None,
        _mock: bool = False,
    ):
        """
        Initialize the Subtensor instance.
        
        Args:
            network: Network name to connect to
            config: Configuration dictionary
            _mock: Whether to use mock substrate interface
        """
        self.network = network
        self.config = config or {}
        self._mock = _mock
        self.substrate = None
        
    def _get_hyperparameter(
        self,
        param_name: str,
        netuid: int,
        block: Optional[int] = None,
        reuse_block: bool = False,
    ) -> Optional[Any]:
        """
        Internal method to retrieve a hyperparameter for a specific subnet.
        
        Args:
            param_name: Name of the hyperparameter to retrieve
            netuid: Unique identifier for the subnet
            block: Block number to query at (None for latest)
            reuse_block: Whether to reuse cached block
            
        Returns:
            The hyperparameter value, or None if not found
        """
        if self.substrate is None:
            raise RuntimeError("Substrate connection not initialized")
            
        result = self.substrate.query(
            module="SubtensorModule",
            storage_function=param_name,
            params=[netuid],
            block_hash=None if block is None else self.substrate.get_block_hash(block),
            reuse_block_hash=reuse_block,
        )
        
        return getattr(result, "value", result)
        
    def get_subnet_hyperparameters(
        self,
        netuid: int,
        block: Optional[int] = None,
        reuse_block: bool = False,
    ) -> Optional["SubnetHyperparameters"]:
        """
        Retrieve hyperparameters for a specific subnet.
        
        Args:
            netuid: Unique identifier for the subnet
            block: Block number to query at (None for latest)
            reuse_block: Whether to reuse cached block
            
        Returns:
            SubnetHyperparameters object or None if subnet doesn't exist
        """
        pass
        
    def get_commitment(
        self,
        netuid: int,
        uid: int,
        block: Optional[int] = None,
    ) -> Optional[str]:
        """
        Retrieve the commitment for a specific neuron.
        
        Args:
            netuid: Unique identifier for the subnet
            uid: Unique identifier for the neuron
            block: Block number to query at (None for latest)
            
        Returns:
            The commitment string, or None if not found
        """
        pass
