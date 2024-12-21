"""
This module defines the `SubtensorWithRetry` class, which wraps around the `Subtensor` module from the Bittensor SDK to
provide retry mechanisms for interacting with blockchain endpoints. It includes fault-tolerant methods for querying and
executing blockchain-related operations with automatic retries on failures.
The `call_with_retry` decorator is used to wrap methods and handle retries transparently.

Primary features:
- Retry mechanisms for `Subtensor` interactions.
- Configurable timeout, retry attempts, and retry intervals.
- Support for various `Subtensor` operations with enhanced logging and error handling.

`SubtensorWithRetry` Class:
- Initializes with multiple endpoint support and manages connections to blockchain nodes.
- Configurable retry behavior using retry seconds or epochs.
- Includes methods for blockchain queries, stake checks, and neuron operations.
- Supports automatic reconnecting and fault-tolerant behavior during operation failures.
"""

import inspect
import time
from functools import wraps, cache
from typing import Optional, Union, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from bittensor.core.metagraph import Metagraph
from bittensor.core.settings import version_as_int
from bittensor.core.subtensor import Subtensor
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor.core.axon import Axon
    from bittensor.core.config import Config
    from bittensor.core.chain_data.delegate_info import DelegateInfo
    from bittensor.core.chain_data.neuron_info import NeuronInfo
    from bittensor.core.chain_data.neuron_info_lite import NeuronInfoLite
    from bittensor.core.chain_data.prometheus_info import PrometheusInfo
    from bittensor.core.chain_data.subnet_info import SubnetInfo
    from bittensor.core.chain_data.subnet_hyperparameters import SubnetHyperparameters
    from bittensor.utils import Certificate, torch
    from bittensor.utils.balance import Balance
    from websockets.sync import client as ws_client
    from bittensor_wallet import Wallet


CHAIN_BLOCK_SECONDS = 12
DEFAULT_SUBNET_TEMPO = 360


class SubtensorWithRetryError(Exception):
    """Error for SubtensorWithRetry."""


@cache
def check_net_uid(method, *args, **kwargs):
    """Extracts and returns the 'netuid' argument from the method's arguments, if present."""
    sig = inspect.signature(method)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()

    if "netuid" in bound_args.arguments:
        return bound_args.arguments["netuid"]
    return None


def call_with_retry(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        self: "SubtensorWithRetry" = args[0]
        last_exception = None
        for endpoint in self._endpoints:
            retries = 0
            while retries < self._retry_attempts:
                retries += 1
                try:
                    if not self._subtensor:
                        self._get_subtensor(endpoint=endpoint)
                    result = method(*args, **kwargs)
                    return result
                except Exception as error:
                    logging.error(
                        f"Attempt [blue]{retries}[/blue] for method [blue]{method.__name__}[/blue] failed. Error: {error}"
                    )
                    last_exception = error
                    if retries < self._retry_attempts:
                        netuid = check_net_uid(method, *args, **kwargs)
                        retry_seconds = self.get_retry_seconds(netuid=netuid)
                        logging.debug(
                            f"Retrying method [blue]{method.__name__}[/blue] call in [blue]{retry_seconds}[/blue] seconds."
                        )
                        time.sleep(retry_seconds)

        err_msg = f"Method '{method.__name__}' failed for all endpoints {self._endpoints} with {self._retry_attempts} attempts."
        logging.critical(err_msg)
        raise SubtensorWithRetryError(err_msg) from last_exception

    return wrapper


def _check_retry_args(
    retry_seconds: Optional[int] = None, retry_epoch: Optional[int] = None
):
    if (retry_seconds and retry_epoch) or (not retry_seconds and not retry_epoch):
        raise ValueError("Either `_retry_seconds` or `_retry_epoch` must be specified.")


class SubtensorWithRetry:
    def __init__(
        self,
        endpoints: list[str],
        retry_seconds: Optional[int] = 0,
        retry_epoch: Optional[int] = None,
        retry_attempts: Optional[int] = None,
        # Subtensor arguments
        config: Optional["Config"] = None,
        log_verbose: bool = False,
        connection_timeout: int = 600,
        websocket: Optional["ws_client.ClientConnection"] = None,
    ):
        """
        Initializes an object with retry configuration and network connection parameters.

        The constructor initializes the retry configuration parameters and other settings such as the endpoints,
        connection timeout, and optional websocket client. It also prepares for Subtensor specific definitions for
        further usage.

        Arguments:
            endpoints: list[str] A list specifying the network endpoints for the object to connect to.
            retry_seconds: Optional[int], default `None`. Retry duration in seconds for operations, if provided.
            retry_epoch: Optional[int], default `None`. Epoch-based retry duration for operations, if provided.
            retry_attempts: Optional[int], default `1`. Number of retry attempts in case of failure.
            config: Optional["Config"], default `None`. Configuration object for Subtensor-specific settings, if provided.
            log_verbose: bool, default `False`. Boolean flag to enable verbose logging.
            connection_timeout: int, default `600`. The maximum duration (in seconds) to wait for a connection.
            websocket: Optional["ws_client.ClientConnection"], default `None`. An optional websocket client connection object.
        """
        _check_retry_args(retry_seconds=retry_seconds, retry_epoch=retry_epoch)
        self._retry_seconds = retry_seconds if not retry_epoch else None
        self._retry_epoch = retry_epoch
        self._retry_attempts = retry_attempts or 1

        self._endpoints = endpoints

        # Subtensor specific definition
        self._config = config
        self._log_verbose = log_verbose
        self._connection_timeout = connection_timeout
        self._websocket = websocket

        self._subtensor = None

    def _get_subtensor(self, endpoint: Optional[str] = None):
        """Initializes the Subtensor instance."""
        logging.debug(
            f"[magenta]Getting connection with endpoint:[/magenta] [blue]{endpoint}[/blue]."
        )
        self._subtensor = Subtensor(
            network=endpoint,
            config=self._config,
            log_verbose=self._log_verbose,
            connection_timeout=self._connection_timeout,
            websocket=self._websocket,
        )
        logging.debug(
            f"[magenta]Subtensor initialized with endpoint:[/magenta] [blue]{endpoint}[/blue]."
        )

    def get_retry_seconds(self, netuid: Optional[int] = None) -> int:
        """Returns the number of seconds to wait before retrying a request based on `retry_second` or `_retry_epoch`.

        Arguments:
            netuid (int): The unique identifier of the subnet. Used in case `_retry_epoch` is specified for class instance.

        Returns:
            int: The number of seconds to wait before retrying a request.
        """
        if self._retry_seconds:
            return self._retry_seconds

        subnet_tempo = DEFAULT_SUBNET_TEMPO
        try:
            subnet_hyperparameters = self._subtensor.get_subnet_hyperparameters(
                netuid=netuid
            )
            subnet_tempo = subnet_hyperparameters.tempo
        except AttributeError:
            logging.debug(
                f"Subtensor instance was not initialized. Use default tempo as [blue]{DEFAULT_SUBNET_TEMPO}"
                f"[/blue] blocks."
            )
        return subnet_tempo * CHAIN_BLOCK_SECONDS

    # Subtensor calls ==================================================================================================

    @call_with_retry
    def get_account_next_index(self, address: str) -> int:
        return self._subtensor.get_account_next_index(address=address)

    @call_with_retry
    def metagraph(
        self, netuid: int, lite: bool = True, block: Optional[int] = None
    ) -> "Metagraph":
        return self._subtensor.metagraph(netuid=netuid, lite=lite, block=block)

    @call_with_retry
    def get_netuids_for_hotkey(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> list[int]:
        return self._subtensor.get_netuids_for_hotkey(
            hotkey_ss58=hotkey_ss58, block=block
        )

    @property
    def block(self) -> int:
        return self.get_current_block()

    @call_with_retry
    def get_current_block(self) -> int:
        return self._subtensor.get_current_block()

    @call_with_retry
    def is_hotkey_registered_any(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> bool:
        return self._subtensor.is_hotkey_registered_any(
            hotkey_ss58=hotkey_ss58, block=block
        )

    @call_with_retry
    def is_hotkey_registered_on_subnet(
        self, hotkey_ss58: str, netuid: int, block: Optional[int] = None
    ) -> bool:
        return self._subtensor.is_hotkey_registered_on_subnet(
            hotkey_ss58=hotkey_ss58, netuid=netuid, block=block
        )

    @call_with_retry
    def is_hotkey_registered(
        self,
        hotkey_ss58: str,
        netuid: Optional[int] = None,
        block: Optional[int] = None,
    ) -> bool:
        return self._subtensor.is_hotkey_registered(
            hotkey_ss58=hotkey_ss58, netuid=netuid, block=block
        )

    @call_with_retry
    def blocks_since_last_update(self, netuid: int, uid: int) -> Optional[int]:
        return self._subtensor.blocks_since_last_update(netuid=netuid, uid=uid)

    @call_with_retry
    def get_block_hash(self, block_id: int) -> str:
        return self._subtensor.get_block_hash(block_id=block_id)

    @call_with_retry
    def weights_rate_limit(self, netuid: int) -> Optional[int]:
        return self._subtensor.weights_rate_limit(netuid=netuid)

    @call_with_retry
    def commit(self, wallet, netuid: int, data: str):
        return self._subtensor.commit(wallet=wallet, netuid=netuid, data=data)

    @call_with_retry
    def subnetwork_n(self, netuid: int, block: Optional[int] = None) -> Optional[int]:
        return self._subtensor.subnetwork_n(netuid=netuid, block=block)

    @call_with_retry
    def get_neuron_for_pubkey_and_subnet(
        self, hotkey_ss58: str, netuid: int, block: Optional[int] = None
    ) -> Optional["NeuronInfo"]:
        return self._subtensor.get_neuron_for_pubkey_and_subnet(
            hotkey_ss58=hotkey_ss58, netuid=netuid, block=block
        )

    @call_with_retry
    def get_neuron_certificate(
        self, hotkey: str, netuid: int, block: Optional[int] = None
    ) -> Optional["Certificate"]:
        return self._subtensor.get_neuron_certificate(
            hotkey=hotkey, netuid=netuid, block=block
        )

    @call_with_retry
    def neuron_for_uid(
        self, uid: int, netuid: int, block: Optional[int] = None
    ) -> "NeuronInfo":
        return self._subtensor.neuron_for_uid(uid=uid, netuid=netuid, block=block)

    @call_with_retry
    def get_subnet_hyperparameters(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[Union[list, "SubnetHyperparameters"]]:
        return self._subtensor.get_subnet_hyperparameters(netuid=netuid, block=block)

    @call_with_retry
    def immunity_period(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        return self._subtensor.immunity_period(netuid=netuid, block=block)

    @call_with_retry
    def get_uid_for_hotkey_on_subnet(
        self, hotkey_ss58: str, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        return self._subtensor.get_uid_for_hotkey_on_subnet(
            hotkey_ss58=hotkey_ss58, netuid=netuid, block=block
        )

    @call_with_retry
    def tempo(self, netuid: int, block: Optional[int] = None) -> Optional[int]:
        return self._subtensor.tempo(netuid=netuid, block=block)

    @call_with_retry
    def get_commitment(self, netuid: int, uid: int, block: Optional[int] = None) -> str:
        return self._subtensor.get_commitment(netuid=netuid, uid=uid, block=block)

    @call_with_retry
    def min_allowed_weights(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        return self._subtensor.min_allowed_weights(netuid=netuid, block=block)

    @call_with_retry
    def max_weight_limit(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[float]:
        return self._subtensor.max_weight_limit(netuid=netuid, block=block)

    @call_with_retry
    def commit_reveal_enabled(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[bool]:
        return self._subtensor.commit_reveal_enabled(netuid=netuid, block=block)

    @call_with_retry
    def get_subnet_reveal_period_epochs(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        return self._subtensor.get_subnet_reveal_period_epochs(
            netuid=netuid, block=block
        )

    @call_with_retry
    def get_prometheus_info(
        self, netuid: int, hotkey_ss58: str, block: Optional[int] = None
    ) -> Optional["PrometheusInfo"]:
        return self._subtensor.get_prometheus_info(
            netuid=netuid, hotkey_ss58=hotkey_ss58, block=block
        )

    @call_with_retry
    def subnet_exists(self, netuid: int, block: Optional[int] = None) -> bool:
        return self._subtensor.subnet_exists(netuid=netuid, block=block)

    @call_with_retry
    def get_all_subnets_info(self, block: Optional[int] = None) -> list["SubnetInfo"]:
        return self._subtensor.get_all_subnets_info(block=block)

    @call_with_retry
    def bonds(
        self, netuid: int, block: Optional[int] = None
    ) -> list[tuple[int, list[tuple[int, int]]]]:
        return self._subtensor.bonds(netuid=netuid, block=block)

    @call_with_retry
    def get_subnet_burn_cost(self, block: Optional[int] = None) -> Optional[str]:
        return self._subtensor.get_subnet_burn_cost(block=block)

    @call_with_retry
    def neurons(self, netuid: int, block: Optional[int] = None) -> list["NeuronInfo"]:
        return self._subtensor.neurons(netuid=netuid, block=block)

    @call_with_retry
    def last_drand_round(self) -> Optional[int]:
        return self._subtensor.last_drand_round()

    @call_with_retry
    def get_current_weight_commit_info(
        self, netuid: int, block: Optional[int] = None
    ) -> list:
        return self._subtensor.get_current_weight_commit_info(
            netuid=netuid, block=block
        )

    @call_with_retry
    def get_total_stake_for_coldkey(
        self, ss58_address: str, block: Optional[int] = None
    ) -> Optional["Balance"]:
        return self._subtensor.get_total_stake_for_coldkey(
            ss58_address=ss58_address, block=block
        )

    @call_with_retry
    def get_total_stake_for_hotkey(
        self, ss58_address: str, block: Optional[int] = None
    ):
        return self._subtensor.get_total_stake_for_hotkey(
            ss58_address=ss58_address, block=block
        )

    @call_with_retry
    def get_total_stake_for_hotkey(
        self, ss58_address: str, block: Optional[int] = None
    ) -> Optional["Balance"]:
        return self._subtensor.get_total_stake_for_hotkey(
            ss58_address=ss58_address, block=block
        )

    @call_with_retry
    def get_total_subnets(self, block: Optional[int] = None) -> Optional[int]:
        return self._subtensor.get_total_subnets(block=block)

    @call_with_retry
    def get_subnets(self, block: Optional[int] = None) -> list[int]:
        return self._subtensor.get_subnets(block=block)

    @call_with_retry
    def neurons_lite(
        self, netuid: int, block: Optional[int] = None
    ) -> list["NeuronInfoLite"]:
        return self._subtensor.neurons_lite(netuid=netuid, block=block)

    @call_with_retry
    def weights(
        self, netuid: int, block: Optional[int] = None
    ) -> list[tuple[int, list[tuple[int, int]]]]:
        return self._subtensor.weights(netuid=netuid, block=block)

    @call_with_retry
    def get_balance(self, address: str, block: Optional[int] = None) -> "Balance":
        return self._subtensor.get_balance(address=address, block=block)

    @call_with_retry
    def get_transfer_fee(
        self, wallet: "Wallet", dest: str, value: Union["Balance", float, int]
    ) -> "Balance":
        return self._subtensor.get_transfer_fee(wallet=wallet, dest=dest, value=value)

    @call_with_retry
    def get_existential_deposit(
        self, block: Optional[int] = None
    ) -> Optional["Balance"]:
        return self._subtensor.get_existential_deposit(block=block)

    @call_with_retry
    def difficulty(self, netuid: int, block: Optional[int] = None) -> Optional[int]:
        return self._subtensor.difficulty(netuid=netuid, block=block)

    @call_with_retry
    def recycle(self, netuid: int, block: Optional[int] = None) -> Optional["Balance"]:
        return self._subtensor.recycle(netuid=netuid, block=block)

    @call_with_retry
    def get_delegate_take(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> Optional[float]:
        return self._subtensor.get_delegate_take(hotkey_ss58=hotkey_ss58, block=block)

    @call_with_retry
    def get_delegate_by_hotkey(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> Optional["DelegateInfo"]:
        return self._subtensor.get_delegate_by_hotkey(
            hotkey_ss58=hotkey_ss58, block=block
        )

    @call_with_retry
    def get_stake_for_coldkey_and_hotkey(
        self, hotkey_ss58: str, coldkey_ss58: str, block: Optional[int] = None
    ) -> Optional["Balance"]:
        return self._subtensor.get_stake_for_coldkey_and_hotkey(
            hotkey_ss58=hotkey_ss58, coldkey_ss58=coldkey_ss58, block=block
        )

    @call_with_retry
    def does_hotkey_exist(self, hotkey_ss58: str, block: Optional[int] = None) -> bool:
        return self._subtensor.does_hotkey_exist(hotkey_ss58=hotkey_ss58, block=block)

    @call_with_retry
    def get_hotkey_owner(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> Optional[str]:
        return self._subtensor.get_hotkey_owner(hotkey_ss58=hotkey_ss58, block=block)

    @call_with_retry
    def get_minimum_required_stake(self) -> "Balance":
        return self._subtensor.get_minimum_required_stake()

    @call_with_retry
    def tx_rate_limit(self, block: Optional[int] = None) -> Optional[int]:
        return self._subtensor.tx_rate_limit(block=block)

    @call_with_retry
    def get_delegates(self, block: Optional[int] = None) -> list["DelegateInfo"]:
        return self._subtensor.get_delegates(block=block)

    @call_with_retry
    def is_hotkey_delegate(self, hotkey_ss58: str, block: Optional[int] = None) -> bool:
        return self._subtensor.is_hotkey_delegate(hotkey_ss58=hotkey_ss58, block=block)

    @call_with_retry
    def test_netuid(self, netuid: int) -> bool:
        return self._subtensor.test_netuid(netuid=netuid)

    # Extrinsics =======================================================================================================

    @call_with_retry
    def set_weights(
        self,
        wallet: "Wallet",
        netuid: int,
        uids: Union[NDArray[np.int64], "torch.LongTensor", list],
        weights: Union[NDArray[np.float32], "torch.FloatTensor", list],
        version_key: int = version_as_int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
        max_retries: int = 5,
    ) -> tuple[bool, str]:
        return self._subtensor.set_weights(
            wallet=wallet,
            netuid=netuid,
            uids=uids,
            weights=weights,
            version_key=version_key,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            max_retries=max_retries,
        )

    @call_with_retry
    def root_set_weights(
        self,
        wallet: "Wallet",
        netuids: Union[NDArray[np.int64], "torch.LongTensor", list],
        weights: Union[NDArray[np.float32], "torch.FloatTensor", list],
        version_key: int = 0,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
    ) -> bool:
        return self._subtensor.root_set_weights(
            wallet=wallet,
            netuids=netuids,
            weights=weights,
            version_key=version_key,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    @call_with_retry
    def register(
        self,
        wallet: "Wallet",
        netuid: int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        max_allowed_attempts: int = 3,
        output_in_place: bool = True,
        cuda: bool = False,
        dev_id: Union[list[int], int] = 0,
        tpb: int = 256,
        num_processes: Optional[int] = None,
        update_interval: Optional[int] = None,
        log_verbose: bool = False,
    ) -> bool:
        return self._subtensor.register(
            wallet=wallet,
            netuid=netuid,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            max_allowed_attempts=max_allowed_attempts,
            output_in_place=output_in_place,
            cuda=cuda,
            dev_id=dev_id,
            tpb=tpb,
            num_processes=num_processes,
            update_interval=update_interval,
            log_verbose=log_verbose,
        )

    @call_with_retry
    def root_register(
        self,
        wallet: "Wallet",
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
    ) -> bool:
        return self._subtensor.root_register(
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    @call_with_retry
    def burned_register(
        self,
        wallet: "Wallet",
        netuid: int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
    ) -> bool:
        return self._subtensor.burned_register(
            wallet=wallet,
            netuid=netuid,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    @call_with_retry
    def serve_axon(
        self,
        netuid: int,
        axon: "Axon",
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        certificate: Optional["Certificate"] = None,
    ) -> bool:
        return self._subtensor.serve_axon(
            netuid=netuid,
            axon=axon,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            certificate=certificate,
        )

    @call_with_retry
    def transfer(
        self,
        wallet: "Wallet",
        dest: str,
        amount: Union["Balance", float],
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        return self._subtensor.transfer(
            wallet=wallet,
            dest=dest,
            amount=amount,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    @call_with_retry
    def commit_weights(
        self,
        wallet: "Wallet",
        netuid: int,
        salt: list[int],
        uids: Union[NDArray[np.int64], list],
        weights: Union[NDArray[np.int64], list],
        version_key: int = version_as_int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
        max_retries: int = 5,
    ) -> tuple[bool, str]:
        return self._subtensor.commit_weights(
            wallet=wallet,
            netuid=netuid,
            salt=salt,
            uids=uids,
            weights=weights,
            version_key=version_key,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            max_retries=max_retries,
        )

    @call_with_retry
    def reveal_weights(
        self,
        wallet: "Wallet",
        netuid: int,
        uids: Union[NDArray[np.int64], list],
        weights: Union[NDArray[np.int64], list],
        salt: Union[NDArray[np.int64], list],
        version_key: int = version_as_int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
        max_retries: int = 5,
    ) -> tuple[bool, str]:
        return self._subtensor.reveal_weights(
            wallet=wallet,
            netuid=netuid,
            uids=uids,
            weights=weights,
            salt=salt,
            version_key=version_key,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            max_retries=max_retries,
        )

    @call_with_retry
    def add_stake(
        self,
        wallet: "Wallet",
        hotkey_ss58: Optional[str] = None,
        amount: Optional[Union["Balance", float]] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        return self._subtensor.add_stake(
            wallet=wallet,
            hotkey_ss58=hotkey_ss58,
            amount=amount,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    @call_with_retry
    def add_stake_multiple(
        self,
        wallet: "Wallet",
        hotkey_ss58s: list[str],
        amounts: Optional[list[Union["Balance", float]]] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        return self._subtensor.add_stake_multiple(
            wallet=wallet,
            hotkey_ss58s=hotkey_ss58s,
            amounts=amounts,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    @call_with_retry
    def unstake(
        self,
        wallet: "Wallet",
        hotkey_ss58: Optional[str] = None,
        amount: Optional[Union["Balance", float]] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        return self._subtensor.unstake(
            wallet=wallet,
            hotkey_ss58=hotkey_ss58,
            amount=amount,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    @call_with_retry
    def unstake_multiple(
        self,
        wallet: "Wallet",
        hotkey_ss58s: list[str],
        amounts: Optional[list[Union["Balance", float]]] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        return self._subtensor.unstake_multiple(
            wallet=wallet,
            hotkey_ss58s=hotkey_ss58s,
            amounts=amounts,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
