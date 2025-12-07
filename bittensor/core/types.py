import argparse
from abc import ABC
from dataclasses import dataclass
from typing import Any, Literal, Optional, TypedDict, Union, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from bittensor.core import settings
from bittensor.core.chain_data import NeuronInfo, NeuronInfoLite
from bittensor.core.config import Config
from bittensor.utils import (
    determine_chain_endpoint_and_network,
    get_caller_name,
    format_error_message,
    networking,
    unlock_key,
    Certificate,
    UnlockStatus,
)
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.utils.balance import Balance
    from scalecodec.types import GenericExtrinsic
    from async_substrate_interface.sync_substrate import ExtrinsicReceipt
    from async_substrate_interface.async_substrate import AsyncExtrinsicReceipt


# Type annotations for UIDs and weights.
UIDs = Union[NDArray[np.int64], list[Union[int]]]
Weights = Union[NDArray[np.float32], list[Union[int, float]], list[int], list[float]]
Salt = Union[NDArray[np.int64], list[int]]


class SubtensorMixin(ABC):
    network: str
    chain_endpoint: str
    log_verbose: bool

    def __str__(self):
        return f"Network: {self.network}, Chain: {self.chain_endpoint}"

    def __repr__(self):
        return self.__str__()

    def _check_and_log_network_settings(self):
        if (
            self.network == "finney"
            or self.chain_endpoint == settings.FINNEY_ENTRYPOINT
        ) and self.log_verbose:
            logging.info(
                f"You are connecting to {self.network} network with endpoint {self.chain_endpoint}."
            )
            logging.debug(
                "We strongly encourage running a local subtensor node whenever possible. "
                "This increases decentralization and resilience of the network."
            )

    @staticmethod
    def config() -> "Config":
        """
        Creates and returns a Bittensor configuration object.

        Returns:
            A Bittensor configuration object configured with arguments added by the `subtensor.add_args` method.
        """
        parser = argparse.ArgumentParser()
        SubtensorMixin.add_args(parser)
        return Config(parser)

    @staticmethod
    def setup_config(network: Optional[str], config: "Config") -> tuple[str, str]:
        """
        Sets up and returns the configuration for the Subtensor network and endpoint.

        This method determines the appropriate network and chain endpoint based on the provided network string or
            configuration object. It evaluates the network and endpoint in the following order of precedence:
            1. Provided network string.
            2. Configured chain endpoint in the `config` object.
            3. Configured network in the `config` object.
            4. Default chain endpoint.
            5. Default network.

        Parameters:
            network: The name of the Subtensor network. If None, the network and endpoint will be determined from the
                `config` object.
            config: The configuration object containing the network and chain endpoint settings.

        Returns:
            tuple: A tuple containing the formatted WebSocket endpoint URL and the evaluated network name.
        """
        if network is None:
            candidates = [
                (
                    config.is_set("subtensor.chain_endpoint"),
                    config.subtensor.chain_endpoint,
                ),
                (config.is_set("subtensor.network"), config.subtensor.network),
                (
                    config.subtensor.get("chain_endpoint"),
                    config.subtensor.chain_endpoint,
                ),
                (config.subtensor.get("network"), config.subtensor.network),
            ]
            for check, config_network in candidates:
                if check:
                    network = config_network

        evaluated_network, evaluated_endpoint = determine_chain_endpoint_and_network(
            network
        )

        return networking.get_formatted_ws_endpoint_url(
            evaluated_endpoint
        ), evaluated_network

    @classmethod
    def help(cls):
        """Print help to stdout."""
        parser = argparse.ArgumentParser()
        cls.add_args(parser)
        print(cls.__new__.__doc__)
        parser.print_help()

    @classmethod
    def add_args(cls, parser: "argparse.ArgumentParser", prefix: Optional[str] = None):
        """
        Adds command-line arguments to the provided ArgumentParser for configuring the Subtensor settings.

        Parameters:
            parser: The ArgumentParser object to which the Subtensor arguments will be added.
            prefix: An optional prefix for the argument names. If provided, the prefix is prepended to each argument name.

        Arguments added:
            --subtensor.network: The Subtensor network flag. Possible values are 'finney', 'test', 'archive', and
                'local'. Overrides the chain endpoint if set.
            --subtensor.chain_endpoint: The Subtensor chain endpoint flag. If set, it overrides the network flag.
            --subtensor._mock: If true, uses a mocked connection to the chain.

        Example:
            parser = argparse.ArgumentParser()
            Subtensor.add_args(parser)
        """
        prefix_str = "" if prefix is None else f"{prefix}."
        try:
            default_network = settings.DEFAULTS.subtensor.network
            default_chain_endpoint = settings.DEFAULTS.subtensor.chain_endpoint

            parser.add_argument(
                f"--{prefix_str}subtensor.network",
                default=default_network,
                type=str,
                help="""The subtensor network flag. The likely choices are:
                                        -- finney (main network)
                                        -- test (test network)
                                        -- archive (archive network +300 blocks)
                                        -- local (local running network)
                                    If this option is set it overloads subtensor.chain_endpoint with
                                    an entry point node from that network.
                                    """,
            )
            parser.add_argument(
                f"--{prefix_str}subtensor.chain_endpoint",
                default=default_chain_endpoint,
                type=str,
                help="""The subtensor endpoint flag. If set, overrides the --network flag.""",
            )
            parser.add_argument(
                f"--{prefix_str}subtensor._mock",
                default=False,
                type=bool,
                help="""If true, uses a mocked connection to the chain.""",
            )

        except argparse.ArgumentError:
            # re-parsing arguments.
            pass


class AxonServeCallParams:
    def __init__(
        self,
        version: int,
        ip: int,
        port: int,
        ip_type: int,
        netuid: int,
        hotkey: str,
        coldkey: str,
        protocol: int,
        placeholder1: int,
        placeholder2: int,
        certificate: Optional[Certificate],
    ):
        self.version = version
        self.ip = ip
        self.port = port
        self.ip_type = ip_type
        self.netuid = netuid
        self.hotkey = hotkey
        self.coldkey = coldkey
        self.protocol = protocol
        self.placeholder1 = placeholder1
        self.placeholder2 = placeholder2
        self.certificate = certificate

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return all(
                getattr(self, attr) == getattr(other, attr) for attr in self.__dict__
            )
        elif isinstance(other, dict):
            return all(getattr(self, attr) == other.get(attr) for attr in self.__dict__)
        elif isinstance(other, (NeuronInfo, NeuronInfoLite)):
            return all(
                [
                    self.version == other.axon_info.version,
                    self.ip == networking.ip_to_int(other.axon_info.ip),
                    self.port == other.axon_info.port,
                    self.ip_type == other.axon_info.ip_type,
                    self.netuid == other.netuid,
                    self.hotkey == other.hotkey,
                    self.coldkey == other.coldkey,
                    self.protocol == other.axon_info.protocol,
                    self.placeholder1 == other.axon_info.placeholder1,
                    self.placeholder2 == other.axon_info.placeholder2,
                ]
            )
        else:
            raise NotImplementedError(
                f"AxonServeCallParams equality not implemented for {type(other)}"
            )

    def copy(self) -> "AxonServeCallParams":
        return self.__class__(
            self.version,
            self.ip,
            self.port,
            self.ip_type,
            self.netuid,
            self.hotkey,
            self.coldkey,
            self.protocol,
            self.placeholder1,
            self.placeholder2,
            self.certificate,
        )

    def as_dict(self) -> dict:
        """Returns a dict representation of this object. If `self.certificate` is `None`, it is not included in this."""
        d: dict = {
            "version": self.version,
            "ip": self.ip,
            "port": self.port,
            "ip_type": self.ip_type,
            "netuid": self.netuid,
            "protocol": self.protocol,
            "placeholder1": self.placeholder1,
            "placeholder2": self.placeholder2,
        }
        if self.certificate:
            d["certificate"] = self.certificate
        return d


class PrometheusServeCallParams(TypedDict):
    """Prometheus serve chain call parameters."""

    version: int
    ip: int
    port: int
    ip_type: int
    netuid: int


@dataclass
class ExtrinsicResponse:
    """
    A standardized response container for handling the extrinsic results submissions and related operations in the SDK.

    This class is designed to give developers a consistent way to represent the outcome of an extrinsic call — whether
    it succeeded or failed — along with useful metadata for debugging, logging, or higher-level business logic.

    The object also implements tuple-like behavior:
      * Iteration yields ``(success, message)``.
      * Indexing is supported: ``response[0] -> success``, ``response[1] -> message``.
      * ``len(response)`` returns 2.

    Attributes:
        success: Indicates if the extrinsic execution was successful.
        message: A status or informational message returned from the execution (e.g., "Successfully registered subnet").
        extrinsic_function: The SDK extrinsic or external function name that was executed (e.g., "add_stake_extrinsic").
        extrinsic: The raw extrinsic object used in the call, if available. This is a ``GenericExtrinsic`` instance
            containing the full payload and metadata of the submitted extrinsic, including call section, method, signer,
            signature, parameters, and encoded bytes. Useful for inspecting or reconstructing the exact transaction
            submitted to the chain.
        extrinsic_fee: The fee charged by the extrinsic, if available.
        extrinsic_receipt: The receipt object of the submitted extrinsic. This is an ``ExtrinsicReceipt`` instance that
            contains the most detailed execution data available, including the block number and hash, triggered events,
            extrinsic index, execution phase, and other low-level details. This allows deep debugging or post-analysis
            of on-chain execution.
        mev_extrinsic: The extrinsic object of the revealed (decrypted and executed) MEV Shield extrinsic. This is
            populated when using MEV Shield protection (``with_mev_protection=True``) and contains the execution details
            of the second extrinsic that decrypts and executes the originally encrypted call. Contains triggered events,
            block information, and other execution metadata. Set to ``None`` for non-MEV Shield transactions or when the
            revealed extrinsic receipt is not available.
        transaction_tao_fee: TAO fee charged by the transaction in TAO (e.g., fee for add_stake), if available.
        transaction_alpha_fee: Alpha fee charged by the transaction (e.g., fee for transfer_stake), if available.
        error: Captures the underlying exception if the extrinsic failed, otherwise `None`.
        data: Arbitrary data returned from the extrinsic, such as decoded events, balance or another extra context.

    Instance methods:
        as_dict: Returns a dictionary representation of this object.
        with_log: Returns itself but with logging message.

    Class methods:
        from_exception: Checks if error is raised or return ExtrinsicResponse accordingly.
        unlock_wallet: Checks if keypair is unlocked and can be used for signing the extrinsic.


    Example:
        import bittensor as bt

        subtensor = bt.SubtensorApi("local")
        wallet = bt.Wallet("alice")

        response = subtensor.subnets.register_subnet(alice_wallet)
        print(response)

        ExtrinsicResponse:
            success: True
            message: Successfully registered subnet
            extrinsic_function: register_subnet_extrinsic
            extrinsic: {'account_id': '0xd43593c715fdd31c...
            transaction_fee: τ1.0
            extrinsic_receipt: Extrinsic Receipt data of of the submitted extrinsic
            mev_extrinsic: None
            transaction_tao_fee: τ1.0
            transaction_alpha_fee: 1.0β
            error: None
            data: None

        success, message = response
        print(success, message)

        True Successfully registered subnet

        print(response[0])
        True
        print(response[1])
        'Successfully registered subnet'
    """

    success: bool = True
    message: Optional[str] = None
    extrinsic_function: Optional[str] = None
    extrinsic: Optional["GenericExtrinsic"] = None
    extrinsic_fee: Optional["Balance"] = None
    extrinsic_receipt: Optional["AsyncExtrinsicReceipt | ExtrinsicReceipt"] = None
    mev_extrinsic: Optional["AsyncExtrinsicReceipt | ExtrinsicReceipt"] = None
    transaction_tao_fee: Optional["Balance"] = None
    transaction_alpha_fee: Optional["Balance"] = None
    error: Optional[Exception] = None
    data: Optional[Any] = None

    def __iter__(self):
        yield self.success
        yield self.message

    def __str__(self):
        _extrinsic_receipt = (
            f"ExtrinsicReceipt<hash:{self.extrinsic_receipt.extrinsic_hash}>\n"
            if self.extrinsic_receipt
            else f"{self.extrinsic_receipt}\n"
        )
        return (
            f"{self.__class__.__name__}:\n"
            f"\tsuccess: {self.success}\n"
            f"\tmessage: {self.message}\n"
            f"\textrinsic_function: {self.extrinsic_function}\n"
            f"\textrinsic: {self.extrinsic}\n"
            f"\textrinsic_fee: {self.extrinsic_fee}\n"
            f"\textrinsic_receipt: {_extrinsic_receipt}\n"
            f"\tmev_extrinsic: {self.mev_extrinsic}\n"
            f"\ttransaction_tao_fee: {self.transaction_tao_fee}\n"
            f"\ttransaction_alpha_fee: {self.transaction_alpha_fee}\n"
            f"\terror: {self.error}\n"
            f"\tdata: {self.data}\n"
        )

    def __repr__(self):
        return repr((self.success, self.message))

    def as_dict(self) -> dict:
        """Represents this object as a dictionary."""
        return {
            "success": self.success,
            "message": self.message,
            "extrinsic_function": self.extrinsic_function,
            "extrinsic": self.extrinsic,
            "extrinsic_fee": self.extrinsic_fee.rao if self.extrinsic_fee else None,
            "transaction_tao_fee": self.transaction_tao_fee.rao
            if self.transaction_tao_fee
            else None,
            "transaction_alpha_fee": str(self.transaction_alpha_fee)
            if self.transaction_alpha_fee
            else None,
            "extrinsic_receipt": self.extrinsic_receipt,
            "error": str(self.error) if self.error else None,
            "data": self.data,
        }

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, (tuple, list)):
            return (self.success, self.message) == tuple(other)
        if isinstance(other, ExtrinsicResponse):
            return (
                self.success == other.success
                and self.message == other.message
                and self.extrinsic_function == other.extrinsic_function
                and self.extrinsic == other.extrinsic
                and self.extrinsic_fee == other.extrinsic_fee
                and self.transaction_tao_fee == other.transaction_tao_fee
                and self.transaction_alpha_fee == other.transaction_alpha_fee
                and self.extrinsic_receipt == other.extrinsic_receipt
                and self.error == other.error
                and self.data == other.data
            )
        return super().__eq__(other)

    def __getitem__(self, index: int) -> Any:
        if index == 0:
            return self.success
        elif index == 1:
            return self.message
        else:
            raise IndexError(
                "ExtrinsicResponse only supports indices 0 (success) and 1 (message)."
            )

    def __len__(self):
        return 2

    def __post_init__(self):
        if self.extrinsic_function is None:
            self.extrinsic_function = get_caller_name(depth=3)

    @classmethod
    def unlock_wallet(
        cls,
        wallet: "Wallet",
        raise_error: bool = False,
        unlock_type: str = "coldkey",
        nonce_key: Optional[str] = None,
    ) -> "ExtrinsicResponse":
        """Check if keypair is unlocked and return ExtrinsicResponse accordingly.

        Parameters:
            wallet: Bittensor Wallet instance.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            unlock_type: The key type, 'coldkey' or 'hotkey'. Or 'both' to check both.
            nonce_key: Key used for generating nonce in extrinsic function.

        Returns:
            Extrinsic Response is used to check if the key is unlocked.

        Note:
            When an extrinsic is signed with the coldkey but internally references or uses the hotkey, both keypairs
            must be validated. Passing unlock_type='both' ensures that authentication is performed against both the
            coldkey and hotkey.
        """
        both = ["coldkey", "hotkey"]
        keys = [unlock_type] if unlock_type in both else both
        unlock = UnlockStatus(False, "")

        for unlock_type in keys:
            unlock = unlock_key(
                wallet, unlock_type=unlock_type, raise_error=raise_error
            )
            if not unlock.success:
                logging.error(unlock.message)

        # If extrinsic uses `unlock_type` and `nonce_key` and `nonce_key` is not public, we need to check the
        # availability of both keys.
        if nonce_key and nonce_key != unlock_type and "pub" not in nonce_key:
            nonce_key_unlock = unlock_key(
                wallet, unlock_type=nonce_key, raise_error=raise_error
            )
            if not nonce_key_unlock.success:
                logging.error(nonce_key_unlock.message)

            return cls(
                success=all([unlock.success, nonce_key_unlock.success]),
                message=unlock.message,
                extrinsic_function=get_caller_name(),
            )

        return cls(
            success=unlock.success,
            message=unlock.message,
            extrinsic_function=get_caller_name(),
        )

    @classmethod
    def from_exception(cls, raise_error: bool, error: Exception) -> "ExtrinsicResponse":
        """Check if error is raised and return ExtrinsicResponse accordingly.
        Parameters:
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            error: Exception raised during extrinsic execution.

        Returns:
            Extrinsic Response with False checks whether to raise an error or simply return the instance.
        """
        if raise_error:
            raise error
        return cls(
            success=False,
            message=format_error_message(error),
            error=error,
            extrinsic_function=get_caller_name(),
        ).with_log()

    def with_log(
        self,
        level: Literal[
            "trace", "debug", "info", "warning", "error", "success"
        ] = "error",
    ) -> "ExtrinsicResponse":
        """Logs provided message with provided level.

        Parameters:
            level: Logging level represented as "trace", "debug", "info", "warning", "error", "success" uses to logging
                message.

        Returns:
            ExtrinsicResponse instance.
        """
        if self.message:
            if level in ["trace", "error"]:
                message = f"[red]{self.message}[/red]"
            elif level == "info":
                message = f"[blue]{self.message}[/blue]"
            elif level == "warning":
                message = f"[yellow]{self.message}[/yellow]"
            elif level == "success":
                message = f"[green]{self.message}[/green]"
            else:
                message = self.message
            getattr(logging, level)(message)
        return self


@dataclass
class BlockInfo:
    """
    Class that holds information about a blockchain block.

    This class encapsulates all relevant information about a block in the blockchain, including its number, hash,
    timestamp, and contents.

    Attributes:
        number: The block number.
        hash: The corresponding block hash.
        timestamp: The timestamp of the block (based on the `Timestamp.Now` extrinsic).
        header: The raw block header returned by the node RPC.
        extrinsics: The list of extrinsics included in the block.
        explorer: The link to block explorer service.
    """

    number: int
    hash: str
    timestamp: Optional[int]
    header: dict
    extrinsics: list
    explorer: str
