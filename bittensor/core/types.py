import argparse
from abc import ABC
from dataclasses import dataclass
from typing import Any, TypedDict, Optional, Union

from scalecodec.types import GenericExtrinsic

import numpy as np
from numpy.typing import NDArray

from bittensor.core import settings
from bittensor.core.chain_data import NeuronInfo, NeuronInfoLite
from bittensor.core.config import Config
from bittensor.utils import determine_chain_endpoint_and_network, networking, Certificate
from bittensor.utils.btlogging import logging

# Type annotations for UIDs and weights.
UIDs = Union[NDArray[np.int64], list[Union[int]]]
Weights = Union[NDArray[np.float32], list[Union[int, float]]]
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

    @staticmethod  # TODO can this be a class method?
    def config() -> "Config":
        """
        Creates and returns a Bittensor configuration object.

        Returns:
            config (bittensor.core.config.Config): A Bittensor configuration object configured with arguments added by
                the `subtensor.add_args` method.
        """
        parser = argparse.ArgumentParser()
        SubtensorMixin.add_args(parser)
        return Config(parser)

    @staticmethod
    def setup_config(network: Optional[str], config: "Config"):
        """
        Sets up and returns the configuration for the Subtensor network and endpoint.

        This method determines the appropriate network and chain endpoint based on the provided network string or
            configuration object. It evaluates the network and endpoint in the following order of precedence:
            1. Provided network string.
            2. Configured chain endpoint in the `config` object.
            3. Configured network in the `config` object.
            4. Default chain endpoint.
            5. Default network.

        Arguments:
            network (Optional[str]): The name of the Subtensor network. If None, the network and endpoint will be
                determined from the `config` object.
            config (bittensor.core.config.Config): The configuration object containing the network and chain endpoint
                settings.

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

        Arguments:
            parser (argparse.ArgumentParser): The ArgumentParser object to which the Subtensor arguments will be added.
            prefix (Optional[str]): An optional prefix for the argument names. If provided, the prefix is prepended to
                each argument name.

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

    def dict(self) -> dict:
        """
        Returns a dict representation of this object. If `self.certificate` is `None`,
        it is not included in this.
        """
        d = {
            "version": self.version,
            "ip": self.ip,
            "port": self.port,
            "ip_type": self.ip_type,
            "netuid": self.netuid,
            "hotkey": self.hotkey,
            "coldkey": self.coldkey,
            "protocol": self.protocol,
            "placeholder1": self.placeholder1,
            "placeholder2": self.placeholder2,
        }
        if self.certificate is not None:
            d["certificate"] = self.certificate
        return d


class PrometheusServeCallParams(TypedDict):
    """Prometheus serve chain call parameters."""

    version: int
    ip: int
    port: int
    ip_type: int
    netuid: int


class ParamWithTypes(TypedDict):
    name: str  # Name of the parameter.
    type: str  # ScaleType string of the parameter.


@dataclass
class ExtrinsicResponse:
    """
    A standardized response container for handling the extrinsic results submissions and related operations in the SDK.

    This class is designed to give developers a consistent way to represent the outcome of an extrinsic call — whether
    it succeeded or failed — along with useful metadata for debugging, logging, or higher-level business logic.

    Attributes:
        success: Indicates if the extrinsic execution was successful.
        message: A status or informational message returned from the execution (e.g., "Successfully registered subnet").
        error: Captures the underlying exception if the extrinsic failed, otherwise `None`.
        data: Arbitrary data returned from the extrinsic, such as decoded events, or extra context.
        extrinsic_function: The name of the SDK extrinsic function that was executed (e.g. "register_subnet_extrinsic").
        extrinsic: The raw extrinsic object used in the call, if available.

    Example:
        import bittensor as bt

        subtensor = bt.SubtensorApi("local")
        wallet = bt.Wallet("alice")

        response = subtensor.subnets.register_subnet(alice_wallet)
        print(response)

        ExtrinsicResponse:
            success: True
            message: Successfully registered subnet
            error: None
            extrinsic_function: register_subnet_extrinsic
            extrinsic: {'account_id': '0xd43593c715fdd31c...

        success, message = response
        print(success, message)

        True Successfully registered subnet
    """

    success: bool = True
    message: str = None
    error: Optional[Exception] = None
    data: Optional[Any] = None
    extrinsic_function: Optional[str] = None
    extrinsic: Optional[GenericExtrinsic] = None

    def __iter__(self):
        yield self.success
        yield self.message

    def __str__(self):
        return str(
            f"ExtrinsicResponse:"
            f"\n\tsuccess: {self.success}"
            f"\n\tmessage: {self.message}"
            f"\n\terror: {self.error}"
            f"\n\textrinsic_function: {self.extrinsic_function}"
            f"\n\textrinsic: {self.extrinsic}"
            f"\n\tdata: {self.data}"
        )

    def __repr__(self):
        return repr((self.success, self.message))
