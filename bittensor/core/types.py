import asyncio
from abc import ABC
import argparse
from functools import partial
from itertools import cycle
from typing import TypedDict, Optional, Union, Type

from async_substrate_interface.sync_substrate import SubstrateInterface
from async_substrate_interface.async_substrate import AsyncSubstrateInterface
from async_substrate_interface.errors import MaxRetriesExceeded

from bittensor.utils import networking, Certificate
from bittensor.utils.btlogging import logging
from bittensor.core import settings
from bittensor.core.config import Config
from bittensor.core.chain_data import NeuronInfo, NeuronInfoLite
from bittensor.utils import determine_chain_endpoint_and_network

SubstrateClass = Type[Union[SubstrateInterface, AsyncSubstrateInterface]]


class RetrySubstrate:
    def __init__(
        self,
        substrate: SubstrateClass,
        main_url: str,
        ss58_format: int,
        type_registry: dict,
        use_remote_preset: bool,
        chain_name: str,
        _mock: bool,
        fallback_chains: Optional[list[str]] = None,
        retry_forever: bool = False,
    ):
        fallback_chains = fallback_chains or []
        self._substrate_class: SubstrateClass = substrate
        self.ss58_format: int = ss58_format
        self.type_registry: dict = type_registry
        self.use_remote_preset: bool = use_remote_preset
        self.chain_name: str = chain_name
        self._mock = _mock
        self.fallback_chains = (
            iter(fallback_chains)
            if not retry_forever
            else cycle(fallback_chains + [main_url])
        )
        initialized = False
        for chain_url in [main_url] + fallback_chains:
            try:
                self._substrate = self._substrate_class(
                    url=chain_url,
                    ss58_format=ss58_format,
                    type_registry=type_registry,
                    use_remote_preset=use_remote_preset,
                    chain_name=chain_name,
                    _mock=_mock,
                )
                initialized = True
                break
            except ConnectionError:
                logging.warning(f"Unable to connect to {chain_url}")
        if not initialized:
            raise ConnectionError(
                f"Unable to connect at any chains specified: {[main_url]+fallback_chains}"
            )

        # retries

        # TODO: properties that need retry logic
        # properties
        # version
        # token_decimals
        # token_symbol
        # name

        retry = (
            self._async_retry
            if self._substrate_class == AsyncSubstrateInterface
            else self._retry
        )

        self._get_block_handler = partial(retry, "_get_block_handler")
        self.apply_type_registry_presets = partial(retry, "apply_type_registry_presets")
        self.close = partial(retry, "close")
        self.compose_call = partial(retry, "compose_call")
        self.connect = partial(retry, "connect")
        self.create_scale_object = partial(retry, "create_scale_object")
        self.create_signed_extrinsic = partial(retry, "create_signed_extrinsic")
        self.create_storage_key = partial(retry, "create_storage_key")
        self.decode_scale = partial(retry, "decode_scale")
        self.encode_scale = partial(retry, "encode_scale")
        self.extension_call = partial(retry, "extension_call")
        self.filter_events = partial(retry, "filter_events")
        self.filter_extrinsics = partial(retry, "filter_extrinsics")
        self.generate_signature_payload = partial(retry, "generate_signature_payload")
        self.get_account_next_index = partial(retry, "get_account_next_index")
        self.get_account_nonce = partial(retry, "get_account_nonce")
        self.get_block = partial(retry, "get_block")
        self.get_block_hash = partial(retry, "get_block_hash")
        self.get_block_header = partial(retry, "get_block_header")
        self.get_block_metadata = partial(retry, "get_block_metadata")
        self.get_block_number = partial(retry, "get_block_number")
        self.get_block_runtime_info = partial(retry, "get_block_runtime_info")
        self.get_block_runtime_version_for = partial(
            retry, "get_block_runtime_version_for"
        )
        self.get_block_timestamp = partial(retry, "get_block_timestamp")
        self.get_chain_finalised_head = partial(retry, "get_chain_finalised_head")
        self.get_chain_head = partial(retry, "get_chain_head")
        self.get_constant = partial(retry, "get_constant")
        self.get_events = partial(retry, "get_events")
        self.get_extrinsics = partial(retry, "get_extrinsics")
        self.get_metadata_call_function = partial(retry, "get_metadata_call_function")
        self.get_metadata_constant = partial(retry, "get_metadata_constant")
        self.get_metadata_error = partial(retry, "get_metadata_error")
        self.get_metadata_errors = partial(retry, "get_metadata_errors")
        self.get_metadata_module = partial(retry, "get_metadata_module")
        self.get_metadata_modules = partial(retry, "get_metadata_modules")
        self.get_metadata_runtime_call_function = partial(
            retry, "get_metadata_runtime_call_function"
        )
        self.get_metadata_runtime_call_functions = partial(
            retry, "get_metadata_runtime_call_functions"
        )
        self.get_metadata_storage_function = partial(
            retry, "get_metadata_storage_function"
        )
        self.get_metadata_storage_functions = partial(
            retry, "get_metadata_storage_functions"
        )
        self.get_parent_block_hash = partial(retry, "get_parent_block_hash")
        self.get_payment_info = partial(retry, "get_payment_info")
        self.get_storage_item = partial(retry, "get_storage_item")
        self.get_type_definition = partial(retry, "get_type_definition")
        self.get_type_registry = partial(retry, "get_type_registry")
        self.init_runtime = partial(retry, "init_runtime")
        self.initialize = partial(retry, "initialize")
        self.is_valid_ss58_address = partial(retry, "is_valid_ss58_address")
        self.load_runtime = partial(retry, "load_runtime")
        self.make_payload = partial(retry, "make_payload")
        self.query = partial(retry, "query")
        self.query_map = partial(retry, "query_map")
        self.query_multi = partial(retry, "query_multi")
        self.query_multiple = partial(retry, "query_multiple")
        self.reload_type_registry = partial(retry, "reload_type_registry")
        self.retrieve_extrinsic_by_hash = partial(retry, "retrieve_extrinsic_by_hash")
        self.retrieve_extrinsic_by_identifier = partial(
            retry, "retrieve_extrinsic_by_identifier"
        )
        self.rpc_request = partial(retry, "rpc_request")
        self.runtime_call = partial(retry, "runtime_call")
        self.search_block_number = partial(retry, "search_block_number")
        self.serialize_constant = partial(retry, "serialize_constant")
        self.serialize_module_call = partial(retry, "serialize_module_call")
        self.serialize_module_error = partial(retry, "serialize_module_error")
        self.serialize_module_event = partial(retry, "serialize_module_event")
        self.serialize_storage_item = partial(retry, "serialize_storage_item")
        self.ss58_decode = partial(retry, "ss58_decode")
        self.ss58_encode = partial(retry, "ss58_encode")
        self.submit_extrinsic = partial(retry, "submit_extrinsic")
        self.subscribe_block_headers = partial(retry, "subscribe_block_headers")
        self.supports_rpc_method = partial(retry, "supports_rpc_method")
        self.ws = self._substrate.ws

    def _retry(self, method, *args, **kwargs):
        try:
            method_ = getattr(self._substrate, method)
            return method_(*args, **kwargs)
        except (MaxRetriesExceeded, ConnectionError, ConnectionRefusedError) as e:
            try:
                self._reinstantiate_substrate()
                method_ = getattr(self._substrate, method)
                return self._retry(method_(*args, **kwargs))
            except StopIteration:
                logging.error(
                    f"Max retries exceeded with {self._substrate.url}. No more fallback chains."
                )
                raise MaxRetriesExceeded

    async def _async_retry(self, method, *args, **kwargs):
        try:
            method_ = getattr(self._substrate, method)
            if asyncio.iscoroutinefunction(method_):
                return await method_(*args, **kwargs)
            else:
                return method_(*args, **kwargs)
        except (MaxRetriesExceeded, ConnectionError, ConnectionRefusedError) as e:
            try:
                self._reinstantiate_substrate(e)
                method_ = getattr(self._substrate, method)
                if asyncio.iscoroutinefunction(method_):
                    return await method_(*args, **kwargs)
                else:
                    return method_(*args, **kwargs)
            except StopIteration:
                logging.error(
                    f"Max retries exceeded with {self._substrate.url}. No more fallback chains."
                )
                raise MaxRetriesExceeded

    def _reinstantiate_substrate(self, e: Optional[Exception] = None) -> None:
        next_network = next(self.fallback_chains)
        if e.__class__ == MaxRetriesExceeded:
            logging.error(
                f"Max retries exceeded with {self._substrate.url}. Retrying with {next_network}."
            )
        else:
            print(f"Connection error. Trying again with {next_network}")
        self._substrate = self._substrate_class(
            url=next_network,
            ss58_format=self.ss58_format,
            type_registry=self.type_registry,
            use_remote_preset=self.use_remote_preset,
            chain_name=self.chain_name,
            _mock=self._mock,
        )


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
            default_network = settings.DEFAULT_NETWORK
            default_chain_endpoint = settings.FINNEY_ENTRYPOINT

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
