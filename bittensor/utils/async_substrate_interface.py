import asyncio
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from hashlib import blake2b
from typing import Optional, Any, Union, Callable, Awaitable, cast

from async_property import async_property
from bittensor_wallet import Keypair
from bt_decode import PortableRegistry, decode as decode_by_type_string, MetadataV15
from scalecodec import GenericExtrinsic
from scalecodec.base import ScaleBytes, ScaleType, RuntimeConfigurationObject
from scalecodec.type_registry import load_type_registry_preset
from scalecodec.types import GenericCall
from substrateinterface.exceptions import (
    SubstrateRequestException,
    ExtrinsicNotFound,
    BlockNotFound,
)
from substrateinterface.storage import StorageKey
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

ResultHandler = Callable[[dict, Any], Awaitable[tuple[dict, bool]]]


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out")


class ExtrinsicReceipt:
    """
    Object containing information of submitted extrinsic. Block hash where extrinsic is included is required
    when retrieving triggered events or determine if extrinsic was successful
    """

    def __init__(
        self,
        substrate: "AsyncSubstrateInterface",
        extrinsic_hash: Optional[str] = None,
        block_hash: Optional[str] = None,
        block_number: Optional[int] = None,
        extrinsic_idx: Optional[int] = None,
        finalized=None,
    ):
        """
        Object containing information of submitted extrinsic. Block hash where extrinsic is included is required
        when retrieving triggered events or determine if extrinsic was successful

        Parameters
        ----------
        substrate
        extrinsic_hash
        block_hash
        finalized
        """
        self.substrate = substrate
        self.extrinsic_hash = extrinsic_hash
        self.block_hash = block_hash
        self.block_number = block_number
        self.finalized = finalized

        self.__extrinsic_idx = extrinsic_idx
        self.__extrinsic = None

        self.__triggered_events: Optional[list] = None
        self.__is_success: Optional[bool] = None
        self.__error_message = None
        self.__weight = None
        self.__total_fee_amount = None

    async def get_extrinsic_identifier(self) -> str:
        """
        Returns the on-chain identifier for this extrinsic in format "[block_number]-[extrinsic_idx]" e.g. 134324-2
        Returns
        -------
        str
        """
        if self.block_number is None:
            if self.block_hash is None:
                raise ValueError(
                    "Cannot create extrinsic identifier: block_hash is not set"
                )

            self.block_number = await self.substrate.get_block_number(self.block_hash)

            if self.block_number is None:
                raise ValueError(
                    "Cannot create extrinsic identifier: unknown block_hash"
                )

        return f"{self.block_number}-{await self.extrinsic_idx}"

    async def retrieve_extrinsic(self):
        if not self.block_hash:
            raise ValueError(
                "ExtrinsicReceipt can't retrieve events because it's unknown which block_hash it is "
                "included, manually set block_hash or use `wait_for_inclusion` when sending extrinsic"
            )
        # Determine extrinsic idx

        block = await self.substrate.get_block(block_hash=self.block_hash)

        extrinsics = block["extrinsics"]

        if len(extrinsics) > 0:
            if self.__extrinsic_idx is None:
                self.__extrinsic_idx = self.__get_extrinsic_index(
                    block_extrinsics=extrinsics, extrinsic_hash=self.extrinsic_hash
                )

            if self.__extrinsic_idx >= len(extrinsics):
                raise ExtrinsicNotFound()

            self.__extrinsic = extrinsics[self.__extrinsic_idx]

    @async_property
    async def extrinsic_idx(self) -> int:
        """
        Retrieves the index of this extrinsic in containing block

        Returns
        -------
        int
        """
        if self.__extrinsic_idx is None:
            await self.retrieve_extrinsic()
        return self.__extrinsic_idx

    @async_property
    async def triggered_events(self) -> list:
        """
        Gets triggered events for submitted extrinsic. block_hash where extrinsic is included is required, manually
        set block_hash or use `wait_for_inclusion` when submitting extrinsic

        Returns
        -------
        list
        """
        if self.__triggered_events is None:
            if not self.block_hash:
                raise ValueError(
                    "ExtrinsicReceipt can't retrieve events because it's unknown which block_hash it is "
                    "included, manually set block_hash or use `wait_for_inclusion` when sending extrinsic"
                )

            if await self.extrinsic_idx is None:
                await self.retrieve_extrinsic()

            self.__triggered_events = []

            for event in await self.substrate.get_events(block_hash=self.block_hash):
                if event["extrinsic_idx"] == await self.extrinsic_idx:
                    self.__triggered_events.append(event)

        return cast(list, self.__triggered_events)

    async def process_events(self):
        if await self.triggered_events:
            self.__total_fee_amount = 0

            # Process fees
            has_transaction_fee_paid_event = False

            for event in await self.triggered_events:
                if (
                    event["event"]["module_id"] == "TransactionPayment"
                    and event["event"]["event_id"] == "TransactionFeePaid"
                ):
                    self.__total_fee_amount = event["event"]["attributes"]["actual_fee"]
                    has_transaction_fee_paid_event = True

            # Process other events
            for event in await self.triggered_events:
                # Check events
                if (
                    event["event"]["module_id"] == "System"
                    and event["event"]["event_id"] == "ExtrinsicSuccess"
                ):
                    self.__is_success = True
                    self.__error_message = None

                    if "dispatch_info" in event["event"]["attributes"]:
                        self.__weight = event["event"]["attributes"]["dispatch_info"][
                            "weight"
                        ]
                    else:
                        # Backwards compatibility
                        self.__weight = event["event"]["attributes"]["weight"]

                elif (
                    event["event"]["module_id"] == "System"
                    and event["event"]["event_id"] == "ExtrinsicFailed"
                ):
                    self.__is_success = False

                    dispatch_info = event["event"]["attributes"]["dispatch_info"]
                    dispatch_error = event["event"]["attributes"]["dispatch_error"]

                    self.__weight = dispatch_info["weight"]

                    if "Module" in dispatch_error:
                        module_index = dispatch_error["Module"][0]["index"]
                        error_index = int.from_bytes(
                            bytes(dispatch_error["Module"][0]["error"]),
                            byteorder="little",
                            signed=False,
                        )

                        if isinstance(error_index, str):
                            # Actual error index is first u8 in new [u8; 4] format
                            error_index = int(error_index[2:4], 16)
                        module_error = self.substrate.metadata.get_module_error(
                            module_index=module_index, error_index=error_index
                        )
                        self.__error_message = {
                            "type": "Module",
                            "name": module_error.name,
                            "docs": module_error.docs,
                        }
                    elif "BadOrigin" in dispatch_error:
                        self.__error_message = {
                            "type": "System",
                            "name": "BadOrigin",
                            "docs": "Bad origin",
                        }
                    elif "CannotLookup" in dispatch_error:
                        self.__error_message = {
                            "type": "System",
                            "name": "CannotLookup",
                            "docs": "Cannot lookup",
                        }
                    elif "Other" in dispatch_error:
                        self.__error_message = {
                            "type": "System",
                            "name": "Other",
                            "docs": "Unspecified error occurred",
                        }

                elif not has_transaction_fee_paid_event:
                    if (
                        event["event"]["module_id"] == "Treasury"
                        and event["event"]["event_id"] == "Deposit"
                    ):
                        self.__total_fee_amount += event["event"]["attributes"]["value"]
                    elif (
                        event["event"]["module_id"] == "Balances"
                        and event["event"]["event_id"] == "Deposit"
                    ):
                        self.__total_fee_amount += event.value["attributes"]["amount"]

    @async_property
    async def is_success(self) -> bool:
        """
        Returns `True` if `ExtrinsicSuccess` event is triggered, `False` in case of `ExtrinsicFailed`
        In case of False `error_message` will contain more details about the error


        Returns
        -------
        bool
        """
        if self.__is_success is None:
            await self.process_events()

        return cast(bool, self.__is_success)

    @async_property
    async def error_message(self) -> Optional[dict]:
        """
        Returns the error message if the extrinsic failed in format e.g.:

        `{'type': 'System', 'name': 'BadOrigin', 'docs': 'Bad origin'}`

        Returns
        -------
        dict
        """
        if self.__error_message is None:
            if await self.is_success:
                return None
            await self.process_events()
        return self.__error_message

    @async_property
    async def weight(self) -> Union[int, dict]:
        """
        Contains the actual weight when executing this extrinsic

        Returns
        -------
        int (WeightV1) or dict (WeightV2)
        """
        if self.__weight is None:
            await self.process_events()
        return self.__weight

    @async_property
    async def total_fee_amount(self) -> int:
        """
        Contains the total fee costs deducted when executing this extrinsic. This includes fee for the validator (
        (`Balances.Deposit` event) and the fee deposited for the treasury (`Treasury.Deposit` event)

        Returns
        -------
        int
        """
        if self.__total_fee_amount is None:
            await self.process_events()
        return cast(int, self.__total_fee_amount)

    # Helper functions
    @staticmethod
    def __get_extrinsic_index(block_extrinsics: list, extrinsic_hash: str) -> int:
        """
        Returns the index of a provided extrinsic
        """
        for idx, extrinsic in enumerate(block_extrinsics):
            if (
                extrinsic.extrinsic_hash
                and f"0x{extrinsic.extrinsic_hash.hex()}" == extrinsic_hash
            ):
                return idx
        raise ExtrinsicNotFound()

    # Backwards compatibility methods
    def __getitem__(self, item):
        return getattr(self, item)

    def __iter__(self):
        for item in self.__dict__.items():
            yield item

    def get(self, name):
        return self[name]


class QueryMapResult:
    def __init__(
        self,
        records: list,
        page_size: int,
        substrate: "AsyncSubstrateInterface",
        module: Optional[str] = None,
        storage_function: Optional[str] = None,
        params: Optional[list] = None,
        block_hash: Optional[str] = None,
        last_key: Optional[str] = None,
        max_results: Optional[int] = None,
        ignore_decoding_errors: bool = False,
    ):
        self.records = records
        self.page_size = page_size
        self.module = module
        self.storage_function = storage_function
        self.block_hash = block_hash
        self.substrate = substrate
        self.last_key = last_key
        self.max_results = max_results
        self.params = params
        self.ignore_decoding_errors = ignore_decoding_errors
        self.loading_complete = False
        self._buffer = iter(self.records)  # Initialize the buffer with initial records

    async def retrieve_next_page(self, start_key) -> list:
        result = await self.substrate.query_map(
            module=self.module,
            storage_function=self.storage_function,
            params=self.params,
            page_size=self.page_size,
            block_hash=self.block_hash,
            start_key=start_key,
            max_results=self.max_results,
            ignore_decoding_errors=self.ignore_decoding_errors,
        )

        # Update last key from new result set to use as offset for next page
        self.last_key = result.last_key
        return result.records

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            # Try to get the next record from the buffer
            return next(self._buffer)
        except StopIteration:
            # If no more records in the buffer, try to fetch the next page
            if self.loading_complete:
                raise StopAsyncIteration

            next_page = await self.retrieve_next_page(self.last_key)
            if not next_page:
                self.loading_complete = True
                raise StopAsyncIteration

            # Update the buffer with the newly fetched records
            self._buffer = iter(next_page)
            return next(self._buffer)

    def __getitem__(self, item):
        return self.records[item]


@dataclass
class Preprocessed:
    queryable: str
    method: str
    params: list
    value_scale_type: str
    storage_item: ScaleType


class RuntimeCache:
    blocks: dict[int, "Runtime"]
    block_hashes: dict[str, "Runtime"]

    def __init__(self):
        self.blocks = {}
        self.block_hashes = {}

    def add_item(
        self, block: Optional[int], block_hash: Optional[str], runtime: "Runtime"
    ):
        if block is not None:
            self.blocks[block] = runtime
        if block_hash is not None:
            self.block_hashes[block_hash] = runtime

    def retrieve(
        self, block: Optional[int] = None, block_hash: Optional[str] = None
    ) -> Optional["Runtime"]:
        if block is not None:
            return self.blocks.get(block)
        elif block_hash is not None:
            return self.block_hashes.get(block_hash)
        else:
            return None


class Runtime:
    block_hash: str
    block_id: int
    runtime_version = None
    transaction_version = None
    cache_region = None
    metadata = None
    type_registry_preset = None

    def __init__(self, chain, runtime_config, metadata, type_registry):
        self.runtime_config = RuntimeConfigurationObject()
        self.config = {}
        self.chain = chain
        self.type_registry = type_registry
        self.runtime_config = runtime_config
        self.metadata = metadata

    @property
    def implements_scaleinfo(self) -> bool:
        """
        Returns True if current runtime implementation a `PortableRegistry` (`MetadataV14` and higher)
        """
        if self.metadata:
            return self.metadata.portable_registry is not None
        else:
            return False

    def reload_type_registry(
        self, use_remote_preset: bool = True, auto_discover: bool = True
    ):
        """
        Reload type registry and preset used to instantiate the SubstrateInterface object. Useful to periodically apply
        changes in type definitions when a runtime upgrade occurred

        Parameters
        ----------
        use_remote_preset: When True preset is downloaded from Github master, otherwise use files from local installed
                           scalecodec package
        auto_discover

        Returns
        -------

        """
        self.runtime_config.clear_type_registry()

        self.runtime_config.implements_scale_info = self.implements_scaleinfo

        # Load metadata types in runtime configuration
        self.runtime_config.update_type_registry(load_type_registry_preset(name="core"))
        self.apply_type_registry_presets(
            use_remote_preset=use_remote_preset, auto_discover=auto_discover
        )

    def apply_type_registry_presets(
        self,
        use_remote_preset: bool = True,
        auto_discover: bool = True,
    ):
        """
        Applies type registry presets to the runtime
        :param use_remote_preset: bool, whether to use presets from remote
        :param auto_discover: bool, whether to use presets from local installed scalecodec package
        """
        if self.type_registry_preset is not None:
            # Load type registry according to preset
            type_registry_preset_dict = load_type_registry_preset(
                name=self.type_registry_preset, use_remote_preset=use_remote_preset
            )

            if not type_registry_preset_dict:
                raise ValueError(
                    f"Type registry preset '{self.type_registry_preset}' not found"
                )

        elif auto_discover:
            # Try to auto discover type registry preset by chain name
            type_registry_name = self.chain.lower().replace(" ", "-")
            try:
                type_registry_preset_dict = load_type_registry_preset(
                    type_registry_name
                )
                self.type_registry_preset = type_registry_name
            except ValueError:
                type_registry_preset_dict = None

        else:
            type_registry_preset_dict = None

        if type_registry_preset_dict:
            # Load type registries in runtime configuration
            if self.implements_scaleinfo is False:
                # Only runtime with no embedded types in metadata need the default set of explicit defined types
                self.runtime_config.update_type_registry(
                    load_type_registry_preset(
                        "legacy", use_remote_preset=use_remote_preset
                    )
                )

            if self.type_registry_preset != "legacy":
                self.runtime_config.update_type_registry(type_registry_preset_dict)

        if self.type_registry:
            # Load type registries in runtime configuration
            self.runtime_config.update_type_registry(self.type_registry)


class RequestManager:
    RequestResults = dict[Union[str, int], list[Union[ScaleType, dict]]]

    def __init__(self, payloads):
        self.response_map = {}
        self.responses = defaultdict(lambda: {"complete": False, "results": []})
        self.payloads_count = len(payloads)

    def add_request(self, item_id: int, request_id: Any):
        """
        Adds an outgoing request to the responses map for later retrieval
        """
        self.response_map[item_id] = request_id

    def overwrite_request(self, item_id: int, request_id: Any):
        """
        Overwrites an existing request in the responses map with a new request_id. This is used
        for multipart responses that generate a subscription id we need to watch, rather than the initial
        request_id.
        """
        self.response_map[request_id] = self.response_map.pop(item_id)
        return request_id

    def add_response(self, item_id: int, response: dict, complete: bool):
        """
        Maps a response to the request for later retrieval
        """
        request_id = self.response_map[item_id]
        self.responses[request_id]["results"].append(response)
        self.responses[request_id]["complete"] = complete

    @property
    def is_complete(self) -> bool:
        """
        Returns whether all requests in the manager have completed
        """
        return (
            all(info["complete"] for info in self.responses.values())
            and len(self.responses) == self.payloads_count
        )

    def get_results(self) -> RequestResults:
        """
        Generates a dictionary mapping the requests initiated to the responses received.
        """
        return {
            request_id: info["results"] for request_id, info in self.responses.items()
        }


class Websocket:
    def __init__(
        self,
        ws_url: str,
        max_subscriptions=1024,
        max_connections=100,
        shutdown_timer=5,
        options: Optional[dict] = None,
    ):
        """
        Websocket manager object. Allows for the use of a single websocket connection by multiple
        calls.

        :param ws_url: Websocket URL to connect to
        :param max_subscriptions: Maximum number of subscriptions per websocket connection
        :param max_connections: Maximum number of connections total
        :param shutdown_timer: Number of seconds to shut down websocket connection after last use
        """
        # TODO allow setting max concurrent connections and rpc subscriptions per connection
        # TODO reconnection logic
        self.ws_url = ws_url
        self.ws: Optional[ClientConnection] = None
        self.id = 0
        self.max_subscriptions = max_subscriptions
        self.max_connections = max_connections
        self.shutdown_timer = shutdown_timer
        self._received = {}
        self._in_use = 0
        self._receiving_task = None
        self._attempts = 0
        self._initialized = False
        self._lock = asyncio.Lock()
        self._exit_task = None
        self._open_subscriptions = 0
        self._options = options if options else {}

    async def __aenter__(self):
        async with self._lock:
            self._in_use += 1
            if self._exit_task:
                self._exit_task.cancel()
            if not self._initialized:
                self._initialized = True
                await self._connect()
                self._receiving_task = asyncio.create_task(self._start_receiving())
        return self

    async def _connect(self):
        self.ws = await asyncio.wait_for(
            connect(self.ws_url, **self._options), timeout=10
        )

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        async with self._lock:
            self._in_use -= 1
            if self._exit_task is not None:
                self._exit_task.cancel()
                try:
                    await self._exit_task
                except asyncio.CancelledError:
                    pass
            if self._in_use == 0 and self.ws is not None:
                self.id = 0
                self._open_subscriptions = 0
                self._exit_task = asyncio.create_task(self._exit_with_timer())

    async def _exit_with_timer(self):
        """
        Allows for graceful shutdown of websocket connection after specified number of seconds, allowing
        for reuse of the websocket connection.
        """
        try:
            await asyncio.sleep(self.shutdown_timer)
            await self.shutdown()
        except asyncio.CancelledError:
            pass

    async def shutdown(self):
        async with self._lock:
            try:
                self._receiving_task.cancel()
                await self._receiving_task
                await self.ws.close()
            except (AttributeError, asyncio.CancelledError):
                pass
            self.ws = None
            self._initialized = False
            self._receiving_task = None
            self.id = 0

    async def _recv(self) -> None:
        try:
            response = json.loads(await cast(ClientProtocol, self.ws).recv())
            async with self._lock:
                self._open_subscriptions -= 1
            if "id" in response:
                self._received[response["id"]] = response
            elif "params" in response:
                self._received[response["params"]["subscription"]] = response
            else:
                raise KeyError(response)
        except ConnectionClosed:
            raise
        except KeyError as e:
            raise e

    async def _start_receiving(self):
        try:
            while True:
                await self._recv()
        except asyncio.CancelledError:
            pass
        except ConnectionClosed:
            # TODO try reconnect, but only if it's needed
            raise

    async def send(self, payload: dict) -> int:
        """
        Sends a payload to the websocket connection.

        :param payload: payload, generate a payload with the AsyncSubstrateInterface.make_payload method
        """
        async with self._lock:
            original_id = self.id
            self.id += 1
            self._open_subscriptions += 1
        try:
            await self.ws.send(json.dumps({**payload, **{"id": original_id}}))
            return original_id
        except ConnectionClosed:
            raise

    async def retrieve(self, item_id: int) -> Optional[dict]:
        """
        Retrieves a single item from received responses dict queue

        :param item_id: id of the item to retrieve

        :return: retrieved item
        """
        while True:
            async with self._lock:
                if item_id in self._received:
                    return self._received.pop(item_id)
            await asyncio.sleep(0.1)


class AsyncSubstrateInterface:
    runtime = None
    registry: Optional[PortableRegistry] = None

    def __init__(
        self,
        chain_endpoint: str,
        use_remote_preset=False,
        auto_discover=True,
        auto_reconnect=True,
        ss58_format=None,
        type_registry=None,
        chain_name=None,
    ):
        """
        The asyncio-compatible version of the subtensor interface commands we use in bittensor
        """
        self.chain_endpoint = chain_endpoint
        self.__chain = chain_name
        options = {
            "max_size": 2**32,
            "write_limit": 2**16,
        }
        self.ws = Websocket(chain_endpoint, options=options)
        self._lock = asyncio.Lock()
        self.last_block_hash: Optional[str] = None
        self.config = {
            "use_remote_preset": use_remote_preset,
            "auto_discover": auto_discover,
            "auto_reconnect": auto_reconnect,
            "rpc_methods": None,
            "strict_scale_decode": True,
        }
        self.initialized = False
        self._forgettable_task = None
        self.ss58_format = ss58_format
        self.type_registry = type_registry
        self.runtime_cache = RuntimeCache()
        self.block_id: Optional[int] = None
        self.runtime_version = None
        self.runtime_config = RuntimeConfigurationObject()
        self.__metadata_cache = {}
        self.type_registry_preset = None
        self.transaction_version = None
        self.metadata = None
        self.metadata_version_hex = "0x0f000000"  # v15

    async def __aenter__(self):
        await self.initialize()

    async def initialize(self):
        """
        Initialize the connection to the chain.
        """
        async with self._lock:
            if not self.initialized:
                if not self.__chain:
                    chain = await self.rpc_request("system_chain", [])
                    self.__chain = chain.get("result")
                self.reload_type_registry()
                await asyncio.gather(self.load_registry(), self.init_runtime(None))
            self.initialized = True

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    def chain(self):
        """
        Returns the substrate chain currently associated with object
        """
        return self.__chain

    async def get_storage_item(self, module: str, storage_function: str):
        if not self.metadata:
            await self.init_runtime()
        metadata_pallet = self.metadata.get_metadata_pallet(module)
        storage_item = metadata_pallet.get_storage_function(storage_function)
        return storage_item

    async def _get_current_block_hash(
        self, block_hash: Optional[str], reuse: bool
    ) -> Optional[str]:
        if block_hash:
            self.last_block_hash = block_hash
            return block_hash
        elif reuse:
            if self.last_block_hash:
                return self.last_block_hash
        return block_hash

    async def load_registry(self):
        metadata_rpc_result = await self.rpc_request(
            "state_call",
            ["Metadata_metadata_at_version", self.metadata_version_hex],
        )
        metadata_option_hex_str = metadata_rpc_result["result"]
        metadata_option_bytes = bytes.fromhex(metadata_option_hex_str[2:])
        metadata_v15 = MetadataV15.decode_from_metadata_option(metadata_option_bytes)
        self.registry = PortableRegistry.from_metadata_v15(metadata_v15)

    async def decode_scale(
        self, type_string, scale_bytes: bytes, return_scale_obj=False
    ):
        """
        Helper function to decode arbitrary SCALE-bytes (e.g. 0x02000000) according to given RUST type_string
        (e.g. BlockNumber). The relevant versioning information of the type (if defined) will be applied if block_hash
        is set

        Parameters
        ----------
        type_string
        scale_bytes
        block_hash
        return_scale_obj: if True the SCALE object itself is returned, otherwise the serialized dict value of the object

        Returns
        -------

        """
        if scale_bytes == b"\x00":
            obj = None
        else:
            obj = decode_by_type_string(type_string, self.registry, scale_bytes)
        return obj

    async def init_runtime(
        self, block_hash: Optional[str] = None, block_id: Optional[int] = None
    ) -> Runtime:
        """
        This method is used by all other methods that deals with metadata and types defined in the type registry.
        It optionally retrieves the block_hash when block_id is given and sets the applicable metadata for that
        block_hash. Also, it applies all the versioned types at the time of the block_hash.

        Because parsing of metadata and type registry is quite heavy, the result will be cached per runtime id.
        In the future there could be support for caching backends like Redis to make this cache more persistent.

        :param block_hash: optional block hash, should not be specified if block_id is
        :param block_id: optional block id, should not be specified if block_hash is

        :returns: Runtime object
        """

        async def get_runtime(block_hash, block_id) -> Runtime:
            # Check if runtime state already set to current block
            if (block_hash and block_hash == self.last_block_hash) or (
                block_id and block_id == self.block_id
            ):
                return Runtime(
                    self.chain,
                    self.runtime_config,
                    self.metadata,
                    self.type_registry,
                )

            if block_id is not None:
                block_hash = await self.get_block_hash(block_id)

            if not block_hash:
                block_hash = await self.get_chain_head()

            self.last_block_hash = block_hash
            self.block_id = block_id

            # In fact calls and storage functions are decoded against runtime of previous block, therefor retrieve
            # metadata and apply type registry of runtime of parent block
            block_header = await self.rpc_request(
                "chain_getHeader", [self.last_block_hash]
            )

            if block_header["result"] is None:
                raise SubstrateRequestException(
                    f'Block not found for "{self.last_block_hash}"'
                )

            parent_block_hash: str = block_header["result"]["parentHash"]

            if (
                parent_block_hash
                == "0x0000000000000000000000000000000000000000000000000000000000000000"
            ):
                runtime_block_hash = self.last_block_hash
            else:
                runtime_block_hash = parent_block_hash

            runtime_info = await self.get_block_runtime_version(
                block_hash=runtime_block_hash
            )

            if runtime_info is None:
                raise SubstrateRequestException(
                    f"No runtime information for block '{block_hash}'"
                )

            # Check if runtime state already set to current block
            if runtime_info.get("specVersion") == self.runtime_version:
                return Runtime(
                    self.chain,
                    self.runtime_config,
                    self.metadata,
                    self.type_registry,
                )

            self.runtime_version = runtime_info.get("specVersion")
            self.transaction_version = runtime_info.get("transactionVersion")

            if not self.metadata:
                if self.runtime_version in self.__metadata_cache:
                    # Get metadata from cache
                    # self.debug_message('Retrieved metadata for {} from memory'.format(self.runtime_version))
                    self.metadata = self.__metadata_cache[self.runtime_version]
                else:
                    self.metadata = await self.get_block_metadata(
                        block_hash=runtime_block_hash, decode=True
                    )
                    # self.debug_message('Retrieved metadata for {} from Substrate node'.format(self.runtime_version))

                    # Update metadata cache
                    self.__metadata_cache[self.runtime_version] = self.metadata

            # Update type registry
            self.reload_type_registry(use_remote_preset=False, auto_discover=True)

            if self.implements_scaleinfo:
                # self.debug_message('Add PortableRegistry from metadata to type registry')
                self.runtime_config.add_portable_registry(self.metadata)

            # Set active runtime version
            self.runtime_config.set_active_spec_version_id(self.runtime_version)

            # Check and apply runtime constants
            ss58_prefix_constant = await self.get_constant(
                "System", "SS58Prefix", block_hash=block_hash
            )

            if ss58_prefix_constant:
                self.ss58_format = ss58_prefix_constant

            # Set runtime compatibility flags
            try:
                _ = self.runtime_config.create_scale_object(
                    "sp_weights::weight_v2::Weight"
                )
                self.config["is_weight_v2"] = True
                self.runtime_config.update_type_registry_types(
                    {"Weight": "sp_weights::weight_v2::Weight"}
                )
            except NotImplementedError:
                self.config["is_weight_v2"] = False
                self.runtime_config.update_type_registry_types({"Weight": "WeightV1"})
            return Runtime(
                self.chain,
                self.runtime_config,
                self.metadata,
                self.type_registry,
            )

        if block_id and block_hash:
            raise ValueError("Cannot provide block_hash and block_id at the same time")

        if not (runtime := self.runtime_cache.retrieve(block_id, block_hash)):
            runtime = await get_runtime(block_hash, block_id)
            self.runtime_cache.add_item(block_id, block_hash, runtime)
        return runtime

    def reload_type_registry(
        self, use_remote_preset: bool = True, auto_discover: bool = True
    ):
        """
        Reload type registry and preset used to instantiate the SubtrateInterface object. Useful to periodically apply
        changes in type definitions when a runtime upgrade occurred

        Parameters
        ----------
        use_remote_preset: When True preset is downloaded from Github master, otherwise use files from local installed scalecodec package
        auto_discover

        Returns
        -------

        """
        self.runtime_config.clear_type_registry()

        self.runtime_config.implements_scale_info = self.implements_scaleinfo

        # Load metadata types in runtime configuration
        self.runtime_config.update_type_registry(load_type_registry_preset(name="core"))
        self.apply_type_registry_presets(
            use_remote_preset=use_remote_preset, auto_discover=auto_discover
        )

    def apply_type_registry_presets(
        self, use_remote_preset: bool = True, auto_discover: bool = True
    ):
        if self.type_registry_preset is not None:
            # Load type registry according to preset
            type_registry_preset_dict = load_type_registry_preset(
                name=self.type_registry_preset, use_remote_preset=use_remote_preset
            )

            if not type_registry_preset_dict:
                raise ValueError(
                    f"Type registry preset '{self.type_registry_preset}' not found"
                )

        elif auto_discover:
            # Try to auto discover type registry preset by chain name
            type_registry_name = self.chain.lower().replace(" ", "-")
            try:
                type_registry_preset_dict = load_type_registry_preset(
                    type_registry_name
                )
                # self.debug_message(f"Auto set type_registry_preset to {type_registry_name} ...")
                self.type_registry_preset = type_registry_name
            except ValueError:
                type_registry_preset_dict = None

        else:
            type_registry_preset_dict = None

        if type_registry_preset_dict:
            # Load type registries in runtime configuration
            if self.implements_scaleinfo is False:
                # Only runtime with no embedded types in metadata need the default set of explicit defined types
                self.runtime_config.update_type_registry(
                    load_type_registry_preset(
                        "legacy", use_remote_preset=use_remote_preset
                    )
                )

            if self.type_registry_preset != "legacy":
                self.runtime_config.update_type_registry(type_registry_preset_dict)

        if self.type_registry:
            # Load type registries in runtime configuration
            self.runtime_config.update_type_registry(self.type_registry)

    @property
    def implements_scaleinfo(self) -> Optional[bool]:
        """
        Returns True if current runtime implementation a `PortableRegistry` (`MetadataV14` and higher)

        Returns
        -------
        bool
        """
        if self.metadata:
            return self.metadata.portable_registry is not None
        else:
            return None

    async def create_storage_key(
        self,
        pallet: str,
        storage_function: str,
        params: Optional[list] = None,
        block_hash: str = None,
    ) -> StorageKey:
        """
        Create a `StorageKey` instance providing storage function details. See `subscribe_storage()`.

        Parameters
        ----------
        pallet: name of pallet
        storage_function: name of storage function
        params: Optional list of parameters in case of a Mapped storage function

        Returns
        -------
        StorageKey
        """
        await self.init_runtime(block_hash=block_hash)

        return StorageKey.create_from_storage_function(
            pallet,
            storage_function,
            params,
            runtime_config=self.runtime_config,
            metadata=self.metadata,
        )

    async def _get_block_handler(
        self,
        block_hash: str,
        ignore_decoding_errors: bool = False,
        include_author: bool = False,
        header_only: bool = False,
        finalized_only: bool = False,
        subscription_handler: Optional[Callable] = None,
    ):
        try:
            await self.init_runtime(block_hash=block_hash)
        except BlockNotFound:
            return None

        async def decode_block(block_data, block_data_hash=None):
            if block_data:
                if block_data_hash:
                    block_data["header"]["hash"] = block_data_hash

                if type(block_data["header"]["number"]) is str:
                    # Convert block number from hex (backwards compatibility)
                    block_data["header"]["number"] = int(
                        block_data["header"]["number"], 16
                    )

                extrinsic_cls = self.runtime_config.get_decoder_class("Extrinsic")

                if "extrinsics" in block_data:
                    for idx, extrinsic_data in enumerate(block_data["extrinsics"]):
                        extrinsic_decoder = extrinsic_cls(
                            data=ScaleBytes(extrinsic_data),
                            metadata=self.metadata,
                            runtime_config=self.runtime_config,
                        )
                        try:
                            extrinsic_decoder.decode(check_remaining=True)
                            block_data["extrinsics"][idx] = extrinsic_decoder

                        except Exception as e:
                            if not ignore_decoding_errors:
                                raise
                            block_data["extrinsics"][idx] = None

                for idx, log_data in enumerate(block_data["header"]["digest"]["logs"]):
                    if type(log_data) is str:
                        # Convert digest log from hex (backwards compatibility)
                        try:
                            log_digest_cls = self.runtime_config.get_decoder_class(
                                "sp_runtime::generic::digest::DigestItem"
                            )

                            if log_digest_cls is None:
                                raise NotImplementedError(
                                    "No decoding class found for 'DigestItem'"
                                )

                            log_digest = log_digest_cls(data=ScaleBytes(log_data))
                            log_digest.decode(
                                check_remaining=self.config.get("strict_scale_decode")
                            )

                            block_data["header"]["digest"]["logs"][idx] = log_digest

                            if include_author and "PreRuntime" in log_digest.value:
                                if self.implements_scaleinfo:
                                    engine = bytes(log_digest[1][0])
                                    # Retrieve validator set
                                    parent_hash = block_data["header"]["parentHash"]
                                    validator_set = await self.query(
                                        "Session", "Validators", block_hash=parent_hash
                                    )

                                    if engine == b"BABE":
                                        babe_predigest = (
                                            self.runtime_config.create_scale_object(
                                                type_string="RawBabePreDigest",
                                                data=ScaleBytes(
                                                    bytes(log_digest[1][1])
                                                ),
                                            )
                                        )

                                        babe_predigest.decode(
                                            check_remaining=self.config.get(
                                                "strict_scale_decode"
                                            )
                                        )

                                        rank_validator = babe_predigest[1].value[
                                            "authority_index"
                                        ]

                                        block_author = validator_set[rank_validator]
                                        block_data["author"] = block_author.value

                                    elif engine == b"aura":
                                        aura_predigest = (
                                            self.runtime_config.create_scale_object(
                                                type_string="RawAuraPreDigest",
                                                data=ScaleBytes(
                                                    bytes(log_digest[1][1])
                                                ),
                                            )
                                        )

                                        aura_predigest.decode(check_remaining=True)

                                        rank_validator = aura_predigest.value[
                                            "slot_number"
                                        ] % len(validator_set)

                                        block_author = validator_set[rank_validator]
                                        block_data["author"] = block_author.value
                                    else:
                                        raise NotImplementedError(
                                            f"Cannot extract author for engine {log_digest.value['PreRuntime'][0]}"
                                        )
                                else:
                                    if (
                                        log_digest.value["PreRuntime"]["engine"]
                                        == "BABE"
                                    ):
                                        validator_set = await self.query(
                                            "Session",
                                            "Validators",
                                            block_hash=block_hash,
                                        )
                                        rank_validator = log_digest.value["PreRuntime"][
                                            "data"
                                        ]["authority_index"]

                                        block_author = validator_set.elements[
                                            rank_validator
                                        ]
                                        block_data["author"] = block_author.value
                                    else:
                                        raise NotImplementedError(
                                            f"Cannot extract author for engine {log_digest.value['PreRuntime']['engine']}"
                                        )

                        except Exception:
                            if not ignore_decoding_errors:
                                raise
                            block_data["header"]["digest"]["logs"][idx] = None

            return block_data

        if callable(subscription_handler):
            rpc_method_prefix = "Finalized" if finalized_only else "New"

            async def result_handler(message, update_nr, subscription_id):
                new_block = await decode_block({"header": message["params"]["result"]})

                subscription_result = subscription_handler(
                    new_block, update_nr, subscription_id
                )

                if subscription_result is not None:
                    # Handler returned end result: unsubscribe from further updates
                    self._forgettable_task = asyncio.create_task(
                        self.rpc_request(
                            f"chain_unsubscribe{rpc_method_prefix}Heads",
                            [subscription_id],
                        )
                    )

                return subscription_result

            result = await self._make_rpc_request(
                [
                    self.make_payload(
                        "_get_block_handler",
                        f"chain_subscribe{rpc_method_prefix}Heads",
                        [],
                    )
                ],
                result_handler=result_handler,
            )

            return result

        else:
            if header_only:
                response = await self.rpc_request("chain_getHeader", [block_hash])
                return await decode_block(
                    {"header": response["result"]}, block_data_hash=block_hash
                )

            else:
                response = await self.rpc_request("chain_getBlock", [block_hash])
                return await decode_block(
                    response["result"]["block"], block_data_hash=block_hash
                )

    async def get_block(
        self,
        block_hash: Optional[str] = None,
        block_number: Optional[int] = None,
        ignore_decoding_errors: bool = False,
        include_author: bool = False,
        finalized_only: bool = False,
    ) -> Optional[dict]:
        """
        Retrieves a block and decodes its containing extrinsics and log digest items. If `block_hash` and `block_number`
        is omitted the chain tip will be retrieve, or the finalized head if `finalized_only` is set to true.

        Either `block_hash` or `block_number` should be set, or both omitted.

        Parameters
        ----------
        block_hash: the hash of the block to be retrieved
        block_number: the block number to retrieved
        ignore_decoding_errors: When set this will catch all decoding errors, set the item to None and continue decoding
        include_author: This will retrieve the block author from the validator set and add to the result
        finalized_only: when no `block_hash` or `block_number` is set, this will retrieve the finalized head

        Returns
        -------
        A dict containing the extrinsic and digest logs data
        """
        if block_hash and block_number:
            raise ValueError("Either block_hash or block_number should be be set")

        if block_number is not None:
            block_hash = await self.get_block_hash(block_number)

            if block_hash is None:
                return

        if block_hash and finalized_only:
            raise ValueError(
                "finalized_only cannot be True when block_hash is provided"
            )

        if block_hash is None:
            # Retrieve block hash
            if finalized_only:
                block_hash = await self.get_chain_finalised_head()
            else:
                block_hash = await self.get_chain_head()

        return await self._get_block_handler(
            block_hash=block_hash,
            ignore_decoding_errors=ignore_decoding_errors,
            header_only=False,
            include_author=include_author,
        )

    async def get_events(self, block_hash: Optional[str] = None) -> list:
        """
        Convenience method to get events for a certain block (storage call for module 'System' and function 'Events')

        Parameters
        ----------
        block_hash

        Returns
        -------
        list
        """

        def convert_event_data(data):
            # Extract phase information
            phase_key, phase_value = next(iter(data["phase"].items()))
            try:
                extrinsic_idx = phase_value[0]
            except IndexError:
                extrinsic_idx = None

            # Extract event details
            module_id, event_data = next(iter(data["event"].items()))
            event_id, attributes_data = next(iter(event_data[0].items()))

            # Convert class and pays_fee dictionaries to their string equivalents if they exist
            attributes = attributes_data
            if isinstance(attributes, dict):
                for key, value in attributes.items():
                    if isinstance(value, dict):
                        # Convert nested single-key dictionaries to their keys as strings
                        sub_key = next(iter(value.keys()))
                        if value[sub_key] == ():
                            attributes[key] = sub_key

            # Create the converted dictionary
            converted = {
                "phase": phase_key,
                "extrinsic_idx": extrinsic_idx,
                "event": {
                    "module_id": module_id,
                    "event_id": event_id,
                    "attributes": attributes,
                },
                "topics": list(data["topics"]),  # Convert topics tuple to a list
            }

            return converted

        events = []

        if not block_hash:
            block_hash = await self.get_chain_head()

        storage_obj = await self.query(
            module="System", storage_function="Events", block_hash=block_hash
        )
        if storage_obj:
            for item in list(storage_obj):
                # print("item!", item)
                events.append(convert_event_data(item))
            # events += list(storage_obj)
        return events

    async def get_block_runtime_version(self, block_hash: str) -> dict:
        """
        Retrieve the runtime version id of given block_hash
        """
        response = await self.rpc_request("state_getRuntimeVersion", [block_hash])
        return response.get("result")

    async def get_block_metadata(
        self, block_hash: Optional[str] = None, decode: bool = True
    ) -> Union[dict, ScaleType]:
        """
        A pass-though to existing JSONRPC method `state_getMetadata`.

        Parameters
        ----------
        block_hash
        decode: True for decoded version

        Returns
        -------

        """
        params = None
        if decode and not self.runtime_config:
            raise ValueError(
                "Cannot decode runtime configuration without a supplied runtime_config"
            )

        if block_hash:
            params = [block_hash]
        response = await self.rpc_request("state_getMetadata", params)

        if "error" in response:
            raise SubstrateRequestException(response["error"]["message"])

        if response.get("result") and decode:
            metadata_decoder = self.runtime_config.create_scale_object(
                "MetadataVersioned", data=ScaleBytes(response.get("result"))
            )
            metadata_decoder.decode()

            return metadata_decoder

        return response

    async def _preprocess(
        self,
        query_for: Optional[list],
        block_hash: Optional[str],
        storage_function: str,
        module: str,
    ) -> Preprocessed:
        """
        Creates a Preprocessed data object for passing to `_make_rpc_request`
        """
        params = query_for if query_for else []
        # Search storage call in metadata
        metadata_pallet = self.metadata.get_metadata_pallet(module)

        if not metadata_pallet:
            raise SubstrateRequestException(f'Pallet "{module}" not found')

        storage_item = metadata_pallet.get_storage_function(storage_function)

        if not metadata_pallet or not storage_item:
            raise SubstrateRequestException(
                f'Storage function "{module}.{storage_function}" not found'
            )

        # SCALE type string of value
        param_types = storage_item.get_params_type_string()
        value_scale_type = storage_item.get_value_type_string()

        if len(params) != len(param_types):
            raise ValueError(
                f"Storage function requires {len(param_types)} parameters, {len(params)} given"
            )

        storage_key = StorageKey.create_from_storage_function(
            module,
            storage_item.value["name"],
            params,
            runtime_config=self.runtime_config,
            metadata=self.metadata,
        )
        method = "state_getStorageAt"
        return Preprocessed(
            str(query_for),
            method,
            [storage_key.to_hex(), block_hash],
            value_scale_type,
            storage_item,
        )

    async def _process_response(
        self,
        response: dict,
        subscription_id: Union[int, str],
        value_scale_type: Optional[str] = None,
        storage_item: Optional[ScaleType] = None,
        runtime: Optional[Runtime] = None,
        result_handler: Optional[ResultHandler] = None,
    ) -> tuple[Union[ScaleType, dict], bool]:
        """
        Processes the RPC call response by decoding it, returning it as is, or setting a handler for subscriptions,
        depending on the specific call.

        :param response: the RPC call response
        :param subscription_id: the subscription id for subscriptions, used only for subscriptions with a result handler
        :param value_scale_type: Scale Type string used for decoding ScaleBytes results
        :param storage_item: The ScaleType object used for decoding ScaleBytes results
        :param runtime: the runtime object, used for decoding ScaleBytes results
        :param result_handler: the result handler coroutine used for handling longer-running subscriptions

        :return: (decoded response, completion)
        """
        result: Union[dict, ScaleType] = response
        if value_scale_type and isinstance(storage_item, ScaleType):
            if not runtime:
                async with self._lock:
                    runtime = Runtime(
                        self.chain,
                        self.runtime_config,
                        self.metadata,
                        self.type_registry,
                    )
            if response.get("result") is not None:
                query_value = response.get("result")
            elif storage_item.value["modifier"] == "Default":
                # Fallback to default value of storage function if no result
                query_value = storage_item.value_object["default"].value_object
            else:
                # No result is interpreted as an Option<...> result
                value_scale_type = f"Option<{value_scale_type}>"
                query_value = storage_item.value_object["default"].value_object
            if isinstance(query_value, str):
                q = bytes.fromhex(query_value[2:])
            elif isinstance(query_value, bytearray):
                q = bytes(query_value)
            else:
                q = query_value
            obj = await self.decode_scale(value_scale_type, q, True)
            result = obj
        if asyncio.iscoroutinefunction(result_handler):
            # For multipart responses as a result of subscriptions.
            message, bool_result = await result_handler(response, subscription_id)
            return message, bool_result
        return result, True

    async def _make_rpc_request(
        self,
        payloads: list[dict],
        value_scale_type: Optional[str] = None,
        storage_item: Optional[ScaleType] = None,
        runtime: Optional[Runtime] = None,
        result_handler: Optional[ResultHandler] = None,
    ) -> RequestManager.RequestResults:
        request_manager = RequestManager(payloads)

        subscription_added = False

        async with self.ws as ws:
            for item in payloads:
                item_id = await ws.send(item["payload"])
                request_manager.add_request(item_id, item["id"])

            while True:
                for item_id in request_manager.response_map.keys():
                    if (
                        item_id not in request_manager.responses
                        or asyncio.iscoroutinefunction(result_handler)
                    ):
                        if response := await ws.retrieve(item_id):
                            if (
                                asyncio.iscoroutinefunction(result_handler)
                                and not subscription_added
                            ):
                                # handles subscriptions, overwrites the previous mapping of {item_id : payload_id}
                                # with {subscription_id : payload_id}
                                try:
                                    item_id = request_manager.overwrite_request(
                                        item_id, response["result"]
                                    )
                                except KeyError:
                                    raise SubstrateRequestException(str(response))
                            decoded_response, complete = await self._process_response(
                                response,
                                item_id,
                                value_scale_type,
                                storage_item,
                                runtime,
                                result_handler,
                            )
                            request_manager.add_response(
                                item_id, decoded_response, complete
                            )
                    if (
                        asyncio.iscoroutinefunction(result_handler)
                        and not subscription_added
                    ):
                        subscription_added = True
                        break

                if request_manager.is_complete:
                    break

        return request_manager.get_results()

    @staticmethod
    def make_payload(id_: str, method: str, params: list) -> dict:
        """
        Creates a payload for making an rpc_request with _make_rpc_request

        :param id_: a unique name you would like to give to this request
        :param method: the method in the RPC request
        :param params: the params in the RPC request

        :return: the payload dict
        """
        return {
            "id": id_,
            "payload": {"jsonrpc": "2.0", "method": method, "params": params},
        }

    async def rpc_request(
        self,
        method: str,
        params: Optional[list],
        block_hash: Optional[str] = None,
        reuse_block_hash: bool = False,
    ) -> Any:
        """
        Makes an RPC request to the subtensor. Use this only if ``self.query`` and ``self.query_multiple`` and
        ``self.query_map`` do not meet your needs.

        :param method: str the method in the RPC request
        :param params: list of the params in the RPC request
        :param block_hash: optional str, the hash of the block — only supply this if not supplying the block
                           hash in the params, and not reusing the block hash
        :param reuse_block_hash: optional bool, whether to reuse the block hash in the params — only mark as True
                                 if not supplying the block hash in the params, or via the `block_hash` parameter

        :return: the response from the RPC request
        """
        block_hash = await self._get_current_block_hash(block_hash, reuse_block_hash)
        params = params or []
        payload_id = f"{method}{random.randint(0, 7000)}"
        payloads = [
            self.make_payload(
                payload_id,
                method,
                params + [block_hash] if block_hash else params,
            )
        ]
        runtime = Runtime(
            self.chain,
            self.runtime_config,
            self.metadata,
            self.type_registry,
        )
        result = await self._make_rpc_request(payloads, runtime=runtime)
        if "error" in result[payload_id][0]:
            raise SubstrateRequestException(result[payload_id][0]["error"]["message"])
        if "result" in result[payload_id][0]:
            return result[payload_id][0]
        else:
            raise SubstrateRequestException(result[payload_id][0])

    async def get_block_hash(self, block_id: int) -> str:
        return (await self.rpc_request("chain_getBlockHash", [block_id]))["result"]

    async def get_chain_head(self) -> str:
        result = await self._make_rpc_request(
            [
                self.make_payload(
                    "rpc_request",
                    "chain_getHead",
                    [],
                )
            ],
            runtime=Runtime(
                self.chain,
                self.runtime_config,
                self.metadata,
                self.type_registry,
            ),
        )
        self.last_block_hash = result["rpc_request"][0]["result"]
        return result["rpc_request"][0]["result"]

    async def compose_call(
        self,
        call_module: str,
        call_function: str,
        call_params: Optional[dict] = None,
        block_hash: Optional[str] = None,
    ) -> GenericCall:
        """
        Composes a call payload which can be used in an extrinsic.

        :param call_module: Name of the runtime module e.g. Balances
        :param call_function: Name of the call function e.g. transfer
        :param call_params: This is a dict containing the params of the call. e.g.
                            `{'dest': 'EaG2CRhJWPb7qmdcJvy3LiWdh26Jreu9Dx6R1rXxPmYXoDk', 'value': 1000000000000}`
        :param block_hash: Use metadata at given block_hash to compose call

        :return: A composed call
        """
        if call_params is None:
            call_params = {}

        await self.init_runtime(block_hash=block_hash)

        call = self.runtime_config.create_scale_object(
            type_string="Call", metadata=self.metadata
        )

        call.encode(
            {
                "call_module": call_module,
                "call_function": call_function,
                "call_args": call_params,
            }
        )

        return call

    async def query_multiple(
        self,
        params: list,
        storage_function: str,
        module: str,
        block_hash: Optional[str] = None,
        reuse_block_hash: bool = False,
    ) -> dict[str, ScaleType]:
        """
        Queries the subtensor. Only use this when making multiple queries, else use ``self.query``
        """
        # By allowing for specifying the block hash, users, if they have multiple query types they want
        # to do, can simply query the block hash first, and then pass multiple query_subtensor calls
        # into an asyncio.gather, with the specified block hash
        block_hash = await self._get_current_block_hash(block_hash, reuse_block_hash)
        if block_hash:
            self.last_block_hash = block_hash
        runtime = await self.init_runtime(block_hash=block_hash)
        preprocessed: tuple[Preprocessed] = await asyncio.gather(
            *[
                self._preprocess([x], block_hash, storage_function, module)
                for x in params
            ]
        )
        all_info = [
            self.make_payload(item.queryable, item.method, item.params)
            for item in preprocessed
        ]
        # These will always be the same throughout the preprocessed list, so we just grab the first one
        value_scale_type = preprocessed[0].value_scale_type
        storage_item = preprocessed[0].storage_item

        responses = await self._make_rpc_request(
            all_info, value_scale_type, storage_item, runtime
        )
        return {
            param: responses[p.queryable][0] for (param, p) in zip(params, preprocessed)
        }

    async def query_multi(
        self, storage_keys: list[StorageKey], block_hash: Optional[str] = None
    ) -> list:
        """
        Query multiple storage keys in one request.

        Example:

        ```
        storage_keys = [
            substrate.create_storage_key(
                "System", "Account", ["F4xQKRUagnSGjFqafyhajLs94e7Vvzvr8ebwYJceKpr8R7T"]
            ),
            substrate.create_storage_key(
                "System", "Account", ["GSEX8kR4Kz5UZGhvRUCJG93D5hhTAoVZ5tAe6Zne7V42DSi"]
            )
        ]

        result = substrate.query_multi(storage_keys)
        ```

        Parameters
        ----------
        storage_keys: list of StorageKey objects
        block_hash: Optional block_hash of state snapshot

        Returns
        -------
        list of `(storage_key, scale_obj)` tuples
        """

        await self.init_runtime(block_hash=block_hash)

        # Retrieve corresponding value
        response = await self.rpc_request(
            "state_queryStorageAt", [[s.to_hex() for s in storage_keys], block_hash]
        )

        if "error" in response:
            raise SubstrateRequestException(response["error"]["message"])

        result = []

        storage_key_map = {s.to_hex(): s for s in storage_keys}

        for result_group in response["result"]:
            for change_storage_key, change_data in result_group["changes"]:
                # Decode result for specified storage_key
                storage_key = storage_key_map[change_storage_key]
                if change_data is None:
                    change_data = b"\x00"
                else:
                    change_data = bytes.fromhex(change_data[2:])
                result.append(
                    (
                        storage_key,
                        await self.decode_scale(
                            storage_key.value_scale_type, change_data
                        ),
                    )
                )

        return result

    async def create_scale_object(
        self,
        type_string: str,
        data: Optional[ScaleBytes] = None,
        block_hash: Optional[str] = None,
        **kwargs,
    ) -> "ScaleType":
        """
        Convenience method to create a SCALE object of type `type_string`, this will initialize the runtime
        automatically at moment of `block_hash`, or chain tip if omitted.

        :param type_string: str Name of SCALE type to create
        :param data: ScaleBytes Optional ScaleBytes to decode
        :param block_hash: Optional block hash for moment of decoding, when omitted the chain tip will be used
        :param kwargs: keyword args for the Scale Type constructor

        :return: The created Scale Type object
        """
        runtime = await self.init_runtime(block_hash=block_hash)
        if "metadata" not in kwargs:
            kwargs["metadata"] = runtime.metadata

        return runtime.runtime_config.create_scale_object(
            type_string, data=data, **kwargs
        )

    async def generate_signature_payload(
        self,
        call: GenericCall,
        era=None,
        nonce: int = 0,
        tip: int = 0,
        tip_asset_id: Optional[int] = None,
        include_call_length: bool = False,
    ) -> ScaleBytes:
        # Retrieve genesis hash
        genesis_hash = await self.get_block_hash(0)

        if not era:
            era = "00"

        if era == "00":
            # Immortal extrinsic
            block_hash = genesis_hash
        else:
            # Determine mortality of extrinsic
            era_obj = self.runtime_config.create_scale_object("Era")

            if isinstance(era, dict) and "current" not in era and "phase" not in era:
                raise ValueError(
                    'The era dict must contain either "current" or "phase" element to encode a valid era'
                )

            era_obj.encode(era)
            block_hash = await self.get_block_hash(
                block_id=era_obj.birth(era.get("current"))
            )

        # Create signature payload
        signature_payload = self.runtime_config.create_scale_object(
            "ExtrinsicPayloadValue"
        )

        # Process signed extensions in metadata
        if "signed_extensions" in self.metadata[1][1]["extrinsic"]:
            # Base signature payload
            signature_payload.type_mapping = [["call", "CallBytes"]]

            # Add signed extensions to payload
            signed_extensions = self.metadata.get_signed_extensions()

            if "CheckMortality" in signed_extensions:
                signature_payload.type_mapping.append(
                    ["era", signed_extensions["CheckMortality"]["extrinsic"]]
                )

            if "CheckEra" in signed_extensions:
                signature_payload.type_mapping.append(
                    ["era", signed_extensions["CheckEra"]["extrinsic"]]
                )

            if "CheckNonce" in signed_extensions:
                signature_payload.type_mapping.append(
                    ["nonce", signed_extensions["CheckNonce"]["extrinsic"]]
                )

            if "ChargeTransactionPayment" in signed_extensions:
                signature_payload.type_mapping.append(
                    ["tip", signed_extensions["ChargeTransactionPayment"]["extrinsic"]]
                )

            if "ChargeAssetTxPayment" in signed_extensions:
                signature_payload.type_mapping.append(
                    ["asset_id", signed_extensions["ChargeAssetTxPayment"]["extrinsic"]]
                )

            if "CheckMetadataHash" in signed_extensions:
                signature_payload.type_mapping.append(
                    ["mode", signed_extensions["CheckMetadataHash"]["extrinsic"]]
                )

            if "CheckSpecVersion" in signed_extensions:
                signature_payload.type_mapping.append(
                    [
                        "spec_version",
                        signed_extensions["CheckSpecVersion"]["additional_signed"],
                    ]
                )

            if "CheckTxVersion" in signed_extensions:
                signature_payload.type_mapping.append(
                    [
                        "transaction_version",
                        signed_extensions["CheckTxVersion"]["additional_signed"],
                    ]
                )

            if "CheckGenesis" in signed_extensions:
                signature_payload.type_mapping.append(
                    [
                        "genesis_hash",
                        signed_extensions["CheckGenesis"]["additional_signed"],
                    ]
                )

            if "CheckMortality" in signed_extensions:
                signature_payload.type_mapping.append(
                    [
                        "block_hash",
                        signed_extensions["CheckMortality"]["additional_signed"],
                    ]
                )

            if "CheckEra" in signed_extensions:
                signature_payload.type_mapping.append(
                    ["block_hash", signed_extensions["CheckEra"]["additional_signed"]]
                )

            if "CheckMetadataHash" in signed_extensions:
                signature_payload.type_mapping.append(
                    [
                        "metadata_hash",
                        signed_extensions["CheckMetadataHash"]["additional_signed"],
                    ]
                )

        if include_call_length:
            length_obj = self.runtime_config.create_scale_object("Bytes")
            call_data = str(length_obj.encode(str(call.data)))

        else:
            call_data = str(call.data)

        payload_dict = {
            "call": call_data,
            "era": era,
            "nonce": nonce,
            "tip": tip,
            "spec_version": self.runtime_version,
            "genesis_hash": genesis_hash,
            "block_hash": block_hash,
            "transaction_version": self.transaction_version,
            "asset_id": {"tip": tip, "asset_id": tip_asset_id},
            "metadata_hash": None,
            "mode": "Disabled",
        }

        signature_payload.encode(payload_dict)

        if signature_payload.data.length > 256:
            return ScaleBytes(
                data=blake2b(signature_payload.data.data, digest_size=32).digest()
            )

        return signature_payload.data

    async def create_signed_extrinsic(
        self,
        call: GenericCall,
        keypair: Keypair,
        era: Optional[dict] = None,
        nonce: Optional[int] = None,
        tip: int = 0,
        tip_asset_id: Optional[int] = None,
        signature: Optional[Union[bytes, str]] = None,
    ) -> "GenericExtrinsic":
        """
        Creates an extrinsic signed by given account details

        :param call: GenericCall to create extrinsic for
        :param keypair: Keypair used to sign the extrinsic
        :param era: Specify mortality in blocks in follow format:
                    {'period': [amount_blocks]} If omitted the extrinsic is immortal
        :param nonce: nonce to include in extrinsics, if omitted the current nonce is retrieved on-chain
        :param tip: The tip for the block author to gain priority during network congestion
        :param tip_asset_id: Optional asset ID with which to pay the tip
        :param signature: Optionally provide signature if externally signed

        :return: The signed Extrinsic
        """
        await self.init_runtime()

        # Check requirements
        if not isinstance(call, GenericCall):
            raise TypeError("'call' must be of type Call")

        # Check if extrinsic version is supported
        if self.metadata[1][1]["extrinsic"]["version"] != 4:  # type: ignore
            raise NotImplementedError(
                f"Extrinsic version {self.metadata[1][1]['extrinsic']['version']} not supported"  # type: ignore
            )

        # Retrieve nonce
        if nonce is None:
            nonce = await self.get_account_nonce(keypair.ss58_address) or 0

        # Process era
        if era is None:
            era = "00"
        else:
            if isinstance(era, dict) and "current" not in era and "phase" not in era:
                # Retrieve current block id
                era["current"] = await self.get_block_number(
                    await self.get_chain_finalised_head()
                )

        if signature is not None:
            if isinstance(signature, str) and signature[0:2] == "0x":
                signature = bytes.fromhex(signature[2:])

            # Check if signature is a MultiSignature and contains signature version
            if len(signature) == 65:
                signature_version = signature[0]
                signature = signature[1:]
            else:
                signature_version = keypair.crypto_type

        else:
            # Create signature payload
            signature_payload = await self.generate_signature_payload(
                call=call, era=era, nonce=nonce, tip=tip, tip_asset_id=tip_asset_id
            )

            # Set Signature version to crypto type of keypair
            signature_version = keypair.crypto_type

            # Sign payload
            signature = keypair.sign(signature_payload)

        # Create extrinsic
        extrinsic = self.runtime_config.create_scale_object(
            type_string="Extrinsic", metadata=self.metadata
        )

        value = {
            "account_id": f"0x{keypair.public_key.hex()}",
            "signature": f"0x{signature.hex()}",
            "call_function": call.value["call_function"],
            "call_module": call.value["call_module"],
            "call_args": call.value["call_args"],
            "nonce": nonce,
            "era": era,
            "tip": tip,
            "asset_id": {"tip": tip, "asset_id": tip_asset_id},
            "mode": "Disabled",
        }

        # Check if ExtrinsicSignature is MultiSignature, otherwise omit signature_version
        signature_cls = self.runtime_config.get_decoder_class("ExtrinsicSignature")
        if issubclass(signature_cls, self.runtime_config.get_decoder_class("Enum")):
            value["signature_version"] = signature_version

        extrinsic.encode(value)

        return extrinsic

    async def get_chain_finalised_head(self):
        """
        A pass-though to existing JSONRPC method `chain_getFinalizedHead`

        Returns
        -------

        """
        response = await self.rpc_request("chain_getFinalizedHead", [])

        if response is not None:
            if "error" in response:
                raise SubstrateRequestException(response["error"]["message"])

            return response.get("result")

    async def runtime_call(
        self,
        api: str,
        method: str,
        params: Optional[Union[list, dict]] = None,
        block_hash: Optional[str] = None,
    ) -> ScaleType:
        """
        Calls a runtime API method

        :param api: Name of the runtime API e.g. 'TransactionPaymentApi'
        :param method: Name of the method e.g. 'query_fee_details'
        :param params: List of parameters needed to call the runtime API
        :param block_hash: Hash of the block at which to make the runtime API call

        :return: ScaleType from the runtime call
        """
        await self.init_runtime()

        if params is None:
            params = {}

        try:
            runtime_call_def = self.runtime_config.type_registry["runtime_api"][api][
                "methods"
            ][method]
            runtime_api_types = self.runtime_config.type_registry["runtime_api"][
                api
            ].get("types", {})
        except KeyError:
            raise ValueError(f"Runtime API Call '{api}.{method}' not found in registry")

        if isinstance(params, list) and len(params) != len(runtime_call_def["params"]):
            raise ValueError(
                f"Number of parameter provided ({len(params)}) does not "
                f"match definition {len(runtime_call_def['params'])}"
            )

        # Add runtime API types to registry
        self.runtime_config.update_type_registry_types(runtime_api_types)
        runtime = Runtime(
            self.chain,
            self.runtime_config,
            self.metadata,
            self.type_registry,
        )

        # Encode params
        param_data = ScaleBytes(bytes())
        for idx, param in enumerate(runtime_call_def["params"]):
            scale_obj = runtime.runtime_config.create_scale_object(param["type"])
            if isinstance(params, list):
                param_data += scale_obj.encode(params[idx])
            else:
                if param["name"] not in params:
                    raise ValueError(f"Runtime Call param '{param['name']}' is missing")

                param_data += scale_obj.encode(params[param["name"]])

        # RPC request
        result_data = await self.rpc_request(
            "state_call", [f"{api}_{method}", str(param_data), block_hash]
        )

        # Decode result
        # TODO update this to use bt-decode
        result_obj = runtime.runtime_config.create_scale_object(
            runtime_call_def["type"]
        )
        result_obj.decode(
            ScaleBytes(result_data["result"]),
            check_remaining=self.config.get("strict_scale_decode"),
        )

        return result_obj

    async def get_account_nonce(self, account_address: str) -> int:
        """
        Returns current nonce for given account address

        :param account_address: SS58 formatted address

        :return: Nonce for given account address
        """
        nonce_obj = await self.runtime_call(
            "AccountNonceApi", "account_nonce", [account_address]
        )
        return nonce_obj.value

    async def get_metadata_constant(self, module_name, constant_name, block_hash=None):
        """
        Retrieves the details of a constant for given module name, call function name and block_hash
        (or chaintip if block_hash is omitted)

        Parameters
        ----------
        module_name
        constant_name
        block_hash

        Returns
        -------
        MetadataModuleConstants
        """

        # await self.init_runtime(block_hash=block_hash)

        for module in self.metadata.pallets:
            if module_name == module.name and module.constants:
                for constant in module.constants:
                    if constant_name == constant.value["name"]:
                        return constant

    async def get_constant(
        self,
        module_name: str,
        constant_name: str,
        block_hash: Optional[str] = None,
        reuse_block_hash: bool = False,
    ) -> "ScaleType":
        """
        Returns the decoded `ScaleType` object of the constant for given module name, call function name and block_hash
        (or chaintip if block_hash is omitted)

        Parameters
        ----------
        :param module_name: Name of the module to query
        :param constant_name: Name of the constant to query
        :param block_hash: Hash of the block at which to make the runtime API call
        :param reuse_block_hash: Reuse last-used block hash if set to true

        :return: ScaleType from the runtime call
        """
        block_hash = await self._get_current_block_hash(block_hash, reuse_block_hash)
        constant = await self.get_metadata_constant(
            module_name, constant_name, block_hash=block_hash
        )
        if constant:
            # Decode to ScaleType
            return await self.decode_scale(
                constant.type,
                bytes(constant.constant_value),
                return_scale_obj=True,
            )
        else:
            return None

    async def get_payment_info(
        self, call: GenericCall, keypair: Keypair
    ) -> dict[str, Any]:
        """
        Retrieves fee estimation via RPC for given extrinsic

        Parameters
        ----------
        call: Call object to estimate fees for
        keypair: Keypair of the sender, does not have to include private key because no valid signature is required

        Returns
        -------
        Dict with payment info

        E.g. `{'class': 'normal', 'partialFee': 151000000, 'weight': {'ref_time': 143322000}}`

        """

        # Check requirements
        if not isinstance(call, GenericCall):
            raise TypeError("'call' must be of type Call")

        if not isinstance(keypair, Keypair):
            raise TypeError("'keypair' must be of type Keypair")

        # No valid signature is required for fee estimation
        signature = "0x" + "00" * 64

        # Create extrinsic
        extrinsic = await self.create_signed_extrinsic(
            call=call, keypair=keypair, signature=signature
        )
        extrinsic_len = self.runtime_config.create_scale_object("u32")
        extrinsic_len.encode(len(extrinsic.data))

        result = await self.runtime_call(
            "TransactionPaymentApi", "query_info", [extrinsic, extrinsic_len]
        )

        return result.value

    async def query(
        self,
        module: str,
        storage_function: str,
        params: Optional[list] = None,
        block_hash: Optional[str] = None,
        raw_storage_key: Optional[bytes] = None,
        subscription_handler=None,
        reuse_block_hash: bool = False,
    ) -> Union["ScaleType"]:
        """
        Queries subtensor. This should only be used when making a single request. For multiple requests,
        you should use ``self.query_multiple``
        """
        block_hash = await self._get_current_block_hash(block_hash, reuse_block_hash)
        if block_hash:
            self.last_block_hash = block_hash
        runtime = await self.init_runtime(block_hash=block_hash)
        preprocessed: Preprocessed = await self._preprocess(
            params, block_hash, storage_function, module
        )
        payload = [
            self.make_payload(
                preprocessed.queryable, preprocessed.method, preprocessed.params
            )
        ]
        value_scale_type = preprocessed.value_scale_type
        storage_item = preprocessed.storage_item

        responses = await self._make_rpc_request(
            payload,
            value_scale_type,
            storage_item,
            runtime,
            result_handler=subscription_handler,
        )
        return responses[preprocessed.queryable][0]

    async def query_map(
        self,
        module: str,
        storage_function: str,
        params: Optional[list] = None,
        block_hash: Optional[str] = None,
        max_results: Optional[int] = None,
        start_key: Optional[str] = None,
        page_size: int = 100,
        ignore_decoding_errors: bool = False,
        reuse_block_hash: bool = False,
    ) -> "QueryMapResult":
        """
        Iterates over all key-pairs located at the given module and storage_function. The storage
        item must be a map.

        Example:

        ```
        result = await substrate.query_map('System', 'Account', max_results=100)

        async for account, account_info in result:
            print(f"Free balance of account '{account.value}': {account_info.value['data']['free']}")
        ```

        Note: it is important that you do not use `for x in result.records`, as this will sidestep possible
        pagination. You must do `async for x in result`.

        :param module: The module name in the metadata, e.g. System or Balances.
        :param storage_function: The storage function name, e.g. Account or Locks.
        :param params: The input parameters in case of for example a `DoubleMap` storage function
        :param block_hash: Optional block hash for result at given block, when left to None the chain tip will be used.
        :param max_results: the maximum of results required, if set the query will stop fetching results when number is
                            reached
        :param start_key: The storage key used as offset for the results, for pagination purposes
        :param page_size: The results are fetched from the node RPC in chunks of this size
        :param ignore_decoding_errors: When set this will catch all decoding errors, set the item to None and continue
                                       decoding
        :param reuse_block_hash: use True if you wish to make the query using the last-used block hash. Do not mark True
                                 if supplying a block_hash

        :return: QueryMapResult object
        """
        params = params or []
        block_hash = await self._get_current_block_hash(block_hash, reuse_block_hash)
        if block_hash:
            self.last_block_hash = block_hash
        runtime = await self.init_runtime(block_hash=block_hash)

        metadata_pallet = runtime.metadata.get_metadata_pallet(module)
        if not metadata_pallet:
            raise ValueError(f'Pallet "{module}" not found')
        storage_item = metadata_pallet.get_storage_function(storage_function)

        if not metadata_pallet or not storage_item:
            raise ValueError(
                f'Storage function "{module}.{storage_function}" not found'
            )

        value_type = storage_item.get_value_type_string()
        param_types = storage_item.get_params_type_string()
        key_hashers = storage_item.get_param_hashers()

        # Check MapType conditions
        if len(param_types) == 0:
            raise ValueError("Given storage function is not a map")
        if len(params) > len(param_types) - 1:
            raise ValueError(
                f"Storage function map can accept max {len(param_types) - 1} parameters, {len(params)} given"
            )

        # Generate storage key prefix
        storage_key = StorageKey.create_from_storage_function(
            module,
            storage_item.value["name"],
            params,
            runtime_config=runtime.runtime_config,
            metadata=runtime.metadata,
        )
        prefix = storage_key.to_hex()

        if not start_key:
            start_key = prefix

        # Make sure if the max result is smaller than the page size, adjust the page size
        if max_results is not None and max_results < page_size:
            page_size = max_results

        # Retrieve storage keys
        response = await self.rpc_request(
            method="state_getKeysPaged",
            params=[prefix, page_size, start_key, block_hash],
        )

        if "error" in response:
            raise SubstrateRequestException(response["error"]["message"])

        result_keys = response.get("result")

        result = []
        last_key = None

        def concat_hash_len(key_hasher: str) -> int:
            """
            Helper function to avoid if statements
            """
            if key_hasher == "Blake2_128Concat":
                return 16
            elif key_hasher == "Twox64Concat":
                return 8
            elif key_hasher == "Identity":
                return 0
            else:
                raise ValueError("Unsupported hash type")

        if len(result_keys) > 0:
            last_key = result_keys[-1]

            # Retrieve corresponding value
            response = await self.rpc_request(
                method="state_queryStorageAt", params=[result_keys, block_hash]
            )

            if "error" in response:
                raise SubstrateRequestException(response["error"]["message"])

            for result_group in response["result"]:
                for item in result_group["changes"]:
                    try:
                        # Determine type string
                        key_type_string = []
                        for n in range(len(params), len(param_types)):
                            key_type_string.append(
                                f"[u8; {concat_hash_len(key_hashers[n])}]"
                            )
                            key_type_string.append(param_types[n])

                        item_key_obj = await self.decode_scale(
                            type_string=f"({', '.join(key_type_string)})",
                            scale_bytes=bytes.fromhex(item[0][len(prefix) :]),
                            return_scale_obj=True,
                        )

                        # strip key_hashers to use as item key
                        if len(param_types) - len(params) == 1:
                            item_key = item_key_obj[1]
                        else:
                            item_key = tuple(
                                item_key_obj[key + 1]
                                for key in range(len(params), len(param_types) + 1, 2)
                            )

                    except Exception as _:
                        if not ignore_decoding_errors:
                            raise
                        item_key = None

                    try:
                        try:
                            item_bytes = bytes.fromhex(item[1][2:])
                        except ValueError:
                            item_bytes = bytes.fromhex(item[1])

                        item_value = await self.decode_scale(
                            type_string=value_type,
                            scale_bytes=item_bytes,
                            return_scale_obj=True,
                        )
                    except Exception as _:
                        if not ignore_decoding_errors:
                            raise
                        item_value = None

                    result.append([item_key, item_value])

        return QueryMapResult(
            records=result,
            page_size=page_size,
            module=module,
            storage_function=storage_function,
            params=params,
            block_hash=block_hash,
            substrate=self,
            last_key=last_key,
            max_results=max_results,
            ignore_decoding_errors=ignore_decoding_errors,
        )

    async def submit_extrinsic(
        self,
        extrinsic: GenericExtrinsic,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
    ) -> "ExtrinsicReceipt":
        """
        Submit an extrinsic to the connected node, with the possibility to wait until the extrinsic is included
         in a block and/or the block is finalized. The receipt returned provided information about the block and
         triggered events

        Parameters
        ----------
        extrinsic: Extrinsic The extrinsic to be sent to the network
        wait_for_inclusion: wait until extrinsic is included in a block (only works for websocket connections)
        wait_for_finalization: wait until extrinsic is finalized (only works for websocket connections)

        Returns
        -------
        ExtrinsicReceipt

        """

        # Check requirements
        if not isinstance(extrinsic, GenericExtrinsic):
            raise TypeError("'extrinsic' must be of type Extrinsics")

        async def result_handler(message: dict, subscription_id) -> tuple[dict, bool]:
            """
            Result handler function passed as an arg to _make_rpc_request as the result_handler
            to handle the results of the extrinsic rpc call, which are multipart, and require
            subscribing to the message

            :param message: message received from the rpc call
            :param subscription_id: subscription id received from the initial rpc call for the subscription

            :returns: tuple containing the dict of the block info for the subscription, and bool for whether
                      the subscription is completed.
            """
            # Check if extrinsic is included and finalized
            if "params" in message and isinstance(message["params"]["result"], dict):
                # Convert result enum to lower for backwards compatibility
                message_result = {
                    k.lower(): v for k, v in message["params"]["result"].items()
                }

                if "finalized" in message_result and wait_for_finalization:
                    # Created as a task because we don't actually care about the result
                    self._forgettable_task = asyncio.create_task(
                        self.rpc_request("author_unwatchExtrinsic", [subscription_id])
                    )
                    return {
                        "block_hash": message_result["finalized"],
                        "extrinsic_hash": "0x{}".format(extrinsic.extrinsic_hash.hex()),
                        "finalized": True,
                    }, True
                elif (
                    "inblock" in message_result
                    and wait_for_inclusion
                    and not wait_for_finalization
                ):
                    # Created as a task because we don't actually care about the result
                    self._forgettable_task = asyncio.create_task(
                        self.rpc_request("author_unwatchExtrinsic", [subscription_id])
                    )
                    return {
                        "block_hash": message_result["inblock"],
                        "extrinsic_hash": "0x{}".format(extrinsic.extrinsic_hash.hex()),
                        "finalized": False,
                    }, True
            return message, False

        if wait_for_inclusion or wait_for_finalization:
            responses = (
                await self._make_rpc_request(
                    [
                        self.make_payload(
                            "rpc_request",
                            "author_submitAndWatchExtrinsic",
                            [str(extrinsic.data)],
                        )
                    ],
                    result_handler=result_handler,
                )
            )["rpc_request"]
            response = next(
                (r for r in responses if "block_hash" in r and "extrinsic_hash" in r),
                None,
            )

            if not response:
                raise SubstrateRequestException(responses)

            # Also, this will be a multipart response, so maybe should change to everything after the first response?
            # The following code implies this will be a single response after the initial subscription id.
            result = ExtrinsicReceipt(
                substrate=self,
                extrinsic_hash=response["extrinsic_hash"],
                block_hash=response["block_hash"],
                finalized=response["finalized"],
            )

        else:
            response = await self.rpc_request(
                "author_submitExtrinsic", [str(extrinsic.data)]
            )

            if "result" not in response:
                raise SubstrateRequestException(response.get("error"))

            result = ExtrinsicReceipt(substrate=self, extrinsic_hash=response["result"])

        return result

    async def get_metadata_call_function(
        self,
        module_name: str,
        call_function_name: str,
        block_hash: Optional[str] = None,
    ) -> Optional[list]:
        """
        Retrieves a list of all call functions in metadata active for given block_hash (or chaintip if block_hash
        is omitted)

        :param module_name: name of the module
        :param call_function_name: name of the call function
        :param block_hash: optional block hash

        :return: list of call functions
        """
        runtime = await self.init_runtime(block_hash=block_hash)

        for pallet in runtime.metadata.pallets:
            if pallet.name == module_name and pallet.calls:
                for call in pallet.calls:
                    if call.name == call_function_name:
                        return call
        return None

    async def get_block_number(self, block_hash: Optional[str] = None) -> int:
        """Async version of `substrateinterface.base.get_block_number` method."""
        response = await self.rpc_request("chain_getHeader", [block_hash])

        if "error" in response:
            raise SubstrateRequestException(response["error"]["message"])

        elif "result" in response:
            if response["result"]:
                return int(response["result"]["number"], 16)

    async def close(self):
        """
        Closes the substrate connection, and the websocket connection.
        """
        try:
            await self.ws.shutdown()
        except AttributeError:
            pass
