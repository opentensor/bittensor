import asyncio
import copy
import ssl
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable, Literal, Optional, Union, cast

import asyncstdlib as a
import scalecodec
from async_substrate_interface import AsyncSubstrateInterface
from async_substrate_interface.substrate_addons import RetryAsyncSubstrate
from async_substrate_interface.utils.storage import StorageKey
from bittensor_drand import get_encrypted_commitment
from bittensor_wallet.utils import SS58_FORMAT
from scalecodec import GenericCall

from bittensor.core.chain_data import (
    CrowdloanConstants,
    CrowdloanInfo,
    DelegateInfo,
    DynamicInfo,
    MetagraphInfo,
    NeuronInfo,
    NeuronInfoLite,
    ProposalVoteData,
    ProxyAnnouncementInfo,
    ProxyConstants,
    ProxyInfo,
    ProxyType,
    RootClaimType,
    SelectiveMetagraphIndex,
    SimSwapResult,
    StakeInfo,
    SubnetHyperparameters,
    SubnetIdentity,
    SubnetInfo,
    WeightCommitInfo,
    decode_account_id,
)
from bittensor.core.chain_data.chain_identity import ChainIdentity
from bittensor.core.chain_data.delegate_info import DelegatedInfo
from bittensor.core.chain_data.utils import (
    decode_block,
    decode_metadata,
    decode_revealed_commitment,
    decode_revealed_commitment_with_hotkey,
)
from bittensor.core.config import Config
from bittensor.core.errors import ChainError, SubstrateRequestException
from bittensor.core.extrinsics.asyncex.children import (
    root_set_pending_childkey_cooldown_extrinsic,
    set_children_extrinsic,
)
from bittensor.core.extrinsics.asyncex.crowdloan import (
    contribute_crowdloan_extrinsic,
    create_crowdloan_extrinsic,
    dissolve_crowdloan_extrinsic,
    finalize_crowdloan_extrinsic,
    refund_crowdloan_extrinsic,
    update_cap_crowdloan_extrinsic,
    update_end_crowdloan_extrinsic,
    update_min_contribution_crowdloan_extrinsic,
    withdraw_crowdloan_extrinsic,
)
from bittensor.core.extrinsics.asyncex.liquidity import (
    add_liquidity_extrinsic,
    modify_liquidity_extrinsic,
    remove_liquidity_extrinsic,
    toggle_user_liquidity_extrinsic,
)
from bittensor.core.extrinsics.asyncex.mev_shield import submit_encrypted_extrinsic
from bittensor.core.extrinsics.asyncex.move_stake import (
    move_stake_extrinsic,
    swap_stake_extrinsic,
    transfer_stake_extrinsic,
)
from bittensor.core.extrinsics.asyncex.proxy import (
    add_proxy_extrinsic,
    announce_extrinsic,
    create_pure_proxy_extrinsic,
    kill_pure_proxy_extrinsic,
    poke_deposit_extrinsic,
    proxy_announced_extrinsic,
    proxy_extrinsic,
    reject_announcement_extrinsic,
    remove_announcement_extrinsic,
    remove_proxies_extrinsic,
    remove_proxy_extrinsic,
)
from bittensor.core.extrinsics.asyncex.registration import (
    burned_register_extrinsic,
    register_extrinsic,
    register_subnet_extrinsic,
    set_subnet_identity_extrinsic,
)
from bittensor.core.extrinsics.asyncex.root import (
    claim_root_extrinsic,
    root_register_extrinsic,
    set_root_claim_type_extrinsic,
)
from bittensor.core.extrinsics.asyncex.serving import (
    publish_metadata_extrinsic,
    serve_axon_extrinsic,
)
from bittensor.core.extrinsics.asyncex.staking import (
    add_stake_extrinsic,
    add_stake_multiple_extrinsic,
    set_auto_stake_extrinsic,
)
from bittensor.core.extrinsics.asyncex.start_call import start_call_extrinsic
from bittensor.core.extrinsics.asyncex.take import set_take_extrinsic
from bittensor.core.extrinsics.asyncex.transfer import transfer_extrinsic
from bittensor.core.extrinsics.asyncex.unstaking import (
    unstake_all_extrinsic,
    unstake_extrinsic,
    unstake_multiple_extrinsic,
)
from bittensor.core.extrinsics.asyncex.weights import (
    commit_timelocked_weights_extrinsic,
    commit_weights_extrinsic,
    reveal_weights_extrinsic,
    set_weights_extrinsic,
)
from bittensor.core.extrinsics.utils import get_transfer_fn_params
from bittensor.core.metagraph import AsyncMetagraph
from bittensor.core.settings import (
    DEFAULT_MEV_PROTECTION,
    DEFAULT_PERIOD,
    TAO_APP_BLOCK_EXPLORER,
    TYPE_REGISTRY,
    version_as_int,
)
from bittensor.core.types import (
    BlockInfo,
    ExtrinsicResponse,
    Salt,
    SubtensorMixin,
    UIDs,
    Weights,
)
from bittensor.utils import (
    Certificate,
    decode_hex_identity_dict,
    format_error_message,
    get_caller_name,
    get_mechid_storage_index,
    is_valid_ss58_address,
    u16_normalized_float,
    u64_normalized_float,
    validate_max_attempts,
)
from bittensor.utils.balance import (
    Balance,
    check_balance_amount,
    fixed_to_float,
)
from bittensor.utils.btlogging import logging
from bittensor.utils.liquidity import (
    LiquidityPosition,
    calculate_fees,
    get_fees,
    price_to_tick,
    tick_to_price,
)

if TYPE_CHECKING:
    from async_substrate_interface import AsyncQueryMapResult
    from async_substrate_interface.types import ScaleObj
    from bittensor_wallet import Keypair, Wallet

    from bittensor.core.axon import Axon


class AsyncSubtensor(SubtensorMixin):
    """Asynchronous interface for interacting with the Bittensor blockchain.

    This class provides a thin layer over the Substrate Interface offering async functionality for Bittensor. This
    includes frequently-used calls for querying blockchain data, managing stakes and liquidity positions, registering
    neurons, submitting weights, and many other functions for participating in Bittensor.

    Notes:
        Key Bittensor concepts used throughout this class:

        - **Coldkey**: The key pair corresponding to a user's overall wallet. Used to transfer, stake, manage subnets.
        - **Hotkey**: A key pair (each wallet may have zero, one, or more) used for neuron operations (mining and
          validation).
        - **Netuid**: Unique identifier for a subnet (0 is the Root Subnet)
        - **UID**: Unique identifier for a neuron registered to a hotkey on a specific subnet.
        - **Metagraph**: Data structure containing the complete state of a subnet at a block.
        - **TAO**: The base network token; subnet 0 stake is in TAO
        - **Alpha**: Subnet-specific token representing some quantity of TAO staked into a subnet.
        - **Rao**: Smallest unit of TAO (1 TAO = 1e9 Rao)
        - Bittensor Glossary <https://docs.learnbittensor.org/glossary>
        - Wallets, Coldkeys and Hotkeys in Bittensor <https://docs.learnbittensor.org/keys/wallets>

    """

    def __init__(
        self,
        network: Optional[str] = None,
        config: Optional["Config"] = None,
        log_verbose: bool = False,
        fallback_endpoints: Optional[list[str]] = None,
        retry_forever: bool = False,
        archive_endpoints: Optional[list[str]] = None,
        websocket_shutdown_timer: Optional[float] = 5.0,
        mock: bool = False,
    ):
        """Initializes an AsyncSubtensor instance for blockchain interaction.

        Parameters:
            network: The network name to connect to (e.g., `finney` for Bittensor mainnet, `test`, for
                Bittensor test network, `local` for a locally deployed blockchain). If `None`, uses the
                default network from config.
            config: Configuration object for the AsyncSubtensor instance. If `None`, uses the default configuration.
            log_verbose: Enables or disables verbose logging.
            fallback_endpoints: List of fallback WebSocket endpoints to use if the primary network endpoint is
                unavailable. These are tried in order when the default endpoint fails.
            retry_forever: Whether to retry connection attempts indefinitely on connection errors.
            mock: Whether this is a mock instance. FOR TESTING ONLY.
            archive_endpoints: List of archive node endpoints for queries requiring historical block data beyond the
                retention period of lite nodes. These are only used when requesting blocks that the current node is
                unable to serve.
            websocket_shutdown_timer: Amount of time (in seconds) to wait after the last response from the chain before
                automatically closing the WebSocket connection. Pass `None` to disable automatic shutdown entirely.

        Returns:
            None
        """
        if config is None:
            config = AsyncSubtensor.config()
        self._config = copy.deepcopy(config)
        self.chain_endpoint, self.network = AsyncSubtensor.setup_config(
            network, self._config
        )

        self.log_verbose = log_verbose
        self._check_and_log_network_settings()

        logging.debug(
            f"Connecting to network: [blue]{self.network}[/blue], "
            f"chain_endpoint: [blue]{self.chain_endpoint}[/blue]..."
        )
        self.substrate = self._get_substrate(
            fallback_endpoints=fallback_endpoints,
            retry_forever=retry_forever,
            _mock=mock,
            archive_endpoints=archive_endpoints,
            ws_shutdown_timer=websocket_shutdown_timer,
        )
        if self.log_verbose:
            logging.set_trace()
            logging.info(
                f"Connected to {self.network} network and {self.chain_endpoint}."
            )

    async def close(self):
        """Closes the connection to the blockchain.

        Use this to explicitly clean up resources and close the network connection instead of waiting for garbage
        collection.

        Returns:
            None

        Example:

            sub = bt.AsyncSubtensor(network="finney")

            # Initialize the connection

            await subtensor.initialize()

            # calls to subtensor

            await subtensor.close()

        """
        if self.substrate:
            await self.substrate.close()

    async def initialize(self):
        """Establishes connection to the blockchain.

        This method establishes the connection to the Bittensor blockchain and should be called after creating an
        AsyncSubtensor instance before making any queries.

        When using the `async with` context manager, this method is called automatically and does not need to be
        invoked explicitly.

        Returns:
            AsyncSubtensor: The initialized instance (self) for method chaining.


        Example:

            subtensor = AsyncSubtensor(network="finney")

            # Initialize the connection

            await subtensor.initialize()

            # calls to subtensor

            await subtensor.close()

        """
        logging.info(
            f"[magenta]Connecting to Substrate:[/magenta] [blue]{self}[/blue][magenta]...[/magenta]"
        )
        try:
            await self.substrate.initialize()
            return self
        except TimeoutError:
            logging.error(
                f"[red]Error[/red]: Timeout occurred connecting to substrate."
                f" Verify your chain and network settings: {self}"
            )
            raise ConnectionError
        except (ConnectionRefusedError, ssl.SSLError) as error:
            logging.error(
                f"[red]Error[/red]: Connection refused when connecting to substrate. "
                f"Verify your chain and network settings: {self}. Error: {error}"
            )
            raise ConnectionError

    async def __aenter__(self):
        return await self.initialize()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.substrate.close()

    # Helpers ==========================================================================================================

    async def _decode_crowdloan_entry(
        self,
        crowdloan_id: int,
        data: dict,
        block_hash: Optional[str] = None,
    ) -> "CrowdloanInfo":
        """
        Internal helper to parse and decode a single Crowdloan record.

        Automatically decodes the embedded `call` field if present (Inline SCALE format).
        """
        call_data = data.get("call")
        if call_data and "Inline" in call_data:
            try:
                inline_bytes = bytes(call_data["Inline"][0][0])
                scale_object = await self.substrate.create_scale_object(
                    type_string="Call",
                    data=scalecodec.ScaleBytes(inline_bytes),
                    block_hash=block_hash,
                )
                decoded_call = scale_object.decode()
                data["call"] = decoded_call
            except Exception as e:
                data["call"] = {"decode_error": str(e), "raw": call_data}

        return CrowdloanInfo.from_dict(crowdloan_id, data)

    @a.lru_cache(maxsize=128)
    async def _get_block_hash(self, block_id: int):
        return await self.substrate.get_block_hash(block_id)

    def _get_substrate(
        self,
        fallback_endpoints: Optional[list[str]] = None,
        retry_forever: bool = False,
        _mock: bool = False,
        archive_endpoints: Optional[list[str]] = None,
        ws_shutdown_timer: float = 5.0,
    ) -> Union[AsyncSubstrateInterface, RetryAsyncSubstrate]:
        """Creates the Substrate instance based on provided arguments.

        This internal method creates either a standard AsyncSubstrateInterface or a RetryAsyncSubstrate depending on
        whether fallback/archive endpoints or infinite retry is requested.

        When `fallback_endpoints`, `archive_endpoints`, or `retry_forever` are provided, a RetryAsyncSubstrate
        is created with automatic failover and exponential backoff retry logic. Otherwise, a standard
        AsyncSubstrateInterface is used.

        Parameters:
            fallback_endpoints: List of fallback WebSocket endpoints to use if the primary endpoint is unavailable.
            retry_forever: Whether to retry connection attempts indefinitely on connection errors.
            _mock: Whether this is a mock instance. Used primarily for testing purposes.
            archive_endpoints: List of archive node endpoints for historical block queries. Archive nodes maintain full
                block history, while lite nodes only keep recent blocks. Use these when querying blocks older than the
                lite node's retention period (typically a few thousand blocks).
            ws_shutdown_timer: Amount of time (in seconds) to wait after the last response from the chain before
                automatically closing the WebSocket connection.

        Returns:
            Either AsyncSubstrateInterface (simple connection) or RetryAsyncSubstrate (with failover and retry logic).
        """
        if fallback_endpoints or retry_forever or archive_endpoints:
            return RetryAsyncSubstrate(
                url=self.chain_endpoint,
                fallback_chains=fallback_endpoints,
                ss58_format=SS58_FORMAT,
                type_registry=TYPE_REGISTRY,
                retry_forever=retry_forever,
                use_remote_preset=True,
                chain_name="Bittensor",
                _mock=_mock,
                archive_nodes=archive_endpoints,
                ws_shutdown_timer=ws_shutdown_timer,
            )
        return AsyncSubstrateInterface(
            url=self.chain_endpoint,
            ss58_format=SS58_FORMAT,
            type_registry=TYPE_REGISTRY,
            use_remote_preset=True,
            chain_name="Bittensor",
            _mock=_mock,
            ws_shutdown_timer=ws_shutdown_timer,
        )

    @property
    async def block(self):
        """Provides an asynchronous getter to retrieve the current block number.

        Returns:
            The current blockchain block number.
        """
        return await self.get_current_block()

    async def determine_block_hash(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[str]:
        """Determine the block hash for the block specified with the provided parameters.

        Ensures that only one of the block specification parameters is used and returns the appropriate block hash
        for blockchain queries.

        Parameter precedence (in order):
            1. If `reuse_block=True` and `block` or `block_hash` is set → raises ValueError
            2. If both `block` and `block_hash` are set → validates they match, raises ValueError if not
            3. If only `block_hash` is set → returns it directly
            4. If only `block` is set → fetches and returns its hash
            5. If none are set → returns `None`

        Parameters:
            block: The block number to get the hash for. If specifying along with `block_hash`, the hash of `block`
                will be checked and compared with the supplied block hash, raising a ValueError if the two do not match.
            block_hash: The hash of the blockchain block (hex string prefixed with `0x`). If specifying along with
                `block`, the hash of `block` will be checked and compared with the supplied block hash, raising a
                ValueError if the two do not match.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block` or `block_hash`.

        Returns:
            The block hash (hex string with `0x` prefix) if one can be determined, `None` otherwise.

        Notes:
            - <https://docs.learnbittensor.org/glossary#block>
        """
        if reuse_block and any([block, block_hash]):
            raise ValueError("Cannot specify both reuse_block and block_hash/block")
        if block and block_hash:
            retrieved_block_hash = await self.get_block_hash(block)
            if retrieved_block_hash != block_hash:
                raise ValueError(
                    "You have supplied a `block_hash` and a `block`, but the block does not map to the same hash as "
                    f"the one you supplied. You supplied `block_hash={block_hash}` for `block={block}`, but this block"
                    f"maps to the block hash {retrieved_block_hash}."
                )
            else:
                return retrieved_block_hash

        # Return the appropriate value.
        if block_hash:
            return block_hash
        if block is not None:
            return await self.get_block_hash(block)
        return None

    async def _runtime_method_exists(
        self, api: str, method: str, block_hash: str
    ) -> bool:
        """
        Check if a runtime call method exists at the given block.

        The complicated logic here comes from the fact that there are two ways in which runtime calls
        are stored: the new and primary method is through the Metadata V15, but the V14 is a good backup (implemented
        around mid 2024)

        Returns:
            True if the runtime call method exists, False otherwise.
        """
        runtime = await self.substrate.init_runtime(block_hash=block_hash)
        if runtime.metadata_v15 is not None:
            metadata_v15_value = runtime.metadata_v15.value()
            apis = {entry["name"]: entry for entry in metadata_v15_value["apis"]}
            try:
                api_entry = apis[api]
                methods = {entry["name"]: entry for entry in api_entry["methods"]}
                _ = methods[method]
                return True
            except KeyError:
                return False
        else:
            try:
                await self.substrate.get_metadata_runtime_call_function(
                    api=api,
                    method=method,
                    block_hash=block_hash,
                )
                return True
            except ValueError:
                return False

    async def _query_with_fallback(
        self,
        *args: tuple[str, str, Optional[list[Any]]],
        block_hash: Optional[str] = None,
        default_value: Any = ValueError,
    ):
        """
        Queries the subtensor node with a given set of args, falling back to the next group if the method
        does not exist at the given block. This method exists to support backwards compatibility for blocks.

        Parameters:
            *args: Tuples containing (module, storage_function, params) in the order they should be attempted.
            block_hash: The hash of the block being queried. If not provided, the chain tip will be used.
            default_value: The default value to return if none of the methods exist at the given block.

        Returns:
            The value returned by the subtensor node, or the default value if none of the methods exist at the given
            block.

        Raises:
            ValueError: If no default value is provided, and none of the methods exist at the given block, a
                ValueError will be raised.

        Example:

            value = await self._query_with_fallback(
                # the first attempt will be made to SubtensorModule.MechanismEmissionSplit with params `[1]`
                ("SubtensorModule", "MechanismEmissionSplit", [1]),
                # if it does not exist at the given block, the next attempt will be made to
                # SubtensorModule.MechanismEmission with params `None`
                ("SubtensorModule", "MechanismEmission", None),
                block_hash="0x1234",
                # if none of the methods exist at the given block, the default value of `None` will be returned
                default_value=None,
            )
        """
        if block_hash is None:
            block_hash = await self.substrate.get_chain_head()
        for module, storage_function, params in args:
            if await self.substrate.get_metadata_storage_function(
                module_name=module,
                storage_name=storage_function,
                block_hash=block_hash,
            ):
                return await self.substrate.query(
                    module=module,
                    storage_function=storage_function,
                    block_hash=block_hash,
                    params=params,
                )
        if not isinstance(default_value, ValueError):
            return default_value
        else:
            raise default_value

    async def _runtime_call_with_fallback(
        self,
        *args: tuple[str, str, Optional[list[Any]] | dict[str, Any]],
        block_hash: Optional[str] = None,
        default_value: Any = ValueError,
    ):
        """
        Makes a runtime call to the subtensor node with a given set of args, falling back to the next group if the
        api.method does not exist at the given block. This method exists to support backwards compatibility for blocks.

        Parameters:
            *args: Tuples containing (api, method, params) in the order they should be attempted.
            block_hash: The hash of the block being queried. If not provided, the chain tip will be used.
            default_value: The default value to return if none of the methods exist at the given block.

        Raises:
            ValueError: If no default value is provided, and none of the methods exist at the given block, a
                ValueError will be raised.

        Example:

            query = await self._runtime_call_with_fallback(
                # the first attempt will be made to SubnetInfoRuntimeApi.get_selective_mechagraph with the
                # given params
                (
                    "SubnetInfoRuntimeApi",
                    "get_selective_mechagraph",
                    [netuid, mechid, [f for f in range(len(SelectiveMetagraphIndex))]],
                ),
                # if it does not exist at the given block, the next attempt will be made as such:
                ("SubnetInfoRuntimeApi", "get_metagraph", [[netuid]]),
                block_hash=block_hash,
                # if none of the methods exist at the given block, the default value will be returned
                default_value=None,
            )

        """
        if block_hash is None:
            block_hash = await self.substrate.get_chain_head()
        for api, method, params in args:
            if await self._runtime_method_exists(
                api=api, method=method, block_hash=block_hash
            ):
                return await self.substrate.runtime_call(
                    api=api,
                    method=method,
                    block_hash=block_hash,
                    params=params,
                )
        if not isinstance(default_value, ValueError):
            return default_value
        else:
            raise default_value

    async def get_hyperparameter(
        self,
        param_name: str,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[Any]:
        """Retrieves a specified hyperparameter for a specific subnet.

        This method queries the blockchain for subnet-specific hyperparameters such as difficulty, tempo, immunity
        period, and other network configuration values. Return types and units vary by parameter.

        Parameters:
            param_name: The name of the hyperparameter storage function to retrieve.
            netuid: The unique identifier of the subnet.
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            The value of the specified hyperparameter if the subnet exists, `None` otherwise. Return type varies
            by parameter (int, float, bool, or Balance).

        Notes:
            - <https://docs.learnbittensor.org/subnets/subnet-hyperparameters>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        if not await self.subnet_exists(
            netuid, block_hash=block_hash, reuse_block=reuse_block
        ):
            logging.error(f"subnet {netuid} does not exist")
            return None

        result = await self.substrate.query(
            module="SubtensorModule",
            storage_function=param_name,
            params=[netuid],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )

        return getattr(result, "value", result)

    async def sim_swap(
        self,
        origin_netuid: int,
        destination_netuid: int,
        amount: "Balance",
        block_hash: Optional[str] = None,
    ) -> SimSwapResult:
        """Simulates a swap/stake operation and calculates the fees and resulting amounts.

        This method queries the SimSwap Runtime API to calculate the swap fees (in Alpha or TAO) and the quantities
        of Alpha or TAO tokens expected as output from the transaction. This simulation does NOT include the
        blockchain extrinsic transaction fee (the fee to submit the transaction itself).

        When moving stake between subnets, the operation may involve swapping Alpha (subnet-specific stake token) to
        TAO (the base network token), then TAO to Alpha on the destination subnet. For subnet 0 (root network), all
        stake is in TAO.

        Parameters:
            origin_netuid: Netuid of the source subnet (0 if add stake).
            destination_netuid: Netuid of the destination subnet.
            amount: Amount to swap/stake as a Balance object. Use `Balance.from_tao(...)` or
             `Balance.from_rao(...)` to create the amount.
            block_hash: The hash of the blockchain block for the query. If `None`, uses the current chain head.

        Returns:
            SimSwapResult: Object containing `alpha_fee`, `tao_fee`, `alpha_amount`, and `tao_amount` fields
            representing the swap fees and output amounts.

        Example:

            # Simulate staking 100 TAO stake to subnet 1

            result = await subtensor.sim_swap(
                origin_netuid=0,
                destination_netuid=1,
                amount=Balance.from_tao(100)
            )
            print(f"Fee: {result.tao_fee.tao} TAO, Output: {result.alpha_amount} Alpha")

        Notes:
            - **Alpha**: Subnet-specific stake token (dynamic TAO)
            - **TAO**: Base network token; subnet 0 uses TAO directly
            - The returned fees do NOT include the extrinsic transaction fee

            - Transaction Fees: <https://docs.learnbittensor.org/learn/fees>
            - Glossary: <https://docs.learnbittensor.org/glossary>
        """
        check_balance_amount(amount)
        block_hash = block_hash or await self.substrate.get_chain_head()
        if origin_netuid > 0 and destination_netuid > 0:
            # for cross-subnet moves where neither origin nor destination is root
            intermediate_result_, sn_price = await asyncio.gather(
                self.query_runtime_api(
                    runtime_api="SwapRuntimeApi",
                    method="sim_swap_alpha_for_tao",
                    params={"netuid": origin_netuid, "alpha": amount.rao},
                    block_hash=block_hash,
                ),
                self.get_subnet_price(origin_netuid, block_hash=block_hash),
            )
            intermediate_result = SimSwapResult.from_dict(
                intermediate_result_, origin_netuid
            )
            result = SimSwapResult.from_dict(
                await self.query_runtime_api(
                    runtime_api="SwapRuntimeApi",
                    method="sim_swap_tao_for_alpha",
                    params={
                        "netuid": destination_netuid,
                        "tao": intermediate_result.tao_amount.rao,
                    },
                    block_hash=block_hash,
                ),
                origin_netuid,
            )
            secondary_fee = (result.tao_fee / sn_price.tao).set_unit(origin_netuid)
            result.alpha_fee = result.alpha_fee + secondary_fee
            return result
        elif origin_netuid > 0:
            # dynamic to tao
            return SimSwapResult.from_dict(
                await self.query_runtime_api(
                    runtime_api="SwapRuntimeApi",
                    method="sim_swap_alpha_for_tao",
                    params={"netuid": origin_netuid, "alpha": amount.rao},
                    block_hash=block_hash,
                ),
                origin_netuid,
            )
        else:
            # tao to dynamic or unstaked to staked tao (SN0)
            return SimSwapResult.from_dict(
                await self.query_runtime_api(
                    runtime_api="SwapRuntimeApi",
                    method="sim_swap_tao_for_alpha",
                    params={"netuid": destination_netuid, "tao": amount.rao},
                    block_hash=block_hash,
                ),
                destination_netuid,
            )

    # Subtensor queries ===========================================================================================

    async def query_constant(
        self,
        module_name: str,
        constant_name: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional["ScaleObj"]:
        """Retrieves a constant from the specified module on the Bittensor blockchain.

        Use this function for nonstandard queries to constants defined within the Bittensor blockchain, if these cannot
        be accessed through other, standard getter methods.

        Parameters:
            module_name: The name of the module containing the constant (e.g., `Balances`, `SubtensorModule`).
            constant_name: The name of the constant to retrieve (e.g., `ExistentialDeposit`).
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            A SCALE-decoded object if found, `None` otherwise. Access the actual value using `.value` attribute.
            Common types include int (for counts/blocks), Balance objects (for amounts in Rao), and booleans.

        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        return await self.substrate.get_constant(
            module_name=module_name,
            constant_name=constant_name,
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )

    async def query_map(
        self,
        module: str,
        name: str,
        params: Optional[list] = None,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> "AsyncQueryMapResult":
        """Queries map storage from any module on the Bittensor blockchain.

        Use this function for nonstandard queries to map storage defined within the Bittensor blockchain, if these cannot
        be accessed through other, standard getter methods.

        Parameters:
            module: The name of the module from which to query the map storage (e.g., "SubtensorModule", "System").
            name: The specific storage function within the module to query (e.g., "Bonds", "Weights").
            params: Parameters to be passed to the query.
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            AsyncQueryMapResult: A data structure representing the map storage if found, None otherwise.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        result = await self.substrate.query_map(
            module=module,
            storage_function=name,
            params=params,
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        return result

    async def query_map_subtensor(
        self,
        name: str,
        params: Optional[list] = None,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> "AsyncQueryMapResult":
        """Queries map storage from the Subtensor module on the Bittensor blockchain.

        Use this function for nonstandard queries to map storage defined within the Bittensor blockchain, if these cannot
        be accessed through other, standard getter methods.

        Parameters:
            name: The name of the map storage function to query.
            params: A list of parameters to pass to the query function.
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            An object containing the map-like data structure, or `None` if not found.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        return await self.substrate.query_map(
            module="SubtensorModule",
            storage_function=name,
            params=params,
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )

    async def query_module(
        self,
        module: str,
        name: str,
        params: Optional[list] = None,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[Union["ScaleObj", Any]]:
        """Queries any module storage on the Bittensor blockchain with the specified parameters and block number.
        This function is a generic query interface that allows for flexible and diverse data retrieval from various
        blockchain modules. Use this function for nonstandard queries to storage defined within the Bittensor
        blockchain, if these cannot be accessed through other, standard getter methods.

        Parameters:
            module: The name of the module from which to query data.
            name: The name of the storage function within the module.
            params: A list of parameters to pass to the query function.
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            An object containing the requested data if found, `None` otherwise.

        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        return await self.substrate.query(
            module=module,
            storage_function=name,
            params=params,
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )

    async def query_runtime_api(
        self,
        runtime_api: str,
        method: str,
        params: Optional[Union[list[Any], dict[str, Any]]],
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[Any]:
        """Queries the runtime API of the Bittensor blockchain, providing a way to interact with the underlying runtime
        and retrieve data encoded in Scale Bytes format. Use this function for nonstandard queries to the runtime
         environment, if these cannot be accessed through other, standard getter methods.

        Parameters:
            runtime_api: The name of the runtime API to query.
            method: The specific method within the runtime API to call.
            params: The parameters to pass to the method call.
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            The decoded result from the runtime API call, or `None` if the call fails.

        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        if not block_hash and reuse_block:
            block_hash = self.substrate.last_block_hash
        result = await self.substrate.runtime_call(
            runtime_api, method, params, block_hash
        )
        return result.value

    async def query_subtensor(
        self,
        name: str,
        params: Optional[list] = None,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[Union["ScaleObj", Any]]:
        """Queries named storage from the Subtensor module on the Bittensor blockchain.

        Use this function for nonstandard queries to storage defined within the Bittensor blockchain, if these cannot
        be accessed through other, standard getter methods.

        Parameters:
            name: The name of the storage function to query.
            params: A list of parameters to pass to the query function.
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            query_response: An object containing the requested data.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        return await self.substrate.query(
            module="SubtensorModule",
            storage_function=name,
            params=params,
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )

    async def state_call(
        self,
        method: str,
        data: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> dict[Any, Any]:
        """Makes a state call to the Bittensor blockchain, allowing for direct queries of the blockchain's state.
        This function is typically used for advanced, nonstandard queries not provided by other getter methods.

        Use this method when you need to query runtime APIs or storage functions that don't have dedicated
        wrapper methods in the SDK. For standard queries, prefer the specific getter methods (e.g., `get_balance`,
        `get_stake`) which provide better type safety and error handling.

        Parameters:
            method: The runtime API method name (e.g., "SubnetInfoRuntimeApi", "get_metagraph").
            data: Hex-encoded string of the SCALE-encoded parameters to pass to the method.
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            The result of the rpc call.

        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        return await self.substrate.rpc_request(
            method="state_call",
            params=[method, data],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )

    # Common subtensor methods =========================================================================================

    async def all_subnets(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[list[DynamicInfo]]:
        """Queries the blockchain for comprehensive information about all subnets, including their dynamic parameters
        and operational status.

        Parameters:
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            Optional[list[DynamicInfo]]: A list of `DynamicInfo` objects, each containing detailed information about
            a subnet, or None if the query fails.
        """
        block_hash = await self.determine_block_hash(
            block=block, block_hash=block_hash, reuse_block=reuse_block
        )
        if not block_hash and reuse_block:
            block_hash = self.substrate.last_block_hash

        query = await self.substrate.runtime_call(
            api="SubnetInfoRuntimeApi",
            method="get_all_dynamic_info",
            block_hash=block_hash,
        )
        subnet_prices = await self.get_subnet_prices(block_hash=block_hash)

        decoded = query.decode()

        if not isinstance(subnet_prices, (SubstrateRequestException, ValueError)):
            for sn in decoded:
                sn.update(
                    {"price": subnet_prices.get(sn["netuid"], Balance.from_tao(0))}
                )
        else:
            logging.warning(
                f"Unable to fetch subnet prices for block {block}, block hash {block_hash}: {subnet_prices}"
            )
        return DynamicInfo.list_from_dicts(decoded)

    async def blocks_since_last_step(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """Queries the blockchain to determine how many blocks have passed since the last epoch step for a specific
        subnet.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            The number of blocks since the last step in the subnet, or None if the query fails.

        Notes:
            - <https://docs.learnbittensor.org/glossary#epoch>
        """
        query = await self.query_subtensor(
            name="BlocksSinceLastStep",
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
            params=[netuid],
        )
        return query.value if query is not None and hasattr(query, "value") else query

    async def blocks_since_last_update(
        self,
        netuid: int,
        uid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """Returns the number of blocks since the last update, or `None` if the subnetwork or UID does not exist.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            uid: The unique identifier of the neuron.
            block: The block number for this query. Do not specify if using block_hash or reuse_block.
            block_hash: The hash of the block for the query. Do not specify if using reuse_block or block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            The number of blocks since the last update, or None if the subnetwork or UID does not exist.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        block = block or await self.substrate.get_block_number(block_hash)
        call = await self.get_hyperparameter(
            param_name="LastUpdate",
            netuid=netuid,
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        return None if call is None else (block - int(call[uid]))

    async def blocks_until_next_epoch(
        self,
        netuid: int,
        tempo: Optional[int] = None,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """Returns the number of blocks until the next epoch of subnet with provided netuid.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            tempo: The tempo of the subnet.
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            The number of blocks until the next epoch of the subnet with provided netuid.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        block = block or await self.substrate.get_block_number(block_hash=block_hash)
        tempo = tempo or await self.tempo(netuid=netuid, block_hash=block_hash)

        if not tempo:
            return None

        # the logic is the same as in SubtensorModule:blocks_until_next_epoch
        netuid_plus_one = int(netuid) + 1
        tempo_plus_one = tempo + 1
        adjusted_block = (block + netuid_plus_one) % (2**64)
        remainder = adjusted_block % tempo_plus_one
        return tempo - remainder

    async def bonds(
        self,
        netuid: int,
        mechid: int = 0,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[tuple[int, list[tuple[int, int]]]]:
        """Retrieves the bond distribution set by subnet validators within a specific subnet.

        Bonds represent a validator's accumulated assessment of each miner's performance over time, which serves as the
        starting point of Yuma Consensus.

        Parameters:
            netuid: Subnet identifier.
            mechid: Subnet mechanism identifier (default 0 for primary mechanism).
            block: The block number for this query. Do not specify if using block_hash or reuse_block.
            block_hash: The hash of the block for the query. Do not specify if using reuse_block or block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            List of tuples, where each tuple contains:
                - validator_uid: The UID of the validator
                - bonds: List of (miner_uid, bond_value) pairs

            Bond values are u16-normalized (0-65535, where 65535 = 1.0 or 100%).

        Example:

            # Get bonds for subnet 1

            bonds = await subtensor.bonds(netuid=1)
            print(bonds[0])

            # example output: (5, [(0, 32767), (1, 16383), (3, 8191)])

            # This means validator UID 5 has bonds: 50% to miner 0, 25% to miner 1, 12.5% to miner 3

        Notes:
            - See: <https://docs.learnbittensor.org/glossary#validator-miner-bonds>
            - See: <https://docs.learnbittensor.org/glossary#yuma-consensus>
        """
        storage_index = get_mechid_storage_index(netuid, mechid)
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        b_map_encoded = await self.substrate.query_map(
            module="SubtensorModule",
            storage_function="Bonds",
            params=[storage_index],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        b_map = []
        async for uid, b in b_map_encoded:
            if b.value is not None:
                b_map.append((uid, b.value))

        return b_map

    async def commit_reveal_enabled(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> bool:
        """Check if commit-reveal mechanism is enabled for a given subnet at a specific block.

        Parameters:
            netuid: The unique identifier of the subnet for which to check the commit-reveal mechanism.
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            True if commit-reveal mechanism is enabled, False otherwise.

        Notes:
            - <https://docs.learnbittensor.org/glossary#commit-reveal>
            - <https://docs.learnbittensor.org/subnets/subnet-hyperparameters>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        call = await self.get_hyperparameter(
            param_name="CommitRevealWeightsEnabled",
            block_hash=block_hash,
            netuid=netuid,
            reuse_block=reuse_block,
        )
        return True if call is True else False

    async def difficulty(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """Retrieves the 'Difficulty' hyperparameter for a specified subnet in the Bittensor network.

        This parameter determines the computational challenge required for neurons to participate in consensus and
         validation processes, using proof of work (POW) registration.


        Parameters:
            netuid: The unique identifier of the subnet.
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            The value of the 'Difficulty' hyperparameter if the subnet exists, `None` otherwise.

        Notes:
            Burn registration is much more common on Bittensor subnets currently, compared to POW registration.

            - <https://docs.learnbittensor.org/subnets/subnet-hyperparameters>
            - <https://docs.learnbittensor.org/validators#validator-registration>
            - <https://docs.learnbittensor.org/miners#miner-registration>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        call = await self.get_hyperparameter(
            param_name="Difficulty",
            netuid=netuid,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        if call is None:
            return None
        return int(call)

    async def does_hotkey_exist(
        self,
        hotkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> bool:
        """Returns true if the hotkey has been associated with a coldkey through account creation.

        This method queries the Subtensor's Owner storage map to check if the hotkey has been paired with a
        coldkey, as it must be before it (the hotkey) can be used for neuron registration.

        The Owner storage map defaults to the zero address (`5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM`)
        for unused hotkeys. This method returns `True` if the Owner value is anything other than this default.

        Parameters:
            hotkey_ss58: The SS58 address of the hotkey.
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            True if the hotkey has been associated with a coldkey, False otherwise.

        Notes:
            - <https://docs.learnbittensor.org/glossary#hotkey>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        result = await self.substrate.query(
            module="SubtensorModule",
            storage_function="Owner",
            params=[hotkey_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        return_val = (
            False
            if result is None
            # not the default key (0x0)
            else result != "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"
        )
        return return_val

    async def get_admin_freeze_window(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> int:
        """Returns the duration, in blocks, of the administrative freeze window at the end of each epoch.

        The admin freeze window is a period at the end of each epoch during which subnet owner
        operations are prohibited. This prevents subnet owners from modifying hyperparameters or performing certain
        administrative actions right before validators submit weights at the epoch boundary.

        Parameters:
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            The number of blocks in the administrative freeze window (default: 10 blocks, ~2 minutes).

        Notes:
            - <https://docs.learnbittensor.org/learn/chain-rate-limits#administrative-freeze-window>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        return (
            await self.substrate.query(
                module="SubtensorModule",
                storage_function="AdminFreezeWindow",
                block_hash=block_hash,
            )
        ).value

    async def get_all_subnets_info(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list["SubnetInfo"]:
        """Retrieves detailed information about all subnets within the Bittensor network.

        Parameters:
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            A list of SubnetInfo objects, each containing detailed information about a subnet.

        """
        result, prices = await asyncio.gather(
            self.query_runtime_api(
                runtime_api="SubnetInfoRuntimeApi",
                method="get_subnets_info_v2",
                params=[],
                block=block,
                block_hash=block_hash,
                reuse_block=reuse_block,
            ),
            self.get_subnet_prices(
                block=block, block_hash=block_hash, reuse_block=reuse_block
            ),
            return_exceptions=True,
        )
        if not result:
            return []

        if not isinstance(prices, (SubstrateRequestException, ValueError)):
            for subnet in result:
                subnet.update({"price": prices.get(subnet["netuid"], 0)})
        else:
            logging.warning(
                f"Unable to fetch subnet prices for block {block}, block hash {block_hash}: {prices}"
            )

        return SubnetInfo.list_from_dicts(result)

    async def get_all_commitments(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> dict[str, str]:
        """Retrieves raw commitment metadata from a given subnet.

        This method retrieves all commitment data for all neurons in a specific subnet. This is useful for analyzing the
        commit-reveal patterns across an entire subnet.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            A mapping of the ss58:commitment with the commitment as a string.

        Example:

            # TODO add example of how to handle realistic commitment data
        """
        query = await self.query_map(
            module="Commitments",
            name="CommitmentOf",
            params=[netuid],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        result = {}
        async for id_, value in query:
            try:
                result[decode_account_id(id_[0])] = decode_metadata(value)
            except Exception as error:
                logging.error(
                    f"Error decoding [red]{id_}[/red] and [red]{value}[/red]: {error}"
                )
        return result

    async def get_all_ema_tao_inflow(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> dict[int, tuple[int, Balance]]:
        """Retrieves the EMA (exponential moving average) of net TAO flows for all subnets.

        The EMA tracks net TAO flows (staking minus unstaking) with a 30-day half-life (~86.8 day window), smoothing
        out short-term fluctuations while capturing sustained staking trends. This metric determines the subnet's share
        of TAO emissions under the current, flow-based model. Positive values indicate net inflow (more staking than unstaking),
        negative values indicate net outflow. Subnets with negative EMA flows receive zero emissions.

        Parameters:
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            Dict mapping netuid to (last_updated_block, ema_flow). The Balance represents the EMA of net TAO flow in
            TAO units. Positive values indicate sustained net inflow, negative values indicate sustained net outflow.

        The EMA uses a smoothing factor α ≈ 0.000003209, creating a 30-day half-life and ~86.8 day window. Only
        direct stake/unstake operations count toward flows; neuron registrations and root claims are excluded.
        Subnet 0 (root network) does not have an EMA TAO flow value.

        Notes:
            - Flow-based emissions: <https://docs.learnbittensor.org/learn/emissions#tao-reserve-injection>
            - EMA smoothing: <https://docs.learnbittensor.org/learn/ema>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        query = await self.substrate.query_map(
            module="SubtensorModule",
            storage_function="SubnetEmaTaoFlow",
            block_hash=block_hash,
        )
        tao_inflow_ema = {}
        async for netuid, (block_updated, tao_bits) in query:
            ema_value = int(fixed_to_float(tao_bits))
            tao_inflow_ema[netuid] = (block_updated, Balance.from_rao(ema_value))
        return tao_inflow_ema

    async def get_all_metagraphs_info(
        self,
        all_mechanisms: bool = False,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[list[MetagraphInfo]]:
        """
        Retrieves a list of MetagraphInfo objects for all subnets

        Parameters:
            all_mechanisms: If True then returns all mechanisms, otherwise only those with index 0 for all subnets.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number at which to perform the query.
            reuse_block: Whether to reuse the last-used block hash when retrieving info.

        Returns:
            List of MetagraphInfo objects for all existing subnets.

        Notes:
            - <https://docs.learnbittensor.org/glossary#metagraph>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        if not block_hash and reuse_block:
            block_hash = self.substrate.last_block_hash
        method = "get_all_mechagraphs" if all_mechanisms else "get_all_metagraphs"
        query = await self.substrate.runtime_call(
            api="SubnetInfoRuntimeApi",
            method=method,
            block_hash=block_hash,
        )
        if query is None or not hasattr(query, "value"):
            return None

        return MetagraphInfo.list_from_dicts(query.value)

    async def get_all_neuron_certificates(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> dict[str, Certificate]:
        """
        Retrieves the TLS certificates for neurons within a specified subnet (netuid) of the Bittensor network.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using `block` or
                `reuse_block`.
            reuse_block: Whether to use the last-used block. Do not set if using `block_hash` or `block`.

        Returns:
            Dictionary mapping neuron hotkey SS58 addresses to their Certificate objects. Only includes neurons
            that have registered certificates.

        Notes:
            This method is used for certificate discovery to establish mutual TLS communication between neurons.
            - <https://docs.learnbittensor.org/subnets/neuron-tls-certificates>
        """
        query_certificates = await self.query_map(
            module="SubtensorModule",
            name="NeuronCertificates",
            params=[netuid],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        output = {}
        async for key, item in query_certificates:
            output[decode_account_id(key)] = Certificate(item.value)
        return output

    async def get_all_revealed_commitments(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> dict[str, tuple[tuple[int, str], ...]]:
        """Retrieves all revealed commitments for a given subnet.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            A dictionary mapping hotkey addresses to tuples of (reveal_block, commitment_message) pairs.
            Each validator can have multiple revealed commitments (up to 10 most recent).

        Example:

            {
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY": ( (12, "Alice message 1"), (152, "Alice message 2") ),
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty": ( (12, "Bob message 1"), (147, "Bob message 2") ),
            }

        Notes:
            - <https://docs.learnbittensor.org/glossary#commit-reveal>

        """
        query = await self.query_map(
            module="Commitments",
            name="RevealedCommitments",
            params=[netuid],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )

        result = {}
        async for pair in query:
            hotkey_ss58_address, commitment_message = (
                decode_revealed_commitment_with_hotkey(pair)
            )
            result[hotkey_ss58_address] = commitment_message
        return result

    async def get_all_subnets_netuid(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[int]:
        """
        Retrieves the list of all subnet unique identifiers (netuids) currently present in the Bittensor network.

        Parameters:
            block: The blockchain block number for the query.
            block_hash: The hash of the block to retrieve the subnet unique identifiers from.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            A list of subnet netuids.

        This function provides a comprehensive view of the subnets within the Bittensor network, offering insights into
        its diversity and scale.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        result = await self.substrate.query_map(
            module="SubtensorModule",
            storage_function="NetworksAdded",
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        subnets = []
        if result.records:
            async for netuid, exists in result:
                if exists:
                    subnets.append(netuid)
        return subnets

    async def get_auto_stakes(
        self,
        coldkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> dict[int, str]:
        """Fetches auto stake destinations for a given wallet across all subnets.

        Parameters:
            coldkey_ss58: Coldkey ss58 address.
            block: The block number for the query.
            block_hash: The block hash for the query.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            Dictionary mapping netuid to hotkey, where:

                - netuid: The unique identifier of the subnet.
                - hotkey: The hotkey of the wallet.

        Notes:
            - <https://docs.learnbittensor.org/miners/autostaking>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        query = await self.substrate.query_map(
            module="SubtensorModule",
            storage_function="AutoStakeDestination",
            params=[coldkey_ss58],
            block_hash=block_hash,
        )

        pairs = {}
        async for netuid, destination in query:
            hotkey_ss58 = decode_account_id(destination.value[0])
            if hotkey_ss58:
                pairs[int(netuid)] = hotkey_ss58

        return pairs

    async def get_balance(
        self,
        address: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Balance:
        """Retrieves the balance for given coldkey.

        This method queries the System module's Account storage to get the current balance of a coldkey address. The
        balance represents the amount of TAO tokens held by the specified address.

        Parameters:
            address: The coldkey address in SS58 format.
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            Balance: The balance object containing the account's TAO balance.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        balance = await self.substrate.query(
            module="System",
            storage_function="Account",
            params=[address],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        return Balance(balance["data"]["free"])

    async def get_balances(
        self,
        *addresses: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> dict[str, Balance]:
        """Retrieves the balance for given coldkey(s).

        This method efficiently queries multiple coldkey addresses in a single batch operation, returning a dictionary
        mapping each address to its corresponding balance. This is more efficient than calling get_balance multiple
        times.

        Parameters:
            *addresses: Variable number of coldkey addresses in SS58 format.
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            A dictionary mapping each address to its Balance object.
        """
        if reuse_block:
            block_hash = self.substrate.last_block_hash
        elif block_hash is None and block is None:
            # Neither block nor block_hash provided, default to head
            block_hash = await self.get_block_hash()
        else:
            block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        calls = [
            (
                await self.substrate.create_storage_key(
                    "System", "Account", [address], block_hash=block_hash
                )
            )
            for address in addresses
        ]
        batch_call = await self.substrate.query_multi(calls, block_hash=block_hash)
        results = {}
        for item in batch_call:
            value = item[1] or {"data": {"free": 0}}
            results.update({item[0].params[0]: Balance(value["data"]["free"])})
        return results

    async def get_current_block(self) -> int:
        """Returns the current block number on the Bittensor blockchain.

        This function provides the latest block number, indicating the most recent state of the blockchain.

        Returns:
            int: The current chain block number.

        Notes:
            - <https://docs.learnbittensor.org/glossary#block>
        """
        return await self.substrate.get_block_number(None)

    async def get_block_hash(self, block: Optional[int] = None) -> str:
        """Retrieves the hash of a specific block on the Bittensor blockchain.

        The block hash is a unique identifier representing the cryptographic hash of the block's content, ensuring its
        integrity and immutability. It is a fundamental aspect of blockchain technology, providing a secure reference
        to each block's data. It is crucial for verifying transactions, ensuring data consistency, and maintaining the
        trustworthiness of the blockchain.

        Parameters:
            block: The block number for which the hash is to be retrieved. If `None`, returns the latest block hash.

        Returns:
            str: The cryptographic hash of the specified block.

        Notes:
            - <https://docs.learnbittensor.org/glossary#block>
        """
        if block is not None:
            return await self._get_block_hash(block)
        else:
            return await self.substrate.get_chain_head()

    async def get_block_info(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
    ) -> Optional[BlockInfo]:
        """Retrieve complete information about a specific block from the Subtensor chain.

        This method aggregates multiple low-level RPC calls into a single structured response, returning both the raw
        on-chain data and high-level decoded metadata for the given block.

        Parameters:
            block: The block number for which the hash is to be retrieved.
            block_hash: The hash of the block to retrieve the block from.

        Returns:
            BlockInfo instance: A dataclass containing all available information about the specified block, including:

                - number: The block number.
                - hash: The corresponding block hash.
                - timestamp: The timestamp of the block (based on the `Timestamp.Now` extrinsic).
                - header: The raw block header returned by the node RPC.
                - extrinsics: The list of decoded extrinsics included in the block.
                - explorer: The link to block explorer service. Always related with finney block data.
        """
        block_info = await self.substrate.get_block(
            block_number=block, block_hash=block_hash, ignore_decoding_errors=True
        )
        if isinstance(block_info, dict) and (header := block_info.get("header")):
            block = block or header.get("number", None)
            block_hash = block_hash or header.get("hash", None)
            extrinsics = cast(list, block_info.get("extrinsics"))
            timestamp = None
            for ext in extrinsics:
                if ext.value_serialized["call"]["call_module"] == "Timestamp":
                    timestamp = ext.value_serialized["call"]["call_args"][0]["value"]
                    break
            return BlockInfo(
                number=block,
                hash=block_hash,
                timestamp=timestamp,
                header=header,
                extrinsics=extrinsics,
                explorer=f"{TAO_APP_BLOCK_EXPLORER}{block}",
            )
        return None

    async def get_children(
        self,
        hotkey_ss58: str,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> tuple[bool, list[tuple[float, str]], str]:
        """Retrieves the children of a given hotkey and netuid.

        This method queries the SubtensorModule's ChildKeys storage function to get the children and formats them before
        returning as a tuple. It provides information about the child neurons that a validator has set for weight
        distribution.

        Parameters:
            hotkey_ss58: The hotkey value.
            netuid: The netuid value.
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            A tuple containing a boolean indicating success or failure, a list of formatted children with their
                proportions, and an error message (if applicable).

        Example:

            # Get children for a hotkey in subnet 1

            success, children, error = await subtensor.get_children(hotkey="5F...", netuid=1)

            if success:
                for proportion, child_hotkey in children:
                    print(f"Child {child_hotkey}: {proportion}")

        Notes:
            - <https://docs.learnbittensor.org/validators/child-hotkeys>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        try:
            children = await self.substrate.query(
                module="SubtensorModule",
                storage_function="ChildKeys",
                params=[hotkey_ss58, netuid],
                block_hash=block_hash,
                reuse_block_hash=reuse_block,
            )
            if children:
                formatted_children = []
                for proportion, child in children.value:
                    # Convert U64 to int
                    formatted_child = decode_account_id(child[0])
                    normalized_proportion = u64_normalized_float(proportion)
                    formatted_children.append((normalized_proportion, formatted_child))
                return True, formatted_children, ""
            else:
                return True, [], ""
        except SubstrateRequestException as e:
            return False, [], format_error_message(e)

    async def get_children_pending(
        self,
        hotkey_ss58: str,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> tuple[
        list[tuple[float, str]],
        int,
    ]:
        """Retrieves the pending children of a given hotkey and netuid.

        This method queries the SubtensorModule's PendingChildKeys storage function to get children that are pending
        approval or in a cooldown period. These are children that have been proposed but not yet finalized.

        Parameters:
            hotkey_ss58: The hotkey value.
            netuid: The netuid value.
            block: The block number for which the children are to be retrieved.
            block_hash: The hash of the block to retrieve the subnet unique identifiers from.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            tuple: A tuple containing:

                - list[tuple[float, str]]: A list of children with their proportions.
                - int: The cool-down block number.

        Notes:
            - <https://docs.learnbittensor.org/validators/child-hotkeys>
        """

        response = await self.substrate.query(
            module="SubtensorModule",
            storage_function="PendingChildKeys",
            params=[netuid, hotkey_ss58],
            block_hash=await self.determine_block_hash(
                block,
                block_hash,
                reuse_block,
            ),
            reuse_block_hash=reuse_block,
        )
        children, cooldown = response.value

        return (
            [
                (
                    u64_normalized_float(proportion),
                    decode_account_id(child[0]),
                )
                for proportion, child in children
            ],
            cooldown,
        )

    async def get_commitment(
        self,
        netuid: int,
        uid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> str:
        """Retrieves the on-chain commitment for a specific neuron in the Bittensor network.

        This method retrieves the commitment data that a neuron has published to the blockchain. Commitments are used in
        the commit-reveal mechanism for secure weight setting and other network operations.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            uid: The unique identifier of the neuron.
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            The commitment data as a string.


            # TODO: add a real example of how to handle realistic commitment data, or chop example

        Notes:
            - <https://docs.learnbittensor.org/glossary#commit-reveal>
        """
        metagraph = await self.metagraph(netuid)
        try:
            hotkey = metagraph.hotkeys[uid]  # type: ignore
        except IndexError:
            logging.error(
                "Your uid is not in the hotkeys. Please double-check your UID."
            )
            return ""

        metadata = cast(
            dict,
            await self.get_commitment_metadata(
                netuid, hotkey, block, block_hash, reuse_block
            ),
        )
        try:
            return decode_metadata(metadata)
        except Exception as error:
            logging.error(error)
            return ""

    async def get_commitment_metadata(
        self,
        netuid: int,
        hotkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Union[str, dict]:
        # TODO: how to handle return data? need good example @roman
        """Fetches raw commitment metadata from specific subnet for given hotkey.

        Parameters:
            netuid: The unique subnet identifier.
            hotkey_ss58: The hotkey ss58 address.
            block: The blockchain block number for the query.
            block_hash: The hash of the block at which to check the hotkey ownership.
            reuse_block: Whether to reuse the last-used blockchain hash.

        Returns:
            The raw commitment metadata. Returns a dict when commitment data exists,
            or an empty string when no commitment is found for the given hotkey on the subnet.

        Notes:
            - <https://docs.learnbittensor.org/glossary#commit-reveal>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        commit_data = await self.substrate.query(
            module="Commitments",
            storage_function="CommitmentOf",
            params=[netuid, hotkey_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        return commit_data

    async def get_crowdloan_constants(
        self,
        constants: Optional[list[str]] = None,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> "CrowdloanConstants":
        """Retrieves runtime configuration constants governing crowdloan behavior and limits on the Bittensor blockchain.

        If a list of constant names is provided, only those constants will be queried.
        Otherwise, all known constants defined in `CrowdloanConstants.field_names()` are fetched.

        These constants define requirements and operational limits for crowdloan campaigns:

        - `AbsoluteMinimumContribution`: Minimum amount per contribution (TAO).
        - `MaxContributors`: Maximum number of unique contributors per crowdloan.
        - `MaximumBlockDuration`: Maximum duration (in blocks) for a crowdloan campaign (60 days = 432,000 blocks on production).
        - `MinimumDeposit`: Minimum deposit required from the creator (TAO).
        - `MinimumBlockDuration`: Minimum duration (in blocks) for a crowdloan campaign (7 days = 50,400 blocks on production).
        - `RefundContributorsLimit`: Maximum number of contributors refunded per `refund_crowdloan` call (typically 50).

        Parameters:
            constants: Specific constant names to query. If `None`, retrieves all constants from `CrowdloanConstants`.
            block: The blockchain block number for the query.
            block_hash: The block hash at which to query. Do not set if using `block` or `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            A `CrowdloanConstants` data object containing the queried constants. Missing constants return `None`.
        Notes:
            These constants enforce contribution floors, duration bounds, and refund batching limits.

            - Crowdloans Overview: <https://docs.learnbittensor.org/subnets/crowdloans>

        """
        result = {}
        const_names = constants or CrowdloanConstants.constants_names()

        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        for const_name in const_names:
            query = await self.query_constant(
                module_name="Crowdloan",
                constant_name=const_name,
                block=block,
                block_hash=block_hash,
                reuse_block=reuse_block,
            )

            if query is not None:
                result[const_name] = query.value

        return CrowdloanConstants.from_dict(result)

    async def get_crowdloan_contributions(
        self,
        crowdloan_id: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> dict[str, "Balance"]:
        """Retrieves all contributions made to a specific crowdloan campaign.

        Returns a mapping of contributor coldkey addresses to their contribution amounts in Rao.

        Parameters:
            crowdloan_id: The unique identifier of the crowdloan.
            block: The blockchain block number for the query.
            block_hash: The block hash at which to query. Do not set if using `block` or `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            Dictionary mapping contributor SS58 addresses to their `Balance` contribution amounts (in Rao).
            Returns empty dictionary if the crowdloan has no contributions or does not exist.

        Notes:
            Contributions are clipped to the remaining cap. Once the cap is reached, no further contributions are accepted.

            - Crowdloans Overview: <https://docs.learnbittensor.org/subnets/crowdloans>
            - Crowdloan Tutorial: <https://docs.learnbittensor.org/subnets/crowdloans/crowdloans-tutorial#step-4-contribute-to-the-crowdloan>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        query = await self.substrate.query_map(
            module="Crowdloan",
            storage_function="Contributions",
            params=[crowdloan_id],
            block_hash=block_hash,
        )

        result = {}

        if query.records:
            async for record in query:
                if record[1].value:
                    result[decode_account_id(record[0])] = Balance.from_rao(
                        record[1].value
                    )

        return result

    async def get_crowdloan_by_id(
        self,
        crowdloan_id: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional["CrowdloanInfo"]:
        """Retrieves detailed information about a specific crowdloan campaign.

        Parameters:
            crowdloan_id: Unique identifier of the crowdloan (auto-incremented starting from 0).
            block: The blockchain block number for the query.
            block_hash: The block hash at which to query. Do not set if using `block` or `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            `CrowdloanInfo` object containing: campaign ID, creator address, creator's deposit,
            minimum contribution amount, end block, funding cap, funds account address, amount raised,
            optional target address, optional embedded call, finalization status, and contributor count.
            Returns `None` if the crowdloan does not exist.

        Notes:

            - Crowdloans Overview: <https://docs.learnbittensor.org/subnets/crowdloans>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        query = await self.substrate.query(
            module="Crowdloan",
            storage_function="Crowdloans",
            params=[crowdloan_id],
            block_hash=block_hash,
        )
        if not query:
            return None
        return await self._decode_crowdloan_entry(
            crowdloan_id=crowdloan_id, data=query.value, block_hash=block_hash
        )

    async def get_crowdloan_next_id(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> int:
        """Retrieves the next available crowdloan identifier.

        Crowdloan IDs are allocated sequentially starting from 0. This method returns the ID that will be
        assigned to the next crowdloan created via :meth:`create_crowdloan`.

        Parameters:
            block: The blockchain block number for the query.
            block_hash: The block hash at which to query. Do not set if using `block` or `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            The next crowdloan ID (integer) to be assigned.

        Notes:
            - Crowdloans Overview: <https://docs.learnbittensor.org/subnets/crowdloans>
            - Crowdloan Tutorial: <https://docs.learnbittensor.org/subnets/crowdloans/crowdloans-tutorial#get-the-crowdloan-id>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        result = await self.substrate.query(
            module="Crowdloan",
            storage_function="NextCrowdloanId",
            block_hash=block_hash,
        )
        return int(result.value or 0)

    async def get_crowdloans(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list["CrowdloanInfo"]:
        """Retrieves all existing crowdloan campaigns with their metadata.

        Returns comprehensive information for all crowdloans registered on the blockchain, including
        both active and finalized campaigns.

        Parameters:
            block: The blockchain block number for the query.
            block_hash: The block hash at which to query. Do not set if using `block` or `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            List of `CrowdloanInfo` objects, each containing: campaign ID, creator address, creator's deposit,
            minimum contribution amount, end block, funding cap, funds account address, amount raised,
            optional target address, optional embedded call, finalization status, and contributor count.
            Returns empty list if no crowdloans exist.

        Notes:
            - Crowdloans Overview: <https://docs.learnbittensor.org/subnets/crowdloans>
            - Crowdloan Lifecycle: <https://docs.learnbittensor.org/subnets/crowdloans#crowdloan-lifecycle>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        query = await self.substrate.query_map(
            module="Crowdloan",
            storage_function="Crowdloans",
            block_hash=block_hash,
        )

        crowdloans = []

        if query.records:
            async for c_id, value_obj in query:
                data = value_obj.value
                if not data:
                    continue
                crowdloans.append(
                    await self._decode_crowdloan_entry(
                        crowdloan_id=c_id, data=data, block_hash=block_hash
                    )
                )

        return crowdloans

    async def get_delegate_by_hotkey(
        self,
        hotkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[DelegateInfo]:
        """Retrieves detailed information about a delegate neuron (validator) based on its hotkey. This function
        provides a comprehensive view of the delegate's status, including its stakes, nominators, and reward
        distribution.

        Parameters:
            hotkey_ss58: The `SS58` address of the delegate's hotkey.
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            Detailed information about the delegate neuron, `None` if not found.

        Notes:

            - <https://docs.learnbittensor.org/glossary#delegate>
            - <https://docs.learnbittensor.org/glossary#nominator>
        """

        result = await self.query_runtime_api(
            runtime_api="DelegateInfoRuntimeApi",
            method="get_delegate",
            params=[hotkey_ss58],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )

        if not result:
            return None

        return DelegateInfo.from_dict(result)

    async def get_delegate_identities(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> dict[str, ChainIdentity]:
        """Fetches delegate identities.

        Delegates are validators that accept stake from other TAO holders (nominators/delegators). This method
        retrieves the on-chain identity information for all delegates, including display name, legal name, web URLs,
        and other metadata they have set.

        Parameters:
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            Dictionary mapping delegate SS58 addresses to their ChainIdentity objects.

        Notes:
            - <https://docs.learnbittensor.org/staking-and-delegation/delegation>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        identities = await self.substrate.query_map(
            module="SubtensorModule",
            storage_function="IdentitiesV2",
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )

        return {
            decode_account_id(ss58_address[0]): ChainIdentity.from_dict(
                decode_hex_identity_dict(identity.value),
            )
            async for ss58_address, identity in identities
        }

    async def get_delegate_take(
        self,
        hotkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> float:
        """
        Retrieves the delegate 'take' percentage for a neuron identified by its hotkey. The 'take' represents the
        percentage of rewards that the delegate claims from its nominators' stakes.

        Parameters:
            hotkey_ss58: The `SS58` address of the neuron's hotkey.
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            float: The delegate take percentage.

        Notes:
            - <https://docs.learnbittensor.org/staking-and-delegation/delegation>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        result = await self.query_subtensor(
            name="Delegates",
            block_hash=block_hash,
            reuse_block=reuse_block,
            params=[hotkey_ss58],
        )

        return u16_normalized_float(result.value)  # type: ignore

    async def get_delegated(
        self,
        coldkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[DelegatedInfo]:
        """Retrieves delegates and their associated stakes for a given nominator coldkey.

        This method identifies all delegates (validators) that a specific coldkey has staked tokens to, along with
        stake amounts and other delegation information. This is useful for account holders to understand their stake
        allocations and involvement in the network's delegation and consensus mechanisms.

        Parameters:
            coldkey_ss58: The SS58 address of the account's coldkey.
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            List of DelegatedInfo objects containing stake amounts and delegate information. Returns empty list if no
            delegations exist for the coldkey.

        Notes:
            - <https://docs.learnbittensor.org/staking-and-delegation/delegation>
        """

        result = await self.query_runtime_api(
            runtime_api="DelegateInfoRuntimeApi",
            method="get_delegated",
            params=[coldkey_ss58],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )

        if not result:
            return []

        return DelegatedInfo.list_from_dicts(result)

    async def get_delegates(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[DelegateInfo]:
        """Fetches all delegates registered on the chain.

        Delegates are validators that accept stake from other TAO holders (nominators/delegators). This method
        retrieves comprehensive information about all delegates including their hotkeys, total stake, nominator count,
        take percentage, and other metadata.

        Parameters:
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            List of DelegateInfo objects containing comprehensive delegate information. Returns empty list if no
            delegates are registered.

        Notes:
            - <https://docs.learnbittensor.org/staking-and-delegation/delegation>
        """
        result = await self.query_runtime_api(
            runtime_api="DelegateInfoRuntimeApi",
            method="get_delegates",
            params=[],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        if result:
            return DelegateInfo.list_from_dicts(result)
        else:
            return []

    async def get_existential_deposit(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Balance:
        """Retrieves the existential deposit amount for the Bittensor blockchain.

        The existential deposit is the minimum amount of TAO required for an account to exist on the blockchain.
        Accounts with balances below this threshold can be reaped (removed) to conserve network resources and prevent
        blockchain bloat from dust accounts.

        Parameters:
            block: The blockchain block number for the query.
            block_hash: Block hash at which to query the deposit amount. If `None`, the current block is used.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            The existential deposit amount in RAO.

        Notes:
            - <https://docs.learnbittensor.org/glossary#existential-deposit>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        result = await self.substrate.get_constant(
            module_name="Balances",
            constant_name="ExistentialDeposit",
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )

        if result is None:
            raise Exception("Unable to retrieve existential deposit amount.")

        return Balance.from_rao(getattr(result, "value", 0))

    async def get_ema_tao_inflow(
        self,
        netuid: int,
        block: Optional[int] = None,
    ) -> Optional[tuple[int, Balance]]:
        """Retrieves the EMA (exponential moving average) of net TAO flow for a specific subnet.

        The EMA tracks net TAO flows (staking minus unstaking) with a 30-day half-life (~86.8 day window), smoothing
        out short-term fluctuations while capturing sustained staking trends. This metric determines the subnet's share
        of TAO emissions under the current, flow-based model. Positive values indicate net inflow (more staking than unstaking),
        negative values indicate net outflow. Subnets with negative EMA flows receive zero emissions.

        Parameters:
            netuid: The unique identifier of the subnet to query.
            block: The block number to query. If `None`, uses latest finalized block.

        Returns:
            Tuple of (last_updated_block, ema_flow) where ema_flow is the EMA of net TAO flow in TAO units.
            Returns None if the subnet does not exist or if querying subnet 0 (root network).

        The EMA uses a smoothing factor α ≈ 0.000003209, creating a 30-day half-life and ~86.8 day window. Only direct
        stake/unstake operations count toward flows; neuron registrations and root claims are excluded. Subnet 0 (root
        network) does not have an EMA TAO flow value and will return None.

        Notes:
            - Flow-based emissions: <https://docs.learnbittensor.org/learn/emissions#tao-reserve-injection>
            - EMA smoothing: <https://docs.learnbittensor.org/learn/ema>
        """
        block_hash = await self.determine_block_hash(block)
        query = await self.substrate.query(
            module="SubtensorModule",
            storage_function="SubnetEmaTaoFlow",
            params=[netuid],
            block_hash=block_hash,
        )

        # sn0 doesn't have EmaTaoInflow
        if query is None:
            return None

        block_updated, tao_bits = query.value
        ema_value = int(fixed_to_float(tao_bits))
        return block_updated, Balance.from_rao(ema_value)

    async def get_hotkey_owner(
        self,
        hotkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[str]:
        """
        Retrieves the owner of the given hotkey at a specific block hash.
        This function queries the blockchain for the owner of the provided hotkey. If the hotkey does not exist at the
        specified block hash, it returns `None`.

        Parameters:
            hotkey_ss58: The SS58 address of the hotkey.
            block: The blockchain block number for the query.
            block_hash: The hash of the block at which to check the hotkey ownership.
            reuse_block: Whether to reuse the last-used blockchain hash.

        Returns:
            Optional[str]: The SS58 address of the owner if the hotkey exists, or None if it doesn't.

        Notes:
            - <https://docs.learnbittensor.org/glossary#hotkey>
            - <https://docs.learnbittensor.org/glossary#subnet>
            - <https://docs.learnbittensor.org/glossary#neuron>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        hk_owner_query = await self.substrate.query(
            module="SubtensorModule",
            storage_function="Owner",
            params=[hotkey_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        exists = False
        if hk_owner_query:
            exists = await self.does_hotkey_exist(hotkey_ss58, block_hash=block_hash)
        hotkey_owner = hk_owner_query if exists else None
        return hotkey_owner

    async def get_last_bonds_reset(
        self,
        netuid: int,
        hotkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ):
        """Retrieves the block number when bonds were last reset for a specific hotkey on a subnet.

        Parameters:
            netuid: The network uid to fetch from.
            hotkey_ss58: The hotkey of the neuron for which to fetch the last bonds reset.
            block: The block number to query.
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using `block` or `reuse_block`.
            reuse_block: Whether to use the last-used block. Do not set if using `block_hash` or `block`.

        Returns:
            The block number when bonds were last reset, or `None` if no bonds reset has occurred.

        Notes:
            - <https://docs.learnbittensor.org/resources/glossary#validator-miner-bonds>
            - <https://docs.learnbittensor.org/resources/glossary#commit-reveal>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        block = await self.substrate.query(
            module="Commitments",
            storage_function="LastBondsReset",
            params=[netuid, hotkey_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        return block

    async def get_last_commitment_bonds_reset_block(
        self,
        netuid: int,
        uid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """
        Retrieves the last block number when the bonds reset were triggered by publish_metadata for a specific neuron.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            uid: The unique identifier of the neuron.
            block: The block number to query.
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using `block` or `reuse_block`.
            reuse_block: Whether to use the last-used block. Do not set if using `block_hash` or `block`.

        Returns:
            The block number when the bonds were last reset, or None if not found.
        """

        metagraph = await self.metagraph(netuid, block=block)
        try:
            hotkey = metagraph.hotkeys[uid]
        except IndexError:
            logging.error(
                "Your uid is not in the hotkeys. Please double-check your UID."
            )
            return None
        block_data = await self.get_last_bonds_reset(
            netuid, hotkey, block, block_hash, reuse_block
        )
        try:
            return decode_block(block_data)
        except TypeError:
            return None

    async def get_liquidity_list(
        self,
        wallet: "Wallet",
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[list[LiquidityPosition]]:
        """Retrieves all liquidity positions for the given wallet on a specified subnet (netuid).
        Calculates associated fee rewards based on current global and tick-level fee data.

        Parameters:
            wallet: Wallet instance to fetch positions for.
            netuid: Subnet unique id.
            block: The block number for which the children are to be retrieved.
            block_hash: The hash of the block to retrieve the subnet unique identifiers from.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            List of liquidity positions, or None if subnet does not exist.

        Notes:
            - <https://docs.learnbittensor.org/liquidity-positions/
            - <https://docs.learnbittensor.org/liquidity-positions/managing-liquidity-positions>
        """
        if not await self.subnet_exists(netuid=netuid):
            logging.debug(f"Subnet {netuid} does not exist.")
            return None

        if not await self.is_subnet_active(netuid=netuid):
            logging.debug(f"Subnet {netuid} is not active.")
            return None

        positions_response = await self.query_map(
            module="Swap",
            name="Positions",
            params=[netuid, wallet.coldkeypub.ss58_address],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        if len(positions_response.records) == 0:
            return []

        block_hash = await self.determine_block_hash(
            block=block, block_hash=block_hash, reuse_block=reuse_block
        )

        # Fetch global fees and current price
        fee_global_tao_query_sk = await self.substrate.create_storage_key(
            pallet="Swap",
            storage_function="FeeGlobalTao",
            params=[netuid],
            block_hash=block_hash,
        )
        fee_global_alpha_query_sk = await self.substrate.create_storage_key(
            pallet="Swap",
            storage_function="FeeGlobalAlpha",
            params=[netuid],
            block_hash=block_hash,
        )
        sqrt_price_query_sk = await self.substrate.create_storage_key(
            pallet="Swap",
            storage_function="AlphaSqrtPrice",
            params=[netuid],
            block_hash=block_hash,
        )

        (
            fee_global_tao_query,
            fee_global_alpha_query,
            sqrt_price_query,
        ) = await self.substrate.query_multi(
            storage_keys=[
                fee_global_tao_query_sk,
                fee_global_alpha_query_sk,
                sqrt_price_query_sk,
            ],
            block_hash=block_hash,
        )

        # convert to floats
        fee_global_tao = fixed_to_float(fee_global_tao_query[1])
        fee_global_alpha = fixed_to_float(fee_global_alpha_query[1])
        sqrt_price = fixed_to_float(sqrt_price_query[1])

        # Fetch global fees and current price
        current_tick = price_to_tick(sqrt_price**2)

        # Fetch positions
        positions_values: list[tuple[dict, int, int]] = []
        positions_storage_keys: list[StorageKey] = []
        async for _, p in positions_response:
            position = p.value

            tick_low_idx = position.get("tick_low")[0]
            tick_high_idx = position.get("tick_high")[0]
            positions_values.append((position, tick_low_idx, tick_high_idx))
            tick_low_sk = await self.substrate.create_storage_key(
                pallet="Swap",
                storage_function="Ticks",
                params=[netuid, tick_low_idx],
                block_hash=block_hash,
            )
            tick_high_sk = await self.substrate.create_storage_key(
                pallet="Swap",
                storage_function="Ticks",
                params=[netuid, tick_high_idx],
                block_hash=block_hash,
            )
            positions_storage_keys.extend([tick_low_sk, tick_high_sk])

        # query all our ticks at once
        ticks_query = await self.substrate.query_multi(
            positions_storage_keys, block_hash=block_hash
        )
        # iterator with just the values
        ticks = iter([x[1] for x in ticks_query])
        positions = []
        for position, tick_low_idx, tick_high_idx in positions_values:
            tick_low = next(ticks)
            tick_high = next(ticks)
            # Calculate fees above/below range for both tokens
            tao_below = get_fees(
                current_tick=current_tick,
                tick=tick_low,
                tick_index=tick_low_idx,
                quote=True,
                global_fees_tao=fee_global_tao,
                global_fees_alpha=fee_global_alpha,
                above=False,
            )
            tao_above = get_fees(
                current_tick=current_tick,
                tick=tick_high,
                tick_index=tick_high_idx,
                quote=True,
                global_fees_tao=fee_global_tao,
                global_fees_alpha=fee_global_alpha,
                above=True,
            )
            alpha_below = get_fees(
                current_tick=current_tick,
                tick=tick_low,
                tick_index=tick_low_idx,
                quote=False,
                global_fees_tao=fee_global_tao,
                global_fees_alpha=fee_global_alpha,
                above=False,
            )
            alpha_above = get_fees(
                current_tick=current_tick,
                tick=tick_high,
                tick_index=tick_high_idx,
                quote=False,
                global_fees_tao=fee_global_tao,
                global_fees_alpha=fee_global_alpha,
                above=True,
            )

            # Calculate fees earned by position
            fees_tao, fees_alpha = calculate_fees(
                position=position,
                global_fees_tao=fee_global_tao,
                global_fees_alpha=fee_global_alpha,
                tao_fees_below_low=tao_below,
                tao_fees_above_high=tao_above,
                alpha_fees_below_low=alpha_below,
                alpha_fees_above_high=alpha_above,
                netuid=netuid,
            )

            positions.append(
                LiquidityPosition(
                    **{
                        "id": position.get("id")[0],
                        "price_low": Balance.from_tao(
                            tick_to_price(position.get("tick_low")[0])
                        ),
                        "price_high": Balance.from_tao(
                            tick_to_price(position.get("tick_high")[0])
                        ),
                        "liquidity": Balance.from_rao(position.get("liquidity")),
                        "fees_tao": fees_tao,
                        "fees_alpha": fees_alpha,
                        "netuid": position.get("netuid"),
                    }
                )
            )

        return positions

    async def get_mechanism_emission_split(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[list[int]]:
        """Returns the emission percentages allocated to each subnet mechanism.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The hash of the block to retrieve the stake from. Do not specify if using `block` or `reuse_block`.
            reuse_block: Whether to use the last-used block. Do not set if using `block_hash` or `block`.

        Returns:
            A list of integers representing the percentage of emission allocated to each subnet mechanism (rounded to
            whole numbers). Returns None if emission is evenly split or if the data is unavailable.

        Notes:
            - <https://docs.learnbittensor.org/subnets/understanding-multiple-mech-subnets>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        result = await self._query_with_fallback(
            ("SubtensorModule", "MechanismEmissionSplit", [netuid]),
            block_hash=block_hash,
            default_value=None,
        )
        if result is None or not hasattr(result, "value"):
            return None

        return [round(i / sum(result.value) * 100) for i in result.value]

    async def get_mechanism_count(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> int:
        """Retrieves the number of mechanisms for the given subnet.

        Parameters:
            netuid: Subnet identifier.
            block: The blockchain block number for the query.
            block_hash: The hash of the block to retrieve the stake from. Do not specify if using `block` or `reuse_block`.
            reuse_block: Whether to use the last-used block. Do not set if using `block_hash` or `block`.

        Returns:
            The number of mechanisms for the given subnet.

        Notes:
            - <https://docs.learnbittensor.org/subnets/understanding-multiple-mech-subnets>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        query = await self._query_with_fallback(
            ("SubtensorModule", "MechanismCountCurrent", [netuid]),
            block_hash=block_hash,
            default_value=None,
        )
        return getattr(query, "value", 1)

    async def get_metagraph_info(
        self,
        netuid: int,
        mechid: int = 0,
        selected_indices: Optional[
            Union[list[SelectiveMetagraphIndex], list[int]]
        ] = None,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[MetagraphInfo]:
        """Retrieves full or partial metagraph information for the specified subnet (netuid).

        A metagraph is a data structure that contains comprehensive information about the current state of a subnet,
        including detailed information on all the nodes (neurons) such as subnet validator stakes and subnet weights
        in the subnet. Metagraph aids in calculating emissions.

        Parameters:
            netuid: The unique identifier of the subnet to query.
            selected_indices: Optional list of SelectiveMetagraphIndex or int values specifying which fields to retrieve.
                If not provided, all available fields will be returned.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number at which to perform the query.
            reuse_block: Whether to reuse the last-used block hash when retrieving info.
            mechid: Subnet mechanism unique identifier.

        Returns:
            MetagraphInfo object with the requested subnet mechanism data, None if the subnet mechanism does not exist.

        Example:

            # Retrieve all fields from the metagraph from subnet 2 mechanism 0

            meta_info = subtensor.get_metagraph_info(netuid=2)

            # Retrieve all fields from the metagraph from subnet 2 mechanism 1

            meta_info = subtensor.get_metagraph_info(netuid=2, mechid=1)

            # Retrieve selective data from the metagraph from subnet 2 mechanism 0

            partial_meta_info = subtensor.get_metagraph_info(
                netuid=2,
                selected_indices=[SelectiveMetagraphIndex.Name, SelectiveMetagraphIndex.OwnerHotkeys]
            )

            # Retrieve selective data from the metagraph from subnet 2 mechanism 1

            partial_meta_info = subtensor.get_metagraph_info(
                netuid=2,
                mechid=1,
                selected_indices=[SelectiveMetagraphIndex.Name, SelectiveMetagraphIndex.OwnerHotkeys]
            )

        Notes:
            - <https://docs.learnbittensor.org/subnets/metagraph>

        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        if not block_hash and reuse_block:
            block_hash = self.substrate.last_block_hash
        if not block_hash:
            block_hash = await self.substrate.get_chain_head()

        # Normalize selected_indices to a list of integers
        if selected_indices is not None:
            indexes = [
                f.value if isinstance(f, SelectiveMetagraphIndex) else f
                for f in selected_indices
            ]
            if 0 not in indexes:
                indexes = [0] + indexes
            query = await self._runtime_call_with_fallback(
                (
                    "SubnetInfoRuntimeApi",
                    "get_selective_mechagraph",
                    [netuid, mechid, indexes],
                ),
                ("SubnetInfoRuntimeApi", "get_selective_metagraph", [netuid, indexes]),
                block_hash=block_hash,
                default_value=ValueError(
                    "You have specified `selected_indices` to retrieve metagraph info selectively, but the "
                    "selective runtime calls are not available at this block (probably too old). Do not specify "
                    "`selected_indices` to retrieve metagraph info selectively."
                ),
            )
        else:
            query = await self._runtime_call_with_fallback(
                (
                    "SubnetInfoRuntimeApi",
                    "get_selective_mechagraph",
                    [netuid, mechid, [f for f in range(len(SelectiveMetagraphIndex))]],
                ),
                ("SubnetInfoRuntimeApi", "get_metagraph", [[netuid]]),
                block_hash=block_hash,
                default_value=None,
            )

        if getattr(query, "value", None) is None:
            logging.error(
                f"Subnet mechanism {netuid}.{mechid if mechid else 0} does not exist."
            )
            return None

        return MetagraphInfo.from_dict(query.value)

    async def get_mev_shield_current_key(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[bytes]:
        """
        Retrieves the CurrentKey from the MevShield pallet storage.

        The CurrentKey contains the ML-KEM-768 public key that is currently being used for encryption in this block.
        This key is rotated from NextKey at the beginning of each block.

        Parameters:
            block: The blockchain block number for the query.
            block_hash: The hash of the block to retrieve the stake from. Do not specify if using block or reuse_block.
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            The ML-KEM-768 public key as bytes (1184 bytes for ML-KEM-768)

        Note:
            If CurrentKey is not set (None in storage), this function returns None. This can happen if no validator has
            announced a key yet.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        query = await self.substrate.query(
            module="MevShield",
            storage_function="CurrentKey",
            block_hash=block_hash,
        )

        if query is None:
            return None

        public_key_bytes = bytes(next(iter(query)))

        # Validate public_key size for ML-KEM-768 (must be exactly 1184 bytes)
        MLKEM768_PUBLIC_KEY_SIZE = 1184
        if len(public_key_bytes) != MLKEM768_PUBLIC_KEY_SIZE:
            raise ValueError(
                f"Invalid ML-KEM-768 public key size: {len(public_key_bytes)} bytes. "
                f"Expected exactly {MLKEM768_PUBLIC_KEY_SIZE} bytes."
            )

        return public_key_bytes

    async def get_mev_shield_next_key(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[bytes]:
        """
        Retrieves the NextKey from the MevShield pallet storage.

        The NextKey contains the ML-KEM-768 public key that will be used for encryption in the next block. This key is
        rotated from NextKey to CurrentKey at the beginning of each block.

        Parameters:
            block: The blockchain block number for the query.
            block_hash: The hash of the block to retrieve the stake from. Do not specify if using block or reuse_block.
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            The ML-KEM-768 public key as bytes (1184 bytes for ML-KEM-768)

        Note:
            If NextKey is not set (None in storage), this function returns None. This can happen if no validator has
            announced the next key yet.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        query = await self.substrate.query(
            module="MevShield",
            storage_function="NextKey",
            block_hash=block_hash,
        )

        if query is None:
            return None

        public_key_bytes = bytes(next(iter(query)))

        # Validate public_key size for ML-KEM-768 (must be exactly 1184 bytes)
        MLKEM768_PUBLIC_KEY_SIZE = 1184
        if len(public_key_bytes) != MLKEM768_PUBLIC_KEY_SIZE:
            raise ValueError(
                f"Invalid ML-KEM-768 public key size: {len(public_key_bytes)} bytes. "
                f"Expected exactly {MLKEM768_PUBLIC_KEY_SIZE} bytes."
            )

        return public_key_bytes

    async def get_mev_shield_submission(
        self,
        submission_id: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[dict[str, str | int | bytes]]:
        """
        Retrieves Submission from the MevShield pallet storage.

        If submission_id is provided, returns a single submission. If submission_id is None, returns all submissions from
        the storage map.

        Parameters:
            submission_id: The hash ID of the submission. Can be a hex string with "0x" prefix or bytes. If None,
                returns all submissions.
            block: The blockchain block number for the query.
            block_hash: The hash of the block to retrieve the stake from. Do not specify if using block or reuse_block.
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            If submission_id is provided: A dictionary containing the submission data if found, None otherwise. The
                dictionary contains:
                - author: The SS58 address of the account that submitted the encrypted extrinsic
                - commitment: The blake2_256 hash of the payload_core (as hex string with "0x" prefix)
                - ciphertext: The encrypted blob as bytes (format: [u16 kem_len][kem_ct][nonce24][aead_ct])
                - submitted_in: The block number when the submission was created

            If submission_id is None: A dictionary mapping submission IDs (as hex strings) to submission dictionaries.

        Note:
            If a specific submission does not exist in storage, this function returns None. If querying all submissions
            and none exist, returns an empty dictionary.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        submission_id = (
            submission_id[2:] if submission_id.startswith("0x") else submission_id
        )
        submission_id_bytes = bytes.fromhex(submission_id)

        query = await self.substrate.query(
            module="MevShield",
            storage_function="Submissions",
            params=[submission_id_bytes],
            block_hash=block_hash,
        )

        if query is None or not isinstance(query, dict):
            return None

        autor = decode_account_id(query.get("author"))
        commitment = bytes(query.get("commitment")[0])
        ciphertext = bytes(query.get("ciphertext")[0])
        submitted_in = query.get("submitted_in")

        return {
            "author": autor,
            "commitment": commitment,
            "ciphertext": ciphertext,
            "submitted_in": submitted_in,
        }

    async def get_mev_shield_submissions(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[dict[str, dict[str, str | int]]]:
        """
        Retrieves all encrypted submissions from the MevShield pallet storage.

        This function queries the MevShield.Submissions storage map and returns all pending encrypted submissions that
        have been submitted via submit_encrypted but not yet executed via execute_revealed.

        Parameters:
            block: The blockchain block number for the query. If None, uses the current block.
            block_hash: The hash of the block to retrieve the submissions from. Do not specify if using block or reuse_block.
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            A dictionary mapping wrapper_id (as hex string with "0x" prefix) to submission data dictionaries. Each
            submission dictionary contains:
            - author: The SS58 address of the account that submitted the encrypted extrinsic
            - commitment: The blake2_256 hash of the payload_core as bytes (32 bytes)
            - ciphertext: The encrypted blob as bytes (format: [u16 kem_len][kem_ct][nonce24][aead_ct])
            - submitted_in: The block number when the submission was created

            Returns None if no submissions exist in storage at the specified block.

        Note:
            Submissions are automatically pruned after KEY_EPOCH_HISTORY blocks (100 blocks) by the pallet's
            on_initialize hook. Only submissions that have been submitted but not yet executed will be present in
            storage.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        query = await self.substrate.query_map(
            module="MevShield",
            storage_function="Submissions",
            block_hash=block_hash,
        )

        result = {}
        async for q in query:
            key, value = q
            value = value.value
            result["0x" + bytes(key[0]).hex()] = {
                "author": decode_account_id(value.get("author")),
                "commitment": bytes(value.get("commitment")[0]),
                "ciphertext": bytes(value.get("ciphertext")[0]),
                "submitted_in": value.get("submitted_in"),
            }

        return result if result else None

    async def get_minimum_required_stake(self):
        """Returns the minimum required stake threshold for nominator cleanup operations.

        This threshold is used ONLY for cleanup after unstaking operations. If a nominator's remaining stake
        falls below this minimum after an unstake, the remaining stake is forcefully cleared and returned
        to the coldkey to prevent dust accounts.

        This is NOT the minimum checked during staking operations. The actual minimum for staking is determined
        by DefaultMinStake (typically 0.001 TAO plus fees).

        Returns:
            The minimum stake threshold as a Balance object. Nominator stakes below this amount
            are automatically cleared after unstake operations.

        Notes:
            - <https://docs.learnbittensor.org/staking-and-delegation/delegation>
        """
        result = await self.substrate.query(
            module="SubtensorModule", storage_function="NominatorMinRequiredStake"
        )

        return Balance.from_rao(getattr(result, "value", 0))

    async def get_netuids_for_hotkey(
        self,
        hotkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[int]:
        """Retrieves a list of subnet UIDs (netuids) where a given hotkey is a member. This function identifies the
        specific subnets within the Bittensor network where the neuron associated with the hotkey is active.

        Parameters:
            hotkey_ss58: The `SS58` address of the neuron's hotkey.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number at which to perform the query.
            reuse_block: Whether to reuse the last-used block hash when retrieving info.

        Returns:
            A list of netuids where the neuron is a member.

        Notes:
            - <https://docs.learnbittensor.org/glossary#hotkey>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        result = await self.substrate.query_map(
            module="SubtensorModule",
            storage_function="IsNetworkMember",
            params=[hotkey_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        netuids = []
        if result.records:
            async for record in result:
                if record[1].value:
                    netuids.append(record[0])
        return netuids

    async def get_neuron_certificate(
        self,
        hotkey_ss58: str,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[Certificate]:
        """
        Retrieves the TLS certificate for a specific neuron identified by its unique identifier (UID) within a specified
        subnet (netuid) of the Bittensor network.

        Parameters:
            hotkey_ss58: The SS58 address of the neuron's hotkey.
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using block or
                reuse_block.
            reuse_block: Whether to use the last-used block. Do not set if using `block_hash` or `block`.

        Returns:
            Certificate object containing the neuron's TLS public key and algorithm, or `None` if the neuron has
            not registered a certificate.

        This function is used for certificate discovery for setting up mutual tls communication between neurons.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        certificate = cast(
            Union[str, dict],
            await self.query_module(
                module="SubtensorModule",
                name="NeuronCertificates",
                block_hash=block_hash,
                reuse_block=reuse_block,
                params=[netuid, hotkey_ss58],
            ),
        )
        try:
            if certificate:
                return Certificate(certificate)

        except AttributeError:
            return None
        return None

    async def get_neuron_for_pubkey_and_subnet(
        self,
        hotkey_ss58: str,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> "NeuronInfo":
        """
        Retrieves information about a neuron based on its public key (hotkey SS58 address) and the specific subnet UID
        (netuid). This function provides detailed neuron information for a particular subnet within the Bittensor
        network.

        Parameters:
            hotkey_ss58: The `SS58` address of the neuron's hotkey.
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The blockchain block number at which to perform the query.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            Detailed information about the neuron if found, `None` otherwise.

        This function is crucial for accessing specific neuron data and understanding its status, stake, and other
        attributes within a particular subnet of the Bittensor ecosystem.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        uid_query = await self.substrate.query(
            module="SubtensorModule",
            storage_function="Uids",
            params=[netuid, hotkey_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        if (uid := getattr(uid_query, "value", None)) is None:
            return NeuronInfo.get_null_neuron()
        else:
            return await self.neuron_for_uid(
                uid=uid,
                netuid=netuid,
                block_hash=block_hash,
                reuse_block=reuse_block,
            )

    async def get_next_epoch_start_block(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """
        Calculates the first block number of the next epoch for the given subnet.

        If `block` is not provided, the current chain block will be used. Epochs are determined based on the subnet's
        tempo (i.e., blocks per epoch). The result is the block number at which the next epoch will begin.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The reference block to calculate from. If `None`, uses the current chain block height.
            block_hash: The blockchain block number at which to perform the query.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            int: The block number at which the next epoch will start.

        Notes:
            - <https://docs.learnbittensor.org/glossary#tempo>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        tempo = await self.tempo(netuid=netuid, block_hash=block_hash)
        current_block = block or await self.block

        if not tempo:
            return None

        blocks_until = await self.blocks_until_next_epoch(
            netuid=netuid, tempo=tempo, block_hash=block_hash
        )

        if blocks_until is None:
            return None

        return current_block + blocks_until + 1

    async def get_owned_hotkeys(
        self,
        coldkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[str]:
        """
        Retrieves all hotkeys owned by a specific coldkey address.

        Parameters:
            coldkey_ss58: The SS58 address of the coldkey to query.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number for the query.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            A list of hotkey SS58 addresses owned by the coldkey.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        owned_hotkeys = await self.substrate.query(
            module="SubtensorModule",
            storage_function="OwnedHotkeys",
            params=[coldkey_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )

        return [decode_account_id(hotkey[0]) for hotkey in owned_hotkeys or []]

    async def get_parents(
        self,
        hotkey_ss58: str,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[tuple[float, str]]:
        """This method retrieves the parent of a given hotkey and netuid. It queries the SubtensorModule's ParentKeys
        storage function to get the children and formats them before returning as a tuple.

        Parameters:
            hotkey_ss58: The child hotkey SS58.
            netuid: The netuid value.
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            A list of formatted parents [(proportion, parent)]

        Notes:
            - <https://docs.learnbittensor.org/validators/child-hotkeys>
            - :meth:`get_children` for retrieving child keys
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        parents = await self.substrate.query(
            module="SubtensorModule",
            storage_function="ParentKeys",
            params=[hotkey_ss58, netuid],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        if parents:
            formatted_parents = []
            for proportion, parent in parents.value:
                # Convert U64 to int
                formatted_child = decode_account_id(parent[0])
                normalized_proportion = u64_normalized_float(proportion)
                formatted_parents.append((normalized_proportion, formatted_child))
            return formatted_parents

        return []

    async def get_proxies(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> dict[str, list[ProxyInfo]]:
        """
        Retrieves all proxy relationships from the chain.

        This method queries the Proxy.Proxies storage map across all accounts and returns a dictionary mapping each real
        account (delegator) to its list of proxy relationships.

        Parameters:
            block: The blockchain block number for the query. If `None`, queries the latest block.
            block_hash: The hash of the block at which to check the parameter. Do not set if using `block` or `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            Dictionary mapping real account SS58 addresses to lists of ProxyInfo objects. Each ProxyInfo contains the
                delegate address, proxy type, and delay for that proxy relationship.

        Notes:
            - This method queries all proxy relationships on the chain, which may be resource-intensive for large
              networks. Consider using :meth:`get_proxies_for_real_account` for querying specific accounts.
            - See: <https://docs.learnbittensor.org/keys/proxies>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        query_map = await self.substrate.query_map(
            module="Proxy",
            storage_function="Proxies",
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )

        proxies = {}
        async for record in query_map:
            real_account, proxy_list = ProxyInfo.from_query_map_record(record)
            proxies[real_account] = proxy_list
        return proxies

    async def get_proxies_for_real_account(
        self,
        real_account_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> tuple[list[ProxyInfo], Balance]:
        """
        Returns proxy/ies associated with the provided real account.

        This method queries the Proxy.Proxies storage for a specific real account and returns all proxy relationships
        where this real account is the delegator. It also returns the deposit amount reserved for these proxies.

        Parameters:
            real_account_ss58: SS58 address of the real account (delegator) whose proxies to retrieve.
            block: The blockchain block number for the query.
            block_hash: The hash of the block at which to check the parameter. Do not set if using `block` or `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            Tuple containing:
                - List of ProxyInfo objects representing all proxy relationships for the real account. Each ProxyInfo
                    contains delegate address, proxy type, and delay.
                - Balance object representing the reserved deposit amount for these proxies. This deposit is held as
                    long as the proxy relationships exist and is returned when proxies are removed.

        Notes:
            - If the account has no proxies, returns an empty list and a zero balance.
            - See: <https://docs.learnbittensor.org/keys/proxies/create-proxy>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        query = await self.substrate.query(
            module="Proxy",
            storage_function="Proxies",
            params=[real_account_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        return ProxyInfo.from_query(query)

    async def get_proxy_announcement(
        self,
        delegate_account_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[ProxyAnnouncementInfo]:
        """
        Retrieves proxy announcements for a specific delegate account.

        This method queries the Proxy.Announcements storage for announcements made by the given delegate proxy account.
        Announcements allow a proxy to declare its intention to execute a call on behalf of a real account after a delay
        period.

        Parameters:
            delegate_account_ss58: SS58 address of the delegate proxy account whose announcements to retrieve.
            block: The blockchain block number for the query. If `None`, queries the latest block.
            block_hash: The hash of the block at which to check the parameter. Do not set if using `block` or `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            List of ProxyAnnouncementInfo objects. Each object contains the real account address, call hash, and block
                height at which the announcement was made.

        Notes:
            - If the delegate has no announcements, returns an empty list.
            - See: <https://docs.learnbittensor.org/keys/proxies>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        query = await self.substrate.query(
            module="Proxy",
            storage_function="Announcements",
            params=[delegate_account_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        return ProxyAnnouncementInfo.from_dict(query.value[0])

    async def get_proxy_announcements(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> dict[str, list[ProxyAnnouncementInfo]]:
        """
        Retrieves all proxy announcements from the chain.

        This method queries the Proxy.Announcements storage map across all delegate accounts and returns a dictionary
        mapping each delegate to its list of pending announcements.

        Parameters:
            block: The blockchain block number for the query. If `None`, queries the latest block.
            block_hash: The hash of the block at which to check the parameter. Do not set if using `block` or `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            Dictionary mapping delegate account SS58 addresses to lists of ProxyAnnouncementInfo objects.
            Each ProxyAnnouncementInfo contains the real account address, call hash, and block height.

        Notes:
            - This method queries all announcements on the chain, which may be resource-intensive for large networks.
              Consider using :meth:`get_proxy_announcement` for querying specific delegates.
            - See: <https://docs.learnbittensor.org/keys/proxies>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        query_map = await self.substrate.query_map(
            module="Proxy",
            storage_function="Announcements",
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        announcements = {}
        async for record in query_map:
            delegate, proxy_list = ProxyAnnouncementInfo.from_query_map_record(record)
            announcements[delegate] = proxy_list
        return announcements

    async def get_proxy_constants(
        self,
        constants: Optional[list[str]] = None,
        as_dict: bool = False,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Union["ProxyConstants", dict]:
        """
        Fetches runtime configuration constants from the `Proxy` pallet.

        This method retrieves on-chain configuration constants that define deposit requirements, proxy limits, and
        announcement constraints for the Proxy pallet. These constants govern how proxy accounts operate within the
        Subtensor network.

        Parameters:
            constants: Optional list of specific constant names to fetch. If omitted, all constants defined in
                `ProxyConstants.constants_names()` are queried. Valid constant names include: "AnnouncementDepositBase",
                "AnnouncementDepositFactor", "MaxProxies", "MaxPending", "ProxyDepositBase", "ProxyDepositFactor".
            as_dict: If True, returns the constants as a dictionary instead of a `ProxyConstants` object.
            block: The blockchain block number for the query. If `None`, queries the latest block.
            block_hash: The hash of the block at which to check the parameter. Do not set if using `block` or `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            If `as_dict` is False: ProxyConstants object containing all requested constants.
            If `as_dict` is True: Dictionary mapping constant names to their values (Balance objects for deposit
                constants, integers for limit constants).

        Notes:
            - All Balance amounts are returned in RAO. Constants reflect the current chain configuration at the specified
              block.
            - See: <https://docs.learnbittensor.org/keys/proxies>
        """
        result = {}
        const_names = constants or ProxyConstants.constants_names()

        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        for const_name in const_names:
            query = await self.query_constant(
                module_name="Proxy",
                constant_name=const_name,
                block=block,
                block_hash=block_hash,
                reuse_block=reuse_block,
            )

            if query is not None:
                result[const_name] = query.value

        proxy_constants = ProxyConstants.from_dict(result)

        return proxy_constants.to_dict() if as_dict else proxy_constants

    async def get_revealed_commitment(
        self,
        netuid: int,
        uid: int,
        block: Optional[int] = None,
    ) -> Optional[tuple[tuple[int, str], ...]]:
        """Returns uid related revealed commitment for a given netuid.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            uid: The neuron uid to retrieve the commitment from.
            block: The block number to retrieve the commitment from.

        Returns:
            A tuple of reveal block and commitment message.

        Example:

            ( (12, "Alice message 1"), (152, "Alice message 2") )
            ( (12, "Bob message 1"), (147, "Bob message 2") )

        Notes:
            - <https://docs.learnbittensor.org/glossary#commit-reveal>
        """
        try:
            meta_info = await self.get_metagraph_info(netuid, block=block)
            if meta_info:
                hotkey_ss58 = meta_info.hotkeys[uid]
            else:
                raise ValueError(f"Subnet with netuid {netuid} does not exist.")
        except IndexError:
            raise ValueError(f"Subnet {netuid} does not have a neuron with uid {uid}.")

        return await self.get_revealed_commitment_by_hotkey(
            netuid=netuid, hotkey_ss58=hotkey_ss58, block=block
        )

    async def get_revealed_commitment_by_hotkey(
        self,
        netuid: int,
        hotkey_ss58: Optional[str] = None,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[tuple[tuple[int, str], ...]]:
        """Retrieves hotkey related revealed commitment for a given subnet.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            hotkey_ss58: The ss58 address of the committee member.
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            A tuple of reveal block and commitment message.

        # TODO: add example to clarify return ordering and units; @roman can you help w this?
        Notes:
            - <https://docs.learnbittensor.org/glossary#commit-reveal>
        """
        if not is_valid_ss58_address(address=hotkey_ss58):
            raise ValueError(f"Invalid ss58 address {hotkey_ss58} provided.")

        query = await self.query_module(
            module="Commitments",
            name="RevealedCommitments",
            params=[netuid, hotkey_ss58],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        if query is None:
            return None
        return tuple(decode_revealed_commitment(pair) for pair in query)

    async def get_root_claim_type(
        self,
        coldkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Union[str, dict]:
        """Return the configured root claim type for a given coldkey.

        The root claim type controls how dividends from staking to the Root Subnet (subnet 0) are processed when they
        are claimed:

        - `Swap` (default): Alpha dividends are swapped to TAO at claim time and restaked on the root subnet.
        - `Keep`: Alpha dividends remain as Alpha on the originating subnets.

        Parameters:
            coldkey_ss58: The SS58 address of the coldkey whose root claim preference to query.
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to query the claim type. Do not specify if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not specify if using `block` or `block_hash`.

        Returns:

            The root claim type as a string, either `Swap` or `Keep`,
            or dict for "KeepSubnets" in format {"KeepSubnets": {"subnets": [1, 2, 3]}}.

        Notes:
            - The claim type applies to both automatic and manual root claims; it does not affect the original TAO stake
              on subnet 0, only how Alpha dividends are treated.
            - See: <https://docs.learnbittensor.org/staking-and-delegation/root-claims>
            - See also: <https://docs.learnbittensor.org/staking-and-delegation/root-claims/managing-root-claims>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        query = await self.substrate.query(
            module="SubtensorModule",
            storage_function="RootClaimType",
            params=[coldkey_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        # Query returns enum as dict: {"Swap": ()} or {"Keep": ()} or {"KeepSubnets": {"subnets": [1, 2, 3]}}
        variant_name = next(iter(query.keys()))
        variant_value = query[variant_name]

        # For simple variants (Swap, Keep), value is empty tuple, return string
        if not variant_value or variant_value == ():
            return variant_name

        # For KeepSubnets, value contains the data, return full dict structure
        if isinstance(variant_value, dict) and "subnets" in variant_value:
            subnets_raw = variant_value["subnets"]
            subnets = list(subnets_raw[0])

            return {variant_name: {"subnets": subnets}}

        return {variant_name: variant_value}

    async def get_root_alpha_dividends_per_subnet(
        self,
        hotkey_ss58: str,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Balance:
        """Retrieves the root alpha dividends per subnet for a given hotkey.

        This storage tracks the root alpha dividends that a hotkey has received on a specific subnet.
        It is updated during block emission distribution when root alpha is distributed to validators.

        Parameters:
            hotkey_ss58: The ss58 address of the root validator hotkey.
            netuid: The unique identifier of the subnet.
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            Balance: The root alpha dividends for this hotkey on this subnet in Rao, with unit set to netuid.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        query = await self.substrate.query(
            module="SubtensorModule",
            storage_function="RootAlphaDividendsPerSubnet",
            params=[netuid, hotkey_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        return Balance.from_rao(query.value).set_unit(netuid=netuid)

    async def get_root_claimable_rate(
        self,
        hotkey_ss58: str,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> float:
        """Return the fraction of root stake currently claimable on a subnet.

        This method returns a normalized rate representing how much Alpha dividends are currently claimable on the given
        subnet relative to the validator's root stake. It is primarily a low-level helper; most users should call
        :meth:`get_root_claimable_stake` instead to obtain a Balance.

        Parameters:
            hotkey_ss58: The SS58 address of the root validator hotkey.
            netuid: The unique identifier of the subnet whose claimable rate to compute.
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to query. Do not specify if using `block` or `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not specify if using `block` or `block_hash`.

        Returns:
            A float representing the claimable rate for this subnet (approximately in the range `[0.0, 1.0]`). A value
            of 0.0 means there are currently no claimable Alpha dividends on the subnet.

        Notes:
            - Use :meth:`get_root_claimable_stake` to retrieve the actual claimable amount as a `Balance` object.
            - See: <https://docs.learnbittensor.org/staking-and-delegation/root-claims/managing-root-claims>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        all_rates = await self.get_root_claimable_all_rates(
            hotkey_ss58=hotkey_ss58,
            block_hash=block_hash,
        )
        return all_rates.get(netuid, 0.0)

    async def get_root_claimable_all_rates(
        self,
        hotkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> dict[int, float]:
        """Retrieves all root claimable rates from a given hotkey address for all subnets with this validator.

        Parameters:
            hotkey_ss58: The SS58 address of the root validator hotkey.
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to query. Do not specify if using `block` or `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not specify if using `block` or `block_hash`.

        Returns:
            Dictionary mapping `netuid` to a float claimable rate (approximately in the range `[0.0, 1.0]`) for that
            subnet. Missing entries imply no claimable Alpha dividends for that subnet.

        Notes:
            - See: <https://docs.learnbittensor.org/staking-and-delegation/root-claims/managing-root-claims>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        query = await self.substrate.query(
            module="SubtensorModule",
            storage_function="RootClaimable",
            params=[hotkey_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        bits_list = next(iter(query.value))
        return {bits[0]: fixed_to_float(bits[1], frac_bits=32) for bits in bits_list}

    async def get_root_claimable_stake(
        self,
        coldkey_ss58: str,
        hotkey_ss58: str,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Balance:
        """Return the currently claimable Alpha staking dividends for a coldkey from a root validator on a subnet.

        Parameters:
            coldkey_ss58: The SS58 address of the delegator's coldkey.
            hotkey_ss58: The SS58 address of the root validator hotkey.
            netuid: The subnet ID where Alpha dividends will be claimed.
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to query. Do not specify if using `block` or `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not specify if using `block` or `block_hash`.

        Returns:
            `Balance` representing the Alpha stake currently available to claim on the specified subnet (unit is the
            subnet's Alpha token).

        Notes:
            - After a successful manual or automatic claim, this value typically drops to zero for that subnet until new
              dividends accumulate.
            - The underlying TAO stake on the Root Subnet remains unaffected; only Alpha dividends are moved or swapped
              according to the configured root claim type.
            - See: <https://docs.learnbittensor.org/staking-and-delegation/root-claims>
            - See also: <https://docs.learnbittensor.org/staking-and-delegation/root-claims/managing-root-claims>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        root_stake = await self.get_stake(
            coldkey_ss58=coldkey_ss58,
            hotkey_ss58=hotkey_ss58,
            netuid=0,  # root netuid
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        root_claimable_rate = await self.get_root_claimable_rate(
            hotkey_ss58=hotkey_ss58,
            netuid=netuid,
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        root_claimable_stake = (root_claimable_rate * root_stake).set_unit(
            netuid=netuid
        )
        root_claimed = await self.get_root_claimed(
            coldkey_ss58=coldkey_ss58,
            hotkey_ss58=hotkey_ss58,
            netuid=netuid,
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        return max(
            root_claimable_stake - root_claimed, Balance(0).set_unit(netuid=netuid)
        )

    async def get_root_claimed(
        self,
        coldkey_ss58: str,
        hotkey_ss58: str,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Balance:
        """Return the total Alpha dividends already claimed for a coldkey from a root validator on a subnet.

        Parameters:
            coldkey_ss58: The SS58 address of the delegator's coldkey.
            hotkey_ss58: The SS58 address of the root validator hotkey.
            netuid: The unique identifier of the subnet.
            block: The block number to query. Do not specify if using `block_hash` or `reuse_block`.
            block_hash: The block hash at which to query. Do not specify if using `block` or `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not specify if using `block` or `block_hash`.

        Returns:
            `Balance` representing the cumulative Alpha stake that has already been claimed from the root validator on
            the specified subnet.

        Notes:
            - See: <https://docs.learnbittensor.org/staking-and-delegation/root-claims/managing-root-claims>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        query = await self.substrate.query(
            module="SubtensorModule",
            storage_function="RootClaimed",
            params=[netuid, hotkey_ss58, coldkey_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        return Balance.from_rao(query.value).set_unit(netuid=netuid)

    async def get_stake(
        self,
        coldkey_ss58: str,
        hotkey_ss58: str,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Balance:
        """
        Returns the stake under a coldkey - hotkey pairing.

        Parameters:
            hotkey_ss58: The SS58 address of the hotkey.
            coldkey_ss58: The SS58 address of the coldkey.
            netuid: The subnet ID.
            block: The block number at which to query the stake information.
            block_hash: The hash of the block to retrieve the stake from. Do not specify if using block
                or reuse_block
            reuse_block: Whether to use the last-used block. Do not set if using `block_hash` or `block`.

        Returns:
            Balance: The stake under the coldkey - hotkey pairing.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)

        alpha_shares = await self.query_subtensor(
            name="Alpha",
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
            params=[hotkey_ss58, coldkey_ss58, netuid],
        )
        hotkey_alpha_result = await self.query_subtensor(
            name="TotalHotkeyAlpha",
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
            params=[hotkey_ss58, netuid],
        )
        hotkey_shares = await self.query_subtensor(
            name="TotalHotkeyShares",
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
            params=[hotkey_ss58, netuid],
        )

        hotkey_alpha: int = getattr(hotkey_alpha_result, "value", 0)
        alpha_shares_as_float = fixed_to_float(alpha_shares)
        hotkey_shares_as_float = fixed_to_float(hotkey_shares)

        if hotkey_shares_as_float == 0:
            return Balance.from_rao(0).set_unit(netuid=netuid)

        stake = alpha_shares_as_float / hotkey_shares_as_float * hotkey_alpha

        return Balance.from_rao(int(stake)).set_unit(netuid=netuid)

    async def get_stake_add_fee(
        self,
        amount: Balance,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Balance:
        """
        Calculates the fee for adding new stake to a hotkey.

        Parameters:
            amount: Amount of stake to add in TAO
            netuid: Netuid of subnet
            block: The block number for which the children are to be retrieved.
            block_hash: The hash of the block to retrieve the subnet unique identifiers from.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            The calculated stake fee as a Balance object in TAO.

        Notes:
            - <https://docs.learnbittensor.org/learn/fees>
        """
        check_balance_amount(amount)
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        sim_swap_result = await self.sim_swap(
            origin_netuid=0,
            destination_netuid=netuid,
            amount=amount,
            block_hash=block_hash,
        )
        return sim_swap_result.tao_fee

    async def get_stake_movement_fee(
        self,
        origin_netuid: int,
        destination_netuid: int,
        amount: Balance,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Balance:
        """
        Calculates the fee for moving stake between hotkeys/subnets/coldkeys.

        Parameters:
            origin_netuid: Netuid of source subnet.
            destination_netuid: Netuid of the destination subnet.
            amount: Amount of stake to move.
            block: The block number for which the children are to be retrieved.
            block_hash: The hash of the block to retrieve the subnet unique identifiers from.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            The calculated stake fee as a Balance object

        Notes:
            - <https://docs.learnbittensor.org/learn/fees>
        """
        check_balance_amount(amount)
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        sim_swap_result = await self.sim_swap(
            origin_netuid=origin_netuid,
            destination_netuid=destination_netuid,
            amount=amount,
            block_hash=block_hash,
        )
        return sim_swap_result.tao_fee

    async def get_stake_for_coldkey_and_hotkey(
        self,
        coldkey_ss58: str,
        hotkey_ss58: str,
        netuids: Optional[UIDs] = None,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> dict[int, StakeInfo]:
        """
        Retrieves all coldkey-hotkey pairing stake across specified (or all) subnets

        Parameters:
            coldkey_ss58: The SS58 address of the coldkey.
            hotkey_ss58: The SS58 address of the hotkey.
            netuids: The subnet IDs to query for. Set to `None` for all subnets.
            block: The block number for which the children are to be retrieved.
            block_hash: The hash of the block to retrieve the subnet unique identifiers from.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            A netuid to StakeInfo mapping of all stakes across all subnets.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        if not block_hash and reuse_block:
            block_hash = self.substrate.last_block_hash
        elif not block_hash:
            block_hash = await self.substrate.get_chain_head()
        if netuids is None:
            all_netuids = await self.get_all_subnets_netuid(block_hash=block_hash)
        else:
            all_netuids = netuids
        results = await asyncio.gather(
            *[
                self.query_runtime_api(
                    runtime_api="StakeInfoRuntimeApi",
                    method="get_stake_info_for_hotkey_coldkey_netuid",
                    params=[hotkey_ss58, coldkey_ss58, netuid],
                    block_hash=block_hash,
                )
                for netuid in all_netuids
            ]
        )
        return {
            netuid: StakeInfo.from_dict(result)
            for (netuid, result) in zip(all_netuids, results)
        }

    async def get_stake_info_for_coldkey(
        self,
        coldkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list["StakeInfo"]:
        """
        Retrieves the stake information for a given coldkey.

        Parameters:
            coldkey_ss58: The SS58 address of the coldkey.
            block: The block number at which to query the stake information.
            block_hash: The hash of the blockchain block number for the query.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            List of StakeInfo objects.
        """
        result = await self.query_runtime_api(
            runtime_api="StakeInfoRuntimeApi",
            method="get_stake_info_for_coldkey",
            params=[coldkey_ss58],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )

        if result is None:
            return []

        stakes: list[StakeInfo] = StakeInfo.list_from_dicts(result)
        return [stake for stake in stakes if stake.stake > 0]

    async def get_stake_info_for_coldkeys(
        self,
        coldkey_ss58s: list[str],
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> dict[str, list["StakeInfo"]]:
        """
        Retrieves the stake information for multiple coldkeys.

        Parameters:
            coldkey_ss58s: A list of SS58 addresses of the coldkeys to query.
            block: The block number at which to query the stake information.
            block_hash: The hash of the blockchain block number for the query.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            The dictionary mapping coldkey addresses to a list of StakeInfo objects.
        """
        query = await self.query_runtime_api(
            runtime_api="StakeInfoRuntimeApi",
            method="get_stake_info_for_coldkeys",
            params=[coldkey_ss58s],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )

        if query is None:
            return {}

        return {
            decode_account_id(ck): StakeInfo.list_from_dicts(st_info)
            for ck, st_info in query
        }

    async def get_stake_for_hotkey(
        self,
        hotkey_ss58: str,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Balance:
        """Retrieves the total stake for a given hotkey on a specific subnet.

        Parameters:
            hotkey_ss58: The SS58 address of the hotkey.
            netuid: The subnet ID to query for.
            block: The blockchain block number for the query.
            block_hash: The hash of the block to retrieve the stake from.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            Balance: The total stake for the hotkey on the specified subnet.
        """
        hotkey_alpha_query = await self.query_subtensor(
            name="TotalHotkeyAlpha",
            params=[hotkey_ss58, netuid],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        balance = Balance.from_rao(hotkey_alpha_query.value)
        balance.set_unit(netuid=netuid)
        return balance

    get_hotkey_stake = get_stake_for_hotkey

    async def get_stake_weight(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[float]:
        """
        Retrieves the stake weight for all hotkeys in a given subnet.

        Parameters:
            netuid: Netuid of subnet.
            block: The block number for which the children are to be retrieved.
            block_hash: The hash of the block to retrieve the subnet unique identifiers from.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            A list of stake weights for all hotkeys in the specified subnet.
        """
        block_hash = await self.determine_block_hash(
            block=block, block_hash=block_hash, reuse_block=reuse_block
        )
        result = await self.substrate.query(
            module="SubtensorModule",
            storage_function="StakeWeight",
            params=[netuid],
            block_hash=block_hash,
        )
        return [u16_normalized_float(w) for w in result]

    async def get_subnet_burn_cost(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[Balance]:
        """
        Retrieves the burn cost for registering a new subnet within the Bittensor network. This cost represents the
            amount of Tao that needs to be locked or burned to establish a new

        Parameters:
            block: The blockchain block number for the query.
            block_hash: The blockchain block_hash of the block id.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            int: The burn cost for subnet registration.

        The subnet burn cost is an important economic parameter, reflecting the network's mechanisms for controlling
            the proliferation of subnets and ensuring their commitment to the network's long-term viability.
        """
        lock_cost = await self.query_runtime_api(
            runtime_api="SubnetRegistrationRuntimeApi",
            method="get_network_registration_cost",
            params=[],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        if lock_cost is not None:
            return Balance.from_rao(lock_cost)
        else:
            return lock_cost

    async def get_subnet_hyperparameters(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional["SubnetHyperparameters"]:
        """
        Retrieves the hyperparameters for a specific subnet within the Bittensor network. These hyperparameters define
        the operational settings and rules governing the subnet's behavior.

        Parameters:
            netuid: The network UID of the subnet to query.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number for the query.
            reuse_block: Whether to reuse the last-used blockchain hash.

        Returns:
            The subnet's hyperparameters, or `None` if not available.

        Understanding the hyperparameters is crucial for comprehending how subnets are configured and managed, and how
        they interact with the network's consensus and incentive mechanisms.
        """
        result = await self.query_runtime_api(
            runtime_api="SubnetInfoRuntimeApi",
            method="get_subnet_hyperparams_v2",
            params=[netuid],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )

        if not result:
            return None

        return SubnetHyperparameters.from_dict(result)

    async def get_subnet_info(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional["SubnetInfo"]:
        """
        Retrieves detailed information about subnet within the Bittensor network.
        This function provides comprehensive data on subnet, including its characteristics and operational parameters.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The block number for which the children are to be retrieved.
            block_hash: The hash of the block to retrieve the subnet unique identifiers from.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            SubnetInfo: A SubnetInfo objects, each containing detailed information about a subnet.

        Gaining insights into the subnet's details assists in understanding the network's composition, the roles of
        different subnets, and their unique features.
        """
        result = await self.query_runtime_api(
            runtime_api="SubnetInfoRuntimeApi",
            method="get_subnet_info_v2",
            params=[netuid],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        if not result:
            return None
        return SubnetInfo.from_dict(result)

    async def get_subnet_owner_hotkey(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[str]:
        """
        Retrieves the hotkey of the subnet owner for a given network UID.

        This function queries the subtensor network to fetch the hotkey of the owner of a subnet specified by its
        netuid. If no data is found or the query fails, the function returns `None`.

        Parameters:
            netuid: The network UID of the subnet to fetch the owner's hotkey for.
            block: The blockchain block number for the query.
            block_hash: The blockchain block_hash representation of the block id.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            The hotkey of the subnet owner if available; None otherwise.
        """
        return await self.query_subtensor(
            name="SubnetOwnerHotkey",
            params=[netuid],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )

    async def get_subnet_price(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Balance:
        """Gets the current Alpha price in TAO for the specified subnet.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The hash of the block to retrieve the stake from. Do not specify if using `block` or `reuse_block`.
            reuse_block: Whether to use the last-used block. Do not set if using `block_hash` or `block`.

        Returns:
            The current Alpha price in TAO units for the specified subnet.

        Notes:
            Subnet 0 (root network) always returns 1 TAO since it uses TAO directly rather than Alpha.
        """
        # SN0 price is always 1 TAO
        if netuid == 0:
            return Balance.from_tao(1)

        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        call = await self.substrate.runtime_call(
            api="SwapRuntimeApi",
            method="current_alpha_price",
            params=[netuid],
            block_hash=block_hash,
        )
        price_rao = call.value
        return Balance.from_rao(price_rao)

    async def get_subnet_prices(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> dict[int, Balance]:
        """Gets the current Alpha price in TAO for all subnets.

        Parameters:
            block: The block number for which the children are to be retrieved.
            block_hash: The hash of the block to retrieve the subnet unique identifiers from.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            A dictionary mapping subnet unique ID (netuid) to the current Alpha price in TAO units.

        Notes:
            Subnet 0 (root network) always has a price of 1 TAO since it uses TAO directly rather than Alpha.
        """
        block_hash = await self.determine_block_hash(
            block=block, block_hash=block_hash, reuse_block=reuse_block
        )

        current_sqrt_prices = await self.substrate.query_map(
            module="Swap",
            storage_function="AlphaSqrtPrice",
            block_hash=block_hash,
            page_size=129,  # total number of subnets
        )

        prices = {}
        async for id_, current_sqrt_price in current_sqrt_prices:
            current_sqrt_price = fixed_to_float(current_sqrt_price)
            current_price = current_sqrt_price * current_sqrt_price
            current_price_in_tao = Balance.from_rao(int(current_price * 1e9))
            prices.update({id_: current_price_in_tao})

        # SN0 price is always 1 TAO
        prices.update({0: Balance.from_tao(1)})
        return prices

    async def get_subnet_reveal_period_epochs(
        self, netuid: int, block: Optional[int] = None, block_hash: Optional[str] = None
    ) -> int:
        """Retrieves the SubnetRevealPeriodEpochs hyperparameter for a specified subnet.

        This hyperparameter determines the number of epochs that must pass before a committed weight can be revealed
        in the commit-reveal mechanism.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query. Do not specify if using `block_hash`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block`.

        Returns:
            The number of epochs in the reveal period for the subnet.

        Notes:
            - <https://docs.learnbittensor.org/glossary#commit-reveal>

        """
        block_hash = await self.determine_block_hash(block, block_hash)
        return await self.get_hyperparameter(
            param_name="RevealPeriodEpochs", block_hash=block_hash, netuid=netuid
        )

    async def get_subnet_validator_permits(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[list[bool]]:
        """
        Retrieves the list of validator permits for a given subnet as boolean values.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            block: The blockchain block number for the query.
            block_hash: The blockchain block_hash representation of the block id.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            A list of boolean values representing validator permits, or None if not available.
        """
        query = await self.query_subtensor(
            name="ValidatorPermit",
            params=[netuid],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        return query.value if query is not None and hasattr(query, "value") else query

    async def get_timelocked_weight_commits(
        self,
        netuid: int,
        mechid: int = 0,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[tuple[str, int, str, int]]:
        """Retrieves CRv4 (Commit-Reveal version 4) weight commit information for a specific subnet.

        This method retrieves timelocked weight commitments made by validators using the commit-reveal mechanism.
        The raw byte/vector encoding from the chain is automatically parsed and converted into a structured format
        via `WeightCommitInfo`.

        Parameters:
            netuid: The unique identifier of the subnet.
            mechid: Subnet mechanism identifier (default 0 for primary mechanism).
            block: The blockchain block number for the query. Do not specify if using `block_hash` or
                `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            A list of commit details, where each item is a tuple containing:

                - ss58_address: The SS58 address of the committer.
                - commit_block: The block number when the commitment was made.
                - commit_message: The commit message (encoded commitment data).
                - reveal_round: The drand round when the commitment can be revealed.

        Notes:
            The list may be empty if there are no commits found.
            - <https://docs.learnbittensor.org/resources/glossary#commit-reveal>
        """
        storage_index = get_mechid_storage_index(netuid, mechid)
        block_hash = await self.determine_block_hash(
            block=block, block_hash=block_hash, reuse_block=reuse_block
        )
        result = await self.substrate.query_map(
            module="SubtensorModule",
            storage_function="TimelockedWeightCommits",
            params=[storage_index],
            block_hash=block_hash,
        )

        commits = result.records[0][1] if result.records else []
        return [WeightCommitInfo.from_vec_u8_v2(commit) for commit in commits]

    async def get_timestamp(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> datetime:
        """Retrieves the datetime timestamp for a given block.

        This method queries the Timestamp pallet to get the block's timestamp. The on-chain timestamp is stored in
        milliseconds (Unix timestamp in milliseconds), which is automatically converted to a Python datetime object
        (Unix timestamp in seconds).

        Parameters:
            block: The blockchain block number for the query. Do not specify if using `block_hash` or
                `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            A datetime object representing the timestamp of the specified block.

        Notes:
            - <https://docs.learnbittensor.org/resources/glossary#block>
        """
        res = await self.query_module(
            "Timestamp",
            "Now",
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        unix = res.value
        return datetime.fromtimestamp(unix / 1000, tz=timezone.utc)

    async def get_total_subnets(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """Retrieves the total number of subnets within the Bittensor network as of a specific blockchain block.

        Parameters:
            block: The blockchain block number for the query.
            block_hash: The blockchain block_hash representation of block id.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            The total number of subnets in the network.

        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        result = await self.substrate.query(
            module="SubtensorModule",
            storage_function="TotalNetworks",
            params=[],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        return getattr(result, "value", None)

    async def get_transfer_fee(
        self,
        wallet: "Wallet",
        destination_ss58: str,
        amount: Balance,
        keep_alive: bool = True,
    ) -> Balance:
        """
        Calculates the transaction fee for transferring tokens from a wallet to a specified destination address. This
        function simulates the transfer to estimate the associated cost, taking into account the current network
        conditions and transaction complexity.

        Parameters:
            wallet: The wallet from which the transfer is initiated.
            destination_ss58: The `SS58` address of the destination account.
            amount: The amount of tokens to be transferred, specified as a Balance object, or in Tao (float) or Rao
                (int) units.
            keep_alive: Whether the transfer fee should be calculated based on keeping the wallet alive (existential
                deposit) or not.

        Returns:
            bittensor.utils.balance.Balance: The estimated transaction fee for the transfer, represented as a Balance
                object.

        Notes:

            - <https://docs.learnbittensor.org/learn/fees>
        """
        check_balance_amount(amount)
        call_params: dict[str, Union[int, str, bool]]
        call_function, call_params = get_transfer_fn_params(
            amount, destination_ss58, keep_alive
        )

        call = await self.compose_call(
            call_module="Balances",
            call_function=call_function,
            call_params=call_params,
        )

        try:
            payment_info = await self.substrate.get_payment_info(
                call=call, keypair=wallet.coldkeypub
            )
        except Exception as e:
            logging.error(f":cross_mark: [red]Failed to get payment info: [/red]{e}")
            payment_info = {"partial_fee": int(2e7)}  # assume  0.02 Tao

        return Balance.from_rao(payment_info["partial_fee"])

    async def get_unstake_fee(
        self,
        netuid: int,
        amount: Balance,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Balance:
        """Calculates the fee for unstaking from a hotkey.

        Parameters:
            netuid: The unique identifier of the subnet.
            amount: Amount of stake to unstake in TAO.
            block: The blockchain block number for the query.
            block_hash: The blockchain block_hash representation of the block id.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            The calculated stake fee as a Balance object in Alpha.

        Notes:
            - <https://docs.learnbittensor.org/learn/fees>
        """
        check_balance_amount(amount)
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        sim_swap_result = await self.sim_swap(
            origin_netuid=netuid,
            destination_netuid=0,
            amount=amount,
            block_hash=block_hash,
        )
        return sim_swap_result.alpha_fee.set_unit(netuid=netuid)

    async def get_vote_data(
        self,
        proposal_hash: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional["ProposalVoteData"]:
        # TODO: is this all deprecated? Didn't subtensor senate stuff get removed?
        """
        Retrieves the voting data for a specific proposal on the Bittensor blockchain. This data includes information
        about how senate members have voted on the proposal.

        Parameters:
            proposal_hash: The hash of the proposal for which voting data is requested.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number to query the voting data.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            An object containing the proposal's voting data, or `None` if not found.

        This function is important for tracking and understanding the decision-making processes within the Bittensor
        network, particularly how proposals are received and acted upon by the governing body.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        vote_data: dict[str, Any] = await self.substrate.query(
            module="Triumvirate",
            storage_function="Voting",
            params=[proposal_hash],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )

        if vote_data is None:
            return None

        return ProposalVoteData.from_dict(vote_data)

    async def get_uid_for_hotkey_on_subnet(
        self,
        hotkey_ss58: str,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """
        Retrieves the unique identifier (UID) for a neuron's hotkey on a specific subnet.

        Parameters:
            hotkey_ss58: The `SS58` address of the neuron's hotkey.
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The blockchain block_hash representation of the block id.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            The UID of the neuron if it is registered on the subnet, `None` otherwise.

        The UID is a critical identifier within the network, linking the neuron's hotkey to its operational and
        governance activities on a particular subnet.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        result = await self.substrate.query(
            module="SubtensorModule",
            storage_function="Uids",
            params=[netuid, hotkey_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        return getattr(result, "value", result)

    async def filter_netuids_by_registered_hotkeys(
        self,
        all_netuids: Iterable[int],
        filter_for_netuids: Iterable[int],
        all_hotkeys: Iterable["Wallet"],
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[int]:
        """
        Filters netuids by combining netuids from all_netuids and netuids with registered hotkeys.

        If filter_for_netuids is empty/None:
            Returns all netuids where hotkeys from all_hotkeys are registered.

        If filter_for_netuids is provided:
            Returns the union of:
            - Netuids from all_netuids that are in filter_for_netuids, AND
            - Netuids with registered hotkeys that are in filter_for_netuids

        This allows you to get netuids that are either in your specified list (all_netuids) or have registered hotkeys,
        as long as they match filter_for_netuids.

        Parameters:
            all_netuids (Iterable[int]): A list of netuids to consider for filtering.
            filter_for_netuids (Iterable[int]): A subset of netuids to restrict the result to. If None/empty, returns
                all netuids with registered hotkeys.
            all_hotkeys (Iterable[Wallet]): Hotkeys to check for registration.
            block (Optional[int]): The blockchain block number for the query.
            block_hash (Optional[str]): hash of the blockchain block number at which to perform the query.
            reuse_block (bool): whether to reuse the last-used blockchain hash when retrieving info.

        Returns:
            The filtered list of netuids (union of filtered all_netuids and registered hotkeys).
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        netuids_with_registered_hotkeys = [
            item
            for sublist in await asyncio.gather(
                *[
                    self.get_netuids_for_hotkey(
                        wallet.hotkey.ss58_address,
                        reuse_block=reuse_block,
                        block_hash=block_hash,
                    )
                    for wallet in all_hotkeys
                ]
            )
            for item in sublist
        ]

        if not filter_for_netuids:
            all_netuids = netuids_with_registered_hotkeys

        else:
            filtered_netuids = [
                netuid for netuid in all_netuids if netuid in filter_for_netuids
            ]

            registered_hotkeys_filtered = [
                netuid
                for netuid in netuids_with_registered_hotkeys
                if netuid in filter_for_netuids
            ]

            # Combine both filtered lists
            all_netuids = filtered_netuids + registered_hotkeys_filtered

        return list(set(all_netuids))

    async def immunity_period(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """
        Retrieves the 'ImmunityPeriod' hyperparameter for a specific subnet. This parameter defines the duration during
        which new neurons are protected from certain network penalties or restrictions.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The blockchain block_hash representation of the block id.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            The value of the 'ImmunityPeriod' hyperparameter if the subnet exists, `None` otherwise.

        The 'ImmunityPeriod' is a critical aspect of the network's governance system, ensuring that new participants
        have a grace period to establish themselves and contribute to the network without facing immediate punitive
        actions.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        call = await self.get_hyperparameter(
            param_name="ImmunityPeriod",
            netuid=netuid,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        return None if call is None else int(call)

    async def is_in_admin_freeze_window(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> bool:
        """
        Returns True if the current block is within the terminal freeze window of the tempo
        for the given subnet. During this window, admin ops are prohibited to avoid interference
        with validator weight submissions.

        Parameters:
            netuid (int): The unique identifier of the subnet.
            block (Optional[int]): The blockchain block number for the query.
            block_hash: The blockchain block_hash representation of the block id.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            bool: True if in freeze window, else False.
        """
        # SN0 doesn't have admin_freeze_window
        if netuid == 0:
            return False

        next_epoch_start_block, window = await asyncio.gather(
            self.get_next_epoch_start_block(
                netuid=netuid,
                block=block,
                block_hash=block_hash,
                reuse_block=reuse_block,
            ),
            self.get_admin_freeze_window(
                block=block, block_hash=block_hash, reuse_block=reuse_block
            ),
        )

        if next_epoch_start_block is not None:
            remaining = next_epoch_start_block - await self.block
            return remaining < window
        return False

    async def is_fast_blocks(self) -> bool:
        """Checks if the node is running with fast blocks enabled.

        Fast blocks have a block time of 10 seconds, compared to the standard 12-second block time. This affects
        transaction timing and network synchronization.

        Returns:
            `True` if fast blocks are enabled (10-second block time), `False` otherwise (12-second block time).

        Notes:
            - <https://docs.learnbittensor.org/resources/glossary#fast-blocks>

        """
        return (
            await self.query_constant("SubtensorModule", "DurationOfStartCall")
        ) == 10

    async def is_hotkey_delegate(
        self,
        hotkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> bool:
        """
        Determines whether a given hotkey (public key) is a delegate on the Bittensor network. This function checks if
        the neuron associated with the hotkey is part of the network's delegation system.

        Parameters:
            hotkey_ss58: The SS58 address of the neuron's hotkey.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number for the query.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            `True` if the hotkey is a delegate, `False` otherwise.

        Being a delegate is a significant status within the Bittensor network, indicating a neuron's involvement in
        consensus and governance processes.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        delegates = await self.get_delegates(
            block_hash=block_hash, reuse_block=reuse_block
        )
        return hotkey_ss58 in [info.hotkey_ss58 for info in delegates]

    async def is_hotkey_registered(
        self,
        hotkey_ss58: str,
        netuid: Optional[int] = None,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> bool:
        """
        Determines whether a given hotkey (public key) is registered in the Bittensor network, either globally across
        any subnet or specifically on a specified subnet. This function checks the registration status of a neuron
        identified by its hotkey, which is crucial for validating its participation and activities within the network.

        Parameters:
            hotkey_ss58: The SS58 address of the neuron's hotkey.
            netuid: The unique identifier of the subnet to check the registration.
            block: The block number for which the children are to be retrieved.
            block_hash: The hash of the block to retrieve the subnet unique identifiers from.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            bool: `True` if the hotkey is registered in the specified context (either any subnet or a specific subnet),
                `False` otherwise.

        This function is important for verifying the active status of neurons in the Bittensor network. It aids in
        understanding whether a neuron is eligible to participate in network processes such as consensus, validation,
        and incentive distribution based on its registration status.
        """
        if netuid is None:
            return await self.is_hotkey_registered_any(
                hotkey_ss58, block, block_hash, reuse_block
            )
        else:
            return await self.is_hotkey_registered_on_subnet(
                hotkey_ss58, netuid, block, block_hash, reuse_block
            )

    async def is_hotkey_registered_any(
        self,
        hotkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> bool:
        """
        Checks if a neuron's hotkey is registered on any subnet within the Bittensor network.

        Parameters:
            hotkey_ss58: The `SS58` address of the neuron's hotkey.
            block: The blockchain block number for the query.
            block_hash: The blockchain block_hash representation of block id.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            bool: `True` if the hotkey is registered on any subnet, False otherwise.
        """
        hotkeys = await self.get_netuids_for_hotkey(
            hotkey_ss58, block, block_hash, reuse_block
        )
        return len(hotkeys) > 0

    async def is_hotkey_registered_on_subnet(
        self,
        hotkey_ss58: str,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> bool:
        """Checks if the hotkey is registered on a given subnet.

        Parameters:
            hotkey_ss58: The SS58 address of the hotkey to check.
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query. Do not specify if using `block_hash` or
                `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            `True` if the hotkey is registered on the specified subnet, `False` otherwise.

        Notes:
            - <https://docs.learnbittensor.org/glossary#hotkey>

        """
        return (
            await self.get_uid_for_hotkey_on_subnet(
                hotkey_ss58,
                netuid,
                block=block,
                block_hash=block_hash,
                reuse_block=reuse_block,
            )
            is not None
        )

    async def is_subnet_active(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> bool:
        """Verifies if a subnet with the provided netuid is active.

        A subnet is considered active if the `start_call` extrinsic has been executed. A newly registered subnet
        may exist but not be active until the subnet owner calls `start_call` to begin emissions.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query. Do not specify if using `block_hash` or
                `reuse_block`.
            block_hash: The block hash at which to check the parameter. Do not set if using `block` or
                `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            `True` if the subnet is active (emissions have started), `False` otherwise.

        Notes:
            - <https://docs.learnbittensor.org/subnets/working-with-subnets>

        """
        query = await self.query_subtensor(
            name="FirstEmissionBlockNumber",
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
            params=[netuid],
        )
        return True if query and query.value > 0 else False

    async def last_drand_round(self) -> Optional[int]:
        """Retrieves the last drand round emitted in Bittensor.

        Drand (distributed randomness) rounds are used to determine when committed weights can be revealed in the
        commit-reveal mechanism. This method returns the most recent drand round number, which corresponds to the
        timing for weight reveals.

        Returns:
            The latest drand round number emitted in Bittensor, or `None` if no round has been stored.

        Notes:
            - <https://docs.learnbittensor.org/resources/glossary#drandtime-lock-encryption>

        """
        result = await self.substrate.query(
            module="Drand", storage_function="LastStoredRound"
        )
        return getattr(result, "value", None)

    async def max_weight_limit(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[float]:
        """Returns the MaxWeightsLimit hyperparameter for a subnet.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            block: The blockchain block number for the query.
            block_hash: The hash of the block at which to query. Do not set if using `block` or `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            The stored maximum weight limit as a normalized float in [0, 1], or `None` if the subnetwork
                does not exist. Note: this value is not actually enforced - the weight validation code uses
                a hardcoded u16::MAX instead.

        Notes:
            - This hyperparameter is now a constant rather than a settable variable.
            - <https://docs.learnbittensor.org/subnets/subnet-hyperparameters>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        call = await self.get_hyperparameter(
            param_name="MaxWeightsLimit",
            netuid=netuid,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        return None if call is None else u16_normalized_float(int(call))

    async def metagraph(
        self,
        netuid: int,
        mechid: int = 0,
        lite: bool = True,
        block: Optional[int] = None,
    ) -> "AsyncMetagraph":
        """
        Returns a synced metagraph for a specified subnet within the Bittensor network.
        The metagraph represents the network's structure, including neuron connections and interactions.

        Parameters:
            netuid: The network UID of the subnet to query.
            mechid: Subnet mechanism identifier.
            lite: If `True`, returns a metagraph using a lightweight sync (no weights, no bonds).
            block: Block number for synchronization, or `None` for the latest block.

        Returns:
            The metagraph representing the subnet's structure and neuron relationships.

        The metagraph is an essential tool for understanding the topology and dynamics of the Bittensor network's
        decentralized architecture, particularly in relation to neuron interconnectivity and consensus processes.
        """
        metagraph = AsyncMetagraph(
            netuid=netuid,
            mechid=mechid,
            network=self.chain_endpoint,
            lite=lite,
            sync=False,
            subtensor=self,
        )
        await metagraph.sync(block=block, lite=lite, subtensor=self)

        return metagraph

    async def min_allowed_weights(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """Returns the MinAllowedWeights hyperparameter for a subnet.

        This hyperparameter sets the minimum length of the weights vector that a validator must submit.
        It checks `weights.len() >= MinAllowedWeights`. For example, a validator could submit `[1000, 0, 0, 0]`
        to satisfy `MinAllowedWeights=4`, but this would fail if `MinAllowedWeights` were set to 5.
        This ensures validators distribute attention across the subnet.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            block: The blockchain block number for the query.
            block_hash: The hash of the block at which to query. Do not set if using `block` or `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            The minimum number of required weight connections, or `None` if the subnetwork does not
                exist or the parameter is not found.

        Notes:
            - <https://docs.learnbittensor.org/subnets/subnet-hyperparameters>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        call = await self.get_hyperparameter(
            param_name="MinAllowedWeights",
            netuid=netuid,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        return None if call is None else int(call)

    async def neuron_for_uid(
        self,
        uid: Optional[int],
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> NeuronInfo:
        """
        Retrieves detailed information about a specific neuron identified by its unique identifier (UID) within a
        specified subnet (netuid) of the Bittensor network. This function provides a comprehensive view of a neuron's
        attributes, including its stake, rank, and operational status.

        Parameters:
            uid: The unique identifier of the neuron.
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number for the query.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            Detailed information about the neuron if found, a null neuron otherwise

        This function is crucial for analyzing individual neurons' contributions and status within a specific subnet,
        offering insights into their roles in the network's consensus and validation mechanisms.
        """
        if uid is None:
            return NeuronInfo.get_null_neuron()

        result = await self.query_runtime_api(
            runtime_api="NeuronInfoRuntimeApi",
            method="get_neuron",
            params=[netuid, uid],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )

        if not result:
            return NeuronInfo.get_null_neuron()

        return NeuronInfo.from_dict(result)

    async def neurons(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[NeuronInfo]:
        """
        Retrieves a list of all neurons within a specified subnet of the Bittensor network.
        This function provides a snapshot of the subnet's neuron population, including each neuron's attributes and
        network interactions.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number for the query.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            A list of NeuronInfo objects detailing each neuron's characteristics in the subnet.

        Understanding the distribution and status of neurons within a subnet is key to comprehending the network's
        decentralized structure and the dynamics of its consensus and governance processes.
        """
        result = await self.query_runtime_api(
            runtime_api="NeuronInfoRuntimeApi",
            method="get_neurons",
            params=[netuid],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )

        if not result:
            return []

        return NeuronInfo.list_from_dicts(result)

    async def neurons_lite(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[NeuronInfoLite]:
        """
        Retrieves a list of neurons in a 'lite' format from a specific subnet of the Bittensor network.
        This function provides a streamlined view of the neurons, focusing on key attributes such as stake and network
        participation.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number for the query.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            A list of simplified neuron information for the subnet.

        This function offers a quick overview of the neuron population within a subnet, facilitating efficient analysis
        of the network's decentralized structure and neuron dynamics.
        """
        result = await self.query_runtime_api(
            runtime_api="NeuronInfoRuntimeApi",
            method="get_neurons_lite",
            params=[netuid],
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )

        if not result:
            return []

        return NeuronInfoLite.list_from_dicts(result)

    async def query_identity(
        self,
        coldkey_ss58: str,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[ChainIdentity]:
        """
        Queries the identity of a neuron on the Bittensor blockchain using the given key. This function retrieves
        detailed identity information about a specific neuron, which is a crucial aspect of the network's decentralized
        identity and governance system.

        Parameters:
            coldkey_ss58: Coldkey used to query the neuron's identity (technically the neuron's coldkey SS58 address).
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number at which to perform the query.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            An object containing the identity information of the neuron if found, `None` otherwise.

        The identity information can include various attributes such as the neuron's stake, rank, and other
        network-specific details, providing insights into the neuron's role and status within the Bittensor network.

        Note:
            See the `Bittensor CLI documentation <https://docs.bittensor.com/reference/btcli>`_ for supported identity
            parameters.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        identity_info = cast(
            dict,
            await self.substrate.query(
                module="SubtensorModule",
                storage_function="IdentitiesV2",
                params=[coldkey_ss58],
                block_hash=block_hash,
                reuse_block_hash=reuse_block,
            ),
        )

        if not identity_info:
            return None

        try:
            return ChainIdentity.from_dict(
                decode_hex_identity_dict(identity_info),
            )
        except TypeError:
            return None

    async def recycle(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[Balance]:
        """Retrieves the 'Burn' hyperparameter for a specified subnet.

        The 'Burn' parameter represents the amount of TAO that is recycled when registering a neuron
        on this subnet. Recycled tokens are removed from circulation but can be re-emitted, unlike
        burned tokens which are permanently removed.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The hash of the block at which to query. Do not set if using `block` or `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            The amount of TAO recycled per neuron registration, or `None` if the subnet does not exist.

        Notes:
            - <https://docs.learnbittensor.org/resources/glossary#recycling-and-burning>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        call = await self.get_hyperparameter(
            param_name="Burn",
            netuid=netuid,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        return None if call is None else Balance.from_rao(int(call))

    async def subnet(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[DynamicInfo]:
        """
        Retrieves the subnet information for a single subnet in the Bittensor network.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The block number to get the subnets at.
            block_hash: The hash of the blockchain block number for the query.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            A DynamicInfo object, containing detailed information about a subnet.
        """
        block_hash = await self.determine_block_hash(
            block=block, block_hash=block_hash, reuse_block=reuse_block
        )

        if not block_hash and reuse_block:
            block_hash = self.substrate.last_block_hash

        query = await self.substrate.runtime_call(
            "SubnetInfoRuntimeApi",
            "get_dynamic_info",
            params=[netuid],
            block_hash=block_hash,
        )
        price = await self.get_subnet_price(
            netuid=netuid,
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )

        if isinstance(decoded := query.decode(), dict):
            if isinstance(price, (SubstrateRequestException, ValueError)):
                price = None
            return DynamicInfo.from_dict({**decoded, "price": price})
        return None

    async def subnet_exists(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> bool:
        """
        Checks if a subnet with the specified unique identifier (netuid) exists within the Bittensor network.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number at which to check the subnet existence.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            `True` if the subnet exists, `False` otherwise.

        This function is critical for verifying the presence of specific subnets in the network, enabling a deeper
        understanding of the network's structure and composition.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        result = await self.substrate.query(
            module="SubtensorModule",
            storage_function="NetworksAdded",
            params=[netuid],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        return getattr(result, "value", False)

    async def subnetwork_n(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """Returns the current number of registered neurons (UIDs) in a subnet.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            block: The blockchain block number for the query.
            block_hash: The hash of the block at which to query. Do not set if using `block` or `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            The current number of registered neurons in the subnet, or `None` if the subnetwork does not exist.

        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        call = await self.get_hyperparameter(
            param_name="SubnetworkN",
            netuid=netuid,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        return None if call is None else int(call)

    async def tempo(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """Returns the Tempo hyperparameter for a subnet.

        Tempo determines the length of an epoch in blocks. It defines how frequently the subnet's consensus mechanism
        runs, calculating emissions and updating rankings. A tempo of 360 blocks equals approximately 72 minutes
        (360 blocks × 12 seconds per block).

        Parameters:
            netuid: The unique identifier of the subnetwork.
            block: The blockchain block number for the query.
            block_hash: The hash of the block at which to query. Do not set if using `block` or `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            The tempo value in blocks, or `None` if the subnetwork does not exist.

        Notes:
            - <https://docs.learnbittensor.org/resources/glossary#tempo>
            - <https://docs.learnbittensor.org/resources/glossary#epoch>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        call = await self.get_hyperparameter(
            param_name="Tempo",
            netuid=netuid,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        return None if call is None else int(call)

    async def tx_rate_limit(
        self,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """
        Retrieves the transaction rate limit for the Bittensor network as of a specific blockchain block.
        This rate limit sets the maximum number of transactions that can be processed within a given time frame.

        Parameters:
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number at which to check the subnet existence.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            The transaction rate limit of the network, `None` if not available.

        The transaction rate limit is an essential parameter for ensuring the stability and scalability of the Bittensor
        network. It helps in managing network load and preventing congestion, thereby maintaining efficient and timely
        transaction processing.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        result = await self.query_subtensor(
            "TxRateLimit", block_hash=block_hash, reuse_block=reuse_block
        )
        return getattr(result, "value", None)

    async def wait_for_block(self, block: Optional[int] = None) -> bool:
        """
        Waits until a specific block is reached on the chain. If no block is specified, waits for the next block.

        Parameters:
            block: The block number to wait for. If `None`, waits for the next block.

        Returns:
            `True` if the target block was reached, `False` if timeout occurred.

        Example:

            # Waits for a specific block

            await subtensor.wait_for_block(block=1234)
        """

        async def handler(block_data: dict):
            logging.debug(
                f"reached block {block_data['header']['number']}. Waiting for block {target_block}"
            )
            if block_data["header"]["number"] >= target_block:
                return True
            return None

        current_block = await self.substrate.get_block()
        current_block_hash = current_block.get("header", {}).get("hash")

        if block is not None:
            target_block = block
        else:
            target_block = current_block["header"]["number"] + 1

        await self.substrate.get_block_handler(
            current_block_hash, header_only=True, subscription_handler=handler
        )
        return True

    async def weights(
        self,
        netuid: int,
        mechid: int = 0,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[tuple[int, list[tuple[int, int]]]]:
        """
        Retrieves the weight distribution set by neurons within a specific subnet of the Bittensor network.
        This function maps each neuron's UID to the weights it assigns to other neurons, reflecting the network's trust
        and value assignment mechanisms.

        Parameters:
            netuid: Subnet unique identifier.
            mechid: Subnet mechanism unique identifier.
            block: The blockchain block number for the query.
            block_hash: The blockchain block_hash representation of the block id.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            A list of tuples mapping each neuron's UID to its assigned weights.

        The weight distribution is a key factor in the network's consensus algorithm and the ranking of neurons,
        influencing their influence and reward allocation within the subnet.
        """
        storage_index = get_mechid_storage_index(netuid, mechid)
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        w_map_encoded = await self.substrate.query_map(
            module="SubtensorModule",
            storage_function="Weights",
            params=[storage_index],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        w_map = []
        async for uid, w in w_map_encoded:
            w_map.append((uid, w.value))

        return w_map

    async def weights_rate_limit(
        self,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[int]:
        """Returns the WeightsSetRateLimit hyperparameter for a subnet.

        This hyperparameter limits how many times a validator can set weights per epoch. It prevents validators
        from spamming weight updates and ensures stable consensus calculations. Once the limit is reached, validators
        must wait until the next epoch to set weights again.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            block: The blockchain block number for the query.
            block_hash: The hash of the block at which to query. Do not set if using `block` or `reuse_block`.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using `block_hash` or `block`.

        Returns:
            The maximum number of weight set operations allowed per epoch, or `None` if the subnetwork does not
                exist or the parameter is not found.
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        call = await self.get_hyperparameter(
            param_name="WeightsSetRateLimit",
            netuid=netuid,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        return None if call is None else int(call)

    # Extrinsics helpers ===============================================================================================
    async def validate_extrinsic_params(
        self,
        call_module: str,
        call_function: str,
        call_params: dict[str, Any],
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ):
        # TODO: Examples: validate against metadata;
        """
        Validate and filter extrinsic parameters against on-chain metadata.

        This method checks that the provided parameters match the expected signature of the given extrinsic (module and
        function) as defined in the Substrate metadata. It raises explicit errors for missing or invalid parameters and
        silently ignores any extra keys not present in the function definition.

        Parameters:
            call_module: The pallet name, e.g. "SubtensorModule" or "AdminUtils".
            call_function: The extrinsic function name, e.g. "set_weights" or "sudo_set_tempo".
            call_params: A dictionary of parameters to validate.
            block: The blockchain block number for the query.
            block_hash: The blockchain block_hash representation of the block id.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            A filtered dictionary containing only the parameters that are valid for the specified extrinsic.

        Raises:
            ValueError: If the given module or function is not found in the chain metadata.
            KeyError: If one or more required parameters are missing.

        Notes:
            This method does not compose or submit the extrinsic. It only ensures that `call_params` conforms to the
            expected schema derived from on-chain metadata.
            See also `compose_call` and `sign_and_send_extrinsic`.

        """
        block_hash = await self.determine_block_hash(
            block=block, block_hash=block_hash, reuse_block=reuse_block
        )

        func_meta = await self.substrate.get_metadata_call_function(
            module_name=call_module,
            call_function_name=call_function,
            block_hash=block_hash,
        )

        if not func_meta:
            raise ValueError(
                f"Call {call_module}.{call_function} not found in chain metadata."
            )

        # Expected params from metadata
        expected_params = func_meta.get_param_info()
        provided_params = {}

        # Validate and filter parameters
        for param_name in expected_params.keys():
            if param_name not in call_params:
                raise KeyError(f"Missing required parameter: '{param_name}'")
            provided_params[param_name] = call_params[param_name]

        # Warn about extra params not defined in metadata
        extra_params = set(call_params.keys()) - set(expected_params.keys())
        if extra_params:
            logging.debug(
                f"Ignoring extra parameters for {call_module}.{call_function}: {extra_params}."
            )
        return provided_params

    async def compose_call(
        self,
        call_module: str,
        call_function: str,
        call_params: dict[str, Any],
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> "GenericCall":
        """Dynamically compose a GenericCall using on-chain Substrate metadata after validating the provided parameters.

        Parameters:
            call_module: Pallet name (e.g. "SubtensorModule", "AdminUtils").
            call_function: Function name (e.g. "set_weights", "sudo_set_tempo").
            call_params: Dictionary of parameters for the call.
            block: The blockchain block number for the query.
            block_hash: The blockchain block_hash representation of the block id.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            GenericCall: Composed call object ready for extrinsic submission.

        # TODO: document whole extrinsic flow
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)

        call_params = await self.validate_extrinsic_params(
            call_module=call_module,
            call_function=call_function,
            call_params=call_params,
            block=block,
            block_hash=block_hash,
            reuse_block=reuse_block,
        )

        logging.debug(
            f"Composing GenericCall -> {call_module}.{call_function} "
            f"with params: {call_params}."
        )
        return await self.substrate.compose_call(
            call_module=call_module,
            call_function=call_function,
            call_params=call_params,
            block_hash=block_hash,
        )

    async def sign_and_send_extrinsic(
        self,
        call: "GenericCall",
        wallet: "Wallet",
        sign_with: str = "coldkey",
        use_nonce: bool = False,
        nonce_key: str = "hotkey",
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        calling_function: Optional[str] = None,
    ) -> ExtrinsicResponse:
        # TODO: Full clear example of sending extrinsic flow
        """
        Helper method to sign and submit an extrinsic call to chain.

        Parameters:
            call: a prepared Call object
            wallet: the wallet whose coldkey will be used to sign the extrinsic
            sign_with: the wallet's keypair to use for the signing. Options are "coldkey", "hotkey", "coldkeypub"
            use_nonce: unique identifier for the transaction related with hot/coldkey.
            nonce_key: the type on nonce to use. Options are "hotkey" or "coldkey".
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: raises the relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: whether to wait until the extrinsic call is included on the chain
            wait_for_finalization: whether to wait until the extrinsic call is finalized on the chain
            calling_function: the name of the calling function.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Raises:
            SubstrateRequestException: Substrate request exception.
        """
        extrinsic_response = ExtrinsicResponse(
            extrinsic_function=calling_function
            if calling_function
            else get_caller_name()
        )
        possible_keys = ("coldkey", "hotkey", "coldkeypub")
        if sign_with not in possible_keys:
            raise AttributeError(
                f"'sign_with' must be either 'coldkey', 'hotkey' or 'coldkeypub', not '{sign_with}'"
            )
        signing_keypair = getattr(wallet, sign_with)
        extrinsic_data = {"call": call, "keypair": signing_keypair}
        if use_nonce:
            if nonce_key not in possible_keys:
                raise AttributeError(
                    f"'nonce_key' must be either 'coldkey', 'hotkey' or 'coldkeypub', not '{nonce_key}'"
                )
            next_nonce = await self.substrate.get_account_next_index(
                getattr(wallet, nonce_key).ss58_address
            )
            extrinsic_data["nonce"] = next_nonce

        if period is not None:
            extrinsic_data["era"] = {"period": period}

        extrinsic_response.extrinsic = await self.substrate.create_signed_extrinsic(
            **extrinsic_data
        )
        try:
            response = await self.substrate.submit_extrinsic(
                extrinsic=extrinsic_response.extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

            # We only wait here if we expect finalization.
            if not wait_for_finalization and not wait_for_inclusion:
                extrinsic_response.extrinsic_fee = await self.get_extrinsic_fee(
                    call=call, keypair=signing_keypair
                )
                extrinsic_response.message = (
                    "Not waiting for finalization or inclusion."
                )
                logging.debug(extrinsic_response.message)
                return extrinsic_response

            extrinsic_response.extrinsic_receipt = response

            if await response.is_success:
                extrinsic_response.extrinsic_fee = Balance.from_rao(
                    await response.total_fee_amount
                )
                extrinsic_response.message = "Success"
                return extrinsic_response

            response_error_message = await response.error_message

            if raise_error:
                raise ChainError.from_error(response_error_message)

            extrinsic_response.success = False
            extrinsic_response.message = format_error_message(response_error_message)
            extrinsic_response.error = response_error_message
            return extrinsic_response

        except SubstrateRequestException as error:
            if raise_error:
                raise

            extrinsic_response.success = False
            extrinsic_response.message = format_error_message(error)
            extrinsic_response.error = error
            return extrinsic_response

    async def get_extrinsic_fee(
        self,
        call: "GenericCall",
        keypair: "Keypair",
    ) -> Balance:
        """Gets the extrinsic fee for a given extrinsic call and keypair.

        This method estimates the transaction fee that will be charged for submitting the extrinsic to the
        blockchain. The fee is returned in Rao (the smallest unit of TAO, where 1 TAO = 1e9 Rao).

        Parameters:
            call: The extrinsic GenericCall object representing the transaction to estimate.
            keypair: The keypair associated with the extrinsic (used to determine the account paying the fee).

        Returns:
            Balance object representing the extrinsic fee in Rao.

        Example:

            # Estimate fee before sending a transfer

            call = await subtensor.compose_call(
                call_module="Balances",
                call_function="transfer",
                call_params={"dest": destination_ss58, "value": amount.rao}
            )
            fee = await subtensor.get_extrinsic_fee(call=call, keypair=wallet.coldkey)
            print(f"Estimated fee: {fee.tao} TAO")

        Notes:
            To create the GenericCall object, use the `compose_call` method with proper parameters.
            - <https://docs.learnbittensor.org/learn/fees>

        """
        payment_info = await self.substrate.get_payment_info(call=call, keypair=keypair)
        return Balance.from_rao(amount=payment_info["partial_fee"])

    # Extrinsics =======================================================================================================

    async def add_stake(
        self,
        wallet: "Wallet",
        netuid: int,
        hotkey_ss58: str,
        amount: Balance,
        safe_staking: bool = False,
        allow_partial_stake: bool = False,
        rate_tolerance: float = 0.005,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """
        Adds a stake from the specified wallet to the neuron identified by the SS58 address of its hotkey in specified
        subnet. Staking is a fundamental process in the Bittensor network that enables neurons to participate actively
        and earn incentives.

        Parameters:
            wallet: The wallet to be used for staking.
            netuid: The unique identifier of the subnet to which the neuron belongs.
            hotkey_ss58: The `ss58` address of the hotkey account to stake to default to the wallet's hotkey.
            amount: The amount of TAO to stake.
            safe_staking: If `True`, enables price safety checks to protect against fluctuating prices. The stake will
                only execute if the price change doesn't exceed the rate tolerance.
            allow_partial_stake: If `True` and safe_staking is enabled, allows partial staking when the full amount would
                exceed the price tolerance. If false, the entire stake fails if it would exceed the tolerance.
            rate_tolerance: The maximum allowed price change ratio when staking. For example, 0.005 = 0.5% maximum price
                increase. Only used when safe_staking is True.
            mev_protection: If `True`, encrypts and submits the staking transaction through the MEV Shield pallet to
                protect against front-running and MEV attacks. The transaction remains encrypted in the mempool until
                validators decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        This function enables neurons to increase their stake in the network, enhancing their influence and potential.
        When safe_staking is enabled, it provides protection against price fluctuations during the time stake is
        executed and the time it is actually processed by the chain.

        Notes:
            - Price Protection: <https://docs.learnbittensor.org/learn/price-protection>
            - Rate Limits: <https://docs.learnbittensor.org/learn/chain-rate-limits#staking-operations-rate-limits>
        """
        check_balance_amount(amount)
        return await add_stake_extrinsic(
            subtensor=self,
            wallet=wallet,
            hotkey_ss58=hotkey_ss58,
            netuid=netuid,
            amount=amount,
            safe_staking=safe_staking,
            allow_partial_stake=allow_partial_stake,
            rate_tolerance=rate_tolerance,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def add_liquidity(
        self,
        wallet: "Wallet",
        netuid: int,
        liquidity: Balance,
        price_low: Balance,
        price_high: Balance,
        hotkey_ss58: Optional[str] = None,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """
        Adds liquidity to the specified price range.

        Parameters:
            wallet: The wallet used to sign the extrinsic (must be unlocked).
            netuid: The UID of the target subnet for which the call is being initiated.
            liquidity: The amount of liquidity to be added.
            price_low: The lower bound of the price tick range. In TAO.
            price_high: The upper bound of the price tick range. In TAO.
            hotkey_ss58: The hotkey with staked TAO in Alpha. If not passed then the wallet hotkey is used.
            mev_protection:` If` True, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Note:
            Adding is allowed even when user liquidity is enabled in specified subnet. Call `toggle_user_liquidity`
            method to enable/disable user liquidity.
        """
        return await add_liquidity_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            liquidity=liquidity,
            price_low=price_low,
            price_high=price_high,
            hotkey_ss58=hotkey_ss58,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def add_stake_multiple(
        self,
        wallet: "Wallet",
        netuids: UIDs,
        hotkey_ss58s: list[str],
        amounts: list[Balance],
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """
        Adds stakes to multiple neurons identified by their hotkey SS58 addresses.
        This bulk operation allows for efficient staking across different neurons from a single wallet.

        Parameters:
            wallet: The wallet used for staking.
            netuids: List of subnet UIDs.
            hotkey_ss58s: List of `SS58` addresses of hotkeys to stake to.
            amounts: List of corresponding TAO amounts to bet for each netuid and hotkey.
            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Notes:
            - Price Protection: <https://docs.learnbittensor.org/learn/price-protection>
            - Rate Limits: <https://docs.learnbittensor.org/learn/chain-rate-limits#staking-operations-rate-limits>
        """
        return await add_stake_multiple_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuids=netuids,
            hotkey_ss58s=hotkey_ss58s,
            amounts=amounts,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def add_proxy(
        self,
        wallet: "Wallet",
        delegate_ss58: str,
        proxy_type: Union[str, "ProxyType"],
        delay: int,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """
        Adds a proxy relationship.

        This method creates a proxy relationship where the delegate can execute calls on behalf of the real account (the
        wallet owner) with restrictions defined by the proxy type and a delay period. A deposit is required and held as
        long as the proxy relationship exists.

        Parameters:
            wallet: Bittensor wallet object.
            delegate_ss58: The SS58 address of the delegate proxy account.
            proxy_type: The type of proxy permissions (e.g., "Any", "NonTransfer", "Governance", "Staking"). Can be a
                string or ProxyType enum value. For available proxy types and their permissions, see the documentation
                link in the Notes section below.
            delay: Optionally, include a delay in blocks. The number of blocks that must elapse between announcing and
                executing a proxied transaction. A delay of `0` means the proxy can be used immediately without
                announcements. A non-zero delay creates a time-lock, requiring the proxy to announce calls via
                :meth:`announce_proxy` before execution, giving the real account time to review and reject unwanted
                operations via :meth:`reject_proxy_announcement`.
            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Notes:
            - A deposit is required when adding a proxy. The deposit amount is determined by runtime constants and is
              returned when the proxy is removed. Use :meth:`get_proxy_constants` to check current deposit requirements.
            - For available proxy types and their specific permissions, see: <https://docs.learnbittensor.org/keys/proxies#types-of-proxies>
            - Bittensor proxies: <https://docs.learnbittensor.org/keys/proxies/create-proxy>

        Warning:
            The `Any` proxy type is dangerous as it grants full permissions to the proxy, including the ability to make
            transfers and manage the account completely. Use with extreme caution.
        """
        return await add_proxy_extrinsic(
            subtensor=self,
            wallet=wallet,
            delegate_ss58=delegate_ss58,
            proxy_type=proxy_type,
            delay=delay,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def announce_proxy(
        self,
        wallet: "Wallet",
        real_account_ss58: str,
        call_hash: str,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """
        Announces a future call that will be executed through a proxy.

        This method allows a proxy account to declare its intention to execute a specific call on behalf of a real
        account after a delay period. The real account can review and either approve (via :meth:`proxy_announced`) or reject
        (via :meth:`reject_proxy_announcement`) the announcement.

        Parameters:
            wallet: Bittensor wallet object (should be the proxy account wallet).
            real_account_ss58: The SS58 address of the real account on whose behalf the call will be made.
            call_hash: The hash of the call that will be executed in the future.
            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Notes:
            - A deposit is required when making an announcement. The deposit is returned when the announcement is
              executed, rejected, or removed. The announcement can be executed after the delay period has passed.
            - Bittensor proxies: <https://docs.learnbittensor.org/keys/proxies>
        """
        return await announce_extrinsic(
            subtensor=self,
            wallet=wallet,
            real_account_ss58=real_account_ss58,
            call_hash=call_hash,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def burned_register(
        self,
        wallet: "Wallet",
        netuid: int,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """
        Registers a neuron on the Bittensor network by recycling TAO. This method of registration involves recycling
        TAO tokens, allowing them to be re-mined by performing work on the network.

        Parameters:
            wallet: The wallet associated with the neuron to be registered.
            netuid: The unique identifier of the subnet.
            mev_protection: If True, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Notes:
            - Rate Limits: <https://docs.learnbittensor.org/learn/chain-rate-limits#registration-rate-limits>
        """
        async with self:
            if netuid == 0:
                return await root_register_extrinsic(
                    subtensor=self,
                    wallet=wallet,
                    mev_protection=mev_protection,
                    period=period,
                    raise_error=raise_error,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                    wait_for_revealed_execution=wait_for_revealed_execution,
                )

            return await burned_register_extrinsic(
                subtensor=self,
                wallet=wallet,
                netuid=netuid,
                mev_protection=mev_protection,
                period=period,
                raise_error=raise_error,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                wait_for_revealed_execution=wait_for_revealed_execution,
            )

    async def claim_root(
        self,
        wallet: "Wallet",
        netuids: "UIDs",
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ):
        """Submit an extrinsic to manually claim accumulated root dividends from one or more subnets.

        Parameters:
            wallet: Bittensor `Wallet` instance.
            netuids: Iterable of subnet IDs to claim from in this call (the chain enforces a maximum number per
                transaction).
            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: Number of blocks during which the transaction remains valid after submission. If the extrinsic is
                not included in a block within this window, it will expire and be rejected.
            raise_error: Whether to raise a Python exception instead of returning a failed `ExtrinsicResponse`.
            wait_for_inclusion: Whether to wait until the extrinsic is included in a block before returning.
            wait_for_finalization: Whether to wait for finalization of the extrinsic in a block before returning.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            `ExtrinsicResponse` describing the result of the extrinsic execution.

        Notes:
            - Only Alpha dividends are claimed; the underlying TAO stake on the Root Subnet remains unchanged.
            - The current root claim type (`Swap` or `Keep`) determines whether claimed Alpha is converted to
              TAO and restaked on root or left as Alpha on the originating subnets.
            - See: <https://docs.learnbittensor.org/staking-and-delegation/root-claims>
            - See also: <https://docs.learnbittensor.org/staking-and-delegation/root-claims/managing-root-claims>
            - Transaction fees: <https://docs.learnbittensor.org/learn/fees>
        """
        return await claim_root_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuids=netuids,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def commit_weights(
        self,
        wallet: "Wallet",
        netuid: int,
        salt: Salt,
        uids: UIDs,
        weights: Weights,
        mechid: int = 0,
        version_key: int = version_as_int,
        max_attempts: int = 5,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = 16,
        raise_error: bool = True,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """
        Commits a hash of the subnet validator's weight vector to the Bittensor blockchain using the provided wallet.
        This action serves as a commitment or snapshot of the validator's current weight distribution.

        Parameters:
            wallet: The wallet associated with the neuron committing the weights.
            netuid: The unique identifier of the subnet.
            salt: list of randomly generated integers as salt to generated weighted hash.
            uids: NumPy array of neuron UIDs for which weights are being committed.
            weights: NumPy array of weight values corresponding to each UID.
            mechid: The subnet mechanism unique identifier.
            version_key: Version key for compatibility with the network.
            max_attempts: The number of maximum attempts to commit weights.
            mev_protection: `If` True, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        This function allows subnet validators to create a tamper-proof record of their weight vector at a specific
        point in time, creating a foundation of transparency and accountability for the Bittensor network.

        Notes:
            - <https://docs.learnbittensor.org/glossary#commit-reveal>
            - Rate Limits: <https://docs.learnbittensor.org/learn/chain-rate-limits#weights-setting-rate-limit>

        """
        attempt = 0
        response = ExtrinsicResponse(False)

        if attempt_check := validate_max_attempts(max_attempts, response):
            return attempt_check

        logging.debug(
            f"Committing weights with params: "
            f"netuid=[blue]{netuid}[/blue], uids=[blue]{uids}[/blue], weights=[blue]{weights}[/blue], "
            f"version_key=[blue]{version_key}[/blue]"
        )

        while attempt < max_attempts and response.success is False:
            try:
                response = await commit_weights_extrinsic(
                    subtensor=self,
                    wallet=wallet,
                    netuid=netuid,
                    mechid=mechid,
                    uids=uids,
                    weights=weights,
                    salt=salt,
                    mev_protection=mev_protection,
                    period=period,
                    raise_error=raise_error,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                    wait_for_revealed_execution=wait_for_revealed_execution,
                )
            except Exception as error:
                return ExtrinsicResponse.from_exception(
                    raise_error=raise_error, error=error
                )
            attempt += 1

        if not response.success:
            logging.debug(
                "No one successful attempt made. "
                "Perhaps it is too soon to commit weights!"
            )
        return response

    async def contribute_crowdloan(
        self,
        wallet: "Wallet",
        crowdloan_id: int,
        amount: "Balance",
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """Contributes TAO to an active crowdloan campaign.

        Contributions must occur before the crowdloan's end block and are subject to minimum contribution
        requirements. If a contribution would push the total raised above the cap, it is automatically clipped
        to fit the remaining amount. Once the cap is reached, further contributions are rejected.


        Parameters:
            wallet: Bittensor wallet instance used to sign the transaction (coldkey pays, coldkey receives emissions).
            crowdloan_id: The unique identifier of the crowdloan to contribute to.
            amount: Amount to contribute (TAO). Must meet or exceed the campaign's `min_contribution`.
            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.

            period: The number of blocks during which the transaction will remain valid after it's submitted.
            raise_error: If `True`, raises an exception rather than returning failure in the response.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            `ExtrinsicResponse` indicating success or failure, with error details if applicable.

        Notes:
            - Contributions can be withdrawn before finalization via `withdraw_crowdloan`.
            - If the campaign does not reach its cap by the end block, contributors can be refunded via `refund_crowdloan`.
            - Contributions are counted toward `MaxContributors` limit per crowdloan.

            - Crowdloans Overview: <https://docs.learnbittensor.org/subnets/crowdloans>
            - Crowdloan Tutorial: <https://docs.learnbittensor.org/subnets/crowdloans/crowdloans-tutorial#step-4-contribute-to-the-crowdloan>
        """
        return await contribute_crowdloan_extrinsic(
            subtensor=self,
            wallet=wallet,
            crowdloan_id=crowdloan_id,
            amount=amount,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def create_crowdloan(
        self,
        wallet: "Wallet",
        deposit: "Balance",
        min_contribution: "Balance",
        cap: "Balance",
        end: int,
        call: Optional["GenericCall"] = None,
        target_address: Optional[str] = None,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """
        Creates a new crowdloan campaign on-chain.

        Parameters:
            wallet: Bittensor Wallet instance used to sign the transaction.
            deposit: Initial deposit in RAO from the creator.
            min_contribution: Minimum contribution amount.
            cap: Maximum cap to be raised.
            end: Block number when the campaign ends.
            call: Runtime call data (e.g., subtensor::register_leased_network).
            target_address: SS58 address to transfer funds to on success.
            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            `ExtrinsicResponse` indicating success or failure. On success, the crowdloan ID can be extracted from the
            `Crowdloan.Created` event in the response.

        Notes:
            - Creator cannot update `call` or `target_address` after creation.
            - Creator can update `cap`, `end`, and `min_contribution` before finalization via `update_*` methods.
            - Use `get_crowdloan_next_id` to determine the ID that will be assigned to the new crowdloan.

            - Crowdloans Overview: <https://docs.learnbittensor.org/subnets/crowdloans>
            - Crowdloan Tutorial: <https://docs.learnbittensor.org/subnets/crowdloans/crowdloans-tutorial#step-3-create-a-crowdloan>
        """
        return await create_crowdloan_extrinsic(
            subtensor=self,
            wallet=wallet,
            deposit=deposit,
            min_contribution=min_contribution,
            cap=cap,
            end=end,
            call=call,
            target_address=target_address,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def create_pure_proxy(
        self,
        wallet: "Wallet",
        proxy_type: Union[str, "ProxyType"],
        delay: int,
        index: int,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """
        Creates a pure proxy account.

        A pure proxy is a keyless account that can only be controlled through proxy relationships. Unlike regular
        proxies, pure proxies do not have their own private keys, making them more secure for certain use cases. The
        pure proxy address is deterministically generated based on the spawner account, proxy type, delay, and index.

        Parameters:
            wallet: Bittensor wallet object.
            proxy_type: The type of proxy permissions for the pure proxy. Can be a string or ProxyType enum value. For
                available proxy types and their permissions, see the documentation link in the Notes section below.
            delay: Optionally, include a delay in blocks. The number of blocks that must elapse between announcing and
                executing a proxied transaction. A delay of `0` means the pure proxy can be used immediately without any
                announcement period. A non-zero delay creates a time-lock, requiring announcements before execution to give
                the spawner time to review/reject.
            index: A salt value (u16, range `0-65535`) used to generate unique pure proxy addresses. This should generally
                be left as `0` unless you are creating batches of proxies. When creating multiple pure proxies with
                identical parameters (same `proxy_type` and `delay`), different index values will produce different SS58
                addresses. This is not a sequential counter—you can use any unique values (e.g., 0, 100, 7, 42) in any
                order. The index must be preserved as it's required for :meth:`kill_pure_proxy`. If creating multiple pure
                proxies in a single batch transaction, each must have a unique index value.
            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Notes:
            - The pure proxy account address can be extracted from the "PureCreated" event in the response. Store the
              spawner address, proxy_type, index, height, and ext_index as they are required to kill the pure proxy later
              via :meth:`kill_pure_proxy`.
            - For available proxy types and their specific permissions, see: <https://docs.learnbittensor.org/keys/proxies#types-of-proxies>
            - Bittensor proxies: <https://docs.learnbittensor.org/keys/proxies/pure-proxies>

        Warning:
            The `Any` proxy type is dangerous as it grants full permissions to the proxy, including the ability to make
            transfers and kill the proxy. Use with extreme caution.
        """
        return await create_pure_proxy_extrinsic(
            subtensor=self,
            wallet=wallet,
            proxy_type=proxy_type,
            delay=delay,
            index=index,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def dissolve_crowdloan(
        self,
        wallet: "Wallet",
        crowdloan_id: int,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """Dissolves a failed or refunded crowdloan, cleaning up storage and returning the creator's deposit.

        This permanently removes the crowdloan from on-chain storage and returns the creator's deposit. Can only
        be called by the creator after all non-creator contributors have been refunded via `refund_crowdloan`.
        This is the final step in the lifecycle of a failed crowdloan (one that did not reach its cap by the end
        block).

        Parameters:
            wallet: Bittensor wallet instance used to sign the transaction (must be the creator's coldkey).
            crowdloan_id: The unique identifier of the crowdloan to dissolve.
            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after submission.
            raise_error: If `True`, raises an exception rather than returning failure in the response.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            `ExtrinsicResponse` indicating success or failure, with error details if applicable.

        Notes:
            - Only the creator can dissolve their own crowdloan.
            - All non-creator contributors must be refunded first via `refund_crowdloan`.
            - The creator's deposit (and any remaining contribution above deposit) is returned.
            - After dissolution, the crowdloan is permanently removed from chain storage.

            - Crowdloans Overview: <https://docs.learnbittensor.org/subnets/crowdloans>
            - Refund and Dissolve: <https://docs.learnbittensor.org/subnets/crowdloans/crowdloans-tutorial#alternative-path-refund-and-dissolve>
        """
        return await dissolve_crowdloan_extrinsic(
            subtensor=self,
            wallet=wallet,
            crowdloan_id=crowdloan_id,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def finalize_crowdloan(
        self,
        wallet: "Wallet",
        crowdloan_id: int,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """Finalizes a successful crowdloan after the cap is fully raised and the end block has passed.

        Finalization executes the stored call (e.g., `register_leased_network`) or transfers raised funds to
        the target address. For subnet lease crowdloans, this registers the subnet, creates a
        `SubnetLeaseBeneficiary` proxy for the creator, and records contributor shares for pro-rata emissions
        distribution. Leftover funds (after registration and proxy costs) are refunded to contributors.

        Only the creator can finalize, and finalization can only occur after both the end block is reached and
        the total raised equals the cap.

        Parameters:
            wallet: Bittensor wallet instance used to sign the transaction (must be the creator's coldkey).
            crowdloan_id: The unique identifier of the crowdloan to finalize.
            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after submission.
            raise_error: If `True`, raises an exception rather than returning failure in the response.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            `ExtrinsicResponse` indicating success or failure. On success, a subnet lease is created (if applicable)
            and contributor shares are recorded for emissions.

        Notes:
            - Only the creator can finalize.
            - Finalization requires `raised == cap` and `current_block >= end`.
            - For subnet leases, emissions are swapped to TAO and distributed to contributors' coldkeys during the lease.
            - Leftover cap (after subnet lock + proxy deposit) is refunded to contributors pro-rata.

            - Crowdloans Overview: <https://docs.learnbittensor.org/subnets/crowdloans>
            - Crowdloan Tutorial: <https://docs.learnbittensor.org/subnets/crowdloans/crowdloans-tutorial#step-5-finalize-the-crowdloan>
            - Emissions Distribution: <https://docs.learnbittensor.org/subnets/crowdloans#emissions-distribution-during-a-lease>
        """
        return await finalize_crowdloan_extrinsic(
            subtensor=self,
            wallet=wallet,
            crowdloan_id=crowdloan_id,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def kill_pure_proxy(
        self,
        wallet: "Wallet",
        pure_proxy_ss58: str,
        spawner: str,
        proxy_type: Union[str, "ProxyType"],
        index: int,
        height: int,
        ext_index: int,
        force_proxy_type: Optional[Union[str, "ProxyType"]] = ProxyType.Any,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """
        Kills (removes) a pure proxy account.

        This method removes a pure proxy account that was previously created via :meth:`create_pure_proxy`. The `kill_pure`
        call must be executed through the pure proxy account itself, with the spawner acting as an "Any" proxy. This
        method automatically handles this by executing the call via :meth:`proxy`.

        Parameters:
            wallet: Bittensor wallet object. The wallet.coldkey.ss58_address must be the spawner of the pure proxy (the
                account that created it via :meth:`create_pure_proxy`). The spawner must have an "Any" proxy relationship
                with the pure proxy.
            pure_proxy_ss58: The SS58 address of the pure proxy account to be killed. This is the address that was
                returned in the :meth:`create_pure_proxy` response.
            spawner: The SS58 address of the spawner account (the account that originally created the pure proxy via
                :meth:`create_pure_proxy`). This should match wallet.coldkey.ss58_address.
            proxy_type: The type of proxy permissions. Can be a string or ProxyType enum value. Must match the
                proxy_type used when creating the pure proxy.
            index: The salt value (u16, range `0-65535`) originally used in :meth:`create_pure_proxy` to generate this
                pure proxy's address. This value, combined with `proxy_type`, `delay`, and `spawner`, uniquely
                identifies the pure proxy to be killed. Must match exactly the index used during creation.
            height: The block number at which the pure proxy was created. This is returned in the "PureCreated" event from
                :meth:`create_pure_proxy` and is required to identify the exact creation transaction.
            ext_index: The extrinsic index within the block at which the pure proxy was created. This is returned in the
                "PureCreated" event from :meth:`create_pure_proxy` and specifies the position of the creation extrinsic
                within the block. Together with `height`, this uniquely identifies the creation transaction.
            force_proxy_type: The proxy type relationship to use when executing `kill_pure` through the proxy mechanism.
                Since pure proxies are keyless and cannot sign transactions, the spawner must act as a proxy for the
                pure proxy to execute `kill_pure`. This parameter specifies which proxy type relationship between the
                spawner and the pure proxy account should be used. The spawner must have a proxy relationship of this
                type (or `Any`) with the pure proxy account. Defaults to `ProxyType.Any` for maximum compatibility. If
                `None`, Substrate will automatically select an available proxy type from the spawner's proxy
                relationships.
            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Notes:
            - The `kill_pure` call must be executed through the pure proxy account itself, with the spawner acting as
              an `Any` proxy. This method automatically handles this by executing the call via :meth:`proxy`. The spawner
              must have an `Any` proxy relationship with the pure proxy for this to work.
            - Bittensor proxies: <https://docs.learnbittensor.org/keys/proxies/pure-proxies>

        Warning:
            All access to this account will be lost. Any funds remaining in the pure proxy account will become
            permanently inaccessible after this operation.
        """
        return await kill_pure_proxy_extrinsic(
            subtensor=self,
            wallet=wallet,
            pure_proxy_ss58=pure_proxy_ss58,
            spawner=spawner,
            proxy_type=proxy_type,
            index=index,
            height=height,
            ext_index=ext_index,
            force_proxy_type=force_proxy_type,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def mev_submit_encrypted(
        self,
        wallet: "Wallet",
        call: "GenericCall",
        signer_keypair: Optional["Keypair"] = None,
        *,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
        blocks_for_revealed_execution: int = 5,
    ) -> ExtrinsicResponse:
        """
        Submits an encrypted extrinsic to the MEV Shield pallet.

        This function encrypts a call using ML-KEM-768 + XChaCha20Poly1305 and submits it to the MevShield pallet. The
        extrinsic remains encrypted in the transaction pool until it is included in a block and decrypted by validators.

        Parameters:
            wallet: The wallet used to sign the extrinsic (must be unlocked, coldkey will be used for signing).
            call: The GenericCall object to encrypt and submit.
            signer_keypair: The keypair used to sign the inner call.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
                think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.
            wait_for_revealed_execution: Whether to wait for the DecryptedExecuted event, indicating that validators
                have successfully decrypted and executed the inner call. If True, the function will poll subsequent
                blocks for the event matching this submission's commitment.
            blocks_for_revealed_execution: Maximum number of blocks to poll for the DecryptedExecuted event after
                inclusion. The function checks blocks from start_block+1 to start_block + blocks_for_revealed_execution.
                Returns immediately if the event is found before the block limit is reached.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Raises:
            ValueError: If NextKey is not available in storage or encryption fails.
            SubstrateRequestException: If the extrinsic fails to be submitted or included.

        Note:
            The encryption uses the public key from NextKey storage, which rotates every block. The payload structure is:
            payload_core = signer_bytes (32B) + nonce (u32 LE, 4B) + SCALE(call)
            plaintext = payload_core + b"\\x01" + signature (64B for sr25519)
            commitment = blake2_256(payload_core)
        """
        return await submit_encrypted_extrinsic(
            subtensor=self,
            wallet=wallet,
            call=call,
            signer_keypair=signer_keypair,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
            blocks_for_revealed_execution=blocks_for_revealed_execution,
        )

    async def modify_liquidity(
        self,
        wallet: "Wallet",
        netuid: int,
        position_id: int,
        liquidity_delta: Balance,
        hotkey_ss58: Optional[str] = None,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """Modifies liquidity in liquidity position by adding or removing liquidity from it.

        Parameters:
            wallet: The wallet used to sign the extrinsic (must be unlocked).
            netuid: The UID of the target subnet for which the call is being initiated.
            position_id: The id of the position record in the pool.
            liquidity_delta: The amount of liquidity to be added or removed (add if positive or remove if negative).
            hotkey_ss58: The hotkey with staked TAO in Alpha. If not passed then the wallet hotkey is used.
            mev_protection:` If` True, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Example:

            import bittensor as bt

            subtensor = bt.AsyncSubtensor(network="local")
            await subtensor.initialize()

            my_wallet = bt.Wallet()

            # if `liquidity_delta` is negative

            my_liquidity_delta = Balance.from_tao(100) * -1
            await subtensor.modify_liquidity(
                wallet=my_wallet,
                netuid=123,
                position_id=2,
                liquidity_delta=my_liquidity_delta
            )

            # if `liquidity_delta` is positive

            my_liquidity_delta = Balance.from_tao(120)
            await subtensor.modify_liquidity(
                wallet=my_wallet,
                netuid=123,
                position_id=2,
                liquidity_delta=my_liquidity_delta
            )

        Note:
            Modifying is allowed even when user liquidity is enabled in specified subnet. Call `toggle_user_liquidity`
            to enable/disable user liquidity.
        """
        return await modify_liquidity_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            position_id=position_id,
            liquidity_delta=liquidity_delta,
            hotkey_ss58=hotkey_ss58,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def move_stake(
        self,
        wallet: "Wallet",
        origin_netuid: int,
        origin_hotkey_ss58: str,
        destination_netuid: int,
        destination_hotkey_ss58: str,
        amount: Optional[Balance] = None,
        move_all_stake: bool = False,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """
        Moves stake to a different hotkey and/or subnet.

        Parameters:
            wallet: The wallet to move stake from.
            origin_netuid: The netuid of the source subnet.
            origin_hotkey_ss58: The SS58 address of the source hotkey.
            destination_netuid: The netuid of the destination subnet.
            destination_hotkey_ss58: The SS58 address of the destination hotkey.
            amount: Amount of stake to move.
            move_all_stake: If `True`, moves all stake from the source hotkey to the destination hotkey.
            mev_protection: I`f` True, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Notes:
            - Price Protection: <https://docs.learnbittensor.org/learn/price-protection>
            - Rate Limits: <https://docs.learnbittensor.org/learn/chain-rate-limits#staking-operations-rate-limits>
        """
        check_balance_amount(amount)
        return await move_stake_extrinsic(
            subtensor=self,
            wallet=wallet,
            origin_netuid=origin_netuid,
            origin_hotkey_ss58=origin_hotkey_ss58,
            destination_netuid=destination_netuid,
            destination_hotkey_ss58=destination_hotkey_ss58,
            amount=amount,
            move_all_stake=move_all_stake,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def poke_deposit(
        self,
        wallet: "Wallet",
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """
        Adjusts deposits made for proxies and announcements based on current values.

        This method recalculates and updates the locked deposit amounts for both proxy relationships and announcements
        for the signing account. It can be used to potentially lower the locked amount if the deposit requirements have
        changed (e.g., due to runtime upgrades or changes in the number of proxies/announcements).

        Parameters:
            wallet: Bittensor wallet object (the account whose deposits will be adjusted).
            mev_protection: `If` True, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Note:
            This method automatically adjusts deposits for both proxy relationships and announcements. No parameters are
            needed as it operates on the account's current state.

        When to use:
            - After runtime upgrade, if deposit constants have changed.
            - After removing proxies/announcements, to free up excess locked funds.
            - Periodically to optimize locked deposit amounts.
        """
        return await poke_deposit_extrinsic(
            subtensor=self,
            wallet=wallet,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def proxy(
        self,
        wallet: "Wallet",
        real_account_ss58: str,
        force_proxy_type: Optional[Union[str, "ProxyType"]],
        call: "GenericCall",
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """
        Executes a call on behalf of the real account through a proxy.

        This method allows a proxy account (delegate) to execute a call on behalf of the real account (delegator). The
        call is subject to the permissions defined by the proxy type and must respect the delay period if one was set
        when the proxy was added.

        Parameters:
            wallet: Bittensor wallet object (should be the proxy account wallet).
            real_account_ss58: The SS58 address of the real account on whose behalf the call is being made.
            force_proxy_type: The type of proxy to use for the call. If `None`, any proxy type can be used. Otherwise,
                must match one of the allowed proxy types. Can be a string or ProxyType enum value.
            call: The inner call to be executed on behalf of the real account.
            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Notes:
            - The call must be permitted by the proxy type. For example, a "NonTransfer" proxy cannot execute transfer
              calls. The delay period must also have passed since the proxy was added.
        """
        return await proxy_extrinsic(
            subtensor=self,
            wallet=wallet,
            real_account_ss58=real_account_ss58,
            force_proxy_type=force_proxy_type,
            call=call,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def proxy_announced(
        self,
        wallet: "Wallet",
        delegate_ss58: str,
        real_account_ss58: str,
        force_proxy_type: Optional[Union[str, "ProxyType"]],
        call: "GenericCall",
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """
        Executes an announced call on behalf of the real account through a proxy.

        This method executes a call that was previously announced via :meth:`announce_proxy`. The call must match the
        call_hash that was announced, and the delay period must have passed since the announcement was made. The real
        account has the opportunity to review and reject the announcement before execution.

        Parameters:
            wallet: Bittensor wallet object (should be the proxy account wallet that made the announcement).
            delegate_ss58: The SS58 address of the delegate proxy account that made the announcement.
            real_account_ss58: The SS58 address of the real account on whose behalf the call will be made.
            force_proxy_type: The type of proxy to use for the call. If `None`, any proxy type can be used. Otherwise,
                must match one of the allowed proxy types. Can be a string or ProxyType enum value.
            call: The inner call to be executed on behalf of the real account (must match the announced call_hash).
            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Notes:
            - The call_hash of the provided call must match the call_hash that was announced. The announcement must not
              have been rejected by the real account, and the delay period must have passed.
        """
        return await proxy_announced_extrinsic(
            subtensor=self,
            wallet=wallet,
            delegate_ss58=delegate_ss58,
            real_account_ss58=real_account_ss58,
            force_proxy_type=force_proxy_type,
            call=call,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def refund_crowdloan(
        self,
        wallet: "Wallet",
        crowdloan_id: int,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """Refunds contributors from a failed crowdloan campaign that did not reach its cap.

        Refunds are batched, processing up to `RefundContributorsLimit` (default 50) contributors per call.
        For campaigns with more contributors, multiple calls are required. Only non-creator contributors are
        refunded; the creator's deposit remains until dissolution via `dissolve_crowdloan`.

        Only the crowdloan creator can call this method for a non-finalized crowdloan.

        Parameters:
            wallet: Bittensor wallet instance used to sign the transaction (must be the crowdloan creator).
            crowdloan_id: The unique identifier of the crowdloan to refund.
            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after submission.
            raise_error: If `True`, raises an exception rather than returning failure in the response.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            `ExtrinsicResponse` indicating success or failure, with error details if applicable.

        Notes:
            - Crowdloans Overview: <https://docs.learnbittensor.org/subnets/crowdloans>
            - Crowdloan Lifecycle: <https://docs.learnbittensor.org/subnets/crowdloans#crowdloan-lifecycle>
            - Refund and Dissolve: <https://docs.learnbittensor.org/subnets/crowdloans/crowdloans-tutorial#alternative-path-refund-and-dissolve>
        """
        return await refund_crowdloan_extrinsic(
            subtensor=self,
            wallet=wallet,
            crowdloan_id=crowdloan_id,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def reject_proxy_announcement(
        self,
        wallet: "Wallet",
        delegate_ss58: str,
        call_hash: str,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """
        Rejects an announcement made by a proxy delegate.

        This method allows the real account to reject an announcement made by a proxy delegate, preventing the announced
        call from being executed. Once rejected, the announcement cannot be executed and the announcement deposit is
        returned to the delegate.

        Parameters:
            wallet: Bittensor wallet object (should be the real account wallet).
            delegate_ss58: The SS58 address of the delegate proxy account whose announcement is being rejected.
            call_hash: The hash of the call that was announced and is now being rejected.
            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Notes:
            - Once rejected, the announcement cannot be executed. The delegate's announcement deposit is returned.
        """
        return await reject_announcement_extrinsic(
            subtensor=self,
            wallet=wallet,
            delegate_ss58=delegate_ss58,
            call_hash=call_hash,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def register(
        self: "AsyncSubtensor",
        wallet: "Wallet",
        netuid: int,
        max_allowed_attempts: int = 3,
        output_in_place: bool = False,
        cuda: bool = False,
        dev_id: Union[list[int], int] = 0,
        tpb: int = 256,
        num_processes: Optional[int] = None,
        update_interval: Optional[int] = None,
        log_verbose: bool = False,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """
        Registers a neuron on the Bittensor subnet with provided netuid using the provided wallet.

        Registration is a critical step for a neuron to become an active participant in the network, enabling it to
        stake, set weights, and receive incentives.

        Parameters:
            wallet: The wallet associated with the neuron to be registered.
            netuid: The unique identifier of the subnet.
            max_allowed_attempts: Maximum number of attempts to register the wallet.
            output_in_place: If `True`, prints the progress of the proof of work to the console in-place. Meaning the
                progress is printed on the same lines.
            cuda: If `true`, the wallet should be registered using CUDA device(s).
            dev_id: The CUDA device id to use, or a list of device ids.
            tpb: The number of threads per block (CUDA).
            num_processes: The number of processes to use to register.
            update_interval: The number of nonces to solve between updates.
            log_verbose: If `true`, the registration process will log more information.
            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        This function facilitates the entry of new neurons into the network, supporting the decentralized growth and
        scalability of the Bittensor ecosystem.

        Notes:
            - Rate Limits: <https://docs.learnbittensor.org/learn/chain-rate-limits#registration-rate-limits>
        """
        return await register_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            max_allowed_attempts=max_allowed_attempts,
            tpb=tpb,
            update_interval=update_interval,
            num_processes=num_processes,
            cuda=cuda,
            dev_id=dev_id,
            output_in_place=output_in_place,
            log_verbose=log_verbose,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def register_subnet(
        self: "AsyncSubtensor",
        wallet: "Wallet",
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """
        Registers a new subnetwork on the Bittensor network.

        Parameters:
            wallet: The wallet to be used for subnet registration.
            mev_protection: If True, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Notes:
            - Rate Limits: <https://docs.learnbittensor.org/learn/chain-rate-limits#network-registration-rate-limit>
        """
        return await register_subnet_extrinsic(
            subtensor=self,
            wallet=wallet,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def remove_proxy_announcement(
        self,
        wallet: "Wallet",
        real_account_ss58: str,
        call_hash: str,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """
        Removes an announcement made by a proxy account.

        This method allows the proxy account to remove its own announcement before it is executed or rejected. This
        frees up the announcement deposit and prevents the call from being executed. Only the proxy account that made
        the announcement can remove it.

        Parameters:
            wallet: Bittensor wallet object (should be the proxy account wallet that made the announcement).
            real_account_ss58: The SS58 address of the real account on whose behalf the call was announced.
            call_hash: The hash of the call that was announced and is now being removed.
            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Notes:
            - Only the proxy account that made the announcement can remove it. The real account can reject it via
              :meth:`reject_proxy_announcement`, but cannot remove it directly.
        """
        return await remove_announcement_extrinsic(
            subtensor=self,
            wallet=wallet,
            real_account_ss58=real_account_ss58,
            call_hash=call_hash,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def remove_liquidity(
        self,
        wallet: "Wallet",
        netuid: int,
        position_id: int,
        hotkey_ss58: Optional[str] = None,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """Remove liquidity and credit balances back to wallet's hotkey stake.

        Parameters:
            wallet: The wallet used to sign the extrinsic (must be unlocked).
            netuid: The UID of the target subnet for which the call is being initiated.
            position_id: The id of the position record in the pool.
            hotkey_ss58: The hotkey with staked TAO in Alpha. If not passed then the wallet hotkey is used.
            mev_protection:` If` True, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Note:
            - Adding is allowed even when user liquidity is enabled in specified subnet. Call `toggle_user_liquidity`
              extrinsic to enable/disable user liquidity.
            - To get the `position_id` use `get_liquidity_list` method.
        """
        return await remove_liquidity_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            position_id=position_id,
            hotkey_ss58=hotkey_ss58,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def remove_proxies(
        self,
        wallet: "Wallet",
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """
        Removes all proxy relationships for the account in a single transaction.

        This method removes all proxy relationships for the signing account in a single call, which is more efficient
        than removing them one by one using :meth:`remove_proxy`. The deposit for all proxies will be returned to the
        account.

        Parameters:
            wallet: Bittensor wallet object. The account whose proxies will be removed (the delegator). All proxy
                relationships where wallet.coldkey.ss58_address is the real account will be removed.
            mev_protection: If `True`, encr`ypts` and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Notes:
            - This removes all proxy relationships for the account, regardless of proxy type or delegate. Use
              :meth:`remove_proxy` if you need to remove specific proxy relationships selectively.
        """
        return await remove_proxies_extrinsic(
            subtensor=self,
            wallet=wallet,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def remove_proxy(
        self,
        wallet: "Wallet",
        delegate_ss58: str,
        proxy_type: Union[str, "ProxyType"],
        delay: int,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """
        Removes a specific proxy relationship.

        This method removes a single proxy relationship between the real account and a delegate. The parameters must
        exactly match those used when the proxy was added via :meth:`add_proxy`. The deposit for this proxy will be returned
        to the account.

        Parameters:
            wallet: Bittensor wallet object.
            delegate_ss58: The SS58 address of the delegate proxy account to remove.
            proxy_type: The type of proxy permissions to remove. Can be a string or ProxyType enum value.
            delay: The announcement delay value (in blocks) for the proxy being removed. Must exactly match the delay
                value that was set when the proxy was originally added via :meth:`add_proxy`. This is a required
                identifier for the specific proxy relationship, not a delay before removal takes effect (removal is
                immediate).
            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Notes:
            - The delegate_ss58, proxy_type, and delay parameters must exactly match those used when the proxy was added.
              Use :meth:`get_proxies_for_real_account` to retrieve the exact parameters for existing proxies.
        """
        return await remove_proxy_extrinsic(
            subtensor=self,
            wallet=wallet,
            delegate_ss58=delegate_ss58,
            proxy_type=proxy_type,
            delay=delay,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def reveal_weights(
        self,
        wallet: "Wallet",
        netuid: int,
        uids: UIDs,
        weights: Weights,
        salt: Salt,
        mechid: int = 0,
        max_attempts: int = 5,
        version_key: int = version_as_int,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = 16,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """
        Reveals the weights for a specific subnet on the Bittensor blockchain using the provided wallet.
        This action serves as a revelation of the neuron's previously committed weight distribution.

        Parameters:
            wallet: Bittensor Wallet instance.
            netuid: The unique identifier of the subnet.
            uids: NumPy array of neuron UIDs for which weights are being revealed.
            weights: NumPy array of weight values corresponding to each UID.
            salt: NumPy array of salt values corresponding to the hash function.
            mechid: The subnet mechanism unique identifier.
            max_attempts: The number of maximum attempts to reveal weights.
            version_key: Version key for compatibility with the network.
            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        This function allows neurons to reveal their previously committed weight distribution, ensuring transparency and
        accountability within the Bittensor network.

        Notes:
            - <https://docs.learnbittensor.org/glossary#commit-reveal>
            - Rate Limits: <https://docs.learnbittensor.org/learn/chain-rate-limits#weights-setting-rate-limit>

        """
        attempt = 0
        response = ExtrinsicResponse(False)

        if attempt_check := validate_max_attempts(max_attempts, response):
            return attempt_check

        while attempt < max_attempts and response.success is False:
            try:
                response = await reveal_weights_extrinsic(
                    subtensor=self,
                    wallet=wallet,
                    netuid=netuid,
                    mechid=mechid,
                    uids=uids,
                    weights=weights,
                    salt=salt,
                    version_key=version_key,
                    mev_protection=mev_protection,
                    period=period,
                    raise_error=raise_error,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                    wait_for_revealed_execution=wait_for_revealed_execution,
                )
            except Exception as error:
                return ExtrinsicResponse.from_exception(
                    raise_error=raise_error, error=error
                )
            attempt += 1

        if not response.success:
            logging.debug("No attempt made. Perhaps it is too soon to reveal weights!")
        return response

    async def root_register(
        self,
        wallet: "Wallet",
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """
        Register neuron by recycling some TAO.

        Parameters:
            wallet: The wallet associated with the neuron to be registered.
            mev_protection: If True, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Notes:
            - Rate Limits: <https://docs.learnbittensor.org/learn/chain-rate-limits#registration-rate-limits>
        """

        return await root_register_extrinsic(
            subtensor=self,
            wallet=wallet,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def root_set_pending_childkey_cooldown(
        self,
        wallet: "Wallet",
        cooldown: int,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """Sets the pending childkey cooldown.

        Parameters:
            wallet: bittensor wallet instance.
            cooldown: the number of blocks to setting pending childkey cooldown.
            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's
                submitted. If the transaction is not included in a block within that number of blocks, it will expire
                and be rejected. You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Note:
            This operation can only be successfully performed if your wallet has root privileges.
        """
        return await root_set_pending_childkey_cooldown_extrinsic(
            subtensor=self,
            wallet=wallet,
            cooldown=cooldown,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def set_auto_stake(
        self,
        wallet: "Wallet",
        netuid: int,
        hotkey_ss58: str,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """Sets the coldkey to automatically stake to the hotkey within specific subnet mechanism.

        Parameters:
            wallet: Bittensor Wallet instance.
            netuid: The subnet unique identifier.
            hotkey_ss58: The SS58 address of the validator's hotkey to which the miner automatically stakes all rewards
                received from the specified subnet immediately upon receipt.
            mev_protection: If T`rue`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Note:
            Use the `get_auto_stakes` method to get the hotkey address of the validator where auto stake is set.
        """
        return await set_auto_stake_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            hotkey_ss58=hotkey_ss58,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def set_children(
        self,
        wallet: "Wallet",
        hotkey_ss58: str,
        netuid: int,
        children: list[tuple[float, str]],
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """
        Allows a coldkey to set children-keys.

        Parameters:
            wallet: bittensor wallet instance.
            hotkey_ss58: The `SS58` address of the neuron's hotkey.
            netuid: The netuid value.
            children: A list of children with their proportions.
            mev_protection: If True, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's
                submitted. If the transaction is not included in a block within that number of blocks, it will expire
                and be rejected. You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Raises:
            DuplicateChild: There are duplicates in the list of children.
            InvalidChild: Child is the hotkey.
            NonAssociatedColdKey: The coldkey does not own the hotkey or the child is the same as the hotkey.
            NotEnoughStakeToSetChildkeys: Parent key doesn't have minimum own stake.
            ProportionOverflow: The sum of the proportions does exceed uint64.
            RegistrationNotPermittedOnRootSubnet: Attempting to register a child on the root network.
            SubnetNotExists: Attempting to register to a non-existent network.
            TooManyChildren: Too many children in request.
            TxRateLimitExceeded: Hotkey hit the rate limit.
            bittensor_wallet.errors.KeyFileError: Failed to decode keyfile data.
            bittensor_wallet.errors.PasswordError: Decryption failed or wrong password for decryption provided.

        Notes:
            - Rate Limits: <https://docs.learnbittensor.org/learn/chain-rate-limits#child-hotkey-operations-rate-limit>
        """
        return await set_children_extrinsic(
            subtensor=self,
            wallet=wallet,
            hotkey_ss58=hotkey_ss58,
            netuid=netuid,
            children=children,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def set_delegate_take(
        self,
        wallet: "Wallet",
        hotkey_ss58: str,
        take: float,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """
        Sets the delegate 'take' percentage for a neuron identified by its hotkey.
        The 'take' represents the percentage of rewards that the delegate claims from its nominators' stakes.

        Parameters:
            wallet: bittensor wallet instance.
            hotkey_ss58: The `SS58` address of the neuron's hotkey.
            take: Percentage reward for the delegate.
            mev_protection:` If` True, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's
                submitted. If the transaction is not included in a block within that number of blocks, it will expire
                and be rejected. You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Raises:
            DelegateTakeTooHigh: Delegate take is too high.
            DelegateTakeTooLow: Delegate take is too low.
            DelegateTxRateLimitExceeded: A transactor exceeded the rate limit for delegate transaction.
            HotKeyAccountNotExists: The hotkey does not exist.
            NonAssociatedColdKey: Request to stake, unstake, or subscribe is made by a coldkey that is not associated
                with the hotkey account.
            bittensor_wallet.errors.PasswordError: Decryption failed or wrong password for decryption provided.
            bittensor_wallet.errors.KeyFileError: Failed to decode keyfile data.

        The delegate take is a critical parameter in the network's incentive structure, influencing the distribution of
        rewards among neurons and their nominators.

        Notes:
            - Rate Limits: <https://docs.learnbittensor.org/learn/chain-rate-limits#delegate-take-rate-limit>
        """
        # u16 representation of the take
        take_u16 = int(take * 0xFFFF)

        current_take = await self.get_delegate_take(hotkey_ss58)
        current_take_u16 = int(current_take * 0xFFFF)

        if current_take_u16 == take_u16:
            message = f"The take for {hotkey_ss58} is already set to {take}."
            logging.debug(f"[green]{message}[/green].")
            return ExtrinsicResponse(True, message)

        logging.debug(f"Updating {hotkey_ss58} take: current={current_take} new={take}")

        response = await set_take_extrinsic(
            subtensor=self,
            wallet=wallet,
            hotkey_ss58=hotkey_ss58,
            take=take_u16,
            action="increase_take" if current_take_u16 < take_u16 else "decrease_take",
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_finalization=wait_for_finalization,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

        if response.success:
            return response

        logging.error(f"[red]{response.message}[/red]")
        return response

    async def set_root_claim_type(
        self,
        wallet: "Wallet",
        new_root_claim_type: "Literal['Swap', 'Keep'] | RootClaimType | dict",
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ):
        """Submit an extrinsic to set the root claim type for the wallet's coldkey.

        The root claim type determines how future Alpha dividends from subnets are handled when they are claimed for
        the wallet's coldkey:

        - `Swap`: Alpha dividends are swapped to TAO at claim time and restaked on the Root Subnet (default).
        - `Keep`: Alpha dividends remain as Alpha on the originating subnets.

        Parameters:

            wallet: Bittensor `Wallet` instance.
            new_root_claim_type: The new root claim type to set. Can be:
                - String: "Swap" or "Keep"
                - RootClaimType: RootClaimType.Swap, RootClaimType.Keep
                - Dict: {"KeepSubnets": {"subnets": [1, 2, 3]}}
                - Callable: RootClaimType.KeepSubnets([1, 2, 3])

            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: Number of blocks for which the transaction remains valid after submission. If the extrinsic is
                not included in a block within this window, it will expire and be rejected.
            raise_error: Whether to raise a Python exception instead of returning a failed `ExtrinsicResponse`.
            wait_for_inclusion: Whether to wait until the extrinsic is included in a block before returning.
            wait_for_finalization: Whether to wait for finalization of the extrinsic in a block before returning.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            `ExtrinsicResponse` describing the result of the extrinsic execution.

        Notes:
            - This setting applies to both automatic and manual root claims going forward; it does not retroactively
              change how already-claimed dividends were processed.
            - Only the treatment of Alpha dividends is affected; the underlying TAO stake on the Root Subnet is
              unchanged.
            - See: <https://docs.learnbittensor.org/staking-and-delegation/root-claims>
            - See also: <https://docs.learnbittensor.org/staking-and-delegation/root-claims/managing-root-claims>
        """
        return await set_root_claim_type_extrinsic(
            subtensor=self,
            wallet=wallet,
            new_root_claim_type=new_root_claim_type,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def set_subnet_identity(
        self,
        wallet: "Wallet",
        netuid: int,
        subnet_identity: SubnetIdentity,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """
        Sets the identity of a subnet for a specific wallet and network.

        Parameters:
            wallet: The wallet instance that will authorize the transaction.
            netuid: The unique ID of the network on which the operation takes place.
            subnet_identity: The identity data of the subnet including attributes like name, GitHub repository, contact,
                URL, discord, description, and any additional metadata.
            mev_protection:` If` True, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's
                submitted. If the transaction is not included in a block within that number of blocks, it will expire
                and be rejected. You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.
        """
        return await set_subnet_identity_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            subnet_name=subnet_identity.subnet_name,
            github_repo=subnet_identity.github_repo,
            subnet_contact=subnet_identity.subnet_contact,
            subnet_url=subnet_identity.subnet_url,
            logo_url=subnet_identity.logo_url,
            discord=subnet_identity.discord,
            description=subnet_identity.description,
            additional=subnet_identity.additional,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def set_weights(
        self,
        wallet: "Wallet",
        netuid: int,
        uids: UIDs,
        weights: Weights,
        mechid: int = 0,
        block_time: float = 12.0,
        commit_reveal_version: int = 4,
        max_attempts: int = 5,
        version_key: int = version_as_int,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """Sets the weight vector for a neuron acting as a validator, specifying the weights assigned to subnet miners
        based on their performance evaluation.

        This method allows subnet validators to submit their weight vectors, which rank the value of each subnet miner's
        work. These weight vectors are used by the Yuma Consensus algorithm to compute emissions for both validators and
        miners.

        The method automatically handles both commit-reveal-enabled subnets (CRv4) and direct weight setting. For
        commit-reveal subnets, weights are committed first and then revealed after the reveal period. The method respects
        rate limiting constraints enforced by `_blocks_weight_limit`.

        Parameters:
            wallet: The wallet associated with the subnet validator setting the weights.
            netuid: The unique identifier of the subnet.
            uids: The list of subnet miner neuron UIDs that the weights are being set for.
            weights: The corresponding weights to be set for each UID, representing the validator's evaluation of each
                miner's performance.
            mechid: The subnet mechanism unique identifier (default 0 for primary mechanism).
            block_time: The block duration in seconds (default 12.0). Used for timing calculations in commit-reveal
                operations.
            commit_reveal_version: The version of the commit-reveal protocol to use (default 4 for CRv4).
            max_attempts: The maximum number of attempts to set weights if rate limiting is encountered (default 5).
            version_key: Version key for compatibility with the network.
            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Example:

            # Set weights directly (for non-commit-reveal subnets)

            response = await subtensor.set_weights(
                wallet=wallet,
                netuid=1,
                uids=[0, 1, 2],
                weights=[0.5, 0.3, 0.2]
            )

            # For commit-reveal subnets, the method automatically handles commit and reveal phases

        Notes:
            This function is crucial in the Yuma Consensus mechanism, where each validator's weight vector contributes
            to the overall weight matrix used to calculate emissions and maintain network consensus.

            - <https://docs.learnbittensor.org/resources/glossary#yuma-consensus>
            - <https://docs.learnbittensor.org/concepts/commit-reveal>
            - Rate Limits: <https://docs.learnbittensor.org/learn/chain-rate-limits#weights-setting-rate-limit>

        """
        attempt = 0
        response = ExtrinsicResponse(False)

        if attempt_check := validate_max_attempts(max_attempts, response):
            return attempt_check

        async def _blocks_weight_limit() -> bool:
            bslu, wrl = await asyncio.gather(
                self.blocks_since_last_update(netuid, uid),
                self.weights_rate_limit(netuid),
            )
            return bslu > wrl

        if (
            uid := await self.get_uid_for_hotkey_on_subnet(
                wallet.hotkey.ss58_address, netuid
            )
        ) is None:
            return ExtrinsicResponse(
                False,
                f"Hotkey {wallet.hotkey.ss58_address} not registered in subnet {netuid}.",
            )

        if await self.commit_reveal_enabled(netuid=netuid):
            # go with `commit_timelocked_mechanism_weights_extrinsic` extrinsic

            while (
                attempt < max_attempts
                and response.success is False
                and await _blocks_weight_limit()
            ):
                logging.debug(
                    f"Committing weights {weights} for subnet [blue]{netuid}[/blue]. "
                    f"Attempt [blue]{attempt + 1}[blue] of [green]{max_attempts}[/green]."
                )
                try:
                    response = await commit_timelocked_weights_extrinsic(
                        subtensor=self,
                        wallet=wallet,
                        netuid=netuid,
                        mechid=mechid,
                        uids=uids,
                        weights=weights,
                        block_time=block_time,
                        commit_reveal_version=commit_reveal_version,
                        version_key=version_key,
                        mev_protection=mev_protection,
                        period=period,
                        raise_error=raise_error,
                        wait_for_inclusion=wait_for_inclusion,
                        wait_for_finalization=wait_for_finalization,
                        wait_for_revealed_execution=wait_for_revealed_execution,
                    )
                except Exception as error:
                    return ExtrinsicResponse.from_exception(
                        raise_error=raise_error, error=error
                    )
                attempt += 1
        else:
            # go with `set_mechanism_weights_extrinsic`

            while (
                attempt < max_attempts
                and response.success is False
                and await _blocks_weight_limit()
            ):
                try:
                    logging.debug(
                        f"Setting weights for subnet #[blue]{netuid}[/blue]. "
                        f"Attempt [blue]{attempt + 1}[/blue] of [green]{max_attempts}[/green]."
                    )
                    response = await set_weights_extrinsic(
                        subtensor=self,
                        wallet=wallet,
                        netuid=netuid,
                        mechid=mechid,
                        uids=uids,
                        weights=weights,
                        version_key=version_key,
                        mev_protection=mev_protection,
                        period=period,
                        raise_error=raise_error,
                        wait_for_inclusion=wait_for_inclusion,
                        wait_for_finalization=wait_for_finalization,
                        wait_for_revealed_execution=wait_for_revealed_execution,
                    )
                except Exception as error:
                    return ExtrinsicResponse.from_exception(
                        raise_error=raise_error, error=error
                    )
                attempt += 1

        if not response.success:
            logging.debug(
                "No one successful attempt made. Perhaps it is too soon to set weights!"
            )
        return response

    async def serve_axon(
        self,
        netuid: int,
        axon: "Axon",
        certificate: Optional[Certificate] = None,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """Registers an Axon endpoint on the network for receiving queries from other neurons.

        This method publishes your neuron's IP address, port, and protocol information to the blockchain, making it
        discoverable by other neurons in the subnet. Optionally, you can include a TLS certificate to enable secure,
        encrypted communication via mutual TLS (mTLS).

        When a certificate is provided, the blockchain stores both your endpoint information and your TLS public key,
        allowing other neurons to discover your certificate and establish encrypted connections. When re-serving with
        updated metadata (including a new certificate), the previous values are overwritten.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            axon: The Axon instance containing your endpoint configuration (IP, port, protocol).
            certificate: Optional TLS certificate for secure communication. Should contain a public key (up to 64
                bytes) and algorithm identifier. If `None`, standard unencrypted serving is used.
            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until
                validators decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after submission. If not
                included in a block within this period, the transaction expires.
            raise_error: If True, raises an exception on failure instead of returning an error response.
            wait_for_inclusion: If True, waits for the transaction to be included in a block before returning.
            wait_for_finalization: If True, waits for the transaction to be finalized on the blockchain.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection
                used.

        Returns:
            ExtrinsicResponse containing the success status and transaction details. On success, the response includes
            the external IP and port that were registered.

        Notes:
            - Rate Limits: <https://docs.learnbittensor.org/learn/chain-rate-limits#serving-rate-limits>
        """
        return await serve_axon_extrinsic(
            subtensor=self,
            netuid=netuid,
            axon=axon,
            certificate=certificate,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def set_commitment(
        self,
        wallet: "Wallet",
        netuid: int,
        data: str,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """Commits arbitrary data to the Bittensor network by publishing metadata.
        # TODO: check with @roman, is this about 'arbitrary data' or 'commit-reveal'? we need a real example here if this is important.
                This method allows neurons to publish arbitrary data to the blockchain, which can be used for various purposes
                such as sharing model updates, configuration data, or other network-relevant information. The data is encoded
                and stored on-chain as metadata.

        Parameters:
            wallet: The wallet associated with the neuron committing the data.
            netuid: The unique identifier of the subnetwork.
            data: The data string to be committed to the network. The data will be encoded as bytes before submission.
            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Notes:
            The data is automatically encoded as bytes before submission. There may be size limits on metadata
            payloads enforced by the chain.

            - <https://docs.learnbittensor.org/resources/glossary#commit-reveal>
            - <https://docs.learnbittensor.org/concepts/commit-reveal>
        """
        return await publish_metadata_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            data_type=f"Raw{len(data)}",
            data=data.encode(),
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def set_reveal_commitment(
        self,
        wallet,
        netuid: int,
        data: str,
        blocks_until_reveal: int = 360,
        block_time: Union[int, float] = 12,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """Commits arbitrary data to the Bittensor network using timelock encryption for reveal scheduling.

        This method commits data that will be automatically revealed after a specified number of blocks using drand
        timelock encryption. The data is encrypted using `get_encrypted_commitment`, which uses drand rounds to ensure
        the data cannot be revealed before the specified reveal time.

        # TODO: check with @roman, is this about 'arbitrary data' or 'commit-reveal'? we need a real example here if this is important, and documentating a real commit reveal flow.

        Parameters:
            block_time: The number of seconds between each block (default 12.0 for standard blocks, 10.0 for fast blocks).
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.

            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution. The response's "data" field contains
            `{"encrypted": encrypted, "reveal_round": reveal_round}` on success.

        Notes:
            A commitment can be set once per subnet epoch and is reset at the next epoch automatically. The timelock
            encryption ensures the data cannot be revealed before the specified drand round.

            - <https://docs.learnbittensor.org/resources/glossary#commit-reveal>
            - <https://docs.learnbittensor.org/resources/glossary#drandtime-lock-encryption>
            - <https://docs.learnbittensor.org/concepts/commit-reveal>
        """

        encrypted, reveal_round = get_encrypted_commitment(
            data, blocks_until_reveal, block_time
        )

        data_ = {"encrypted": encrypted, "reveal_round": reveal_round}
        response = await publish_metadata_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            data_type="TimelockEncrypted",
            data=data_,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )
        response.data = data_
        return response

    async def start_call(
        self,
        wallet: "Wallet",
        netuid: int,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """Submits a start_call extrinsic to the blockchain to trigger emission start for a subnet.

        This method initiates the emission mechanism for a newly registered subnet. Once called, the subnet becomes
        "active" and begins receiving TAO emissions. Only the subnet owner (the wallet that registered the subnet) is
        authorized to call this method.

        Parameters:
            wallet: The wallet used to sign the extrinsic (must be unlocked and must be the subnet owner).
            netuid: The unique identifier of the target subnet for which emissions are being started.
            mev_protection: If True, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Notes:
            Only the subnet owner can call this method. After successful execution, the subnet becomes active and
            eligible for TAO emissions.

            - <https://docs.learnbittensor.org/subnets/create-a-subnet>
        """
        return await start_call_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def swap_stake(
        self,
        wallet: "Wallet",
        hotkey_ss58: str,
        origin_netuid: int,
        destination_netuid: int,
        amount: Balance,
        safe_swapping: bool = False,
        allow_partial_stake: bool = False,
        rate_tolerance: float = 0.005,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """Moves stake between subnets while keeping the same coldkey-hotkey pair ownership.

        This method swaps stake from one subnet to another, effectively moving the same stake amount (minus fees) from
        the origin subnet to the destination subnet. Like subnet hopping - same owner, same hotkey, just changing which
        subnet the stake is in.

        The `amount` parameter is specified as a Balance object (in TAO or Alpha units depending on the subnet). The
        actual amount received may be less due to swap fees and potential slippage. When `safe_swapping` is enabled, the
        method uses price ratio checks to protect against unfavorable price movements during the swap.

        Parameters:
            wallet: The wallet to swap stake from.
            hotkey_ss58: The SS58 address of the hotkey whose stake is being swapped.
            origin_netuid: The netuid from which stake is removed.
            destination_netuid: The netuid to which stake is added.
            amount: The amount to swap as a Balance object (in TAO or Alpha units). The actual amount received may be
                less due to swap fees and slippage.
            safe_swapping: If `True`, enables price safety checks to protect against fluctuating prices. The swap
                will only execute if the price ratio between subnets doesn't exceed the rate tolerance.
            allow_partial_stake: If `True` and `safe_swapping` is enabled, allows partial stake swaps when the full
                amount would exceed the price tolerance. If `False`, the entire swap fails if it would exceed the
                tolerance.
            rate_tolerance: The maximum allowed increase in the price ratio between subnets
                (origin_price/destination_price). For example, 0.005 = 0.5% maximum increase. Only used when
                `safe_swapping` is `True`.
                safe_staking is True.
            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Notes:
            The price ratio for swap_stake in safe mode is calculated as: origin_subnet_price / destination_subnet_price.
            When `safe_swapping` is enabled, the swap will only execute if:
            - With `allow_partial_stake=False`: The entire swap amount can be executed without the price ratio
              increasing more than `rate_tolerance`.
            - With `allow_partial_stake=True`: A partial amount will be swapped up to the point where the price ratio
              would increase by `rate_tolerance`.
            - Price Protection: <https://docs.learnbittensor.org/learn/price-protection>
            - Rate Limits: <https://docs.learnbittensor.org/learn/chain-rate-limits#staking-operations-rate-limits>

            - <https://docs.learnbittensor.org/navigating-subtensor/swap-stake>
        """
        check_balance_amount(amount)
        return await swap_stake_extrinsic(
            subtensor=self,
            wallet=wallet,
            hotkey_ss58=hotkey_ss58,
            origin_netuid=origin_netuid,
            destination_netuid=destination_netuid,
            amount=amount,
            safe_swapping=safe_swapping,
            allow_partial_stake=allow_partial_stake,
            rate_tolerance=rate_tolerance,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def toggle_user_liquidity(
        self,
        wallet: "Wallet",
        netuid: int,
        enable: bool,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """Toggles the user liquidity feature for a specified subnet.

        This method enables or disables user liquidity positions for a subnet. Only the subnet owner (the wallet that
        registered the subnet) is authorized to call this method.

        Parameters:
            wallet: The wallet used to sign the extrinsic (must be unlocked and must be the subnet owner).
            netuid: The unique identifier of the target subnet for which user liquidity is being toggled.
            enable: Boolean indicating whether to enable (`True`) or disable (`False`) user liquidity.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected.
            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Notes:
            Only the subnet owner can execute this call successfully.

            - <https://docs.learnbittensor.org/liquidity-positions/liquidity-positions>
        """
        return await toggle_user_liquidity_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            enable=enable,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def transfer(
        self,
        wallet: "Wallet",
        destination_ss58: str,
        amount: Optional[Balance],
        transfer_all: bool = False,
        keep_alive: bool = True,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """Transfers TAO tokens from the source wallet to a destination address.

        This method transfers TAO tokens from the wallet's coldkey to the specified destination address. The amount is
        specified as a Balance object (in TAO or Rao units). Use `get_transfer_fee` to pre-estimate the transaction fee
        before sending.

        When `keep_alive=True`, the transfer ensures the source account maintains at least the existential deposit
        amount. If `keep_alive=False`, the transfer may reduce the source account balance below the existential deposit,
        which could result in the account being reaped (removed) from the chain.

        Parameters:
            wallet: Source wallet for the transfer (must be unlocked).
            destination_ss58: Destination SS58 address for the transfer.
            amount: Amount of TAO to transfer as a Balance object. If `None` and `transfer_all=True`, transfers all
                available balance minus fees and existential deposit (if `keep_alive=True`).
            transfer_all: If `True`, transfers all available tokens (minus fees and existential deposit if
                `keep_alive=True`). Ignored if `amount` is specified.
            keep_alive: If `True`, ensures the source account maintains at least the existential deposit amount. If
                `False`, the transfer may reduce the balance below the existential deposit, potentially causing the
                account to be reaped.
            mev_protection:` If` True, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Notes:
            The existential deposit is the minimum balance required to keep an account alive on the chain. Use
            :meth:`get_existential_deposit` to query the current value.

            - <https://docs.learnbittensor.org/resources/glossary#existential-deposit>
            - <https://docs.learnbittensor.org/resources/glossary#transfer>
        """
        check_balance_amount(amount)
        return await transfer_extrinsic(
            subtensor=self,
            wallet=wallet,
            destination_ss58=destination_ss58,
            amount=amount,
            transfer_all=transfer_all,
            keep_alive=keep_alive,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def transfer_stake(
        self,
        wallet: "Wallet",
        destination_coldkey_ss58: str,
        hotkey_ss58: str,
        origin_netuid: int,
        destination_netuid: int,
        amount: Balance,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """
        Transfers stake from one subnet to another while changing the coldkey owner.

        Parameters:
            wallet: The wallet to transfer stake from.
            destination_coldkey_ss58: The destination coldkey SS58 address.
            hotkey_ss58: The hotkey SS58 address associated with the stake.
            origin_netuid: The source subnet UID.
            destination_netuid: The destination subnet UID.
            amount: Amount to transfer.
            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Notes:
            - Price Protection: <https://docs.learnbittensor.org/learn/price-protection>
            - Rate Limits: <https://docs.learnbittensor.org/learn/chain-rate-limits#staking-operations-rate-limits>
        """
        check_balance_amount(amount)
        return await transfer_stake_extrinsic(
            subtensor=self,
            wallet=wallet,
            destination_coldkey_ss58=destination_coldkey_ss58,
            hotkey_ss58=hotkey_ss58,
            origin_netuid=origin_netuid,
            destination_netuid=destination_netuid,
            amount=amount,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def unstake(
        self,
        wallet: "Wallet",
        netuid: int,
        hotkey_ss58: str,
        amount: Balance,
        allow_partial_stake: bool = False,
        safe_unstaking: bool = False,
        rate_tolerance: float = 0.005,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """
        Removes a specified amount of stake from a single hotkey account. This function is critical for adjusting
        individual neuron stakes within the Bittensor network.

        Parameters:
            wallet: The wallet associated with the neuron from which the stake is being removed.
            netuid: The unique identifier of the subnet.
            hotkey_ss58: The `SS58` address of the hotkey account to unstake from.
            amount: The amount of alpha to unstake. If not specified, unstakes all. Alpha amount.
            allow_partial_stake: If `True` and safe_staking is enabled, allows partial unstaking when
                the full amount would exceed the price tolerance. If false, the entire unstake fails if it would
                exceed the tolerance.
            rate_tolerance: The maximum allowed price change ratio when unstaking. For example,
                0.005 = 0.5% maximum price decrease. Only used when safe_staking is True.
            safe_unstaking: If `True`, enables price safety checks to protect against fluctuating prices. The unstake
                will only execute if the price change doesn't exceed the rate tolerance.
            mev_protection: If T`rue`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        This function supports flexible stake management, allowing neurons to adjust their network participation and
        potential reward accruals.

        Notes:
            - Price Protection: <https://docs.learnbittensor.org/learn/price-protection>
            - Rate Limits: <https://docs.learnbittensor.org/learn/chain-rate-limits#staking-operations-rate-limits>
        """
        check_balance_amount(amount)
        return await unstake_extrinsic(
            subtensor=self,
            wallet=wallet,
            hotkey_ss58=hotkey_ss58,
            netuid=netuid,
            amount=amount,
            allow_partial_stake=allow_partial_stake,
            rate_tolerance=rate_tolerance,
            safe_unstaking=safe_unstaking,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def unstake_all(
        self,
        wallet: "Wallet",
        netuid: int,
        hotkey_ss58: str,
        rate_tolerance: Optional[float] = 0.005,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """Unstakes all TAO/Alpha associated with a hotkey from the specified subnet on the Bittensor network.

        This method unstakes all stake from a hotkey on a specific subnet. When `rate_tolerance` is specified, the method
        uses safe unstaking behavior to protect against unfavorable price movements due to liquidity/price impact. The
        `rate_tolerance` parameter limits the maximum price change ratio during the unstaking operation.

        Parameters:
            wallet: The wallet of the stake owner (must be unlocked).
            netuid: The unique identifier of the subnet.
            hotkey_ss58: The SS58 address of the hotkey to unstake from.
            rate_tolerance: The maximum allowed price change ratio when unstaking (default 0.005 = 0.5% maximum price
                decrease). If `None`, unstaking proceeds without price limit protection. Only used for subnets with
                liquidity pools where price impact may occur.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected.
            mev_protection: If` True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Example:

            # If you would like to unstake all stakes in all subnets safely, use default `rate_tolerance` or pass your
            # value:

            import bittensor as bt

            subtensor = bt.AsyncSubtensor()
            wallet = bt.Wallet("my_wallet")
            netuid = 14
            hotkey = "5%SOME_HOTKEY_WHERE_IS_YOUR_STAKE_NOW%"

            wallet_stakes = await subtensor.get_stake_info_for_coldkey(coldkey_ss58=wallet.coldkey.ss58_address)

            for stake in wallet_stakes:
                result = await subtensor.unstake_all(
                    wallet=wallet,
                    hotkey_ss58=stake.hotkey_ss58,
                    netuid=stake.netuid,
                )
                print(result)

            # If you would like to unstake all stakes in all subnets unsafely, use `rate_tolerance=None`:

            import bittensor as bt

            subtensor = bt.AsyncSubtensor()
            wallet = bt.Wallet("my_wallet")
            netuid = 14
            hotkey = "5%SOME_HOTKEY_WHERE_IS_YOUR_STAKE_NOW%"

            wallet_stakes = await subtensor.get_stake_info_for_coldkey(coldkey_ss58=wallet.coldkey.ss58_address)

            for stake in wallet_stakes:
                result = await subtensor.unstake_all(
                    wallet=wallet,
                    hotkey_ss58=stake.hotkey_ss58,
                    netuid=stake.netuid,
                    rate_tolerance=None,
                )
                print(result)

        Notes:
            - Slippage: <https://docs.learnbittensor.org/learn/slippage>
            - Price Protection: <https://docs.learnbittensor.org/learn/price-protection>
            - Rate Limits: <https://docs.learnbittensor.org/learn/chain-rate-limits#staking-operations-rate-limits>
            - Managing Stake with SDK: <https://docs.learnbittensor.org/staking-and-delegation/managing-stake-sdk>

        """
        return await unstake_all_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            hotkey_ss58=hotkey_ss58,
            rate_tolerance=rate_tolerance,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def unstake_multiple(
        self,
        wallet: "Wallet",
        netuids: UIDs,
        hotkey_ss58s: list[str],
        amounts: Optional[list[Balance]] = None,
        unstake_all: bool = False,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """
        Performs batch unstaking from multiple hotkey accounts, allowing a neuron to reduce its staked amounts
        efficiently. This function is useful for managing the distribution of stakes across multiple neurons.

        Parameters:
            wallet: The wallet linked to the coldkey from which the stakes are being withdrawn.
            netuids: Subnets unique IDs.
            hotkey_ss58s: A list of hotkey `SS58` addresses to unstake from.
            amounts: The amounts of TAO to unstake from each hotkey. If not provided, unstakes all.
            unstake_all: If `True`, unstakes all tokens. Amounts are ignored.
            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Notes:
            - Slippage: <https://docs.learnbittensor.org/learn/slippage>
            - Price Protection: <https://docs.learnbittensor.org/learn/price-protection>
            - Rate Limits: <https://docs.learnbittensor.org/learn/chain-rate-limits#staking-operations-rate-limits>
            - Managing Stake with SDK: <https://docs.learnbittensor.org/staking-and-delegation/managing-stake-sdk>

        """
        return await unstake_multiple_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuids=netuids,
            hotkey_ss58s=hotkey_ss58s,
            amounts=amounts,
            unstake_all=unstake_all,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def update_cap_crowdloan(
        self,
        wallet: "Wallet",
        crowdloan_id: int,
        new_cap: "Balance",
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """Updates the fundraising cap of an active (non-finalized) crowdloan.

        Allows the creator to adjust the maximum total contribution amount before finalization. The new cap
        must be at least equal to the amount already raised. This is useful for adjusting campaign goals based
        on contributor feedback or changing subnet costs.

        Parameters:
            wallet: Bittensor wallet instance used to sign the transaction (must be the creator's coldkey).
            crowdloan_id: The unique identifier of the crowdloan to update.
            new_cap: The new fundraising cap (TAO). Must be `>= raised`.
            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after submission.
            raise_error: If `True`, raises an exception rather than returning failure in the response.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            `ExtrinsicResponse` indicating success or failure, with error details if applicable.

        Notes:
            - Only the creator can update the cap.
            - The crowdloan must not be finalized.
            - The new cap must be `>=` the total funds already raised.

            - Crowdloans Overview: <https://docs.learnbittensor.org/subnets/crowdloans>
            - Update Parameters: <https://docs.learnbittensor.org/subnets/crowdloans#crowdloan-lifecycle>
        """
        return await update_cap_crowdloan_extrinsic(
            subtensor=self,
            wallet=wallet,
            crowdloan_id=crowdloan_id,
            new_cap=new_cap,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def update_end_crowdloan(
        self,
        wallet: "Wallet",
        crowdloan_id: int,
        new_end: int,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """Updates the end block of an active (non-finalized) crowdloan.

        Allows the creator to extend (or shorten) the contribution period before finalization. The new end block
        must be in the future and respect the minimum and maximum duration bounds defined in the runtime constants.
        This is useful for extending campaigns that need more time to reach their cap or shortening campaigns with
        sufficient contributions.

        Parameters:
            wallet: Bittensor wallet instance used to sign the transaction (must be the creator's coldkey).
            crowdloan_id: The unique identifier of the crowdloan to update.
            new_end: The new block number at which the crowdloan will end. Must be between `MinimumBlockDuration`
                (7 days = 50,400 blocks) and `MaximumBlockDuration` (60 days = 432,000 blocks) from the current block.
            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after submission.
            raise_error: If `True`, raises an exception rather than returning failure in the response.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            `ExtrinsicResponse` indicating success or failure, with error details if applicable.

        Notes:
            - Only the creator can update the end block.
            - The crowdloan must not be finalized.
            - The new end block must respect duration bounds (`MinimumBlockDuration` to `MaximumBlockDuration`).

            - Crowdloans Overview: <https://docs.learnbittensor.org/subnets/crowdloans>
            - Update Parameters: <https://docs.learnbittensor.org/subnets/crowdloans#crowdloan-lifecycle>
        """
        return await update_end_crowdloan_extrinsic(
            subtensor=self,
            wallet=wallet,
            crowdloan_id=crowdloan_id,
            new_end=new_end,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def update_min_contribution_crowdloan(
        self,
        wallet: "Wallet",
        crowdloan_id: int,
        new_min_contribution: "Balance",
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """Updates the minimum contribution amount of an active (non-finalized) crowdloan.

        Allows the creator to adjust the minimum per-contribution amount before finalization. The new value must
        meet or exceed the `AbsoluteMinimumContribution` constant. This is useful for adjusting contribution
        requirements based on the number of expected contributors or campaign strategy.

        Parameters:
            wallet: Bittensor wallet instance used to sign the transaction (must be the creator's coldkey).
            crowdloan_id: The unique identifier of the crowdloan to update.
            new_min_contribution: The new minimum contribution amount (TAO). Must be `>= AbsoluteMinimumContribution`.
            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after submission.
            raise_error: If `True`, raises an exception rather than returning failure in the response.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            `ExtrinsicResponse` indicating success or failure, with error details if applicable.

        Notes:
            - Only the creator can update the minimum contribution.
            - The crowdloan must not be finalized.
            - The new minimum must be `>= AbsoluteMinimumContribution` (check via `get_crowdloan_constants`).

            - Crowdloans Overview: <https://docs.learnbittensor.org/subnets/crowdloans>
            - Update Parameters: <https://docs.learnbittensor.org/subnets/crowdloans#crowdloan-lifecycle>
        """
        return await update_min_contribution_crowdloan_extrinsic(
            subtensor=self,
            wallet=wallet,
            crowdloan_id=crowdloan_id,
            new_min_contribution=new_min_contribution,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )

    async def withdraw_crowdloan(
        self,
        wallet: "Wallet",
        crowdloan_id: int,
        *,
        mev_protection: bool = DEFAULT_MEV_PROTECTION,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
        wait_for_revealed_execution: bool = True,
    ) -> ExtrinsicResponse:
        """Withdraws a contribution from an active (not yet finalized or dissolved) crowdloan.

        Contributors can withdraw their contributions at any time before finalization. For regular contributors,
        the full contribution amount is returned. For the creator, only amounts exceeding the initial deposit can
        be withdrawn; the deposit itself remains locked until dissolution.

        Parameters:
            wallet: Bittensor wallet instance used to sign the transaction (coldkey must match a contributor).
            crowdloan_id: The unique identifier of the crowdloan to withdraw from.
            mev_protection: If `True`, encrypts and submits the transaction through the MEV Shield pallet to protect
                against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
                decrypt and execute it. If `False`, submits the transaction directly without encryption.
            period: The number of blocks during which the transaction will remain valid after submission, after which
                it will be rejected.
            raise_error: If `True`, raises an exception rather than returning False in the response, in case the
               transaction fails.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.
            wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

        Returns:
            `ExtrinsicResponse` indicating success or failure, with error details if applicable.

        Notes:

            - Crowdloans Overview: <https://docs.learnbittensor.org/subnets/crowdloans>
            - Crowdloan Lifecycle: <https://docs.learnbittensor.org/subnets/crowdloans#crowdloan-lifecycle>
            - Withdraw: <https://docs.learnbittensor.org/subnets/crowdloans/crowdloans-tutorial#optional-withdraw>
        """
        return await withdraw_crowdloan_extrinsic(
            subtensor=self,
            wallet=wallet,
            crowdloan_id=crowdloan_id,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )


async def get_async_subtensor(
    network: Optional[str] = None,
    config: Optional["Config"] = None,
    mock: bool = False,
    log_verbose: bool = False,
) -> "AsyncSubtensor":
    """Factory method to create an initialized AsyncSubtensor instance.

    This function creates an AsyncSubtensor instance and automatically initializes the connection to the blockchain.
    This is useful when you don't want to manually call `await subtensor.initialize()` after instantiation.

    Parameters:
        network: The network name to connect to (e.g., `finney` for Bittensor mainnet, `test` for test network,
            `local` for a locally deployed blockchain). If `None`, uses the default network from config.
        config: Configuration object for the AsyncSubtensor instance. If `None`, uses the default configuration.
        mock: Whether this is a mock instance. FOR TESTING ONLY.
        log_verbose: Enables or disables verbose logging.

    Returns:
        An initialized AsyncSubtensor instance ready for use.

    Example:

        # Create and initialize in one step
        subtensor = await get_async_subtensor(network="finney")
        # Ready to use immediately
        block = await subtensor.get_current_block()

    """
    sub = AsyncSubtensor(
        network=network, config=config, mock=mock, log_verbose=log_verbose
    )
    await sub.initialize()
    return sub
