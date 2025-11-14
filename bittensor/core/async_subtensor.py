import asyncio
import copy
import ssl
from datetime import datetime, timezone
from typing import cast, Optional, Any, Union, Iterable, TYPE_CHECKING, Literal

import asyncstdlib as a
import scalecodec
from async_substrate_interface import AsyncSubstrateInterface
from async_substrate_interface.substrate_addons import RetryAsyncSubstrate
from async_substrate_interface.utils.storage import StorageKey
from bittensor_drand import get_encrypted_commitment
from bittensor_wallet.utils import SS58_FORMAT
from scalecodec import GenericCall

from bittensor.core.chain_data import (
    CrowdloanInfo,
    CrowdloanConstants,
    DelegateInfo,
    DynamicInfo,
    MetagraphInfo,
    NeuronInfoLite,
    NeuronInfo,
    ProposalVoteData,
    ProxyAnnouncementInfo,
    ProxyInfo,
    ProxyConstants,
    ProxyType,
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
from bittensor.core.extrinsics.asyncex.move_stake import (
    transfer_stake_extrinsic,
    swap_stake_extrinsic,
    move_stake_extrinsic,
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
    remove_proxy_extrinsic,
    remove_proxies_extrinsic,
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
)
from bittensor.core.extrinsics.asyncex.serving import serve_axon_extrinsic
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
    version_as_int,
    DEFAULT_PERIOD,
    TYPE_REGISTRY,
    TAO_APP_BLOCK_EXPLORER,
)
from bittensor.core.types import (
    BlockInfo,
    ExtrinsicResponse,
    ParamWithTypes,
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
    fixed_to_float,
    check_balance_amount,
)
from bittensor.utils.btlogging import logging
from bittensor.utils.liquidity import (
    calculate_fees,
    get_fees,
    tick_to_price,
    price_to_tick,
    LiquidityPosition,
)

if TYPE_CHECKING:
    from async_substrate_interface.types import ScaleObj
    from bittensor_wallet import Keypair, Wallet
    from bittensor.core.axon import Axon
    from async_substrate_interface import AsyncQueryMapResult


class AsyncSubtensor(SubtensorMixin):
    """Asynchronous interface for interacting with the Bittensor blockchain.

    This class provides a thin layer over the Substrate Interface offering async functionality for Bittensor. This
    includes frequently-used calls for querying blockchain data, managing stakes and liquidity positions, registering
    neurons, submitting weights, and many other functions for participating in Bittensor.

    Notes:
        Key Bittensor concepts used throughout this class:

        - **Coldkey**: The key pair corresponding to a user's overall wallet. Used to transfer, stake, manage subnets.
        - **Hotkey**: A key pair (each wallet may have zero, one, or more) used for neuron operations (mining and validation).
        - **Netuid**: Unique identifier for a subnet (0 is the Root Subnet)
        - **UID**: Unique identifier for a neuron registered to a hotkey on a specific subnet.
        - **Metagraph**: Data structure containing the complete state of a subnet at a block.        
        - **TAO**: The base network token; subnet 0 stake is in TAO
        - **Alpha**: Subnet-specific token representing some quantity of TAO staked into a subnet.
        - **Rao**: Smallest unit of TAO (1 TAO = 1e9 Rao)

        See: Bittensor Glossary <https://docs.learnbittensor.org/glossary>
        See: Wallets, Coldkeys and Hotkeys in Bittensor <https://docs.learnbittensor.org/keys/wallets>

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
            network: The network name to connect to (e.g., ``"finney"`` for Bittensor mainnet, ``"test"``, for
                Bittensor test network, ``"local"`` for a locally deployed blockchain).If ``None``, uses the
                default network from config.
            config: Configuration object for the AsyncSubtensor instance. If ``None``, uses the default configuration.
            log_verbose: Enables or disables verbose logging.
            fallback_endpoints: List of fallback WebSocket endpoints to use if the primary network endpoint is
                unavailable. These are tried in order when the default endpoint fails.
            retry_forever: Whether to retry connection attempts indefinitely on connection errors.
            mock: Whether this is a mock instance. FOR TESTING ONLY.
            archive_endpoints: List of archive node endpoints for queries requiring historical block data beyond the
                retention period of lite nodes. These are only used when requesting blocks that the current node is
                unable to serve.
            websocket_shutdown_timer: Amount of time (in seconds) to wait after the last response from the chain before
                automatically closing the WebSocket connection. Pass ``None`` to disable automatic shutdown entirely.

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

        Example::
            
            sub = bt.AsyncSubtensor(network="finney")
            # Initialize the connection
            await subtensor.initialize()
            # calls to subtensor
            await subtensor.close()

        """
        if self.substrate:
            await self.substrate.close()

    async def initialize(self):
        """
         connection to the blockchain.

        This method establishes the connection to the Bittensor blockchain and should be called after creating an
        AsyncSubtensor instance before making any queries.

        When using the ``async with`` context manager, this method is called automatically and does not need to be
        invoked explicitly.

        Returns:
            AsyncSubtensor: The initialized instance (self) for method chaining.


        Example::

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

        When ``fallback_endpoints``, ``archive_endpoints``, or ``retry_forever`` are provided, a RetryAsyncSubstrate
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
        """Provides an asynchronous getter to retrieve the current block."""
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
            1. If ``reuse_block=True`` and ``block`` or ``block_hash`` is set → raises ValueError
            2. If both ``block`` and ``block_hash`` are set → validates they match, raises ValueError if not
            3. If only ``block_hash`` is set → returns it directly
            4. If only ``block`` is set → fetches and returns its hash
            5. If none are set → returns None

        Parameters:
            block: The block number to get the hash for. If specifying along with ``block_hash``, the hash of ``block``
                will be checked and compared with the supplied block hash, raising a ValueError if the two do not match.
            block_hash: The hash of the blockchain block (hex string prefixed with ``0x``). If specifying along with
                ``block``, the hash of ``block`` will be checked and compared with the supplied block hash, raising a
                ValueError if the two do not match.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using ``block`` or ``block_hash``.

        Returns:
            The block hash (hex string with ``0x`` prefix) if one can be determined, ``None`` otherwise.

        Notes:
            See: <https://docs.learnbittensor.org/glossary#block>
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

    async def encode_params(
        self,
        call_definition: dict[str, list["ParamWithTypes"]],
        params: Union[list[Any], dict[str, Any]],
    ) -> str:
        """Deprecated
        """
        param_data = scalecodec.ScaleBytes(b"")

        for i, param in enumerate(call_definition["params"]):
            scale_obj = await self.substrate.create_scale_object(param["type"])
            if isinstance(params, list):
                param_data += scale_obj.encode(params[i])
            else:
                if param["name"] not in params:
                    raise ValueError(f"Missing param {param['name']} in params dict.")

                param_data += scale_obj.encode(params[param["name"]])

        return param_data.to_hex()

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
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            The value of the specified hyperparameter if the subnet exists, ``None`` otherwise. Return type varies
            by parameter (int, float, bool, or Balance).

        Notes:
            See: <https://docs.learnbittensor.org/subnets/subnet-hyperparameters>
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
            origin_netuid: Netuid of the source subnet (0 for root/TAO staking).
            destination_netuid: Netuid of the destination subnet.
            amount: Amount to swap/stake as a Balance object. Use ``Balance.from_tao(...)`` or
             ``Balance.from_rao(...)`` to create the amount.
            block_hash: The hash of the blockchain block for the query. If ``None``, uses the current chain head.

        Returns:
            SimSwapResult: Object containing ``alpha_fee``, ``tao_fee``, ``alpha_amount``, and ``tao_amount`` fields
            representing the swap fees and output amounts.

        Example::

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

            See:
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
            module_name: The name of the module containing the constant (e.g., ``"Balances"``, ``"SubtensorModule"``).
            constant_name: The name of the constant to retrieve (e.g., ``"ExistentialDeposit"``).
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            A SCALE-decoded object if found, ``None`` otherwise. Access the actual value using ``.value`` attribute.
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

        Use this function for nonstandard queries to constants defined within the Bittensor blockchain, if these cannot
        be accessed through other, standard getter methods.

        Parameters:
            module: The name of the module from which to query the map storage (e.g., "SubtensorModule", "System").
            name: The specific storage function within the module to query (e.g., "Bonds", "Weights").
            params: Parameters to be passed to the query.
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

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

        Use this function for nonstandard queries to constants defined within the Bittensor blockchain, if these cannot
        be accessed through other, standard getter methods.

        Parameters:
            name: The name of the map storage function to query.
            params: A list of parameters to pass to the query function.
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            An object containing the map-like data structure, or ``None`` if not found.        
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
        blockchain modules. Use this function for nonstandard queries to constants defined within the Bittensor
        blockchain, if these cannot be accessed through other, standard getter methods.

        Parameters:
            module: The name of the module from which to query data.
            name: The name of the storage function within the module.
            params: A list of parameters to pass to the query function.
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            An object containing the requested data if found, ``None`` otherwise.

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
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            The decoded result from the runtime API call, or ``None`` if the call fails.

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

        Use this function for nonstandard queries to constants defined within the Bittensor blockchain, if these cannot
        be accessed through other, standard getter methods.

        Parameters:
            name: The name of the storage function to query.
            params: A list of parameters to pass to the query function.
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

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

        Parameters:
            method: The method name for the state call.
            data: The data to be passed to the method.
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

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
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            Optional[list[DynamicInfo]]: A list of ``DynamicInfo`` objects, each containing detailed information about
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
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            The number of blocks since the last step in the subnet, or None if the query fails.

        # TODO glossary link for epoch
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
        """Returns the number of blocks since the last update, or ``None`` if the subnetwork or UID does not exist.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            uid: The unique identifier of the neuron.
            block: The block number for this query. Do not specify if using block_hash or reuse_block.
            block_hash: The hash of the block for the query. Do not specify if using reuse_block or block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

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
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            List of tuples, where each tuple contains:
                - validator_uid: The UID of the validator
                - bonds: List of (miner_uid, bond_value) pairs
            
            Bond values are u16-normalized (0-65535, where 65535 = 1.0 or 100%).

        Example::

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
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            True if commit-reveal mechanism is enabled, False otherwise.

        Example::

            # Check if commit-reveal is enabled for subnet 1
            enabled = await subtensor.commit_reveal_enabled(netuid=1)

            # Check at specific block
            enabled = await subtensor.commit_reveal_enabled(netuid=1, block=1000000)

        Notes:
            See also: <https://docs.learnbittensor.org/glossary#commit-reveal>
            See: <https://docs.learnbittensor.org/subnets/subnet-hyperparameters>            
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
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            The value of the 'Difficulty' hyperparameter if the subnet exists, None otherwise.

        Example::

            # Get difficulty for subnet 1
            difficulty = await subtensor.difficulty(netuid=1)

            # Get difficulty at specific block
            difficulty = await subtensor.difficulty(netuid=1, block=1000000)

        Notes:
            Burn registration is much more common on Bittensor subnets currently, compared to POW registration.

            See also:
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

        The Owner storage map defaults to the zero address (``5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM``) 
        for unused hotkeys. This method returns True if the Owner value is anything other than this default.

        Parameters:
            hotkey_ss58: The SS58 address of the hotkey.
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            True if the hotkey has been associated with a coldkey, False otherwise.

        Example::

            # Check if hotkey exists
            exists = await subtensor.does_hotkey_exist(hotkey_ss58="5F...")

            # Check at specific block
            exists = await subtensor.does_hotkey_exist(hotkey_ss58="5F...", block=1000000)

        Notes:
            See: <https://docs.learnbittensor.org/glossary#hotkey>
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
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            The number of blocks in the administrative freeze window (default: 10 blocks, ~2 minutes).

        Notes:
            See: <https://docs.learnbittensor.org/learn/chain-rate-limits#administrative-freeze-window>
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
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

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
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            A mapping of the ss58:commitment with the commitment as a string.

        Example::

            # Get all commitments for subnet 1
            commitments = await subtensor.get_all_commitments(netuid=1)

            # Iterate over all commitments
            for hotkey, commitment in commitments.items():
                print(f"Hotkey {hotkey}: {commitment}")
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
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            Dict mapping netuid to (last_updated_block, ema_flow). The Balance represents the EMA of net TAO flow
            in TAO units. Positive values indicate sustained net inflow, negative values indicate sustained net outflow.

        The EMA uses a smoothing factor α ≈ 0.000003209, creating a 30-day half-life and ~86.8 day window. Only direct
        stake/unstake operations count toward flows; neuron registrations and root claims are excluded. Subnet 0 (root
        network) does not have an EMA TAO flow value.

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
            See also: See <https://docs.learnbittensor.org/glossary#metagraph>
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
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using block or
                reuse_block.
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            {ss58: Certificate} for the key/Certificate pairs on the subnet

        This function is used for certificate discovery for setting up mutual tls communication between neurons.
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
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            A dictionary mapping hotkey addresses to tuples of (reveal_block, commitment_message) pairs.
            Each validator can have multiple revealed commitments (up to 10 most recent).

        Example::
            
            {
                "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY": ( (12, "Alice message 1"), (152, "Alice message 2") ),
                "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty": ( (12, "Bob message 1"), (147, "Bob message 2") ),
            }

        Notes:
            See also:
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
            See: <https://docs.learnbittensor.org/miners/autostaking>
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
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            Balance: The balance object containing the account's TAO balance.

        Example::

            # Get balance for an address
            balance = await subtensor.get_balance(address="5F...")
            print(f"Balance: {balance.tao} TAO")

            # Get balance at specific block
            balance = await subtensor.get_balance(address="5F...", block=1000000)
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
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            A dictionary mapping each address to its Balance object.

        Example::

            # Get balances for multiple addresses
            balances = await subtensor.get_balances("5F...", "5G...", "5H...")
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

        This function provides the latest block number, indicating the most recent state of the blockchain. Knowing
        the current block number is essential for querying real-time data and performing time-sensitive operations on
        the blockchain. It serves as a reference point for network activities and data synchronization.

        Returns:
            int: The current chain block number.

        Example::

            # Get current block number
            current_block = await subtensor.get_current_block()
            print(f"Current block: {current_block}")

            block = await subtensor.get_current_block()
            if block > 1000000:
                print("Network has progressed past block 1M")

        Notes:
            See also: <https://docs.learnbittensor.org/glossary#block>
        """
        return await self.substrate.get_block_number(None)

    async def get_block_hash(self, block: Optional[int] = None) -> str:
        """Retrieves the hash of a specific block on the Bittensor blockchain.

        The block hash is a unique identifier representing the cryptographic hash of the block's content, ensuring its
        integrity and immutability. It is a fundamental aspect of blockchain technology, providing a secure reference
        to each block's data. It is crucial for verifying transactions, ensuring data consistency, and maintaining the
        trustworthiness of the blockchain.

        Parameters:
            block: The block number for which the hash is to be retrieved. If ``None``, returns the latest block hash.

        Returns:
            str: The cryptographic hash of the specified block.

        Example::

            # Get hash for specific block
            block_hash = await subtensor.get_block_hash(block=1000000)
            print(f"Block 1000000 hash: {block_hash}")

            # Get latest block hash
            latest_hash = await subtensor.get_block_hash()
            print(f"Latest block hash: {latest_hash}")

        Notes:
            See also: <https://docs.learnbittensor.org/glossary#block>
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
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            A tuple containing a boolean indicating success or failure, a list of formatted children with their
                proportions, and an error message (if applicable).

        Example::

            # Get children for a hotkey in subnet 1
            success, children, error = await subtensor.get_children(hotkey="5F...", netuid=1)

            if success:
                for proportion, child_hotkey in children:
                    print(f"Child {child_hotkey}: {proportion}")

        Notes:
            See: <https://docs.learnbittensor.org/validators/child-hotkeys>
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
            See: <https://docs.learnbittensor.org/validators/child-hotkeys>
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
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            The commitment data as a string.

        Example::

            # Get commitment for UID 5 in subnet 1
            commitment = await subtensor.get_commitment(netuid=1, uid=5)
            print(f"Commitment: {commitment}")

            # Get commitment at specific block
            commitment = await subtensor.get_commitment(
                netuid=1,
                uid=5,
                block=1000000
            )

        Notes:
            See: <https://docs.learnbittensor.org/glossary#commit-reveal>
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
        # TODO: State whether this returns raw SCALE-encoded metadata vs decoded dict, plus examples for both cases.
        """Fetches raw commitment metadata from specific subnet for given hotkey.

        Parameters:
            netuid: The unique subnet identifier.
            hotkey_ss58: The hotkey ss58 address.
            block: The blockchain block number for the query.
            block_hash: The hash of the block at which to check the hotkey ownership.
            reuse_block: Whether to reuse the last-used blockchain hash.

        Returns:
            The raw commitment metadata from specific subnet for given hotkey.
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
        """
        Fetches runtime configuration constants from the `Crowdloan` pallet.

        If a list of constant names is provided, only those constants will be queried.
        Otherwise, all known constants defined in `CrowdloanConstants.field_names()` are fetched.

        Parameters:
            constants: A list of specific constant names to fetch from the pallet. If omitted, all constants from
                `CrowdloanConstants` are queried.
            block: The blockchain block number for the query.
            block_hash: The hash of the block at which to check the hotkey ownership.
            reuse_block: Whether to reuse the last-used blockchain hash.

        Returns:
            CrowdloanConstants:
                A structured dataclass containing the retrieved values. Missing constants are returned as `None`.

        Example::

            print(subtensor.get_crowdloan_constants())
            CrowdloanConstants(
                AbsoluteMinimumContribution=τ1.000000000,
                MaxContributors=1000,
                MaximumBlockDuration=86400,
                MinimumDeposit=τ10.000000000,
                MinimumBlockDuration=600,
                RefundContributorsLimit=50
            )

            crowdloan_consts = subtensor.get_crowdloan_constants(
                constants=["MaxContributors", "RefundContributorsLimit"]
            )
            print(crowdloan_consts)
            CrowdloanConstants(MaxContributors=1000, RefundContributorsLimit=50)

            print(crowdloan_consts.MaxContributors)
            1000
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
        """
        Returns a mapping of contributor SS58 addresses to their contribution amounts for a specific crowdloan.

        Parameters:
            crowdloan_id: The unique identifier of the crowdloan.
            block: The blockchain block number for the query.
            block_hash: The hash of the block at which to check the hotkey ownership.
            reuse_block: Whether to reuse the last-used blockchain hash.

        Returns:
            Dict[address -> Balance].
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
        """
        Returns detailed information about a specific crowdloan by ID.

        Parameters:
            crowdloan_id: Unique identifier of the crowdloan.
            block: The blockchain block number for the query.
            block_hash: The hash of the block at which to check the hotkey ownership.
            reuse_block: Whether to reuse the last-used blockchain hash.

        Returns:
            CrowdloanInfo if found, else None.
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
        """
        Returns the next available crowdloan ID (auto-increment value).

        Parameters:
            block: The blockchain block number for the query.
            block_hash: The hash of the block at which to check the hotkey ownership.
            reuse_block: Whether to reuse the last-used blockchain hash.

        Returns:
            The next crowdloan ID to be used when creating a new campaign.
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
        """Returns a list of all existing crowdloans with their metadata.

        Parameters:
            block: The blockchain block number for the query.
            block_hash: The hash of the block at which to check the hotkey ownership.
            reuse_block: Whether to reuse the last-used blockchain hash.

        Returns:
            List of CrowdloanInfo which contains (id, creator, cap, raised, end, finalized, etc.)

        Notes:
            See: <https://docs.learnbittensor.org/subnets/crowdloans>
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
            hotkey_ss58: The ``SS58`` address of the delegate's hotkey.
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            Detailed information about the delegate neuron, ``None`` if not found.

        Notes:
            This function is essential for understanding the roles and influence of delegate neurons within the Bittensor
            network's consensus and governance structures.
            
            See also:
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
        """
        Fetches delegates identities from the chain.

        Parameters:
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            Dict {ss58: ChainIdentity, ...}
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
            hotkey_ss58: The ``SS58`` address of the neuron's hotkey.
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            float: The delegate take percentage.

        The delegate take is a critical parameter in the network's incentive structure, influencing the distribution of
        rewards among neurons and their nominators.
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
        """
        Retrieves a list of delegates and their associated stakes for a given coldkey. This function identifies the
        delegates that a specific account has staked tokens on.

        Parameters:
            coldkey_ss58: The ``SS58`` address of the account's coldkey.
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            A list containing the delegated information for the specified coldkey.

        This function is important for account holders to understand their stake allocations and their involvement in
        the network's delegation and consensus mechanisms.
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
        """
        Fetches all delegates on the chain

        Parameters:
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            List of DelegateInfo objects, or an empty list if there are no delegates.
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
        """
        Retrieves the existential deposit amount for the Bittensor blockchain.
        The existential deposit is the minimum amount of TAO required for an account to exist on the blockchain.
        Accounts with balances below this threshold can be reaped to conserve network resources.

        Parameters:
            block: The blockchain block number for the query.
            block_hash: Block hash at which to query the deposit amount. If ``None``, the current block is used.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            The existential deposit amount.

        The existential deposit is a fundamental economic parameter in the Bittensor network, ensuring efficient use of
        storage and preventing the proliferation of dust accounts.
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
            block: The block number to query. If None, uses latest finalized block.

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
        specified block hash, it returns None.

        Parameters:
            hotkey_ss58: The SS58 address of the hotkey.
            block: The blockchain block number for the query.
            block_hash: The hash of the block at which to check the hotkey ownership.
            reuse_block: Whether to reuse the last-used blockchain hash.

        Returns:
            Optional[str]: The SS58 address of the owner if the hotkey exists, or None if it doesn't.

        Notes:
            See also:
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
    ) -> bytes:
        """
        Retrieves the last bonds reset triggered at commitment from given subnet for a specific hotkey.

        Parameters:
            netuid: The network uid to fetch from.
            hotkey_ss58: The hotkey of the neuron for which to fetch the last bonds reset.
            block: The block number to query.
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using block or reuse_block.
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            bytes: The last bonds reset data for the specified hotkey and netuid.
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
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using block or reuse_block.
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

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
            See also:
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
            block_hash: The hash of the block to retrieve the stake from. Do not specify if using block or reuse_block.
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            A list of integers representing the percentage of emission allocated to each subnet mechanism (rounded to
            whole numbers). Returns None if emission is evenly split or if the data is unavailable.

        Notes:
            See: <https://docs.learnbittensor.org/subnets/understanding-multiple-mech-subnets>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        result = await self.substrate.query(
            module="SubtensorModule",
            storage_function="MechanismEmissionSplit",
            params=[netuid],
            block_hash=block_hash,
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
            block_hash: The hash of the block to retrieve the stake from. Do not specify if using block or reuse_block.
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            The number of mechanisms for the given subnet.

        Notes:
            See: <https://docs.learnbittensor.org/subnets/understanding-multiple-mech-subnets>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        query = await self.substrate.query(
            module="SubtensorModule",
            storage_function="MechanismCountCurrent",
            params=[netuid],
            block_hash=block_hash,
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

        Example::

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
            See also:
            - <https://docs.learnbittensor.org/subnets/metagraph>

        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        if not block_hash and reuse_block:
            block_hash = self.substrate.last_block_hash

        indexes = (
            [
                f.value if isinstance(f, SelectiveMetagraphIndex) else f
                for f in selected_indices
            ]
            if selected_indices is not None
            else [f for f in range(len(SelectiveMetagraphIndex))]
        )

        query = await self.substrate.runtime_call(
            api="SubnetInfoRuntimeApi",
            method="get_selective_mechagraph",
            params=[netuid, mechid, indexes if 0 in indexes else [0] + indexes],
            block_hash=block_hash,
        )
        if getattr(query, "value", None) is None:
            logging.error(
                f"Subnet mechanism {netuid}.{mechid if mechid else 0} does not exist."
            )
            return None

        return MetagraphInfo.from_dict(query.value)

    async def get_minimum_required_stake(self):
        """
        Returns the minimum required stake for nominators in the Subtensor network.

        Returns:
            Balance: The minimum required stake as a Balance object.
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
        """Retrieves a list of subnet UIDs (netuids) for which a given hotkey is a member. This function identifies the
        specific subnets within the Bittensor network where the neuron associated with the hotkey is active.

        Parameters:
            hotkey_ss58: The ``SS58`` address of the neuron's hotkey.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number at which to perform the query.
            reuse_block: Whether to reuse the last-used block hash when retrieving info.

        Returns:
            A list of netuids where the neuron is a member.

        Notes:
            See also:
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
            hotkey_ss58: The hotkey to query.
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The hash of the block to retrieve the parameter from. Do not specify if using block or
                reuse_block.
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            the certificate of the neuron if found, ``None`` otherwise.

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
            hotkey_ss58: The ``SS58`` address of the neuron's hotkey.
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The blockchain block number at which to perform the query.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            Detailed information about the neuron if found, ``None`` otherwise.

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

        If ``block`` is not provided, the current chain block will be used. Epochs are determined based on the subnet's
        tempo (i.e., blocks per epoch). The result is the block number at which the next epoch will begin.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The reference block to calculate from. If None, uses the current chain block height.
            block_hash: The blockchain block number at which to perform the query.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            int: The block number at which the next epoch will start.

        Notes:
            See also: <https://docs.learnbittensor.org/glossary#tempo>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        blocks_since_last_step = await self.blocks_since_last_step(
            netuid=netuid, block=block, block_hash=block_hash, reuse_block=reuse_block
        )
        tempo = await self.tempo(
            netuid=netuid, block=block, block_hash=block_hash, reuse_block=reuse_block
        )

        block = block or await self.substrate.get_block_number(block_hash=block_hash)
        if block and blocks_since_last_step is not None and tempo:
            return block - blocks_since_last_step + tempo + 1
        return None

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
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            A list of formatted parents [(proportion, parent)]

        Notes:
            See also:
            - <https://docs.learnbittensor.org/validators/child-hotkeys>
            - `get_children()` for retrieving child keys
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
            block: The blockchain block number for the query. If None, queries the latest block.
            block_hash: The hash of the block at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            Dictionary mapping real account SS58 addresses to lists of ProxyInfo objects. Each ProxyInfo contains the
                delegate address, proxy type, and delay for that proxy relationship.

        Note:
            This method queries all proxy relationships on the chain, which may be resource-intensive for large
            networks. Consider using `get_proxies_for_real_account()` for querying specific accounts.
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
            block_hash: The hash of the block at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            Tuple containing:
                - List of ProxyInfo objects representing all proxy relationships for the real account. Each ProxyInfo
                    contains delegate address, proxy type, and delay.
                - Balance object representing the reserved deposit amount for these proxies. This deposit is held as
                    long as the proxy relationships exist and is returned when proxies are removed.

        Note:
            If the account has no proxies, returns an empty list and a zero balance.
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
            block: The blockchain block number for the query. If None, queries the latest block.
            block_hash: The hash of the block at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            List of ProxyAnnouncementInfo objects. Each object contains the real account address, call hash, and block
                height at which the announcement was made.

        Note:
            If the delegate has no announcements, returns an empty list.
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
            block: The blockchain block number for the query. If None, queries the latest block.
            block_hash: The hash of the block at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            Dictionary mapping delegate account SS58 addresses to lists of ProxyAnnouncementInfo objects.
            Each ProxyAnnouncementInfo contains the real account address, call hash, and block height.

        Note:
            This method queries all announcements on the chain, which may be resource-intensive for large networks.
            Consider using `get_proxy_announcement()` for querying specific delegates.
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
            block: The blockchain block number for the query. If None, queries the latest block.
            block_hash: The hash of the block at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            If `as_dict` is False: ProxyConstants object containing all requested constants.
            If `as_dict` is True: Dictionary mapping constant names to their values (Balance objects for deposit
                constants, integers for limit constants).

        Note:
            All Balance amounts are returned in RAO. Constants reflect the current chain configuration at the specified
            block.
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

        Example::
            
            ( (12, "Alice message 1"), (152, "Alice message 2") )
            ( (12, "Bob message 1"), (147, "Bob message 2") )

        Notes:
            See: <https://docs.learnbittensor.org/glossary#commit-reveal>
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
        # TODO: Clarify return ordering and units; add Examples
        """Retrieves hotkey related revealed commitment for a given subnet.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            hotkey_ss58: The ss58 address of the committee member.
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            A tuple of reveal block and commitment message.

        Notes:
            See: <https://docs.learnbittensor.org/glossary#commit-reveal>
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
    ) -> str:
        """Retrieves the root claim type for a given coldkey address.

        The root claim type determines how dividends from staking to the ROOT network (subnet 0) are handled:
        - "Swap": Future Root Alpha Emissions are swapped to TAO at claim time and added to your root stake
        - "Keep": Future Root Alpha Emissions are kept as Alpha

        Parameters:
            coldkey_ss58: The ss58 address of the coldkey.
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            RootClaimType value in string representation. Could be `Swap` or `Keep`.

        Notes:
            See: <https://docs.learnbittensor.org/staking-and-delegation/root-claims>
        """
        block_hash = await self.determine_block_hash(block, block_hash, reuse_block)
        query = await self.substrate.query(
            module="SubtensorModule",
            storage_function="RootClaimType",
            params=[coldkey_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        return next(iter(query.keys()))

    async def get_root_claimable_rate(
        self,
        hotkey_ss58: str,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> float:
        """Retrieves the root claimable rate from a given hotkey address for provided netuid.

        Parameters:
            hotkey_ss58: The ss58 address of the root validator.
            netuid: The unique identifier of the subnet to get the rate.
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            The rate of claimable stake from validator's hotkey ss58 address for provided subnet.
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
            hotkey_ss58: The ss58 address of the root validator.
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            The rate of claimable stake from validator's hotkey ss58 address for provided subnet.
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
        """
        Retrieves the root claimable stake for a given coldkey address.

        Parameters:
            coldkey_ss58: Delegate's ColdKey ss58 address.
            hotkey_ss58: The root validator hotkey ss58 address.
            netuid: Delegate's netuid where stake will be claimed.
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            Available for claiming root stake.

        Note:
            After manual claim, claimable (available) stake will be added to subtends stake.
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
        """Retrieves the root claimed Alpha shares for coldkey from hotkey in provided subnet.

        Parameters:
            coldkey_ss58: The ss58 address of the staker.
            hotkey_ss58: The ss58 address of the root validator.
            netuid: The unique identifier of the subnet.
            block: The block number to query. Do not specify if using block_hash or reuse_block.
            block_hash: The block hash at which to check the parameter. Do not set if using block or reuse_block.
            reuse_block: Whether to reuse the last-used block hash. Do not set if using block_hash or block.

        Returns:
            The number of Alpha stake claimed from the root validator in Rao.
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
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

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
            See: <https://docs.learnbittensor.org/learn/fees>
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
            See: <https://docs.learnbittensor.org/learn/fees>
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
            netuids: The subnet IDs to query for. Set to ``None`` for all subnets.
            block: The block number for which the children are to be retrieved.
            block_hash: The hash of the block to retrieve the subnet unique identifiers from.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            A {netuid: StakeInfo} pairing of all stakes across all subnets.
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
    ) -> Optional[list["StakeInfo"]]:
        """
        Retrieves the stake information for a given coldkey.

        Parameters:
            coldkey_ss58: The SS58 address of the coldkey.
            block: The block number at which to query the stake information.
            block_hash: The hash of the blockchain block number for the query.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            An optional list of StakeInfo objects, or ``None`` if no stake information is found.
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

    async def get_stake_for_hotkey(
        self,
        hotkey_ss58: str,
        netuid: int,
        block: Optional[int] = None,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Balance:
        """
        Retrieves the stake information for a given hotkey.

        Parameters:
            hotkey_ss58: The SS58 address of the hotkey.
            netuid: The subnet ID to query for.
            block: The block number for which the children are to be retrieved.
            block_hash: The hash of the block to retrieve the subnet unique identifiers from.
            reuse_block: Whether to reuse the last-used block hash.
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
            The subnet's hyperparameters, or ``None`` if not available.

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
        netuid. If no data is found or the query fails, the function returns None.

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
            block_hash: The hash of the block to retrieve the stake from. Do not specify if using block or reuse_block.
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            The current Alpha price in TAO units for the specified subnet.
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
        """Gets the current Alpha price in TAO for a specified subnet.

        Parameters:
            block: The block number for which the children are to be retrieved.
            block_hash: The hash of the block to retrieve the subnet unique identifiers from.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            A dictionary mapping subnet unique ID to the current Alpha price in TAO units.
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
        """Retrieve the SubnetRevealPeriodEpochs hyperparameter."""
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
        """
        # TODO: Expand CRv4 acronym in a Notes section and link to commit-reveal v4 docs; define 'reveal_round'.
        # TODO: Clarify byte/vector encoding -> parsed structure conversion via `WeightCommitInfo`.
        Retrieves CRv4 weight commit information for a specific subnet.

        Parameters:
            netuid: Subnet identifier.
            mechid: Subnet mechanism identifier.
            block: The blockchain block number for the query.
            block_hash: The hash of the block to retrieve the stake from. Do not specify if using block or reuse_block.
            reuse_block: Whether to use the last-used block. Do not set if using block_hash or block.

        Returns:
            A list of commit details, where each item contains:

                - ss58_address: The address of the committer.
                - commit_block: The block number when the commitment was made.
                - commit_message: The commit message.
                - reveal_round: The round when the commitment was revealed.

            The list may be empty if there are no commits found.
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
        """
        # TODO: Link to 'Timestamp' pallet docs and note millisecond to seconds conversion (unix/1000).
        Retrieves the datetime timestamp for a given block.

        Parameters:
            block: The blockchain block number for the query.
            block_hash: The blockchain block_hash representation of the block id.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            datetime object for the timestamp of the block.
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
        """
        Retrieves the total number of subnets within the Bittensor network as of a specific blockchain block.

        Parameters:
            block: The blockchain block number for the query.
            block_hash: The blockchain block_hash representation of block id.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            The total number of subnets in the network.

        Understanding the total number of subnets is essential for assessing the network's growth and the extent of its
        decentralized infrastructure.
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
            destination_ss58: The ``SS58`` address of the destination account.
            amount: The amount of tokens to be transferred, specified as a Balance object, or in Tao (float) or Rao
                (int) units.
            keep_alive: Whether the transfer fee should be calculated based on keeping the wallet alive (existential
                deposit) or not.

        Returns:
            bittensor.utils.balance.Balance: The estimated transaction fee for the transfer, represented as a Balance
                object.

        Notes:
            Estimating the transfer fee is essential for planning and executing token transactions, ensuring that the
            wallet has sufficient funds to cover both the transfer amount and the associated costs. This function provides
            a crucial tool for managing financial operations within the Bittensor network.
            
            See: <https://docs.learnbittensor.org/learn/fees>
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
        """
        Calculates the fee for unstaking from a hotkey.

        Parameters:
            netuid: The unique identifier of the subnet.
            amount: Amount of stake to unstake in TAO.
            block: The blockchain block number for the query.
            block_hash: The blockchain block_hash representation of the block id.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            The calculated stake fee as a Balance object in Alpha.

        Notes:
            See: <https://docs.learnbittensor.org/learn/fees>
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
        # TODO: Add Examples; specify `proposal_hash` format; link to governance docs; Raises for not found.
        """
        Retrieves the voting data for a specific proposal on the Bittensor blockchain. This data includes information
        about how senate members have voted on the proposal.

        Parameters:
            proposal_hash: The hash of the proposal for which voting data is requested.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number to query the voting data.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            An object containing the proposal's voting data, or ``None`` if not found.

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
            hotkey_ss58: The ``SS58`` address of the neuron's hotkey.
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The blockchain block_hash representation of the block id.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            The UID of the neuron if it is registered on the subnet, ``None`` otherwise.

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
        Filters a given list of all netuids for certain specified netuids and hotkeys

        Parameters:
            all_netuids: A list of netuids to filter.
            filter_for_netuids: A subset of all_netuids to filter from the main list.
            all_hotkeys: Hotkeys to filter from the main list.
            block: The blockchain block number for the query.
            block_hash: hash of the blockchain block number at which to perform the query.
            reuse_block: whether to reuse the last-used blockchain hash when retrieving info.

        Returns:
            The filtered list of netuids.
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
            The value of the 'ImmunityPeriod' hyperparameter if the subnet exists, ``None`` otherwise.

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

    async def is_fast_blocks(self):
        # TODO: Document what "fast blocks" means (e.g., 10s blocks) and link to chain configuration docs.
        """Returns True if the node is running with fast blocks. False if not."""
        return (
            await self.query_constant("SubtensorModule", "DurationOfStartCall")
        ).value == 10

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
            ``True`` if the hotkey is a delegate, ``False`` otherwise.

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
            bool: ``True`` if the hotkey is registered in the specified context (either any subnet or a specific subnet),
                ``False`` otherwise.

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
            hotkey_ss58: The ``SS58`` address of the neuron's hotkey.
            block: The blockchain block number for the query.
            block_hash: The blockchain block_hash representation of block id.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            bool: ``True`` if the hotkey is registered on any subnet, False otherwise.

        This function is essential for determining the network-wide presence and participation of a neuron.
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
        """Checks if the hotkey is registered on a given netuid."""
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
        # TODO: Link to 'start_call' explanation and clarify edge cases (e.g., network just registered but not started).
        """Verify if subnet with provided netuid is active.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The blockchain block_hash representation of block id.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            ``True`` if subnet is active, ``False`` otherwise.

        Note:
            This means whether the ``start_call`` was initiated or not.
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
        """
        # TODO: Link to 'drand' glossary and explain how drand rounds map to commit-reveal reveal timing.
        Retrieves the last drand round emitted in bittensor. This corresponds when committed weights will be revealed.

        Returns:
            int: The latest Drand round emitted in bittensor.
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
        """
        Returns network MaxWeightsLimit hyperparameter.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            block: The blockchain block number for the query.
            block_hash: The blockchain block_hash representation of block id.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            The value of the MaxWeightsLimit hyperparameter, or ``None`` if the subnetwork does not exist or the
                parameter is not found.
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
            lite: If true, returns a metagraph using a lightweight sync (no weights, no bonds).
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
        """
        Returns network MinAllowedWeights hyperparameter.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            block: The blockchain block number for the query.
            block_hash: The blockchain block_hash representation of block id.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            The value of the MinAllowedWeights hyperparameter, or ``None`` if the subnetwork does not exist or the
                parameter is not found.
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
            An object containing the identity information of the neuron if found, ``None`` otherwise.

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
        """
        Retrieves the 'Burn' hyperparameter for a specified subnet. The 'Burn' parameter represents the amount of Tao
        that is effectively recycled within the Bittensor network.

        Parameters:
            netuid: The unique identifier of the subnet.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number for the query.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            The value of the 'Burn' hyperparameter if the subnet exists, ``None`` otherwise.

        Understanding the 'Burn' rate is essential for analyzing the network registration usage, particularly how it is
        correlated with user activity and the overall cost of participation in a given subnet.
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
            ``True`` if the subnet exists, ``False`` otherwise.

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
        """
        Returns network SubnetworkN hyperparameter.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number at which to check the subnet existence.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            The value of the SubnetworkN hyperparameter, or ``None`` if the subnetwork does not exist or the parameter
                is not found.
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
        """
        Returns network Tempo hyperparameter.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            block: The blockchain block number for the query.
            block_hash: The hash of the blockchain block number at which to check the subnet existence.
            reuse_block: Whether to reuse the last-used block hash.

        Returns:
            The value of the Tempo hyperparameter, or ``None`` if the subnetwork does not exist or the parameter is not
                found.
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
            The transaction rate limit of the network, ``None`` if not available.

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
            block: The block number to wait for. If ``None``, waits for the next block.

        Returns:
            ``True`` if the target block was reached, ``False`` if timeout occurred.

        Example::

            import bittensor as bt
            subtensor = bt.Subtensor()

            await subtensor.wait_for_block() # Waits for next block
            await subtensor.wait_for_block(block=1234) # Waits for a specific block
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
        """
        Returns network WeightsSetRateLimit hyperparameter.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            block: The blockchain block number for the query.
            block_hash: The blockchain block_hash representation of the block id.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            Optional[int]: The value of the WeightsSetRateLimit hyperparameter, or ``None`` if the subnetwork does not
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
        # TODO: Add Raises documentation for missing required params (KeyError) and invalid calls (ValueError).
        # TODO: Examples: validate against metadata; See Also: `compose_call`.
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
        # TODO: Document Returns type (`GenericCall`) clearly; Examples composing a Balances transfer call.
        # TODO: Raises propagated from `validate_extrinsic_params`; See Also: `sign_and_send_extrinsic`.
        """
        Dynamically compose a GenericCall using on-chain Substrate metadata after validating the provided parameters.

        Parameters:
            call_module: Pallet name (e.g. "SubtensorModule", "AdminUtils").
            call_function: Function name (e.g. "set_weights", "sudo_set_tempo").
            call_params: Dictionary of parameters for the call.
            block: The blockchain block number for the query.
            block_hash: The blockchain block_hash representation of the block id.
            reuse_block: Whether to reuse the last-used blockchain block hash.

        Returns:
            GenericCall: Composed call object ready for extrinsic submission.
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
        # TODO: Clarify async behavior and when to set inclusion/finalization; Returns include fee when available.
        # TODO: Document Raises for invalid sign_with/nonce_key; Examples with common paths.
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
            extrinsic_response.extrinsic_receipt = response
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
    ):
        # TODO: Specify units (Rao), Examples estimating fee before sending; link to transfer fee helpers.
        """
        Get extrinsic fee for a given extrinsic call and keypair for a given SN's netuid.

        Parameters:
            call: The extrinsic GenericCall.
            keypair: The keypair associated with the extrinsic.

        Returns:
            Balance object representing the extrinsic fee in RAO.

        Note:
            To create the GenericCall object use `compose_call` method with proper parameters.
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
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
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
            safe_staking: If true, enables price safety checks to protect against fluctuating prices. The stake will
                only execute if the price change doesn't exceed the rate tolerance.
            allow_partial_stake: If true and safe_staking is enabled, allows partial staking when the full amount would
                exceed the price tolerance. If false, the entire stake fails if it would exceed the tolerance.
            rate_tolerance: The maximum allowed price change ratio when staking. For example, 0.005 = 0.5% maximum price
                increase. Only used when safe_staking is True.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        This function enables neurons to increase their stake in the network, enhancing their influence and potential.
        When safe_staking is enabled, it provides protection against price fluctuations during the time stake is
        executed and the time it is actually processed by the chain.
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
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def add_liquidity(
        self,
        wallet: "Wallet",
        netuid: int,
        liquidity: Balance,
        price_low: Balance,
        price_high: Balance,
        hotkey_ss58: Optional[str] = None,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
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
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Note: Adding is allowed even when user liquidity is enabled in specified subnet. Call ``toggle_user_liquidity``
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
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def add_stake_multiple(
        self,
        wallet: "Wallet",
        netuids: UIDs,
        hotkey_ss58s: list[str],
        amounts: list[Balance],
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Adds stakes to multiple neurons identified by their hotkey SS58 addresses.
        This bulk operation allows for efficient staking across different neurons from a single wallet.

        Parameters:
            wallet: The wallet used for staking.
            netuids: List of subnet UIDs.
            hotkey_ss58s: List of ``SS58`` addresses of hotkeys to stake to.
            amounts: List of corresponding TAO amounts to bet for each netuid and hotkey.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        This function is essential for managing stakes across multiple neurons, reflecting the dynamic and collaborative
        nature of the Bittensor network.
        """
        return await add_stake_multiple_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuids=netuids,
            hotkey_ss58s=hotkey_ss58s,
            amounts=amounts,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def add_proxy(
        self,
        wallet: "Wallet",
        delegate_ss58: str,
        proxy_type: Union[str, "ProxyType"],
        delay: int,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
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
                string or ProxyType enum value.
            delay: The number of blocks before the proxy can be used.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Note:
            A deposit is required when adding a proxy. The deposit amount is determined by runtime constants and is
            returned when the proxy is removed. Use `get_proxy_constants()` to check current deposit requirements.
        """
        return await add_proxy_extrinsic(
            subtensor=self,
            wallet=wallet,
            delegate_ss58=delegate_ss58,
            proxy_type=proxy_type,
            delay=delay,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def announce_proxy(
        self,
        wallet: "Wallet",
        real_account_ss58: str,
        call_hash: str,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Announces a future call that will be executed through a proxy.

        This method allows a proxy account to declare its intention to execute a specific call on behalf of a real
        account after a delay period. The real account can review and either approve (via `proxy_announced()`) or reject
        (via `reject_proxy_announcement()`) the announcement.

        Parameters:
            wallet: Bittensor wallet object (should be the proxy account wallet).
            real_account_ss58: The SS58 address of the real account on whose behalf the call will be made.
            call_hash: The hash of the call that will be executed in the future.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Note:
            A deposit is required when making an announcement. The deposit is returned when the announcement is
            executed, rejected, or removed. The announcement can be executed after the delay period has passed.
        """
        return await announce_extrinsic(
            subtensor=self,
            wallet=wallet,
            real_account_ss58=real_account_ss58,
            call_hash=call_hash,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def burned_register(
        self,
        wallet: "Wallet",
        netuid: int,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Registers a neuron on the Bittensor network by recycling TAO. This method of registration involves recycling
        TAO tokens, allowing them to be re-mined by performing work on the network.

        Parameters:
            wallet: The wallet associated with the neuron to be registered.
            netuid: The unique identifier of the subnet.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.
        """
        async with self:
            if netuid == 0:
                return await root_register_extrinsic(
                    subtensor=self,
                    wallet=wallet,
                    period=period,
                    raise_error=raise_error,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )

            return await burned_register_extrinsic(
                subtensor=self,
                wallet=wallet,
                netuid=netuid,
                period=period,
                raise_error=raise_error,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

    async def claim_root(
        self,
        wallet: "Wallet",
        netuids: "UIDs",
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ):
        """Claims the root emissions for a coldkey.

        Parameters:
            wallet: Bittensor Wallet instance.
            netuids: The netuids to claim root emissions for.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.
        """
        return await claim_root_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuids=netuids,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
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
        period: Optional[int] = 16,
        raise_error: bool = True,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
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
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        This function allows subnet validators to create a tamper-proof record of their weight vector at a specific
        point in time, creating a foundation of transparency and accountability for the Bittensor network.

        Notes:
            See also: <https://docs.learnbittensor.org/glossary#commit-reveal>,
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
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                    period=period,
                    raise_error=raise_error,
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
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Contributes funds to an active crowdloan campaign.

        Parameters:
            wallet: Bittensor Wallet instance used to sign the transaction.
            crowdloan_id: The unique identifier of the crowdloan to contribute to.
            amount: Amount to contribute.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.
        """
        return await contribute_crowdloan_extrinsic(
            subtensor=self,
            wallet=wallet,
            crowdloan_id=crowdloan_id,
            amount=amount,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
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
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
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
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.
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
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def create_pure_proxy(
        self,
        wallet: "Wallet",
        proxy_type: Union[str, "ProxyType"],
        delay: int,
        index: int,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Creates a pure proxy account.

        A pure proxy is a keyless account that can only be controlled through proxy relationships. Unlike regular
        proxies, pure proxies do not have their own private keys, making them more secure for certain use cases. The
        pure proxy address is deterministically generated based on the spawner account, proxy type, delay, and index.

        Parameters:
            wallet: Bittensor wallet object.
            proxy_type: The type of proxy permissions for the pure proxy. Can be a string or ProxyType enum value.
            delay: The number of blocks before the pure proxy can be used.
            index: The index to use for generating the pure proxy account address.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Note:
            The pure proxy account address can be extracted from the "PureCreated" event in the response. Store the
            spawner address, proxy_type, index, height, and ext_index as they are required to kill the pure proxy later
            via `kill_pure_proxy()`.
        """
        return await create_pure_proxy_extrinsic(
            subtensor=self,
            wallet=wallet,
            proxy_type=proxy_type,
            delay=delay,
            index=index,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def dissolve_crowdloan(
        self,
        wallet: "Wallet",
        crowdloan_id: int,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Dissolves a completed or failed crowdloan campaign after all refunds are processed.

        This permanently removes the campaign from on-chain storage and refunds the creator's remaining deposit, if
        applicable. Can only be called by the campaign creator.

        Parameters:
            wallet: Bittensor Wallet instance used to sign the transaction.
            crowdloan_id: The unique identifier of the crowdloan to dissolve.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Notes:
            - Only the creator can dissolve their own crowdloan.
            - All contributors (except the creator) must have been refunded first.
            - The creator’s remaining contribution (deposit) is returned during dissolution.
            - After this call, the crowdloan is removed from chain storage.
        """
        return await dissolve_crowdloan_extrinsic(
            subtensor=self,
            wallet=wallet,
            crowdloan_id=crowdloan_id,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def finalize_crowdloan(
        self,
        wallet: "Wallet",
        crowdloan_id: int,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Finalizes a successful crowdloan campaign once the cap has been reached and the end block has passed.

        This executes the stored call or transfers the raised funds to the target address, completing the campaign.

        Parameters:
            wallet: Bittensor Wallet instance used to sign the transaction.
            crowdloan_id: The unique identifier of the crowdloan to finalize.
            period: The number of blocks during which the transaction will remain valid after it's submitted.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.
        """
        return await finalize_crowdloan_extrinsic(
            subtensor=self,
            wallet=wallet,
            crowdloan_id=crowdloan_id,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
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
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Kills (removes) a pure proxy account.

        This method removes a pure proxy account that was previously created via `create_pure_proxy()`. The `kill_pure`
        call must be executed through the pure proxy account itself, with the spawner acting as an "Any" proxy. This
        method automatically handles this by executing the call via `proxy()`.

        Parameters:
            wallet: Bittensor wallet object. The wallet.coldkey.ss58_address must be the spawner of the pure proxy (the
                account that created it via `create_pure_proxy()`). The spawner must have an "Any" proxy relationship
                with the pure proxy.
            pure_proxy_ss58: The SS58 address of the pure proxy account to be killed. This is the address that was
                returned in the `create_pure_proxy()` response.
            spawner: The SS58 address of the spawner account (the account that originally created the pure proxy via
                `create_pure_proxy()`). This should match wallet.coldkey.ss58_address.
            proxy_type: The type of proxy permissions. Can be a string or ProxyType enum value. Must match the
                proxy_type used when creating the pure proxy.
            index: The disambiguation index originally passed to `create_pure()`.
            height: The block height at which the pure proxy was created.
            ext_index: The extrinsic index at which the pure proxy was created.
            force_proxy_type: The proxy type relationship to use when executing `kill_pure` through the proxy mechanism.
                Since pure proxies are keyless and cannot sign transactions, the spawner must act as a proxy for the
                pure proxy to execute `kill_pure`. This parameter specifies which proxy type relationship between the
                spawner and the pure proxy account should be used. The spawner must have a proxy relationship of this
                type (or `Any`) with the pure proxy account. Defaults to `ProxyType.Any` for maximum compatibility. If
                `None`, Substrate will automatically select an available proxy type from the spawner's proxy
                relationships.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Note:
            The `kill_pure` call must be executed through the pure proxy account itself, with the spawner acting as an
            "Any" proxy. This method automatically handles this by executing the call via `proxy()`. The spawner must
            have an "Any" proxy relationship with the pure proxy for this to work.

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
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def modify_liquidity(
        self,
        wallet: "Wallet",
        netuid: int,
        position_id: int,
        liquidity_delta: Balance,
        hotkey_ss58: Optional[str] = None,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """Modifies liquidity in liquidity position by adding or removing liquidity from it.

        Parameters:
            wallet: The wallet used to sign the extrinsic (must be unlocked).
            netuid: The UID of the target subnet for which the call is being initiated.
            position_id: The id of the position record in the pool.
            liquidity_delta: The amount of liquidity to be added or removed (add if positive or remove if negative).
            hotkey_ss58: The hotkey with staked TAO in Alpha. If not passed then the wallet hotkey is used.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Example::

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
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
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
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
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
            move_all_stake: If true, moves all stake from the source hotkey to the destination hotkey.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.
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
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def poke_deposit(
        self,
        wallet: "Wallet",
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Adjusts deposits made for proxies and announcements based on current values.

        This method recalculates and updates the locked deposit amounts for both proxy relationships and announcements
        for the signing account. It can be used to potentially lower the locked amount if the deposit requirements have
        changed (e.g., due to runtime upgrades or changes in the number of proxies/announcements).

        Parameters:
            wallet: Bittensor wallet object (the account whose deposits will be adjusted).
            period: The number of blocks during which the transaction will remain valid after it's submitted.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.

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
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def proxy(
        self,
        wallet: "Wallet",
        real_account_ss58: str,
        force_proxy_type: Optional[Union[str, "ProxyType"]],
        call: "GenericCall",
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Executes a call on behalf of the real account through a proxy.

        This method allows a proxy account (delegate) to execute a call on behalf of the real account (delegator). The
        call is subject to the permissions defined by the proxy type and must respect the delay period if one was set
        when the proxy was added.

        Parameters:
            wallet: Bittensor wallet object (should be the proxy account wallet).
            real_account_ss58: The SS58 address of the real account on whose behalf the call is being made.
            force_proxy_type: The type of proxy to use for the call. If None, any proxy type can be used. Otherwise,
                must match one of the allowed proxy types. Can be a string or ProxyType enum value.
            call: The inner call to be executed on behalf of the real account.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Note:
            The call must be permitted by the proxy type. For example, a "NonTransfer" proxy cannot execute transfer
            calls. The delay period must also have passed since the proxy was added.
        """
        return await proxy_extrinsic(
            subtensor=self,
            wallet=wallet,
            real_account_ss58=real_account_ss58,
            force_proxy_type=force_proxy_type,
            call=call,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def proxy_announced(
        self,
        wallet: "Wallet",
        delegate_ss58: str,
        real_account_ss58: str,
        force_proxy_type: Optional[Union[str, "ProxyType"]],
        call: "GenericCall",
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Executes an announced call on behalf of the real account through a proxy.

        This method executes a call that was previously announced via `announce_proxy()`. The call must match the
        call_hash that was announced, and the delay period must have passed since the announcement was made. The real
        account has the opportunity to review and reject the announcement before execution.

        Parameters:
            wallet: Bittensor wallet object (should be the proxy account wallet that made the announcement).
            delegate_ss58: The SS58 address of the delegate proxy account that made the announcement.
            real_account_ss58: The SS58 address of the real account on whose behalf the call will be made.
            force_proxy_type: The type of proxy to use for the call. If None, any proxy type can be used. Otherwise,
                must match one of the allowed proxy types. Can be a string or ProxyType enum value.
            call: The inner call to be executed on behalf of the real account (must match the announced call_hash).
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Note:
            The call_hash of the provided call must match the call_hash that was announced. The announcement must not
            have been rejected by the real account, and the delay period must have passed.
        """
        return await proxy_announced_extrinsic(
            subtensor=self,
            wallet=wallet,
            delegate_ss58=delegate_ss58,
            real_account_ss58=real_account_ss58,
            force_proxy_type=force_proxy_type,
            call=call,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def refund_crowdloan(
        self,
        wallet: "Wallet",
        crowdloan_id: int,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Refunds contributors from a failed or expired crowdloan campaign.

        This call attempts to refund up to the limit defined by `RefundContributorsLimit` in a single dispatch. If there are
        more contributors than the limit, the call may need to be executed multiple times until all refunds are processed.

        Parameters:
            wallet: Bittensor Wallet instance used to sign the transaction.
            crowdloan_id: The unique identifier of the crowdloan to refund.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Notes:
            - Can be called by only creator signed account.
            - Refunds contributors (excluding the creator) whose funds were locked in a failed campaign.
            - Each call processes a limited number of refunds (`RefundContributorsLimit`).
            - If the campaign has too many contributors, multiple refund calls are required.
        """
        return await refund_crowdloan_extrinsic(
            subtensor=self,
            wallet=wallet,
            crowdloan_id=crowdloan_id,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def reject_proxy_announcement(
        self,
        wallet: "Wallet",
        delegate_ss58: str,
        call_hash: str,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
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
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Note:
            Once rejected, the announcement cannot be executed. The delegate's announcement deposit is returned.
        """
        return await reject_announcement_extrinsic(
            subtensor=self,
            wallet=wallet,
            delegate_ss58=delegate_ss58,
            call_hash=call_hash,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
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
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Registers a neuron on the Bittensor subnet with provided netuid using the provided wallet.

        Registration is a critical step for a neuron to become an active participant in the network, enabling it to
        stake, set weights, and receive incentives.

        Parameters:
            wallet: The wallet associated with the neuron to be registered.
            netuid: The unique identifier of the subnet.
            max_allowed_attempts: Maximum number of attempts to register the wallet.
            output_in_place: If true, prints the progress of the proof of work to the console in-place. Meaning the
                progress is printed on the same lines.
            cuda: If ``true``, the wallet should be registered using CUDA device(s).
            dev_id: The CUDA device id to use, or a list of device ids.
            tpb: The number of threads per block (CUDA).
            num_processes: The number of processes to use to register.
            update_interval: The number of nonces to solve between updates.
            log_verbose: If ``true``, the registration process will log more information.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        This function facilitates the entry of new neurons into the network, supporting the decentralized growth and
        scalability of the Bittensor ecosystem.
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
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def register_subnet(
        self: "AsyncSubtensor",
        wallet: "Wallet",
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Registers a new subnetwork on the Bittensor network.

        Parameters:
            wallet: The wallet to be used for subnet registration.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.
        """
        return await register_subnet_extrinsic(
            subtensor=self,
            wallet=wallet,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def remove_proxy_announcement(
        self,
        wallet: "Wallet",
        real_account_ss58: str,
        call_hash: str,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
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
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Note:
            Only the proxy account that made the announcement can remove it. The real account can reject it via
            `reject_proxy_announcement()`, but cannot remove it directly.
        """
        return await remove_announcement_extrinsic(
            subtensor=self,
            wallet=wallet,
            real_account_ss58=real_account_ss58,
            call_hash=call_hash,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def remove_liquidity(
        self,
        wallet: "Wallet",
        netuid: int,
        position_id: int,
        hotkey_ss58: Optional[str] = None,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """Remove liquidity and credit balances back to wallet's hotkey stake.

        Parameters:
            wallet: The wallet used to sign the extrinsic (must be unlocked).
            netuid: The UID of the target subnet for which the call is being initiated.
            position_id: The id of the position record in the pool.
            hotkey_ss58: The hotkey with staked TAO in Alpha. If not passed then the wallet hotkey is used.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

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
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def remove_proxies(
        self,
        wallet: "Wallet",
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Removes all proxy relationships for the account in a single transaction.

        This method removes all proxy relationships for the signing account in a single call, which is more efficient
        than removing them one by one using `remove_proxy()`. The deposit for all proxies will be returned to the
        account.

        Parameters:
            wallet: Bittensor wallet object. The account whose proxies will be removed (the delegator). All proxy
                relationships where wallet.coldkey.ss58_address is the real account will be removed.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Note:
            This removes all proxy relationships for the account, regardless of proxy type or delegate. Use
            `remove_proxy()` if you need to remove specific proxy relationships selectively.
        """
        return await remove_proxies_extrinsic(
            subtensor=self,
            wallet=wallet,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def remove_proxy(
        self,
        wallet: "Wallet",
        delegate_ss58: str,
        proxy_type: Union[str, "ProxyType"],
        delay: int,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Removes a specific proxy relationship.

        This method removes a single proxy relationship between the real account and a delegate. The parameters must
        exactly match those used when the proxy was added via `add_proxy()`. The deposit for this proxy will be returned
        to the account.

        Parameters:
            wallet: Bittensor wallet object.
            delegate_ss58: The SS58 address of the delegate proxy account to remove.
            proxy_type: The type of proxy permissions to remove. Can be a string or ProxyType enum value.
            delay: The number of blocks before the proxy removal takes effect.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Note:
            The delegate_ss58, proxy_type, and delay parameters must exactly match those used when the proxy was added.
            Use `get_proxies_for_real_account()` to retrieve the exact parameters for existing proxies.
        """
        return await remove_proxy_extrinsic(
            subtensor=self,
            wallet=wallet,
            delegate_ss58=delegate_ss58,
            proxy_type=proxy_type,
            delay=delay,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
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
        period: Optional[int] = 16,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
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
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        This function allows neurons to reveal their previously committed weight distribution, ensuring transparency and
        accountability within the Bittensor network.

        See also: <https://docs.learnbittensor.org/glossary#commit-reveal>,
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
                    period=period,
                    raise_error=raise_error,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
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
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Register neuron by recycling some TAO.

        Parameters:
            wallet: The wallet associated with the neuron to be registered.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.
        """

        return await root_register_extrinsic(
            subtensor=self,
            wallet=wallet,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def root_set_pending_childkey_cooldown(
        self,
        wallet: "Wallet",
        cooldown: int,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """Sets the pending childkey cooldown.

        Parameters:
            wallet: bittensor wallet instance.
            cooldown: the number of blocks to setting pending childkey cooldown.
            period: The number of blocks during which the transaction will remain valid after it's
                submitted. If the transaction is not included in a block within that number of blocks, it will expire
                and be rejected. You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Note: This operation can only be successfully performed if your wallet has root privileges.
        """
        return await root_set_pending_childkey_cooldown_extrinsic(
            subtensor=self,
            wallet=wallet,
            cooldown=cooldown,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def set_auto_stake(
        self,
        wallet: "Wallet",
        netuid: int,
        hotkey_ss58: str,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """Sets the coldkey to automatically stake to the hotkey within specific subnet mechanism.

        Parameters:
            wallet: Bittensor Wallet instance.
            netuid: The subnet unique identifier.
            hotkey_ss58: The SS58 address of the validator's hotkey to which the miner automatically stakes all rewards
                received from the specified subnet immediately upon receipt.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.

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
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def set_children(
        self,
        wallet: "Wallet",
        hotkey_ss58: str,
        netuid: int,
        children: list[tuple[float, str]],
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Allows a coldkey to set children-keys.

        Parameters:
            wallet: bittensor wallet instance.
            hotkey_ss58: The `SS58` address of the neuron's hotkey.
            netuid: The netuid value.
            children: A list of children with their proportions.
            period: The number of blocks during which the transaction will remain valid after it's
                submitted. If the transaction is not included in a block within that number of blocks, it will expire
                and be rejected. You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.

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
        """
        return await set_children_extrinsic(
            subtensor=self,
            wallet=wallet,
            hotkey_ss58=hotkey_ss58,
            netuid=netuid,
            children=children,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def set_delegate_take(
        self,
        wallet: "Wallet",
        hotkey_ss58: str,
        take: float,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Sets the delegate 'take' percentage for a neuron identified by its hotkey.
        The 'take' represents the percentage of rewards that the delegate claims from its nominators' stakes.

        Parameters:
            wallet: bittensor wallet instance.
            hotkey_ss58: The ``SS58`` address of the neuron's hotkey.
            take: Percentage reward for the delegate.
            period: The number of blocks during which the transaction will remain valid after it's
                submitted. If the transaction is not included in a block within that number of blocks, it will expire
                and be rejected. You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.

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
            period=period,
            raise_error=raise_error,
            wait_for_finalization=wait_for_finalization,
            wait_for_inclusion=wait_for_inclusion,
        )

        if response.success:
            return response

        logging.error(f"[red]{response.message}[/red]")
        return response

    async def set_root_claim_type(
        self,
        wallet: "Wallet",
        new_root_claim_type: Literal["Swap", "Keep"],
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ):
        """Sets the root claim type for the coldkey in provided wallet.

        Parameters:
            wallet: Bittensor Wallet instance.
            new_root_claim_type: The new root claim type to set. Could be either "Swap" or "Keep".
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.
        """
        return await set_root_claim_type_extrinsic(
            subtensor=self,
            wallet=wallet,
            new_root_claim_type=new_root_claim_type,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def set_subnet_identity(
        self,
        wallet: "Wallet",
        netuid: int,
        subnet_identity: SubnetIdentity,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Sets the identity of a subnet for a specific wallet and network.

        Parameters:
            wallet: The wallet instance that will authorize the transaction.
            netuid: The unique ID of the network on which the operation takes place.
            subnet_identity: The identity data of the subnet including attributes like name, GitHub repository, contact,
                URL, discord, description, and any additional metadata.
            period: The number of blocks during which the transaction will remain valid after it's
                submitted. If the transaction is not included in a block within that number of blocks, it will expire
                and be rejected. You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.

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
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
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
        period: Optional[int] = 8,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        # TODO: Add glossary links for 'weights', 'commit-reveal', and 'Yuma Consensus'.
        # TODO: Clarify `block_time` units, CRv4 behavior, and rate-limit gating (_blocks_weight_limit) with links.
        # TODO: Provide a short example for both commit-reveal-enabled and direct set paths.
        Sets the weight vector for a neuron acting as a validator, specifying the weights assigned to subnet miners
        based on their performance evaluation.

        This method allows subnet validators to submit their weight vectors, which rank the value of each subnet miner's
        work. These weight vectors are used by the Yuma Consensus algorithm to compute emissions for both validators and
        miners.

        Parameters:
            wallet: The wallet associated with the subnet validator setting the weights.
            netuid: The unique identifier of the subnet.
            uids: The list of subnet miner neuron UIDs that the weights are being set for.
            weights: The corresponding weights to be set for each UID, representing the validator's evaluation of each
                miner's performance.
            mechid: The subnet mechanism unique identifier.
            block_time: The number of seconds for block duration.
            commit_reveal_version: The version of the chain commit-reveal protocol to use.
            max_attempts: The number of maximum attempts to set weights.
            version_key: Version key for compatibility with the network.
            period: The number of blocks during which the transaction will remain valid after it's
                submitted. If the transaction is not included in a block within that number of blocks, it will expire
                and be rejected. You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        This function is crucial in the Yuma Consensus mechanism, where each validator's weight vector contributes to
        the overall weight matrix used to calculate emissions and maintain network consensus.

        Notes:
            See <https://docs.learnbittensor.org/glossary#yuma-consensus>
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
                        period=period,
                        raise_error=raise_error,
                        wait_for_inclusion=wait_for_inclusion,
                        wait_for_finalization=wait_for_finalization,
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
                        period=period,
                        raise_error=raise_error,
                        wait_for_inclusion=wait_for_inclusion,
                        wait_for_finalization=wait_for_finalization,
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
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        # TODO: Link to 'Axon' concept docs and provide a minimal example (ports, TLS cert expectations).
        # TODO: Clarify what happens when re-serving with updated metadata.
        Registers an ``Axon`` serving endpoint on the Bittensor network for a specific neuron.

        This function is used to set up the Axon, a key component of a neuron that handles incoming queries and data
        processing tasks.

        Parameters:
            netuid: The unique identifier of the subnetwork.
            axon: The Axon instance to be registered for serving.
            certificate: Certificate to use for TLS. If ``None``, no TLS will be used.
            period: The number of blocks during which the transaction will remain valid after it's
                submitted. If the transaction is not included in a block within that number of blocks, it will expire
                and be rejected. You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Waits for the transaction to be included in a block.
            wait_for_finalization: Waits for the transaction to be finalized on the blockchain.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        By registering an Axon, the neuron becomes an active part of the network's distributed computing infrastructure,
        contributing to the collective intelligence of Bittensor.
        """
        return await serve_axon_extrinsic(
            subtensor=self,
            netuid=netuid,
            axon=axon,
            certificate=certificate,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def set_commitment(
        self,
        wallet: "Wallet",
        netuid: int,
        data: str,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        # TODO: Clarify size limits for metadata payloads and encoding; link to commit-reveal docs.
        Commits arbitrary data to the Bittensor network by publishing metadata.

        This method allows neurons to publish arbitrary data to the blockchain, which can be used for various purposes
        such as sharing model updates, configuration data, or other network-relevant information.

        Parameters:
            wallet (bittensor_wallet.Wallet): The wallet associated with the neuron committing the data.
            netuid (int): The unique identifier of the subnetwork.
            data (str): The data to be committed to the network.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Example::

            # Commit some data to subnet 1
            success = await subtensor.commit(wallet=my_wallet, netuid=1, data="Hello Bittensor!")

            # Commit with custom period
            success = await subtensor.commit(wallet=my_wallet, netuid=1, data="Model update v2.0", period=100)

        Note: See <https://docs.learnbittensor.org/glossary#commit-reveal>
        """
        return await publish_metadata_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            data_type=f"Raw{len(data)}",
            data=data.encode(),
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def set_reveal_commitment(
        self,
        wallet,
        netuid: int,
        data: str,
        blocks_until_reveal: int = 360,
        block_time: Union[int, float] = 12,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        # TODO: Link to commit-reveal timelock docs; clarify security model of `get_encrypted_commitment`.
        # TODO: Spell out expected epoch math: blocks_until_reveal vs tempo; add example for fast vs standard blocks.
        Commits arbitrary data to the Bittensor network by publishing metadata.

        Parameters:
            wallet: The wallet associated with the neuron committing the data.
            netuid: The unique identifier of the subnetwork.
            data: The data to be committed to the network.
            blocks_until_reveal: The number of blocks from now after which the data will be revealed. Then number of
                blocks in one epoch.
            block_time: The number of seconds between each block.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Note:
            A commitment can be set once per subnet epoch and is reset at the next epoch in the chain automatically.
            Successful extrinsic's the "data" field contains {"encrypted": encrypted, "reveal_round": reveal_round}.
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
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
        response.data = data_
        return response

    async def start_call(
        self,
        wallet: "Wallet",
        netuid: int,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> ExtrinsicResponse:
        """
        # TODO: Add glossary link for 'start_call' and 'emission start'; clarify who is authorized to call this.
        Submits a start_call extrinsic to the blockchain, to trigger the start call process for a subnet (used to start
        a new subnet's emission mechanism).

        Parameters:
            wallet: The wallet used to sign the extrinsic (must be unlocked).
            netuid: The UID of the target subnet for which the call is being initiated.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.
        """
        return await start_call_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
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
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        # TODO: Link to SimSwap docs and define price ratio logic used in safe mode; clarify partial swapping semantics.
        # TODO: State units for `amount` and note interaction with fees and slippage.
        Moves stake between subnets while keeping the same coldkey-hotkey pair ownership.
        Like subnet hopping - same owner, same hotkey, just changing which subnet the stake is in.

        Parameters:
            wallet: The wallet to swap stake from.
            hotkey_ss58: The SS58 address of the hotkey whose stake is being swapped.
            origin_netuid: The netuid from which stake is removed.
            destination_netuid: The netuid to which stake is added.
            amount: The amount to swap.
            safe_swapping: If true, enables price safety checks to protect against fluctuating prices. The swap
                will only execute if the price ratio between subnets doesn't exceed the rate tolerance.
            allow_partial_stake: If true and safe_staking is enabled, allows partial stake swaps when the full amount
                would exceed the price tolerance. If false, the entire swap fails if it would exceed the tolerance.
            rate_tolerance: The maximum allowed increase in the price ratio between subnets
                (origin_price/destination_price). For example, 0.005 = 0.5% maximum increase. Only used when
                safe_staking is True.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If the
                transaction is not included in a block within that number of blocks, it will expire and be rejected. You
                can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the inclusion of the transaction.
            wait_for_finalization: Whether to wait for the finalization of the transaction.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        The price ratio for swap_stake in safe mode is calculated as: origin_subnet_price / destination_subnet_price
        When safe_staking is enabled, the swap will only execute if:
            - With allow_partial_stake=False: The entire swap amount can be executed without the price ratio increasing
            more than rate_tolerance.
            - With allow_partial_stake=True: A partial amount will be swapped up to the point where the price ratio
            would increase by rate_tolerance.
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
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def toggle_user_liquidity(
        self,
        wallet: "Wallet",
        netuid: int,
        enable: bool,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        # TODO: Document authorization (subnet owner only); Examples enabling/disabling; link to liquidity docs.
        """Allow to toggle user liquidity for specified subnet.

        Parameters:
            wallet: The wallet used to sign the extrinsic (must be unlocked).
            netuid: The UID of the target subnet for which the call is being initiated.
            enable: Boolean indicating whether to enable user liquidity.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Note: The call can be executed successfully by the subnet owner only.
        """
        return await toggle_user_liquidity_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            enable=enable,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def transfer(
        self,
        wallet: "Wallet",
        destination_ss58: str,
        amount: Optional[Balance],
        transfer_all: bool = False,
        keep_alive: bool = True,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> ExtrinsicResponse:
        """
        # TODO: Warn about existential deposit implications (keep_alive) and link to 'ExistentialDeposit' constant.
        # TODO: Clarify `amount` units and suggest using `get_transfer_fee` to pre-estimate fees.
        Transfer token of amount to destination.

        Parameters:
            wallet: Source wallet for the transfer.
            destination_ss58: Destination address for the transfer.
            amount: Number of tokens to transfer. `None` is transferring all.
            transfer_all: Flag to transfer all tokens.
            keep_alive: Flag to keep the connection alive.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.
        """
        check_balance_amount(amount)
        return await transfer_extrinsic(
            subtensor=self,
            wallet=wallet,
            destination_ss58=destination_ss58,
            amount=amount,
            transfer_all=transfer_all,
            keep_alive=keep_alive,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def transfer_stake(
        self,
        wallet: "Wallet",
        destination_coldkey_ss58: str,
        hotkey_ss58: str,
        origin_netuid: int,
        destination_netuid: int,
        amount: Balance,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
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
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.
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
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
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
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Removes a specified amount of stake from a single hotkey account. This function is critical for adjusting
        individual neuron stakes within the Bittensor network.

        Parameters:
            wallet: The wallet associated with the neuron from which the stake is being removed.
            netuid: The unique identifier of the subnet.
            hotkey_ss58: The ``SS58`` address of the hotkey account to unstake from.
            amount: The amount of alpha to unstake. If not specified, unstakes all. Alpha amount.
            allow_partial_stake: If true and safe_staking is enabled, allows partial unstaking when
                the full amount would exceed the price tolerance. If false, the entire unstake fails if it would
                exceed the tolerance.
            rate_tolerance: The maximum allowed price change ratio when unstaking. For example,
                0.005 = 0.5% maximum price decrease. Only used when safe_staking is True.
            safe_unstaking: If true, enables price safety checks to protect against fluctuating prices. The unstake
                will only execute if the price change doesn't exceed the rate tolerance.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        This function supports flexible stake management, allowing neurons to adjust their network participation and
        potential reward accruals.
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
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def unstake_all(
        self,
        wallet: "Wallet",
        netuid: int,
        hotkey_ss58: str,
        rate_tolerance: Optional[float] = 0.005,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        # TODO: Link to liquidity/price impact docs to explain `rate_tolerance` and safe unstaking behavior.
        """Unstakes all TAO/Alpha associated with a hotkey from the specified subnets on the Bittensor network.

        Parameters:
            wallet: The wallet of the stake owner.
            netuid: The unique identifier of the subnet.
            hotkey_ss58: The SS58 address of the hotkey to unstake from.
            rate_tolerance: The maximum allowed price change ratio when unstaking. For example, 0.005 = 0.5% maximum
                price decrease. If not passed (None), then unstaking goes without price limit.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Example::

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
        """
        return await unstake_all_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            hotkey_ss58=hotkey_ss58,
            rate_tolerance=rate_tolerance,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def unstake_multiple(
        self,
        wallet: "Wallet",
        netuids: UIDs,
        hotkey_ss58s: list[str],
        amounts: Optional[list[Balance]] = None,
        unstake_all: bool = False,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Performs batch unstaking from multiple hotkey accounts, allowing a neuron to reduce its staked amounts
        efficiently. This function is useful for managing the distribution of stakes across multiple neurons.

        Parameters:
            wallet: The wallet linked to the coldkey from which the stakes are being withdrawn.
            netuids: Subnets unique IDs.
            hotkey_ss58s: A list of hotkey `SS58` addresses to unstake from.
            amounts: The amounts of TAO to unstake from each hotkey. If not provided, unstakes all.
            unstake_all: If true, unstakes all tokens. If `True` amounts are ignored.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        This function allows for strategic reallocation or withdrawal of stakes, aligning with the dynamic stake
        management aspect of the Bittensor network.
        """
        return await unstake_multiple_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuids=netuids,
            hotkey_ss58s=hotkey_ss58s,
            amounts=amounts,
            unstake_all=unstake_all,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def update_cap_crowdloan(
        self,
        wallet: "Wallet",
        crowdloan_id: int,
        new_cap: "Balance",
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Updates the fundraising cap (maximum total contribution) of a non-finalized crowdloan.

        Only the creator of the crowdloan can perform this action, and the new cap must be greater than or equal to the
        current amount already raised.

        Parameters:
            wallet: Bittensor Wallet instance used to sign the transaction.
            crowdloan_id: The unique identifier of the crowdloan to update.
            new_cap: The new fundraising cap (in TAO or Balance).
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Notes:
            - Only the creator can update the cap.
            - The crowdloan must not be finalized.
            - The new cap must be greater than or equal to the total funds already raised.
        """
        return await update_cap_crowdloan_extrinsic(
            subtensor=self,
            wallet=wallet,
            crowdloan_id=crowdloan_id,
            new_cap=new_cap,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def update_end_crowdloan(
        self,
        wallet: "Wallet",
        crowdloan_id: int,
        new_end: int,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Updates the end block of a non-finalized crowdloan campaign.

        Only the creator of the crowdloan can perform this action. The new end block must be valid — meaning it cannot be in
        the past and must respect the minimum and maximum duration limits enforced by the chain.

        Parameters:
            wallet: Bittensor Wallet instance used to sign the transaction.
            crowdloan_id: The unique identifier of the crowdloan to update.
            new_end: The new block number at which the crowdloan will end.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Notes:
            - Only the creator can call this extrinsic.
            - The crowdloan must not be finalized.
            - The new end block must be later than the current block and within valid duration bounds (between
                `MinimumBlockDuration` and `MaximumBlockDuration`).
        """
        return await update_end_crowdloan_extrinsic(
            subtensor=self,
            wallet=wallet,
            crowdloan_id=crowdloan_id,
            new_end=new_end,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def update_min_contribution_crowdloan(
        self,
        wallet: "Wallet",
        crowdloan_id: int,
        new_min_contribution: "Balance",
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Updates the minimum contribution amount of a non-finalized crowdloan.

        Only the creator of the crowdloan can perform this action, and the new value must be greater than or equal to the
        absolute minimum contribution defined in the chain configuration.

        Parameters:
            wallet: Bittensor Wallet instance used to sign the transaction.
            crowdloan_id: The unique identifier of the crowdloan to update.
            new_min_contribution: The new minimum contribution amount (in TAO or Balance).
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Notes:
            - Can only be called by the creator of the crowdloan.
            - The crowdloan must not be finalized.
            - The new minimum contribution must not fall below the absolute minimum defined in the runtime.
        """
        return await update_min_contribution_crowdloan_extrinsic(
            subtensor=self,
            wallet=wallet,
            crowdloan_id=crowdloan_id,
            new_min_contribution=new_min_contribution,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def withdraw_crowdloan(
        self,
        wallet: "Wallet",
        crowdloan_id: int,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """
        Withdraws a contribution from an active (not yet finalized or dissolved) crowdloan.

        Parameters:
            wallet: Wallet instance used to sign the transaction (must be unlocked).
            crowdloan_id: The unique identifier of the crowdloan to withdraw from.
            period: The number of blocks during which the transaction will remain valid after it's submitted. If
                the transaction is not included in a block within that number of blocks, it will expire and be rejected.
                You can think of it as an expiration date for the transaction.
            raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
            wait_for_inclusion: Whether to wait for the extrinsic to be included in a block.
            wait_for_finalization: Whether to wait for finalization of the extrinsic.

        Returns:
            ExtrinsicResponse: The result object of the extrinsic execution.

        Note:
            - Regular contributors can fully withdraw their contribution before finalization.
            - The creator cannot withdraw the initial deposit, but may withdraw any amount exceeding his deposit.
        """
        return await withdraw_crowdloan_extrinsic(
            subtensor=self,
            wallet=wallet,
            crowdloan_id=crowdloan_id,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )


async def get_async_subtensor(
    network: Optional[str] = None,
    config: Optional["Config"] = None,
    mock: bool = False,
    log_verbose: bool = False,
) -> "AsyncSubtensor":
    """Factory method to create an initialized AsyncSubtensor.
    Mainly useful for when you don't want to run `await subtensor.initialize()` after instantiation.
    """
    sub = AsyncSubtensor(
        network=network, config=config, mock=mock, log_verbose=log_verbose
    )
    await sub.initialize()
    return sub
