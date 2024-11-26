import asyncio
import ssl
from typing import Optional, Any, Union, TypedDict, Iterable

import aiohttp
import numpy as np
import scalecodec
from bittensor_wallet import Wallet
from bittensor_wallet.utils import SS58_FORMAT
from numpy.typing import NDArray
from scalecodec import GenericCall
from scalecodec.base import RuntimeConfiguration
from scalecodec.type_registry import load_type_registry_preset
from substrateinterface.exceptions import SubstrateRequestException

from bittensor.core.chain_data import (
    DelegateInfo,
    custom_rpc_type_registry,
    StakeInfo,
    NeuronInfoLite,
    NeuronInfo,
    SubnetHyperparameters,
    decode_account_id,
)
from bittensor.core.extrinsics.async_registration import register_extrinsic
from bittensor.core.extrinsics.async_root import (
    set_root_weights_extrinsic,
    root_register_extrinsic,
)
from bittensor.core.extrinsics.async_transfer import transfer_extrinsic
from bittensor.core.extrinsics.async_weights import (
    commit_weights_extrinsic,
    set_weights_extrinsic,
)
from bittensor.core.settings import (
    TYPE_REGISTRY,
    DEFAULTS,
    NETWORK_MAP,
    DELEGATES_DETAILS_URL,
    DEFAULT_NETWORK,
)
from bittensor.core.settings import version_as_int
from bittensor.utils import (
    torch,
    ss58_to_vec_u8,
    format_error_message,
    decode_hex_identity_dict,
    validate_chain_endpoint,
    hex_to_bytes,
)
from bittensor.utils.async_substrate_interface import (
    AsyncSubstrateInterface,
    TimeoutException,
)
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging
from bittensor.utils.delegates_details import DelegatesDetails
from bittensor.utils.weight_utils import generate_weight_hash


class ParamWithTypes(TypedDict):
    name: str  # Name of the parameter.
    type: str  # ScaleType string of the parameter.


class ProposalVoteData:
    index: int
    threshold: int
    ayes: list[str]
    nays: list[str]
    end: int

    def __init__(self, proposal_dict: dict) -> None:
        self.index = proposal_dict["index"]
        self.threshold = proposal_dict["threshold"]
        self.ayes = self.decode_ss58_tuples(proposal_dict["ayes"])
        self.nays = self.decode_ss58_tuples(proposal_dict["nays"])
        self.end = proposal_dict["end"]

    @staticmethod
    def decode_ss58_tuples(line: tuple):
        """Decodes a tuple of ss58 addresses formatted as bytes tuples."""
        return [decode_account_id(line[x][0]) for x in range(len(line))]


def _decode_hex_identity_dict(info_dictionary: dict[str, Any]) -> dict[str, Any]:
    """Decodes a dictionary of hexadecimal identities."""
    for k, v in info_dictionary.items():
        if isinstance(v, dict):
            item = next(iter(v.values()))
        else:
            item = v
        if isinstance(item, tuple) and item:
            if len(item) > 1:
                try:
                    info_dictionary[k] = (
                        bytes(item).hex(sep=" ", bytes_per_sep=2).upper()
                    )
                except UnicodeDecodeError:
                    logging.error(f"Could not decode: {k}: {item}.")
            else:
                try:
                    info_dictionary[k] = bytes(item[0]).decode("utf-8")
                except UnicodeDecodeError:
                    logging.error(f"Could not decode: {k}: {item}.")
        else:
            info_dictionary[k] = item

    return info_dictionary


class AsyncSubtensor:
    """Thin layer for interacting with Substrate Interface. Mostly a collection of frequently-used calls."""

    def __init__(self, network: str = DEFAULT_NETWORK):
        if network in NETWORK_MAP:
            self.chain_endpoint = NETWORK_MAP[network]
            self.network = network
            if network == "local":
                logging.warning(
                    "Warning: Verify your local subtensor is running on port 9944."
                )
        else:
            is_valid, _ = validate_chain_endpoint(network)
            if is_valid:
                self.chain_endpoint = network
                if network in NETWORK_MAP.values():
                    self.network = next(
                        key for key, value in NETWORK_MAP.items() if value == network
                    )
                else:
                    self.network = "custom"
            else:
                logging.info(
                    f"Network not specified or not valid. Using default chain endpoint: [blue]{NETWORK_MAP[DEFAULTS.subtensor.network]}[/blue]."
                )
                logging.info(
                    "You can set this for commands with the [blue]--network[/blue] flag, or by setting this in the config."
                )
                self.chain_endpoint = NETWORK_MAP[DEFAULTS.subtensor.network]
                self.network = DEFAULTS.subtensor.network

        self.substrate = AsyncSubstrateInterface(
            chain_endpoint=self.chain_endpoint,
            ss58_format=SS58_FORMAT,
            type_registry=TYPE_REGISTRY,
            chain_name="Bittensor",
        )

    def __str__(self):
        return f"Network: {self.network}, Chain: {self.chain_endpoint}"

    async def __aenter__(self):
        logging.info(
            f"[magenta]Connecting to Substrate:[/magenta] [blue]{self}[/blue][magenta]...[/magenta]"
        )
        try:
            async with self.substrate:
                return self
        except TimeoutException:
            logging.error(
                f"[red]Error[/red]: Timeout occurred connecting to substrate. Verify your chain and network settings: {self}"
            )
            raise ConnectionError
        except (ConnectionRefusedError, ssl.SSLError) as error:
            logging.error(
                f"[red]Error[/red]: Connection refused when connecting to substrate. "
                f"Verify your chain and network settings: {self}. Error: {error}"
            )
            raise ConnectionError

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.substrate.close()

    async def encode_params(
        self,
        call_definition: dict[str, list["ParamWithTypes"]],
        params: Union[list[Any], dict[str, Any]],
    ) -> str:
        """Returns a hex encoded string of the params using their types."""
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
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[Any]:
        """
        Retrieves a specified hyperparameter for a specific subnet.

        Args:
            param_name (str): The name of the hyperparameter to retrieve.
            netuid (int): The unique identifier of the subnet.
            block_hash (Optional[str]): The hash of blockchain block number for the query.
            reuse_block (bool): Whether to reuse the last-used block hash.

        Returns:
            The value of the specified hyperparameter if the subnet exists, or None
        """
        if not await self.subnet_exists(netuid, block_hash):
            logging.error(f"subnet {netuid} does not exist")
            return None

        result = await self.substrate.query(
            module="SubtensorModule",
            storage_function=param_name,
            params=[netuid],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )

        return result

    # Common subtensor methods =========================================================================================

    async def get_current_block(self) -> int:
        """
        Returns the current block number on the Bittensor blockchain. This function provides the latest block number, indicating the most recent state of the blockchain.

        Returns:
            int: The current chain block number.

        Knowing the current block number is essential for querying real-time data and performing time-sensitive operations on the blockchain. It serves as a reference point for network activities and data synchronization.
        """
        return await self.substrate.get_block_number()

    async def get_block_hash(self, block_id: Optional[int] = None):
        """
        Retrieves the hash of a specific block on the Bittensor blockchain. The block hash is a unique identifier representing the cryptographic hash of the block's content, ensuring its integrity and immutability.

        Args:
            block_id (int): The block number for which the hash is to be retrieved.

        Returns:
            str: The cryptographic hash of the specified block.

        The block hash is a fundamental aspect of blockchain technology, providing a secure reference to each block's data. It is crucial for verifying transactions, ensuring data consistency, and maintaining the trustworthiness of the blockchain.
        """
        if block_id:
            return await self.substrate.get_block_hash(block_id)
        else:
            return await self.substrate.get_chain_head()

    async def is_hotkey_registered_any(
        self, hotkey_ss58: str, block_hash: Optional[str] = None
    ) -> bool:
        """
        Checks if a neuron's hotkey is registered on any subnet within the Bittensor network.

        Args:
            hotkey_ss58 (str): The ``SS58`` address of the neuron's hotkey.
            block_hash (Optional[str]): The blockchain block_hash representation of block id.

        Returns:
            bool: ``True`` if the hotkey is registered on any subnet, False otherwise.

        This function is essential for determining the network-wide presence and participation of a neuron.
        """
        return len(await self.get_netuids_for_hotkey(hotkey_ss58, block_hash)) > 0

    async def get_subnet_burn_cost(
        self, block_hash: Optional[str] = None
    ) -> Optional[str]:
        """
        Retrieves the burn cost for registering a new subnet within the Bittensor network. This cost represents the amount of Tao that needs to be locked or burned to establish a new subnet.

        Args:
            block_hash (Optional[int]): The blockchain block_hash of the block id.

        Returns:
            int: The burn cost for subnet registration.

        The subnet burn cost is an important economic parameter, reflecting the network's mechanisms for controlling the proliferation of subnets and ensuring their commitment to the network's long-term viability.
        """
        lock_cost = await self.query_runtime_api(
            runtime_api="SubnetRegistrationRuntimeApi",
            method="get_network_registration_cost",
            params=[],
            block_hash=block_hash,
        )

        return lock_cost

    async def get_total_subnets(
        self, block_hash: Optional[str] = None
    ) -> Optional[int]:
        """
        Retrieves the total number of subnets within the Bittensor network as of a specific blockchain block.

        Args:
            block_hash (Optional[str]): The blockchain block_hash representation of block id.

        Returns:
            Optional[str]: The total number of subnets in the network.

        Understanding the total number of subnets is essential for assessing the network's growth and the extent of its decentralized infrastructure.
        """
        result = await self.substrate.query(
            module="SubtensorModule",
            storage_function="TotalNetworks",
            params=[],
            block_hash=block_hash,
        )
        return result

    async def get_subnets(self, block_hash: Optional[str] = None) -> list[int]:
        """
        Retrieves the list of all subnet unique identifiers (netuids) currently present in the Bittensor network.

        Args:
            block_hash (Optional[str]): The hash of the block to retrieve the subnet unique identifiers from.

        Returns:
            A list of subnet netuids.

        This function provides a comprehensive view of the subnets within the Bittensor network,
        offering insights into its diversity and scale.
        """
        result = await self.substrate.query_map(
            module="SubtensorModule",
            storage_function="NetworksAdded",
            block_hash=block_hash,
            reuse_block_hash=True,
        )
        return (
            []
            if result is None or not hasattr(result, "records")
            else [netuid async for netuid, exists in result if exists]
        )

    async def is_hotkey_delegate(
        self,
        hotkey_ss58: str,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> bool:
        """
        Determines whether a given hotkey (public key) is a delegate on the Bittensor network. This function checks if the neuron associated with the hotkey is part of the network's delegation system.

        Args:
            hotkey_ss58 (str): The SS58 address of the neuron's hotkey.
            block_hash (Optional[str]): The hash of the blockchain block number for the query.
            reuse_block (Optional[bool]): Whether to reuse the last-used block hash.

        Returns:
            `True` if the hotkey is a delegate, `False` otherwise.

        Being a delegate is a significant status within the Bittensor network, indicating a neuron's involvement in consensus and governance processes.
        """
        delegates = await self.get_delegates(
            block_hash=block_hash, reuse_block=reuse_block
        )
        return hotkey_ss58 in [info.hotkey_ss58 for info in delegates]

    async def get_delegates(
        self, block_hash: Optional[str] = None, reuse_block: bool = False
    ) -> list[DelegateInfo]:
        """
        Fetches all delegates on the chain

        Args:
            block_hash (Optional[str]): hash of the blockchain block number for the query.
            reuse_block (Optional[bool]): whether to reuse the last-used block hash.

        Returns:
            List of DelegateInfo objects, or an empty list if there are no delegates.
        """
        hex_bytes_result = await self.query_runtime_api(
            runtime_api="DelegateInfoRuntimeApi",
            method="get_delegates",
            params=[],
            block_hash=block_hash,
            reuse_block=reuse_block,
        )
        if hex_bytes_result is not None:
            return DelegateInfo.list_from_vec_u8(hex_to_bytes(hex_bytes_result))
        else:
            return []

    async def get_stake_info_for_coldkey(
        self,
        coldkey_ss58: str,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[StakeInfo]:
        """
        Retrieves stake information associated with a specific coldkey. This function provides details about the stakes held by an account, including the staked amounts and associated delegates.

        Args:
            coldkey_ss58 (str): The ``SS58`` address of the account's coldkey.
            block_hash (Optional[str]): The hash of the blockchain block number for the query.
            reuse_block (bool): Whether to reuse the last-used block hash.

        Returns:
            A list of StakeInfo objects detailing the stake allocations for the account.

        Stake information is vital for account holders to assess their investment and participation in the network's delegation and consensus processes.
        """
        encoded_coldkey = ss58_to_vec_u8(coldkey_ss58)

        hex_bytes_result = await self.query_runtime_api(
            runtime_api="StakeInfoRuntimeApi",
            method="get_stake_info_for_coldkey",
            params=[encoded_coldkey],
            block_hash=block_hash,
            reuse_block=reuse_block,
        )

        if hex_bytes_result is None:
            return []

        return StakeInfo.list_from_vec_u8(hex_to_bytes(hex_bytes_result))

    async def get_stake_for_coldkey_and_hotkey(
        self, hotkey_ss58: str, coldkey_ss58: str, block_hash: Optional[str] = None
    ) -> Balance:
        """
        Retrieves stake information associated with a specific coldkey and hotkey.

        Args:
            hotkey_ss58 (str): the hotkey SS58 address to query
            coldkey_ss58 (str): the coldkey SS58 address to query
            block_hash (Optional[str]): the hash of the blockchain block number for the query.

        Returns:
            Stake Balance for the given coldkey and hotkey
        """
        _result = await self.substrate.query(
            module="SubtensorModule",
            storage_function="Stake",
            params=[hotkey_ss58, coldkey_ss58],
            block_hash=block_hash,
        )
        return Balance.from_rao(_result or 0)

    async def query_runtime_api(
        self,
        runtime_api: str,
        method: str,
        params: Optional[Union[list[list[int]], dict[str, int], list[int]]],
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional[str]:
        """
        Queries the runtime API of the Bittensor blockchain, providing a way to interact with the underlying runtime and retrieve data encoded in Scale Bytes format. This function is essential for advanced users who need to interact with specific runtime methods and decode complex data types.

        Args:
            runtime_api (str): The name of the runtime API to query.
            method (str): The specific method within the runtime API to call.
            params (Optional[Union[list[list[int]], dict[str, int]]]): The parameters to pass to the method call.
            block_hash (Optional[str]): The hash of the blockchain block number at which to perform the query.
            reuse_block (bool): Whether to reuse the last-used block hash.

        Returns:
            The Scale Bytes encoded result from the runtime API call, or ``None`` if the call fails.

        This function enables access to the deeper layers of the Bittensor blockchain, allowing for detailed and specific interactions with the network's runtime environment.
        """
        call_definition = TYPE_REGISTRY["runtime_api"][runtime_api]["methods"][method]

        data = (
            "0x"
            if params is None
            else await self.encode_params(
                call_definition=call_definition, params=params
            )
        )
        api_method = f"{runtime_api}_{method}"

        json_result = await self.substrate.rpc_request(
            method="state_call",
            params=[api_method, data, block_hash] if block_hash else [api_method, data],
            reuse_block_hash=reuse_block,
        )

        if json_result is None:
            return None

        return_type = call_definition["type"]

        as_scale_bytes = scalecodec.ScaleBytes(json_result["result"])

        rpc_runtime_config = RuntimeConfiguration()
        rpc_runtime_config.update_type_registry(load_type_registry_preset("legacy"))
        rpc_runtime_config.update_type_registry(custom_rpc_type_registry)

        obj = rpc_runtime_config.create_scale_object(return_type, as_scale_bytes)
        if obj.data.to_hex() == "0x0400":  # RPC returned None result
            return None

        return obj.decode()

    async def get_balance(
        self,
        *addresses: str,
        block_hash: Optional[str] = None,
    ) -> dict[str, Balance]:
        """
        Retrieves the balance for given coldkey(s)

        Args:
            addresses (str): coldkey addresses(s).
            block_hash (Optional[str]): the block hash, optional.

        Returns:
            Dict of {address: Balance objects}.
        """
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

    async def get_transfer_fee(
        self, wallet: "Wallet", dest: str, value: Union["Balance", float, int]
    ) -> "Balance":
        """
        Calculates the transaction fee for transferring tokens from a wallet to a specified destination address. This function simulates the transfer to estimate the associated cost, taking into account the current network conditions and transaction complexity.

        Args:
            wallet (bittensor_wallet.Wallet): The wallet from which the transfer is initiated.
            dest (str): The ``SS58`` address of the destination account.
            value (Union[bittensor.utils.balance.Balance, float, int]): The amount of tokens to be transferred, specified as a Balance object, or in Tao (float) or Rao (int) units.

        Returns:
            bittensor.utils.balance.Balance: The estimated transaction fee for the transfer, represented as a Balance object.

        Estimating the transfer fee is essential for planning and executing token transactions, ensuring that the wallet has sufficient funds to cover both the transfer amount and the associated costs. This function provides a crucial tool for managing financial operations within the Bittensor network.
        """
        if isinstance(value, float):
            value = Balance.from_tao(value)
        elif isinstance(value, int):
            value = Balance.from_rao(value)

        if isinstance(value, Balance):
            call = await self.substrate.compose_call(
                call_module="Balances",
                call_function="transfer_allow_death",
                call_params={"dest": dest, "value": value.rao},
            )

            try:
                payment_info = await self.substrate.get_payment_info(
                    call=call, keypair=wallet.coldkeypub
                )
            except Exception as e:
                logging.error(
                    f":cross_mark: [red]Failed to get payment info: [/red]{e}"
                )
                payment_info = {"partialFee": int(2e7)}  # assume  0.02 Tao

            return Balance.from_rao(payment_info["partialFee"])
        else:
            fee = Balance.from_rao(int(2e7))
            logging.error(
                "To calculate the transaction fee, the value must be Balance, float, or int. Received type: %s. Fee "
                "is %s",
                type(value),
                2e7,
            )
            return fee

    async def get_total_stake_for_coldkey(
        self,
        *ss58_addresses,
        block_hash: Optional[str] = None,
    ) -> dict[str, Balance]:
        """
        Returns the total stake held on a coldkey.

        Args:
            ss58_addresses (tuple[str]): The SS58 address(es) of the coldkey(s)
            block_hash (str): The hash of the block number to retrieve the stake from.

        Returns:
            Dict in view {address: Balance objects}.
        """
        calls = [
            (
                await self.substrate.create_storage_key(
                    "SubtensorModule",
                    "TotalColdkeyStake",
                    [address],
                    block_hash=block_hash,
                )
            )
            for address in ss58_addresses
        ]
        batch_call = await self.substrate.query_multi(calls, block_hash=block_hash)
        results = {}
        for item in batch_call:
            results.update({item[0].params[0]: Balance.from_rao(item[1] or 0)})
        return results

    async def get_total_stake_for_hotkey(
        self,
        *ss58_addresses,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> dict[str, Balance]:
        """
        Returns the total stake held on a hotkey.

        Args:
            ss58_addresses (tuple[str]): The SS58 address(es) of the hotkey(s)
            block_hash (str): The hash of the block number to retrieve the stake from.
            reuse_block (bool): Whether to reuse the last-used block hash when retrieving info.

        Returns:
            Dict {address: Balance objects}.
        """
        results = await self.substrate.query_multiple(
            params=[s for s in ss58_addresses],
            module="SubtensorModule",
            storage_function="TotalHotkeyStake",
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        return {k: Balance.from_rao(r or 0) for (k, r) in results.items()}

    async def get_netuids_for_hotkey(
        self,
        hotkey_ss58: str,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[int]:
        """
        Retrieves a list of subnet UIDs (netuids) for which a given hotkey is a member. This function identifies the specific subnets within the Bittensor network where the neuron associated with the hotkey is active.

        Args:
            hotkey_ss58 (str): The ``SS58`` address of the neuron's hotkey.
            block_hash (Optional[str]): The hash of the blockchain block number at which to perform the query.
            reuse_block (Optional[bool]): Whether to reuse the last-used block hash when retrieving info.

        Returns:
            A list of netuids where the neuron is a member.
        """

        result = await self.substrate.query_map(
            module="SubtensorModule",
            storage_function="IsNetworkMember",
            params=[hotkey_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        return (
            [record[0] async for record in result if record[1]]
            if result and hasattr(result, "records")
            else []
        )

    async def subnet_exists(
        self, netuid: int, block_hash: Optional[str] = None, reuse_block: bool = False
    ) -> bool:
        """
        Checks if a subnet with the specified unique identifier (netuid) exists within the Bittensor network.

        Args:
            netuid (int): The unique identifier of the subnet.
            block_hash (Optional[str]): The hash of the blockchain block number at which to check the subnet existence.
            reuse_block (bool): Whether to reuse the last-used block hash.

        Returns:
            `True` if the subnet exists, `False` otherwise.

        This function is critical for verifying the presence of specific subnets in the network,
        enabling a deeper understanding of the network's structure and composition.
        """
        result = await self.substrate.query(
            module="SubtensorModule",
            storage_function="NetworksAdded",
            params=[netuid],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        return result

    async def filter_netuids_by_registered_hotkeys(
        self,
        all_netuids: Iterable[int],
        filter_for_netuids: Iterable[int],
        all_hotkeys: Iterable[Wallet],
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[int]:
        """
        Filters a given list of all netuids for certain specified netuids and hotkeys

        Args:
            all_netuids (Iterable[int]): A list of netuids to filter.
            filter_for_netuids (Iterable[int]): A subset of all_netuids to filter from the main list
            all_hotkeys (Iterable[Wallet]): Hotkeys to filter from the main list
            block_hash (str): hash of the blockchain block number at which to perform the query.
            reuse_block (bool): whether to reuse the last-used blockchain hash when retrieving info.

        Returns:
            The filtered list of netuids.
        """
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

    async def get_existential_deposit(
        self, block_hash: Optional[str] = None, reuse_block: bool = False
    ) -> Balance:
        """
        Retrieves the existential deposit amount for the Bittensor blockchain.
        The existential deposit is the minimum amount of TAO required for an account to exist on the blockchain.
        Accounts with balances below this threshold can be reaped to conserve network resources.

        Args:
            block_hash (str): Block hash at which to query the deposit amount. If `None`, the current block is used.
            reuse_block (bool): Whether to reuse the last-used blockchain block hash.

        Returns:
            The existential deposit amount.

        The existential deposit is a fundamental economic parameter in the Bittensor network, ensuring efficient use of storage and preventing the proliferation of dust accounts.
        """
        result = await self.substrate.get_constant(
            module_name="Balances",
            constant_name="ExistentialDeposit",
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )

        if result is None:
            raise Exception("Unable to retrieve existential deposit amount.")

        return Balance.from_rao(result)

    async def neurons(
        self, netuid: int, block_hash: Optional[str] = None, reuse_block: bool = False
    ) -> list[NeuronInfo]:
        """
        Retrieves a list of all neurons within a specified subnet of the Bittensor network.
        This function provides a snapshot of the subnet's neuron population, including each neuron's attributes and network interactions.

        Args:
            netuid (int): The unique identifier of the subnet.
            block_hash (str): The hash of the blockchain block number for the query.
            reuse_block (bool): Whether to reuse the last-used blockchain block hash.

        Returns:
            A list of NeuronInfo objects detailing each neuron's characteristics in the subnet.

        Understanding the distribution and status of neurons within a subnet is key to comprehending the network's decentralized structure and the dynamics of its consensus and governance processes.
        """
        hex_bytes_result = await self.query_runtime_api(
            runtime_api="NeuronInfoRuntimeApi",
            method="get_neurons",
            params=[
                netuid
            ],
            block_hash=block_hash,
            reuse_block=reuse_block,
        )

        if hex_bytes_result is None:
            return []

        return NeuronInfo.list_from_vec_u8(hex_to_bytes(hex_bytes_result))

    async def neurons_lite(
        self, netuid: int, block_hash: Optional[str] = None, reuse_block: bool = False
    ) -> list[NeuronInfoLite]:
        """
        Retrieves a list of neurons in a 'lite' format from a specific subnet of the Bittensor network.
        This function provides a streamlined view of the neurons, focusing on key attributes such as stake and network participation.

        Args:
            netuid (int): The unique identifier of the subnet.
            block_hash (str): The hash of the blockchain block number for the query.
            reuse_block (bool): Whether to reuse the last-used blockchain block hash.

        Returns:
            A list of simplified neuron information for the subnet.

        This function offers a quick overview of the neuron population within a subnet, facilitating efficient analysis of the network's decentralized structure and neuron dynamics.
        """
        hex_bytes_result = await self.query_runtime_api(
            runtime_api="NeuronInfoRuntimeApi",
            method="get_neurons_lite",
            params=[
                netuid
            ],  # TODO check to see if this can accept more than one at a time
            block_hash=block_hash,
            reuse_block=reuse_block,
        )

        if hex_bytes_result is None:
            return []

        return NeuronInfoLite.list_from_vec_u8(hex_to_bytes(hex_bytes_result))

    async def get_neuron_for_pubkey_and_subnet(
        self,
        hotkey_ss58: str,
        netuid: int,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> "NeuronInfo":
        """
        Retrieves information about a neuron based on its public key (hotkey SS58 address) and the specific subnet UID (netuid). This function provides detailed neuron information for a particular subnet within the Bittensor network.

        Args:
            hotkey_ss58 (str): The ``SS58`` address of the neuron's hotkey.
            netuid (int): The unique identifier of the subnet.
            block_hash (Optional[int]): The blockchain block number at which to perform the query.
            reuse_block (bool): Whether to reuse the last-used blockchain block hash.

        Returns:
            Optional[bittensor.core.chain_data.neuron_info.NeuronInfo]: Detailed information about the neuron if found, ``None`` otherwise.

        This function is crucial for accessing specific neuron data and understanding its status, stake, and other attributes within a particular subnet of the Bittensor ecosystem.
        """
        uid = await self.substrate.query(
            module="SubtensorModule",
            storage_function="Uids",
            params=[netuid, hotkey_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        if uid is None:
            return NeuronInfo.get_null_neuron()

        params = [netuid, uid]
        json_body = await self.substrate.rpc_request(
            method="neuronInfo_getNeuron",
            params=params,
        )

        if not (result := json_body.get("result", None)):
            return NeuronInfo.get_null_neuron()

        return NeuronInfo.from_vec_u8(bytes(result))

    async def neuron_for_uid(
        self, uid: Optional[int], netuid: int, block_hash: Optional[str] = None
    ) -> NeuronInfo:
        """
        Retrieves detailed information about a specific neuron identified by its unique identifier (UID) within a specified subnet (netuid) of the Bittensor network. This function provides a comprehensive view of a neuron's attributes, including its stake, rank, and operational status.

        Args:
            uid (int): The unique identifier of the neuron.
            netuid (int): The unique identifier of the subnet.
            block_hash (str): The hash of the blockchain block number for the query.

        Returns:
            Detailed information about the neuron if found, a null neuron otherwise

        This function is crucial for analyzing individual neurons' contributions and status within a specific subnet, offering insights into their roles in the network's consensus and validation mechanisms.
        """
        if uid is None:
            return NeuronInfo.get_null_neuron()

        params = [netuid, uid, block_hash] if block_hash else [netuid, uid]
        json_body = await self.substrate.rpc_request(
            method="neuronInfo_getNeuron",
            params=params,  # custom rpc method
        )
        if not (result := json_body.get("result", None)):
            return NeuronInfo.get_null_neuron()

        bytes_result = bytes(result)
        return NeuronInfo.from_vec_u8(bytes_result)

    async def get_delegated(
        self,
        coldkey_ss58: str,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> list[tuple[DelegateInfo, Balance]]:
        """
        Retrieves a list of delegates and their associated stakes for a given coldkey. This function identifies the delegates that a specific account has staked tokens on.

        Args:
            coldkey_ss58 (str): The `SS58` address of the account's coldkey.
            block_hash (Optional[str]): The hash of the blockchain block number for the query.
            reuse_block (bool): Whether to reuse the last-used blockchain block hash.

        Returns:
            A list of tuples, each containing a delegate's information and staked amount.

        This function is important for account holders to understand their stake allocations and their involvement in the network's delegation and consensus mechanisms.
        """

        block_hash = (
            block_hash
            if block_hash
            else (self.substrate.last_block_hash if reuse_block else None)
        )
        encoded_coldkey = ss58_to_vec_u8(coldkey_ss58)
        json_body = await self.substrate.rpc_request(
            method="delegateInfo_getDelegated",
            params=([block_hash, encoded_coldkey] if block_hash else [encoded_coldkey]),
        )

        if not (result := json_body.get("result")):
            return []

        return DelegateInfo.delegated_list_from_vec_u8(bytes(result))

    async def query_identity(
        self,
        key: str,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> dict:
        """
        Queries the identity of a neuron on the Bittensor blockchain using the given key. This function retrieves detailed identity information about a specific neuron, which is a crucial aspect of the network's decentralized identity and governance system.

        Args:
            key (str): The key used to query the neuron's identity, typically the neuron's SS58 address.
            block_hash (str): The hash of the blockchain block number at which to perform the query.
            reuse_block (bool): Whether to reuse the last-used blockchain block hash.

        Returns:
            An object containing the identity information of the neuron if found, ``None`` otherwise.

        The identity information can include various attributes such as the neuron's stake, rank, and other network-specific details, providing insights into the neuron's role and status within the Bittensor network.

        Note:
            See the `Bittensor CLI documentation <https://docs.bittensor.com/reference/btcli>`_ for supported identity parameters.
        """

        identity_info = await self.substrate.query(
            module="Registry",
            storage_function="IdentityOf",
            params=[key],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        try:
            return _decode_hex_identity_dict(identity_info["info"])
        except TypeError:
            return {}

    async def weights(
        self, netuid: int, block_hash: Optional[str] = None
    ) -> list[tuple[int, list[tuple[int, int]]]]:
        """
        Retrieves the weight distribution set by neurons within a specific subnet of the Bittensor network.
        This function maps each neuron's UID to the weights it assigns to other neurons, reflecting the network's trust and value assignment mechanisms.

        Args:
            netuid (int): The network UID of the subnet to query.
            block_hash (str): The hash of the blockchain block for the query.

        Returns:
            A list of tuples mapping each neuron's UID to its assigned weights.

        The weight distribution is a key factor in the network's consensus algorithm and the ranking of neurons, influencing their influence and reward allocation within the subnet.
        """
        # TODO look into seeing if we can speed this up with storage query
        w_map_encoded = await self.substrate.query_map(
            module="SubtensorModule",
            storage_function="Weights",
            params=[netuid],
            block_hash=block_hash,
        )
        w_map = [(uid, w or []) async for uid, w in w_map_encoded]

        return w_map

    async def bonds(
        self, netuid: int, block_hash: Optional[str] = None
    ) -> list[tuple[int, list[tuple[int, int]]]]:
        """
        Retrieves the bond distribution set by neurons within a specific subnet of the Bittensor network.
        Bonds represent the investments or commitments made by neurons in one another, indicating a level of trust and perceived value. This bonding mechanism is integral to the network's market-based approach to measuring and rewarding machine intelligence.

        Args:
            netuid (int): The network UID of the subnet to query.
            block_hash (Optional[str]): The hash of the blockchain block number for the query.

        Returns:
            List of tuples mapping each neuron's UID to its bonds with other neurons.

        Understanding bond distributions is crucial for analyzing the trust dynamics and market behavior within the subnet. It reflects how neurons recognize and invest in each other's intelligence and contributions, supporting diverse and niche systems within the Bittensor ecosystem.
        """
        b_map_encoded = await self.substrate.query_map(
            module="SubtensorModule",
            storage_function="Bonds",
            params=[netuid],
            block_hash=block_hash,
        )
        b_map = [(uid, b) async for uid, b in b_map_encoded]

        return b_map

    async def does_hotkey_exist(
        self,
        hotkey_ss58: str,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> bool:
        """
        Returns true if the hotkey is known by the chain and there are accounts.

        Args:
            hotkey_ss58 (str): The SS58 address of the hotkey.
            block_hash (Optional[str]): The hash of the block number to check the hotkey against.
            reuse_block (bool): Whether to reuse the last-used blockchain hash.

        Returns:
            `True` if the hotkey is known by the chain and there are accounts, `False` otherwise.
        """
        _result = await self.substrate.query(
            module="SubtensorModule",
            storage_function="Owner",
            params=[hotkey_ss58],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        result = decode_account_id(_result[0])
        return_val = (
            False
            if result is None
            else result != "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"
        )
        return return_val

    async def get_hotkey_owner(
        self, hotkey_ss58: str, block_hash: str
    ) -> Optional[str]:
        """
        Retrieves the owner of the given hotkey at a specific block hash.
        This function queries the blockchain for the owner of the provided hotkey. If the hotkey does not exist at the specified block hash, it returns None.

        Args:
            hotkey_ss58 (str): The SS58 address of the hotkey.
            block_hash (str): The hash of the block at which to check the hotkey ownership.

        Returns:
            Optional[str]: The SS58 address of the owner if the hotkey exists, or None if it doesn't.
        """
        hk_owner_query = await self.substrate.query(
            module="SubtensorModule",
            storage_function="Owner",
            params=[hotkey_ss58],
            block_hash=block_hash,
        )
        val = decode_account_id(hk_owner_query[0])
        if val:
            exists = await self.does_hotkey_exist(hotkey_ss58, block_hash=block_hash)
        else:
            exists = False
        hotkey_owner = val if exists else None
        return hotkey_owner

    async def sign_and_send_extrinsic(
        self,
        call: "GenericCall",
        wallet: "Wallet",
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> tuple[bool, str]:
        """
        Helper method to sign and submit an extrinsic call to chain.

        Args:
            call (scalecodec.types.GenericCall): a prepared Call object
            wallet (bittensor_wallet.Wallet): the wallet whose coldkey will be used to sign the extrinsic
            wait_for_inclusion (bool): whether to wait until the extrinsic call is included on the chain
            wait_for_finalization (bool): whether to wait until the extrinsic call is finalized on the chain

        Returns:
            (success, error message)
        """
        extrinsic = await self.substrate.create_signed_extrinsic(
            call=call, keypair=wallet.coldkey
        )  # sign with coldkey
        try:
            response = await self.substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )
            # We only wait here if we expect finalization.
            if not wait_for_finalization and not wait_for_inclusion:
                return True, ""
            await response.process_events()
            if await response.is_success:
                return True, ""
            else:
                return False, format_error_message(
                    await response.error_message, substrate=self.substrate
                )
        except SubstrateRequestException as e:
            return False, format_error_message(e, substrate=self.substrate)

    async def get_children(self, hotkey: str, netuid: int) -> tuple[bool, list, str]:
        """
        This method retrieves the children of a given hotkey and netuid. It queries the SubtensorModule's ChildKeys storage function to get the children and formats them before returning as a tuple.

        Args:
            hotkey (str): The hotkey value.
            netuid (int): The netuid value.

        Returns:
            A tuple containing a boolean indicating success or failure, a list of formatted children, and an error message (if applicable)
        """
        try:
            children = await self.substrate.query(
                module="SubtensorModule",
                storage_function="ChildKeys",
                params=[hotkey, netuid],
            )
            if children:
                formatted_children = []
                for proportion, child in children:
                    # Convert U64 to int
                    formatted_child = decode_account_id(child[0])
                    int_proportion = int(proportion)
                    formatted_children.append((int_proportion, formatted_child))
                return True, formatted_children, ""
            else:
                return True, [], ""
        except SubstrateRequestException as e:
            return False, [], format_error_message(e, self.substrate)

    async def get_subnet_hyperparameters(
        self, netuid: int, block_hash: Optional[str] = None
    ) -> Optional[Union[list, SubnetHyperparameters]]:
        """
        Retrieves the hyperparameters for a specific subnet within the Bittensor network. These hyperparameters define the operational settings and rules governing the subnet's behavior.

        Args:
            netuid (int): The network UID of the subnet to query.
            block_hash (Optional[str]): The hash of the blockchain block number for the query.

        Returns:
            The subnet's hyperparameters, or `None` if not available.

        Understanding the hyperparameters is crucial for comprehending how subnets are configured and managed, and how they interact with the network's consensus and incentive mechanisms.
        """
        hex_bytes_result = await self.query_runtime_api(
            runtime_api="SubnetInfoRuntimeApi",
            method="get_subnet_hyperparams",
            params=[netuid],
            block_hash=block_hash,
        )

        if hex_bytes_result is None:
            return []

        return SubnetHyperparameters.from_vec_u8(hex_to_bytes(hex_bytes_result))

    async def get_vote_data(
        self,
        proposal_hash: str,
        block_hash: Optional[str] = None,
        reuse_block: bool = False,
    ) -> Optional["ProposalVoteData"]:
        """
        Retrieves the voting data for a specific proposal on the Bittensor blockchain. This data includes information about how senate members have voted on the proposal.

        Args:
            proposal_hash (str): The hash of the proposal for which voting data is requested.
            block_hash (Optional[str]): The hash of the blockchain block number to query the voting data.
            reuse_block (bool): Whether to reuse the last-used blockchain block hash.

        Returns:
            An object containing the proposal's voting data, or `None` if not found.

        This function is important for tracking and understanding the decision-making processes within the Bittensor network, particularly how proposals are received and acted upon by the governing body.
        """
        vote_data = await self.substrate.query(
            module="Triumvirate",
            storage_function="Voting",
            params=[proposal_hash],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
        if vote_data is None:
            return None
        else:
            return ProposalVoteData(vote_data)

    async def get_delegate_identities(
        self, block_hash: Optional[str] = None
    ) -> dict[str, DelegatesDetails]:
        """
        Fetches delegates identities from the chain and GitHub. Preference is given to chain data, and missing info is filled-in by the info from GitHub. At some point, we want to totally move away from fetching this info from GitHub, but chain data is still limited in that regard.

        Args:
            block_hash (str): the hash of the blockchain block for the query

        Returns:
            Dict {ss58: DelegatesDetails, ...}

        """
        timeout = aiohttp.ClientTimeout(10.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            identities_info, response = await asyncio.gather(
                self.substrate.query_map(
                    module="Registry",
                    storage_function="IdentityOf",
                    block_hash=block_hash,
                ),
                session.get(DELEGATES_DETAILS_URL),
            )

            all_delegates_details = {
                decode_account_id(ss58_address[0]): DelegatesDetails.from_chain_data(
                    decode_hex_identity_dict(identity["info"])
                )
                for ss58_address, identity in identities_info
            }

            if response.ok:
                all_delegates: dict[str, Any] = await response.json(content_type=None)

                for delegate_hotkey, delegate_details in all_delegates.items():
                    delegate_info = all_delegates_details.setdefault(
                        delegate_hotkey,
                        DelegatesDetails(
                            display=delegate_details.get("name", ""),
                            web=delegate_details.get("url", ""),
                            additional=delegate_details.get("description", ""),
                            pgp_fingerprint=delegate_details.get("fingerprint", ""),
                        ),
                    )
                    delegate_info.display = (
                        delegate_info.display or delegate_details.get("name", "")
                    )
                    delegate_info.web = delegate_info.web or delegate_details.get(
                        "url", ""
                    )
                    delegate_info.additional = (
                        delegate_info.additional
                        or delegate_details.get("description", "")
                    )
                    delegate_info.pgp_fingerprint = (
                        delegate_info.pgp_fingerprint
                        or delegate_details.get("fingerprint", "")
                    )

        return all_delegates_details

    async def is_hotkey_registered(self, netuid: int, hotkey_ss58: str) -> bool:
        """Checks to see if the hotkey is registered on a given netuid"""
        result = await self.substrate.query(
            module="SubtensorModule",
            storage_function="Uids",
            params=[netuid, hotkey_ss58],
        )
        if result is not None:
            return True
        else:
            return False

    async def get_uid_for_hotkey_on_subnet(
        self, hotkey_ss58: str, netuid: int, block_hash: Optional[str] = None
    ) -> Optional[int]:
        """
        Retrieves the unique identifier (UID) for a neuron's hotkey on a specific subnet.

        Args:
            hotkey_ss58 (str): The ``SS58`` address of the neuron's hotkey.
            netuid (int): The unique identifier of the subnet.
            block_hash (Optional[str]): The blockchain block_hash representation of the block id.

        Returns:
            Optional[int]: The UID of the neuron if it is registered on the subnet, ``None`` otherwise.

        The UID is a critical identifier within the network, linking the neuron's hotkey to its operational and governance activities on a particular subnet.
        """
        result = await self.substrate.query(
            module="SubtensorModule",
            storage_function="Uids",
            params=[netuid, hotkey_ss58],
            block_hash=block_hash,
        )
        return result

    async def weights_rate_limit(self, netuid: int) -> Optional[int]:
        """
        Returns network WeightsSetRateLimit hyperparameter.

        Args:
            netuid (int): The unique identifier of the subnetwork.

        Returns:
            Optional[int]: The value of the WeightsSetRateLimit hyperparameter, or ``None`` if the subnetwork does not exist or the parameter is not found.
        """
        call = await self.get_hyperparameter(
            param_name="WeightsSetRateLimit", netuid=netuid
        )
        return None if call is None else int(call)

    async def blocks_since_last_update(self, netuid: int, uid: int) -> Optional[int]:
        """
        Returns the number of blocks since the last update for a specific UID in the subnetwork.

        Args:
            netuid (int): The unique identifier of the subnetwork.
            uid (int): The unique identifier of the neuron.

        Returns:
            Optional[int]: The number of blocks since the last update, or ``None`` if the subnetwork or UID does not exist.
        """
        call = await self.get_hyperparameter(param_name="LastUpdate", netuid=netuid)
        return None if call is None else await self.get_current_block() - int(call[uid])

    # Extrinsics =======================================================================================================

    async def transfer(
        self,
        wallet: "Wallet",
        destination: str,
        amount: float,
        transfer_all: bool,
    ) -> bool:
        """
        Transfer token of amount to destination.

        Args:
            wallet (bittensor_wallet.Wallet): Source wallet for the transfer.
            destination (str): Destination address for the transfer.
            amount (float): Amount of tokens to transfer.
            transfer_all (bool): Flag to transfer all tokens.

        Returns:
            `True` if the transferring was successful, otherwise `False`.
        """
        return await transfer_extrinsic(
            self,
            wallet,
            destination,
            Balance.from_tao(amount),
            transfer_all,
        )

    async def register(
        self,
        wallet: "Wallet",
        netuid: int,
        block_hash: Optional[str] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> bool:
        """
        Register neuron by recycling some TAO.

        Args:
            wallet (bittensor_wallet.Wallet): Bittensor wallet instance.
            netuid (int): Subnet uniq id.
            block_hash (Optional[str]): The hash of the blockchain block for the query.
            wait_for_inclusion (bool): Waits for the transaction to be included in a block. Default is ``False``.
            wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain. Default is ``False``.

        Returns:
            `True` if registration was successful, otherwise `False`.
        """
        logging.info(
            f"Registering on netuid [blue]0[/blue] on network: [blue]{self.network}[/blue]"
        )

        # Check current recycle amount
        logging.info("Fetching recycle amount & balance.")
        block_hash = block_hash if block_hash else await self.get_block_hash()
        recycle_call, balance_ = await asyncio.gather(
            self.get_hyperparameter(param_name="Burn", netuid=netuid, reuse_block=True),
            self.get_balance(wallet.coldkeypub.ss58_address, block_hash=block_hash),
        )
        current_recycle = Balance.from_rao(int(recycle_call))
        try:
            balance: Balance = balance_[wallet.coldkeypub.ss58_address]
        except TypeError as e:
            logging.error(f"Unable to retrieve current recycle. {e}")
            return False
        except KeyError:
            logging.error("Unable to retrieve current balance.")
            return False

        # Check balance is sufficient
        if balance < current_recycle:
            logging.error(
                f"[red]Insufficient balance {balance} to register neuron. Current recycle is {current_recycle} TAO[/red]."
            )
            return False

        return await root_register_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    async def pow_register(
        self: "AsyncSubtensor",
        wallet: "Wallet",
        netuid: int,
        processors: int,
        update_interval: int,
        output_in_place: bool,
        verbose: bool,
        use_cuda: bool,
        dev_id: Union[list[int], int],
        threads_per_block: int,
    ):
        """Register neuron."""
        return await register_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuid=netuid,
            tpb=threads_per_block,
            update_interval=update_interval,
            num_processes=processors,
            cuda=use_cuda,
            dev_id=dev_id,
            output_in_place=output_in_place,
            log_verbose=verbose,
        )

    async def set_weights(
        self,
        wallet: "Wallet",
        netuid: int,
        uids: Union[NDArray[np.int64], "torch.LongTensor", list],
        weights: Union[NDArray[np.float32], "torch.FloatTensor", list],
        version_key: int = version_as_int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
        max_retries: int = 5,
    ):
        """
        Sets the inter-neuronal weights for the specified neuron. This process involves specifying the influence or trust a neuron places on other neurons in the network, which is a fundamental aspect of Bittensor's decentralized learning architecture.

        Args:
            wallet (bittensor_wallet.Wallet): The wallet associated with the neuron setting the weights.
            netuid (int): The unique identifier of the subnet.
            uids (Union[NDArray[np.int64], torch.LongTensor, list]): The list of neuron UIDs that the weights are being set for.
            weights (Union[NDArray[np.float32], torch.FloatTensor, list]): The corresponding weights to be set for each UID.
            version_key (int): Version key for compatibility with the network.  Default is ``int representation of Bittensor version.``.
            wait_for_inclusion (bool): Waits for the transaction to be included in a block. Default is ``False``.
            wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain. Default is ``False``.
            max_retries (int): The number of maximum attempts to set weights. Default is ``5``.

        Returns:
            tuple[bool, str]: ``True`` if the setting of weights is successful, False otherwise. And `msg`, a string value describing the success or potential error.

        This function is crucial in shaping the network's collective intelligence, where each neuron's learning and contribution are influenced by the weights it sets towards others【81†source】.
        """
        uid = await self.get_uid_for_hotkey_on_subnet(
            wallet.hotkey.ss58_address, netuid
        )
        retries = 0
        success = False
        message = "No attempt made. Perhaps it is too soon to set weights!"
        while retries < max_retries and await self.blocks_since_last_update(
            netuid, uid
        ) > await self.weights_rate_limit(netuid):
            try:
                logging.info(
                    f"Setting weights for subnet #[blue]{netuid}[/blue]. Attempt [blue]{retries + 1} of {max_retries}[/blue]."
                )
                success, message = await set_weights_extrinsic(
                    subtensor=self,
                    wallet=wallet,
                    netuid=netuid,
                    uids=uids,
                    weights=weights,
                    version_key=version_key,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )
            except Exception as e:
                logging.error(f"Error setting weights: {e}")
            finally:
                retries += 1

        return success, message

    async def root_set_weights(
        self,
        wallet: "Wallet",
        netuids: list[int],
        weights: list[float],
    ) -> bool:
        """
        Set weights for root network.

        Args:
            wallet (bittensor_wallet.Wallet): bittensor wallet instance.
            netuids (list[int]): The list of subnet uids.
            weights (list[float]): The list of weights to be set.

        Returns:
            `True` if the setting of weights is successful, `False` otherwise.
        """
        netuids_ = np.array(netuids, dtype=np.int64)
        weights_ = np.array(weights, dtype=np.float32)
        logging.info(f"Setting weights in network: [blue]{self.network}[/blue]")
        # Run the set weights operation.
        return await set_root_weights_extrinsic(
            subtensor=self,
            wallet=wallet,
            netuids=netuids_,
            weights=weights_,
            version_key=0,
            wait_for_finalization=True,
            wait_for_inclusion=True,
        )

    async def commit_weights(
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
        """
        Commits a hash of the neuron's weights to the Bittensor blockchain using the provided wallet.
        This action serves as a commitment or snapshot of the neuron's current weight distribution.

        Args:
            wallet (bittensor_wallet.Wallet): The wallet associated with the neuron committing the weights.
            netuid (int): The unique identifier of the subnet.
            salt (list[int]): list of randomly generated integers as salt to generated weighted hash.
            uids (np.ndarray): NumPy array of neuron UIDs for which weights are being committed.
            weights (np.ndarray): NumPy array of weight values corresponding to each UID.
            version_key (int): Version key for compatibility with the network. Default is ``int representation of Bittensor version.``.
            wait_for_inclusion (bool): Waits for the transaction to be included in a block. Default is ``False``.
            wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain. Default is ``False``.
            max_retries (int): The number of maximum attempts to commit weights. Default is ``5``.

        Returns:
            tuple[bool, str]: ``True`` if the weight commitment is successful, False otherwise. And `msg`, a string value describing the success or potential error.

        This function allows neurons to create a tamper-proof record of their weight distribution at a specific point in time, enhancing transparency and accountability within the Bittensor network.
        """
        retries = 0
        success = False
        message = "No attempt made. Perhaps it is too soon to commit weights!"

        logging.info(
            f"Committing weights with params: netuid={netuid}, uids={uids}, weights={weights}, version_key={version_key}"
        )

        # Generate the hash of the weights
        commit_hash = generate_weight_hash(
            address=wallet.hotkey.ss58_address,
            netuid=netuid,
            uids=list(uids),
            values=list(weights),
            salt=salt,
            version_key=version_key,
        )

        while retries < max_retries:
            try:
                success, message = await commit_weights_extrinsic(
                    subtensor=self,
                    wallet=wallet,
                    netuid=netuid,
                    commit_hash=commit_hash,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )
                if success:
                    break
            except Exception as e:
                logging.error(f"Error committing weights: {e}")
            finally:
                retries += 1

        return success, message
