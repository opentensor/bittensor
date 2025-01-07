import asyncio
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Iterable, Optional, Union

import numpy as np
from numpy.typing import NDArray

from bittensor.core.async_subtensor import AsyncSubtensor
from bittensor.utils.substrate_interface import SubstrateInterface
from bittensor.core.settings import version_as_int
from bittensor.utils import execute_coroutine, torch, get_event_loop

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.async_subtensor import ProposalVoteData
    from bittensor.core.axon import Axon
    from bittensor.core.config import Config
    from bittensor.core.metagraph import Metagraph
    from bittensor.core.chain_data.delegate_info import DelegateInfo
    from bittensor.core.chain_data.neuron_info import NeuronInfo
    from bittensor.core.chain_data.neuron_info_lite import NeuronInfoLite
    from bittensor.core.chain_data.stake_info import StakeInfo
    from bittensor.core.chain_data.subnet_hyperparameters import SubnetHyperparameters
    from bittensor.core.chain_data.subnet_info import SubnetInfo
    from bittensor.utils.balance import Balance
    from bittensor.utils import Certificate
    from bittensor.utils.substrate_interface import QueryMapResult
    from bittensor.utils.delegates_details import DelegatesDetails
    from scalecodec.types import ScaleType


class Subtensor:
    # get static methods from AsyncSubtensor
    config = AsyncSubtensor.config
    setup_config = AsyncSubtensor.setup_config
    help = AsyncSubtensor.help
    add_args = AsyncSubtensor.add_args
    determine_chain_endpoint_and_network = (
        AsyncSubtensor.determine_chain_endpoint_and_network
    )

    def __init__(
        self,
        network: Optional[str] = None,
        config: Optional["Config"] = None,
        _mock: bool = False,
        log_verbose: bool = False,
    ):
        self.event_loop = get_event_loop()
        self.network = network
        self._config = config
        self.log_verbose = log_verbose
        self.async_subtensor = AsyncSubtensor(
            network=network,
            config=config,
            log_verbose=log_verbose,
            event_loop=self.event_loop,
            _mock=_mock,
        )

        self.substrate = SubstrateInterface(
            url=self.async_subtensor.chain_endpoint,
            _mock=_mock,
            substrate=self.async_subtensor.substrate,
        )
        self.chain_endpoint = self.async_subtensor.chain_endpoint

    def __str__(self):
        return self.async_subtensor.__str__()

    def __repr__(self):
        return self.async_subtensor.__repr__()

    def execute_coroutine(self, coroutine) -> Any:
        return execute_coroutine(coroutine, self.event_loop)

    def close(self):
        execute_coroutine(self.async_subtensor.close())

    # Subtensor queries ===========================================================================================

    def query_constant(
        self, module_name: str, constant_name: str, block: Optional[int] = None
    ) -> Optional["ScaleType"]:
        return self.execute_coroutine(
            self.async_subtensor.query_constant(
                module_name=module_name, constant_name=constant_name, block=block
            )
        )

    def query_map(
        self,
        module: str,
        name: str,
        block: Optional[int] = None,
        params: Optional[list] = None,
    ) -> "QueryMapResult":
        return self.execute_coroutine(
            self.async_subtensor.query_map(
                module=module, name=name, block=block, params=params
            )
        )

    def query_map_subtensor(
        self, name: str, block: Optional[int] = None, params: Optional[list] = None
    ) -> "QueryMapResult":
        return self.execute_coroutine(
            self.async_subtensor.query_map_subtensor(
                name=name, block=block, params=params
            )
        )

    def query_module(
        self,
        module: str,
        name: str,
        block: Optional[int] = None,
        params: Optional[list] = None,
    ) -> "ScaleType":
        return self.execute_coroutine(
            self.async_subtensor.query_module(
                module=module,
                name=name,
                block=block,
                params=params,
            )
        )

    def query_runtime_api(
        self,
        runtime_api: str,
        method: str,
        params: Optional[Union[list[int], dict[str, int]]] = None,
        block: Optional[int] = None,
    ) -> Optional[str]:
        return self.execute_coroutine(
            coroutine=self.async_subtensor.query_runtime_api(
                runtime_api=runtime_api,
                method=method,
                params=params,
                block=block,
            )
        )

    def query_subtensor(
        self, name: str, block: Optional[int] = None, params: Optional[list] = None
    ) -> "ScaleType":
        return self.execute_coroutine(
            self.async_subtensor.query_subtensor(name=name, block=block, params=params)
        )

    def state_call(
        self, method: str, data: str, block: Optional[int] = None
    ) -> dict[Any, Any]:
        return self.execute_coroutine(
            self.async_subtensor.state_call(method=method, data=data, block=block)
        )

    # Common subtensor calls ===========================================================================================

    @property
    def block(self) -> int:
        return self.get_current_block()

    def blocks_since_last_update(self, netuid: int, uid: int) -> Optional[int]:
        return self.execute_coroutine(
            self.async_subtensor.blocks_since_last_update(netuid=netuid, uid=uid)
        )

    def bonds(
        self, netuid: int, block: Optional[int] = None
    ) -> list[tuple[int, list[tuple[int, int]]]]:
        return self.execute_coroutine(
            self.async_subtensor.bonds(netuid=netuid, block=block),
        )

    def commit(self, wallet, netuid: int, data: str) -> bool:
        return self.execute_coroutine(
            self.async_subtensor.commit(wallet=wallet, netuid=netuid, data=data)
        )

    def commit_reveal_enabled(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[bool]:
        return self.execute_coroutine(
            self.async_subtensor.commit_reveal_enabled(netuid=netuid, block=block)
        )

    def difficulty(self, netuid: int, block: Optional[int] = None) -> Optional[int]:
        return self.execute_coroutine(
            self.async_subtensor.difficulty(netuid=netuid, block=block),
        )

    def does_hotkey_exist(self, hotkey_ss58: str, block: Optional[int] = None) -> bool:
        return self.execute_coroutine(
            self.async_subtensor.does_hotkey_exist(hotkey_ss58=hotkey_ss58, block=block)
        )

    def get_all_subnets_info(self, block: Optional[int] = None) -> list["SubnetInfo"]:
        return self.execute_coroutine(
            self.async_subtensor.get_all_subnets_info(block=block),
        )

    def get_balance(self, address: str, block: Optional[int] = None) -> "Balance":
        return self.execute_coroutine(
            self.async_subtensor.get_balance(address, block=block),
        )

    def get_balances(
        self,
        *addresses: str,
        block: Optional[int] = None,
    ) -> dict[str, "Balance"]:
        return self.execute_coroutine(
            self.async_subtensor.get_balances(*addresses, block=block),
        )

    def get_current_block(self) -> int:
        return self.execute_coroutine(
            coroutine=self.async_subtensor.get_current_block(),
        )

    @lru_cache(maxsize=128)
    def get_block_hash(self, block: Optional[int] = None) -> str:
        return self.execute_coroutine(
            coroutine=self.async_subtensor.get_block_hash(block=block),
        )

    def get_children(
        self, hotkey: str, netuid: int, block: Optional[int] = None
    ) -> tuple[bool, list, str]:
        return self.execute_coroutine(
            self.async_subtensor.get_children(
                hotkey=hotkey, netuid=netuid, block=block
            ),
        )

    def get_commitment(self, netuid: int, uid: int, block: Optional[int] = None) -> str:
        return self.execute_coroutine(
            self.async_subtensor.get_commitment(netuid=netuid, uid=uid, block=block),
        )

    def get_current_weight_commit_info(
        self, netuid: int, block: Optional[int] = None
    ) -> list:
        return self.execute_coroutine(
            self.async_subtensor.get_current_weight_commit_info(
                netuid=netuid, block=block
            ),
        )

    def get_delegate_by_hotkey(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> Optional["DelegateInfo"]:
        return self.execute_coroutine(
            self.async_subtensor.get_delegate_by_hotkey(
                hotkey_ss58=hotkey_ss58, block=block
            ),
        )

    def get_delegate_identities(
        self, block: Optional[int] = None
    ) -> dict[str, "DelegatesDetails"]:
        return self.execute_coroutine(
            self.async_subtensor.get_delegate_identities(block=block),
        )

    def get_delegate_take(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> Optional[float]:
        return self.execute_coroutine(
            self.async_subtensor.get_delegate_take(
                hotkey_ss58=hotkey_ss58, block=block
            ),
        )

    def get_delegated(
        self, coldkey_ss58: str, block: Optional[int] = None
    ) -> list[tuple["DelegateInfo", "Balance"]]:
        return self.execute_coroutine(
            self.async_subtensor.get_delegated(coldkey_ss58=coldkey_ss58, block=block),
        )

    def get_delegates(self, block: Optional[int] = None) -> list["DelegateInfo"]:
        return self.execute_coroutine(
            self.async_subtensor.get_delegates(block=block),
        )

    def get_existential_deposit(
        self, block: Optional[int] = None
    ) -> Optional["Balance"]:
        return self.execute_coroutine(
            self.async_subtensor.get_existential_deposit(block=block),
        )

    def get_hotkey_owner(
        self, hotkey_ss58: str, block: Optional[int] = None
    ) -> Optional[str]:
        return self.execute_coroutine(
            self.async_subtensor.get_hotkey_owner(hotkey_ss58=hotkey_ss58, block=block),
        )

    def get_minimum_required_stake(self) -> "Balance":
        return self.execute_coroutine(
            self.async_subtensor.get_minimum_required_stake(),
        )

    def get_netuids_for_hotkey(
        self, hotkey_ss58: str, block: Optional[int] = None, reuse_block: bool = False
    ) -> list[int]:
        return self.execute_coroutine(
            self.async_subtensor.get_netuids_for_hotkey(
                hotkey_ss58=hotkey_ss58, block=block, reuse_block=reuse_block
            ),
        )

    def get_neuron_certificate(
        self, hotkey: str, netuid: int, block: Optional[int] = None
    ) -> Optional["Certificate"]:
        return self.execute_coroutine(
            self.async_subtensor.get_neuron_certificate(hotkey, netuid, block=block),
        )

    def get_neuron_for_pubkey_and_subnet(
        self, hotkey_ss58: str, netuid: int, block: Optional[int] = None
    ) -> Optional["NeuronInfo"]:
        return self.execute_coroutine(
            self.async_subtensor.get_neuron_for_pubkey_and_subnet(
                hotkey_ss58, netuid, block=block
            ),
        )

    def get_stake_for_coldkey_and_hotkey(
        self, hotkey_ss58: str, coldkey_ss58: str, block: Optional[int] = None
    ) -> Optional["Balance"]:
        return self.execute_coroutine(
            self.async_subtensor.get_stake_for_coldkey_and_hotkey(
                hotkey_ss58=hotkey_ss58, coldkey_ss58=coldkey_ss58, block=block
            ),
        )

    def get_stake_info_for_coldkey(
        self, coldkey_ss58: str, block: Optional[int] = None
    ) -> list["StakeInfo"]:
        return self.execute_coroutine(
            self.async_subtensor.get_stake_info_for_coldkey(
                coldkey_ss58=coldkey_ss58, block=block
            ),
        )

    def get_subnet_burn_cost(self, block: Optional[int] = None) -> Optional[str]:
        return self.execute_coroutine(
            self.async_subtensor.get_subnet_burn_cost(block=block),
        )

    def get_subnet_hyperparameters(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[Union[list, "SubnetHyperparameters"]]:
        return self.execute_coroutine(
            self.async_subtensor.get_subnet_hyperparameters(netuid=netuid, block=block),
        )

    def get_subnet_reveal_period_epochs(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        return self.execute_coroutine(
            self.async_subtensor.get_subnet_reveal_period_epochs(
                netuid=netuid, block=block
            ),
        )

    def get_subnets(self, block: Optional[int] = None) -> list[int]:
        return self.execute_coroutine(
            self.async_subtensor.get_subnets(block=block),
        )

    def get_total_stake_for_coldkey(
        self, ss58_address: str, block: Optional[int] = None
    ) -> Optional["Balance"]:
        result = self.execute_coroutine(
            self.async_subtensor.get_total_stake_for_coldkey(ss58_address, block=block),
        )
        return next(iter(result.values()), None) if isinstance(result, dict) else None

    def get_total_stake_for_hotkey(
        self, ss58_address: str, block: Optional[int] = None
    ) -> Optional["Balance"]:
        result = self.execute_coroutine(
            self.async_subtensor.get_total_stake_for_hotkey(
                *[ss58_address], block=block
            ),
        )
        return next(iter(result.values()), None) if isinstance(result, dict) else None

    def get_total_subnets(self, block: Optional[int] = None) -> Optional[int]:
        return self.execute_coroutine(
            self.async_subtensor.get_total_subnets(block=block),
        )

    def get_transfer_fee(
        self, wallet: "Wallet", dest: str, value: Union["Balance", float, int]
    ) -> "Balance":
        return self.execute_coroutine(
            self.async_subtensor.get_transfer_fee(
                wallet=wallet, dest=dest, value=value
            ),
        )

    def get_vote_data(
        self, proposal_hash: str, block: Optional[int] = None
    ) -> Optional["ProposalVoteData"]:
        return self.execute_coroutine(
            self.async_subtensor.get_vote_data(
                proposal_hash=proposal_hash, block=block
            ),
        )

    def get_uid_for_hotkey_on_subnet(
        self, hotkey_ss58: str, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        return self.execute_coroutine(
            self.async_subtensor.get_uid_for_hotkey_on_subnet(
                hotkey_ss58=hotkey_ss58, netuid=netuid, block=block
            ),
        )

    def filter_netuids_by_registered_hotkeys(
        self,
        all_netuids: Iterable[int],
        filter_for_netuids: Iterable[int],
        all_hotkeys: Iterable["Wallet"],
        block: Optional[int],
    ) -> list[int]:
        return self.execute_coroutine(
            self.async_subtensor.filter_netuids_by_registered_hotkeys(
                all_netuids=all_netuids,
                filter_for_netuids=filter_for_netuids,
                all_hotkeys=all_hotkeys,
                block=block,
            ),
        )

    def immunity_period(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        return self.execute_coroutine(
            self.async_subtensor.immunity_period(netuid=netuid, block=block),
        )

    def is_hotkey_delegate(self, hotkey_ss58: str, block: Optional[int] = None) -> bool:
        return self.execute_coroutine(
            self.async_subtensor.is_hotkey_delegate(
                hotkey_ss58=hotkey_ss58, block=block
            ),
        )

    def is_hotkey_registered(
        self,
        hotkey_ss58: str,
        netuid: Optional[int] = None,
        block: Optional[int] = None,
    ) -> bool:
        return self.execute_coroutine(
            self.async_subtensor.is_hotkey_registered(
                hotkey_ss58=hotkey_ss58, netuid=netuid, block=block
            ),
        )

    def is_hotkey_registered_any(
        self,
        hotkey_ss58: str,
        block: Optional[int] = None,
    ) -> bool:
        return self.execute_coroutine(
            self.async_subtensor.is_hotkey_registered_any(
                hotkey_ss58=hotkey_ss58,
                block=block,
            ),
        )

    def is_hotkey_registered_on_subnet(
        self, hotkey_ss58: str, netuid: int, block: Optional[int] = None
    ) -> bool:
        return self.get_uid_for_hotkey_on_subnet(hotkey_ss58, netuid, block) is not None

    def last_drand_round(self) -> Optional[int]:
        return self.execute_coroutine(
            self.async_subtensor.last_drand_round(),
        )

    def max_weight_limit(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[float]:
        return self.execute_coroutine(
            self.async_subtensor.max_weight_limit(netuid=netuid, block=block),
        )

    def metagraph(
        self, netuid: int, lite: bool = True, block: Optional[int] = None
    ) -> "Metagraph":  # type: ignore
        return self.execute_coroutine(
            self.async_subtensor.metagraph(netuid=netuid, lite=lite, block=block),
        )

    def min_allowed_weights(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        return self.execute_coroutine(
            self.async_subtensor.min_allowed_weights(netuid=netuid, block=block),
        )

    def neuron_for_uid(
        self, uid: int, netuid: int, block: Optional[int] = None
    ) -> "NeuronInfo":
        return self.execute_coroutine(
            self.async_subtensor.neuron_for_uid(uid=uid, netuid=netuid, block=block),
        )

    def neurons(self, netuid: int, block: Optional[int] = None) -> list["NeuronInfo"]:
        return self.execute_coroutine(
            self.async_subtensor.neurons(netuid=netuid, block=block),
        )

    def neurons_lite(
        self, netuid: int, block: Optional[int] = None
    ) -> list["NeuronInfoLite"]:
        return self.execute_coroutine(
            self.async_subtensor.neurons_lite(netuid=netuid, block=block),
        )

    def query_identity(self, key: str, block: Optional[int] = None) -> Optional[str]:
        return self.execute_coroutine(
            self.async_subtensor.query_identity(key=key, block=block),
        )

    def recycle(self, netuid: int, block: Optional[int] = None) -> Optional["Balance"]:
        return self.execute_coroutine(
            self.async_subtensor.recycle(netuid=netuid, block=block),
        )

    def subnet_exists(self, netuid: int, block: Optional[int] = None) -> bool:
        return self.execute_coroutine(
            self.async_subtensor.subnet_exists(netuid=netuid, block=block),
        )

    def subnetwork_n(self, netuid: int, block: Optional[int] = None) -> Optional[int]:
        return self.execute_coroutine(
            self.async_subtensor.subnetwork_n(netuid=netuid, block=block),
        )

    def tempo(self, netuid: int, block: Optional[int] = None) -> Optional[int]:
        return self.execute_coroutine(
            self.async_subtensor.tempo(netuid=netuid, block=block),
        )

    def tx_rate_limit(self, block: Optional[int] = None) -> Optional[int]:
        return self.execute_coroutine(
            self.async_subtensor.tx_rate_limit(block=block),
        )

    def weights(
        self, netuid: int, block: Optional[int] = None
    ) -> list[tuple[int, list[tuple[int, int]]]]:
        return self.execute_coroutine(
            self.async_subtensor.weights(netuid=netuid, block=block),
        )

    def weights_rate_limit(
        self, netuid: int, block: Optional[int] = None
    ) -> Optional[int]:
        return self.execute_coroutine(
            self.async_subtensor.weights_rate_limit(netuid=netuid, block=block),
        )

    # Extrinsics =======================================================================================================

    def add_stake(
        self,
        wallet: "Wallet",
        hotkey_ss58: Optional[str] = None,
        amount: Optional[Union["Balance", float]] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        return self.execute_coroutine(
            self.async_subtensor.add_stake(
                wallet=wallet,
                hotkey_ss58=hotkey_ss58,
                amount=amount,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            ),
        )

    def add_stake_multiple(
        self,
        wallet: "Wallet",
        hotkey_ss58s: list[str],
        amounts: Optional[list[Union["Balance", float]]] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        return self.execute_coroutine(
            self.async_subtensor.add_stake_multiple(
                wallet=wallet,
                hotkey_ss58s=hotkey_ss58s,
                amounts=amounts,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            ),
        )

    def burned_register(
        self,
        wallet: "Wallet",
        netuid: int,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
    ) -> bool:
        return self.execute_coroutine(
            self.async_subtensor.burned_register(
                wallet=wallet,
                netuid=netuid,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            ),
        )

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
        return self.execute_coroutine(
            self.async_subtensor.commit_weights(
                wallet=wallet,
                netuid=netuid,
                salt=salt,
                uids=uids,
                weights=weights,
                version_key=version_key,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                max_retries=max_retries,
            ),
        )

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
        return self.execute_coroutine(
            self.async_subtensor.register(
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
            ),
        )

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
        return self.execute_coroutine(
            self.async_subtensor.reveal_weights(
                wallet=wallet,
                netuid=netuid,
                uids=uids,
                weights=weights,
                salt=salt,
                version_key=version_key,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                max_retries=max_retries,
            ),
        )

    def root_register(
        self,
        wallet: "Wallet",
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
    ) -> bool:
        return execute_coroutine(
            self.async_subtensor.root_register(
                wallet=wallet,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            ),
        )

    def root_set_weights(
        self,
        wallet: "Wallet",
        netuids: list[int],
        weights: list[float],
        version_key: int = 0,
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = False,
    ) -> bool:
        return self.execute_coroutine(
            self.async_subtensor.root_set_weights(
                wallet=wallet,
                netuids=netuids,
                weights=weights,
                version_key=version_key,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            ),
        )

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
        return self.execute_coroutine(
            self.async_subtensor.set_weights(
                wallet=wallet,
                netuid=netuid,
                uids=uids,
                weights=weights,
                version_key=version_key,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                max_retries=max_retries,
            )
        )

    def serve_axon(
        self,
        netuid: int,
        axon: "Axon",
        wait_for_inclusion: bool = False,
        wait_for_finalization: bool = True,
        certificate: Optional["Certificate"] = None,
    ) -> bool:
        return self.execute_coroutine(
            self.async_subtensor.serve_axon(
                netuid=netuid,
                axon=axon,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                certificate=certificate,
            ),
        )

    def transfer(
        self,
        wallet: "Wallet",
        dest: str,
        amount: Union["Balance", float],
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
        transfer_all: bool = False,
        keep_alive: bool = True,
    ) -> bool:
        return self.execute_coroutine(
            self.async_subtensor.transfer(
                wallet=wallet,
                destination=dest,
                amount=amount,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                transfer_all=transfer_all,
                keep_alive=keep_alive,
            ),
        )

    def unstake(
        self,
        wallet: "Wallet",
        hotkey_ss58: Optional[str] = None,
        amount: Optional[Union["Balance", float]] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        return self.execute_coroutine(
            self.async_subtensor.unstake(
                wallet=wallet,
                hotkey_ss58=hotkey_ss58,
                amount=amount,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            ),
        )

    def unstake_multiple(
        self,
        wallet: "Wallet",
        hotkey_ss58s: list[str],
        amounts: Optional[list[Union["Balance", float]]] = None,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = False,
    ) -> bool:
        return self.execute_coroutine(
            self.async_subtensor.unstake_multiple(
                wallet=wallet,
                hotkey_ss58s=hotkey_ss58s,
                amounts=amounts,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            ),
        )
