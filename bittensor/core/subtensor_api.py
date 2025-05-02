from typing import Optional, Union, TYPE_CHECKING
from bittensor.core.subtensor import Subtensor as _Subtensor
from bittensor.core.async_subtensor import AsyncSubtensor as _AsyncSubtensor

if TYPE_CHECKING:
    from bittensor.core.config import Config


class _Extrinsics:
    def __init__(self, subtensor: Union["_Subtensor", "_AsyncSubtensor"]):
        self.add_stake = subtensor.add_stake
        self.add_stake_multiple = subtensor.add_stake_multiple
        self.burned_register = subtensor.burned_register
        self.commit_weights = subtensor.commit_weights
        self.move_stake = subtensor.move_stake
        self.register = subtensor.register
        self.register_subnet = subtensor.register_subnet
        self.reveal_weights = subtensor.reveal_weights
        self.root_register = subtensor.root_register
        self.root_set_weights = subtensor.root_set_weights
        self.set_subnet_identity = subtensor.set_subnet_identity
        self.set_weights = subtensor.set_weights
        self.serve_axon = subtensor.serve_axon
        self.start_call = subtensor.start_call
        self.swap_stake = subtensor.swap_stake
        self.transfer = subtensor.transfer
        self.transfer_stake = subtensor.transfer_stake
        self.unstake = subtensor.unstake
        self.unstake_multiple = subtensor.unstake_multiple


class _Queries:
    def __init__(self, subtensor: Union["_Subtensor", "_AsyncSubtensor"]):
        self.query_constant = subtensor.query_constant
        self.query_map = subtensor.query_map
        self.query_map_subtensor = subtensor.query_map_subtensor
        self.query_module = subtensor.query_module
        self.query_runtime_api = subtensor.query_runtime_api
        self.query_subtensor = subtensor.query_subtensor


class _Subnets:
    def __init__(self, subtensor: Union["_Subtensor", "_AsyncSubtensor"]):
        self.all_subnets = subtensor.all_subnets
        self.get_all_subnets_info = subtensor.get_all_subnets_info
        self.get_neuron_for_pubkey_and_subnet = subtensor.get_neuron_for_pubkey_and_subnet
        self.get_subnet_burn_cost = subtensor.get_subnet_burn_cost
        self.get_subnet_hyperparameters = subtensor.get_subnet_hyperparameters
        self.get_subnet_owner_hotkey = subtensor.get_subnet_owner_hotkey
        self.get_subnet_reveal_period_epochs = subtensor.get_subnet_reveal_period_epochs
        self.get_subnet_validator_permits = subtensor.get_subnet_validator_permits
        self.get_subnets = subtensor.get_subnets
        self.get_total_subnets = subtensor.get_total_subnets
        self.get_uid_for_hotkey_on_subnet = subtensor.get_uid_for_hotkey_on_subnet
        self.is_hotkey_registered_on_subnet = subtensor.is_hotkey_registered_on_subnet
        self.register_subnet = subtensor.register_subnet
        self.set_subnet_identity = subtensor.set_subnet_identity
        self.subnet = subtensor.subnet
        self.subnet_exists = subtensor.subnet_exists
        self.subnetwork_n = subtensor.subnetwork_n


class SubtensorApi:
    def __init__(
            self,
            network: Optional[str] = None,
            config: Optional["Config"] = None,
            _mock: bool = False,
            log_verbose: bool = False,
            async_subtensor: bool = False,
    ):
        if async_subtensor:
            self._subtensor = _AsyncSubtensor(network=network, config=config, _mock=_mock, log_verbose=log_verbose)
        else:
            self._subtensor = _Subtensor(network=network, config=config, _mock=_mock, log_verbose=log_verbose)

        self.network = network
        self._mock = _mock
        self.log_verbose = log_verbose
        self.is_async = async_subtensor

        self.config = None
        self.start_call = None
        self.chain_endpoint = None
        self.substrate = None
        self.close = None

        self.add_important_fields()

    def add_important_fields(self):
        """Adds important fields from the subtensor instance to this API instance."""
        self.substrate = self._subtensor.substrate
        self.chain_endpoint = self._subtensor.chain_endpoint
        self.config = self._subtensor.config
        self.start_call = self._subtensor.start_call
        self.close = self._subtensor.close

    def __str__(self):
        return f"<Network: {self.network}, Chain: {self.chain_endpoint}, {'Async version' if self.is_async else 'Sync version'}>"

    def __repr__(self):
        return self.__str__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self):
        return await self._subtensor.__aenter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.substrate.close()

    @property
    def block(self):
        return self._subtensor.block

    @property
    def extrinsics(self):
        return _Extrinsics(self._subtensor)

    @property
    def queries(self):
        return _Queries(self._subtensor)

    @property
    def subnets(self):
        return _Subnets(self._subtensor)
