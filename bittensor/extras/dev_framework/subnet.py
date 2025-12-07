from typing import Optional, Union
from collections import namedtuple
from bittensor_wallet import Wallet

from bittensor.core.extrinsics.asyncex.utils import (
    sudo_call_extrinsic as async_sudo_call_extrinsic,
)
from bittensor.core.extrinsics.utils import sudo_call_extrinsic
from bittensor.core.settings import DEFAULT_PERIOD
from bittensor.core.types import ExtrinsicResponse
from bittensor.extras import SubtensorApi
from bittensor.utils.btlogging import logging
from .calls import *  # noqa: F401#
from .utils import (
    is_instance_namedtuple,
    split_command,
    ACTIVATE_SUBNET,
    STEPS,
    REGISTER_NEURON,
    REGISTER_SUBNET,
)

CALL_RECORD = namedtuple("CALL_RECORD", ["idx", "operation", "response"])

# Use this constant to set netuid in set_hyperparameter using class `self._netuid`
NETUID = "SN_NETUID"


class TestSubnet:
    """Class for managing test subnet operations."""

    def __init__(
        self,
        subtensor: SubtensorApi,
        netuid: Optional[int] = None,
        period: Optional[int] = DEFAULT_PERIOD,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ):
        if not isinstance(subtensor, SubtensorApi):
            raise TypeError("subtensor must be an instance of `SubtensorApi`.")
        self.s: SubtensorApi = subtensor
        self.period = period
        self.raise_error = raise_error
        self.wait_for_inclusion = wait_for_inclusion
        self.wait_for_finalization = wait_for_finalization

        self._netuid = netuid
        self._owner: Optional[Wallet] = None
        self._calls: list[CALL_RECORD] = []

    def __str__(self):
        return f"TestSubnet(netuid={self._netuid}, owner={self._owner})"

    @property
    def calls(self) -> list[CALL_RECORD]:
        return self._calls

    @property
    def netuid(self) -> int:
        return self._netuid

    @property
    def owner(self) -> Wallet:
        return self._owner

    def execute_steps(self, steps: list[Union[STEPS, tuple]]):
        """Executes a multiple steps synchronously."""
        for step in steps:
            self.execute_one(step)

    async def async_execute_steps(self, steps: list[Union[STEPS, tuple]]):
        """Executes a multiple steps asynchronously."""
        for step in steps:
            await self.async_execute_one(step)

    def execute_one(self, step: Union[STEPS, tuple]) -> ExtrinsicResponse:
        """Executes one step synchronously."""
        # subnet registration
        response = None
        if isinstance(step, REGISTER_SUBNET):
            assert isinstance(step.wallet, Wallet), (
                "Bittensor Wallet instance must be provided to register subnet."
            )
            response = self._register_subnet(step.wallet)

        # subnet activation
        if isinstance(step, ACTIVATE_SUBNET):
            assert isinstance(step.wallet, Wallet), (
                "subnet owner wallet must be provided to activate subnet."
            )
            owner_wallet = step.wallet
            netuid = step.netuid if isinstance(step.netuid, int) else self._netuid
            response = self._activate_subnet(owner_wallet, netuid)

        # add neuron to subnet
        if isinstance(step, REGISTER_NEURON):
            assert isinstance(step.wallet, Wallet), (
                "neuron wallet must be provided to register in subnet."
            )
            self._check_netuid()
            neuron_wallet = step.wallet
            netuid = step.netuid if isinstance(step.netuid, int) else self._netuid
            response = self._register_neuron(neuron_wallet, netuid)

        if is_instance_namedtuple(step):
            (
                sudo_or_owner_wallet,
                call_module,
                sudo_call,
                call_function,
                call_params,
            ) = split_command(step)
            assert isinstance(sudo_or_owner_wallet, Wallet), (
                "sudo wallet must be provided to the class to set tempo."
            )

            # use `NETUID` for netuid field in namedtuple for using TestSubnet.netuid in setting hyperparameter.
            if hasattr(step, "netuid") and getattr(step, "netuid") == NETUID:
                call_params.update({"netuid": self._netuid})

            if call_function.startswith("sudo_") and sudo_call is None:
                sudo_call = True

            response = self.set_hyperparameter(
                sudo_or_owner_wallet=sudo_or_owner_wallet,
                call_function=call_function,
                call_module=call_module,
                call_params=call_params,
                sudo_call=sudo_call,
            )
        if not response:
            raise NotImplementedError(
                f"Execution for step {step} with type {type(step)}."
            )
        return response

    async def async_execute_one(self, step: Union[STEPS, tuple]) -> ExtrinsicResponse:
        """Executes one step asynchronously."""
        response = None
        # subnet registration
        if isinstance(step, REGISTER_SUBNET):
            assert isinstance(step.wallet, Wallet), (
                "Bittensor Wallet instance must be provided to register subnet."
            )
            response = await self._async_register_subnet(step.wallet)

        # subnet activation
        if isinstance(step, ACTIVATE_SUBNET):
            assert isinstance(step.wallet, Wallet), (
                "subnet owner wallet must be provided to activate subnet."
            )
            owner_wallet = step.wallet
            netuid = step.netuid if isinstance(step.netuid, int) else self._netuid
            response = await self._async_activate_subnet(owner_wallet, netuid)

        # add neuron to subnet
        if isinstance(step, REGISTER_NEURON):
            assert isinstance(step.wallet, Wallet), (
                "neuron wallet must be provided to register in subnet."
            )
            self._check_netuid()
            neuron_wallet = step.wallet
            netuid = step.netuid if isinstance(step.netuid, int) else self._netuid
            response = await self._async_register_neuron(neuron_wallet, netuid)

        if is_instance_namedtuple(step):
            (
                sudo_or_owner_wallet,
                call_module,
                sudo_call,
                call_function,
                call_params,
            ) = split_command(step)
            assert isinstance(sudo_or_owner_wallet, Wallet), (
                "sudo wallet must be provided to the class to set tempo."
            )

            # use `NETUID` for netuid field in namedtuple for using TestSubnet.netuid in setting hyperparameter.
            if hasattr(step, "netuid") and getattr(step, "netuid") == NETUID:
                call_params.update({"netuid": self._netuid})

            if call_function.startswith("sudo_") and sudo_call is None:
                sudo_call = True

            response = await self.async_set_hyperparameter(
                sudo_or_owner_wallet=sudo_or_owner_wallet,
                call_function=call_function,
                call_module=call_module,
                call_params=call_params,
                sudo_call=sudo_call,
            )
        if not response:
            raise NotImplementedError(
                f"Execution for step {step} with type {type(step)}."
            )
        return response

    def _register_subnet(
        self,
        owner_wallet: Wallet,
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """Register subnet on the chain."""
        self._check_register_subnet()
        response = self.s.subnets.register_subnet(
            wallet=owner_wallet,
            period=period or self.period,
            raise_error=raise_error or self.raise_error,
            wait_for_inclusion=wait_for_inclusion or self.wait_for_inclusion,
            wait_for_finalization=wait_for_finalization or self.wait_for_finalization,
        )
        self._check_response(response)
        self._netuid = self.s.subnets.get_total_subnets() - 1
        if response.success:
            self._owner = owner_wallet
            logging.console.info(f"Subnet [blue]{self._netuid}[/blue] was registered.")
        self._add_call_record(REGISTER_SUBNET.__name__, response)
        return response

    async def _async_register_subnet(
        self,
        owner_wallet: Wallet,
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """Register subnet on the chain."""
        self._check_register_subnet()
        response = await self.s.subnets.register_subnet(
            wallet=owner_wallet,
            period=period or self.period,
            raise_error=raise_error or self.raise_error,
            wait_for_inclusion=wait_for_inclusion or self.wait_for_inclusion,
            wait_for_finalization=wait_for_finalization or self.wait_for_finalization,
        )
        self._check_response(response)
        self._netuid = await self.s.subnets.get_total_subnets() - 1
        if response.success:
            self._owner = owner_wallet
            logging.console.info(f"Subnet [blue]{self._netuid}[/blue] was registered.")
        self._add_call_record(REGISTER_SUBNET.__name__, response)
        return response

    def _activate_subnet(
        self,
        owner_wallet: Wallet,
        netuid: Optional[int] = None,
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """Activate subnet."""
        self._check_netuid()
        current_block = self.s.block
        activation_block = self.s.queries.query_constant(
            "SubtensorModule", "DurationOfStartCall"
        ).value
        # added 10 blocks bc 2.5 seconds is not always enough for the chain to update.
        self.s.wait_for_block(current_block + activation_block + 1)
        response = self.s.subnets.start_call(
            wallet=owner_wallet,
            netuid=netuid or self._netuid,
            period=period or self.period,
            raise_error=raise_error or self.raise_error,
            wait_for_inclusion=wait_for_inclusion or self.wait_for_inclusion,
            wait_for_finalization=wait_for_finalization or self.wait_for_finalization,
        )
        if self._check_response(response):
            logging.console.info(f"Subnet [blue]{self._netuid}[/blue] was activated.")
        assert self.s.subnets.is_subnet_active(self._netuid), (
            "Subnet was not activated."
        )
        self._add_call_record(ACTIVATE_SUBNET.__name__, response)
        return response

    async def _async_activate_subnet(
        self,
        owner_wallet: Wallet,
        netuid: Optional[int] = None,
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """Activate subnet."""
        self._check_netuid()
        current_block = await self.s.block
        activation_block = (
            await self.s.queries.query_constant(
                "SubtensorModule", "DurationOfStartCall"
            )
        ).value
        # added 10 blocks bc 2.5 seconds is not always enough for the chain to update.
        await self.s.wait_for_block(current_block + activation_block + 1)

        response = await self.s.subnets.start_call(
            wallet=owner_wallet,
            netuid=netuid or self._netuid,
            period=period or self.period,
            raise_error=raise_error or self.raise_error,
            wait_for_inclusion=wait_for_inclusion or self.wait_for_inclusion,
            wait_for_finalization=wait_for_finalization or self.wait_for_finalization,
        )
        if self._check_response(response):
            logging.console.info(f"Subnet [blue]{self._netuid}[/blue] was activated.")
        assert await self.s.subnets.is_subnet_active(self._netuid), (
            "Subnet was not activated."
        )
        self._add_call_record(ACTIVATE_SUBNET.__name__, response)
        return response

    def _register_neuron(
        self,
        neuron_wallet: Wallet,
        netuid: Optional[int] = None,
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """Add neuron to the subnet."""
        response = self.s.subnets.burned_register(
            wallet=neuron_wallet,
            netuid=netuid or self._netuid,
            period=period or self.period,
            raise_error=raise_error or self.raise_error,
            wait_for_inclusion=wait_for_inclusion or self.wait_for_inclusion,
            wait_for_finalization=wait_for_finalization or self.wait_for_finalization,
        )
        self._check_response(response)
        if response.success:
            logging.console.info(
                f"Neuron [blue]{neuron_wallet.name.capitalize()} <{neuron_wallet.hotkey.ss58_address}>[/blue] was "
                f"registered in subnet [blue]{self._netuid}[/blue]."
            )
        self._add_call_record(REGISTER_NEURON.__name__, response)
        return response

    async def _async_register_neuron(
        self,
        neuron_wallet: Wallet,
        netuid: Optional[int] = None,
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """Add neuron to the subnet."""
        response = await self.s.subnets.burned_register(
            wallet=neuron_wallet,
            netuid=netuid or self._netuid,
            period=period or self.period,
            raise_error=raise_error or self.raise_error,
            wait_for_inclusion=wait_for_inclusion or self.wait_for_inclusion,
            wait_for_finalization=wait_for_finalization or self.wait_for_finalization,
        )
        self._check_response(response)
        if response.success:
            logging.console.info(
                f"Neuron [blue]{neuron_wallet.name.capitalize()} <{neuron_wallet.hotkey.ss58_address}>[/blue] was "
                f"registered in subnet [blue]{self._netuid}[/blue]."
            )
        self._add_call_record(REGISTER_NEURON.__name__, response)
        return response

    def set_hyperparameter(
        self,
        sudo_or_owner_wallet: Wallet,
        call_function: str,
        call_module: str,
        call_params: dict,
        sudo_call: bool = False,
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """Set hyperparameter for the chain or subnet."""
        response = sudo_call_extrinsic(
            subtensor=self.s.inner_subtensor,
            wallet=sudo_or_owner_wallet,
            call_function=call_function,
            call_module=call_module,
            call_params=call_params,
            period=period or self.period,
            raise_error=raise_error or self.raise_error,
            wait_for_inclusion=wait_for_inclusion or self.wait_for_inclusion,
            wait_for_finalization=wait_for_finalization or self.wait_for_finalization,
            root_call=not sudo_call,
        )

        if self._check_response(response):
            logging.console.info(
                f"Hyperparameter [blue]{call_function}[/blue] was set successfully with params [blue]{call_params}[/blue]."
            )
        self._add_call_record("SET_HYPERPARAMETER", response)
        return response

    async def async_set_hyperparameter(
        self,
        sudo_or_owner_wallet: Wallet,
        call_function: str,
        call_module: str,
        call_params: dict,
        sudo_call: bool = False,
        period: Optional[int] = None,
        raise_error: bool = False,
        wait_for_inclusion: bool = True,
        wait_for_finalization: bool = True,
    ) -> ExtrinsicResponse:
        """Set hyperparameter for the chain or subnet."""
        response = await async_sudo_call_extrinsic(
            subtensor=self.s.inner_subtensor,
            wallet=sudo_or_owner_wallet,
            call_function=call_function,
            call_module=call_module,
            call_params=call_params,
            period=period or self.period,
            raise_error=raise_error or self.raise_error,
            wait_for_inclusion=wait_for_inclusion or self.wait_for_inclusion,
            wait_for_finalization=wait_for_finalization or self.wait_for_finalization,
            root_call=not sudo_call,
        )

        if self._check_response(response):
            logging.console.info(
                f"Hyperparameter [blue]{call_function}[/blue] was set successfully with params [blue]{call_params}[/blue]."
            )
        self._add_call_record("SET_HYPERPARAMETER", response)
        return response

    def wait_next_epoch(self, netuid: Optional[int] = None):
        """Sync wait until the next epoch first block is reached."""
        netuid = netuid or self._netuid
        if not netuid:
            self._check_netuid()
        current_block = self.s.block
        next_epoch_block = self.s.subnets.get_next_epoch_start_block(netuid)
        logging.console.info(
            f"Waiting for next epoch first block: [blue]{next_epoch_block}[/blue]. "
            f"Current block: [blue]{current_block}[/blue]."
        )
        self.s.wait_for_block(next_epoch_block + 1)

    async def async_wait_next_epoch(self, netuid: Optional[int] = None):
        """Async wait until the next epoch first block is reached."""
        netuid = netuid or self._netuid
        if not netuid:
            self._check_netuid()
        current_block = await self.s.block
        next_epoch_block = await self.s.subnets.get_next_epoch_start_block(netuid)
        logging.console.info(
            f"Waiting for next epoch first block: [blue]{next_epoch_block}[/blue]. "
            f"Current block: [blue]{current_block}[/blue]."
        )
        await self.s.wait_for_block(next_epoch_block + 1)

    def _check_netuid(self):
        assert self._netuid is not None, (
            "Subnet must be registered before any subnet action is performed."
        )

    def _add_call_record(self, operation: str, response: ExtrinsicResponse):
        """Add extrinsic response to the calls list."""
        self._calls.append(CALL_RECORD(len(self._calls), operation, response))

    def _check_response(self, response: ExtrinsicResponse) -> bool:
        """Check if the call was successful."""
        if response.success:
            return True

        if self.raise_error:
            if response.error:
                raise response.error
            raise RuntimeError(response.message)

        logging.console.warning(response.message)
        return False

    def _check_register_subnet(self):
        """Avoids multiple subnet registrations within the same class instance."""
        if self._netuid:
            raise RuntimeError(
                "This instance already has associated netuid. Cannot register again. "
                "To register a new subnet, create a new instance of TestSubnet class."
            )
