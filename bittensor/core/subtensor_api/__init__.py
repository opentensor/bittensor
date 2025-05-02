from typing import Optional, Union, TYPE_CHECKING

from bittensor.core.async_subtensor import AsyncSubtensor as _AsyncSubtensor
from bittensor.core.subtensor import Subtensor as _Subtensor
from .chain import Chain as _Chain
from .commitments import Commitments as _Commitments
from .delegates import Delegates as _Delegates
from .extrinsics import Extrinsics as _Extrinsics
from .metagraphs import Metagraphs as _Metagraphs
from .neurons import Neurons as _Neurons
from .queries import Queries as _Queries
from .stakes import Stakes as _Stakes
from .subnets import Subnets as _Subnets
from .utils import add_classic_fields as _add_classic_fields
from .wallets import Wallets as _Wallets

if TYPE_CHECKING:
    from bittensor.core.config import Config


class SubtensorApi:
    """Subtensor API class.

    Arguments:
        network: The network to connect to. Defaults to `None` -> `finney`.
        config: Bittensor configuration object. Defaults to `None`.
        log_verbose: If `True`, sets the subtensor to log verbosely. Defaults to `False`.
        async_subtensor: If `True`, uses the async subtensor to create the connection. Defaults to `False`.
        subtensor_fields: If `True`, all methods from the Subtensor class will be added to the root level of this class.

    Example:
        # sync version
        import bittensor as bt

        subtensor = bt.SubtensorApi()
        print(subtensor.block)
        print(subtensor.delegates.get_delegate_identities())
        subtensor.chain.tx_rate_limit()

        # async version
        import bittensor as bt

        subtensor = bt.SubtensorApi(async_subtensor=True)
        async with subtensor:
            print(await subtensor.block)
            print(await subtensor.delegates.get_delegate_identities())
            print(await subtensor.chain.tx_rate_limit())

        # using `subtensor_fields`
        import bittensor as bt

        subtensor = bt.SubtensorApi(subtensor_fields=True)
        print(subtensor.bonds(0))
    """

    def __init__(
        self,
        network: Optional[str] = None,
        config: Optional["Config"] = None,
        log_verbose: bool = False,
        async_subtensor: bool = False,
        subtensor_fields: bool = False,
        _mock: bool = False,
    ):
        self.network = network
        self._mock = _mock
        self.log_verbose = log_verbose
        self.is_async = async_subtensor
        self._config = config
        # assigned only for async instance
        self.initialize = None
        self._subtensor = self._get_subtensor()

        # fix naming collision
        self._neurons = _Neurons(self._subtensor)

        # define empty fields
        self.substrate = self._subtensor.substrate
        self.add_args = self._subtensor.add_args
        self.chain_endpoint = self._subtensor.chain_endpoint
        self.close = self._subtensor.close
        self.config = self._subtensor.config
        self.setup_config = self._subtensor.setup_config
        self.help = self._subtensor.help

        self.determine_block_hash = self._subtensor.determine_block_hash
        self.encode_params = self._subtensor.encode_params
        self.sign_and_send_extrinsic = self._subtensor.sign_and_send_extrinsic
        self.start_call = self._subtensor.start_call
        self.wait_for_block = self._subtensor.wait_for_block
        if subtensor_fields:
            _add_classic_fields(self)

    def _get_subtensor(self) -> Union["_Subtensor", "_AsyncSubtensor"]:
        """Returns the subtensor instance based on the provided config and subtensor type flag."""
        if self.is_async:
            self.initialize = self._subtensor.initialize
            return _AsyncSubtensor(
                network=self.network,
                config=self._config,
                _mock=self._mock,
                log_verbose=self.log_verbose,
            )
        else:
            return _Subtensor(
                network=self.network,
                config=self._config,
                _mock=self._mock,
                log_verbose=self.log_verbose,
            )

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
    def chain(self):
        return _Chain(self._subtensor)

    @property
    def commitments(self):
        return _Commitments(self._subtensor)

    @property
    def delegates(self):
        return _Delegates(self._subtensor)

    @property
    def extrinsics(self):
        return _Extrinsics(self._subtensor)

    @property
    def metagraphs(self):
        return _Metagraphs(self._subtensor)

    @property
    def neurons(self):
        return self._neurons

    @neurons.setter
    def neurons(self, value):
        self._neurons = value

    @property
    def queries(self):
        return _Queries(self._subtensor)

    @property
    def stakes(self):
        return _Stakes(self._subtensor)

    @property
    def subnets(self):
        return _Subnets(self._subtensor)

    @property
    def wallets(self):
        return _Wallets(self._subtensor)
