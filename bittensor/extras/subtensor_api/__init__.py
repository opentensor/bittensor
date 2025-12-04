from typing import Optional, Union, TYPE_CHECKING

from bittensor.core.async_subtensor import AsyncSubtensor as _AsyncSubtensor
from bittensor.core.subtensor import Subtensor as _Subtensor
from .chain import Chain as _Chain
from .commitments import Commitments as _Commitments
from .crowdloans import Crowdloans as _Crowdloans
from .delegates import Delegates as _Delegates
from .extrinsics import Extrinsics as _Extrinsics
from .metagraphs import Metagraphs as _Metagraphs
from .mev_shield import MevShield as _MevShield
from .neurons import Neurons as _Neurons
from .proxy import Proxy as _Proxy
from .queries import Queries as _Queries
from .staking import Staking as _Staking
from .subnets import Subnets as _Subnets
from .utils import add_legacy_methods as _add_classic_fields
from .wallets import Wallets as _Wallets

if TYPE_CHECKING:
    from bittensor.core.config import Config


class SubtensorApi:
    """Subtensor API class.

    Parameters:
        network: The network to connect to.
        config: Bittensor configuration object.
        legacy_methods: If `True`, all methods from the Subtensor class will be added to the root level of this class.
        fallback_endpoints: List of fallback endpoints to use if default or provided network is not available.
        retry_forever: Whether to retry forever on connection errors.
        log_verbose: Enables or disables verbose logging.
        mock: Whether this is a mock instance. Mainly just for use in testing.
        archive_endpoints: Similar to fallback_endpoints, but specifically only archive nodes. Will be used in cases
            where you are requesting a block that is too old for your current (presumably lite) node.
        websocket_shutdown_timer: Amount of time, in seconds, to wait after the last response from the chain to close
            the connection. Only applicable to AsyncSubtensor. If `None` is passed to this, the automatic shutdown
            process is disabled.

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

        # using `legacy_methods`
        import bittensor as bt

        subtensor = bt.SubtensorApi(legacy_methods=True)
        print(subtensor.bonds(0))

        # using `fallback_endpoints` or `retry_forever`
        import bittensor as bt

        subtensor = bt.SubtensorApi(
            network="finney",
            fallback_endpoints=["wss://localhost:9945", "wss://some-other-endpoint:9945"],
            retry_forever=True,
        )
        print(subtensor.block)
    """

    def __init__(
        self,
        network: Optional[str] = None,
        config: Optional["Config"] = None,
        async_subtensor: bool = False,
        legacy_methods: bool = False,
        fallback_endpoints: Optional[list[str]] = None,
        retry_forever: bool = False,
        log_verbose: bool = False,
        mock: bool = False,
        archive_endpoints: Optional[list[str]] = None,
        websocket_shutdown_timer: Optional[float] = 5.0,
    ):
        self.network = network
        self._fallback_endpoints = fallback_endpoints
        self._archive_endpoints = archive_endpoints
        self._retry_forever = retry_forever
        self._ws_shutdown_timer = websocket_shutdown_timer
        self._mock = mock
        self.log_verbose = log_verbose
        self.is_async = async_subtensor
        self._config = config

        # assigned only for async instance
        self.initialize = None
        self.inner_subtensor = self._get_subtensor()

        # fix naming collision
        self._neurons = _Neurons(self.inner_subtensor)

        # define empty fields
        self.substrate = self.inner_subtensor.substrate
        self.chain_endpoint = self.inner_subtensor.chain_endpoint
        self.close = self.inner_subtensor.close
        self.config = self.inner_subtensor.config
        self.setup_config = self.inner_subtensor.setup_config
        self.help = self.inner_subtensor.help

        self.determine_block_hash = self.inner_subtensor.determine_block_hash
        self.compose_call = self.inner_subtensor.compose_call
        self.sign_and_send_extrinsic = self.inner_subtensor.sign_and_send_extrinsic
        self.start_call = self.inner_subtensor.start_call
        self.wait_for_block = self.inner_subtensor.wait_for_block

        # adds all Subtensor methods into main level os SubtensorApi class
        if legacy_methods:
            _add_classic_fields(self)

    def _get_subtensor(self) -> Union["_Subtensor", "_AsyncSubtensor"]:
        """Returns the subtensor instance based on the provided config and subtensor type flag."""
        if self.is_async:
            _subtensor = _AsyncSubtensor(
                network=self.network,
                config=self._config,
                log_verbose=self.log_verbose,
                fallback_endpoints=self._fallback_endpoints,
                retry_forever=self._retry_forever,
                mock=self._mock,
                archive_endpoints=self._archive_endpoints,
                websocket_shutdown_timer=self._ws_shutdown_timer,
            )
            self.initialize = _subtensor.initialize
            return _subtensor
        else:
            return _Subtensor(
                network=self.network,
                config=self._config,
                log_verbose=self.log_verbose,
                fallback_endpoints=self._fallback_endpoints,
                retry_forever=self._retry_forever,
                mock=self._mock,
                archive_endpoints=self._archive_endpoints,
            )

    def _determine_chain_endpoint(self) -> str:
        """Determines the connection and mock flag."""
        if self._mock:
            return "Mock"
        return self.substrate.url

    def __str__(self):
        return (
            f"<Network: {self.network}, "
            f"Chain: {self._determine_chain_endpoint()}, "
            f"{'Async version' if self.is_async else 'Sync version'}>"
        )

    def __repr__(self):
        return self.__str__()

    def __enter__(self):
        if self.is_async:
            raise NotImplementedError(
                "Async version of SubtensorApi cannot be used with sync context manager."
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_async:
            raise NotImplementedError(
                "Async version of SubtensorApi cannot be used with sync context manager."
            )
        self.close()

    async def __aenter__(self):
        if not self.is_async:
            raise NotImplementedError(
                "Sync version of SubtensorApi cannot be used with async context manager."
            )
        await self.inner_subtensor.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not self.is_async:
            raise NotImplementedError(
                "Sync version of SubtensorApi cannot be used with async context manager."
            )
        await self.substrate.close()

    @classmethod
    def add_args(cls, parser):
        _Subtensor.add_args(parser)

    @property
    def block(self):
        """Returns current chain block number."""
        return self.inner_subtensor.block

    @property
    def chain(self):
        """Property of interaction with chain methods."""
        return _Chain(self.inner_subtensor)

    @property
    def commitments(self):
        """Property to access commitments methods."""
        return _Commitments(self.inner_subtensor)

    @property
    def crowdloans(self):
        """Property to access crowdloans methods."""
        return _Crowdloans(self.inner_subtensor)

    @property
    def delegates(self):
        """Property to access delegates methods."""
        return _Delegates(self.inner_subtensor)

    @property
    def extrinsics(self):
        """Property to access extrinsics methods."""
        return _Extrinsics(self.inner_subtensor)

    @property
    def metagraphs(self):
        """Property to access metagraphs methods."""
        return _Metagraphs(self.inner_subtensor)

    @property
    def mev_shield(self):
        """Property to access MEV Shield methods."""
        return _MevShield(self.inner_subtensor)

    @property
    def neurons(self):
        """Property to access neurons methods."""
        return self._neurons

    @neurons.setter
    def neurons(self, value):
        """Setter for neurons property."""
        self._neurons = value

    @property
    def proxies(self):
        """Property to access subtensor proxy methods."""
        return _Proxy(self.inner_subtensor)

    @property
    def queries(self):
        """Property to access subtensor queries methods."""
        return _Queries(self.inner_subtensor)

    @property
    def staking(self):
        """Property to access staking methods."""
        return _Staking(self.inner_subtensor)

    @property
    def subnets(self):
        """Property of interaction with subnets methods."""
        return _Subnets(self.inner_subtensor)

    @property
    def wallets(self):
        """Property of interaction methods with cold/hotkeys, and balances, etc."""
        return _Wallets(self.inner_subtensor)
