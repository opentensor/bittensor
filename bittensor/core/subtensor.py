import asyncio
import argparse
import copy
import functools
from typing import Optional, TYPE_CHECKING

from bittensor.core import settings
from bittensor.core.config import Config
from bittensor.utils import networking
from bittensor.core.async_subtensor import AsyncSubtensor
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor.utils.substrate_interface import AsyncSubstrateInterface


def event_loop_is_running():
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


class SubstrateWrapper:
    def __init__(
        self,
        substrate: "AsyncSubstrateInterface",
        event_loop: asyncio.AbstractEventLoop,
    ):
        self._async_instance = substrate
        self.event_loop = event_loop

    def __del__(self):
        self.event_loop.run_until_complete(self._async_instance.close())

    def __getattr__(self, name):
        attr = getattr(self._async_instance, name)

        if asyncio.iscoroutinefunction(attr):

            def sync_method(*args, **kwargs):
                return self.event_loop.run_until_complete(attr(*args, **kwargs))

            return sync_method
        elif asyncio.iscoroutine(attr):
            # indicates this is an async_property
            return self.event_loop.run_until_complete(attr)
        else:
            return attr


class Subtensor(AsyncSubtensor):
    """
    This is an experimental subtensor class that utilises the underlying AsyncSubtensor
    """

    def __init__(
        self,
        network: Optional[str] = None,
        config: Optional["Config"] = None,
        _mock: bool = False,
        log_verbose: bool = False,
        connection_timeout: int = 600,
    ) -> None:
        if event_loop_is_running():
            raise RuntimeError(
                "You are attempting to invoke the sync Subtensor with an already running event loop."
                " You should be using AsyncSubtensor."
            )

        if config is None:
            config = Subtensor.config()
        self._config = copy.deepcopy(config)

        # Setup config.subtensor.network and config.subtensor.chain_endpoint
        self.chain_endpoint, self.network = Subtensor.setup_config(
            network, self._config
        )

        if (
            self.network == "finney"
            or self.chain_endpoint == settings.FINNEY_ENTRYPOINT
        ) and log_verbose:
            logging.info(
                f"You are connecting to {self.network} network with endpoint {self.chain_endpoint}."
            )
            logging.debug(
                "We strongly encourage running a local subtensor node whenever possible. "
                "This increases decentralization and resilience of the network."
            )
            logging.debug(
                "In a future release, local subtensor will become the default endpoint. "
                "To get ahead of this change, please run a local subtensor node and point to it."
            )

        self.log_verbose = log_verbose
        self._connection_timeout = connection_timeout
        self._subtensor = AsyncSubtensor(network=self.chain_endpoint)
        self._event_loop = asyncio.get_event_loop()
        self._event_loop.run_until_complete(self._subtensor.__aenter__())
        self.substrate = SubstrateWrapper(self._subtensor.substrate, self._event_loop)
        self._init_subtensor()

    def _init_subtensor(self):
        for attr_name in dir(AsyncSubtensor):
            attr = getattr(AsyncSubtensor, attr_name)
            if asyncio.iscoroutinefunction(attr):

                def sync_method(a, *args, **kwargs):
                    return self._event_loop.run_until_complete(a(*args, **kwargs))

                setattr(
                    self,
                    attr_name,
                    functools.partial(sync_method, getattr(self._subtensor, attr_name)),
                )

    @property
    def block(self) -> int:
        return self._event_loop.run_until_complete(self._subtensor.get_current_block())

    @staticmethod
    def determine_chain_endpoint_and_network(
        network: str,
    ) -> tuple[Optional[str], Optional[str]]:
        """Determines the chain endpoint and network from the passed network or chain_endpoint.

        Args:
            network (str): The network flag. The choices are: ``finney`` (main network), ``archive`` (archive network +300 blocks), ``local`` (local running network), ``test`` (test network).

        Returns:
            tuple[Optional[str], Optional[str]]: The network and chain endpoint flag. If passed, overrides the ``network`` argument.
        """

        if network is None:
            return None, None
        if network in settings.NETWORKS:
            return network, settings.NETWORK_MAP[network]
        else:
            if (
                network == settings.FINNEY_ENTRYPOINT
                or "entrypoint-finney.opentensor.ai" in network
            ):
                return "finney", settings.FINNEY_ENTRYPOINT
            elif (
                network == settings.FINNEY_TEST_ENTRYPOINT
                or "test.finney.opentensor.ai" in network
            ):
                return "test", settings.FINNEY_TEST_ENTRYPOINT
            elif (
                network == settings.ARCHIVE_ENTRYPOINT
                or "archive.chain.opentensor.ai" in network
            ):
                return "archive", settings.ARCHIVE_ENTRYPOINT
            elif "127.0.0.1" in network or "localhost" in network:
                return "local", network
            else:
                return "unknown", network

    @staticmethod
    def config() -> "Config":
        """
        Creates and returns a Bittensor configuration object.

        Returns:
            config (bittensor.core.config.Config): A Bittensor configuration object configured with arguments added by the `subtensor.add_args` method.
        """
        parser = argparse.ArgumentParser()
        Subtensor.add_args(parser)
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

        Args:
            network (Optional[str]): The name of the Subtensor network. If None, the network and endpoint will be determined from the `config` object.
            config (bittensor.core.config.Config): The configuration object containing the network and chain endpoint settings.

        Returns:
            tuple: A tuple containing the formatted WebSocket endpoint URL and the evaluated network name.
        """
        if network is not None:
            (
                evaluated_network,
                evaluated_endpoint,
            ) = Subtensor.determine_chain_endpoint_and_network(network)
        else:
            if config.is_set("subtensor.chain_endpoint"):
                (
                    evaluated_network,
                    evaluated_endpoint,
                ) = Subtensor.determine_chain_endpoint_and_network(
                    config.subtensor.chain_endpoint
                )

            elif config.subtensor.get("chain_endpoint"):
                (
                    evaluated_network,
                    evaluated_endpoint,
                ) = Subtensor.determine_chain_endpoint_and_network(
                    config.subtensor.chain_endpoint
                )

            elif config.is_set("subtensor.network"):
                (
                    evaluated_network,
                    evaluated_endpoint,
                ) = Subtensor.determine_chain_endpoint_and_network(
                    config.subtensor.network
                )

            elif config.subtensor.get("network"):
                (
                    evaluated_network,
                    evaluated_endpoint,
                ) = Subtensor.determine_chain_endpoint_and_network(
                    config.subtensor.network
                )

            else:
                (
                    evaluated_network,
                    evaluated_endpoint,
                ) = Subtensor.determine_chain_endpoint_and_network(
                    settings.DEFAULTS.subtensor.network
                )

        return (
            networking.get_formatted_ws_endpoint_url(evaluated_endpoint),
            evaluated_network,
        )

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

        Args:
            parser (argparse.ArgumentParser): The ArgumentParser object to which the Subtensor arguments will be added.
            prefix (Optional[str]): An optional prefix for the argument names. If provided, the prefix is prepended to each argument name.

        Arguments added:
            --subtensor.network: The Subtensor network flag. Possible values are 'finney', 'test', 'archive', and 'local'. Overrides the chain endpoint if set.
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
