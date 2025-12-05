from typing import Union
from bittensor.core.subtensor import Subtensor as _Subtensor
from bittensor.core.async_subtensor import AsyncSubtensor as _AsyncSubtensor


class Proxy:
    """Class for managing proxy operations on the Bittensor network.

    This class provides access to all proxy-related operations, including creating and managing both standard and pure
    proxy relationships, handling proxy announcements, and querying proxy data. It works with both synchronous
    `Subtensor` and asynchronous `AsyncSubtensor` instances.

    Proxies enable secure delegation of account permissions by allowing a delegate account to perform certain operations
    on behalf of a real account, with restrictions defined by the proxy type and optional time-lock delays.

    Notes:
        - For comprehensive documentation on proxies, see: <https://docs.learnbittensor.org/keys/proxies>
        - For creating and managing proxies, see: <https://docs.learnbittensor.org/keys/proxies/create-proxy>
        - For pure proxy documentation, see: <https://docs.learnbittensor.org/keys/proxies/pure-proxies>
        - For available proxy types and their permissions, see: <https://docs.learnbittensor.org/keys/proxies#types-of-proxies>

    """

    def __init__(self, subtensor: Union["_Subtensor", "_AsyncSubtensor"]):
        self.add_proxy = subtensor.add_proxy
        self.announce_proxy = subtensor.announce_proxy
        self.create_pure_proxy = subtensor.create_pure_proxy
        self.get_proxies = subtensor.get_proxies
        self.get_proxies_for_real_account = subtensor.get_proxies_for_real_account
        self.get_proxy_announcement = subtensor.get_proxy_announcement
        self.get_proxy_announcements = subtensor.get_proxy_announcements
        self.get_proxy_constants = subtensor.get_proxy_constants
        self.kill_pure_proxy = subtensor.kill_pure_proxy
        self.poke_deposit = subtensor.poke_deposit
        self.proxy_announced = subtensor.proxy_announced
        self.proxy = subtensor.proxy
        self.reject_proxy_announcement = subtensor.reject_proxy_announcement
        self.remove_proxies = subtensor.remove_proxies
        self.remove_proxy = subtensor.remove_proxy
        self.remove_proxy_announcement = subtensor.remove_proxy_announcement
