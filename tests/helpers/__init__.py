import os
from .helpers import (  # noqa: F401
    CLOSE_IN_VALUE,
    __mock_wallet_factory__,
)
from bittensor_wallet.mock.wallet_mock import (  # noqa: F401
    get_mock_coldkey,
    get_mock_hotkey,
    get_mock_keypair,
    get_mock_wallet,
)


def is_running_in_circleci():
    """Checks that tests are running in the app.circleci.com environment."""
    return os.getenv("CIRCLECI") == "true"
