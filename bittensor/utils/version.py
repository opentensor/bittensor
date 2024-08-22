from typing import Optional
from pathlib import Path
import time
from packaging.version import Version

import bittensor
import requests

VERSION_CHECK_THRESHOLD = 86400


class VersionCheckError(Exception):
    pass


def _get_version_file_path() -> Path:
    return Path.home() / ".bittensor" / ".last_known_version"


def _get_version_from_file(version_file: Path) -> Optional[str]:
    try:
        mtime = version_file.stat().st_mtime
        bittensor.logging.debug(f"Found version file, last modified: {mtime}")
        diff = time.time() - mtime

        if diff >= VERSION_CHECK_THRESHOLD:
            bittensor.logging.debug("Version file expired")
            return None

        return version_file.read_text()
    except FileNotFoundError:
        bittensor.logging.debug("No bitensor version file found")
        return None
    except OSError:
        bittensor.logging.exception("Failed to read version file")
        return None


def _get_version_from_pypi(timeout: int = 15) -> str:
    bittensor.logging.debug(
        f"Checking latest Bittensor version at: {bittensor.__pipaddress__}"
    )
    try:
        response = requests.get(bittensor.__pipaddress__, timeout=timeout)
        latest_version = response.json()["info"]["version"]
        return latest_version
    except requests.exceptions.RequestException:
        bittensor.logging.exception("Failed to get latest version from pypi")
        raise


def get_and_save_latest_version(timeout: int = 15) -> str:
    version_file = _get_version_file_path()

    if last_known_version := _get_version_from_file(version_file):
        return last_known_version

    latest_version = _get_version_from_pypi(timeout)

    try:
        version_file.write_text(latest_version)
    except OSError:
        bittensor.logging.exception("Failed to save latest version to file")

    return latest_version


def check_version(timeout: int = 15):
    """
    Check if the current version of Bittensor is up to date with the latest version on PyPi.
    Raises a VersionCheckError if the version check fails.
    """

    pass
    # TODO: bring this back before merging to main.
    try:
        latest_version = get_and_save_latest_version(timeout)

        if Version(latest_version) > Version(bittensor.__version__):
            print(
                "\u001b[33mBittensor Version: Current {}/Latest {}\nPlease update to the latest version at your earliest convenience. "
                "Run the following command to upgrade:\n\n\u001b[0mpython -m pip install --upgrade bittensor".format(
                    bittensor.__version__, latest_version
                )
            )
    except Exception as e:
        raise VersionCheckError("Version check failed") from e


def version_checking(timeout: int = 15):
    """
    Deprecated, kept for backwards compatibility. Use check_version() instead.
    """
    pass

    from warnings import warn

    warn(
        "version_checking() is deprecated, please use check_version() instead",
        DeprecationWarning,
    )

    try:
        check_version(timeout)
    except VersionCheckError:
        bittensor.logging.exception("Version check failed")
