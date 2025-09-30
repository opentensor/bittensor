import time
from pathlib import Path
from typing import Optional

import requests
from packaging.version import Version, InvalidVersion

from bittensor import __name__
from bittensor.core.settings import __version__, PIPADDRESS
from bittensor.utils.btlogging import logging

VERSION_CHECK_THRESHOLD = 86400


class VersionCheckError(Exception):
    """Exception raised for errors in the version check process."""


def _get_version_file_path() -> Path:
    return Path.home() / ".bittensor" / ".last_known_version"


def _get_version_from_file(version_file: Path) -> Optional[str]:
    try:
        mtime = version_file.stat().st_mtime
        logging.debug(f"Found version file, last modified: {mtime}")
        diff = time.time() - mtime

        if diff >= VERSION_CHECK_THRESHOLD:
            logging.debug("Version file expired")
            return None

        return version_file.read_text()
    except FileNotFoundError:
        logging.debug("No bittensor version file found")
        return None
    except OSError:
        logging.exception("Failed to read version file")
        return None


def _get_version_from_pypi(timeout: int = 15) -> str:
    logging.debug(f"Checking latest Bittensor version at: {PIPADDRESS}")
    try:
        response = requests.get(PIPADDRESS, timeout=timeout)
        latest_version = response.json()["info"]["version"]
        return latest_version
    except requests.exceptions.RequestException:
        logging.exception("Failed to get latest version from pypi")
        raise


def get_and_save_latest_version(timeout: int = 15) -> str:
    """
    Retrieves and saves the latest version of Bittensor.

    Parameters:
        timeout: The timeout for the request to PyPI in seconds.

    Returns:
        The latest version of Bittensor.
    """
    version_file = _get_version_file_path()

    if last_known_version := _get_version_from_file(version_file):
        return last_known_version

    latest_version = _get_version_from_pypi(timeout)

    try:
        version_file.write_text(latest_version)
    except OSError:
        logging.exception("Failed to save latest version to file")

    return latest_version


def check_version(timeout: int = 15):
    """
    Check if the current version of Bittensor is up-to-date with the latest version on PyPi.
    Raises a VersionCheckError if the version check fails.

    Parameters:
        timeout: The timeout for the request to PyPI in seconds.
    """

    try:
        latest_version = get_and_save_latest_version(timeout)

        if Version(latest_version) > Version(__version__):
            print(
                f"\u001b[33mBittensor Version: Current {__version__}/Latest {latest_version}\n"
                f"Please update to the latest version at your earliest convenience. "
                "Run the following command to upgrade:\n\n\u001b[0mpython -m pip install --upgrade bittensor"
            )
        pass
    except Exception as e:
        raise VersionCheckError("Version check failed") from e


def check_latest_version_in_pypi():
    """Check for the latest version of the package on PyPI."""
    package_name = __name__
    url = f"https://pypi.org/pypi/{package_name}/json"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        latest_version = response.json()["info"]["version"]
        installed_version = __version__
        try:
            if Version(installed_version) < Version(latest_version):
                print(
                    f"\n🔔 New version is available `{package_name} v.{latest_version}`"
                )
                print("📦 Use command `pip install --upgrade bittensor` to update.")
        except InvalidVersion:
            # stay silent if InvalidVersion
            pass
    except (requests.RequestException, KeyError) as e:
        # stay silent if not internet connection or pypi.org issue
        pass
