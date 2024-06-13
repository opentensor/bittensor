import time
from typing import Optional

import aiohttp
from aiohttp import ClientError
from aiopath import AsyncPath
from packaging.version import Version

import bittensor

# 24 hours
VERSION_CHECK_THRESHOLD = 86400


class VersionCheckError(Exception):
    pass


async def _get_version_file_path() -> AsyncPath:
    home_path = await AsyncPath.home()
    return home_path / ".bittensor" / ".last_known_version"


async def _get_version_from_file(version_file: AsyncPath) -> Optional[str]:
    try:
        mtime = await version_file.stat()
        mtime = mtime.st_mtime
        bittensor.logging.debug(f"Found version file, last modified: {mtime}")
        diff = time.time() - mtime

        if diff >= VERSION_CHECK_THRESHOLD:
            bittensor.logging.debug("Version file expired")
            return None

        return await version_file.read_text()
    except FileNotFoundError:
        bittensor.logging.debug("No bitensor version file found")
        return None
    except OSError:
        bittensor.logging.exception("Failed to read version file")
        return None


async def _get_version_from_pypi(timeout: int = 15) -> str:
    bittensor.logging.debug(
        f"Checking latest Bittensor version at: {bittensor.__pipaddress__}"
    )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                bittensor.__pipaddress__, timeout=timeout
            ) as response:
                response_json = await response.json()
                return response_json["info"]["version"]
    except ClientError as e:
        bittensor.logging.exception(f"Failed to get latest version from pypi: {e}")
        raise


async def get_and_save_latest_version(timeout: int = 15) -> str:
    version_file = await _get_version_file_path()

    if last_known_version := await _get_version_from_file(version_file):
        return last_known_version

    latest_version = await _get_version_from_pypi(timeout)

    try:
        await version_file.write_text(latest_version)
    except OSError:
        bittensor.logging.exception("Failed to save latest version to file")

    return latest_version


async def check_version(timeout: int = 15):
    """
    Check if the current version of Bittensor is up to date with the latest version on PyPi.
    Raises a VersionCheckError if the version check fails.
    """

    try:
        latest_version = await get_and_save_latest_version(timeout)

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

    from warnings import warn

    warn(
        "version_checking() is deprecated, please use check_version() instead",
        DeprecationWarning,
    )

    try:
        # To:do - Decide on event loop and make changes here
        check_version(timeout)
    except VersionCheckError:
        bittensor.logging.exception("Version check failed")
