from typing import Optional
from pathlib import Path
import time

import bittensor
import requests

VERSION_CHECK_THRESHOLD = 86400


def _get_version_file_path() -> Path:
    return Path.home() / ".bittensor" / ".version"


def _get_version_from_file(version_file: Path) -> Optional[str]:
    if not version_file.exists():
        bittensor.logging.debug("No bitensor version file found")
        return None

    mtime = version_file.stat().st_mtime
    bittensor.logging.debug(f"Found version file, last modified: {mtime}")
    diff = time.time() - mtime

    if diff >= VERSION_CHECK_THRESHOLD:
        bittensor.logging.debug("Version file expired")
        return None

    try:
        return version_file.read_text()
    except Exception as e:
        bittensor.logging.error(f"Failed to read version file: {e}")
        return None


def _get_version_from_pypi(timeout: int = 15) -> str:
    try:
        bittensor.logging.debug(
            f"Checking latest Bittensor version at: {bittensor.__pipaddress__}"
        )
        response = requests.get(bittensor.__pipaddress__, timeout=timeout)
        latest_version = response.json()["info"]["version"]

    except requests.exceptions.Timeout:
        bittensor.logging.error("Version check failed due to timeout")
    except requests.exceptions.RequestException as e:
        bittensor.logging.error(f"Version check failed due to request failure: {e}")
    else:
        return latest_version

    raise


def get_latest_version(timeout: int = 15) -> str:
    version_file = _get_version_file_path()
    latest_version = _get_version_from_file(version_file)

    if not latest_version:
        try:
            latest_version = _get_version_from_pypi(timeout)
        except Exception:
            return None
        try:
            version_file.write_text(latest_version)
        except Exception as e:
            bittensor.logging.error(f"Failed to write version file: {e}")

    return latest_version


def version_checking(timeout: int = 15):
    try:
        latest_version = get_latest_version(timeout)
    except Exception:
        return

    version_split = latest_version.split(".")
    latest_version_as_int = (
        (100 * int(version_split[0]))
        + (10 * int(version_split[1]))
        + (1 * int(version_split[2]))
    )

    if latest_version_as_int > bittensor.__version_as_int__:
        print(
            "\u001b[33mBittensor Version: Current {}/Latest {}\nPlease update to the latest version at your earliest convenience. "
            "Run the following command to upgrade:\n\n\u001b[0mpython -m pip install --upgrade bittensor".format(
                bittensor.__version__, latest_version
            )
        )
