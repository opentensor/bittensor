# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2022 Opentensor Foundation
# Copyright © 2023 Opentensor Technologies Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Module providing common helper functions for working with Subtensor."""

import json
import logging
import os
from typing import Dict, Optional, Union, Any

from substrateinterface.base import SubstrateInterface

_logger = logging.getLogger("subtensor.errors_handler")

_USER_HOME_DIR = os.path.expanduser("~")
_BT_DIR = os.path.join(_USER_HOME_DIR, ".bittensor")
_ERRORS_FILE_PATH = os.path.join(_BT_DIR, "subtensor_errors_map.json")
_ST_BUILD_ID = "subtensor_build_id"

# Create directory if it doesn't exist
os.makedirs(_BT_DIR, exist_ok=True)


# Pallet's typing class `PalletMetadataV14` is defined only at
# https://github.com/polkascan/py-scale-codec/blob/master/scalecodec/type_registry/core.json#L1024
# A class object is created dynamically at runtime.
# Circleci linter complains about string represented classes like 'PalletMetadataV14'.
def _get_errors_from_pallet(pallet) -> Optional[Dict[str, Dict[str, str]]]:
    """Extracts and returns error information from the given pallet metadata.

    Args:
        pallet (PalletMetadataV14): The pallet metadata containing error definitions.

    Returns:
        dict[str, str]: A dictionary of errors indexed by their IDs.

    Raises:
        ValueError: If the pallet does not contain error definitions or the list is empty.
    """
    if not hasattr(pallet, "errors") or not pallet.errors:
        _logger.warning(
            "The pallet does not contain any error definitions or the list is empty."
        )
        return None

    return {
        str(error["index"]): {
            "name": error["name"],
            "description": " ".join(error["docs"]),
        }
        for error in pallet.errors
    }


def _save_errors_to_cache(uniq_version: str, errors: Dict[str, Dict[str, str]]):
    """Saves error details and unique version identifier to a JSON file.

    Args:
        uniq_version (str): Unique version identifier for the Subtensor build.
        errors (dict[str, str]): Error information to be cached.
    """
    data = {_ST_BUILD_ID: uniq_version, "errors": errors}
    try:
        with open(_ERRORS_FILE_PATH, "w") as json_file:
            json.dump(data, json_file, indent=4)
    except IOError as e:
        _logger.warning(f"Error saving to file: {e}")


def _get_errors_from_cache() -> Optional[Dict[str, Dict[str, Dict[str, str]]]]:
    """Retrieves and returns the cached error information from a JSON file, if it exists.

    Returns:
            A dictionary containing error information.
    """
    if not os.path.exists(_ERRORS_FILE_PATH):
        return None

    try:
        with open(_ERRORS_FILE_PATH, "r") as json_file:
            data = json.load(json_file)
    except IOError as e:
        _logger.warning(f"Error reading from file: {e}")
        return None

    return data


def get_subtensor_errors(
    substrate: SubstrateInterface,
) -> Union[Dict[str, Dict[str, str]], Dict[Any, Any]]:
    """Fetches or retrieves cached Subtensor error definitions using metadata.

    Args:
        substrate (SubstrateInterface): Instance of SubstrateInterface to access metadata.

    Returns:
        dict[str, str]: A dictionary containing error information.
    """
    if not substrate.metadata:
        substrate.get_metadata()

    cached_errors_map = _get_errors_from_cache()
    # TODO: Talk to the Nucleus team about a unique identification for each assembly (subtensor). Before that, use
    #  the metadata value for `subtensor_build_id`
    subtensor_build_id = substrate.metadata[0].value

    if not cached_errors_map or subtensor_build_id != cached_errors_map.get(
        _ST_BUILD_ID
    ):
        pallet = substrate.metadata.get_metadata_pallet("SubtensorModule")
        subtensor_errors_map = _get_errors_from_pallet(pallet)

        if not subtensor_errors_map:
            return {}

        _save_errors_to_cache(
            uniq_version=substrate.metadata[0].value, errors=subtensor_errors_map
        )
        _logger.info(f"File {_ERRORS_FILE_PATH} has been updated.")
        return subtensor_errors_map
    else:
        return cached_errors_map.get("errors", {})
