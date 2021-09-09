""" Function for loading secretSeed 
"""
from substrateinterface import Keypair
import json
from loguru import logger

class KeyFileError(Exception):
    """ Overwrite the built-in KeyError
    """

def load_keypair_from_data(data) -> Keypair:
    try:
        data = json.loads(data)
        if "secretSeed" not in data:
            raise KeyFileError("Keyfile corrupt")

        return Keypair.create_from_seed(data['secretSeed'])
    except BaseException as e:
        logger.debug(e)
        raise KeyFileError("Keyfile corrupt") from e
