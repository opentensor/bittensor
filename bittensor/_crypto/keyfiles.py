""" Function for loading secretSeed 
"""
import json
from substrateinterface import Keypair
from loguru import logger

class KeyFileError(Exception):
    """ Overwrite the built-in CryptoKeyError
    """

def load_keypair_from_data(data) -> Keypair:
    """ Get keypair from data seed
    """
    try:
        data = json.loads(data)
        if "secretSeed" not in data:
            raise KeyFileError("Keyfile corrupt")

        return Keypair.create_from_seed(data['secretSeed'])
    except BaseException as e:
        logger.debug(e)
        raise KeyFileError("Keyfile corrupt") from e
