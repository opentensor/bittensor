import argparse
from io import StringIO
import traceback as tb
from bittensor.utils.replicate_utils import ReplicateUtility
from munch import Munch

from bittensor.synapse import Synapse
from bittensor.dendrite import Dendrite
from bittensor.axon import Axon
from bittensor.metagraph import Metagraph
from bittensor.nucleus import Nucleus
from bittensor.utils.asyncio import Asyncio
from bittensor.subtensor.interface import Keypair
from bittensor.exceptions.handlers import rollbar
from loguru import logger
from bittensor.crypto import is_encrypted
from bittensor.utils import Cli
from bittensor.crypto import decrypt_keypair
from cryptography.exceptions import InvalidSignature, InvalidKey
from cryptography.fernet import InvalidToken
import json


import os

class FailedConnectToChain(Exception):
    pass

class FailedSubscribeToChain(Exception):
    pass

class FailedToEnterSession(Exception):
    pass

class FailedToPollChain(Exception):
    pass

class KeyFileError(Exception):
    pass

class KeyError(Exception):
    pass

class Session:
    def __init__(self, config):
        self.config = config
        self.metagraph = Metagraph(self.config)
        self.nucleus = Nucleus(self.config)
        self.axon = Axon(self.config, self.nucleus)
        self.dendrite = Dendrite(self.config)

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--session.keyfile', required=False, default='~/.bittensor/key', help="Path to the bittensor key file")
        return parser

    @staticmethod   
    def check_config(config: Munch) -> Munch:
        Session.__check_key_path(config.session.keyfile)

    @staticmethod
    def __check_key_path(path):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            logger.error("--session.keyfile {} does not exist", path)
            raise KeyFileError

        if not os.path.isfile(path):
            logger.error("--session.keyfile {} is not a file", path)
            raise KeyFileError

        if not os.access(path, os.R_OK):
            logger.error("--session.keyfile {} is not readable", path)
            raise KeyFileError

    @staticmethod
    def load_keypair(config):
        logger.info("Loading keypair")
        keyfile = os.path.expanduser(config.session.keyfile)
        with open(keyfile, 'rb') as file:
            data = file.read()

            if is_encrypted(data):
                data = Session.decrypt_data(data)

            keypair = Session.load_keypair_from_data(data)
            config.session.coldkey = keypair.public_key

        # ---- Create a new hotkey
        hotkey_mnemonic = Keypair.generate_mnemonic()
        logger.info('hotkey: {}', hotkey_mnemonic)
        hotkey = Keypair.create_from_mnemonic(hotkey_mnemonic)
        config.session.keypair = hotkey

    @staticmethod
    def load_keypair_from_data(data) -> Keypair:
        try:
            data = json.loads(data)
            if "secretSeed" not in data:
                raise KeyFileError("Keyfile corrupt")

            return Keypair.create_from_seed(data['secretSeed'])
        except BaseException as e:
            logger.debug(e)
            logger.error("Invalid keypair")
            raise KeyError

    @staticmethod
    def decrypt_data(data):
        try:
            password = Cli.ask_password()
            return decrypt_keypair(data, password)
        except (InvalidSignature, InvalidKey, InvalidToken):
            logger.error("Invalid password")
            raise KeyError

    def __del__(self):
        self.stop()

    def serve(self, synapse: Synapse):
        r""" Serves a Synapse to the axon server replacing the previous Synapse if exists.

            Args:
                synapse (:obj:`bittensor.Synapse`, `required`): 
                    synapse object to serve on the axon server.
        """
        self.axon.serve(synapse)

    def __enter__(self):
        rollbar.init() # If a rollbar token is present, this will enable error reporting to rollbar

        logger.trace('session enter')
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """ Defines the exit protocol from asyncio task.

        Args:
            exc_type (Type): The type of the exception.
            exc_value (RuntimeError): The value of the exception, typically RuntimeError. 
            exc_traceback (traceback): The traceback that can be printed for this exception, detailing where error actually happend.

        Returns:
            Session: present instance of session.
        """        
        self.stop()
        if exc_value:

            top_stack = StringIO()
            tb.print_stack(file=top_stack)
            top_lines = top_stack.getvalue().strip('\n').split('\n')[:-4]
            top_stack.close()

            full_stack = StringIO()
            full_stack.write('Traceback (most recent call last):\n')
            full_stack.write('\n'.join(top_lines))
            full_stack.write('\n')
            tb.print_tb(exc_traceback, file=full_stack)
            full_stack.write('{}: {}'.format(exc_type.__name__, str(exc_value)))
            sinfo = full_stack.getvalue()
            full_stack.close()
            # Log the combined stack
            logger.error('Exception:{}'.format(sinfo))

            if rollbar.is_enabled():
                rollbar.send_exception()

        return self

    def start(self):
        # Stop background grpc threads for serving the synapse object.
        logger.info('Start axon server...')
        try:
            self.axon.start()
        except Exception as e:
            logger.error('SESSION: Failed to start axon server with error: {}', e)
            raise FailedToEnterSession

        logger.trace('Connect to chain ...')
        try:
            connected = self.metagraph.connect()
            if not connected:
                logger.error('SESSION: Timeout while subscribing to the chain endpoint')
                raise FailedConnectToChain
        except Exception as e:
            logger.error('SESSION: Error while connecting to the chain endpoint: {}', e)
            raise FailedToEnterSession

        logger.info('Subscribe to chain ...')
        try:
            is_subscribed = self.metagraph.subscribe(12)
            if is_subscribed == False:
                logger.error('SESSION: Error while subscribing to the chain endpoint: {}', e)
                raise FailedToEnterSession

        except Exception as e:
            logger.error('SESSION: Error while subscribing to the chain endpoint: {}', e)
            raise FailedToEnterSession

    def stop(self):

        logger.info('Shutting down the Axon server ...')
        try:
            self.axon.stop()
        except Exception as e:
            logger.error('SESSION: Error while stopping axon server: {} ', e)

        # Stop background grpc threads for serving synapse objects.
        logger.info('Unsubscribe from chain ...')
        try:
            self.metagraph.unsubscribe()
        except Exception as e:
            logger.error('SESSION: Error while unsubscribing to the chain endpoint: {}', e)

    def subscribe (self):
       self.metagraph.subscribe()

    def unsubscribe (self):
        self.metagraph.unsubscribe()
