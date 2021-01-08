import argparse
from io import StringIO
import traceback as tb
from munch import Munch
from loguru import logger

from bittensor.synapse import Synapse
from bittensor.dendrite import Dendrite
from bittensor.axon import Axon
from bittensor.metagraph import Metagraph
from bittensor.nucleus import Nucleus
from bittensor.subtensor.interface import Keypair
from bittensor.exceptions.handlers import rollbar
from bittensor.crypto import is_encrypted, decrypt_data
from bittensor.utils import Cli
from bittensor.crypto import decrypt_keypair
from cryptography.exceptions import InvalidSignature, InvalidKey
from cryptography.fernet import InvalidToken
from bittensor.crypto.keyfiles import KeyFileError, load_keypair_from_data

import json
import re
import stat
import os

class FailedConnectToChain(Exception):
    pass

class FailedSubscribeToChain(Exception):
    pass

class FailedToEnterNeuron(Exception):
    pass

class FailedToPollChain(Exception):
    pass

class Neuron:
    def __init__(self, config):
        self.config = config
        self.metagraph = Metagraph(self.config)
        self.nucleus = Nucleus(self.config, self.metagraph)
        self.axon = Axon(self.config, self.nucleus, self.metagraph)
        self.dendrite = Dendrite(self.config, self.metagraph)

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--neuron.hotkeyfile', required=False, default='~/.bittensor/wallets/default/hotkeys/default', 
                                help='''The path to your bittensor hot key file,
                                        Hotkeys should not hold tokens and are only used
                                        for suscribing and setting weights from running code.
                                        Hotkeys are linked to coldkeys through the metagraph''')
        parser.add_argument('--neuron.coldkeyfile', required=False, default='~/.bittensor/wallets/default/coldkeypub.txt', 
                                help='''The path to your bittensor cold publickey text file.
                                        Coldkeys can hold tokens and should be encrypted on your device.
                                        The coldkey must be used to stake and unstake funds from a running node.
                                        On subscribe this coldkey account is linked to the associated hotkey on the subtensor chain.
                                        Only this key is capable of making staking and unstaking requests for this neuron.''')
        Axon.add_args(parser)
        Dendrite.add_args(parser)
        Metagraph.add_args(parser)
        Nucleus.add_args(parser)
        return parser

    @staticmethod   
    def check_config(config: Munch):
        Dendrite.check_config(config)
        Metagraph.check_config(config)
        Axon.check_config(config)
        Nucleus.check_config(config)
        Neuron.__check_hot_key_path(config.neuron.hotkeyfile)
        Neuron.__check_cold_key_path(config.neuron.coldkeyfile)
        try:
            Neuron.load_hotkeypair(config)
            Neuron.load_cold_key(config)
        except (KeyError):
            logger.error("Invalid password")
            quit()
        except KeyFileError:
            logger.error("Keyfile corrupt")
            quit()

    @staticmethod
    def __check_hot_key_path(path):
        path = os.path.expanduser(path)

        if not os.path.isfile(path):
            logger.error("--neuron.hotkeyfile {} is not a file", path)
            logger.error("You can create keys with: bittensor-cli new_wallet")
            raise KeyFileError

        if not os.access(path, os.R_OK):
            logger.error("--neuron.hotkeyfile {} is not readable", path)
            logger.error("Ensure you have proper privileges to read the file {}", path)
            raise KeyFileError

        if Neuron.__is_world_readable(path):
            logger.error("--neuron.hotkeyfile {} is world readable.", path)
            logger.error("Ensure you have proper privileges to read the file {}", path)
            raise KeyFileError

    @staticmethod
    def __is_world_readable(path):
        st = os.stat(path)
        return st.st_mode & stat.S_IROTH

    @staticmethod
    def __check_cold_key_path(path):
        path = os.path.expanduser(path)

        if not os.path.isfile(path):
            logger.error("--neuron.coldkeyfile {} does not exist", path)
            raise KeyFileError

        if not os.path.isfile(path):
            logger.error("--neuron.coldkeyfile {} is not a file", path)
            raise KeyFileError

        if not os.access(path, os.R_OK):
            logger.error("--neuron.coldkeyfile {} is not readable", path)
            raise KeyFileError

        with open(path, "r") as file:
            key = file.readline().strip()
            if not re.match("^0x[a-z0-9]{64}$", key):
                logger.error("Cold key file corrupt")
                raise KeyFileError

    @staticmethod
    def __create_keypair() -> Keypair:
        return Keypair.create_from_mnemonic(Keypair.generate_mnemonic())

    @staticmethod
    def __save_keypair(keypair : Keypair, path : str):
        path = os.path.expanduser(path)
        with open(path, 'w') as file:
            json.dump(keypair.toDict(), file)
            file.close()

        os.chmod(path, stat.S_IWUSR | stat.S_IRUSR)

    @staticmethod
    def __has_keypair(path):
        path = os.path.expanduser(path)

        return os.path.exists(path)

    @staticmethod
    def load_cold_key(config):
        path = config.neuron.coldkeyfile
        path = os.path.expanduser(path)
        with open(path, "r") as file:
            config.neuron.coldkey = file.readline().strip()
        logger.info("Loaded coldkey: {}", config.neuron.coldkey)

    @staticmethod
    def load_hotkeypair(config):
        keyfile = os.path.expanduser(config.neuron.hotkeyfile)
        with open(keyfile, 'rb') as file:
            data = file.read()
            if is_encrypted(data):
                password = Cli.ask_password()
                data = decrypt_data(password, data)
            hotkey = load_keypair_from_data(data)
            config.neuron.keypair = hotkey
            logger.info("Loaded hotkey: {}", config.neuron.keypair.public_key)

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
        logger.trace('Neuron enter')
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """ Defines the exit protocol from asyncio task.

        Args:
            exc_type (Type): The type of the exception.
            exc_value (RuntimeError): The value of the exception, typically RuntimeError. 
            exc_traceback (traceback): The traceback that can be printed for this exception, detailing where error actually happend.

        Returns:
            Neuron: present instance of Neuron.
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
        try:
            self.axon.start()
            logger.info('Started Axon server')
        except Exception as e:
            logger.error('Neuron: Failed to start axon server with error: {}', e)
            raise FailedToEnterNeuron

        try:
            code, message = self.metagraph.connect(timeout=12)
            if code != Metagraph.ConnectSuccess:
                logger.error('Neuron: Timeout while subscribing to the chain endpoint with message {}', message)
                logger.error('Check that your internet connection is working and the chain endpoint {} is available', self.config.metagraph.chain_endpoint)
                failed_connect_msg_help = ''' The subtensor chain endpoint should likely be one of the following choices:
                                        -- localhost:9944 -- (your locally running node)
                                        -- feynman.akira.bittensor.com:9944 (testnet)
                                        -- feynman.kusanagi.bittensor.com:9944 (mainnet)
                                    To connect to your local node you will need to run the subtensor locally: (See: docs/running_a_validator.md)
                                '''
                logger.error(failed_connect_msg_help)
                raise FailedConnectToChain
        except Exception as e:
            logger.error('Neuron: Error while connecting to the chain endpoint: {}', e)
            raise FailedToEnterNeuron

        try:
            code, message = self.metagraph.subscribe(timeout=12)
            if code != Metagraph.SubscribeSuccess:
                logger.error('Neuron: Error while subscribing to the chain endpoint with message: {}', message)
                raise FailedToEnterNeuron

        except Exception as e:
            logger.error('Neuron: Error while subscribing to the chain endpoint: {}', e)
            raise FailedToEnterNeuron

        try:
            self.metagraph.sync()
            logger.info(self.metagraph)
        except Exception as e:
            logger.error('Neuron: Error while syncing chain state with error {}', e)
            raise FailedToEnterNeuron

    def stop(self):

        logger.info('Shutting down the Axon server ...')
        try:
            self.axon.stop()
            logger.info('Axon server stopped')
        except Exception as e:
            logger.error('Neuron: Error while stopping axon server: {} ', e)

