'''
The MIT License (MIT)
Copyright © 2021 Opentensor.ai

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.
'''
import argparse
import json
import os
import re
import stat

from munch import Munch
from loguru import logger

import bittensor
from bittensor.crypto import is_encrypted, decrypt_data
from bittensor.crypto import decrypt_keypair
from bittensor.crypto.keyfiles import KeyFileError, load_keypair_from_data


class Wallet():
    def __init__(self, config: Munch = None):
        if config == None:
            config = Wallet.build_config()
        self.config = config
        try:
            self.load_hotkeypair()
            self.load_cold_key()
        except (KeyError):
            logger.error("Invalid password")
            quit()
        except KeyFileError:
            logger.error("Keyfile corrupt")
            quit()

    def load_cold_key(self):
        path = self.config.wallet.coldkeyfile
        path = os.path.expanduser(path)
        with open(path, "r") as file:
            self.coldkey = file.readline().strip()
        logger.info("Loaded coldkey: {}", self.coldkey)

    def load_hotkeypair(self):
        keyfile = os.path.expanduser(self.config.wallet.hotkeyfile)
        with open(keyfile, 'rb') as file:
            data = file.read()
            if is_encrypted(data):
                password = bittensor.utils.Cli.ask_password()
                data = decrypt_data(password, data)
            hotkey = load_keypair_from_data(data)
            self.keypair = hotkey
            logger.info("Loaded hotkey: {}", self.keypair.public_key)
        
    @staticmethod   
    def build_config() -> Munch:
        # Parses and returns a config Munch for this object.
        parser = argparse.ArgumentParser(); 
        Wallet.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        Wallet.check_config(config)
        return config

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        try:
            parser.add_argument('--wallet.hotkeyfile', required=False, default='~/.bittensor/wallets/default/hotkeys/default', 
                                    help='''The path to your bittensor hot key file,
                                            Hotkeys should not hold tokens and are only used
                                            for suscribing and setting weights from running code.
                                            Hotkeys are linked to coldkeys through the metagraph''')
            parser.add_argument('--wallet.coldkeyfile', required=False, default='~/.bittensor/wallets/default/coldkeypub.txt', 
                                    help='''The path to your bittensor cold publickey text file.
                                            Coldkeys can hold tokens and should be encrypted on your device.
                                            The coldkey must be used to stake and unstake funds from a running node.
                                            On subscribe this coldkey account is linked to the associated hotkey on the subtensor chain.
                                            Only this key is capable of making staking and unstaking requests for this neuron.''')
        except:
            pass

    @staticmethod   
    def check_config(config: Munch):
        Wallet.__check_hot_key_path(config.wallet.hotkeyfile)
        Wallet.__check_cold_key_path(config.wallet.coldkeyfile)

    @staticmethod
    def __check_hot_key_path(path):
        path = os.path.expanduser(path)

        if not os.path.isfile(path):
            logger.error("--wallet.hotkeyfile {} is not a file", path)
            logger.error("You can create keys with: bittensor-cli new_wallet")
            raise KeyFileError

        if not os.access(path, os.R_OK):
            logger.error("--wallet.hotkeyfile {} is not readable", path)
            logger.error("Ensure you have proper privileges to read the file {}", path)
            raise KeyFileError

        if Wallet.__is_world_readable(path):
            logger.error("--wallet.hotkeyfile {} is world readable.", path)
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
            logger.error("--wallet.coldkeyfile {} does not exist", path)
            raise KeyFileError

        if not os.path.isfile(path):
            logger.error("--wallet.coldkeyfile {} is not a file", path)
            raise KeyFileError

        if not os.access(path, os.R_OK):
            logger.error("--wallet.coldkeyfile {} is not readable", path)
            raise KeyFileError

        with open(path, "r") as file:
            key = file.readline().strip()
            if not re.match("^0x[a-z0-9]{64}$", key):
                logger.error("Cold key file corrupt")
                raise KeyFileError

    @staticmethod
    def __create_keypair() -> bittensor.subtensor.interface.Keypair:
        return bittensor.subtensor.interface.Keypair.create_from_mnemonic(bittensor.subtensor.interface.Keypair.generate_mnemonic())

    @staticmethod
    def __save_keypair(keypair : bittensor.subtensor.interface.Keypair, path : str):
        path = os.path.expanduser(path)
        with open(path, 'w') as file:
            json.dump(keypair.toDict(), file)
            file.close()

        os.chmod(path, stat.S_IWUSR | stat.S_IRUSR)

    @staticmethod
    def __has_keypair(path):
        path = os.path.expanduser(path)
        return os.path.exists(path)