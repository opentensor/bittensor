# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

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

from argparse import ArgumentParser
import json
import sys
import os
import stat
from password_strength import PasswordPolicy
import getpass

from loguru import logger

import bittensor
from bittensor.substrate import Keypair
from bittensor.crypto.keyfiles import load_keypair_from_data, KeyFileError
from termcolor import colored
from bittensor.crypto import encrypt, is_encrypted, decrypt_data, KeyError

class cli_utils():

    @staticmethod
    def ask_password():
        print ('here')
        return getpass.getpass("Enter password to unlock key: ")

    @staticmethod
    def load_key(path) -> Keypair:
        path = os.path.expanduser(path)
        password = getpass.getpass("Enter password to unlock key: ")
        try:
            with open(path, 'rb') as file:
                data = file.read()
                if is_encrypted(data):
                    password = getpass.getpass("Enter password to unlock key: ")
                    print ('here2')
                    bittensor.__logger__.log('USER-INFO', "decrypting key... (this may take a few moments)")
                    print ('here3')
                    data = decrypt_data(password, data)

                return load_keypair_from_data(data)

        except KeyError:
            bittensor.__logger__.log('USER-CRITICAL', "Invalid password")
            quit()
        except KeyFileError as e:
            bittensor.__logger__.log('USER-CRITICAL', "Keyfile corrupt")
            raise e
        
    @staticmethod
    def enable_debug(should_debug):
        if not should_debug:
            logger.remove()
            logger.add(sink=sys.stderr, level="INFO")

    @staticmethod
    def create_wallet_dir_if_not_exists(wallet_dir):
        wallet_dir = os.path.expanduser(wallet_dir)
        if os.path.exists(wallet_dir):
            if os.path.isdir(wallet_dir):
                return
            else:
                bittensor.__logger__.log('USER-CRITICAL', "{} exists, but is not a directory. Aborting".format(wallet_dir))
                quit()
        os.mkdir(wallet_dir)

    @staticmethod
    def create_hotkeys_dir_if_not_exists(hotkeys_dir):
        hotkeys_dir = os.path.expanduser(hotkeys_dir)
        if os.path.exists(hotkeys_dir):
            if os.path.isdir(hotkeys_dir):
                return
            else:
                bittensor.__logger__.log('USER-CRITICAL', "{} exists, but is not a directory. Aborting".format(hotkeys_dir))
                quit()
        os.mkdir(hotkeys_dir)

    @staticmethod
    def create_wallets_dir_if_not_exists():
        wallet_dir = "~/.bittensor/wallets"
        wallet_dir = os.path.expanduser(wallet_dir)
        if os.path.exists(wallet_dir):
            if os.path.isdir(wallet_dir):
                return
            else:
                bittensor.__logger__.log('USER-CRITICAL', "~/.bittensor/wallets exists, but is not a directory. Aborting")
                quit()
        os.mkdir(wallet_dir)

    @staticmethod
    def validate_wallet_name( wallet_name:str ) -> str:
        if wallet_name == None:
            choice = input("Use 'default' as wallet ? (y/N) ")
            if choice == "y":
                return 'default'
            else:
                return input("Wallet name: ")
        else:
            return wallet_name

    @staticmethod
    def validate_hotkey_name( hotkey_name:str ) -> str:
        if hotkey_name == None:
            choice = input("Use 'default' as hotkey name ? (y/N) ")
            if choice == "y":
                return 'default'
            else:
                return input("Hotkey name: ")
        else:
            return hotkey_name
            
    @staticmethod
    def may_overwrite( file:str ):
        choice = input("File %s already exists. Overwrite ? (y/N) " % file)
        if choice == "y":
            return True
        else:
            return False

    @staticmethod
    def validate_path(path):
        path = os.path.expanduser(path)

        if not os.path.isfile(path):
            bittensor.__logger__.log('USER-CRITICAL', "{} is not a file. Aborting".format(path))
            quit()

        if not os.access(path, os.R_OK):
            bittensor.__logger__.log('USER-CRITICAL', "{} is not readable. Aborting".format(path))
            quit()

    @staticmethod
    def create_dirs():
        path = '~/.bittensor/wallets/'
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def validate_create_path( keyfile ):
        keyfile = os.path.expanduser(keyfile)
        if os.path.isfile(keyfile):
            if os.access(keyfile, os.W_OK):
                if cli_utils.may_overwrite( keyfile ):
                    return keyfile
                else:
                    quit()
            else:
                bittensor.__logger__.log('USER-CRITICAL', "No write access for  %s" % keyfile)
                quit()
        else:
            pdir = os.path.dirname(keyfile)
            if os.access(pdir, os.W_OK):
                return keyfile
            else:
                bittensor.__logger__.log('USER-CRITICAL', "No write access for  %s" % keyfile)
                quit()

    @staticmethod
    def write_pubkey_to_text_file( keyfile, pubkey_str:str ):
        keyfile = os.path.expanduser(keyfile)
        with open(keyfile + "pub.txt", "w") as pubfile:
            pubfile.write(pubkey_str.strip())

    @staticmethod
    def input_password():
        valid = False
        while not valid:
            password = getpass.getpass("Specify password for key encryption: ")
            valid = cli_utils.validate_password(password)

        return password

    @staticmethod
    def validate_password(password):
        policy = PasswordPolicy.from_names(
            strength=0.20,
            entropybits=10,
            length=6,
        )
        if not password:
            return False

        tested_pass = policy.password(password)
        result = tested_pass.test()
        if len(result) > 0:
            bittensor.__logger__.log('USER-CRITICAL', "Password not strong enough. Try increasing the length of the password or the password complexity")
            return False

        password_verification = getpass.getpass("Retype your password: ")
        if password != password_verification:
            bittensor.__logger__.log('USER-CRITICAL', "Passwords do not match")
            return False

        return True
    
    @staticmethod
    def validate_generate_mnemonic(mnemonic):
        if len(mnemonic) not in [12,15,18,21,24]:
            bittensor.__logger__.log('USER-CRITICAL', "Mnemonic has invalid size. This should be 12,15,18,21 or 24 words")
            quit()

        try:
            keypair = Keypair.create_from_mnemonic(" ".join(mnemonic))
            return keypair
        except ValueError as e:
            bittensor.__logger__.log('USER-CRITICAL', str(e))
            quit()

    @staticmethod
    def gen_new_key(words):
        mnemonic = Keypair.generate_mnemonic(words)
        keypair = Keypair.create_from_mnemonic(mnemonic)
        return keypair

    @staticmethod
    def display_mnemonic_msg( kepair : Keypair ):
        mnemonic = kepair.mnemonic
        bittensor.__logger__.log('USER-CRITICAL', "\nIMPORTANT: Store this mnemonic in a secure (preferable offline place), as anyone " \
                    "who has possesion of this mnemonic can use it to regenerate the key and access your tokens. \n")
        bittensor.__logger__.log('USER-SUCCESS', "The mnemonic to the new key is:\n\n%s\n" % mnemonic)
        bittensor.__logger__.log('USER-INFO', "You can use the mnemonic to recreate the key in case it gets lost. The command to use to regenerate the key using this mnemonic is:")
        bittensor.__logger__.log('USER-INFO', "bittensor-cli regen --mnemonic %s" % mnemonic)
        bittensor.__logger__.log('USER-INFO', '')


    @staticmethod
    def save_keys(path, data):
        bittensor.__logger__.log('USER-INFO', "Writing key to %s" % path)
        with open(path, "wb") as keyfile:
            keyfile.write(data)
    
    @staticmethod
    def set_file_permissions(path):
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        pass

    @staticmethod
    def confirm_no_password():
        bittensor.__logger__.log('USER-INFO', "*** WARNING ***")
        bittensor.__logger__.log('USER-INFO', "You have not specified the --password flag.")
        bittensor.__logger__.log('USER-INFO', "This means that the generated key will be stored as plaintext in the keyfile")
        bittensor.__logger__.log('USER-INFO', "The benefit of this is that you will not be prompted for a password when bittensor starts")
        bittensor.__logger__.log('USER-INFO', "The drawback is that an attacker has access to the key if they have access to the account bittensor runs on")
        bittensor.__logger__.log('USER-INFO', "")
        choice = input("Do you wish to proceed? (Y/n) ")
        if choice in ["n", "N"]:
            return False

        return True