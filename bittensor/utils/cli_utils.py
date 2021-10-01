""" Utils for cli, eg. create and validate wallet dir/password/keypair name
"""
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

import sys
import os
import stat
import getpass

from termcolor import colored
from password_strength import PasswordPolicy
from loguru import logger
from substrateinterface import Keypair

from bittensor._crypto.keyfiles import load_keypair_from_data, KeyFileError
from bittensor._crypto import is_encrypted, decrypt_file, CryptoKeyError

class cli_utils():
    """ Utils for cli, eg. create and validate wallet dir/password/keypair name
    """

    @staticmethod
    def load_key(path) -> Keypair:
        """ Get Keypair from the path, where passward has to be provided
        """
        path = os.path.expanduser(path)
        try:
            if is_encrypted(path):
                password = cli_utils.ask_password()
                print("decrypting key... (this may take a few moments)")
                data = decrypt_file(password, path)

            return load_keypair_from_data(data)

        except CryptoKeyError:
            print(colored("Invalid password", 'red'))
            quit()
        except KeyFileError as e:
            print(colored("Keyfile corrupt", 'red'))
            raise e
        
    @staticmethod
    def enable_debug(should_debug):
        """ If not debug, log messages to stderr with the minimum level as INFO
        """
        if not should_debug:
            logger.remove()
            logger.add(sink=sys.stderr, level="INFO")

    @staticmethod
    def create_wallet_dir_if_not_exists(wallet_dir):
        """ Check if wallet_dir exist, if not then make dir for wallet_dir
        """
        wallet_dir = os.path.expanduser(wallet_dir)
        if os.path.exists(wallet_dir):
            if os.path.isdir(wallet_dir):
                return
            else:
                print(colored("{} exists, but is not a directory. Aborting".format(wallet_dir), 'red'))
                quit()
        os.mkdir(wallet_dir)

    @staticmethod
    def create_hotkeys_dir_if_not_exists(hotkeys_dir):
        """ Check if hotkeys_dir exist, if not then make dir for hotkeys_dir
        """
        hotkeys_dir = os.path.expanduser(hotkeys_dir)
        if os.path.exists(hotkeys_dir):
            if os.path.isdir(hotkeys_dir):
                return
            else:
                print(colored("{} exists, but is not a directory. Aborting".format(hotkeys_dir), 'red'))
                quit()
        os.mkdir(hotkeys_dir)

    @staticmethod
    def create_wallets_dir_if_not_exists():
        """ Check if walletS dir exists, if not then create walletS dir to store all the wallets
        """
        wallet_dir = "~/.bittensor/wallets"
        wallet_dir = os.path.expanduser(wallet_dir)
        if os.path.exists(wallet_dir):
            if os.path.isdir(wallet_dir):
                return
            else:
                print(colored("~/.bittensor/wallets exists, but is not a directory. Aborting", 'red'))
                quit()
        os.mkdir(wallet_dir)

    @staticmethod
    def validate_wallet_name( wallet_name:str ) -> str:
        """ Confirm the chosen wallet name with the user 
        """
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
        """ Confirm the chosen hotkey name with the user 
        """
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
        """ Confirm to overwrite the file with the user
        """
        choice = input("File %s already exists. Overwrite ? (y/N) " % file)
        if choice == "y":
            return True
        else:
            return False

    @staticmethod
    def validate_path(path):
        """ Validate if the path is a file and accessible
        """
        path = os.path.expanduser(path)

        if not os.path.isfile(path):
            print(colored("{} is not a file. Aborting".format(path), 'red'))
            quit()

        if not os.access(path, os.R_OK):
            print(colored("{} is not readable. Aborting".format(path), 'red'))
            quit()

    @staticmethod
    def create_dirs():
        """ Create walletS dir to store all the wallets
        """
        path = '~/.bittensor/wallets/'
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def validate_create_path( keyfile, overwrite: bool = False ):
        """ Check if we can overwrite the keyfile with the os and the user
        """
        keyfile = os.path.expanduser(keyfile)
        if os.path.isfile(keyfile):
            if os.access(keyfile, os.W_OK):
                if overwrite:
                    return keyfile

                elif cli_utils.may_overwrite( keyfile ):
                    return keyfile

                else:
                    quit()
            else:
                print(colored("No write access for  %s" % keyfile, 'red'))
                quit()
        else:
            pdir = os.path.dirname(keyfile)
            if os.access(pdir, os.W_OK):
                return keyfile
            else:
                print(colored("No write access for  %s" % keyfile, 'red'))
                quit()

    @staticmethod
    def write_pubkey_to_text_file( keyfile, pubkey_str:str ):
        """ Write  public key to text file
        """
        keyfile = os.path.expanduser(keyfile)
        with open(keyfile + "pub.txt", "w") as pubfile:
            pubfile.write(pubkey_str.strip())

    @staticmethod
    def input_password():
        """ Ask user to input a password
        """
        valid = False
        while not valid:
            password = getpass.getpass("Specify password for key encryption: ")
            valid = cli_utils.validate_password(password)

        return password

    @staticmethod
    def ask_password():
        """ Ask user to input a password
        """
        password = getpass.getpass("Enter password to unlock key: ")
        return password
    @staticmethod
    def validate_password(password):
        """ The policy to validate the strength of password
        """
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
            print(colored('Password not strong enough. Try increasing the length of the password or the password complexity'))
            return False

        password_verification = getpass.getpass("Retype your password: ")
        if password != password_verification:
            print("Passwords do not match")
            return False

        return True
    
    @staticmethod
    def validate_generate_mnemonic(mnemonic):
        """ Create keypair from mnemonic
        """
        if len(mnemonic) not in [12,15,18,21,24]:
            print(colored("Mnemonic has invalid size. This should be 12,15,18,21 or 24 words", 'red'))
            quit()

        try:
            keypair = Keypair.create_from_mnemonic(" ".join(mnemonic))
            return keypair
        except ValueError as e:
            print(colored(str(e), "red"))
            quit()

    @staticmethod
    def gen_new_key(words):
        """ Generate new public/privete keypair 
        1. gen mnemonic 
        2. gen keypair from mnemonic
        """
        mnemonic = Keypair.generate_mnemonic(words)
        keypair = Keypair.create_from_mnemonic(mnemonic)
        return keypair

    @staticmethod
    def display_mnemonic_msg( kepair : Keypair ):
        """ Displaying the mnemonic and warning message to keep mnemonic safe
        """
        mnemonic = kepair.mnemonic
        mnemonic_green = colored(mnemonic, 'green')
        print (colored("\nIMPORTANT: Store this mnemonic in a secure (preferable offline place), as anyone " \
                    "who has possesion of this mnemonic can use it to regenerate the key and access your tokens. \n", "red"))
        print ("The mnemonic to the new key is:\n\n%s\n" % mnemonic_green)
        print ("You can use the mnemonic to recreate the key in case it gets lost. The command to use to regenerate the key using this mnemonic is:")
        print("bittensor-cli regen --mnemonic %s" % mnemonic)
        print('')


    @staticmethod
    def save_keys(path, data):
        """ Write the key(data) to path
        """
        print("Writing key to %s" % path)
        with open(path, "wb") as keyfile:
            keyfile.write(data)
    
    @staticmethod
    def set_file_permissions(path):
        """ Set permission to be read and write by owner
        """
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)

    @staticmethod
    def confirm_no_password():
        """ Warn the owner for not providing password
        """
        print(colored('*** WARNING ***', 'white'))
        print(colored('You have not specified the --password flag.', 'white'))
        print(colored('This means that the generated key will be stored as plaintext in the keyfile', 'white'))
        print(colored('The benefit of this is that you will not be prompted for a password when bittensor starts', 'white'))
        print(colored('The drawback is that an attacker has access to the key if they have access to the account bittensor runs on', 'white'))
        print()
        choice = input("Do you wish to proceed? (Y/n) ")
        if choice in ["n", "N"]:
            return False

        return True
