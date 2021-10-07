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

from bittensor._wallet.wallet_utils import wallet_utils, KeyFileError, CryptoKeyError

class cli_utils():
    """ Utils for cli, eg. create and validate wallet dir/password/keypair name
    """

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
    def ask_password_to_encrypt():
        """ Ask user to input a password
        """
        valid = False
        while not valid:
            password = getpass.getpass("Specify password for key encryption: ")
            valid = cli_utils.validate_password(password)

        return password

    @staticmethod
    def ask_password_to_decrypt():
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
