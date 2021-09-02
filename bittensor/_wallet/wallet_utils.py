
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

import os
import stat
import json
import bittensor
import getpass
import random
import string
from password_strength import PasswordPolicy
from ansible_vault import Vault
from substrateinterface import Keypair
from termcolor import colored

def may_write_to_path( full_path:str, overwrite, force_through_user_input):
    if not os.access( full_path, os.W_OK ):
        return False
    if not os.access( os.path.dirname (full_path), os.W_OK ) :
        return False
    if os.path.isfile(full_path):
        if overwrite or force_through_user_input:
            return True
        choice = input("File %s already exists. Overwrite ? (y/N) " % full_path)
        if choice == "y":
            return True
        else:
            return False

def set_file_permissions(path):
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    pass

def validate_password(password):
    policy = PasswordPolicy.from_names(
        strength=0.20,
        entropybits=10,
        length=6,
    )
    if not password:
        return False

    tested_pass = policy.password( password )
    result = tested_pass.test()
    if len(result) > 0:
        print(colored('Password not strong enough. Try increasing the length of the password or the password complexity'))
        return False

    password_verification = getpass.getpass("Retype your password: ")
    if password != password_verification:
        print("Passwords do not match")
        return False

    return True

def input_password( force_through_user_input ):
    valid = False
    if force_through_user_input:
        generated_password = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        print(colored('***WARNING***: you are encrypting this keypair with the follwoing randomly generate password: {}'.format(generated_password), 'red'))
        return generated_password
    while not valid:
        password = getpass.getpass("Specify password for key encryption: ")
        valid = validate_password(password)
    return password

def write_pubkey_to_text_file( full_path, keypair:Keypair, overwrite ):
    pub_fullpath = os.path.expanduser( full_path ) + "pub.txt",
    if not may_write_to_path( pub_fullpath, overwrite ):
        print(colored("No write access for {}" % pub_fullpath, 'red'))
    with open(pub_fullpath, "w") as pubfile:
        pubfile.write( keypair.public_key.strip() )

def keypair_to_dict( keypair: Keypair ):
    return {
        'accountId': keypair.public_key,
        'publicKey': keypair.public_key,
        'secretPhrase': keypair.mnemonic,
        'secretSeed': "0x" + keypair.seed_hex,
        'ss58Address': keypair.ss58_address
    }

def generate_new_keypair( words: int ):
    mnemonic = Keypair.generate_mnemonic( words )
    keypair = Keypair.create_from_mnemonic( mnemonic )
    return keypair

def display_mnemonic_msg( kepair : Keypair ):
    mnemonic = kepair.mnemonic
    mnemonic_green = colored(mnemonic, 'green')
    print (colored("\nIMPORTANT: Store this mnemonic in a secure (preferable offline place), as anyone " \
                "who has possesion of this mnemonic can use it to regenerate the key and access your tokens. \n", "red"))
    print ("The mnemonic to the new key is:\n\n%s\n" % mnemonic_green)
    print ("You can use the mnemonic to recreate the key in case it gets lost. The command to use to regenerate the key using this mnemonic is:")
    print("bittensor regen --mnemonic %s" % mnemonic)
    print('')

def confirm_no_password( force_through_user_input ):
    print(colored('*** WARNING ***', 'white'))
    print(colored('You have specified not using a password.', 'white'))
    print(colored('This means that the generated key will be stored as plaintext in the keyfile', 'white'))
    print(colored('The benefit of this is that you will not be prompted for a password when bittensor starts', 'white'))
    print(colored('The drawback is that an attacker has access to the key if they have access to the account bittensor runs on', 'white'))
    print()
    if force_through_user_input:
        return True
    choice = input("Do you wish to proceed? (Y/n) ")
    if choice in ["n", "N"]:
        return False

    return True

def create_new_encrypted_keypair ( 
        path: str,
        name: str,
        n_words: int, 
        use_password: bool, 
        overwrite:bool,
        is_coldkey:bool,
        force_through_user_input:bool,
    ) -> 'bittensor.Wallet':  
    r""" Creates a new (possibly encrypted) keypair at the specified path with name.
            Args:
                path (`type`:str)
                    The file path we are saving the keypair under.
                name (`type`:str)
                    The name file we are saving the keypair to.
                n_words (`type`:int):
                    The number of mnemonic words to use.
                use_password (`type`:bool):
                    If True, the keypair will be encrypted and passowrd protected.
                overwrite (`type`:bool):
                    If True, the keypair will overwrite the file already saved 
                    under this path (if it exists.)
                is_coldkey (`type`:bool):
                    If True, the keypair is treated as the coldkey and the .ss58.pub and .ed255.pub are created
                force_through_user_input (`type`:bool):
                    If True, the file function forces all operations by passing 'yes'

    """
    # Check write permissions.
    full_path = os.path.expanduser(os.path.join(path, name))
    if not may_write_to_path( full_path, overwrite ):
        print(colored("No write access for {}" % full_path, 'red'))

    # Create Key.
    keypair = generate_new_keypair( n_words )
    display_mnemonic_msg( keypair )

    # Optionally Encrypt
    keypair_data = json.dumps( keypair_to_dict( keypair ) ).encode()
    if use_password:
        keypair_data = json.dumps( keypair_to_dict( keypair ) ).encode()
        vault = Vault( input_password( force_through_user_input ) )
        vault.dump( keypair_data, open( full_path, 'w') )
    else:
        if is_coldkey:
            confirm_no_password( force_through_user_input )

    # Optionally write pubkey ed255 to file.
    if is_coldkey:
        write_pubkey_to_text_file( full_path, keypair, overwrite )

    # Set permissions.
    set_file_permissions( full_path )
