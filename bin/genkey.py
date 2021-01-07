#!/bin/python3

from bittensor.subtensor.interface import Keypair
from termcolor import colored
from bittensor.crypto import encrypt

from argparse import ArgumentParser
import json
import os
import stat
from password_strength import PasswordPolicy
import getpass


def create_config_dir_if_not_exists():
    config_dir = "~/.bittensor"
    config_dir = os.path.expanduser(config_dir)
    if os.path.exists(config_dir):
        if os.path.isdir(config_dir):
            return
        else:
            print(colored("~/.bittensor exists, but is not a directory. Aborting", 'red'))
            quit()
    os.mkdir(config_dir)

def may_overwrite(file):
    choice = input("File %s already exists. Overwrite ? (y/N) " % file)
    if choice == "y":
        return True
    else:
        return False

def validate_path(keyfile):
    keyfile = os.path.expanduser(keyfile)

    if os.path.isfile(keyfile):
        if os.access(keyfile, os.W_OK):
            if may_overwrite(keyfile):
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

def input_password():
    valid = False
    while not valid:
        password = getpass.getpass("Specify password for key encryption: ")
        valid = validate_password(password)

    return password

def validate_password(password):
    policy = PasswordPolicy.from_names(
        strength=0.66,
        entropybits=30,
        length=8,
    )

    if not password:
        return False

    tested_pass = policy.password(password)
    result = tested_pass.test()
    if len(result) > 0:
        print(colored('Password not strong enough. Try increasing the length of the password or the password comlexity'))
        return False


    password_verification = getpass.getpass("Retype your password: ")
    if password != password_verification:
        print("Passwords do not match")
        return False

    return True

def validate_generate_mnemonic(mnemonic):
    if len(mnemonic) not in [12,15,18,21,24]:
        print(colored("Mnemonic has invalid size. This should be 12,15,18,21 or 24 words", 'red'))
        quit()

    try:
        keypair = Keypair.create_from_mnemonic(" ".join(mnemonic))
        return keypair
    except ValueError as e:
        print(colored(str(e), "red"))
        quit()



def gen_new_key(words):
    mnemonic = Keypair.generate_mnemonic(words)
    keypair = Keypair.create_from_mnemonic(mnemonic)

    return keypair


def display_mnemonic_msg(kepair : Keypair):
    mnemonic =kepair.mnemonic
    mnemonic_green = colored(mnemonic, 'green')
    print ("The mnemonic to the new key is:\n\n%s\n" % mnemonic_green)
    print ("You can use the mnemonic to recreate the key in case it gets lost. The command to use to regenerate the key using this mnemonic is:")
    print("python3 ./genkey.py regen --mnemonic %s" % mnemonic)
    print('')
    print (colored("It is important to store this mnemonic in a secure (preferable offline place), as anyone " \
                   "who has possesion of this mnemonic can use it to regenerate the key and access your tokens", "white"))



def save_keys(path, data):
    print("Writing key to %s" % path)
    with open(path, "wb") as keyfile:
        keyfile.write(data)

def set_file_permissions(path):
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    pass

def confirm_no_password():
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



def main():
    create_config_dir_if_not_exists()

    parser = ArgumentParser(description="Generate a key for bittensor")
    cmd_parsers = parser.add_subparsers(dest='command', required=True)

    new_key_parser = cmd_parsers.add_parser('new')
    new_key_parser.add_argument('--words',
                                type=int,
                                choices=[12,15,18,21,24],
                                default=12,
                                help="The amount of words the mnemonic representing the key will contain")
    new_key_parser.add_argument('--password', action='store_true', help='Protect the generated bittensor key with a password')
    new_key_parser.add_argument('--keyfile', help='The destination path of the keyfile (default: ~/.bittensor/keys)',
                        default='~/.bittensor/key')

    regen_key_parser = cmd_parsers.add_parser('regen')
    regen_key_parser.add_argument("--mnemonic", required=True, nargs="+")
    regen_key_parser.add_argument('--password', action='store_true', help='Protect the generated bittensor key with a password')
    regen_key_parser.add_argument('--keyfile', help='The destination path of the keyfile (default: ~/.bittensor/keys)',
                        default='~/.bittensor/key')


    args = parser.parse_args()
    keyfile = validate_path(args.keyfile)

    if not args.password and not confirm_no_password():
        quit()

    if args.command == 'new':
        keypair = gen_new_key(args.words)
        display_mnemonic_msg(keypair)
    else: # Implies regen
        keypair = validate_generate_mnemonic(args.mnemonic)

    data = json.dumps(keypair.toDict()).encode()

    if args.password:
        password = input_password()
        print("Encrypting key, this might take a while")
        data = encrypt(data, password)


    save_keys(keyfile, data)
    set_file_permissions(keyfile)

if __name__ == '__main__':
    main()





