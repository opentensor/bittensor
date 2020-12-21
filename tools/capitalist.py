#!/bin/python3

from argparse import ArgumentParser
from bittensor.subtensor.client import WSClient
from bittensor.subtensor.interface import Keypair
from loguru import logger
import json
from bittensor.crypto import is_encrypted, decrypt_data, KeyError
from bittensor.crypto.keyfiles import load_keypair_from_data, KeyFileError
from bittensor.utils import Cli
import asyncio


import os

def __validate_path(path):
    path = os.path.expanduser(path)

    if not os.path.isfile(path):
        logger.error("{} is not a file. Aborting", path)
        quit()

    if not os.access(path, os.R_OK):
        logger.error("{} is not readable. Aborting", path)
        quit()

def load_key(path) -> Keypair:
    path = os.path.expanduser(path)
    try:
        with open(path, 'rb') as file:
            data = file.read()
            if is_encrypted(data):
                password = Cli.ask_password()
                data = decrypt_data(password, data)

            return load_keypair_from_data(data)

    except KeyError:
        logger.error("Invalid password")
        quit()
    except KeyFileError as e:
        logger.error("Keyfile corrupt")
        raise e


async def balance(socket, keypair : Keypair):
    client = WSClient(socket=socket, keypair=keypair)
    client.connect()

    await client.is_connected()

    balance = await client.get_balance(keypair.ss58_address)
    logger.info("Balance: {}", balance)
    pass




'''

Functions :
- Generate cold key
- View balance
- View hotkeys associated with the supplied cold key
- Stake funds into hotkey ( one by one / amount devided equally over keys)
- Unstake funds into coldkey (one by one / withdraw all)


'''

def main():
    parser = ArgumentParser(description="Capitalism yeey")
    parser.add_argument("--chain-endpoint", default="localhost:9944", required=False, help="The endpoint to the subtensor chain <hostname/ip>:<port>")
    parser.add_argument("--cold-key", default='~/.bittensor/cold_key', help="Path to the cold key")

    cmd_parsers = parser.add_subparsers(dest='command', required=True)
    balance_parser = cmd_parsers.add_parser('balance')

    args = parser.parse_args()
    endpoint = args.chain_endpoint

    print(args)

    __validate_path(args.cold_key)
    keypair = load_key(args.cold_key)

    if (args.command == "balance"):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(balance(endpoint, keypair))



    print(keypair)


if __name__ == '__main__':
    main()