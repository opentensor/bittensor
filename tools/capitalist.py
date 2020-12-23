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
import sys

import os

from prettytable import PrettyTable
from subtensor.client import Neuron


class CommandExecutor:
    __keypair : Keypair
    __client : WSClient
    def __init__(self, keypair : Keypair, client : WSClient):
        self.__keypair = keypair
        self.__client = client

    async def connect(self):
        self.__client.connect()
        await self.__client.is_connected()

    async def _associated_neurons(self):
        pubkey = self.__keypair.public_key

        logger.debug("Retrieving all nodes associated with cold key : {}", pubkey)

        neurons = await self.__client.neurons(decorator=True)

        result = filter(lambda x : x.coldkey == pubkey, neurons )# These are the neurons associated with the provided cold key

        associated_neurons = list(result)

        # Load stakes
        for neuron in associated_neurons:
            neuron.stake = await self.__client.get_stake_for_uid(neuron.uid)


        return associated_neurons




    async def overview(self):
        balance = await self.__client.get_balance(self.__keypair.ss58_address)
        logger.info("Balance: {}", balance)

        neurons = await self._associated_neurons()

        t = PrettyTable(["uid", "ip", "stake"])
        t.align = 'l'
        for neuron in neurons:
            t.add_row([neuron.uid, neuron.ip, neuron.stake])

        print(t.get_string())









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



def enable_debug(should_debug):
    if not should_debug:
        logger.remove()
        logger.add(sink=sys.stderr, level="INFO")


async def overview(socket, keypair : Keypair):


    client = WSClient(socket=socket, keypair=keypair)
    exec = CommandExecutor(keypair, client)
    await exec.connect()
    overview = await exec.overview()



    pass


async def get_hotkeys(keypair : Keypair):
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
    parser.add_argument("--chain-endpoint", default="feynman.kusanagi.bittensor.com:9944", required=False, help="The endpoint to the subtensor chain <hostname/ip>:<port>")
    parser.add_argument("--cold-key", default='~/.bittensor/cold_key', help="Path to the cold key")
    parser.add_argument("--debug", default=False, help="Turn on debugging information", action="store_true")

    cmd_parsers = parser.add_subparsers(dest='command', required=True)
    overview_parser = cmd_parsers.add_parser('overview')

    args = parser.parse_args()
    endpoint = args.chain_endpoint
    enable_debug(args.debug)

    __validate_path(args.cold_key)
    keypair = load_key(args.cold_key)

    if (args.command == "overview"):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(overview(endpoint, keypair))



    print(keypair)


if __name__ == '__main__':
    main()