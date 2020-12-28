#!/bin/python3

from argparse import ArgumentParser

from bittensor.balance import Balance
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
from bittensor.subtensor.client import Neuron, Neurons


class CommandExecutor:
    __keypair : Keypair
    __client : WSClient
    def __init__(self, keypair : Keypair, client : WSClient):
        self.__keypair = keypair
        self.__client = client

    async def connect(self):
        self.__client.connect()
        await self.__client.is_connected()

    async def _associated_neurons(self) -> Neurons:
        pubkey = self.__keypair.public_key

        logger.debug("Retrieving all nodes associated with cold key : {}", pubkey)

        neurons = await self.__client.neurons(decorator=True)

        result = filter(lambda x : x.coldkey == pubkey, neurons )# These are the neurons associated with the provided cold key

        associated_neurons = Neurons(result)

        # Load stakes
        for neuron in associated_neurons:
            neuron.stake = await self.__client.get_stake_for_uid(neuron.uid)


        return associated_neurons


    async def overview(self):
        await self.connect()
        balance = await self.__client.get_balance(self.__keypair.ss58_address)
        neurons = await self._associated_neurons()

        print("BALANCE: %s : [%s]" % (self.__keypair.ss58_address, balance))
        print()
        print("--===[[ STAKES ]]===--")
        t = PrettyTable(["UID", "IP", "STAKE"])
        t.align = 'l'
        for neuron in neurons:
            t.add_row([neuron.uid, neuron.ip, neuron.stake])

        print(t.get_string())

    async def unstake_all(self):
        await self.connect()
        neurons = await self._associated_neurons()
        for neuron in neurons:
            neuron.stake = await self.__client.get_stake_for_uid(neuron.uid)
            await self.__client.unstake(neuron.stake, neuron.hotkey)

    async def unstake(self, uid, amount : Balance):
        await self.connect()
        neurons = await self._associated_neurons()
        neuron = neurons.get_by_uid(uid)
        if not neuron:
            logger.error("Neuron with uid {} is not associated with this cold key")
            quit()

        neuron.stake = await self.__client.get_stake_for_uid(uid)
        if amount > neuron.stake:
            logger.error("Neuron with uid {} does not have enough stake ({}) to be able to unstake {}", uid, neuron.stake, amount)
            quit()

        await self.__client.unstake(amount, neuron.hotkey)

    async def stake(self, uid, amount : Balance):
        await self.connect()
        balance = await self.__client.get_balance(self.__keypair.ss58_address)
        if balance < amount:
            logger.error("Not enough balance ({}) to stake {}", balance, amount)

        neurons = await self._associated_neurons()
        neuron = neurons.get_by_uid(uid)
        if not neuron:
            logger.error("Neuron with uid {} is not associated with this cold key")
            quit()

        await self.__client.add_stake(amount, neuron.hotkey)






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

def get_client(endpoint, keypair):
    return WSClient(socket=endpoint, keypair=keypair)

def enable_debug(should_debug):
    if not should_debug:
        logger.remove()
        logger.add(sink=sys.stderr, level="INFO")




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
    unstake_parser = cmd_parsers.add_parser('unstake')
    unstake_parser.add_argument('--all', dest="unstake_all", action='store_true')
    unstake_parser.add_argument('--uid', dest="uid", type=int, required=False)
    unstake_parser.add_argument('--amount', dest="amount", type=float, required=False)

    stake_parser = cmd_parsers.add_parser('stake')
    stake_parser.add_argument('--uid', dest="uid", type=int, required=False)
    stake_parser.add_argument('--amount', dest="amount", type=float, required=False)


    args = parser.parse_args()

    endpoint = args.chain_endpoint
    enable_debug(args.debug)

    __validate_path(args.cold_key)
    keypair = load_key(args.cold_key)

    client = get_client(endpoint, keypair)
    executor = CommandExecutor(keypair, client)
    loop = asyncio.get_event_loop()

    if args.command == "overview":
        loop.run_until_complete(executor.overview())

    if args.command == "unstake":
        if args.unstake_all:
            confirm = input("This will remove all stake from associated neurons, and transfer the balance in the account associated with the cold key. Continue? (y/N) ")
            if confirm not in (["Y", 'y']):
                quit()
            loop.run_until_complete(executor.unstake_all())
            quit()


        if not args.uid:
            logger.error("The --uid argument is required for this command")
            quit()

        if not args.amount:
            logger.error("The --amount argument is required for this command")
            quit()

        amount = Balance.from_float(args.amount)

        loop.run_until_complete(executor.unstake(args.uid, amount))

    if args.command == "stake":
        if not args.uid:
            logger.error("The --uid argument is required for this command")
            quit()

        if not args.amount:
            logger.error("The --amount argument is required for this command")
            quit()

        amount = Balance.from_float(args.amount)
        loop.run_until_complete(executor.stake(args.uid, amount))



if __name__ == '__main__':
    main()
