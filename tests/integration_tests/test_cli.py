import os
import random
import subprocess
import sys
import time
import pytest
import bittensor

from substrateinterface.base import Keypair
from _pytest.fixtures import fixture
from sys import platform   
from loguru import logger
from unittest.mock import MagicMock


@fixture(scope="function")
def setup_chain():

    operating_system = "OSX" if platform == "darwin" else "Linux"
    path = "./bin/chain/{}/node-subtensor".format(operating_system)
    logger.info(path)
    if not path:
        logger.error("make sure the NODE_SUBTENSOR_BIN env var is set and points to the node-subtensor binary")
        sys.exit()

    # Select a port
    port = select_port()

    # Delete existing test wallets
    subprocess.run("rm -rf ~/.bittensor/wallets/*testwallet",  shell=True, check=True)
    
    # Purge chain first
    subprocess.Popen([path, 'purge-chain', '--dev', '-y'], close_fds=True)
    proc = subprocess.Popen([path, '--dev', '--port', str(port+1), '--ws-port', str(port), '--rpc-port', str(port + 2), '--tmp'], close_fds=True, shell=False)


    # Wait 4 seconds for the node to come up
    time.sleep(4)

    yield port

    # Wait 4 seconds for the node to come up
    time.sleep(4)

    # Kill process
    os.system("kill %i" % proc.pid)

@pytest.fixture(scope="session", autouse=True)
def initialize_tests():
    # Kill any running process before running tests
    os.system("pkill node-subtensor")

def select_port():
    port = random.randrange(1000, 65536, 5)
    return port


def generate_wallet(coldkey : 'Keypair' = None, hotkey: 'Keypair' = None):
    wallet = bittensor.wallet()   

    if not coldkey:
        coldkey = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    if not hotkey:
        hotkey = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())

    wallet.set_coldkey(coldkey, encrypt=False, overwrite=True)
    wallet.set_coldkeypub(coldkey, encrypt=False, overwrite=True)    
    wallet.set_hotkey(hotkey, encrypt=False, overwrite=True)

    return wallet

def setup_subtensor( port:int ):
    chain_endpoint = "localhost:{}".format(port)
    subtensor = bittensor.subtensor(
        chain_endpoint = chain_endpoint,
    )
    return subtensor, port

def construct_config():
    defaults = bittensor.Config()
    bittensor.subtensor.add_defaults( defaults )
    bittensor.dendrite.add_defaults( defaults )
    bittensor.axon.add_defaults( defaults )
    bittensor.wallet.add_defaults( defaults )
    bittensor.dataset.add_defaults( defaults )
    
    return defaults

def test_check_configs():
    commands = ["run", "transfer", "register", "unstake", 
    "stake", "overview", "new_coldkey", "new_hotkey", 
    "regen_coldkey", "regen_hotkey", "metagraph", "weights", 
    "set_weights", "inspect"]
    config = construct_config()
    config.no_prompt = True
    config.model = "template_miner"
    config.dest = "no_prompt"
    config.amount = 1
    config.mnemonic = "this is a mnemonic"
    config.uids = [1,2,3]
    config.weights = [0.25, 0.25, 0.25, 0.25]

    cli = bittensor.cli
    
    for cmd in commands:
        config.command = cmd
        cli.check_config(config)



def test_overview( setup_chain ):
        subtensor, port = setup_subtensor(setup_chain)
        subtensor.register = MagicMock(return_value = True)  
        
        config = construct_config()

        config.command = "overview"
        config.subtensor.network = "local"
        config.subtensor.chain_endpoint = "localhost:{}".format(port)
        config.no_prompt = True

        cli = bittensor.cli(config)
        cli.run()

def test_register( setup_chain ):
    subtensor, port = setup_subtensor(setup_chain)
    subtensor.register = MagicMock(return_value = True)  
    
    config = construct_config()

    config.command = "register"
    config.subtensor.network = "local"
    config.subtensor.chain_endpoint = "localhost:{}".format(port)
    config.no_prompt = True

    cli = bittensor.cli(config)
    cli.run()

def test_stake( setup_chain ):
    subtensor, port = setup_subtensor(setup_chain)
    subtensor.register = MagicMock(return_value = True)  
    
    config = construct_config()

    config.command = "stake"
    config.amount = 1
    config.subtensor.network = "local"
    config.stake_all = False
    config.subtensor.chain_endpoint = "localhost:{}".format(port)
    config.model = "template_miner"
    config.no_prompt = True

    cli = bittensor.cli(config)
    cli.run()


def test_unstake( setup_chain ):
    subtensor, port = setup_subtensor(setup_chain)
    subtensor.register = MagicMock(return_value = True)  
    
    #subtensor.register(wallet=wallet)    
    config = construct_config()

    config.command = "unstake"
    config.amount = 1
    config.subtensor.network = "local"
    config.unstake_all = False
    config.dest = "no_prompt"
    config.subtensor.chain_endpoint = "localhost:{}".format(port)
    config.model = "template_miner"
    config.no_prompt = True

    cli = bittensor.cli(config)
    cli.run()

def test_new_coldkey( setup_chain ):
    subtensor, port = setup_subtensor(setup_chain)
    subtensor.register = MagicMock(return_value = True)  
    
    #subtensor.register(wallet=wallet)    
    config = construct_config()
    config.wallet.name = "new_coldkey_testwallet"
    config.command = "new_coldkey"
    config.amount = 1
    config.subtensor.network = "local"
    config.dest = "no_prompt"
    config.subtensor.chain_endpoint = "localhost:{}".format(port)
    config.model = "template_miner"
    config.n_words = 12
    config.use_password = False
    config.no_prompt = True

    cli = bittensor.cli(config)
    cli.run()

def test_new_hotkey( setup_chain ):
    subtensor, port = setup_subtensor(setup_chain)
    subtensor.register = MagicMock(return_value = True)  
    
    #subtensor.register(wallet=wallet)    
    config = construct_config()
    config.wallet.name = "new_hotkey_testwallet"
    config.command = "new_hotkey"
    config.amount = 1
    config.subtensor.network = "local"
    config.dest = "no_prompt"
    config.subtensor.chain_endpoint = "localhost:{}".format(port)
    config.model = "template_miner"
    config.n_words = 12
    config.use_password = False
    config.no_prompt = True

    cli = bittensor.cli(config)
    cli.run()

def test_regen_coldkey( setup_chain ):
    subtensor, port = setup_subtensor(setup_chain)
    subtensor.register = MagicMock(return_value = True)  
    
    #subtensor.register(wallet=wallet)    
    config = construct_config()
    config.wallet.name = "regen_coldkey_testwallet"
    config.command = "regen_coldkey"
    config.amount = 1
    config.subtensor.network = "local"
    config.dest = "no_prompt"
    config.subtensor.chain_endpoint = "localhost:{}".format(port)
    config.model = "template_miner"
    config.mnemonic = "faculty decade seven jelly gospel axis next radio grain radio remain gentle"
    config.n_words = 12
    config.use_password = False
    config.no_prompt = True

    cli = bittensor.cli(config)
    cli.run()

def test_regen_hotkey( setup_chain ):
    subtensor, port = setup_subtensor(setup_chain)
    subtensor.register = MagicMock(return_value = True)  
    
    #subtensor.register(wallet=wallet)    
    config = construct_config()
    config.wallet.name = "regen_hotkey_testwallet"
    config.command = "regen_hotkey"
    config.amount = 1
    config.subtensor.network = "local"
    config.subtensor.chain_endpoint = "localhost:{}".format(port)
    config.model = "template_miner"
    config.mnemonic = "faculty decade seven jelly gospel axis next radio grain radio remain gentle"
    config.n_words = 12
    config.use_password = False
    config.no_prompt = True

    cli = bittensor.cli(config)
    cli.run()

def test_metagraph( setup_chain ):
    subtensor, port = setup_subtensor(setup_chain)
    subtensor.register = MagicMock(return_value = True)  
    
    #subtensor.register(wallet=wallet)    
    config = construct_config()
    config.wallet.name = "metagraph_testwallet"
    config.command = "metagraph"
    config.subtensor.network = "local"
    config.no_prompt = True
    config.subtensor.chain_endpoint = "localhost:{}".format(port)

    cli = bittensor.cli(config)
    cli.run()

def test_weights( setup_chain ):
    subtensor, port = setup_subtensor(setup_chain)
    subtensor.register = MagicMock(return_value = True)  
    
    #subtensor.register(wallet=wallet)    
    config = construct_config()
    config.wallet.name = "weights_testwallet"
    config.command = "weights"
    config.subtensor.network = "local"
    config.no_prompt = True
    config.subtensor.chain_endpoint = "localhost:{}".format(port)

    cli = bittensor.cli(config)
    cli.run()

def test_set_weights( setup_chain ):
    subtensor, port = setup_subtensor(setup_chain)
    subtensor.register = MagicMock(return_value = True)  
    
    #subtensor.register(wallet=wallet)    
    config = construct_config()
    config.wallet.name = "set_weights_testwallet"
    config.subtensor.network = "local"
    config.no_prompt = True
    config.uids = [1, 2, 3, 4]
    config.weights = [0.25, 0.25, 0.25, 0.25]
    config.subtensor.chain_endpoint = "localhost:{}".format(port)
    config.n_words = 12
    config.use_password = False

    # First create a new hotkey
    config.command = "new_hotkey"
    cli = bittensor.cli(config)
    cli.run()
    
    # Now set the weights
    config.command = "set_weights"
    cli = bittensor.cli(config)
    cli.run()

def test_inspect( setup_chain ):
    subtensor, port = setup_subtensor(setup_chain)
    subtensor.register = MagicMock(return_value = True)  
    
    #subtensor.register(wallet=wallet)    
    config = construct_config()
    config.wallet.name = "inspect_testwallet"
    config.subtensor.network = "local"
    config.no_prompt = True
    config.subtensor.chain_endpoint = "localhost:{}".format(port)
    config.n_words = 12
    config.use_password = False

    # First create a new coldkey
    config.command = "new_coldkey"
    cli = bittensor.cli(config)
    cli.run()

    # Now create a new hotkey
    config.command = "new_hotkey"
    cli = bittensor.cli(config)
    cli.run()

    # Now inspect it
    config.command = "inspect"
    cli = bittensor.cli(config)
    cli.run()

    config.command = "list"
    cli = bittensor.cli(config)
    cli.run()

