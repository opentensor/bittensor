import os
import sys
import time
import pytest
import asyncio
import random
import subprocess

from typing import List
from loguru import logger
logger = logger.opt(ansi=True)
from pytest import fixture

import bittensor
from bittensor.utils.balance import Balance

@pytest.fixture(scope="session", autouse=True)
def initialize_tests():
    # Kill any running process before running tests
    os.system("pkill node-subtensor")

def select_port():
    port = random.randrange(1000, 65536, 5)
    return port

@fixture(scope="function")
def setup_chain():
    path = os.getenv("NODE_SUBTENSOR_BIN", None)
    logger.info(path)
    if not path:
        logger.error("make sure the NODE_SUBTENSOR_BIN env var is set and points to the node-subtensor binary")
        quit()

    # Select a port
    port = select_port()

    proc = subprocess.Popen([path, '--dev', '--port', str(port+1), '--ws-port', str(port), '--rpc-port', str(port + 2), '--tmp'], close_fds=True, shell=False)

    # Wait 2 seconds for the node to come up
    time.sleep(2)

    yield port

    # Kill process
    os.system("kill %i" % proc.pid)


def connect( port:int ):
    chain_endpoint = "localhost:%i" % port
    subtensor = bittensor.subtensor.Subtensor(
        chain_endpoint = chain_endpoint,
    )
    subtensor.connect()
    return subtensor

async def add_stake( port:int, wallet:bittensor.wallet.Wallet, amount: Balance ):
    subtensor = connect( port )
    await subtensor.is_connected()

    # Get the uid of the new neuron
    uid = await subtensor.get_uid_for_pubkey( wallet.hotkey.public_key )
    assert uid is not None

    # Get the amount of stake, should be 0
    result = await subtensor.get_stake_for_uid( uid )
    assert int(result) == int(Balance(0))

    # Add stake to new neuron
    await subtensor.add_stake( wallet = wallet, amount = amount, hotkey_id = hotkeypair.public_key )

def generate_wallet( name:str = 'pytest' ):
    wallet = bittensor.wallet.Wallet(
        path = '/tmp/pytest',
        name = name,
        hotkey = 'pytest'
    )
    if not wallet.has_coldkey:
        wallet.create_new_coldkey(use_password=False)
    if not wallet.has_hotkey:
        wallet.create_new_hotkey(use_password=False)
    assert wallet.has_coldkey
    assert wallet.has_hotkey
    return wallet
    
def subscribe( subtensor, wallet):
    subtensor.subscribe(
        wallet = wallet,
        ip = "8.8.8.8", 
        port = 6666, 
        modality = bittensor.proto.Modality.TEXT,
        wait_for_finalization = True,
        timeout = 6 * bittensor.__blocktime__,
    )
    assert subtensor.async_is_subscribed (
        wallet = wallet,
        ip = "8.8.8.8",
        port = 6666, 
        modality = bittensor.proto.Modality.TEXT,
    )

'''
connect() tests
'''

def test_connect_success(setup_chain):
    logger.error(setup_chain)
    wallet = generate_wallet()
    subtensor = connect(setup_chain)
    result = subtensor.is_connected()
    assert result is True

'''
subscribe() tests
'''

def test_subscribe_success(setup_chain):
    wallet = generate_wallet()
    subtensor = connect(setup_chain)
    subtensor.is_connected()
    subscribe(subtensor, wallet)
    uid = subtensor.get_uid_for_pubkey(wallet.hotkey.public_key)
    assert uid is not None

def test_get_balance_success(setup_chain):
    wallet = generate_wallet()
    subtensor = connect(setup_chain)
    subtensor.is_connected()
    result = subtensor.get_balance(wallet.hotkey.ss58_address)
    assert result == Balance(0)

def test_get_uid_for_pubkey_succes(setup_chain):
    wallet = generate_wallet()
    subtensor = connect(setup_chain)
    subtensor.is_connected()
    subscribe(subtensor, wallet)
    result = subtensor.get_uid_for_pubkey(wallet.hotkey.public_key)
    assert result is not None

def test_get_current_block(setup_chain):
    wallet = generate_wallet()
    subtensor = connect(setup_chain)
    subtensor.is_connected()
    block = subtensor.get_current_block()
    assert block >= 0

def test_get_active(setup_chain):
    wallet = generate_wallet()
    subtensor = connect(setup_chain)
    subtensor.is_connected()
    subscribe(subtensor, wallet)
    result = subtensor.get_active()
    assert isinstance(result, List)
    assert len(result) > 0
    elem = result[0]
    assert isinstance(elem[0], str)
    assert elem[0][:2] == "0x"
    assert len(elem[0][2:]) == 64
    assert isinstance(elem[1], int)

def test_get_stake_for_uid___unknown_uid(setup_chain):
    wallet = generate_wallet()
    subtensor = connect(setup_chain)
    subtensor.is_connected()
    result = subtensor.get_stake_for_uid(999999999)
    assert int(result) == 0

def test_get_neuron_for_uid(setup_chain):
    wallet = generate_wallet()
    subtensor = connect(setup_chain)
    subtensor.is_connected()
    subscribe(subtensor, wallet)
    uid = subtensor.get_uid_for_pubkey(wallet.hotkey.public_key)
    result = subtensor.get_neuron_for_uid(uid)

    assert isinstance(result, dict)
    assert "coldkey" in result
    assert "hotkey" in result
    assert "ip_type" in result
    assert "modality" in result
    assert "port" in result
    assert "uid" in result

    assert result['coldkey'] == wallet.coldkey.public_key
    assert result['hotkey'] == wallet.hotkey.public_key
    assert result['ip_type'] == 4
    assert result['modality'] == 0
    assert result['port'] == 6666
    assert result['uid'] == uid

def test_get_last_emit_data_for_uid__success(setup_chain):
    wallet = generate_wallet()
    subtensor = connect(setup_chain)
    subtensor.is_connected()
    subscribe(subtensor, wallet)
    uid = subtensor.get_uid_for_pubkey(wallet.hotkey.public_key)
    result = subtensor.get_last_emit_data_for_uid( uid )
    current_block = subtensor.get_current_block()
    assert result < current_block

def test_get_last_emit_data_for_uid__no_uid(setup_chain):
    wallet = generate_wallet()
    subtensor = connect(setup_chain)
    subtensor.is_connected()
    result = subtensor.get_last_emit_data_for_uid( 999999 )
    assert result is None

def test_get_neurons(setup_chain):
    walletA = generate_wallet('A')
    walletB = generate_wallet('B')

    subtensorA = connect(setup_chain)
    subtensorB = connect(setup_chain)
    subtensorA.is_connected()
    subtensorB.is_connected()

    subscribe( subtensorA, walletA )
    subscribe( subtensorB, walletB )

    result = subtensorA.neurons()
    assert isinstance(result, List)
    assert len(result) >= 2
    elem = result[0]
    assert isinstance(elem, list)
    assert isinstance(elem[0], int) # This is the uid, which is the first element of the list
    assert isinstance(elem[1], dict) # The second element is a neuron dict

    assert "coldkey" in elem[1]
    assert "hotkey" in elem[1]
    assert "ip_type" in elem[1]
    assert "modality" in elem[1]
    assert "port" in elem[1]
    assert "uid" in elem[1]

    assert isinstance(elem[1]['coldkey'], str)
    assert isinstance(elem[1]['hotkey'], str)
    assert isinstance(elem[1]['ip_type'], int)
    assert isinstance(elem[1]['modality'], int)
    assert isinstance(elem[1]['port'], int)
    assert isinstance(elem[1]['uid'], int)
    assert elem[1]['uid'] == elem[0]

def test_set_weights_success(setup_chain):
    walletA = generate_wallet('A')
    walletB = generate_wallet('B')

    subtensorA = connect(setup_chain)
    subtensorB = connect(setup_chain)
    subtensorA.is_connected()
    subtensorB.is_connected()

    subscribe( subtensorA, walletA )
    subscribe( subtensorB, walletB )

    uidA = subtensorA.get_uid_for_pubkey(walletA.hotkey.public_key)
    uidB = subtensorB.get_uid_for_pubkey(walletB.hotkey.public_key)

    w_uids = [uidA, uidB]
    w_vals = [pow(2, 31)-1, pow(2,31)-1]
    subtensorA.set_weights(
        wallet = self.wallet, 
        destinations = w_uids, 
        values = w_vals, 
        wait_for_finalization=True, 
        timeout = 4 * bittensor.__blocktime__
    )
    subtensorB.set_weights (
        wallet = self.wallet, 
        destinations = w_uids, 
        values = w_vals, 
        wait_for_finalization=True, 
        timeout = 4 * bittensor.__blocktime__
    )

    result_uids = subtensorA.weight_uids_for_uid(uidA)
    result_vals = subtensorA.weight_vals_for_uid(uidA)
    assert result_uids == w_uids
    assert result_vals == w_vals

def test_get_stake_for_uid___has_no_stake(setup_chain):
    wallet = generate_wallet()
    subtensor = connect(setup_chain)
    subtensor.is_connected()
    subscribe(subtensor, wallet)
    uid = subtensor.get_uid_for_pubkey(wallet.hotkey.public_key)
    result = subtensor.get_stake_for_uid(uid)
    assert int(result) == 0

# TODO(const): Tests to be added back once functionality for Alice and Bob are added.
# -- transfer funds to accounts
# -- stake and unstake real funds