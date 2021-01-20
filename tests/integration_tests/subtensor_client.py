import asyncio
import random
import subprocess
from typing import List

from bittensor.balance import Balance
from bittensor.subtensor.client import WSClient
from bittensor.subtensor.interface import Keypair
from loguru import logger
import pytest
import os
import sys

# logger.remove() # Shut up loguru
from pytest import fixture
import time

@pytest.fixture(scope="session", autouse=True)
def initialize_tests():
    # Kill any running process before running tests
    os.system("pkill node-subtensor")



def select_port():
    port = random.randrange(1000, 65536, 5)

    # Check if port is in use
    proc1 = subprocess.Popen(['ss', '-tulpn'], stdout=subprocess.PIPE)
    proc2 = subprocess.Popen(['grep', str(port)], stdin=proc1.stdout,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    output = proc2.stdout

    if str(port) in output:
        return select_port()
    else:
        return port


@fixture(scope="function")
def setup_chain():
    path = os.getenv("NODE_SUBTENSOR_BIN", None)
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


def connect(keypair, port):
    socket = "localhost:%i" % port
    client = WSClient(socket, keypair)

    client.connect()
    return client


async def add_stake(port, coldkeypair : Keypair, hotkeypair : Keypair, amount : Balance):
    client = connect(coldkeypair, port)
    await client.is_connected()

    # Get the uid of the new neuron
    uid = await client.get_uid_for_pubkey(hotkeypair.public_key)
    assert uid is not None

    # Get the amount of stake, should be 0
    result = await client.get_stake_for_uid(uid)
    assert int(result) == int(Balance(0))

    # Add stake to new neuron
    await client.add_stake(amount, hotkeypair.public_key)


def generate_keypair():
    return Keypair.create_from_mnemonic(Keypair.generate_mnemonic())

'''
connect() tests
'''

@pytest.mark.asyncio
async def test_connect_success(setup_chain):
    logger.error(setup_chain)
    hotkeypair = Keypair.create_from_uri("//Alice")
    client = connect(hotkeypair, setup_chain)
    result = await client.is_connected()
    assert result is True


'''
subscribe() tests
'''

@pytest.mark.asyncio
async def test_subscribe_success(setup_chain):
    hotkeypair = generate_keypair()
    client = connect(hotkeypair, setup_chain)
    coldkeypair = Keypair.create_from_uri("//Alice")

    await client.is_connected()

    await client.subscribe("8.8.8.8", 6666, 0, coldkeypair.public_key)

    await asyncio.sleep(10) # Sleep for at least one block to give extrinsic the chance to be put onto the blockchain
    uid = await client.get_uid_for_pubkey(hotkeypair.public_key)
    assert uid is not None

'''
get_balance() tests
'''

@pytest.mark.asyncio
async def test_get_balance_success(setup_chain):
    hotkeypair = Keypair.create_from_uri("//Alice")
    client = connect(hotkeypair, setup_chain)

    await client.is_connected()

    result = await client.get_balance(hotkeypair.ss58_address)
    assert int(result) == pow(10, 9)

@pytest.mark.asyncio
async def test_get_balance_no_balance(setup_chain):
    hotkeypair = generate_keypair()
    client = connect(hotkeypair,setup_chain)

    await client.is_connected()

    result = await client.get_balance(hotkeypair.ss58_address)
    assert result == Balance(0)

'''
get_stake() tests
'''
@pytest.mark.asyncio
async def test_add_stake_success(setup_chain):
    coldkeypair = Keypair.create_from_uri("//Alice")
    hotkeypair  = generate_keypair()

    client = connect(hotkeypair, setup_chain)
    await client.is_connected()

    # Subscribe a new neuron with the hotkey
    await client.subscribe("8.8.8.8", 6666, 0, coldkeypair.public_key)
    await asyncio.sleep(10)

    # Now switch the connection the use the coldkey

    client = connect(coldkeypair, setup_chain)
    await client.is_connected()

    # Get the uid of the new neuron
    uid = await client.get_uid_for_pubkey(hotkeypair.public_key)
    assert uid is not None

    # Get the amount of stake, should be 0
    result = await client.get_stake_for_uid(uid)
    assert int (result) == int(Balance(0))

    # Add stake to new neuron
    result = await client.add_stake(Balance(4000), hotkeypair.public_key)

    assert result is not None
    assert 'extrinsic_hash' in result

    # Wait for the extrinsic to complete
    await asyncio.sleep(10)

    # Get the amount of stake
    result = await client.get_stake_for_uid(uid)
    assert int(result) == int(Balance(4000))


'''
@todo Build more tests. The above only tests the successful putting of the extrinsic on the chain
What is needed is logic that handles errors generated when handling the extrinsic
'''

@pytest.mark.asyncio
async def test_transfer_success(setup_chain):
    coldkey_alice = Keypair.create_from_uri("//Alice")
    coldkey_bob = Keypair.create_from_uri("//Bob")

    client = connect(coldkey_alice, setup_chain)
    await client.is_connected()

    balance_alice = await client.get_balance(coldkey_alice.ss58_address)
    balance_bob   = await client.get_balance(coldkey_bob.ss58_address)

    result = await client.transfer(coldkey_bob.public_key, Balance(pow(10, 4)))
    assert result is not None
    assert 'extrinsic_hash' in result

    # Wait until extrinsic is processed
    await asyncio.sleep(10)

    balance_alice_new = await client.get_balance(coldkey_alice.ss58_address)
    balance_bob_new = await client.get_balance(coldkey_bob.ss58_address)

    assert balance_alice_new < balance_alice
    assert balance_bob_new > balance_bob

@pytest.mark.asyncio
async def test_unstake_success(setup_chain):
    coldkeypair = Keypair.create_from_uri("//Alice")
    hotkeypair = generate_keypair()

    hotkey_client = connect(hotkeypair,setup_chain)
    await hotkey_client.is_connected()

    # Subscribe a new neuron with the hotkey
    await hotkey_client.subscribe("8.8.8.8", 6666, 0, coldkeypair.public_key)
    await asyncio.sleep(10)

    # Now switch the connection the use the coldkey

    coldkey_client = connect(coldkeypair, setup_chain)
    await coldkey_client.is_connected()

    # Get the uid of the new neuron
    uid = await coldkey_client.get_uid_for_pubkey(hotkeypair.public_key)
    logger.error(uid)
    assert uid is not None

    # Get the amount of stake, should be 0
    result = await coldkey_client.get_stake_for_uid(uid)
    assert int(result) == int(Balance(0))

    # Get the balance for the cold key, we use this for later comparison
    balance = await coldkey_client.get_balance(coldkeypair.public_key)

    # Add stake to new neuron
    result = await coldkey_client.add_stake(Balance(4000), hotkeypair.public_key)
    logger.info(result)

    assert result is not None
    assert 'extrinsic_hash' in result

    # Wait for the extrinsic to complete
    await asyncio.sleep(10)

    # Get current balance, should be 4000 less than first balance
    result = await coldkey_client.get_balance(coldkeypair.ss58_address)
    assert int(result) == int(balance) - 4000

    # Get the amount of stake, should be 4000
    result = await coldkey_client.get_stake_for_uid(uid)
    assert int(result) == int(Balance(4000))

    # Now do the actual unstake

    # Reconnect with coldkey account
    coldkey_client =  connect(coldkeypair, setup_chain)
    await coldkey_client.is_connected()

    # Do unstake
    result = await coldkey_client.unstake(Balance(4000), hotkeypair.public_key)
    assert result is not None
    assert 'extrinsic_hash' in result

    await asyncio.sleep(10)

    # Check if balance is the same as what we started with
    new_balance = await coldkey_client.get_balance(coldkeypair.ss58_address)
    assert int(new_balance) == int(balance)

@pytest.mark.asyncio
async def test_set_weights_success(setup_chain):
    hotkeypair_alice = Keypair.create_from_uri("//Alice")
    hotkeypair_bob = Keypair.create_from_uri("//Bob")

    coldkeypair = generate_keypair()

    client_alice = connect(hotkeypair_alice,setup_chain)
    client_bob   = connect(hotkeypair_bob,setup_chain)
    await client_alice.is_connected()
    await client_bob.is_connected()

    # Subscribe both alice and bob
    await client_alice.subscribe("8.8.8.8", 666, 0, coldkeypair.public_key)
    await client_bob.subscribe("8.8.8.8", 666, 0, coldkeypair.public_key)

    await asyncio.sleep(10)
    alice_uid = await client_alice.get_uid_for_pubkey(hotkeypair_alice.public_key)
    bob_uid = await client_bob.get_uid_for_pubkey(hotkeypair_bob.public_key)

    w_uids = [alice_uid, bob_uid]
    w_vals = [pow(2, 31)-1, pow(2,31)-1] # 50/50 distro

    result = await client_alice.set_weights(w_uids, w_vals, wait_for_inclusion=False)
    assert result is not None
    assert "extrinsic_hash" in result

    await asyncio.sleep(10)

    result = await client_alice.weight_uids_for_uid(alice_uid)
    assert result == w_uids

    result = await client_alice.weight_vals_for_uid(alice_uid)
    assert result == w_vals

@pytest.mark.asyncio
async def test_get_current_block(setup_chain):
    keypair = Keypair.create_from_uri("//Alice")
    client = connect(keypair, setup_chain)

    await client.is_connected()
    result = await client.get_current_block()
    assert result >= 0

@pytest.mark.asyncio
async def test_get_active(setup_chain):
    keypair = Keypair.create_from_uri("//Alice")
    client = connect(keypair,setup_chain)

    await client.is_connected()

    # Subscribe at least one
    await client.subscribe("8.8.8.8", 666, 0, keypair.public_key)
    await asyncio.sleep(10)

    result = await client.get_active()

    assert isinstance(result, List)
    assert len(result) > 0

    elem = result[0]
    assert isinstance(elem[0], str)
    assert elem[0][:2] == "0x"
    assert len(elem[0][2:]) == 64
    assert isinstance(elem[1], int)

@pytest.mark.asyncio
async def test_get_uid_for_pubkey_succes(setup_chain):
    keypair = generate_keypair()
    client = connect(keypair,setup_chain)
    await client.is_connected()

    # subscribe first
    await client.subscribe("8.8.8.8", 6666, 0, keypair.public_key)
    await asyncio.sleep(10)

    # Get the id
    result = await client.get_uid_for_pubkey(keypair.public_key)
    assert result is not None

# @Todo write tests for non happy paths

@pytest.mark.asyncio
async def test_get_neuron_for_uid(setup_chain):
    hotkey = generate_keypair()
    coldkey = generate_keypair()

    client = connect(hotkey,setup_chain)
    await client.is_connected()

    # subscribe first
    await client.subscribe("8.8.8.8", 6666, 0, coldkey.public_key)
    await asyncio.sleep(10)

    uid = await client.get_uid_for_pubkey(hotkey.public_key)

    result = await client.get_neuron_for_uid(uid)

    assert isinstance(result, dict)
    assert "coldkey" in result
    assert "hotkey" in result
    assert "ip_type" in result
    assert "modality" in result
    assert "port" in result
    assert "uid" in result

    assert result['coldkey'] == coldkey.public_key
    assert result['hotkey'] == hotkey.public_key
    assert result['ip_type'] == 4
    assert result['modality'] == 0
    assert result['port'] == 6666
    assert result['uid'] == uid


@pytest.mark.asyncio
async def test_get_neurons(setup_chain):
    hotkey_1 = generate_keypair()
    hotkey_2 = generate_keypair()

    coldkey = generate_keypair()

    client_1 = connect(hotkey_1,setup_chain)
    client_2 = connect(hotkey_2,setup_chain)

    await client_1.is_connected()
    await client_2.is_connected()

    # subscribe 2 neurons
    await client_1.subscribe("8.8.8.8", 6666, 0, coldkey.public_key)
    await client_2.subscribe("8.8.8.8", 6666, 0, coldkey.public_key)
    await asyncio.sleep(10)

    result = await client_1.neurons()
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

'''
get_stake_for_uid() tests
'''

@pytest.mark.asyncio
async def test_get_stake_for_uid___has_stake(setup_chain):
    hotkeyp = generate_keypair()
    coldkeyp = Keypair.create_from_uri("//Alice")

    client = connect(hotkeyp,setup_chain)
    await client.is_connected()

    await client.subscribe("8.8.8.8", 667, 0, coldkeyp.public_key)
    await asyncio.sleep(10)

    uid = await client.get_uid_for_pubkey(hotkeyp.public_key)

    await add_stake(setup_chain, coldkeyp, hotkeyp, Balance(4000))
    await asyncio.sleep(10)

    result = await client.get_stake_for_uid(uid)
    assert int(result) == 4000


@pytest.mark.asyncio
async def test_get_stake_for_uid___has_no_stake(setup_chain):
    hotkeyp = generate_keypair()
    coldkeyp = Keypair.create_from_uri("//Alice")

    client = connect(hotkeyp, setup_chain)
    await client.is_connected()

    await client.subscribe("8.8.8.8", 667, 0, coldkeyp.public_key)
    await asyncio.sleep(10)

    uid = await client.get_uid_for_pubkey(hotkeyp.public_key)

    result = await client.get_stake_for_uid(uid)
    assert int(result) == 0



@pytest.mark.asyncio
async def test_get_stake_for_uid___unknown_uid(setup_chain):
    hotkeyp = generate_keypair()

    client = connect(hotkeyp, setup_chain)
    await client.is_connected()

    result = await client.get_stake_for_uid(999999999)
    assert int(result) == 0


@pytest.mark.asyncio
async def test_weight_uids_for_uid__weight_vals_for_uid(setup_chain):
    hotkeypair_1 = generate_keypair()
    hotkeypair_2 = generate_keypair()

    coldkeypair = generate_keypair()

    client_1 = connect(hotkeypair_1, setup_chain)
    client_2   = connect(hotkeypair_2, setup_chain)
    await client_1.is_connected()
    await client_2.is_connected()

    # Subscribe both alice and bob
    await client_1.subscribe("8.8.8.8", 666, 0, coldkeypair.public_key)
    await client_2.subscribe("8.8.8.8", 666, 0, coldkeypair.public_key)

    await asyncio.sleep(10)
    uid_1 = await client_1.get_uid_for_pubkey(hotkeypair_1.public_key)
    uid_2 = await client_2.get_uid_for_pubkey(hotkeypair_2.public_key)

    w_uids = [uid_1, uid_2]
    w_vals = [pow(2, 31)-1, pow(2,31)-1] # 50/50 distro

    await client_1.set_weights(w_uids, w_vals, wait_for_inclusion=False)
    await asyncio.sleep(10)

    result = await client_1.weight_vals_for_uid(uid_1)
    assert isinstance(result, List)
    assert len(result) == 2
    assert isinstance(result[0], int)
    assert isinstance(result[1], int)

    result = await client_1.weight_uids_for_uid(uid_1)
    assert isinstance(result, List)
    assert len(result) == 2
    assert isinstance(result[0], int)
    assert isinstance(result[1], int)

    assert result[0] == uid_1
    assert result[1] == uid_2

@pytest.mark.asyncio
async def test_get_last_emit_data_for_uid__success(setup_chain):
    hotkeypair_1 = generate_keypair()
    coldkeypair = generate_keypair()

    client = connect(hotkeypair_1, setup_chain)
    await client.is_connected()

    current_block = await client.get_current_block()
    await client.subscribe("8.8.8.8", 666, 0, coldkeypair.public_key)
    await asyncio.sleep(10)
    uid = await client.get_uid_for_pubkey(hotkeypair_1.public_key)

    result = await client.get_last_emit_data_for_uid(uid)

    assert result in [current_block, current_block + 1, current_block + 2]


@pytest.mark.asyncio
async def test_get_last_emit_data_for_uid__no_uid(setup_chain):
    hotkey = generate_keypair()

    client = connect(hotkey, setup_chain)
    await client.is_connected()

    result = await client.get_last_emit_data_for_uid(99999)
    assert result is None



@pytest.mark.asyncio
async def test_get_last_emit_data(setup_chain):
    hotkey_1 = generate_keypair()
    hotkey_2 = generate_keypair()

    coldkey = generate_keypair()

    client_1 = connect(hotkey_1, setup_chain)
    client_2 = connect(hotkey_2, setup_chain)

    await client_1.is_connected()
    await client_2.is_connected()

    # subscribe 2 neurons
    await client_1.subscribe("8.8.8.8", 6666, 0, coldkey.public_key)
    await client_2.subscribe("8.8.8.8", 6666, 0, coldkey.public_key)
    await asyncio.sleep(10)


    result = await client_1.get_last_emit_data()
    assert isinstance(result, List)
    assert len(result) >= 2
    elem = result[0]
    assert isinstance(elem, List)
    assert isinstance(elem[0], int)
    assert isinstance(elem[1], int)

