import os
import sys
import time
import pytest
import asyncio
import random
import subprocess

from typing import List
from loguru import logger
from pytest import fixture

import bittensor
from bittensor.utils.balance import Balance

from bittensor.wallet import Wallet
from bittensor.substrate import Keypair


BLOCK_REWARD = 500_000_000
TRANSACTION_FEE = 100
TRANSACTION_FEE_ADD_STAKE = 100 * 145  # Fee per byte * extrinsic length
TRANSACTION_FEE_UNSTAKE = 100 * 145
TRANSACTION_FEE_TRANSFER = 100 * 139

class WalletStub(Wallet):
    def __init__(self, coldkey_pair: 'Keypair', hotkey_pair: 'Keypair'):
        self._hotkey = hotkey_pair
        self._coldkey = coldkey_pair
        self._coldkeypub = coldkey_pair.public_key



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

def add_stake( subtensor, wallet: 'wallet:bittensor.wallet.Wallet', amount: 'Balance' ):
    # Get the uid of the new neuron
    uid = subtensor.get_uid_for_pubkey( wallet.hotkey.public_key )
    assert uid is not None

    # Add stake to new neuron
    result = subtensor.add_stake( wallet = wallet, amount = amount, hotkey_id = wallet.hotkey.public_key, wait_for_finalization=True, timeout=30 )
    assert result == True

def generate_wallet(coldkey_pair : 'Keypair' = None, hotkey_pair: 'Keypair' = None):
    if not coldkey_pair:
        coldkey_pair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())

    if not hotkey_pair:
        hotkey_pair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())

    return WalletStub(coldkey_pair=coldkey_pair, hotkey_pair=hotkey_pair)

    
def subscribe( subtensor, wallet):
    subtensor.subscribe(
        wallet = wallet,
        ip = "8.8.8.8", 
        port = 6666, 
        modality = bittensor.proto.Modality.TEXT,
        wait_for_finalization = True,
        timeout = 6 * bittensor.__blocktime__,
    )
    assert subtensor.is_subscribed (
        wallet = wallet,
        ip = "8.8.8.8",
        port = 6666, 
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

'''
get_balance() tests
'''

def test_get_balance_no_balance(setup_chain):
    wallet = generate_wallet()
    subtensor = connect(setup_chain)
    subtensor.is_connected()
    result = subtensor.get_balance(wallet.hotkey.ss58_address)
    assert result == Balance(0)


def test_get_balance_success(setup_chain):
    hotkey_pair = Keypair.create_from_uri('//Alice')
    subtensor = connect(setup_chain)
    subtensor.is_connected()
    result = subtensor.get_balance(hotkey_pair.ss58_address)
    assert int(result) == pow(10, 9)



'''
add_stake() tests
'''

def test_add_stake_success(setup_chain):
    coldkeypair = Keypair.create_from_uri("//Alice")
    hotkeypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())

    wallet = generate_wallet(coldkey_pair=coldkeypair, hotkey_pair=hotkeypair)
    subtensor = connect(setup_chain)
    subtensor.is_connected()

    # Subscibe the hotkey using Alice's cold key, which has TAO
    subscribe(subtensor, wallet)

    uid = subtensor.get_uid_for_pubkey(hotkeypair.public_key)
    assert uid is not None

    #Verify the node has 0 stake
    result = subtensor.get_stake_for_uid(uid)
    assert int(result) == int(Balance(0))

    # Get balance
    balance_pre = subtensor.get_balance(coldkeypair.ss58_address)

    # Timeout is 30, because 3 * blocktime does not work.
    result = subtensor.add_stake(wallet, Balance(4000), hotkeypair.public_key, wait_for_finalization=True, timeout=30)
    assert result == True

    # Check if the amount of stake ends up in the hotkey account
    result = subtensor.get_stake_for_uid(uid)
    assert int(result) == int(Balance(4000))

    # Check if the balances had reduced by the amount of stake + the transaction fee for the staking operation
    balance_post = subtensor.get_balance(coldkeypair.ss58_address)
    assert int(balance_post) == int(balance_pre) - (4000 + TRANSACTION_FEE_ADD_STAKE)


'''
unstake() tests
'''

def test_unstake_success(setup_chain):
    coldkeypair = Keypair.create_from_uri('//Alice')
    hotkey_pair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    wallet = generate_wallet(coldkey_pair=coldkeypair, hotkey_pair=hotkey_pair)

    subtensor = connect(setup_chain)
    subtensor.is_connected()

    subscribe(subtensor, wallet)

    # Get the balance for the cold key, we use this for later comparison
    balance_pre = int(subtensor.get_balance(coldkeypair.public_key))

    add_stake(subtensor, wallet, Balance(4000))

    # Determine the cost of the add_stake transaction
    balance_post = int(subtensor.get_balance(coldkeypair.public_key))
    transaction_fee_add_stake = balance_pre - balance_post - 4000

    logger.error("Trans_fee add_stake: {}", transaction_fee_add_stake)

    # unstake incurs a transaction fee that is added to the block reward
    result = subtensor.unstake(amount=Balance(3000), wallet=wallet, hotkey_id=hotkey_pair.public_key, wait_for_finalization=True, timeout=30)
    assert result is True

    transaction_fee_unstake = balance_post - int(subtensor.get_balance(coldkeypair.public_key)) + 3000
    logger.error("Trans_fee add_stake: {}", transaction_fee_unstake)

    assert int(transaction_fee_unstake) == TRANSACTION_FEE_UNSTAKE

    # At this point, the unstake transaction fee is in the transaction_fee_pool, and will make it into the block
    # reward the next block. However, in order to get this reward into the hotkey account of the neuron,
    # and emit needs to take place. This is why the expectation does not include the unstake transaction fee

    uid = subtensor.get_uid_for_pubkey(hotkey_pair.public_key)
    stake = subtensor.get_stake_for_uid(uid)
    expectation = 1000 + (3 * BLOCK_REWARD) + TRANSACTION_FEE_ADD_STAKE

    assert int(stake) == expectation


'''
get_stake_for_uid() tests
'''

def test_get_stake_for_uid___has_stake(setup_chain):
    hotkeypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    coldkeypair = Keypair.create_from_uri('//Alice')

    wallet = generate_wallet(coldkey_pair=coldkeypair, hotkey_pair=hotkeypair)
    subtensor = connect(setup_chain)
    subtensor.is_connected()

    subscribe(subtensor, wallet)
    uid = subtensor.get_uid_for_pubkey(hotkeypair.public_key)

    add_stake(subtensor,wallet,Balance(4000))

    result = subtensor.get_stake_for_uid(uid)
    assert int(result) == 4000


def test_get_stake_for_uid___has_no_stake(setup_chain):
    hotkeypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    coldkeypair = Keypair.create_from_uri('//Alice')

    wallet = generate_wallet(coldkey_pair=coldkeypair, hotkey_pair=hotkeypair)
    subtensor = connect(setup_chain)
    subtensor.is_connected()

    subscribe(subtensor, wallet)
    uid = subtensor.get_uid_for_pubkey(hotkeypair.public_key)

    result = subtensor.get_stake_for_uid(uid)
    assert int(result) == 0

def test_get_stake_for_uid___unknown_uid(setup_chain):
    client = connect(setup_chain)
    client.is_connected()

    result = client.get_stake_for_uid(999999999)
    assert int(result) == 0




'''
transfer() tests
'''

def test_transfer_success(setup_chain):
    coldkey_alice = Keypair.create_from_uri("//Alice")
    coldkey_bob = Keypair.create_from_uri("//Bob")

    subtensor = connect(setup_chain)
    subtensor.is_connected()

    balance_alice_pre = subtensor.get_balance(coldkey_alice.ss58_address)
    balance_bob_pre = subtensor.get_balance(coldkey_bob.ss58_address)

    wallet_alice = generate_wallet(coldkey_pair=coldkey_alice)

    result = subtensor.transfer(wallet=wallet_alice, dest=coldkey_bob.ss58_address, amount=Balance(10_000), wait_for_finalization=True, timeout=30)
    assert result is True

    balance_alice_post = subtensor.get_balance(coldkey_alice.ss58_address)
    balance_bob_post = subtensor.get_balance(coldkey_bob.ss58_address)

    assert int(balance_alice_post) == int(balance_alice_pre) - (10000 + TRANSACTION_FEE_TRANSFER)
    assert int(balance_bob_post) == int(balance_bob_pre) + 10000





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
    subtensor = connect(setup_chain)
    subtensor.is_connected()
    result = subtensor.get_last_emit_data_for_uid( 999999 )
    assert result is None

def test_get_neurons(setup_chain):
    walletA = generate_wallet()
    walletB = generate_wallet()

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
    walletA = generate_wallet()
    walletB = generate_wallet()

    subtensorA = connect(setup_chain)
    subtensorB = connect(setup_chain)
    subtensorA.is_connected()
    subtensorB.is_connected()

    subscribe(subtensorA, walletA)
    subscribe(subtensorB, walletB)

    uidA = subtensorA.get_uid_for_pubkey(walletA.hotkey.public_key)
    uidB = subtensorB.get_uid_for_pubkey(walletB.hotkey.public_key)

    w_uids = [uidA, uidB]
    w_vals = [pow(2, 31) - 1, pow(2, 31) - 1]
    subtensorA.set_weights(
        destinations=w_uids,
        values=w_vals,
        wait_for_finalization=True,
        timeout=4 * bittensor.__blocktime__,
        wallet=walletA
    )
    subtensorB.set_weights(
        destinations=w_uids,
        values=w_vals,
        wait_for_finalization=True,
        timeout=4 * bittensor.__blocktime__,
        wallet=walletB
    )

    result_uids = subtensorA.weight_uids_for_uid(uidA)
    result_vals = subtensorA.weight_vals_for_uid(uidA)
    assert result_uids == w_uids
    assert result_vals == w_vals


def test_set_weights_success_transaction_fee(setup_chain):
    coldkeyA = Keypair.create_from_uri('//Alice')
    coldkeyB = Keypair.create_from_uri('//Bob')

    walletA = generate_wallet(coldkey_pair=coldkeyA)
    walletB = generate_wallet(coldkey_pair=coldkeyB)

    subtensorA = connect(setup_chain)
    subtensorB = connect(setup_chain)
    subtensorA.is_connected()
    subtensorB.is_connected()

    subscribe(subtensorA, walletA)  # Sets a self weight of 1
    subscribe(subtensorB, walletB)  # Sets a self weight of 1

    uidA = subtensorA.get_uid_for_pubkey(walletA.hotkey.public_key)
    uidB = subtensorB.get_uid_for_pubkey(walletB.hotkey.public_key)

    stake = 4000
    transaction_fee = 14355

    # Add stake to the hotkey account, so we can do tests on the transaction fees of the set_weights function
    # Keep in mind, this operation incurs transaction fees that are appended to the block_reward
    subtensorA.add_stake(wallet=walletA, amount=Balance(stake), hotkey_id=walletA.hotkey.public_key,
                         wait_for_finalization=True, timeout=30)


    # At this point both neurons have equal stake, with self-weight set, so they receive each 50% of the block reward

    blocknr_pre = subtensorA.get_current_block()

    w_uids = [uidA, uidB]
    w_vals = [0, 1]
    subtensorA.set_weights(
        destinations=w_uids,
        values=w_vals,
        wait_for_finalization=True,
        timeout=4 * bittensor.__blocktime__,
        wallet=walletA
    )

    blocknr_post = subtensorA.get_current_block()
    blocks_passed = blocknr_post - blocknr_pre

    logger.error(blocks_passed)

    # Check the stakes
    stakeA = subtensorA.get_stake_for_uid(uidA)

    expectation = int(stake + (0.99 * BLOCK_REWARD * 3) + 0.99 * TRANSACTION_FEE_ADD_STAKE)

    assert int(stakeA) == expectation  # 1_485_018_355
