import os
import sys
import time
from typing import List
from unittest.mock import MagicMock
import pytest
import random
import subprocess
import torch

from torch import Tensor
from loguru import logger
from pytest import fixture
from sys import platform   

import bittensor
from bittensor.utils.balance import Balance
import bittensor.utils.weight_utils as weight_utils

from substrateinterface.utils.ss58 import ss58_decode
from substrateinterface import Keypair

BLOCK_REWARD = 500_000_000
PARTIAL_FEE = 125_000_146

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
    
    # Purge chain first
    subprocess.Popen([path, 'purge-chain', '--dev', '-y'], close_fds=True, shell=False)
    
    proc = subprocess.Popen([path, '--dev', '--port', str(port+1), '--ws-port', str(port), '--rpc-port', str(port + 2), '--tmp'], close_fds=True, shell=False)

    # Wait 4 seconds for the node to come up
    time.sleep(4)

    yield port

    # Wait 4 seconds for the node to come up
    time.sleep(4)

    # Kill process
    os.system("kill %i" % proc.pid)

class WalletStub( bittensor.Wallet ):
        def __init__(self, coldkey_pair: 'Keypair', hotkey_pair: 'Keypair'):
            self._hotkey = hotkey_pair
            self._coldkey = coldkey_pair
            self._coldkeypub = "0x" + coldkey_pair.public_key.hex()

@pytest.fixture(scope="session", autouse=True)
def initialize_tests():
    # Kill any running process before running tests
    os.system("pkill node-subtensor")

def select_port():
    port = random.randrange(1000, 65536, 5)
    return port


def generate_wallet(coldkey : 'Keypair' = None, hotkey: 'Keypair' = None):
    wallet = bittensor.wallet(_mock=True)   

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
    return subtensor

def add_stake( subtensor, wallet: 'bittensor.Wallet', amount: 'int' ):
    # Get the uid of the new neuron
    uid = subtensor.get_uid_for_hotkey( wallet.hotkey.ss58_address )
    assert uid is not None

    # Add stake to new neuron
    result = subtensor.add_stake( wallet = wallet, amount = amount, wait_for_finalization=False )
    assert result == True

def unstake( subtensor, wallet: 'bittensor.Wallet', amount: 'int' ):
    # Get the uid of the new neuron
    uid = subtensor.get_uid_for_hotkey( wallet.hotkey.ss58_address )
    assert uid is not None

    # Add stake to new neuron
    result = subtensor.unstake( wallet = wallet, amount = amount, wait_for_finalization=False )
    assert result == True


'''
get_balance() tests
'''

def test_get_balance_no_balance(setup_chain):
    wallet = bittensor.wallet()
    subtensor = setup_subtensor(setup_chain)
    subtensor.register = MagicMock(return_value = True)  
    
    subtensor.register(wallet=wallet)    
    result = subtensor.get_balance(wallet.hotkey.ss58_address)
    assert result == Balance(0)

def test_get_balance_success(setup_chain):
    wallet = bittensor.wallet()
    hotkey_pair = Keypair.create_from_uri('//Bob')
    subtensor = setup_subtensor(setup_chain)
    #subtensor.register = MagicMock(return_value = True)  
    
    subtensor.register(wallet=wallet)    
    result = subtensor.get_balance(hotkey_pair.ss58_address)
    # Here 1040631877269 is the default account balance for Alice on substrate
    assert result == Balance(0)

'''
add_stake() tests
'''

def test_add_stake_success(setup_chain):
    coldkey = Keypair.create_from_uri("//Alice")
    hotkey = Keypair.create_from_uri('//Alice')

    wallet = generate_wallet(coldkey=coldkey, hotkey=hotkey)
    subtensor = setup_subtensor(setup_chain)
    
    # Register the hotkey using Alice's cold key
    subtensor.register(wallet=wallet)   
    uid = subtensor.get_uid_for_hotkey(hotkey.ss58_address)
    assert uid is not None

    # Verify the node has 0 stake
    result = wallet.get_stake(subtensor=subtensor)
    assert result == Balance(0)

    # Get balance. This should be default account balance value of Alice (1152921504606846976)
    balance_pre = subtensor.get_balance(coldkey.ss58_address)
    add_stake(subtensor, wallet, 100)

    # # Check if the amount of stake ends up in the hotkey account
    result = wallet.get_stake(subtensor=subtensor)
    assert result == Balance.from_tao(100)

    # Check if the balances had reduced by the amount of stake 
    balance_post = subtensor.get_balance(coldkey.ss58_address)

    assert balance_post == Balance(int(balance_pre) - PARTIAL_FEE - int(Balance.from_tao(100)))

'''
unstake() tests
'''

def test_unstake_success(setup_chain):
    coldkey = Keypair.create_from_uri('//Alice')
    hotkey = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    
    wallet = generate_wallet(coldkey=coldkey, hotkey=hotkey)
    subtensor = setup_subtensor(setup_chain)
        
    # Register the hotkey using Alice's cold key
    subtensor.register(wallet=wallet)   

    uid = subtensor.get_uid_for_hotkey(hotkey.ss58_address)
    assert uid is not None
    
    add_stake(subtensor, wallet, 100)

    # unstake incurs a transaction fee that is added to the block reward
    unstake(subtensor, wallet, 100)

    stake_post = wallet.get_stake(subtensor=subtensor)

    assert int(stake_post) == 0


'''
transfer() tests
'''

def test_transfer_success(setup_chain):
    coldkey_alice = Keypair.create_from_uri("//Alice")
    coldkey_bob = Keypair.create_from_uri("//Bob")

    subtensor = setup_subtensor(setup_chain)

    balance_alice_pre = subtensor.get_balance(coldkey_alice.ss58_address)
    balance_bob_pre = subtensor.get_balance(coldkey_bob.ss58_address)

    wallet_alice = generate_wallet(coldkey=coldkey_alice)
    result = subtensor.transfer(wallet=wallet_alice, dest=coldkey_bob.ss58_address, amount=100, wait_for_finalization=True)
    assert result == True

    balance_alice_post = subtensor.get_balance(coldkey_alice.ss58_address)
    balance_bob_post = subtensor.get_balance(coldkey_bob.ss58_address)

    
    # Bob should have its original balance + 100 that we transferred
    assert int(balance_bob_post) == int(balance_bob_pre) +  int(Balance.from_tao(100))



def test_get_uid_for_hotkey_success(setup_chain):
    wallet = generate_wallet()
    subtensor = setup_subtensor(setup_chain)
    # Register the hotkey using Alice's cold key
    subtensor.register(wallet=wallet)   
    uid = subtensor.get_uid_for_hotkey(wallet.hotkey.ss58_address)
    assert uid is not None

def test_get_current_block(setup_chain):
    subtensor = setup_subtensor(setup_chain)
    block = subtensor.get_current_block()
    assert block >= 0

def test_get_active(setup_chain):
    wallet = generate_wallet()
    subtensor = setup_subtensor(setup_chain)
    metagraph = bittensor.metagraph(subtensor=subtensor)

    subtensor.register(wallet=wallet)  
    metagraph.sync()
    result = metagraph.active

    assert isinstance(result, Tensor)
    assert len(result) > 0
    result = metagraph.hotkeys
    elem = result[0]
    elem = "0x" + ss58_decode(elem)
    assert isinstance(elem[0], str)
    assert elem[:2] == "0x"
    assert len(elem[2:]) == 64


def test_get_last_emit_data_for_uid__success(setup_chain):
    wallet = generate_wallet()
    subtensor = setup_subtensor(setup_chain)
    metagraph = bittensor.metagraph(subtensor=subtensor)    
    subtensor.register(wallet=wallet)      
    uid = subtensor.get_uid_for_hotkey(wallet.hotkey.ss58_address)
    metagraph.sync()
    result = metagraph.last_update[uid]
    current_block = subtensor.get_current_block()
    assert result.item() <= current_block


def test_get_neurons(setup_chain):
    wallet_a = generate_wallet()
    subtensor = setup_subtensor(setup_chain)
    subtensor.register(wallet=wallet_a)   

    wallet_b = generate_wallet()
    subtensor.register(wallet=wallet_b)   

    result = subtensor.neurons()
    assert isinstance(result, List)
    assert len(result) >= 2
    elem = result[0]
    assert isinstance(elem.uid, int) # This is the uid, which is the first element of the list

    assert hasattr(elem, 'coldkey')
    assert hasattr(elem, 'hotkey')
    assert hasattr(elem, 'ip_type')
    assert hasattr(elem, 'modality')
    assert hasattr(elem, 'port')
    assert hasattr(elem, 'uid')

    elem = result[1]

    assert hasattr(elem, 'coldkey')
    assert hasattr(elem, 'hotkey')
    assert hasattr(elem, 'ip_type')
    assert hasattr(elem, 'modality')
    assert hasattr(elem, 'port')
    assert hasattr(elem, 'uid')


def test_set_weights_success(setup_chain):
    subtensor = setup_subtensor(setup_chain)

    wallet_a = generate_wallet()
    wallet_b = generate_wallet()
    subtensor.register(wallet=wallet_a)  
    subtensor.register(wallet=wallet_b)  


    uid_a = subtensor.get_uid_for_hotkey(wallet_a.hotkey.ss58_address)
    uid_b = subtensor.get_uid_for_hotkey(wallet_b.hotkey.ss58_address)

    w_uids = torch.tensor([uid_a, uid_b])
    w_vals = torch.tensor([pow(2, 31) - 1, pow(2, 31)])
    
    subtensor.set_weights(
        uids = w_uids,
        weights = w_vals,
        wait_for_finalization=True,
        wallet=wallet_a
    )
    
    subtensor.set_weights(
        uids = w_uids,
        weights = w_vals,
        wait_for_finalization=True,
        wallet=wallet_b
    )

    result_uids, result_vals = weight_utils.convert_weights_and_uids_for_emit( w_uids, w_vals )
    assert result_uids == w_uids.tolist()
    assert result_vals == w_vals.tolist()

def test_solve_for_difficulty_fast( setup_chain ):
    subtensor, port = setup_subtensor(setup_chain)
    nonce, block_number, block_hash, difficulty, seal = bittensor.utils.solve_for_difficulty_fast(subtensor)

    assert difficulty == 10000

def test_indexed_values_to_dataframe( setup_chain ):
    subtensor, port = setup_subtensor(setup_chain)
    wallet = generate_wallet()

    wallet.register(subtensor=subtensor)
    nn = subtensor.neuron_for_pubkey(wallet.hotkey.ss58_address)
    metagraph = bittensor.metagraph(subtensor=subtensor)
    metagraph.sync()
    idx_df = bittensor.utils.indexed_values_to_dataframe( prefix = 'w_i_{}'.format(nn.uid), index = [nn.uid], values = metagraph.W[:, nn.uid] )
    assert idx_df.values[0][0] == 1.0

    idx_df = bittensor.utils.indexed_values_to_dataframe( prefix = nn.uid, index = [nn.uid], values = metagraph.W[:, nn.uid] )
    assert idx_df.values[0][0] == 1.0

    idx_df = bittensor.utils.indexed_values_to_dataframe( prefix = nn.uid, index = torch.LongTensor([nn.uid]), values = metagraph.W[:, nn.uid] )
    assert idx_df.values[0][0] == 1.0

    # Need to check for errors 
    with pytest.raises(ValueError):
        idx_df = bittensor.utils.indexed_values_to_dataframe( prefix = b'w_i', index = [nn.uid], values = metagraph.W[:, nn.uid] )
    with pytest.raises(ValueError):
        idx_df = bittensor.utils.indexed_values_to_dataframe( prefix = 'w_i_{}'.format(nn.uid), index = nn.uid, values = metagraph.W[:, nn.uid] )
    with pytest.raises(ValueError):
        idx_df = bittensor.utils.indexed_values_to_dataframe( prefix = 'w_i_{}'.format(nn.uid), index = [nn.uid], values = nn.uid)
