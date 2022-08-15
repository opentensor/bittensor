import binascii
import hashlib
import bittensor
import sys
import subprocess
import time
import pytest
import os 
import random
import torch
import multiprocessing

from sys import platform   
from substrateinterface.base import Keypair
from _pytest.fixtures import fixture
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
    
    # Delete existing wallets
    #subprocess.Popen(["rm", '-r', '~/.bittensor/wallets/*testwallet'], close_fds=True, shell=False)

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
    return subtensor, port

def construct_config():
    defaults = bittensor.Config()
    bittensor.subtensor.add_defaults( defaults )
    bittensor.dendrite.add_defaults( defaults )
    bittensor.axon.add_defaults( defaults )
    bittensor.wallet.add_defaults( defaults )
    bittensor.dataset.add_defaults( defaults )
    
    return defaults

def test_unbiased_topk():
    input_tensor = torch.FloatTensor([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    topk = bittensor.utils.unbiased_topk(input_tensor, 2)
    assert torch.all(torch.eq(topk[0], torch.Tensor([10., 9.])))
    assert torch.all(torch.eq(topk[1], torch.Tensor([9, 8])))

def test_hex_bytes_to_u8_list():
    nonce = 1
    nonce_bytes = binascii.hexlify(nonce.to_bytes(8, 'little'))
    hex_bytes_list = bittensor.utils.hex_bytes_to_u8_list(nonce_bytes)

    assert len(hex_bytes_list) == 8
    assert hex_bytes_list[0] == 1
    assert hex_bytes_list[-1] == 0

def test_u8_list_to_hex():
    hex_bytes_list = [1, 0, 0, 0, 0, 0, 0, 0]
    assert bittensor.utils.u8_list_to_hex(hex_bytes_list) == 1

def test_create_seal_hash():
   block_hash = '0xba7ea4eb0b16dee271dbef5911838c3f359fcf598c74da65a54b919b68b67279'
   nonce = 10
   seal_hash = bittensor.utils.create_seal_hash(block_hash, nonce)
   assert seal_hash == b'\xc5\x01B6"\xa8\xa5FDPK\xe49\xad\xdat\xbb:\x87d\x13/\x86\xc6:I8\x9b\x88\xf0\xc20'

def test_seal_meets_difficulty():
    block_hash = '0xba7ea4eb0b16dee271dbef5911838c3f359fcf598c74da65a54b919b68b67279'
    nonce = 10
    nonce_bytes = binascii.hexlify(nonce.to_bytes(8, 'little'))
    block_bytes = block_hash.encode('utf-8')[2:]
    pre_seal = nonce_bytes + block_bytes
    seal = hashlib.sha256( bytearray(bittensor.utils.hex_bytes_to_u8_list(pre_seal)) ).digest()

    difficulty = 1
    meets = bittensor.utils.seal_meets_difficulty( seal, difficulty )
    assert meets == True

    difficulty = 10
    meets = bittensor.utils.seal_meets_difficulty( seal, difficulty )
    assert meets == False

def test_solve_for_difficulty():
    block_hash = '0xba7ea4eb0b16dee271dbef5911838c3f359fcf598c74da65a54b919b68b67279'
    nonce, seal = bittensor.utils.solve_for_difficulty(block_hash, 1)

    assert nonce == 0
    assert seal == b'\xe2d\xbc\x10Tu|\xd0nQ\x1f\x15wTd\xb0\x18\x8f\xc7\xe7:\x12\xc6>\\\xbe\xac\xc5/v\xa7\xce'

    nonce, seal = bittensor.utils.solve_for_difficulty(block_hash, 10)
    assert nonce == 2
    assert seal == b'\x19\xf2H1mB3\xa3y\xda\xe7)\xc7P\x93t\xe5o\xbc$\x14sQ\x10\xc3M\xc6\x90M8vq'

def test_solve_for_difficulty_fast():
    block_hash = '0xba7ea4eb0b16dee271dbef5911838c3f359fcf598c74da65a54b919b68b67279'
    subtensor = MagicMock()
    subtensor.get_current_block = MagicMock( return_value=1 )
    subtensor.difficulty = 1
    subtensor.substrate = MagicMock()
    subtensor.substrate.get_block_hash = MagicMock( return_value=block_hash )
    wallet = MagicMock()
    wallet.is_registered = MagicMock( return_value=False )
    num_proc: int = 1

    solution = bittensor.utils.solve_for_difficulty_fast( subtensor, wallet, num_processes=num_proc )   
    seal = solution.seal

    assert bittensor.utils.seal_meets_difficulty(seal, 1)
    
    subtensor.difficulty = 10
    solution = bittensor.utils.solve_for_difficulty_fast( subtensor, wallet, num_processes=num_proc )
    seal = solution.seal
    assert bittensor.utils.seal_meets_difficulty(seal, 10)
    
def test_solve_for_difficulty_fast_registered_already():
    # tests if the registration stops after the first block of nonces
    for _ in range(10):
        workblocks_before_is_registered = random.randint(1, 4)
        # return False each work block but return True after a random number of blocks
        is_registered_return_values = [False for _ in range(workblocks_before_is_registered)] + [True] + [False, False]

        block_hash = '0xba7ea4eb0b16dee271dbef5911838c3f359fcf598c74da65a54b919b68b67279'
        subtensor = MagicMock()
        subtensor.get_current_block = MagicMock( return_value=1 )
        subtensor.difficulty = 100000000000# set high to make solving take a long time
        subtensor.substrate = MagicMock()
        subtensor.substrate.get_block_hash = MagicMock( return_value=block_hash )
        wallet = MagicMock()
        wallet.is_registered = MagicMock( side_effect=is_registered_return_values )

        # all arugments should return None to indicate an early return
        solution = bittensor.utils.solve_for_difficulty_fast( subtensor, wallet, num_processes = 1, update_interval = 1000)

        assert solution is None
        # called every time until True
        assert wallet.is_registered.call_count == workblocks_before_is_registered + 1

def test_solve_for_difficulty_fast_missing_hash():
    block_hash = '0xba7ea4eb0b16dee271dbef5911838c3f359fcf598c74da65a54b919b68b67279'
    subtensor = MagicMock()
    subtensor.get_current_block = MagicMock( return_value=1 )
    subtensor.difficulty = 1
    subtensor.substrate = MagicMock()
    subtensor.substrate.get_block_hash = MagicMock( side_effect= [None, None] + [block_hash]*20)
    wallet = MagicMock()
    wallet.is_registered = MagicMock( return_value=False )
    num_proc: int = 1

    solution = bittensor.utils.solve_for_difficulty_fast( subtensor, wallet, num_processes=num_proc )
    seal = solution.seal
    assert bittensor.utils.seal_meets_difficulty(seal, 1)
    
    subtensor.difficulty = 10
    solution = bittensor.utils.solve_for_difficulty_fast( subtensor, wallet, num_processes=num_proc )
    seal = solution.seal
    assert bittensor.utils.seal_meets_difficulty(seal, 10)

def test_is_valid_ss58_address():
    keypair = bittensor.Keypair.create_from_mnemonic(
        bittensor.Keypair.generate_mnemonic(
            words=12
        ), ss58_format=bittensor.__ss58_format__
    )
    good_address = keypair.ss58_address
    bad_address = good_address[:-1] + 'a'
    assert bittensor.utils.is_valid_ss58_address(good_address)
    assert not bittensor.utils.is_valid_ss58_address(bad_address)

def test_is_valid_ed25519_pubkey():
    keypair = bittensor.Keypair.create_from_mnemonic(
        bittensor.Keypair.generate_mnemonic(
            words=12
        ), ss58_format=bittensor.__ss58_format__
    )
    good_pubkey = keypair.public_key.hex()
    bad_pubkey = good_pubkey[:-1] # needs to be 64 chars
    assert bittensor.utils.is_valid_ed25519_pubkey(good_pubkey)
    assert not bittensor.utils.is_valid_ed25519_pubkey(bad_pubkey)

    # Test with bytes
    good_pubkey = keypair.public_key
    bad_pubkey = good_pubkey[:-1] # needs to be 32 bytes
    assert bittensor.utils.is_valid_ed25519_pubkey(good_pubkey)
    assert not bittensor.utils.is_valid_ed25519_pubkey(bad_pubkey)

def test_registration_diff_pack_unpack():
        fake_diff = pow(2, 31)# this is under 32 bits
        
        mock_diff = multiprocessing.Array('Q', [0, 0], lock=True) # [high, low]
        
        bittensor.utils.registration_diff_pack(fake_diff, mock_diff)
        assert bittensor.utils.registration_diff_unpack(mock_diff) == fake_diff

        fake_diff = pow(2, 32) * pow(2, 4) # this should be too large if the bit shift is wrong (32 + 4 bits)
        
        bittensor.utils.registration_diff_pack(fake_diff, mock_diff)
        assert bittensor.utils.registration_diff_unpack(mock_diff) == fake_diff

if __name__ == "__main__":
    test_solve_for_difficulty_fast_registered_already()