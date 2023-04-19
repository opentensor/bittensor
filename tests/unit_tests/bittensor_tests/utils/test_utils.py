import binascii
import hashlib
import math
import multiprocessing
import os
import random
import subprocess
import sys
import time
import unittest
from sys import platform
from types import SimpleNamespace
from typing import Dict, Union
from unittest.mock import MagicMock, patch

import pytest
from _pytest.fixtures import fixture

from ddt import data, ddt, unpack

import torch
from loguru import logger
from substrateinterface.base import Keypair

import bittensor
from bittensor.utils.registration import _CUDASolver, _SolverBase


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

class TestRegistrationHelpers(unittest.TestCase):
    def test_create_seal_hash(self):
        block_and_hotkey_hash = '0xba7ea4eb0b16dee271dbef5911838c3f359fcf598c74da65a54b919b68b67279'
        block_and_hotkey_hash_bytes = bytes.fromhex(block_and_hotkey_hash[2:])
        nonce = 10
        seal_hash = bittensor.utils.registration._create_seal_hash(block_and_hotkey_hash_bytes, nonce)
        self.assertEqual(seal_hash, b'\xc5\x01B6"\xa8\xa5FDPK\xe49\xad\xdat\xbb:\x87d\x13/\x86\xc6:I8\x9b\x88\xf0\xc20')

    def test_seal_meets_difficulty(self):
        block_hash = '0xba7ea4eb0b16dee271dbef5911838c3f359fcf598c74da65a54b919b68b67279'
        nonce = 10
        limit = int(math.pow(2,256))- 1  
        nonce_bytes = nonce.to_bytes(8, 'little')
        block_bytes = block_hash.encode('utf-8')[2:]
        pre_seal = nonce_bytes + block_bytes
        seal = hashlib.sha256( bytearray(pre_seal) ).digest()

        difficulty = 1
        meets = bittensor.utils.registration._seal_meets_difficulty( seal, difficulty, limit )
        assert meets == True

        difficulty = 10
        meets = bittensor.utils.registration._seal_meets_difficulty( seal, difficulty, limit )
        assert meets == False

    def test_solve_for_difficulty_fast(self):
        block_hash = '0xba7ea4eb0b16dee271dbef5911838c3f359fcf598c74da65a54b919b68b67279'
        subtensor = MagicMock()
        subtensor.get_current_block = MagicMock( return_value=1 )
        subtensor.difficulty = MagicMock( return_value=1 )
        subtensor.substrate = MagicMock()
        subtensor.substrate.get_block_hash = MagicMock( return_value=block_hash )
        wallet = MagicMock(
            hotkey = Keypair.create_from_mnemonic(Keypair.generate_mnemonic()),
            is_registered = MagicMock( return_value=False )
        )
        num_proc: int = 1
        limit = int(math.pow(2,256))- 1  

        solution = bittensor.utils.registration._solve_for_difficulty_fast( subtensor, wallet, netuid = -1, num_processes=num_proc )   
        seal = solution.seal

        assert bittensor.utils.registration._seal_meets_difficulty(seal, 1, limit)
        
        subtensor.difficulty = MagicMock( return_value=10 )
        solution = bittensor.utils.registration._solve_for_difficulty_fast( subtensor, wallet, netuid = -1, num_processes=num_proc )
        seal = solution.seal
        assert bittensor.utils.registration._seal_meets_difficulty(seal, 10, limit)
    
    def test_solve_for_difficulty_fast_registered_already(self):
        # tests if the registration stops after the first block of nonces
        for _ in range(10):
            workblocks_before_is_registered = random.randint(1, 4)
            # return False each work block but return True after a random number of blocks
            is_registered_return_values = [False for _ in range(workblocks_before_is_registered)] + [True] + [False, False]

            block_hash = '0xba7ea4eb0b16dee271dbef5911838c3f359fcf598c74da65a54b919b68b67279'
            subtensor = MagicMock()
            subtensor.get_current_block = MagicMock( return_value=1 )
            subtensor.difficulty = MagicMock( return_value=int(1e10)) # set high to make solving take a long time
            subtensor.substrate = MagicMock()
            subtensor.substrate.get_block_hash = MagicMock( return_value=block_hash )
            wallet = MagicMock(
                hotkey = Keypair.create_from_mnemonic(Keypair.generate_mnemonic()),
                is_registered = MagicMock( side_effect=is_registered_return_values )
            )

            # all arugments should return None to indicate an early return
            solution = bittensor.utils.registration._solve_for_difficulty_fast( subtensor, wallet, netuid = -1, num_processes = 1, update_interval = 1000)

            assert solution is None
            # called every time until True
            assert wallet.is_registered.call_count == workblocks_before_is_registered + 1

    def test_solve_for_difficulty_fast_missing_hash(self):
        block_hash = '0xba7ea4eb0b16dee271dbef5911838c3f359fcf598c74da65a54b919b68b67279'
        subtensor = MagicMock()
        subtensor.get_current_block = MagicMock( return_value=1 )
        subtensor.difficulty = MagicMock( return_value=1 )
        subtensor.substrate = MagicMock()
        subtensor.substrate.get_block_hash = MagicMock( side_effect= [None, None] + [block_hash]*20)
        wallet = MagicMock(
            hotkey = Keypair.create_from_mnemonic(Keypair.generate_mnemonic()),
            is_registered = MagicMock( return_value=False )
        )
        num_proc: int = 1
        limit = int(math.pow(2,256))- 1  

        solution = bittensor.utils.registration._solve_for_difficulty_fast( subtensor, wallet, netuid = -1, num_processes=num_proc )
        seal = solution.seal
        assert bittensor.utils.registration._seal_meets_difficulty(seal, 1, limit)
        
        subtensor.difficulty = MagicMock( return_value=10 )
        solution = bittensor.utils.registration._solve_for_difficulty_fast( subtensor, wallet, netuid = -1, num_processes=num_proc )
        seal = solution.seal
        assert bittensor.utils.registration._seal_meets_difficulty(seal, 10, limit)

    def test_registration_diff_pack_unpack_under_32_bits(self):
        fake_diff = pow(2, 31)# this is under 32 bits
        
        mock_diff = multiprocessing.Array('Q', [0, 0], lock=True) # [high, low]
        
        bittensor.utils.registration._registration_diff_pack(fake_diff, mock_diff)
        assert bittensor.utils.registration._registration_diff_unpack(mock_diff) == fake_diff

    def test_registration_diff_pack_unpack_over_32_bits(self):
        mock_diff = multiprocessing.Array('Q', [0, 0], lock=True) # [high, low]
        fake_diff = pow(2, 32) * pow(2, 4) # this should be too large if the bit shift is wrong (32 + 4 bits)
        
        bittensor.utils.registration._registration_diff_pack(fake_diff, mock_diff)
        assert bittensor.utils.registration._registration_diff_unpack(mock_diff) == fake_diff

    def test_hash_block_with_hotkey(self):
        block_hash = "0xc444e4205857add79a0427401aa2518d11e85f32377eff9a946d180a54697459"
        block_hash_bytes = bytes.fromhex(block_hash[2:])
        
        hotkey_pubkey_hex = "0xba3189e99e75b6097cd94a5ecc771016b83c8432d35d14a03ab731b07112f559"
        hotkey_bytes = bytes.fromhex(hotkey_pubkey_hex[2:])
        
        expected_hash_hex = '0x7869b61229641b33a355dc34d4ef48f8d82166635237f9f10bcb215b8cb48161'
        expected_hash = bytes.fromhex(expected_hash_hex[2:])

        result_hash = bittensor.utils.registration._hash_block_with_hotkey(block_hash_bytes, hotkey_bytes)
        self.assertEqual(result_hash, expected_hash)

    def test_update_curr_block(self):
        curr_block, curr_block_num, curr_diff = _SolverBase.create_shared_memory()

        block_number: int = 1
        block_bytes = bytes.fromhex('9dda24e4199df410e18a43044b3069078f796922b0247b8749aecb577b09bd59')
        diff: int = 1
        hotkey_bytes = bytes.fromhex('0'*64)
        lock: Union[multiprocessing.Lock, MagicMock] = MagicMock()

        bittensor.utils.registration._update_curr_block(curr_diff, curr_block, curr_block_num, block_number, block_bytes, diff, hotkey_bytes, lock)

        self.assertEqual(curr_block_num.value, block_number)
        
        self.assertEqual(curr_diff[0], diff >> 32)
        self.assertEqual(curr_diff[1], diff & 0xFFFFFFFF)

        hash_of_block_and_hotkey = bittensor.utils.registration._hash_block_with_hotkey(block_bytes, hotkey_bytes)
        self.assertEqual(curr_block[:], [int(byte_) for byte_ in hash_of_block_and_hotkey])

    def test_solve_for_nonce_block(self):
        nonce_start = 0
        nonce_end = 10_000
        block_and_hotkey_hash_bytes = bytes.fromhex('9dda24e4199df410e18a43044b3069078f796922b0247b8749aecb577b09bd59')
        
        limit = limit = int(math.pow(2,256)) - 1  
        block_number = 1

        difficulty = 1
        result = bittensor.utils.registration._solve_for_nonce_block(nonce_start, nonce_end, block_and_hotkey_hash_bytes, difficulty, limit, block_number)

        self.assertIsNotNone(result)
        self.assertEqual(result.block_number, block_number)
        self.assertEqual(result.difficulty, difficulty)

        # Make sure seal meets difficulty
        self.assertTrue(bittensor.utils.registration._seal_meets_difficulty(result.seal, difficulty, limit))

        # Test with a higher difficulty
        difficulty = 10
        result = bittensor.utils.registration._solve_for_nonce_block(nonce_start, nonce_end, block_and_hotkey_hash_bytes, difficulty, limit, block_number)

        self.assertIsNotNone(result)
        self.assertEqual(result.block_number, block_number)
        self.assertEqual(result.difficulty, difficulty)

        # Make sure seal meets difficulty
        self.assertTrue(bittensor.utils.registration._seal_meets_difficulty(result.seal, difficulty, limit))

class TestSS58Utils(unittest.TestCase):
    def test_is_valid_ss58_address(self):
        keypair = bittensor.Keypair.create_from_mnemonic(
            bittensor.Keypair.generate_mnemonic(
                words=12
            ), ss58_format=bittensor.__ss58_format__
        )
        good_address = keypair.ss58_address
        bad_address = good_address[:-1] + 'a'
        assert bittensor.utils.is_valid_ss58_address(good_address)
        assert not bittensor.utils.is_valid_ss58_address(bad_address)

    def test_is_valid_ss58_address_legacy(self):
        keypair = bittensor.Keypair.create_from_mnemonic(
            bittensor.Keypair.generate_mnemonic(
                words=12
            ), ss58_format=42 # should be fine for legacy ss58
        )
        good_address = keypair.ss58_address
        bad_address = good_address[:-1] + 'a'
        assert bittensor.utils.is_valid_ss58_address(good_address)
        assert not bittensor.utils.is_valid_ss58_address(bad_address)

    def test_is_valid_ed25519_pubkey(self):
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


class TestUpdateCurrentBlockDuringRegistration(unittest.TestCase):
    def test_check_for_newest_block_and_update_same_block(self):
        # if the block is the same, the function should return the same block number
        subtensor = MagicMock()
        current_block_num: int = 1
        subtensor.get_current_block = MagicMock( return_value=current_block_num )
        mock_hotkey_bytes = bytes.fromhex('0'*63 + '1')

        self.assertEqual(bittensor.utils.registration._check_for_newest_block_and_update(
            subtensor,
            -1, # netuid
            current_block_num, # current block number is the same as the new block number
            mock_hotkey_bytes, # mock hotkey bytes
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        ), current_block_num)

    def test_check_for_newest_block_and_update_new_block(self):
        # if the block is new, the function should return the new block_number
        mock_block_hash = '0xba7ea4eb0b16dee271dbef5911838c3f359fcf598c74da65a54b919b68b67279'
        mock_hotkey_bytes = bytes.fromhex('0'*63 + '1')

        current_block_num: int = 1
        current_diff: int = 0

        mock_substrate = MagicMock(
            get_block_hash=MagicMock(
                return_value=mock_block_hash
            ),

        )
        subtensor = MagicMock(
            substrate=mock_substrate,
            difficulty=MagicMock(return_value=current_diff + 1), # new diff
        )
        subtensor.get_current_block = MagicMock( return_value=current_block_num + 1 ) # new block

        mock_update_curr_block = MagicMock()

        mock_solvers = [
            MagicMock(
                newBlockEvent=MagicMock(
                    set=MagicMock()
                )
        ), 
        MagicMock(
            newBlockEvent=MagicMock(
                set=MagicMock()
            )
        )]

        mock_curr_stats = MagicMock(
            block_number=current_block_num,
            block_hash=b'',
            difficulty=0,
        )

        self.assertEqual(bittensor.utils.registration._check_for_newest_block_and_update(
            subtensor,
            -1, # netuid
            MagicMock(),
            mock_hotkey_bytes,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            mock_update_curr_block,
            MagicMock(),
            mock_solvers,
            mock_curr_stats,
        ), current_block_num + 1)      

        # check that the update_curr_block function was called
        mock_update_curr_block.assert_called_once()

        # check that the solvers got the event 
        for solver in mock_solvers:
            solver.newBlockEvent.set.assert_called_once()

        # check the stats were updated
        self.assertEqual(mock_curr_stats.block_number, current_block_num + 1)
        self.assertEqual(mock_curr_stats.block_hash, mock_block_hash)
        self.assertEqual(mock_curr_stats.difficulty, current_diff + 1)

class TestGetBlockWithRetry(unittest.TestCase):
    class MockException(Exception):
        pass

    def test_get_block_with_retry_network_error_exit(self):
        mock_subtensor = MagicMock(
            get_current_block=MagicMock(return_value=1),
            difficulty=MagicMock(return_value=1),
            substrate=MagicMock(
                get_block_hash=MagicMock(side_effect=self.MockException('network error'))
            )
        )
        with pytest.raises(self.MockException):
            # this should raise an exception because the network error is retried only 3 times
            bittensor.utils.registration._get_block_with_retry(mock_subtensor, -1)

    def test_get_block_with_retry_network_error_no_error(self):
        mock_subtensor = MagicMock(
            get_current_block=MagicMock(return_value=1),
            difficulty=MagicMock(return_value=1),
            substrate=MagicMock(
                get_block_hash=MagicMock(return_value=b'ba7ea4eb0b16dee271dbef5911838c3f359fcf598c74da65a54b919b68b67279')
            )
        )

        # this should not raise an exception because there is no error
        bittensor.utils.registration._get_block_with_retry(mock_subtensor, -1)

    def test_get_block_with_retry_network_error_none_twice(self):
        # Should retry twice then succeed on the third try
        tries = 0
        def block_none_twice(block_hash: bytes):
            nonlocal tries
            if tries == 1:
                return block_hash
            else:
                tries += 1
                return None

        
        mock_subtensor = MagicMock(
            get_current_block=MagicMock(return_value=1),
            difficulty=MagicMock(return_value=1),
            substrate=MagicMock(
                get_block_hash=MagicMock(side_effect=block_none_twice(b'ba7ea4eb0b16dee271dbef5911838c3f359fcf598c74da65a54b919b68b67279'))
            )
        )
        
        # this should not raise an exception because there is no error on the third try
        bittensor.utils.registration._get_block_with_retry(mock_subtensor, -1)
class TestPOWNotStale(unittest.TestCase):
    def test_pow_not_stale_same_block_number(self):
        mock_subtensor = MagicMock(
            get_current_block=MagicMock(return_value=1),
        )
        mock_solution = bittensor.utils.registration.POWSolution(
            block_number= 1, # 3 less than current block number
            nonce= 1,
            difficulty= 1,
            seal= b'',
        )

        assert not mock_solution.is_stale(mock_subtensor)

    def test_pow_not_stale_diff_block_number(self):
        mock_subtensor = MagicMock(
            get_current_block=MagicMock(return_value=2),
        )
        mock_solution = bittensor.utils.registration.POWSolution(
            block_number= 1, # 1 less than current block number
            nonce= 1,
            difficulty= 1,
            seal= b'',
        )

        assert not mock_solution.is_stale(mock_subtensor)

        mock_subtensor = MagicMock(
            get_current_block=MagicMock(return_value=3),
        )
        mock_solution = bittensor.utils.registration.POWSolution(
            block_number= 1, # 2 less than current block number
            nonce= 1,
            difficulty= 1,
            seal= b'',
        )

        assert not mock_solution.is_stale(mock_subtensor)

        mock_subtensor = MagicMock(
            get_current_block=MagicMock(return_value=4),
        )
        mock_solution = bittensor.utils.registration.POWSolution(
            block_number= 1, # 3 less than current block number
            nonce= 1,
            difficulty= 1,
            seal= b'',
        )

        assert not mock_solution.is_stale(mock_subtensor)

    def test_pow_not_stale_diff_block_number_too_old(self):
        mock_subtensor = MagicMock(
            get_current_block=MagicMock(return_value=5),
        )
        mock_solution = bittensor.utils.registration.POWSolution(
            block_number= 1, # 4 less than current block number
            nonce= 1,
            difficulty= 1,
            seal= b'',
        )

        assert mock_solution.is_stale(mock_subtensor)
    
class TestPOWCalled(unittest.TestCase):
    def test_pow_called_for_cuda(self):
        class MockException(Exception):
            pass
        mock_compose_call = MagicMock(side_effect=MockException)

        mock_subtensor = bittensor.subtensor(_mock=True)
        mock_subtensor.get_neuron_for_pubkey_and_subnet=MagicMock(is_null=True)
        mock_subtensor.substrate = MagicMock(
            __enter__= MagicMock(return_value=MagicMock(
                compose_call=mock_compose_call
            )),
            __exit__ = MagicMock(return_value=None),
        )

        mock_wallet = SimpleNamespace(
            hotkey=bittensor.Keypair.create_from_seed(
                '0x' + '0' * 64, ss58_format=bittensor.__ss58_format__
            ),
            coldkeypub=SimpleNamespace(
                ss58_address=''
            )
        )

        mock_pow_is_stale = MagicMock(return_value=False)

        mock_result = MagicMock(
            spec = bittensor.utils.registration.POWSolution,
            block_number=1,
            nonce=random.randint(0, pow(2, 32)),
            difficulty=1,
            seal=b'\x00' * 64,
            is_stale=mock_pow_is_stale,
        )
        

        with patch('torch.cuda.is_available', return_value=True) as mock_cuda_available:
            with patch(
                'bittensor._subtensor.extrinsics.registration.create_pow',
                return_value=mock_result
            ) as mock_create_pow:
                # Should exit early
                with pytest.raises(MockException):
                    mock_subtensor.register(mock_wallet, netuid=-1, cuda=True, prompt=False)

                mock_pow_is_stale.assert_called_once()
                mock_create_pow.assert_called_once()
                mock_cuda_available.assert_called_once()

                call0 = mock_pow_is_stale.call_args
                _, kwargs = call0
                assert kwargs['subtensor'] == mock_subtensor

                mock_compose_call.assert_called_once()
                call1 = mock_compose_call.call_args
                assert call1[1]['call_function'] == 'register'
                call_params = call1[1]['call_params']
                assert call_params['nonce'] == mock_result.nonce

class TestCUDASolverRun(unittest.TestCase):      
    def test_multi_cuda_run_updates_nonce_start(self):
        class MockException(Exception):
            pass

        TPB: int = 512
        update_interval: int = 70_000
        nonce_limit: int = int(math.pow(2, 64)) - 1

        mock_solver_self = MagicMock(
            spec=_CUDASolver,
            TPB=TPB,
            dev_id=0,
            update_interval=update_interval,
            stopEvent=MagicMock(is_set=MagicMock(return_value=False)),
            newBlockEvent=MagicMock(is_set=MagicMock(return_value=False)),
            finished_queue=MagicMock(put=MagicMock()),
            limit=10000,
            proc_num=0,
        )  

        
        with patch('bittensor.utils.registration._solve_for_nonce_block_cuda',
            side_effect=[None, MockException] # first call returns mocked no solution, second call raises exception
        ) as mock_solve_for_nonce_block_cuda: 
        
            # Should exit early
            with pytest.raises(MockException):
                _CUDASolver.run(mock_solver_self)
            
            mock_solve_for_nonce_block_cuda.assert_called()
            calls = mock_solve_for_nonce_block_cuda.call_args_list
            self.assertEqual(len(calls), 2, f"solve_for_nonce_block_cuda was called {len(calls)}. Expected 2") # called only twice
            
            # args, kwargs
            args_call_0, _ = calls[0]
            initial_nonce_start: int = args_call_0[0] # fist arg should be nonce_start
            self.assertIsInstance(initial_nonce_start, int)
            
            args_call_1, _ = calls[1]
            nonce_start_after_iteration: int = args_call_1[0] # first arg should be nonce_start
            self.assertIsInstance(nonce_start_after_iteration, int)

            # verify nonce_start is updated after each iteration
            self.assertNotEqual(nonce_start_after_iteration, initial_nonce_start, "nonce_start was not updated after iteration")
            ## Should incerase by the number of nonces tried == TPB * update_interval
            self.assertEqual(nonce_start_after_iteration, (initial_nonce_start + update_interval * TPB) % nonce_limit,  "nonce_start was not updated by the correct amount")

@ddt
class TestExplorerURL(unittest.TestCase):
    network_map: Dict[str, str] = {
        "nakamoto": "https://polkadot.js.org/apps/?rpc=wss://archivelb.nakamoto.opentensor.ai:9943#/explorer",
        "example": "https://polkadot.js.org/apps/?rpc=wss://example.example.com#/explorer",
        "nobunaga": "https://polkadot.js.org/apps/?rpc=wss://nobunaga.bittensor.com:9943#/explorer",
        # "bad": None # no explorer for this network
    }

    @data(
        ("nobunaga", "https://polkadot.js.org/apps/?rpc=wss://nobunaga.bittensor.com:9943#/explorer"),
        ("nakamoto", "https://polkadot.js.org/apps/?rpc=wss://archivelb.nakamoto.opentensor.ai:9943#/explorer"),
        ("example", "https://polkadot.js.org/apps/?rpc=wss://example.example.com#/explorer"),
        ("bad", None),
        ("", None),
        ("networknamewithoutexplorer", None)
    )
    @unpack
    def test_get_explorer_root_url_by_network_from_map(self, network: str, expected: str) -> str:
        self.assertEqual(bittensor.utils.get_explorer_root_url_by_network_from_map(network, self.network_map), expected)

    @data(
        ("nobunaga", "0x123", "https://polkadot.js.org/apps/?rpc=wss://nobunaga.bittensor.com:9943#/explorer/query/0x123"),
        ("example", "0x123", "https://polkadot.js.org/apps/?rpc=wss://example.example.com#/explorer/query/0x123"),
        ("bad", "0x123", None),
        ("", "0x123", None),
        ("networknamewithoutexplorer", "0x123", None)
    )
    @unpack
    def test_get_explorer_url_for_network_by_network_and_block_hash(self, network: str, block_hash: str, expected: str) -> str:
        self.assertEqual(bittensor.utils.get_explorer_url_for_network(network, block_hash, self.network_map), expected)


if __name__ == "__main__":
    unittest.main()
