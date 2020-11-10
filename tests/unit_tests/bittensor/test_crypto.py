from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from bittensor import bittensor_pb2_grpc as bittensor_grpc
from bittensor import bittensor_pb2
from bittensor.crypto import Crypto

import bittensor
import os
import unittest

class TestCrypto(unittest.TestCase):
    random_synapse = None

    def setUp(self):
        self.random_synapse = self.generate_random_synapse()

    def generate_random_synapse(self):
        private_key = Crypto.generate_private_ed25519()
        public_key = Crypto.public_key_from_private(private_key)
        synapse = bittensor_pb2.Synapse(
            version = bittensor.__version__,
            neuron_key = Crypto.public_key_to_string(public_key),
            synapse_key = Crypto.public_key_to_string(public_key),
            address = '0.0.0.0',
            port = 12231,
            block_hash = Crypto.lastest_block_hash()
        )
        return private_key, synapse

    def test_proof_of_work(self):
        difficulty = 1
        _, synapse = self.random_synapse
        synapse = Crypto.fill_proof_of_work(synapse, difficulty)
        assert Crypto.count_zeros(synapse.proof_of_work) >= difficulty

    def test_signature(self):
        private, synapse = self.random_synapse
        synapse = Crypto.sign_synapse(private, synapse)
        assert Crypto.verify_synapse(synapse)
        
    def test_false_signature(self):
        private, synapse = self.random_synapse
        synapse = Crypto.sign_synapse(private, synapse)
        synapse.signature = os.urandom(256)
        assert Crypto.verify_synapse(synapse) == False

    def test_check_signature(self):
        private, synapse = self.random_synapse
        synapse = Crypto.sign_synapse(private, synapse)
        synapse.signature = os.urandom(256)
        assert Crypto.check_signature(synapse) == False
    
    def test_difficulty(self):
        private, synapse = self.random_synapse
        synapse = Crypto.sign_synapse(private, synapse)
        digest = Crypto.digest(synapse)
        difficulty = Crypto.difficulty(digest)
        assert difficulty == 0

