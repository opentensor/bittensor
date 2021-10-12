import bittensor
from unittest.mock import MagicMock
import os
import shutil

# Init dirs.
if os.path.exists('/tmp/pytest'):
    shutil.rmtree('/tmp/pytest')

def test_create():
    keyfile = bittensor.keyfile (path = '/tmp/pytest/keyfile' )
    alice = bittensor.Keypair.create_from_uri ('/Alice')
    keyfile.set_keypair(alice, encrypt=True, overwrite=True, password = 'thisisafakepassword')
    assert keyfile.is_readable()
    assert keyfile.is_writable()
    assert keyfile.is_encrypted()
    keyfile.decrypt( password = 'thisisafakepassword' )
    assert not keyfile.is_encrypted()
    keyfile.encrypt( password = 'thisisafakepassword' )
    assert keyfile.is_encrypted()
    keyfile.decrypt( password = 'thisisafakepassword' )
    assert not keyfile.is_encrypted()
    keyfile.get_keypair( password = 'thisisafakepassword' ).ss58_address == alice.ss58_address
    keyfile.get_keypair( password = 'thisisafakepassword' ).mnemonic == alice.mnemonic
    keyfile.get_keypair( password = 'thisisafakepassword' ).seed_hex == alice.seed_hex
    keyfile.get_keypair( password = 'thisisafakepassword' ).private_key == alice.private_key
    keyfile.get_keypair( password = 'thisisafakepassword' ).public_key == alice.public_key
    bob = bittensor.Keypair.create_from_uri ('/Bob')
    keyfile.set_keypair(bob, encrypt=True, overwrite=True, password = 'thisisafakepassword')
    keyfile.get_keypair( password = 'thisisafakepassword' ).ss58_address == bob.ss58_address
    keyfile.get_keypair( password = 'thisisafakepassword' ).mnemonic == bob.mnemonic
    keyfile.get_keypair( password = 'thisisafakepassword' ).seed_hex == bob.seed_hex
    keyfile.get_keypair( password = 'thisisafakepassword' ).private_key == bob.private_key
    keyfile.get_keypair( password = 'thisisafakepassword' ).public_key == bob.public_key
