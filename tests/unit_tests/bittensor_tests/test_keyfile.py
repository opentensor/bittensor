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

def test_legacy_coldkey():
    keyfile = bittensor.keyfile (path = '/tmp/pytest/coldlegacy_keyfile' )
    keyfile.make_dirs()
    keyfile_data = b'0x32939b6abc4d81f02dff04d2b8d1d01cc8e71c5e4c7492e4fa6a238cdca3512f'
    with open('/tmp/pytest/coldlegacy_keyfile', "wb") as keyfile_obj:
        keyfile_obj.write( keyfile_data )
    assert keyfile.keyfile_data == keyfile_data
    keyfile.encrypt( password = 'this is the fake password' )
    keyfile.decrypt( password = 'this is the fake password' )
    keypair_bytes = b'{"accountId": "0x32939b6abc4d81f02dff04d2b8d1d01cc8e71c5e4c7492e4fa6a238cdca3512f", "publicKey": "0x32939b6abc4d81f02dff04d2b8d1d01cc8e71c5e4c7492e4fa6a238cdca3512f", "secretPhrase": null, "secretSeed": null, "ss58Address": "5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"}'
    assert keyfile.keyfile_data == keypair_bytes
    assert keyfile.get_keypair().ss58_address == "5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm"
    assert keyfile.get_keypair().public_key == "0x32939b6abc4d81f02dff04d2b8d1d01cc8e71c5e4c7492e4fa6a238cdca3512f"

test_legacy_coldkey()