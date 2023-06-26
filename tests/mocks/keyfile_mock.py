# The MIT License (MIT)

# Copyright © 2021 Yuma Rao
# Copyright © 2022 Opentensor Foundation
# Copyright © 2023 Opentensor Technologies

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from bittensor_wallet import serialized_keypair_to_keyfile_data, Keyfile
from bittensor_wallet import Keypair

class MockKeyfile( Keyfile ):
    """ Defines an interface to a mocked keyfile object (nothing is created on device) keypair is treated as non encrypted and the data is just the string version.
    """
    def __init__( self, path: str ):
        super().__init__( path )
        
        self._mock_keypair = Keypair.create_from_mnemonic( mnemonic = 'arrive produce someone view end scout bargain coil slight festival excess struggle' )
        self._mock_data = serialized_keypair_to_keyfile_data( self._mock_keypair )

    def __str__(self):
        if not self.exists_on_device():
            return "Keyfile (empty, {})>".format( self.path )
        if self.is_encrypted():
            return "Keyfile (encrypted, {})>".format( self.path )
        else:
            return "Keyfile (decrypted, {})>".format( self.path )

    def __repr__(self):
        return self.__str__()

    @property
    def keypair( self ) -> 'Keypair':
        return self._mock_keypair

    @property
    def data( self ) -> bytes:
        return bytes(self._mock_data)

    @property
    def keyfile_data( self ) -> bytes:
        return bytes( self._mock_data)

    def set_keypair ( self, keypair: 'Keypair', encrypt: bool = True, overwrite: bool = False, password:str = None):
        self._mock_keypair = keypair
        self._mock_data = serialized_keypair_to_keyfile_data( self._mock_keypair )

    def get_keypair(self, password: str = None) -> 'Keypair':
        return self._mock_keypair

    def make_dirs( self ):
        return

    def exists_on_device( self ) -> bool:
        return True

    def is_readable( self ) -> bool:
        return True

    def is_writable( self ) -> bool:
        return True

    def is_encrypted ( self ) -> bool:
        return False

    def encrypt( self, password: str = None):
        raise ValueError('Cannot encrypt a mock keyfile')

    def decrypt( self, password: str = None):
        return
