from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric import ed25519


class Identity():

    def __init__(self):
        self._private = Ed25519PrivateKey.generate()
        self._public = self._private.public_key()

    def public_key(self):
        return '0x%s' % self.public_bytes().hex()

    def public_bytes(self):
        return self._public.public_bytes(serialization.Encoding.Raw,
                                         serialization.PublicFormat.Raw)

    @staticmethod
    def public_bytes_from_string(string):
        result = bytes.fromhex(string[2:])
        return result

    @staticmethod
    def public_from_string(string):
        result = Identity.public_bytes_from_string(string)
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(result)
        return public_key

    def sign(self, digest):
        return self._private.sign(digest)
