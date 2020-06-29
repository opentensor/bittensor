from opentensor_proto import opentensor_pb2_grpc as proto_grpc
from opentensor_proto import opentensor_pb2 as proto_pb2

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization

import os

def test_empty_protos():
    proto_pb2.TensorMessage()
    return

def test_signed_message_fail():
    
    # Generate key.
    private = ed25519.Ed25519PrivateKey.generate()
    public = private.public_key()
    
    # Get public key as bytes.
    public_bytes = public.public_bytes (
            serialization.Encoding.Raw, 
            serialization.PublicFormat.Raw 
    )

    # Create nounce. A string representation of random bytes. 
    nounce = os.urandom(12)

    # Content.
    content = os.urandom(100)
    
    # Create SHA256 digest.
    digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
    digest.update(public_bytes)
    digest.update(nounce)
    digest.update(content)
    real_sha256_digest = digest.finalize()
   
    # Creat false SHA256 digest.
    false_content = os.urandom(100)
    
    digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
    digest.update(public_bytes)
    digest.update(nounce)
    digest.update(false_content)
    false_sha256_digest = digest.finalize()

    # Create signature
    signature = private.sign(real_sha256_digest)
    
    # Build message
    fwd_request = proto_pb2.TensorMessage (
        public_key = public_bytes, 
        nounce = nounce,
        digest = false_sha256_digest,
        content = content,
        signature = signature
    )
 
    # Server side, load key.
    in_public_key = ed25519.Ed25519PublicKey.from_public_bytes(fwd_request.public_key)
    
    # Verify the authenticity of the message.
    try:
        in_public_key.verify(fwd_request.signature, fwd_request.digest)
    except:
        return True
    
    assert(False)


def test_signed_message():
    
    # Generate key.
    private = ed25519.Ed25519PrivateKey.generate()
    public = private.public_key()
    
    # Get public key as bytes.
    public_bytes = public.public_bytes (
            serialization.Encoding.Raw, 
            serialization.PublicFormat.Raw 
    )

    # Create nounce. A string representation of random bytes. 
    nounce = os.urandom(12)

    # Content.
    content = os.urandom(100)

    # Create SHA256 digest.
    digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
    digest.update(public_bytes)
    digest.update(nounce)
    digest.update(content)
    sha256_digest = digest.finalize()
    
    # Create signature
    signature = private.sign(sha256_digest)
    
    # Build message
    fwd_request = proto_pb2.TensorMessage (
        public_key = public_bytes, 
        nounce = nounce,
        digest = sha256_digest,
        content = content,
        signature = signature
    )
 
    # Server side, load key.
    in_public_key = ed25519.Ed25519PublicKey.from_public_bytes(fwd_request.public_key)
    
    # Verify the authenticity of the message.
    ou = in_public_key.verify(fwd_request.signature, fwd_request.digest)

    print (ou)


    
