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
    
    # Get source_key as bytes.
    source_key = public.public_bytes (
            serialization.Encoding.Raw, 
            serialization.PublicFormat.Raw 
    )

    # Create nounce. A string representation of random bytes. 
    nounce = os.urandom(12)

    # Create SHA256 digest.
    digest = hashes.Hash(hashes.SHA1(), backend=default_backend())
    digest.update(source_key)
    digest.update(nounce)
    real_sha1_digest = digest.finalize()
   
    # False nounce.
    false_nounce = os.urandom(12)
    
    digest = hashes.Hash(hashes.SHA1(), backend=default_backend())
    digest.update(source_key)
    digest.update(false_nounce)
    false_sha1_digest = digest.finalize()

    # Create signature
    signature = private.sign(real_sha1_digest)
    
    # Build message
    fwd_request = proto_pb2.TensorMessage (
        source_key = source_key, 
        nounce = false_nounce,
        signature = signature
    )
 
    # Server side, load key.
    in_source_key = ed25519.Ed25519PublicKey.from_public_bytes(fwd_request.source_key)
    
    # Recreate digest.
    digest = hashes.Hash(hashes.SHA1(), backend=default_backend())
    digest.update(fwd_request.source_key)
    digest.update(fwd_request.nounce)
    target_side_digest = digest.finalize()

    # Verify the authenticity of the message.
    try:
        in_source_key.verify(fwd_request.signature, target_side_digest)
    except:
        return True
    
    assert(False)


def test_signed_message():
    
    # Generate key.
    private = ed25519.Ed25519PrivateKey.generate()
    public = private.public_key()
    
    # Get source_key as bytes.
    source_key = public.public_bytes (
            serialization.Encoding.Raw, 
            serialization.PublicFormat.Raw 
    )

    # Create nounce. A string representation of random bytes. 
    nounce = os.urandom(12)

    # Create SHA1 digest.
    digest = hashes.Hash(hashes.SHA1(), backend=default_backend())
    digest.update(source_key)
    digest.update(nounce)
    sha1_digest = digest.finalize()
   
    # Sign the digest.
    signature = private.sign(sha1_digest)
    
    # Build message
    fwd_request = proto_pb2.TensorMessage (
        source_key = source_key, 
        nounce = nounce,
        signature = signature
    )
 
    # Server side, load key.
    in_source_key = ed25519.Ed25519PublicKey.from_public_bytes(fwd_request.source_key)
    
    # Recreate digest.
    digest = hashes.Hash(hashes.SHA1(), backend=default_backend())
    digest.update(fwd_request.source_key)
    digest.update(fwd_request.nounce)
    target_side_digest = digest.finalize()

    # Verify the authenticity of the message.
    in_source_key.verify(fwd_request.signature, target_side_digest)
    
    return True
