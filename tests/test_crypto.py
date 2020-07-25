from opentensor import opentensor_pb2_grpc as proto_grpc
from opentensor import opentensor_pb2 as proto_pb2
import opentensor

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization

import os


def test_empty_protos():
    proto_pb2.TensorMessage()
    return


def test_signed_message_fail():

    # Generate key.
    identity = opentensor.Identity()

    # Get source_key as bytes.
    public_bytes = identity.public_bytes()

    # Create nounce. A string representation of random bytes.
    nounce = 1

    # Create SHA256 digest.
    digest = hashes.Hash(hashes.SHA1(), backend=default_backend())
    digest.update(public_bytes)
    digest.update(bytes(nounce))
    real_sha1_digest = digest.finalize()

    # False nounce.
    false_nounce = 0

    digest = hashes.Hash(hashes.SHA1(), backend=default_backend())
    digest.update(public_bytes)
    digest.update(bytes(false_nounce))
    false_sha1_digest = digest.finalize()

    # Create signature
    signature = identity.sign(real_sha1_digest)

    # Build message
    fwd_request = proto_pb2.TensorMessage(neuron_key=identity.public_key(),
                                          nounce=false_nounce,
                                          signature=signature)

    # Server side, load key.
    in_public_bytes = opentensor.Identity.public_bytes_from_string(
        fwd_request.neuron_key)
    in_public_key = opentensor.Identity.public_from_string(
        fwd_request.neuron_key)

    # Recreate digest.
    digest = hashes.Hash(hashes.SHA1(), backend=default_backend())
    digest.update(in_public_bytes)
    digest.update(bytes(fwd_request.nounce))
    target_side_digest = digest.finalize()

    # Verify the authenticity of the message.
    try:
        in_public_key.verify(fwd_request.signature, target_side_digest)
    except:
        return True

    assert (False)


def test_signed_message():

    # Generate key.
    identity = opentensor.Identity()

    # Get source_key as bytes.
    public_bytes = identity.public_bytes()

    # Create nounce. A string representation of random bytes.
    nounce = 1

    # Create SHA1 digest.
    digest = hashes.Hash(hashes.SHA1(), backend=default_backend())
    digest.update(public_bytes)
    digest.update(bytes(nounce))
    sha1_digest = digest.finalize()

    # Sign the digest.
    signature = identity.sign(sha1_digest)

    # Build message
    fwd_request = proto_pb2.TensorMessage(neuron_key=identity.public_key(),
                                          nounce=nounce,
                                          signature=signature)

    # Server side, load key.
    in_public_bytes = opentensor.Identity.public_bytes_from_string(
        fwd_request.neuron_key)
    in_public_key = opentensor.Identity.public_from_string(
        fwd_request.neuron_key)

    # Recreate digest.
    digest = hashes.Hash(hashes.SHA1(), backend=default_backend())
    digest.update(in_public_bytes)
    digest.update(bytes(fwd_request.nounce))
    target_side_digest = digest.finalize()

    # Verify the authenticity of the message.
    in_public_key.verify(fwd_request.signature, target_side_digest)

    return True
