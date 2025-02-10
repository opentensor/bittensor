"""Helper functions used to calculate keys for Substrate storage items"""

from hashlib import blake2b

import xxhash


def blake2_256(data):
    """
    Helper function to calculate a 32 bytes Blake2b hash for provided data, used as key for Substrate storage items
    """
    return blake2b(data, digest_size=32).digest()


def blake2_128(data):
    """
    Helper function to calculate a 16 bytes Blake2b hash for provided data, used as key for Substrate storage items
    """
    return blake2b(data, digest_size=16).digest()


def blake2_128_concat(data):
    """
    Helper function to calculate a 16 bytes Blake2b hash for provided data, concatenated with data, used as key
    for Substrate storage items
    """
    return blake2b(data, digest_size=16).digest() + data


def xxh128(data):
    """
    Helper function to calculate a 2 concatenated xxh64 hash for provided data, used as key for several Substrate
    """
    storage_key1 = bytearray(xxhash.xxh64(data, seed=0).digest())
    storage_key1.reverse()

    storage_key2 = bytearray(xxhash.xxh64(data, seed=1).digest())
    storage_key2.reverse()

    return storage_key1 + storage_key2


def two_x64_concat(data):
    """
    Helper function to calculate a xxh64 hash with concatenated data for provided data,
    used as key for several Substrate
    """
    storage_key = bytearray(xxhash.xxh64(data, seed=0).digest())
    storage_key.reverse()

    return storage_key + data


def xxh64(data):
    storage_key = bytearray(xxhash.xxh64(data, seed=0).digest())
    storage_key.reverse()

    return storage_key


def identity(data):
    return data
