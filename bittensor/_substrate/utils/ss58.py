# Python Substrate Interface Library
#
# Copyright 2018-2020 Stichting Polkascan (Polkascan Foundation).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#  ss58.py

""" SS58 is a simple address format designed for Substrate based chains.
    Encoding/decoding according to specification on https://wiki.parity.io/External-Address-Format-(SS58)

"""
import base58
from hashlib import blake2b

from scalecodec import ScaleBytes
from scalecodec.types import U8, U16, U32, U64


def ss58_decode(address, valid_address_type=None):
    """
    Decodes given SS58 encoded address to an account ID
    Parameters
    ----------
    address: e.g. EaG2CRhJWPb7qmdcJvy3LiWdh26Jreu9Dx6R1rXxPmYXoDk
    valid_address_type

    Returns
    -------
    Decoded string AccountId
    """
    checksum_prefix = b'SS58PRE'

    ss58_format = base58.b58decode(address)

    if valid_address_type and ss58_format[0] != valid_address_type:
        raise ValueError("Invalid Address type")

    # Determine checksum length according to length of address string
    if len(ss58_format) in [3, 4, 6, 10]:
        checksum_length = 1
    elif len(ss58_format) in [5, 7, 11, 35]:
        checksum_length = 2
    elif len(ss58_format) in [8, 12]:
        checksum_length = 3
    elif len(ss58_format) in [9, 13]:
        checksum_length = 4
    elif len(ss58_format) in [14]:
        checksum_length = 5
    elif len(ss58_format) in [15]:
        checksum_length = 6
    elif len(ss58_format) in [16]:
        checksum_length = 7
    elif len(ss58_format) in [17]:
        checksum_length = 8
    else:
        raise ValueError("Invalid address length")

    checksum = blake2b(checksum_prefix + ss58_format[0:-checksum_length]).digest()

    if checksum[0:checksum_length] != ss58_format[-checksum_length:]:
        raise ValueError("Invalid checksum")

    return ss58_format[1:len(ss58_format)-checksum_length].hex()


def ss58_encode(address, address_type=42):
    """
    Encodes an account ID to an Substrate address according to provided address_type

    Parameters
    ----------
    address
    address_type

    Returns
    -------

    """
    checksum_prefix = b'SS58PRE'

    if type(address) is bytes or type(address) is bytearray:
        address_bytes = address
    else:
        address_bytes = bytes.fromhex(address.replace('0x', ''))

    if len(address_bytes) == 32:
        # Checksum size is 2 bytes for public key
        checksum_length = 2
    elif len(address_bytes) in [1, 2, 4, 8]:
        # Checksum size is 1 byte for account index
        checksum_length = 1
    else:
        raise ValueError("Invalid length for address")

    address_format = bytes([address_type]) + address_bytes
    checksum = blake2b(checksum_prefix + address_format).digest()

    return base58.b58encode(address_format + checksum[:checksum_length]).decode()


def ss58_encode_account_index(account_index, address_type=42):
    """
    Encodes an AccountIndex to an Substrate address according to provided address_type

    Parameters
    ----------
    account_index
    address_type

    Returns
    -------

    """

    if 0 <= account_index <= 2**8 - 1:
        account_idx_encoder = U8()
    elif 2**8 <= account_index <= 2**16 - 1:
        account_idx_encoder = U16()
    elif 2**16 <= account_index <= 2**32 - 1:
        account_idx_encoder = U32()
    elif 2**32 <= account_index <= 2**64 - 1:
        account_idx_encoder = U64()
    else:
        raise ValueError("Value too large for an account index")

    return ss58_encode(account_idx_encoder.encode(account_index).data, address_type)


def ss58_decode_account_index(address, valid_address_type=42):
    """
    Decodes given SS58 encoded address to an AccountIndex

    Parameters
    ----------
    address
    valid_address_type

    Returns
    -------
    Decoded int AccountIndex
    """
    account_index_bytes = ss58_decode(address, valid_address_type)

    if len(account_index_bytes) == 2:
        return U8(ScaleBytes('0x{}'.format(account_index_bytes))).decode()
    if len(account_index_bytes) == 4:
        return U16(ScaleBytes('0x{}'.format(account_index_bytes))).decode()
    if len(account_index_bytes) == 8:
        return U32(ScaleBytes('0x{}'.format(account_index_bytes))).decode()
    if len(account_index_bytes) == 16:
        return U64(ScaleBytes('0x{}'.format(account_index_bytes))).decode()
    else:
        raise ValueError("Invalid account index length")

