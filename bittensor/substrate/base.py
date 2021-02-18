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

import warnings
from hashlib import blake2b

import binascii
import json
import logging
import re

import requests
from typing import Optional

from scalecodec.exceptions import RemainingScaleBytesNotEmptyException
from websocket import create_connection, WebSocketConnectionClosedException

from scalecodec import ScaleBytes, GenericCall
from scalecodec.base import ScaleDecoder, RuntimeConfigurationObject, ScaleType
from scalecodec.block import ExtrinsicsDecoder, EventsDecoder, LogDigest, Extrinsic
from scalecodec.metadata import MetadataDecoder
from scalecodec.type_registry import load_type_registry_preset
from scalecodec.updater import update_type_registries

from .key import extract_derive_path
from .utils.hasher import blake2_256, two_x64_concat, xxh128, blake2_128, blake2_128_concat, identity
from .exceptions import SubstrateRequestException, ConfigurationError, StorageFunctionNotFound, BlockHashNotFound, \
    ExtrinsicNotFound
from .constants import *
from .utils.ss58 import ss58_decode, ss58_encode, is_valid_ss58_address

from bip39 import bip39_to_mini_secret, bip39_generate
import sr25519
import ed25519


__all__ = ['Keypair', 'KeypairType', 'SubstrateInterface', 'ExtrinsicReceipt', 'logger']

logger = logging.getLogger(__name__)


class KeypairType:
    ED25519 = 0
    SR25519 = 1


class Keypair:

    def __init__(self, ss58_address=None, public_key=None, private_key=None, ss58_format=None,
                 address_type=None, seed_hex=None,
                 crypto_type=KeypairType.SR25519):
        """
        Allows generation of Keypairs from a variety of input combination, such as a public/private key combination, a
        mnemonic or a uri containing soft and hard derivation paths. With these Keypairs data can be signed and verified

        Parameters
        ----------
        ss58_address: Substrate address
        public_key: hex string or bytes of public_key key
        private_key: hex string or bytes of private key
        ss58_format: Substrate address format, default = 42
        address_type: (deprecated) replaced by ss58_format
        seed_hex: hex string of seed
        crypto_type: Use KeypairType.SR25519 or KeypairType.ED25519 cryptography for generating the Keypair
        """

        self.crypto_type = crypto_type
        self.seed_hex = seed_hex
        self.derive_path = None

        if ss58_address and not public_key:
            public_key = ss58_decode(ss58_address, valid_ss58_format=ss58_format)

        if not public_key:
            raise ValueError('No SS58 formatted address or public key provided')

        if type(public_key) is bytes:
            public_key = public_key.hex()

        public_key = '0x{}'.format(public_key.replace('0x', ''))

        if len(public_key) != 66:
            raise ValueError('Public key should be 32 bytes long')

        if address_type is not None:
            warnings.warn("Keyword 'address_type' will be replaced by 'ss58_format'", DeprecationWarning)
            ss58_format = address_type

        self.ss58_format = ss58_format

        if not ss58_address:
            ss58_address = ss58_encode(public_key, ss58_format=ss58_format)

        self.public_key = public_key

        self.ss58_address = ss58_address

        if private_key:

            if type(private_key) is bytes:
                private_key = private_key.hex()

            private_key = '0x{}'.format(private_key.replace('0x', ''))

            if self.crypto_type == KeypairType.SR25519 and len(private_key) != 130:
                raise ValueError('Secret key should be 64 bytes long')

        self.private_key = private_key

        self.mnemonic = None

    @classmethod
    def generate_mnemonic(cls, words=12):
        """
        Generates a new seed phrase with given amount of words (default 12)

        Parameters
        ----------
        words: The amount of words to generate, valid values are 12, 15, 18, 21 and 24

        Returns
        -------
        Seed phrase
        """
        return bip39_generate(words)

    @classmethod
    def create_from_mnemonic(cls, mnemonic, ss58_format=42, address_type=None, crypto_type=KeypairType.SR25519):
        """
        Create a Keypair for given memonic

        Parameters
        ----------
        mnemonic: Seed phrase
        ss58_format: Substrate address format
        address_type: (deprecated)
        crypto_type: Use `KeypairType.SR25519` or `KeypairType.ED25519` cryptography for generating the Keypair

        Returns
        -------
        Keypair
        """
        seed_array = bip39_to_mini_secret(mnemonic, "")

        if address_type is not None:
            warnings.warn("Keyword 'address_type' will be replaced by 'ss58_format'", DeprecationWarning)
            ss58_format = address_type

        keypair = cls.create_from_seed(
            seed_hex=binascii.hexlify(bytearray(seed_array)).decode("ascii"),
            ss58_format=ss58_format,
            crypto_type=crypto_type
        )
        keypair.mnemonic = mnemonic

        return keypair

    @classmethod
    def create_from_seed(
            cls, seed_hex: str, ss58_format: Optional[int] = 42, address_type=None, crypto_type=KeypairType.SR25519
    ) -> 'Keypair':
        """
        Create a Keypair for given seed

        Parameters
        ----------
        seed_hex: hex string of seed
        ss58_format: Substrate address format
        address_type: (deprecated)
        crypto_type: Use KeypairType.SR25519 or KeypairType.ED25519 cryptography for generating the Keypair

        Returns
        -------
        Keypair
        """

        if address_type is not None:
            warnings.warn("Keyword 'address_type' will be replaced by 'ss58_format'", DeprecationWarning)
            ss58_format = address_type

        if crypto_type == KeypairType.SR25519:
            public_key, private_key = sr25519.pair_from_seed(bytes.fromhex(seed_hex.replace('0x', '')))
        elif crypto_type == KeypairType.ED25519:
            private_key, public_key = ed25519.ed_from_seed(bytes.fromhex(seed_hex.replace('0x', '')))
        else:
            raise ValueError('crypto_type "{}" not supported'.format(crypto_type))

        public_key = public_key.hex()
        private_key = private_key.hex()

        ss58_address = ss58_encode(f'0x{public_key}', ss58_format)

        return cls(
            ss58_address=ss58_address, public_key=public_key, private_key=private_key,
            ss58_format=ss58_format, crypto_type=crypto_type, seed_hex=seed_hex
        )

    @classmethod
    def create_from_uri(
            cls, suri: str, ss58_format: Optional[int] = 42, address_type=None, crypto_type=KeypairType.SR25519
    ) -> 'Keypair':
        """
        Creates Keypair for specified suri in following format: `<mnemonic>/<soft-path>//<hard-path>`

        Parameters
        ----------
        suri:
        ss58_format: Substrate address format
        address_type: (deprecated)
        crypto_type: Use KeypairType.SR25519 or KeypairType.ED25519 cryptography for generating the Keypair

        Returns
        -------
        Keypair
        """

        if address_type is not None:
            warnings.warn("Keyword 'address_type' will be replaced by 'ss58_format'", DeprecationWarning)
            ss58_format = address_type

        if suri and suri.startswith('/'):
            suri = DEV_PHRASE + suri

        suri_regex = re.match(r'^(?P<phrase>\w+( \w+)*)(?P<path>(//?[^/]+)*)(///(?P<password>.*))?$', suri)

        suri_parts = suri_regex.groupdict()

        if suri_parts['password']:
            raise NotImplementedError("Passwords in suri not supported")

        derived_keypair = cls.create_from_mnemonic(
            suri_parts['phrase'], ss58_format=ss58_format, crypto_type=crypto_type
        )

        if suri_parts['path'] != '':

            derived_keypair.derive_path = suri_parts['path']

            if crypto_type not in [KeypairType.SR25519]:
                raise NotImplementedError('Derivation paths for this crypto type not supported')

            derive_junctions = extract_derive_path(suri_parts['path'])

            child_pubkey = bytes.fromhex(derived_keypair.public_key[2:])
            child_privkey = bytes.fromhex(derived_keypair.private_key[2:])

            for junction in derive_junctions:

                if junction.is_hard:

                    _, child_pubkey, child_privkey = sr25519.hard_derive_keypair(
                        (junction.chain_code, child_pubkey, child_privkey),
                        b''
                    )

                else:

                    _, child_pubkey, child_privkey = sr25519.derive_keypair(
                        (junction.chain_code, child_pubkey, child_privkey),
                        b''
                    )

            derived_keypair = Keypair(public_key=child_pubkey, private_key=child_privkey, ss58_format=ss58_format)

        return derived_keypair

    @classmethod
    def create_from_private_key(
        cls, private_key, public_key=None, ss58_address=None, ss58_format=None, crypto_type=KeypairType.SR25519,
        address_type=None
    ):
        """
        Creates Keypair for specified public/private keys
        Parameters
        ----------
        private_key: hex string or bytes of private key
        public_key: hex string or bytes of public key
        ss58_address: Substrate address
        ss58_format: Substrate address format, default = 42
        address_type: (deprecated)
        crypto_type: Use KeypairType.SR25519 or KeypairType.ED25519 cryptography for generating the Keypair

        Returns
        -------
        Keypair
        """
        if address_type is not None:
            warnings.warn("Keyword 'address_type' will be replaced by 'ss58_format'", DeprecationWarning)
            ss58_format = address_type

        return cls(
            ss58_address=ss58_address, public_key=public_key, private_key=private_key,
            ss58_format=ss58_format, crypto_type=crypto_type
        )

    def sign(self, data):
        """
        Creates a signature for given data

        Parameters
        ----------
        data: data to sign in `Scalebytes`, bytes or hex string format

        Returns
        -------
        signature in hex string format

        """
        if type(data) is ScaleBytes:
            data = bytes(data.data)
        elif data[0:2] == '0x':
            data = bytes.fromhex(data[2:])
        else:
            data = data.encode()

        if not self.private_key:
            raise ConfigurationError('No private key set to create signatures')

        if self.crypto_type == KeypairType.SR25519:

            signature = sr25519.sign((bytes.fromhex(self.public_key[2:]), bytes.fromhex(self.private_key[2:])), data)
        elif self.crypto_type == KeypairType.ED25519:
            signature = ed25519.ed_sign(bytes.fromhex(self.public_key[2:]), bytes.fromhex(self.private_key[2:]), data)
        else:
            raise ConfigurationError("Crypto type not supported")

        return "0x{}".format(signature.hex())

    def verify(self, data, signature):
        """
        Verifies data with specified signature

        Parameters
        ----------
        data: data to be verified in `Scalebytes`, bytes or hex string format
        signature: signature in bytes or hex string format

        Returns
        -------
        True if data is signed with this Keypair, otherwise False
        """

        if type(data) is ScaleBytes:
            data = bytes(data.data)
        elif data[0:2] == '0x':
            data = bytes.fromhex(data[2:])
        else:
            data = data.encode()

        if type(signature) is str and signature[0:2] == '0x':
            signature = bytes.fromhex(signature[2:])

        if type(signature) is not bytes:
            raise TypeError("Signature should be of type bytes or a hex-string")

        if self.crypto_type == KeypairType.SR25519:
            return sr25519.verify(signature, data, bytes.fromhex(self.public_key[2:]))
        elif self.crypto_type == KeypairType.ED25519:
            return ed25519.ed_verify(signature, data, bytes.fromhex(self.public_key[2:]))
        else:
            raise ConfigurationError("Crypto type not supported")

    def __repr__(self):
        return '<Keypair (ss58_address={})>'.format(self.ss58_address)

    def toDict(self):
        return {
            'accountId': self.public_key,
            'publicKey': self.public_key,
            'secretPhrase': self.mnemonic,
            'secretSeed': "0x" + self.seed_hex,
            'ss58Address': self.ss58_address
        }


class SubstrateInterface:

    def __init__(self, url=None, websocket=None, ss58_format=None, type_registry=None, type_registry_preset=None,
                 cache_region=None, address_type=None, runtime_config=None, use_remote_preset=False):
        """
        A specialized class in interfacing with a Substrate node.

        Parameters
        ----------
        url: the URL to the substrate node, either in format https://127.0.0.1:9933 or wss://127.0.0.1:9944
        ss58_format: The address type which account IDs will be SS58-encoded to Substrate addresses. Defaults to 42, for Kusama the address type is 2
        type_registry: A dict containing the custom type registry in format: {'types': {'customType': 'u32'},..}
        type_registry_preset: The name of the predefined type registry shipped with the SCALE-codec, e.g. kusama
        cache_region: a Dogpile cache region as a central store for the metadata cache
        use_remote_preset: When True preset is downloaded from Github master, otherwise use files from local installed scalecodec package
        """

        if (not url and not websocket) or (url and websocket):
            raise ValueError("Either 'url' or 'websocket' must be provided")

        if address_type is not None:
            warnings.warn("Keyword 'address_type' will be replaced by 'ss58_format'", DeprecationWarning)
            ss58_format = address_type

        # Initialize lazy loading variables
        self.__version = None
        self.__name = None
        self.__properties = None
        self.__chain = None

        self.__token_decimals = None
        self.__token_symbol = None
        self.__ss58_format = None

        self.cache_region = cache_region

        self.ss58_format = ss58_format
        self.type_registry_preset = type_registry_preset
        self.type_registry = type_registry

        self.request_id = 1
        self.url = url
        self.websocket = None

        self.mock_extrinsics = None
        self.default_headers = {
            'content-type': "application/json",
            'cache-control': "no-cache"
        }

        self.metadata_decoder = None

        self.runtime_version = None
        self.transaction_version = None

        self.block_hash = None
        self.block_id = None

        self.metadata_cache = {}
        self.type_registry_cache = {}

        if not runtime_config:
            runtime_config = RuntimeConfigurationObject()

        self.runtime_config = runtime_config

        self.debug = False

        self.reload_type_registry(use_remote_preset=use_remote_preset)

    def connect (self):
        if self.url and (self.url[0:6] == 'wss://' or self.url[0:5] == 'ws://'):
            self.debug_message("Connecting to {} ...".format(self.url))
            self.websocket = create_connection(
                self.url,
                max_size=2 ** 32,
                read_limit=2 ** 32,
                write_limit=2 ** 32,
            )

    def is_connected (self) -> bool:
        return self.websocket.connected

    def debug_message(self, message):
        logger.debug(message)

    def rpc_request(self, method, params, result_handler=None):
        """
        Method that handles the actual RPC request to the Substrate node. The other implemented functions eventually
        use this method to perform the request.

        Parameters
        ----------
        result_handler: Callback of function that processes the result received from the node
        method: method of the JSONRPC request
        params: a list containing the parameters of the JSONRPC request

        Returns
        -------
        a dict with the parsed result of the request.
        """
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": self.request_id
        }

        self.debug_message('RPC request #{}: "{}"'.format(self.request_id, method))

        if self.websocket:
            try:
                self.websocket.send(json.dumps(payload))

                if callable(result_handler):
                    # If result handler is set, pass result through and loop until handler return not None
                    event_number = 0
                    json_body = None
                    while not json_body:
                        result = json.loads(self.websocket.recv())
                        self.debug_message("Websocket result [{}] Received from node: {}".format(event_number, result))

                        # Check if response has error
                        if 'error' in result:
                            raise SubstrateRequestException(result['error'])

                        callback_result = result_handler(result)
                        if callback_result:
                            json_body = callback_result

                        event_number += 1
                else:

                    json_body = json.loads(self.websocket.recv())

            except WebSocketConnectionClosedException:
                if self.url:
                    # Try to reconnect websocket and retry rpc_request
                    self.debug_message("Connection Closed; Trying to reconnecting...")
                    self.connect_websocket()

                    return self.rpc_request(method=method, params=params, result_handler=result_handler)
                else:
                    # websocket connection is externally created, re-raise exception
                    raise

        else:

            if result_handler:
                raise ConfigurationError("Result handlers only available for websockets (ws://) connections")

            response = requests.request("POST", self.url, data=json.dumps(payload), headers=self.default_headers)

            if response.status_code != 200:
                raise SubstrateRequestException("RPC request failed with HTTP status code {}".format(response.status_code))

            json_body = response.json()

        self.request_id += 1
        return json_body

    @property
    def name(self):
        if self.__name is None:
            self.__name = self.rpc_request("system_name", []).get('result')
        return self.__name

    @property
    def properties(self):
        if self.__properties is None:
            self.__properties = self.rpc_request("system_properties", []).get('result')
        return self.__properties

    @property
    def chain(self):
        if self.__chain is None:
            self.__chain = self.rpc_request("system_chain", []).get('result')
        return self.__chain

    @property
    def version(self):
        if self.__version is None:
            self.__version = self.rpc_request("system_version", []).get('result')
        return self.__version

    @property
    def token_decimals(self):
        if self.__token_decimals is None:
            self.__token_decimals = self.properties.get('tokenDecimals')
        return self.__token_decimals

    @token_decimals.setter
    def token_decimals(self, value):
        if type(value) is not int and value is not None:
            raise TypeError('Token decimals must be an int')
        self.__token_decimals = value

    @property
    def token_symbol(self):
        if self.__token_symbol is None:
            if self.properties:
                self.__token_symbol = self.properties.get('tokenSymbol')
            else:
                self.__token_symbol = 'UNIT'
        return self.__token_symbol

    @token_symbol.setter
    def token_symbol(self, value):
        self.__token_symbol = value

    @property
    def ss58_format(self):
        if self.__ss58_format is None:
            if self.properties:
                self.__ss58_format = self.properties.get('ss58Format')
            else:
                self.__ss58_format = 42
        return self.__ss58_format

    @ss58_format.setter
    def ss58_format(self, value):
        if type(value) is not int and value is not None:
            raise TypeError('ss58_format must be an int')
        self.__ss58_format = value

    def get_chain_head(self):
        """
        A pass-though to existing JSONRPC method `chain_getHead`

        Returns
        -------

        """
        response = self.rpc_request("chain_getHead", [])

        if 'error' in response:
            raise SubstrateRequestException(response['error']['message'])

        return response.get('result')

    def get_chain_finalised_head(self):
        """
        A pass-though to existing JSONRPC method `chain_getFinalisedHead`

        Returns
        -------

        """
        response = self.rpc_request("chain_getFinalisedHead", [])

        if 'error' in response:
            raise SubstrateRequestException(response['error']['message'])

        return response.get('result')

    def get_chain_block(self, block_hash=None, block_id=None, metadata_decoder=None):
        """
        A pass-though to existing JSONRPC method `chain_getBlock`. For a decoded version see `get_runtime_block()`

        Parameters
        ----------
        block_hash
        block_id
        metadata_decoder

        Returns
        -------

        """

        if block_id:
            block_hash = self.get_block_hash(block_id)

        response = self.rpc_request("chain_getBlock", [block_hash])

        if 'error' in response:
            raise SubstrateRequestException(response['error']['message'])
        else:
            result = response.get('result')

            if self.mock_extrinsics:
                # Extend extrinsics with mock_extrinsics for e.g. performance tests
                result['block']['extrinsics'].extend(self.mock_extrinsics)

            # Decode extrinsics
            if metadata_decoder:

                result['block']['header']['number'] = int(result['block']['header']['number'], 16)

                for idx, extrinsic_data in enumerate(result['block']['extrinsics']):
                    extrinsic_decoder = ExtrinsicsDecoder(
                        data=ScaleBytes(extrinsic_data),
                        metadata=metadata_decoder,
                        runtime_config=self.runtime_config
                    )
                    extrinsic_decoder.decode()
                    result['block']['extrinsics'][idx] = extrinsic_decoder.value

                for idx, log_data in enumerate(result['block']['header']["digest"]["logs"]):
                    log_digest = LogDigest(ScaleBytes(log_data), runtime_config=self.runtime_config)
                    log_digest.decode()
                    result['block']['header']["digest"]["logs"][idx] = log_digest.value

            return result

    def get_block_hash(self, block_id):
        """
        A pass-though to existing JSONRPC method `chain_getBlockHash`

        Parameters
        ----------
        block_id

        Returns
        -------

        """
        response = self.rpc_request("chain_getBlockHash", [block_id])

        if 'error' in response:
            raise SubstrateRequestException(response['error']['message'])
        else:
            return response.get('result')

    def get_block_header(self, block_hash):
        """
        A pass-though to existing JSONRPC method `chain_getHeader`

        Parameters
        ----------
        block_hash

        Returns
        -------

        """
        response = self.rpc_request("chain_getHeader", [block_hash])

        if 'error' in response:
            raise SubstrateRequestException(response['error']['message'])
        else:
            return response.get('result')

    def get_block_number(self, block_hash):
        """
        A convenience method to get the block number for given block_hash

        Parameters
        ----------
        block_hash

        Returns
        -------

        """
        response = self.rpc_request("chain_getHeader", [block_hash])

        if 'error' in response:
            raise SubstrateRequestException(response['error']['message'])

        elif 'result' in response:

            if response['result']:
                return int(response['result']['number'], 16)

    def get_block_metadata(self, block_hash=None, decode=True):
        """
        A pass-though to existing JSONRPC method `state_getMetadata`. For a decoded version see `get_runtime_metadata()`

        Parameters
        ----------
        block_hash
        decode: DEPRECATED use `get_runtime_metadata()` for decoded version

        Returns
        -------

        """
        params = None
        if block_hash:
            params = [block_hash]
        response = self.rpc_request("state_getMetadata", params)

        if 'error' in response:
            raise SubstrateRequestException(response['error']['message'])

        if response.get('result') and decode:
            metadata_decoder = MetadataDecoder(ScaleBytes(response.get('result')), runtime_config=self.runtime_config)
            metadata_decoder.decode()

            return metadata_decoder

        return response

    def get_storage(self, block_hash, module, function, params=None, return_scale_type=None, hasher=None,
                    spec_version_id='default', metadata=None, metadata_version=None):
        """
        Retrieves the storage entry for given module, function and optional parameters at given block.

        DEPRECATED: use `query()`

        Parameters
        ----------
        block_hash
        module
        function
        params
        return_scale_type: Scale type string to interprete result
        hasher: Hashing method used to determine storage key, defaults to 'Twox64Concat' if not provided
        spec_version_id: DEPRECATED
        metadata
        metadata_version: Version index of Metadata, e.g. 9 for MetadataV9

        Returns
        -------

        """
        storage_hash = self.generate_storage_hash(
            storage_module=module,
            storage_function=function,
            params=params,
            hasher=hasher,
            metadata_version=metadata_version
        )
        response = self.rpc_request("state_getStorageAt", [storage_hash, block_hash])

        if 'error' in response:
            raise SubstrateRequestException(response['error']['message'])

        elif 'result' in response:

            if return_scale_type and response.get('result'):
                obj = ScaleDecoder.get_decoder_class(
                    type_string=return_scale_type,
                    data=ScaleBytes(response.get('result')),
                    metadata=metadata,
                    runtime_config=self.runtime_config
                )
                return obj.decode()
            else:
                return response.get('result')
        else:
            raise SubstrateRequestException("Error occurred during retrieval of events")

    def get_storage_by_key(self, block_hash, storage_key):
        """
        A pass-though to existing JSONRPC method `state_getStorageAt`

        Parameters
        ----------
        block_hash
        storage_key

        Returns
        -------

        """

        response = self.rpc_request("state_getStorageAt", [storage_key, block_hash])
        if 'result' in response:
            return response.get('result')
        elif 'error' in response:
            raise SubstrateRequestException(response['error']['message'])
        else:
            raise SubstrateRequestException("Unknown error occurred during retrieval of events")

    def get_block_events(self, block_hash, metadata_decoder=None):
        """
        A convenience method to fetch the undecoded events from storage

        Parameters
        ----------
        block_hash
        metadata_decoder

        Returns
        -------

        """

        if metadata_decoder and metadata_decoder.version.index >= 9:
            storage_hash = STORAGE_HASH_SYSTEM_EVENTS_V9
        else:
            storage_hash = STORAGE_HASH_SYSTEM_EVENTS

        response = self.rpc_request("state_getStorageAt", [storage_hash, block_hash])

        if 'error' in response:
            raise SubstrateRequestException(response['error']['message'])

        if response.get('result'):

            if metadata_decoder:
                # Process events
                events_decoder = EventsDecoder(
                    data=ScaleBytes(response.get('result')),
                    metadata=metadata_decoder,
                    runtime_config=self.runtime_config
                )
                events_decoder.decode()

                return events_decoder

            else:
                return response
        else:
            raise SubstrateRequestException("Error occurred during retrieval of events")

    def get_block_runtime_version(self, block_hash):
        """
        Retrieve the runtime version id of given block_hash
        Parameters
        ----------
        block_hash

        Returns
        -------

        """
        response = self.rpc_request("chain_getRuntimeVersion", [block_hash])

        if 'error' in response:
            raise SubstrateRequestException(response['error']['message'])

        return response.get('result')

    def generate_storage_hash(self, storage_module, storage_function, params=None, hasher=None, key2_hasher=None, metadata_version=None):
        """
        Generate a storage key for given module/function

        Parameters
        ----------
        storage_module
        storage_function
        params: Parameters of the storage function, provided in scale encoded hex-bytes
        hasher: Hashing method used to determine storage key, defaults to 'Twox64Concat' if not provided
        metadata_version: Version index of Metadata, e.g. 9 for MetadataV9

        Returns
        -------

        """

        if not metadata_version or metadata_version >= 9:
            storage_hash = xxh128(storage_module.encode()) + xxh128(storage_function.encode())

            if params:

                if type(params) is not list:
                    params = [params]

                for idx, param in enumerate(params):
                    if idx == 0:
                        param_hasher = hasher
                    elif idx == 1:
                        param_hasher = key2_hasher
                    else:
                        raise ValueError('Unexpected third parameter for storage call')

                    params_key = bytes()

                    if type(param) is str:
                        params_key += binascii.unhexlify(param)
                    elif type(param) is ScaleBytes:
                        params_key += param.data
                    elif isinstance(param, ScaleDecoder):
                        params_key += param.data.data

                    if not param_hasher:
                        param_hasher = 'Twox128'

                    if param_hasher == 'Blake2_256':
                        storage_hash += blake2_256(params_key)

                    elif param_hasher == 'Blake2_128':
                        storage_hash += blake2_128(params_key)

                    elif param_hasher == 'Blake2_128Concat':
                        storage_hash += blake2_128_concat(params_key)

                    elif param_hasher == 'Twox128':
                        storage_hash += xxh128(params_key)

                    elif param_hasher == 'Twox64Concat':
                        storage_hash += two_x64_concat(params_key)

                    elif param_hasher == 'Identity':
                        storage_hash += identity(params_key)

                    else:
                        raise ValueError('Unknown storage hasher "{}"'.format(param_hasher))

            return '0x{}'.format(storage_hash)

        else:
            storage_hash = storage_module.encode() + b" " + storage_function.encode()

            if params:
                storage_hash += binascii.unhexlify(params)

            # Determine hasher function
            if not hasher:
                hasher = 'Twox128'

            if hasher == 'Blake2_256':
                return "0x{}".format(blake2_256(storage_hash))

            elif hasher == 'Twox128':
                return "0x{}".format(xxh128(storage_hash))

            elif hasher == 'Twox64Concat':
                return "0x{}".format(two_x64_concat(storage_hash))

    def convert_storage_parameter(self, scale_type, value):
        if scale_type == 'AccountId':
            if value[0:2] != '0x':
                return '0x{}'.format(ss58_decode(value, self.ss58_format))

        return value

    # Runtime functions used by Substrate API

    def init_runtime(self, block_hash=None, block_id=None):
        """
        This method is used by all other methods that deals with metadata and types defined in the type registry.
        It optionally retrieves the block_hash when block_id is given and sets the applicable metadata for that
        block_hash. Also it applies all the versioned types at the time of the block_hash.

        Because parsing of metadata and type registry is quite heavy, the result will be cached per runtime id.
        In the future there could be support for caching backends like Redis to make this cache more persistent.

        Parameters
        ----------
        block_hash
        block_id

        Returns
        -------

        """

        if block_id and block_hash:
            raise ValueError('Cannot provide block_hash and block_id at the same time')

        # Check if runtime state already set to current block
        if (block_hash and block_hash == self.block_hash) or (block_id and block_id == self.block_id):
            return

        if block_id is not None:
            block_hash = self.get_block_hash(block_id)

        self.block_hash = block_hash
        self.block_id = block_id

        # In fact calls and storage functions are decoded against runtime of previous block, therefor retrieve
        # metadata and apply type registry of runtime of parent block
        block_header = self.get_block_header(block_hash=self.block_hash) or {}
        parent_block_hash = block_header.get('parentHash')

        if parent_block_hash == '0x0000000000000000000000000000000000000000000000000000000000000000':
            runtime_block_hash = self.block_hash
        else:
            runtime_block_hash = parent_block_hash

        runtime_info = self.get_block_runtime_version(block_hash=runtime_block_hash)

        if runtime_info is None:
            raise SubstrateRequestException(f"No runtime information for block '{block_hash}'")

        # Check if runtime state already set to current block
        if runtime_info.get("specVersion") == self.runtime_version:
            return

        self.runtime_version = runtime_info.get("specVersion")
        self.transaction_version = runtime_info.get("transactionVersion")

        # Set active runtime version
        self.runtime_config.set_active_spec_version_id(self.runtime_version)

        if self.runtime_version not in self.metadata_cache and self.cache_region:
            # Try to retrieve metadata from Dogpile cache
            cached_metadata = self.cache_region.get('METADATA_{}'.format(self.runtime_version))
            if cached_metadata:
                self.debug_message('Retrieved metadata for {} from Redis'.format(self.runtime_version))
                self.metadata_cache[self.runtime_version] = cached_metadata

        if self.runtime_version in self.metadata_cache:
            # Get metadata from cache
            self.debug_message('Retrieved metadata for {} from memory'.format(self.runtime_version))
            self.metadata_decoder = self.metadata_cache[self.runtime_version]
        else:
            self.metadata_decoder = self.get_block_metadata(block_hash=runtime_block_hash, decode=True)
            self.debug_message('Retrieved metadata for {} from Substrate node'.format(self.runtime_version))

            # Update metadata cache
            self.metadata_cache[self.runtime_version] = self.metadata_decoder

            if self.cache_region:
                self.debug_message('Stored metadata for {} in Redis'.format(self.runtime_version))
                self.cache_region.set('METADATA_{}'.format(self.runtime_version), self.metadata_decoder)

    def iterate_map(self, module, storage_function, block_hash=None):
        """
        iterates over all key-pairs localted at the given module and storage_function. The storage
        item must be a map.

        Parameters
        ----------
        module: The module name in the metadata, e.g. Balances or Account.
        storage_function: The storage function name, e.g. FreeBalance or AccountNonce.
        block_hash: Optional block hash, when left to None the chain tip will be used.

        Returns
        -------
        A two dimensional list of key-value pairs, both decoded into the given type, e.g.
        [[k1, v1], [k2, v2], ...]
        """
        self.init_runtime(block_hash=block_hash)

        key_type = None
        value_type = None
        concat_hash_len = None

        storage_item = self.get_metadata_storage_function(module, storage_function, block_hash=block_hash)
        storage_module = self.get_metadata_module(module)

        if not storage_item or not storage_module:
            raise ValueError(f'Specified storage function "{module}.{storage_function}" not found in metadata')

        if 'MapType' in storage_item.type:
            key_type = storage_item.type['MapType']['key']
            value_type = storage_item.type['MapType']['value']
            if storage_item.type['MapType']['hasher'] == "Blake2_128Concat":
                concat_hash_len = 32
            elif storage_item.type['MapType']['hasher'] == "Twox64Concat":
                concat_hash_len = 16
            elif storage_item.type['MapType']['hasher'] == "Identity":
                concat_hash_len = 0
            else:
                raise ValueError('Unsupported hash type')
        else:
            raise ValueError('Given storage is not a map')

        prefix = self.generate_storage_hash(storage_module.prefix, storage_item.name)
        prefix_len = len(prefix)
        response = self.rpc_request(method="state_getPairs", params=[prefix, block_hash])

        if 'error' in response:
            raise SubstrateRequestException(response['error']['message'])

        pairs = response.get('result')

        # convert keys to the portion that needs to be decoded.
        pairs = map(lambda kp: ["0x" + kp[0][prefix_len + concat_hash_len:], kp[1]], pairs)

        # decode both of them
        pairs = map(
            lambda kp: [self.decode_scale(key_type, kp[0]), self.decode_scale(value_type, kp[1])],
            list(pairs)
        )

        return list(pairs)

    def query(self, module, storage_function, params=None, block_hash=None) -> Optional[ScaleType]:
        """
        Retrieves the storage entry for given module, function and optional parameters at given block hash

        Parameters
        ----------
        module: The module name in the metadata, e.g. Balances or Account
        storage_function: The storage function name, e.g. FreeBalance or AccountNonce
        params: list of params, in the decoded format of the applicable ScaleTypes
        block_hash: Optional block hash, when omitted the chain tip will be used

        Returns
        -------
        ScaleType
        """

        self.init_runtime(block_hash=block_hash)

        # Search storage call in metadata
        metadata_module = self.get_metadata_module(module, block_hash=block_hash)
        storage_item = self.get_metadata_storage_function(module, storage_function, block_hash=block_hash)

        if not metadata_module or not storage_item:
            raise StorageFunctionNotFound('Storage function "{}.{}" not found'.format(module, storage_function))

        key2_hasher = None

        if 'PlainType' in storage_item.type:
            hasher = 'Twox64Concat'
            return_scale_type = storage_item.type.get('PlainType')
            if params:
                raise ValueError('Storage call of type "PlainType" doesn\'t accept params')

        elif 'MapType' in storage_item.type:

            map_type = storage_item.type.get('MapType')
            hasher = map_type.get('hasher')
            return_scale_type = map_type.get('value')

            if not params or len(params) != 1:
                raise ValueError('Storage call of type "MapType" requires 1 parameter')

            # Encode parameter
            params[0] = self.convert_storage_parameter(map_type['key'], params[0])
            param_obj = ScaleDecoder.get_decoder_class(
                type_string=map_type['key'], runtime_config=self.runtime_config
            )
            params[0] = param_obj.encode(params[0])

        elif 'DoubleMapType' in storage_item.type:

            map_type = storage_item.type.get('DoubleMapType')
            hasher = map_type.get('hasher')
            key2_hasher = map_type.get('key2Hasher')
            return_scale_type = map_type.get('value')

            if not params or len(params) != 2:
                raise ValueError('Storage call of type "DoubleMapType" requires 2 parameters')

            # Encode parameter 1
            params[0] = self.convert_storage_parameter(map_type['key1'], params[0])
            param_obj = ScaleDecoder.get_decoder_class(
                type_string=map_type['key1'], runtime_config=self.runtime_config
            )
            params[0] = param_obj.encode(params[0])

            # Encode parameter 2
            params[1] = self.convert_storage_parameter(map_type['key2'], params[1])
            param_obj = ScaleDecoder.get_decoder_class(
                type_string=map_type['key2'], runtime_config=self.runtime_config
            )
            params[1] = param_obj.encode(params[1])

        else:
            raise NotImplementedError("Storage type not implemented")

        storage_hash = self.generate_storage_hash(
            storage_module=metadata_module.prefix,
            storage_function=storage_function,
            params=params,
            hasher=hasher,
            key2_hasher=key2_hasher,
            metadata_version=self.metadata_decoder.version.index
        )

        response = self.rpc_request("state_getStorageAt", [storage_hash, block_hash])

        if 'error' in response:
            raise SubstrateRequestException(response['error']['message'])

        if 'result' in response:

            if return_scale_type and response.get('result'):
                obj = ScaleDecoder.get_decoder_class(
                    type_string=return_scale_type,
                    data=ScaleBytes(response.get('result')),
                    metadata=self.metadata_decoder,
                    runtime_config=self.runtime_config
                )
                obj.decode()
                return obj

        return None

    def get_runtime_state(self, module, storage_function, params=None, block_hash=None):
        warnings.warn("'get_runtime_state' will be replaced by 'query'", DeprecationWarning)

        obj = self.query(module, storage_function, params=params, block_hash=block_hash)
        return {'result': obj.value if obj else None}

    def get_events(self, block_hash=None) -> list:
        """
        Convenience method to get events for a certain block (storage call for module 'System' and function 'Events')

        Parameters
        ----------
        block_hash

        Returns
        -------
        list
        """
        events = []
        storage_obj = self.query(module="System", storage_function="Events", block_hash=block_hash)
        if storage_obj:
            events += storage_obj.elements
        return events

    def get_runtime_events(self, block_hash=None):

        warnings.warn("'get_runtime_events' will be replaced by 'get_events'", DeprecationWarning)

        """
        Convenience method to get events for a certain block (storage call for module 'System' and function 'Events')

        Parameters
        ----------
        block_hash

        Returns
        -------
        Collection of events
        """
        return self.get_runtime_state(
            module="System",
            storage_function="Events",
            block_hash=block_hash
        )

    def get_runtime_metadata(self, block_hash=None):
        """
        Retrieves and decodes the metadata for given block or chaintip if block_hash is omitted.

        Parameters
        ----------
        block_hash

        Returns
        -------

        """
        params = None
        if block_hash:
            params = [block_hash]
        response = self.rpc_request("state_getMetadata", params)

        if 'error' in response:
            raise SubstrateRequestException(response['error']['message'])

        if 'result' in response:
            metadata_decoder = MetadataDecoder(ScaleBytes(response.get('result')), runtime_config=self.runtime_config)
            response['result'] = metadata_decoder.decode()

        return response

    def compose_call(self, call_module, call_function, call_params=(), block_hash=None):
        """
        Composes a call payload which can be used as an unsigned extrinsic or a proposal.

        Parameters
        ----------
        call_module: Name of the runtime module e.g. Balances
        call_function: Name of the call function e.g. transfer
        call_params: This is a dict containing the params of the call. e.g. `{'dest': 'EaG2CRhJWPb7qmdcJvy3LiWdh26Jreu9Dx6R1rXxPmYXoDk', 'value': 1000000000000}`
        block_hash: Use metadata at given block_hash to compose call

        Returns
        -------
        GenericCall
        """
        self.init_runtime(block_hash=block_hash)

        call = ScaleDecoder.get_decoder_class(
            type_string='Call', metadata=self.metadata_decoder, runtime_config=self.runtime_config
        )

        call.encode({
            'call_module': call_module,
            'call_function': call_function,
            'call_args': call_params
        })

        return call

    def get_account_nonce(self, account_address):
        """
        Returns current nonce for given account address

        Parameters
        ----------
        account_address: SS58 formatted address

        Returns
        -------
        int
        """
        account_info = self.query('System', 'Account', [account_address])
        if account_info:
            return account_info.value.get('nonce', 0)

    def generate_signature_payload(self, call, era=None, nonce=0, tip=0, include_call_length=False):

        # Retrieve genesis hash
        genesis_hash = self.get_block_hash(0)

        if not era:
            era = '00'

        if era == '00':
            # Immortal extrinsic
            block_hash = genesis_hash
        else:
            # Determine mortality of extrinsic
            era_obj = ScaleDecoder.get_decoder_class('Era', runtime_config=self.runtime_config)

            if isinstance(era, dict) and 'current' not in era and 'phase' not in era:
                raise ValueError('The era dict must contain either "current" or "phase" element to encode a valid era')

            era_obj.encode(era)
            block_hash = self.get_block_hash(block_id=era_obj.birth(era.get('current')))

        # Create signature payload
        signature_payload = ScaleDecoder.get_decoder_class('ExtrinsicPayloadValue', runtime_config=self.runtime_config)

        if include_call_length:

            length_obj = self.runtime_config.get_decoder_class('Bytes')
            call_data = str(length_obj().encode(str(call.data)))

        else:
            call_data = str(call.data)

        payload_dict = {
            'call': call_data,
            'era': era,
            'nonce': nonce,
            'tip': tip,
            'specVersion': self.runtime_version,
            'genesisHash': genesis_hash,
            'blockHash': block_hash
        }

        if self.transaction_version is not None:
            payload_dict['transactionVersion'] = self.transaction_version

        signature_payload.encode(payload_dict)

        if signature_payload.data.length > 256:
            return ScaleBytes(data=blake2b(signature_payload.data.data, digest_size=32).digest())

        return signature_payload.data

    def create_signed_extrinsic(self, call, keypair: Keypair, era=None, nonce=None, tip=0, signature=None):
        """
        Creates a extrinsic signed by given account details

        Parameters
        ----------
        call: GenericCall to create extrinsic for
        keypair: Keypair used to sign the extrinsic
        era: Specify mortality in blocks in follow format: {'period': <amount_blocks>} If omitted the extrinsic is immortal
        nonce: nonce to include in extrinsics, if omitted the current nonce is retrieved on-chain
        tip: specify tip to gain priority during network congestion
        signature: Optionally provide signature if externally signed

        Returns
        -------
        ExtrinsicsDecoder The signed Extrinsic
        """

        # Check requirements
        if not isinstance(call, GenericCall):
            raise TypeError("'call' must be of type Call")

        # Retrieve nonce
        if nonce is None:
            nonce = self.get_account_nonce(keypair.public_key) or 0

        # Process era
        if era is None:
            era = '00'
        else:
            if isinstance(era, dict) and 'current' not in era and 'phase' not in era:
                # Retrieve current block id
                era['current'] = self.get_block_number(self.get_chain_finalised_head())

        if signature is not None:

            signature = signature.replace('0x', '')

            # Check if signature is a MultiSignature and contains signature version
            if len(signature) == 130:
                signature_version = int(signature[0:2], 16)
                signature = '0x{}'.format(signature[2:])
            else:
                signature_version = keypair.crypto_type

        else:
            # Create signature payload
            signature_payload = self.generate_signature_payload(call=call, era=era, nonce=nonce, tip=tip)

            # Set Signature version to crypto type of keypair
            signature_version = keypair.crypto_type

            # Sign payload
            signature = keypair.sign(signature_payload)

        # Create extrinsic
        extrinsic = ScaleDecoder.get_decoder_class(
            type_string='Extrinsic', metadata=self.metadata_decoder, runtime_config=self.runtime_config
        )

        extrinsic.encode({
            'account_id': keypair.public_key,
            'signature_version': signature_version,
            'signature': signature,
            'call_function': call.value['call_function'],
            'call_module': call.value['call_module'],
            'call_args': call.value['call_args'],
            'nonce': nonce,
            'era': era,
            'tip': tip
        })

        # Set extrinsic hash
        extrinsic.extrinsic_hash = extrinsic.generate_hash()

        return extrinsic

    def create_unsigned_extrinsic(self, call):
        """
        Create unsigned extrinsic for given `Call`
        Parameters
        ----------
        call: GenericCall the call the extrinsic should contain

        Returns
        -------
        ExtrinsicsDecoder
        """
        # Create extrinsic
        extrinsic = ScaleDecoder.get_decoder_class(
            type_string='Extrinsic', metadata=self.metadata_decoder, runtime_config=self.runtime_config
        )

        extrinsic.encode({
            'call_function': call.value['call_function'],
            'call_module': call.value['call_module'],
            'call_args': call.value['call_args']
        })

        return extrinsic

    def submit_extrinsic(self, extrinsic, wait_for_inclusion=False, wait_for_finalization=False) -> "ExtrinsicReceipt":
        """

        Parameters
        ----------
        extrinsic: ExtrinsicsDecoder The extinsic to be send to the network
        wait_for_inclusion: wait until extrinsic is included in a block (only works for websocket connections)
        wait_for_finalization: wait until extrinsic is finalized (only works for websocket connections)

        Returns
        -------
        The hash of the extrinsic submitted to the network

        """

        # Check requirements
        if extrinsic.__class__.__name__ != 'ExtrinsicsDecoder':
            raise TypeError("'extrinsic' must be of type ExtrinsicsDecoder")

        def result_handler(result):
            # Check if extrinsic is included and finalized
            if 'params' in result and type(result['params']['result']) is dict:
                if 'finalized' in result['params']['result'] and wait_for_finalization:
                    return {
                        'block_hash': result['params']['result']['finalized'],
                        'extrinsic_hash': '0x{}'.format(extrinsic.extrinsic_hash),
                        'finalized': True
                    }
                elif 'inBlock' in result['params']['result'] and wait_for_inclusion and not wait_for_finalization:
                    return {
                        'block_hash': result['params']['result']['inBlock'],
                        'extrinsic_hash': '0x{}'.format(extrinsic.extrinsic_hash),
                        'finalized': False
                    }

        if wait_for_inclusion or wait_for_finalization:
            response = self.rpc_request(
                "author_submitAndWatchExtrinsic",
                [str(extrinsic.data)],
                result_handler=result_handler
            )

            result = ExtrinsicReceipt(
                substrate=self,
                extrinsic_hash=response['extrinsic_hash'],
                block_hash=response['block_hash'],
                finalized=response['finalized']
            )

        else:

            response = self.rpc_request("author_submitExtrinsic", [str(extrinsic.data)])

            if 'result' not in response:
                raise SubstrateRequestException(response.get('error'))

            result = ExtrinsicReceipt(
                substrate=self,
                extrinsic_hash=response['result']
            )

        return result

    def get_payment_info(self, call, keypair):
        """
        Retrieves fee estimation via RPC for given extrinsic

        Parameters
        ----------
        call Call object to estimate fees for
        keypair Keypair of the sender, does not have to include private key because no valid signature is required

        Returns
        -------
        Dict with payment info

        E.g. `{'class': 'normal', 'partialFee': 151000000, 'weight': 217238000}`

        """

        # Check requirements
        if not isinstance(call, GenericCall):
            raise TypeError("'call' must be of type Call")

        if not isinstance(keypair, Keypair):
            raise TypeError("'keypair' must be of type Keypair")

        # No valid signature is required for fee estimation
        signature = '0x' + '00' * 64

        # Create extrinsic
        extrinsic = self.create_signed_extrinsic(
            call=call,
            keypair=keypair,
            signature=signature
        )

        payment_info = self.rpc_request('payment_queryInfo', [str(extrinsic.data)])

        # convert partialFee to int
        if 'result' in payment_info:
            payment_info['result']['partialFee'] = int(payment_info['result']['partialFee'])
            return payment_info['result']
        else:
            raise SubstrateRequestException(payment_info['error']['message'])

    def process_metadata_typestring(self, type_string):
        """
        Process how given type_string is decoded with active runtime and type registry

        Parameters
        ----------
        type_string: RUST variable type, e.g. `Vec<Address>`

        Returns
        -------

        dict of properties for given type_string

        E.g.

        `{
            "type_string": "Vec<Address>",
            "decoder_class": "Vec",
            "is_primitive_runtime": false,
            "is_primitive_core": false,
            "spec_version": 1030
        }`

        """
        decoder_class_obj = None

        type_info = {
            "type_string": type_string,
            "decoder_class": None,
            "is_primitive_runtime": None,
            "is_primitive_core": False,
            "spec_version": self.runtime_version
        }

        if self.runtime_version not in self.type_registry_cache:
            self.type_registry_cache[self.runtime_version] = {}

        # Check if already added
        if type_string.lower() in self.type_registry_cache[self.runtime_version]:
            return self.type_registry_cache[self.runtime_version][type_string.lower()]['decoder_class']

        # Try to get decoder class
        decoder_class = self.runtime_config.get_decoder_class(type_string)

        if not decoder_class:

            # Not in type registry, try get hard coded decoder classes
            try:
                decoder_class_obj = ScaleDecoder.get_decoder_class(
                    type_string=type_string, runtime_config=self.runtime_config
                )
                decoder_class = decoder_class_obj.__class__
            except NotImplementedError as e:
                decoder_class = None

        # Process classes that contain subtypes (e.g. Option<ChangesTrieConfiguration>)
        if decoder_class_obj and decoder_class_obj.sub_type:
            type_info["is_primitive_runtime"] = False

            # Try to split on ',' (e.g. ActiveRecovery<BlockNumber, BalanceOf, AccountId>)
            if not re.search('[<()>]', decoder_class_obj.sub_type):
                for element in decoder_class_obj.sub_type.split(','):
                    if element not in ['T', 'I']:
                        self.process_metadata_typestring(element.strip())

        # Process classes that contain type_mapping (e.g. Struct and Enum)
        if decoder_class and hasattr(decoder_class, 'type_mapping') and decoder_class.type_mapping:

            if type_string[0] == '(':
                type_info["is_primitive_runtime"] = False

            for key, data_type in decoder_class.type_mapping:
                self.process_metadata_typestring(data_type)

        # Try to get superclass as actual decoding class if not root level 'ScaleType'
        if decoder_class and len(decoder_class.__mro__) > 1 and decoder_class.__mro__[1].__name__ != 'ScaleType':
            decoder_class = decoder_class.__mro__[1]

        if decoder_class:
            type_info['decoder_class'] = decoder_class.__name__

            if type_info["is_primitive_runtime"] is None:
                type_info["is_primitive_runtime"] = True

            if type_info["is_primitive_runtime"] and type_string.lower() in ScaleDecoder.PRIMITIVES:
                type_info["is_primitive_core"] = True
        else:
            type_info["is_primitive_runtime"] = None
            type_info["is_primitive_core"] = None

        self.type_registry_cache[self.runtime_version][type_string.lower()] = type_info

        return decoder_class

    def get_type_registry(self, block_hash=None):
        """
        Generates an exhaustive list of which RUST types exist in the runtime specified at given block_hash (or
        chaintip if block_hash is omitted)

        Parameters
        ----------
        block_hash: Chaintip will be used if block_hash is omitted

        Returns
        -------

        """
        self.init_runtime(block_hash=block_hash)

        if self.runtime_version not in self.type_registry_cache:

            for module in self.metadata_decoder.metadata.modules:

                # Storage backwards compt check
                if module.storage and isinstance(module.storage, list):
                    storage_functions = module.storage
                elif module.storage and isinstance(getattr(module.storage, 'value'), dict):
                    storage_functions = module.storage.items
                else:
                    storage_functions = []

                if len(module.calls or []) > 0:
                    for idx, call in enumerate(module.calls):
                        for arg in call.args:
                            self.process_metadata_typestring(arg.type)

                if len(module.events or []) > 0:
                    for event_index, event in enumerate(module.events):

                        for arg_index, arg in enumerate(event.args):
                            self.process_metadata_typestring(arg)

                if len(storage_functions) > 0:
                    for idx, storage in enumerate(storage_functions):

                        # Determine type
                        type_key1 = None
                        type_key2 = None
                        type_value = None

                        if storage.type.get('PlainType'):
                            type_value = storage.type.get('PlainType')

                        elif storage.type.get('MapType'):
                            type_key1 = storage.type['MapType'].get('key')
                            type_value = storage.type['MapType'].get('value')

                        elif storage.type.get('DoubleMapType'):
                            type_key1 = storage.type['DoubleMapType'].get('key1')
                            type_key2 = storage.type['DoubleMapType'].get('key2')
                            type_value = storage.type['DoubleMapType'].get('value')

                        self.process_metadata_typestring(type_value)

                        if type_key1:
                            self.process_metadata_typestring(type_key1)

                        if type_key2:
                            self.process_metadata_typestring(type_key2)

                if len(module.constants or []) > 0:
                    for idx, constant in enumerate(module.constants):

                        # Check if types already registered in database
                        self.process_metadata_typestring(constant.type)

        return self.type_registry_cache[self.runtime_version]

    def get_type_definition(self, type_string, block_hash=None):
        """
        Retrieves decoding specifications of given type_string

        Parameters
        ----------
        type_string: RUST variable type, e.g. Vec<Address>
        block_hash

        Returns
        -------

        """
        type_registry = self.get_type_registry(block_hash=block_hash)
        return type_registry.get(type_string.lower())

    def get_metadata_modules(self, block_hash=None):
        """
        Retrieves a list of modules in metadata for given block_hash (or chaintip if block_hash is omitted)

        Parameters
        ----------
        block_hash

        Returns
        -------

        """
        self.init_runtime(block_hash=block_hash)

        return [{
            'metadata_index': idx,
            'module_id': module.get_identifier(),
            'name': module.name,
            'prefix': module.prefix,
            'spec_version': self.runtime_version,
            'count_call_functions': len(module.calls or []),
            'count_storage_functions': len(module.calls or []),
            'count_events': len(module.events or []),
            'count_constants': len(module.constants or []),
            'count_errors': len(module.errors or []),
        } for idx, module in enumerate(self.metadata_decoder.metadata.modules)]

    def get_metadata_module(self, name, block_hash=None):
        """
        Retrieves modules in metadata by name for given block_hash (or chaintip if block_hash is omitted)

        Parameters
        ----------
        name
        block_hash

        Returns
        -------
        MetadataModule
        """
        self.init_runtime(block_hash=block_hash)

        for module in self.metadata_decoder.metadata.modules:
            if module.name == name:
                return module

    def get_metadata_call_functions(self, block_hash=None):
        """
        Retrieves a list of all call functions in metadata active for given block_hash (or chaintip if block_hash is omitted)

        Parameters
        ----------
        block_hash

        Returns
        -------

        """
        self.init_runtime(block_hash=block_hash)

        call_list = []

        for call_index, (module, call) in self.metadata_decoder.call_index.items():
            call_list.append(
                self.serialize_module_call(
                    module, call, self.runtime_version, call_index
                )
            )
        return call_list

    def get_metadata_call_function(self, module_name, call_function_name, block_hash=None):
        """
        Retrieves the details of a call function given module name, call function name and block_hash
        (or chaintip if block_hash is omitted)

        Parameters
        ----------
        module_name
        call_function_name
        block_hash

        Returns
        -------

        """
        self.init_runtime(block_hash=block_hash)

        for call_index, (module, call) in self.metadata_decoder.call_index.items():
            if module.name == module_name and \
                    call.get_identifier() == call_function_name:
                return call

    def get_metadata_events(self, block_hash=None):
        """
        Retrieves a list of all events in metadata active for given block_hash (or chaintip if block_hash is omitted)

        Parameters
        ----------
        block_hash

        Returns
        -------

        """

        self.init_runtime(block_hash=block_hash)

        event_list = []

        for event_index, (module, event) in self.metadata_decoder.event_index.items():
            event_list.append(
                self.serialize_module_event(
                    module, event, self.runtime_version, event_index
                )
            )

        return event_list

    def get_metadata_event(self, module_name, event_name, block_hash=None):
        """
        Retrieves the details of an event for given module name, call function name and block_hash
        (or chaintip if block_hash is omitted)

        Parameters
        ----------
        module_name
        event_name
        block_hash

        Returns
        -------

        """

        self.init_runtime(block_hash=block_hash)

        for event_index, (module, event) in self.metadata_decoder.event_index.items():
            if module.name == module_name and \
                    event.name == event_name:
                return event

    def get_metadata_constants(self, block_hash=None):
        """
        Retrieves a list of all constants in metadata active at given block_hash (or chaintip if block_hash is omitted)

        Parameters
        ----------
        block_hash

        Returns
        -------

        """

        self.init_runtime(block_hash=block_hash)

        constant_list = []

        for module_idx, module in enumerate(self.metadata_decoder.metadata.modules):
            for constant in module.constants or []:
                constant_list.append(
                    self.serialize_constant(
                        constant, module, self.runtime_version
                    )
                )

        return constant_list

    def get_metadata_constant(self, module_name, constant_name, block_hash=None):
        """
        Retrieves the details of a constant for given module name, call function name and block_hash
        (or chaintip if block_hash is omitted)

        Parameters
        ----------
        module_name
        constant_name
        block_hash

        Returns
        -------

        """

        self.init_runtime(block_hash=block_hash)

        for module_idx, module in enumerate(self.metadata_decoder.metadata.modules):

            if module_name == module.name and module.constants:

                for constant in module.constants:
                    if constant_name == constant.name:
                        return constant

    def get_metadata_storage_functions(self, block_hash=None):
        """
        Retrieves a list of all storage functions in metadata active at given block_hash (or chaintip if block_hash is
        omitted)

        Parameters
        ----------
        block_hash

        Returns
        -------

        """
        self.init_runtime(block_hash=block_hash)

        storage_list = []

        for module_idx, module in enumerate(self.metadata_decoder.metadata.modules):
            if module.storage:
                for storage in module.storage.items:
                    storage_list.append(
                        self.serialize_storage_item(
                            storage_item=storage,
                            module=module,
                            spec_version_id=self.runtime_version
                        )
                    )

        return storage_list

    def get_metadata_storage_function(self, module_name, storage_name, block_hash=None):
        """
        Retrieves the details of a storage function for given module name, call function name and block_hash

        Parameters
        ----------
        module_name
        storage_name
        block_hash

        Returns
        -------

        """
        self.init_runtime(block_hash=block_hash)

        for module_idx, module in enumerate(self.metadata_decoder.metadata.modules):
            if module.name == module_name and module.storage:
                for storage in module.storage.items:
                    if storage.name == storage_name:
                        return storage

    def get_metadata_errors(self, block_hash=None):
        """
        Retrieves a list of all errors in metadata active at given block_hash (or chaintip if block_hash is omitted)

        Parameters
        ----------
        block_hash

        Returns
        -------

        """
        self.init_runtime(block_hash=block_hash)

        error_list = []

        for module_idx, module in enumerate(self.metadata_decoder.metadata.modules):
            if module.errors:
                for error in module.errors:
                    error_list.append(
                        self.serialize_module_error(
                            module=module, error=error, spec_version=self.runtime_version
                        )
                    )

        return error_list

    def get_metadata_error(self, module_name, error_name, block_hash=None):
        """
        Retrieves the details of an error for given module name, call function name and block_hash

        Parameters
        ----------
        module_name
        error_name
        block_hash

        Returns
        -------

        """
        self.init_runtime(block_hash=block_hash)

        for module_idx, module in enumerate(self.metadata_decoder.metadata.modules):
            if module.name == module_name and module.errors:
                for error in module.errors:
                    if error_name == error.name:
                        return error

    def get_runtime_block(self, block_hash=None, block_id=None, ignore_decoding_errors=False):
        """
        Retrieves a block with method `chain_getBlock` and in addition decodes extrinsics and log items

        Parameters
        ----------
        block_hash
        block_id
        ignore_decoding_errors: When True no exception will be raised if decoding of extrinsics failes and add as `None` instead

        Returns
        -------

        """
        self.init_runtime(block_hash=block_hash, block_id=block_id)

        response = self.rpc_request("chain_getBlock", [self.block_hash])

        if 'error' in response:
            raise SubstrateRequestException(response['error']['message'])

        response = response.get('result')

        if response:
            response['block']['header']['number'] = int(response['block']['header']['number'], 16)

            for idx, extrinsic_data in enumerate(response['block']['extrinsics']):
                extrinsic_decoder = Extrinsic(
                    data=ScaleBytes(extrinsic_data),
                    metadata=self.metadata_decoder,
                    runtime_config=self.runtime_config
                )
                try:
                    extrinsic_decoder.decode()
                    response['block']['extrinsics'][idx] = extrinsic_decoder.value

                except (RemainingScaleBytesNotEmptyException, KeyError, IndexError, ValueError):
                    if not ignore_decoding_errors:
                        raise
                    response['block']['extrinsics'][idx] = None

            for idx, log_data in enumerate(response['block']['header']["digest"]["logs"]):
                log_digest = LogDigest(ScaleBytes(log_data), runtime_config=self.runtime_config)
                try:
                    log_digest.decode()
                    response['block']['header']["digest"]["logs"][idx] = log_digest.value

                except (RemainingScaleBytesNotEmptyException, KeyError, IndexError, ValueError):
                    if not ignore_decoding_errors:
                        raise
                    response['block']['header']["digest"]["logs"][idx] = None

        return response

    def get_block_extrinsics(self, block_hash: str = None, block_id: int = None, ignore_decoding_errors=False) -> list:
        """
        Retrieves a list of `Extrinsic` objects for given block_hash or block_id

        Parameters
        ----------
        block_hash
        block_id
        ignore_decoding_errors: When True no exception will be raised if decoding of extrinsics failes and add as `None` instead

        Returns
        -------
        list
        """
        self.init_runtime(block_hash=block_hash, block_id=block_id)

        response = self.rpc_request("chain_getBlock", [self.block_hash])

        if 'error' in response:
            raise SubstrateRequestException(response['error']['message'])

        if response.get('result') is None:
            raise BlockHashNotFound(f"{block_hash} not found")

        extrinsics = []

        for extrinsic_data in response['result']['block']['extrinsics']:
            extrinsic = Extrinsic(
                data=ScaleBytes(extrinsic_data),
                metadata=self.metadata_decoder,
                runtime_config=self.runtime_config
            )
            try:
                extrinsic.decode()
            except (RemainingScaleBytesNotEmptyException, KeyError, IndexError, ValueError):
                if not ignore_decoding_errors:
                    raise
                extrinsic = None

            extrinsics.append(extrinsic)

        return extrinsics

    def decode_scale(self, type_string, scale_bytes, block_hash=None, return_scale_obj=False):
        """
        Helper function to decode arbitrary SCALE-bytes (e.g. 0x02000000) according to given RUST type_string
        (e.g. BlockNumber). The relevant versioning information of the type (if defined) will be applied if block_hash
        is set

        Parameters
        ----------
        type_string
        scale_bytes
        block_hash
        return_scale_obj: if True the SCALE object itself is returned, otherwise the serialized dict value of the object

        Returns
        -------

        """
        self.init_runtime(block_hash=block_hash)

        if type(scale_bytes) == str:
            scale_bytes = ScaleBytes(scale_bytes)

        obj = ScaleDecoder.get_decoder_class(
            type_string=type_string,
            data=scale_bytes,
            metadata=self.metadata_decoder,
            runtime_config=self.runtime_config
        )

        obj.decode()

        if return_scale_obj:
            return obj
        else:
            return obj.value

    def encode_scale(self, type_string, value, block_hash=None):
        """
        Helper function to encode arbitrary data into SCALE-bytes for given RUST type_string

        Parameters
        ----------
        type_string
        value
        block_hash

        Returns
        -------

        """
        self.init_runtime(block_hash=block_hash)

        obj = ScaleDecoder.get_decoder_class(
            type_string=type_string, metadata=self.metadata_decoder, runtime_config=self.runtime_config
        )
        return obj.encode(value)

    def ss58_encode(self, public_key: str) -> str:
        """
        Helper function to encode a public key to SS58 address

        Parameters
        ----------
        public_key

        Returns
        -------
        SS58 address
        """
        return ss58_encode(public_key, ss58_format=self.ss58_format)

    def ss58_decode(self, ss58_address: str) -> str:
        """
        Helper function to decode a SS58 address to a public key

        Parameters
        ----------
        ss58_address

        Returns
        -------
        Public key
        """
        return ss58_decode(ss58_address, valid_ss58_format=self.ss58_format)

    def is_valid_ss58_address(self, value: str) -> bool:
        """
        Helper function to validate given value as ss58_address for current network/ss58_format

        Parameters
        ----------
        value

        Returns
        -------
        bool
        """
        return is_valid_ss58_address(value, valid_ss58_format=self.ss58_format)

    # Serializing helper function

    def serialize_storage_item(self, storage_item, module, spec_version_id):
        """
        Helper function to serialize a storage item

        Parameters
        ----------
        storage_item
        module
        spec_version_id

        Returns
        -------

        """
        storage_dict = {
            "storage_name": storage_item.name,
            "storage_modifier": storage_item.modifier,
            "storage_fallback_scale": storage_item.fallback,
            "storage_fallback": None,
            "documentation": '\n'.join(storage_item.docs),
            "module_id": module.get_identifier(),
            "module_prefix": module.prefix,
            "module_name": module.name,
            "spec_version": spec_version_id,
            "type_key1": None,
            "type_key2": None,
            "type_hasher_key1": None,
            "type_hasher_key2": None,
            "type_value": None,
            "type_is_linked": None
        }

        type_class, type_info = next(iter(storage_item.type.items()))

        storage_dict["type_class"] = type_class

        if type_class == 'PlainType':
            storage_dict["type_value"] = type_info

        elif type_class == 'MapType':
            storage_dict["type_value"] = type_info["value"]
            storage_dict["type_key1"] = type_info["key"]
            storage_dict["type_hasher_key1"] = type_info["hasher"]
            storage_dict["type_is_linked"] = type_info["isLinked"]

        elif type_class == 'DoubleMapType':

            storage_dict["type_value"] = type_info["value"]
            storage_dict["type_key1"] = type_info["key1"]
            storage_dict["type_key2"] = type_info["key2"]
            storage_dict["type_hasher_key1"] = type_info["hasher"]
            storage_dict["type_hasher_key2"] = type_info["key2Hasher"]

        if storage_item.fallback != '0x00':
            # Decode fallback
            try:
                fallback_obj = ScaleDecoder.get_decoder_class(
                    type_string=storage_dict["type_value"],
                    data=ScaleBytes(storage_item.fallback),
                    runtime_config=self.runtime_config
                )
                storage_dict["storage_fallback"] = fallback_obj.decode()
            except Exception:
                storage_dict["storage_fallback"] = '[decoding error]'

        return storage_dict

    def serialize_constant(self, constant, module, spec_version_id):
        """
        Helper function to serialize a constant

        Parameters
        ----------
        constant
        module
        spec_version_id

        Returns
        -------

        """
        try:
            value_obj = ScaleDecoder.get_decoder_class(
                type_string=constant.type, data=ScaleBytes(constant.constant_value), runtime_config=self.runtime_config
            )
            constant_decoded_value = value_obj.decode()
        except Exception:
            constant_decoded_value = '[decoding error]'

        return {
            "constant_name": constant.name,
            "constant_type": constant.type,
            "constant_value": constant_decoded_value,
            "constant_value_scale": constant.constant_value,
            "documentation": '\n'.join(constant.docs),
            "module_id": module.get_identifier(),
            "module_prefix": module.prefix,
            "module_name": module.name,
            "spec_version": spec_version_id
        }

    def serialize_module_call(self, module, call, spec_version, call_index):
        """
        Helper function to serialize a call function

        Parameters
        ----------
        module
        call
        spec_version
        call_index

        Returns
        -------

        """
        return {
            "call_id": call.get_identifier(),
            "call_name": call.name,
            "call_args": [call_arg.value for call_arg in call.args],
            "lookup": '0x{}'.format(call_index),
            "documentation": '\n'.join(call.docs),
            "module_id": module.get_identifier(),
            "module_prefix": module.prefix,
            "module_name": module.name,
            "spec_version": spec_version
        }

    def serialize_module_event(self, module, event, spec_version, event_index):
        """
        Helper function to serialize an event

        Parameters
        ----------
        module
        event
        spec_version
        event_index

        Returns
        -------

        """
        return {
            "event_id": event.name,
            "event_name": event.name,
            "event_args": [
                  {
                    "event_arg_index": idx,
                    "type": arg
                  } for idx, arg in enumerate(event.args)
                ],
            "lookup": '0x{}'.format(event_index),
            "documentation": '\n'.join(event.docs),
            "module_id": module.get_identifier(),
            "module_prefix": module.prefix,
            "module_name": module.name,
            "spec_version": spec_version
        }

    def serialize_module_error(self, module, error, spec_version):
        """
        Helper function to serialize an error

        Parameters
        ----------
        module
        error
        spec_version

        Returns
        -------

        """
        return {
            "error_name": error.name,
            "documentation": '\n'.join(error.docs),
            "module_id": module.get_identifier(),
            "module_prefix": module.prefix,
            "module_name": module.name,
            "spec_version": spec_version
        }

    def update_type_registry_presets(self):
        try:
            update_type_registries()
            self.reload_type_registry(use_remote_preset=False)
            return True
        except Exception:
            return False

    def reload_type_registry(self, use_remote_preset: bool = True):
        """
        Reload type registry and preset used to instantiate the SubtrateInterface object. Useful to periodically apply
        changes in type definitions when a runtime upgrade occurred

        Parameters
        ----------
        use_remote_preset: When True preset is downloaded from Github master, otherwise use files from local installed scalecodec package

        Returns
        -------

        """
        self.runtime_config.clear_type_registry()

        if self.type_registry_preset:
            # Load type registry according to preset
            type_registry_preset_dict = load_type_registry_preset(
                name=self.type_registry_preset, use_remote_preset=use_remote_preset
            )

            if not type_registry_preset_dict:
                raise ValueError(f"Type registry preset '{self.type_registry_preset}' not found")
        else:
            # Try to auto discover type registry preset by chain name
            type_registry_preset_dict = load_type_registry_preset(self.chain.lower())

            if not type_registry_preset_dict:
                raise ValueError(f"Could not auto-detect type registry preset for chain '{self.chain}'")

            self.debug_message(f"Auto set type_registry_preset to {self.chain.lower()} ...")
            self.type_registry_preset = self.chain.lower()

        if type_registry_preset_dict:
            # Load type registries in runtime configuration
            self.runtime_config.update_type_registry(
                load_type_registry_preset("default", use_remote_preset=use_remote_preset)
            )

            if self.type_registry_preset != "default":
                self.runtime_config.update_type_registry(type_registry_preset_dict)

        if self.type_registry:
            # Load type registries in runtime configuration
            self.runtime_config.update_type_registry(self.type_registry)


class ExtrinsicReceipt:

    def __init__(self, substrate: SubstrateInterface, extrinsic_hash: str, block_hash: str = None, finalized=None):
        """
        Object containing information of submitted extrinsic. Block hash where extrinsic is included is required
        when retrieving triggered events or determine if extrinsic was succesfull

        Parameters
        ----------
        substrate
        extrinsic_hash
        block_hash
        finalized
        """
        self.substrate = substrate
        self.extrinsic_hash = extrinsic_hash
        self.block_hash = block_hash
        self.finalized = finalized

        self.__extrinsic_idx = None
        self.__extrinsic = None

        self.__triggered_events = None
        self.__is_success = None
        self.__error_message = None
        self.__weight = None
        self.__total_fee_amount = None

    def retrieve_extrinsic(self):
        if not self.block_hash:
            raise ValueError("ExtrinsicReceipt can't retrieve events because it's unknown which block_hash it is "
                             "included, manually set block_hash or use `wait_for_inclusion` when sending extrinsic")
        # Determine extrinsic idx

        extrinsics = self.substrate.get_block_extrinsics(block_hash=self.block_hash)

        if len(extrinsics) > 0:
            self.__extrinsic_idx = self.__get_extrinsic_index(
                block_extrinsics=extrinsics,
                extrinsic_hash=self.extrinsic_hash
            )

            self.__extrinsic = extrinsics[self.__extrinsic_idx]

    @property
    def extrinsic_idx(self) -> int:
        """
        Retrieves the index of this extrinsic in containing block

        Returns
        -------
        int
        """
        if self.__extrinsic_idx is None:
            self.retrieve_extrinsic()
        return self.__extrinsic_idx

    @property
    def extrinsic(self) -> Extrinsic:
        """
        Retrieves the `Extrinsic` subject of this receipt

        Returns
        -------
        Extrinsic
        """
        if self.__extrinsic is None:
            self.retrieve_extrinsic()
        return self.__extrinsic

    @property
    def triggered_events(self) -> list:
        """
        Gets triggered events for submitted extrinsic. block_hash where extrinsic is included is required, manually
        set block_hash or use `wait_for_inclusion` when submitting extrinsic

        Returns
        -------
        list
        """
        if self.__triggered_events is None:
            if not self.block_hash:
                raise ValueError("ExtrinsicReceipt can't retrieve events because it's unknown which block_hash it is "
                                 "included, manually set block_hash or use `wait_for_inclusion` when sending extrinsic")

            if self.extrinsic_idx is None:
                self.retrieve_extrinsic()

            self.__triggered_events = []

            for event in self.substrate.get_events(block_hash=self.block_hash):
                if event.extrinsic_idx == self.extrinsic_idx:
                    self.__triggered_events.append(event)

        return self.__triggered_events

    def process_events(self):
        if self.triggered_events:

            self.__total_fee_amount = 0

            for event in self.triggered_events:
                # Check events
                if event.event_module.name == 'System' and event.event.name == 'ExtrinsicSuccess':
                    self.__is_success = True
                    self.__error_message = None

                    for param in event.params:
                        if param['type'] == 'DispatchInfo':
                            self.__weight = param['value']['weight']

                elif event.event_module.name == 'System' and event.event.name == 'ExtrinsicFailed':
                    self.__is_success = False

                    for param in event.params:
                        if param['type'] == 'DispatchError':
                            if 'Module' in param['value']:
                                module_error = self.substrate.metadata_decoder.get_module_error(
                                    module_index=param['value']['Module']['index'],
                                    error_index=param['value']['Module']['error']
                                )
                                self.__error_message = {
                                    'type': 'Module',
                                    'name': module_error.name,
                                    'docs': module_error.docs
                                }
                            elif 'BadOrigin' in param['value']:
                                self.__error_message = {
                                    'type': 'System',
                                    'name': 'BadOrigin',
                                    'docs': 'Bad origin'
                                }
                            elif 'CannotLookup' in param['value']:
                                self.__error_message = {
                                    'type': 'System',
                                    'name': 'CannotLookup',
                                    'docs': 'Cannot lookup'
                                }
                            elif 'Other' in param['value']:
                                self.__error_message = {
                                    'type': 'System',
                                    'name': 'Other',
                                    'docs': 'Unspecified error occurred'
                                }

                        if param['type'] == 'DispatchInfo':
                            self.__weight = param['value']['weight']

                elif event.event_module.name == 'Treasury' and event.event.name == 'Deposit':
                    self.__total_fee_amount += event.params[0]['value']

                elif event.event_module.name == 'Balances' and event.event.name == 'Deposit':
                    self.__total_fee_amount += event.params[1]['value']

    @property
    def is_success(self) -> bool:
        """
        Returns `True` if `ExtrinsicSuccess` event is triggered, `False` in case of `ExtrinsicFailed`
        In case of False `error_message` will contain more details about the error


        Returns
        -------
        bool
        """
        if self.__is_success is None:
            self.process_events()

        return self.__is_success

    @property
    def error_message(self) -> Optional[dict]:
        """
        Returns the error message if the extrinsic failed in format e.g.:

        `{'type': 'System', 'name': 'BadOrigin', 'docs': 'Bad origin'}`

        Returns
        -------
        dict
        """
        if self.__error_message is None:
            if self.is_success:
                return None
            self.process_events()
        return self.__error_message

    @property
    def weight(self) -> int:
        """
        Contains the actual weight when executing this extrinsic

        Returns
        -------
        int
        """
        if self.__weight is None:
            self.process_events()
        return self.__weight

    @property
    def total_fee_amount(self) -> int:
        """
        Contains the total fee costs deducted when executing this extrinsic. This includes fee for the validator (
        (`Balances.Deposit` event) and the fee deposited for the treasury (`Treasury.Deposit` event)

        Returns
        -------
        int
        """
        if self.__total_fee_amount is None:
            self.process_events()
        return self.__total_fee_amount

    # Helper functions
    @staticmethod
    def __get_extrinsic_index(block_extrinsics: list, extrinsic_hash: str) -> int:
        """
        Returns the index of a provided extrinsic
        """
        for idx, extrinsic in enumerate(block_extrinsics):
            if extrinsic.extrinsic_hash == extrinsic_hash.replace('0x', ''):
                return idx
        raise ExtrinsicNotFound()

    # Backwards compatibility methods
    def __getitem__(self, item):
        return getattr(self, item)

    def __iter__(self):
        for item in self.__dict__.items():
            yield item

    def get(self, name):
        return self[name]
