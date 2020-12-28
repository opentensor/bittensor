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

import unittest
import pytest
from unittest.mock import MagicMock

from scalecodec import ScaleBytes
from scalecodec.metadata import MetadataDecoder

from bittensor.subtensor.interface import SubstrateWSInterface
from .fixtures import metadata_v12_hex


class TestRuntimeState(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.substrate = SubstrateWSInterface(host='dummy', port=666, address_type=42, type_registry_preset='kusama')

    @pytest.mark.asyncio
    async def test_plaintype_call(self):

        def mocked_request(method, params):
            if method == 'chain_getRuntimeVersion':
                return {
                    "jsonrpc": "2.0",
                    "result": {"specVersion": 2023},
                    "id": 1
                }
            if method == 'state_getStorageAt':
                return {
                    "jsonrpc": "2.0",
                    "result": '0x0800000000000000482d7c0900000000020000000100000000000000000000000000020000',
                    "id": 1
                }

        metadata_decoder = MetadataDecoder(ScaleBytes(metadata_v12_hex))
        metadata_decoder.decode()
        self.substrate.get_block_metadata = MagicMock(return_value=metadata_decoder)
        self.substrate.rpc_request = MagicMock(side_effect=mocked_request)

        response = await self.substrate.get_runtime_state(
            module='System',
            storage_function='Events'
        )

        self.assertEqual(len(response['result']), 2)

        self.assertEqual(response['result'][0]['module_id'], 'System')
        self.assertEqual(response['result'][0]['event_id'], 'ExtrinsicSuccess')
        self.assertEqual(response['result'][1]['module_id'], 'System')
        self.assertEqual(response['result'][1]['event_id'], 'ExtrinsicSuccess')

    @pytest.mark.asyncio
    async def test_maptype_call(self):

        def mocked_request(method, params):
            if method == 'chain_getRuntimeVersion':
                return {
                    "jsonrpc": "2.0",
                    "result": {"specVersion": 2023},
                    "id": 1
                }
            elif method == 'state_getStorageAt':
                return {
                    'jsonrpc': '2.0',
                    'result': '0x00000000030000c16ff28623000000000000000000000000000000000000000000000000000000c16ff286230000000000000000000000c16ff28623000000000000000000',
                    'id': 1
                }

        self.substrate.rpc_request = MagicMock(side_effect=mocked_request)
        metadata_decoder = MetadataDecoder(ScaleBytes(metadata_v12_hex))
        metadata_decoder.decode()
        self.substrate.get_block_metadata = MagicMock(return_value=metadata_decoder)

        response = await self.substrate.get_runtime_state(
            module='System',
            storage_function='Account',
            params=['5GNJqTPyNqANBkUVMN1LPPrxXnFouWXoe2wNSmmEoLctxiZY']
        )

        self.assertEqual(response['result'], {
            'data':
                {
                    'feeFrozen': 10000000000000000,
                    'free': 10000000000000000,
                    'miscFrozen': 10000000000000000,
                    'reserved': 0
                },
                'nonce': 0,
                'refcount': 3
        })

    @pytest.mark.asyncio
    async def test_iterate_map(self):

        def mocked_request(method, params):
            if method == 'chain_getRuntimeVersion':
                return {
                    "jsonrpc": "2.0",
                    "result": {"specVersion": 2023},
                    "id": 1
                }
            elif method == 'state_getPairs':
                return {
                    "jsonrpc": "2.0",
                    "result": [
                        ['0x5f3e4907f716ac89b6347d15ececedca3ed14b45ed20d054f05e37e2542cfe70e535263148daaf49be5ddb1579b72e84524fc29e78609e3caf42e85aa118ebfe0b0ad404b5bdd25f',
                         '0xd43593c715fdd31c61141abd04a99fd6822c8558854ccde39a5684e7a56da27d']
                    ],
                    "id": 1
                }

        self.substrate.rpc_request = MagicMock(side_effect=mocked_request)
        metadata_decoder = MetadataDecoder(ScaleBytes(metadata_v12_hex))
        metadata_decoder.decode()
        self.substrate.get_block_metadata = MagicMock(return_value=metadata_decoder)

        all_bonded_stash_ctrls = self.substrate.iterate_map(
            module='Staking',
            storage_function='Bonded',
            block_hash='0x7d56e0ff8d3c57f77ea6a1eeef1cd2c0157a7b24d5a1af0f802ca242617922bf'
        )

        self.assertEqual(all_bonded_stash_ctrls, [[
            '0xbe5ddb1579b72e84524fc29e78609e3caf42e85aa118ebfe0b0ad404b5bdd25f',
            '0xd43593c715fdd31c61141abd04a99fd6822c8558854ccde39a5684e7a56da27d'
        ]])


if __name__ == '__main__':
    unittest.main()
