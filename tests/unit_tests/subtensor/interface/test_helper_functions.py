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
from unittest.mock import MagicMock

from scalecodec import ScaleBytes, Bytes
from scalecodec.metadata import MetadataDecoder

from subtensorinterface import SubstrateInterface
from test.fixtures import metadata_v12_hex


class TestHelperFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.substrate = SubstrateInterface(url='dummy', address_type=42, type_registry_preset='kusama')
        metadata_decoder = MetadataDecoder(ScaleBytes(metadata_v12_hex))
        metadata_decoder.decode()
        cls.substrate.get_block_metadata = MagicMock(return_value=metadata_decoder)

        def mocked_request(method, params):
            if method == 'chain_getRuntimeVersion':
                return {
                    "jsonrpc": "2.0",
                    "result": {"specVersion": 2023},
                    "id": 1
                }

        cls.substrate.rpc_request = MagicMock(side_effect=mocked_request)

    def test_decode_scale(self):
        self.assertEqual(self.substrate.decode_scale('Compact<u32>', '0x08'), 2)

    def test_encode_scale(self):
        self.assertEqual(self.substrate.encode_scale('Compact<u32>', 3), '0x0c')

    def test_get_type_definition(self):
        self.assertDictEqual(self.substrate.get_type_definition('Bytes'), {
            'decoder_class': 'Bytes',
            'is_primitive_core': False,
            'is_primitive_runtime': True,
            'spec_version': 2023,
            'type_string': 'Bytes'}
        )

    def test_get_metadata_modules(self):
        for module in self.substrate.get_metadata_modules():
            self.assertIn('module_id', module)
            self.assertIn('name', module)
            self.assertEqual(module['spec_version'], 2023)

    def test_get_metadata_call_function(self):
        call_function = self.substrate.get_metadata_call_function("Balances", "transfer")
        self.assertEqual(call_function['module_name'], "Balances")
        self.assertEqual(call_function['call_name'], "transfer")
        self.assertEqual(call_function['spec_version'], 2023)

    def test_get_metadata_event(self):
        event = self.substrate.get_metadata_event("Balances", "Transfer")
        self.assertEqual(event['module_name'], "Balances")
        self.assertEqual(event['event_name'], "Transfer")
        self.assertEqual(event['spec_version'], 2023)

    def test_get_metadata_constant(self):
        constant = self.substrate.get_metadata_constant("System", "BlockHashCount")
        self.assertEqual(constant['module_name'], "System")
        self.assertEqual(constant['constant_name'], "BlockHashCount")
        self.assertEqual(constant['spec_version'], 2023)

    def test_get_metadata_storage_function(self):
        storage = self.substrate.get_metadata_storage_function("System", "Account")
        self.assertEqual(storage['module_name'], "System")
        self.assertEqual(storage['storage_name'], "Account")
        self.assertEqual(storage['spec_version'], 2023)

    def test_get_metadata_error(self):
        error = self.substrate.get_metadata_error("System", "InvalidSpecName")
        self.assertEqual(error['module_name'], "System")
        self.assertEqual(error['error_name'], "InvalidSpecName")
        self.assertEqual(error['spec_version'], 2023)


if __name__ == '__main__':
    unittest.main()
