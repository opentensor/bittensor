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

from scalecodec.type_registry import load_type_registry_preset
from subtensorinterface import SubstrateInterface, Keypair, SubstrateRequestException
from test import settings


class CreateExtrinsicsTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.kusama_substrate = SubstrateInterface(
            url=settings.KUSAMA_NODE_URL,
            address_type=2,
            type_registry_preset='kusama'
        )

        cls.polkadot_substrate = SubstrateInterface(
            url=settings.POLKADOT_NODE_URL,
            address_type=0,
            type_registry_preset='polkadot'
        )

    def test_compatibility_polkadot_runtime(self):
        type_reg = load_type_registry_preset("polkadot")

        runtime_data = self.polkadot_substrate.rpc_request('state_getRuntimeVersion', [])
        self.assertLessEqual(
            runtime_data['result']['specVersion'], type_reg.get('runtime_id'), 'Current runtime is incompatible'
        )

    def test_compatibility_kusama_runtime(self):
        type_reg = load_type_registry_preset("kusama")

        runtime_data = self.polkadot_substrate.rpc_request('state_getRuntimeVersion', [])
        self.assertLessEqual(
            runtime_data['result']['specVersion'], type_reg.get('runtime_id'), 'Current runtime is incompatible'
        )

    def test_create_balance_transfer(self):
        # Create new keypair
        mnemonic = Keypair.generate_mnemonic()
        keypair = Keypair.create_from_mnemonic(mnemonic, address_type=2)

        for substrate in [self.kusama_substrate, self.polkadot_substrate]:

            # Create balance transfer call
            call = substrate.compose_call(
                call_module='Balances',
                call_function='transfer',
                call_params={
                    'dest': 'EaG2CRhJWPb7qmdcJvy3LiWdh26Jreu9Dx6R1rXxPmYXoDk',
                    'value': 2 * 10 ** 3
                }
            )

            extrinsic = substrate.create_signed_extrinsic(call=call, keypair=keypair)

            self.assertEqual(extrinsic.address.value, keypair.public_key)
            self.assertEqual(extrinsic.call_module.name, 'Balances')
            self.assertEqual(extrinsic.call.name, 'transfer')

            # Randomly created account should always have 0 nonce, otherwise account already exists
            self.assertEqual(extrinsic.nonce.value, 0)

            try:
                substrate.submit_extrinsic(extrinsic)

                self.fail('Should raise no funds to pay fees exception')

            except SubstrateRequestException as e:
                # Extrinsic should be successful if account had balance, eitherwise 'Bad proof' error should be raised
                self.assertEqual(e.args[0]['data'], 'Inability to pay some fees (e.g. account balance too low)')

    def test_create_mortal_extrinsic(self):
        # Create new keypair
        mnemonic = Keypair.generate_mnemonic()
        keypair = Keypair.create_from_mnemonic(mnemonic, address_type=2)

        for substrate in [self.kusama_substrate, self.polkadot_substrate]:

            # Create balance transfer call
            call = substrate.compose_call(
                call_module='Balances',
                call_function='transfer',
                call_params={
                    'dest': 'EaG2CRhJWPb7qmdcJvy3LiWdh26Jreu9Dx6R1rXxPmYXoDk',
                    'value': 2 * 10 ** 3
                }
            )

            extrinsic = substrate.create_signed_extrinsic(call=call, keypair=keypair, era={'period': 64})

            try:
                substrate.submit_extrinsic(extrinsic)

                self.fail('Should raise no funds to pay fees exception')

            except SubstrateRequestException as e:
                # Extrinsic should be successful if account had balance, eitherwise 'Bad proof' error should be raised
                self.assertEqual(e.args[0]['data'], 'Inability to pay some fees (e.g. account balance too low)')

    def test_create_unsigned_extrinsic(self):

        call = self.kusama_substrate.compose_call(
            call_module='Timestamp',
            call_function='set',
            call_params={
                'now': 1602857508000,
            }
        )

        extrinsic = self.kusama_substrate.create_unsigned_extrinsic(call)
        self.assertEqual(str(extrinsic.data), '0x280402000ba09cc0317501')

    def test_payment_info(self):
        keypair = Keypair(ss58_address="EaG2CRhJWPb7qmdcJvy3LiWdh26Jreu9Dx6R1rXxPmYXoDk")

        call = self.kusama_substrate.compose_call(
            call_module='Balances',
            call_function='transfer',
            call_params={
                'dest': 'EaG2CRhJWPb7qmdcJvy3LiWdh26Jreu9Dx6R1rXxPmYXoDk',
                'value': 2 * 10 ** 3
            }
        )
        payment_info = self.kusama_substrate.get_payment_info(call=call, keypair=keypair)

        self.assertIn('class', payment_info)
        self.assertIn('partialFee', payment_info)
        self.assertIn('weight', payment_info)

        self.assertGreater(payment_info['partialFee'], 0)


if __name__ == '__main__':
    unittest.main()
