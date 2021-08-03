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

import os
import sys

sys.path.append(os.path.abspath('../../py-scale-codec'))

import unittest
import pytest

from scalecodec.base import RuntimeConfiguration

from substrateinterface import SubstrateInterface
from .settings import *


class KusamaTypeRegistryTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.substrate = SubstrateInterface(
            address_type=2,
            type_registry_preset='kusama',
            url='test.kusanagi.bittensor.com:9944'
        )

    @pytest.mark.asyncio
    async def test_type_registry_compatibility(self):
        with self.substrate as substrate:
            for scale_type in await substrate.get_type_registry():
                obj = RuntimeConfiguration().get_decoder_class(scale_type)
                self.assertIsNotNone(obj, '{} not supported'.format(scale_type))


class PolkadotTypeRegistryTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.substrate = SubstrateInterface(
            address_type=0,
            type_registry_preset='polkadot',
            url='test.kusanagi.bittensor.com:9944'
        )

    @pytest.mark.asyncio
    async def test_type_registry_compatibility(self):
        with self.substrate as substrate:
            for scale_type in await substrate.get_type_registry():
                obj = RuntimeConfiguration().get_decoder_class(scale_type)
                self.assertIsNotNone(obj, '{} not supported'.format(scale_type))


if __name__ == '__main__':
    unittest.main()
