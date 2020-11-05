from substrateinterface import SubstrateInterface

import bittensor

class Subtensor:
    def __init__(self, config):
        self._config = config
        self.__substrate = SubstrateInterface(
            url=config.chain_endpoint,
            address_type=42,
            type_registry_preset='substrate-node-template'
        )

    def current_block(self):
        result = self.__substrate.get_runtime_block()
        current_block = result['block']['header']['parentHash']
        return current_block

    def height(self):
        result = self.__substrate.get_runtime_block()
        height = result['block']['header']['number']
        return height

    def get_balance(self, keypair):
        balance_info = self.__substrate.get_runtime_state(
            module='System',
            storage_function='Account',
            params=[keypair.ss58_address]
        ).get('result')
        return balance_info['data']['free'] / 10**12

    def setWeight(self, keypair, publdest, value):
        call = self.__substrate.compose_call(
            call_module='SubtensorModule',
            call_function='set_weight',
            call_params={'dest': publdest, 'value': value}
        )
        extrinsic = self.__substrate.create_signed_extrinsic(call=call, keypair=keypair)
        self.__substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)

