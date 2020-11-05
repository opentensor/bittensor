from substrateinterface import SubstrateInterface
from loguru import logger
import bittensor

class Subtensor:
    def __init__(self, config):
        self._config = config
        try:
            self.__substrate = SubstrateInterface(
                url=config.chain_endpoint,
                address_type=42,
                type_registry_preset='substrate-node-template'
            )
        except:
            self.__substrate = None
            logger.error('Failed to connect to substrate chain endpoing {}', config.chain_endpoint)
            logger.info('Continuing with local execution')

    def current_block(self):
        if self.__substrate == None:
            return ""
        try:
            result = self.__substrate.get_runtime_block()
            current_block = result['block']['header']['parentHash']
            return current_block
        except:
            return ""

    def height(self):
        if self.__substrate == None:
            return 0
        try:
            result = self.__substrate.get_runtime_block()
            height = result['block']['header']['number']
            return height
        except:
            return 0

    def get_balance(self, keypair):
        if self.__substrate == None:
            return 0.0
        try:
            balance_info = self.__substrate.get_runtime_state(
                module='System',
                storage_function='Account',
                params=[keypair.ss58_address]
            ).get('result')
            return balance_info['data']['free'] / 10**12
        except:
            return 0.0

    def setWeight(self, keypair, publdest, value):
        if self.__substrate == None:
            return
        try:
            call = self.__substrate.compose_call(
                call_module='SubtensorModule',
                call_function='set_weight',
                call_params={'dest': publdest, 'value': value}
            )
            extrinsic = self.__substrate.create_signed_extrinsic(call=call, keypair=keypair)
            self.__substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)
        except:
            pass

