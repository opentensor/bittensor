from bittensor.subtensor.interface import SubstrateWSInterface, Keypair
import bittensor.utils.networking as net
from loguru import logger
from bittensor.balance import Balance
from .neurons import Neuron, Neurons

class WSClient:
    custom_type_registry = {
        "runtime_id": 2,
        "types": {
            "NeuronMetadataOf": {
                "type": "struct",
                "type_mapping": [["ip", "u128"], ["port", "u16"], ["ip_type", "u8"], ["uid", "u64"], ["modality", "u8"], ["hotkey", "AccountId"], ["coldkey", "AccountId"]]
            }
        }
    }


    def __init__(self, socket : str, keypair: Keypair):
        host, port = socket.split(":")

        self.substrate = SubstrateWSInterface(
            host=host,
            port=int(port),
            address_type = 42,
            type_registry_preset='substrate-node-template',
            type_registry=self.custom_type_registry,
        )

        self.__keypair = keypair

    '''
    PUBLIC METHODS
    '''

    def connect(self):
        logger.trace("connect() C")
        self.substrate.connect()

    def is_connected(self):
        return self.substrate.is_connected()


    async def subscribe(self, ip: str, port: int, modality: int, coldkey: str):
        ip_as_int  = net.ip_to_int(ip)
        params = {
            'ip': ip_as_int,
            'port': port, 
            'ip_type': 4,
            'modality': modality,
            'coldkey': coldkey
        }

        call = await self.substrate.compose_call(
            call_module='SubtensorModule',
            call_function='subscribe',
            call_params=params
        )

        extrinsic = await self.substrate.create_signed_extrinsic(call=call, keypair=self.__keypair)
        await self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)  # Waiting for inclusion and other does not work



    async def get_balance(self, address):
        logger.debug("Getting balance for: {}", address)
        result  = await self.substrate.get_runtime_state(
            module='System',
            storage_function='Account',
            params=[address],
            block_hash=None
        )

        balance_info = result.get('result')
        if not balance_info:
            logger.debug("{} has no balance", address)
            return Balance(0)

        balance = balance_info['data']['free']
        logger.debug("{} has {} rao", address, balance)
        return Balance(balance)

    async def add_stake(self, amount : Balance, hotkey_id):
        logger.info("Adding stake of {} rao from coldkey {} to hotkey {}", amount.rao, self.__keypair.public_key, hotkey_id)
        call = await self.substrate.compose_call(
            call_module='SubtensorModule',
            call_function='add_stake',
            call_params={
                'hotkey': hotkey_id,
                'ammount_staked': amount.rao
            }
        )

        extrinsic = await self.substrate.create_signed_extrinsic(call=call, keypair=self.__keypair)
        result = await self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)
        return result

    async def transfer(self, dest:str, amount: Balance):
        logger.debug("Requesting transfer of {}, from {} to {}", amount.rao, self.__keypair.public_key, dest)
        call = await self.substrate.compose_call(
            call_module='Balances',
            call_function='transfer',
            call_params={
                'dest': dest, 
                'value': amount.rao
            }
        )
        extrinsic = await self.substrate.create_signed_extrinsic(call=call, keypair=self.__keypair)
        result = await self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)
        return result

    async def unstake(self, amount : Balance, hotkey_id):
        logger.debug("Requesting unstake of {} rao for hotkey: {} to coldkey: {}", amount.rao, hotkey_id, self.__keypair.public_key)
        call = await self.substrate.compose_call(
            call_module='SubtensorModule',
            call_function='remove_stake',
            call_params={'ammount_unstaked': amount.rao, 'hotkey': hotkey_id}
        )

        extrinsic = await self.substrate.create_signed_extrinsic(call=call, keypair=self.__keypair)
        result = await self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)
        return result

    # async def set_weights(self, destinations, values, keypair, wait_for_inclusion=False):
    #     call = await self.substrate.compose_call(
    #         call_module = 'SubtensorModule',
    #         call_function = 'set_weights',
    #         call_params = {'dests': destinations, 'weights': values}
    #     )
    #
    #     extrinsic = await self.substrate.create_signed_extrinsic(call=call, keypair=keypair)
    #     await self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=wait_for_inclusion)


    async def set_weights(self, destinations, values, wait_for_inclusion=False):
        call = await self.substrate.compose_call(
            call_module='SubtensorModule',
            call_function='set_weights',
            call_params = {'dests': destinations, 'weights': values}
        )
        extrinsic = await self.substrate.create_signed_extrinsic(call=call, keypair=self.__keypair)
        result = await self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=wait_for_inclusion)
        return result

    async def get_current_block(self):
        return await self.substrate.get_block_number(None)

    async def get_active(self) -> int:
        result = await self.substrate.iterate_map(
            module='SubtensorModule',
            storage_function='Active',
        )

        return result

    async def get_uid_for_pubkey(self, pubkey = str) -> int:
        result = await self.substrate.get_runtime_state(
            module='SubtensorModule',
            storage_function='Active',
            params=[pubkey]
        )

        if result['result'] is None:
            return None

        return int(result['result'])

    async def get_neuron_for_uid(self, uid:int):

        result = await self.substrate.get_runtime_state(
                module='SubtensorModule',
                storage_function='Neurons',
                params=[uid]
        )
        return result['result']
    
    async def neurons(self, uid=None, decorator=False):

        # Todo (parall4x, 23-12-2020) Get rid of this decorator flag. This should be refactored into something that returns Neuron objects only
        if uid:
            result = await self.substrate.get_runtime_state(
                module='SubtensorModule',
                storage_function='Neurons',
                params=[uid]
            )
            return Neurons.from_list(result['result']) if decorator else result['result']

        else:
            neurons = await self.substrate.iterate_map(
                module='SubtensorModule',
                storage_function='Neurons'
            )
            return Neurons.from_list(neurons) if decorator else neurons

    async def get_stake_for_uid(self, uid) -> Balance:
        stake = await self.substrate.get_runtime_state(
            module='SubtensorModule',
            storage_function='Stake',
            params = [uid]
        )

        result = stake['result']

        if not result:
            return Balance(0)

        return Balance(result)

    async def weight_uids_for_uid(self, uid):
        result = await self.substrate.get_runtime_state(
            module='SubtensorModule',
            storage_function='WeightUids',
            params = [uid]
        )
        return result['result']

    async def weight_vals_for_uid(self, uid):
        result = await self.substrate.get_runtime_state(
            module='SubtensorModule',
            storage_function='WeightVals',
            params = [uid]
        )
        return result['result']

    async def get_last_emit_data_for_uid(self, uid):
        result = await self.substrate.get_runtime_state(
            module='SubtensorModule',
            storage_function='LastEmit',
            params = [uid]
        )
        return result['result']

    async def get_last_emit_data(self):
        result = await self.substrate.iterate_map(
            module='SubtensorModule',
            storage_function='LastEmit'
        )
        return result



