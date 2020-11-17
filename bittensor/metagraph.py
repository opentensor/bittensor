
import asyncio
import bittensor
import netaddr
import time

from loguru import logger
from bittensor import bittensor_pb2
from substrateinterface import SubstrateInterface, Keypair
from typing import List

custom_type_registry = {
    "runtime_id": 2, 
    "types": {
            "NeuronMetadata": {
                    "type": "struct", 
                    "type_mapping": [["ip", "u128"], ["port", "u16"], ["ip_type", "u8"]]
                }
        }
}

def int_to_ip(int_val):
    return str(netaddr.IPAddress(int_val))
 
def ip_to_int(str_val):
    return int(netaddr.IPAddress(str_val))

class Metagraph():

    def __init__(self, config, keypair):
        """Initializes a new Metagraph subtensor interface.
        Args:
            config (bittensor.Config): An bittensor config object.
        """
        self._config = config
        self.__keypair = keypair
        self.substrate = SubstrateInterface(
            url=self._config.session_settings.chain_endpoint,
            address_type=42,
            type_registry_preset='substrate-node-template',
            type_registry=custom_type_registry
        )


    def synapses (self) -> List[bittensor_pb2.Synapse]:
        neurons = self.substrate.iterate_map(
            module='SubtensorModule',
            storage_function='Neurons'
        )
        # Add from list.
        synapses = []
        for synapse in neurons:
            if synapse[0] != self.__keypair.public_key:
                # Create a new bittensor_pb2.Synapse proto.
                synapse_proto = bittensor_pb2.Synapse(
                    version=bittensor.__version__,
                    neuron_key=synapse[0],
                    synapse_key=synapse[0],
                    address=int_to_ip(synapse[1]['ip']),
                    port=int(synapse[1]['port']),
                )
                synapses.append(synapse_proto)
        for syn in synapses:
            logger.info('synapse {}', syn)
        return synapses

    def connect(self, timeout) -> bool:
        time_elapsed = 0
        while time_elapsed < timeout:
            time.sleep(1)
            time_elapsed += 1
            try:
                self.substrate.get_runtime_block()
                return True
            except:
                continue
        return False

    def subscribe (self, timeout) -> bool:
        params = {'ip': ip_to_int(self._config.session_settings.remote_ip), 'port': self._config.session_settings.axon_port, 'ip_type': 4}
        call = self.substrate.compose_call(
            call_module='SubtensorModule',
            call_function='subscribe',
            call_params=params
        )
        extrinsic = self.substrate.create_signed_extrinsic(call=call, keypair=self.__keypair)
        self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)
        time_elapsed = 0
        while time_elapsed < timeout:
            time.sleep(1)
            time_elapsed += 1
            neurons = self.substrate.iterate_map(
                module='SubtensorModule',
                storage_function='Neurons'
            )
            for n in neurons:
                if n[0] == self.__keypair.public_key:
                    return True
                    break
        return False
            

    def unsubscribe (self, timeout):
        logger.info('Unsubscribe from chain endpoint')
        call = self.substrate.compose_call(
            call_module='SubtensorModule',
            call_function='unsubscribe'
        )
        extrinsic = self.substrate.create_signed_extrinsic(call=call, keypair=self.__keypair)
        self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)
        time_elapsed = 0
        while time_elapsed < timeout:
            neurons = self.substrate.iterate_map(
                module='SubtensorModule',
                storage_function='Neurons'
            )
            i_exist = False
            for n in neurons:
                if n[0] == self.__keypair.public_key:
                    i_exist = True
                    break
            if i_exist == False:
                return True
            time.sleep(1)
            time_elapsed += 1
        return False