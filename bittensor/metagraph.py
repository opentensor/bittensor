
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
            type_registry=custom_type_registry,
        )

    def neurons (self) -> List[bittensor_pb2.Neuron]:
        
        # Get current block for filtering.
        current_block = self.substrate.get_block_number(None)

        # Pull the last emit data from all nodes.
        last_emit_data = self.substrate.iterate_map(
            module='SubtensorModule',
            storage_function='LastEmit'
        )
        last_emit_map = {}
        for el in last_emit_data:
            last_emit_map[el[0]] = int(el[1])

        # Pull all neuron metadata.
        neuron_metadata = self.substrate.iterate_map(
            module='SubtensorModule',
            storage_function='Neurons'
        )

        # Filter neurons.
        neuron_map = {}
        for n_meta in neuron_metadata:
            # Create a new bittensor_pb2.Neuron proto.
            public_key = n_meta[0]

            # Filter nodes based on neuron last emit.
            if current_block - int(last_emit_map[public_key]) > 100:
                continue

            # Create neuron proto.
            ipstr = int_to_ip(n_meta[1]['ip'])
            port = int(n_meta[1]['port'])
            neuron_proto = bittensor_pb2.Neuron(
                version=bittensor.__version__,
                public_key=public_key,
                address=ipstr,
                port=port
            )

            # Check overlap ip-port, take the most recent neuron.
            ip_port = ipstr + str(port)
            if ip_port not in neuron_map:
                neuron_map[ip_port] = neuron_proto
            else:
                public_key2 = neuron_map[ip_port].public_key
                last_emit_1 = last_emit_map[public_key]
                last_emit_2 = last_emit_map[public_key2]
                if last_emit_1 >= last_emit_2:
                    neuron_map[ip_port] = neuron_proto
                
        # Return list of non-filtered neurons.
        neurons_list = neuron_map.values()
        return neurons_list

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

        logger.info(params)
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