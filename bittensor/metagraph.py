
import asyncio
import bittensor
import struct 
import socket
import time

from loguru import logger
from bittensor import bittensor_pb2
from substrateinterface import SubstrateInterface, Keypair
from typing import List

# Helper functions for converting between IPs and integers.
def ip2int(addr):
    return struct.unpack("!I", socket.inet_aton(addr))[0]

def int2ip(addr):
    return socket.inet_ntoa(struct.pack("!I", addr))

custom_type_registry = {
    "runtime_id": 2, 
    "types": {
            "PeerMetadata": {
                    "type": "struct", 
                    "type_mapping": [["ip", "u128"], ["port", "u16"], ["ip_type", "u8"]]
                }
        }
}

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
        # Add myself.
        synapses = [ bittensor_pb2.Synapse(
                version=bittensor.__version__,
                neuron_key=self.__keypair.public_key,
                synapse_key=self.__keypair.public_key,
                address=self._config.session_settings.remote_ip,
                port=self._config.session_settings.axon_port,
        )]
        # Add from list.
        for synapse in neurons:
            if synapse[0] != self.__keypair.public_key:
                # Create a new bittensor_pb2.Synapse proto.
                synapse_proto = bittensor_pb2.Synapse(
                    version=bittensor.__version__,
                    neuron_key=synapse[0],
                    synapse_key=synapse[0],
                    address=int2ip(synapse[1]['ip']),
                    port=int(synapse[1]['ip']),
                )
            synapses.append(synapse_proto)
        return synapses

    def subscribe (self):
        params = {'ip': ip2int(self._config.session_settings.remote_ip), 'port': self._config.session_settings.axon_port, 'ip_type': 4}
        call = self.substrate.compose_call(
            call_module='SubtensorModule',
            call_function='subscribe',
            call_params=params
        )
        extrinsic = self.substrate.create_signed_extrinsic(call=call, keypair=self.__keypair)
        self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)
        time_elapsed = 0
        is_not_subscribed = True
        while is_not_subscribed:
            time.sleep(3)
            logger.info('.')
            time_elapsed += 3
            neurons = self.substrate.iterate_map(
                module='SubtensorModule',
                storage_function='Neurons'
            )
            for n in neurons:
                if n[0] == self.__keypair.public_key:
                    is_not_subscribed = False
                    logger.info('Subscribed.')
                    break
            

    def unsubscribe (self):
        logger.info('Unsubscribe from chain endpoint')
        call = self.substrate.compose_call(
            call_module='SubtensorModule',
            call_function='unsubscribe'
        )
        extrinsic = self.substrate.create_signed_extrinsic(call=call, keypair=self.__keypair)
        self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)