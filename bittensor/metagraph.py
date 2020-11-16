
import bittensor

from loguru import logger
from bittensor import bittensor_pb2
from substrateinterface import SubstrateInterface, Keypair
from typing import List

# Substrate custom type interface.
custom_type_registry = {
    "runtime_id": 2, 
    "types": {
            "NeuronMetadata": {
                    "type": "struct", 
                    "type_mapping": [["ip", "u128"], ["port", "u16"], ["ip_type", "u8"]]
                }
        }
}

# Helper functions for converting between IPs and integers.
def ip2int(addr):
    return struct.unpack("!I", socket.inet_aton(addr))[0]

def int2ip(addr):
    return socket.inet_ntoa(struct.pack("!I", addr))

class Metagraph():

    def __init__(self, config: bittensor.Config):
        """Initializes a new Metagraph subtensor interface.
        Args:
            config (bittensor.Config): An bittensor config object.
        """
        self.config = config
        self.substrate = SubstrateInterface(
            url=self.config.chain_endpoint,
            address_type=42,
            type_registry_preset='substrate-node-template',
            type_registry=custom_type_registry
        )

    def synapses (self) -> List[bittensor_pb2.Synapse]:
        neurons = self.substrate.iterate_map(
            module='SubtensorModule',
            storage_function='Neurons')
        synapses = []
        for synapse in neurons:
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
        params = {'ip': ip2int(self.config.remote_ip), 'port': self.config.axon_port, 'ip_type': 4}
        logger.info('Subscribe to chain endpoint {} with params {}', self.config.chain_endpoint, params)
        call = self.substrate.compose_call(
            call_module='SubtensorModule',
            call_function='subscribe',
            call_params=params,
        )
        extrinsic = self.substrate.create_signed_extrinsic(call=call, keypair=self.__keypair)
        self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)

    def unsubscribe (self):
        call = self.substrate.compose_call(
                call_module='SubtensorModule',
                call_function='unsubscribe'
        )
        extrinsic = self.substrate.create_signed_extrinsic(call=call, keypair=self.__keypair)
        self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)
    
