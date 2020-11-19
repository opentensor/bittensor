
import asyncio
import bittensor
import math
import netaddr
import numpy
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
        r"""Initializes a new Metagraph subtensor interface.
        Args:
            config (bittensor.Config): 
                An bittensor config object.
            keypair (substrateinterface.Keypair): 
                An bittensor keys object.
        """
        self._config = config
        self.__keypair = keypair
        self.substrate = SubstrateInterface(
            url=self._config.session_settings.chain_endpoint,
            address_type=42,
            type_registry_preset='substrate-node-template',
            type_registry=custom_type_registry,
        )
<<<<<<< HEAD
        
        # Record of the last poll iteration.
=======
>>>>>>> 17e573b1164c6034b62724702fc09755da3ca859
        self._last_poll = -math.inf
        self._neurons = []
        self._ranks = None
        self._weights = None

    def _pollchain(self):
        r""" Polls the substrate chain for the current graph state.
        """
        # Pull weights vals and fill key'd map.
        weight_vals_map = {}
        weight_vals_list = self.substrate.iterate_map(
            module='SubtensorModule',
            storage_function='Weight_vals',
        )
        for (key, val) in weight_vals_list:
            weight_vals_map[key] = val
        del weight_vals_list

        # Pull weights keys and fill key'd map.
        weight_keys_map = {}
        weight_keys_list = self.substrate.iterate_map(
            module='SubtensorModule',
            storage_function='Weight_keys',
        )
        for (key, val) in weight_keys_list:
            weight_keys_map[key] = val
        del weight_keys_list

        # Pull Stake and fill key'd map.
        stake_map = {}
        stake_list = self.substrate.iterate_map(
            module='SubtensorModule',
            storage_function='Stake'
        )
        for (key, val) in stake_list:
            stake_map[key] = val
        del stake_list

        # Pull Emit and fill key'd map.
        last_emit_map = {}
        last_emit_list = self.substrate.iterate_map(
            module='SubtensorModule',
            storage_function='LastEmit'
        )
        for (key, val) in last_emit_list:
            last_emit_map[key] = val
        del last_emit_list

        # Pull Neurons and fill key'd map.
        neuron_meta_map = {}
        neuron_meta_list = self.substrate.iterate_map(
            module='SubtensorModule',
            storage_function='Neurons'
        )
        for (key, val) in neuron_meta_list:
            neuron_meta_map[key] = val
        del neuron_meta_list

<<<<<<< HEAD
        # Fill full neuron set
        # Contains all staked nodes + ranked nodes + weighted nodes
        neurons_set = set()
        for n_key in neuron_meta_map.keys():
            neurons_set.add(n_key)
        for n_key in stake_map.keys():
            neurons_set.add(n_key)
        for n_key in weight_keys_map.keys():
            for j_key in weight_keys_map[n_key]:
                neurons_set.add(j_key)
=======
        # Filter neurons.
        neuron_map = {}

        # Fill self.
        self_neuron_proto = bittensor_pb2.Neuron(
                version=bittensor.__version__,
                public_key=self.__keypair.public_key,
                address=self._config.session_settings.remote_ip,
                port=self._config.session_settings.axon_port
        )
        self_endpoint_key = str(self_neuron_proto.address) + str(self_neuron_proto.port)
        neuron_map[self_endpoint_key] = self_neuron_proto

        for n_meta in neuron_metadata:
            # Create a new bittensor_pb2.Neuron proto.
            public_key = n_meta[0]
>>>>>>> 17e573b1164c6034b62724702fc09755da3ca859

        # Filter neurons into active set.
        # Map from ip+port key to neuron meta.
        active_neurons_map = {}
        for n_key in list(neurons_set):

            # Filter neurons without metadata.
            neuron_meta = None
            if n_key not in neuron_meta_map:
                continue
            neuron_meta = neuron_meta_map[n_key]

            # Check if endpoint exists.
            ipstr = int_to_ip(neuron_meta['ip'])
            port = int(neuron_meta['port'])
            endpoint_key = str(ipstr) + str(port)
            neuron_proto = bittensor_pb2.Neuron(
                version=bittensor.__version__,
                public_key=n_key,
                address=ipstr,
                port=port
            )

            # Check if endpoint is taken.
            if endpoint_key in active_neurons_map:

                # Competitor.
                n_key_b = active_neurons_map[endpoint_key].public_key
                
                # Get last emit.
                last_emit_a = None
                if n_key in last_emit_map.keys():
                    last_emit_a = last_emit_map[n_key]

                # Get last emit.
                last_emit_b = None
                if n_key_b in last_emit_map.keys():
                    last_emit_b = last_emit_map[n_key]

                # Replace neuron in active set.
                if last_emit_a >= last_emit_b:
                    active_neurons_map[endpoint_key] = neuron_proto

<<<<<<< HEAD
            # Add neuron to active set.
            else:
                active_neurons_map[endpoint_key] = neuron_proto

        # Fill self.
        self_neuron_proto = bittensor_pb2.Neuron(
                version=bittensor.__version__,
                public_key=self.__keypair.public_key,
                address=self._config.session_settings.remote_ip,
                port=self._config.session_settings.axon_port
        )
        self_endpoint_key = str(self_neuron_proto.address) + str(self_neuron_proto.port)
        active_neurons_map[self_endpoint_key] = self_neuron_proto

        # Set self._neurons to active set.
        self._neurons = active_neurons_map.values()

        # Fill ranks map. Score for each in neuron set.
        ranks_numpy = numpy.zeros(len(self._neurons))
        for i, proto in enumerate(self._neurons):
            ranks_numpy[i] = ranks_map[proto.public_key]
        self._ranks = torch.Tensor(np_ranks)


        ranks_map = {}
        for n_key in list(self._neurons):
            ranks_map[n_key] = 0.0
        # Update ranks, fill weights.
        for n_key in list(neurons_set):
            n_stake = stake_map[n_key]
            n_wkeys = weight_keys_map[n_key]
            n_wvals = weight_vals_map[n_key]
            for (j_key, w_ij) in list(zip(n_wkeys, n_wvals)):
                ranks_map [j_key] += n_stake + w_ij

        # Fill weights double map.
        weights_double_map = {}
        for i_key in list(neurons_set):
            weights_double_map[i_key] = {}
            i_wkeys = weight_keys_map[i_key]
            i_wvals = weight_vals_map[i_key]
            for (j_key, w_ij) in list(zip(i_wkeys, i_wvals)):
                weights_double_map[i_key][j_key] = w_ij
        
        ranks_numpy = np.array(len(self._neurons))
        for i, proto in enumerate(self._neurons):
            ranks_numpy[i] = ranks_map[proto.public_key]
        self._ranks = torch.Tensor(np_ranks)

        weights_numpy = np.array( (len(self._neurons), len(self._neurons)) )
        for i, proto_i in enumerate(self._neurons):
            for j, proto_j in enumerate(self._neurons):
                if proto_j.public_key in weights_double_map[proto_i.public_key]:
                    weights_numpy[i, j] = weights_double_map[proto_i.public_key][proto_j.public_key]
        self._weights = torch.Tensor(weights_numpy)

        return self._neurons, self._weights, self._ranks
        

    def neurons (self, poll_every_seconds: int = 20) -> List[bittensor_pb2.Neuron]:
=======
    def neurons (self, poll_every_seconds: int = 15) -> List[bittensor_pb2.Neuron]:
>>>>>>> 17e573b1164c6034b62724702fc09755da3ca859
        if (time.time() - self._last_poll) > poll_every_seconds:
            self._last_poll = time.time()
            self._pollchain()
        return self._neurons

    def ranks (self, poll_every_seconds: int = 20) -> List[bittensor_pb2.Neuron]:
        if (time.time() - self._last_poll) > poll_every_seconds:
            self._last_poll = time.time()
            self._pollchain()
        return self._ranks

    def weights (self, poll_every_seconds: int = 20) -> List[bittensor_pb2.Neuron]:
        if (time.time() - self._last_poll) > poll_every_seconds:
            self._last_poll = time.time()
            self._pollchain()
        return self._weights
        
        
    def connect(self, timeout) -> bool:
        time_elapsed = 0
        while time_elapsed < timeout:
            time.sleep(1)
            time_elapsed += 1
            try:
                self.substrate.get_runtime_block()
                return True
            except Exception as e:
                logger.warn("Exception occured during connection: {}".format)
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