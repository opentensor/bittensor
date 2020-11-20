
import asyncio
import collections
import bittensor
import math
import netaddr
import numpy
import time
<<<<<<< HEAD
import torch
=======
import threading
>>>>>>> 6f0025b79c01083b6d75d504d7eadd3107bfd21b

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

        # Record of last time a chain poll was called.
        self._last_poll = -math.inf

<<<<<<< HEAD
        # Map from neuron pubkey -> neuron index
        self._pubkey_index_map = {}

        # Number of neruons in graph.
        self._n = 0
=======
        # Thread variables
        self.chain_polling_thread = None
        self.poll_every_seconds = 15
        self._running = False

    def _pollchain(self):
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
>>>>>>> 6f0025b79c01083b6d75d504d7eadd3107bfd21b

        # List of bittensor_pb2.Neurons ordered by index
        self._neurons_list = []

        # List of List of weight_keys ordered by index
        self._weight_keys = []
        self._weight_vals = []
        self._weights_torch = None

        # List of stake values ordered by index
        self._stake_list = []
        self._stake_torch = None

        # List of emit values ordered by index
        self._emit_list = []
        self._emit_torch = None

        # List of last poll ordered by index
        self._poll_list = []
        self._poll_torch = None

    def n (self) -> int:
        """ Returns the number of neurons in the network.

        Returns:
            n: (int): neuron count.
        """
        return self._n

    def neurons(self) -> List[bittensor_pb2.Neuron]:
        """ Returns the neurons information of each active in the network.

        Returns:
            neurons: (List[bittensor_pb2.Neuron]): neuron info ordered by index.
        """
        return self._neurons_list

    def emit (self) -> torch.Tensor:
        """ Returns the last block emit time of each active neuron in the network.

        Returns:
            emit: (:obj:`torch.LongTensor` of shape :obj:`(self.n)`): neuron emit block ordered by index.
        """
        return self._emit_torch

    def poll (self) -> torch.Tensor:
        """ Returns the metagraph poll block of each active neuron in the network.

        Returns:
            poll: (:obj:`torch.LongTensor` of shape :obj:`(self.n)`): neuron poll block ordered by index.
        """
        return self._poll_torch

    def stake (self) -> torch.Tensor:
        """ Returns the stake of each active neuron in the network.

        Returns:
            stake: (:obj:`torch.LongTensor` of shape :obj:`(self.n)`): neuron stake ordered by index.
        """
        return self._stake_torch

    def weights (self) -> torch.Tensor:
        """ Returns the stake of each active neuron in the network.

        Returns:
            weights: (:obj:`torch.FloatTensor` of shape :obj:`(self.n, self.n)`): neuron stake ordered by index.
        """
        return self._weights_torch

    def pollchain(self):
        """ Polls the chain state for information about peers.
        
        """
        current_block = self.substrate.get_block_number(None)
        emits = self.substrate.iterate_map(
            module='SubtensorModule',
            storage_function='LastEmit',
        )
        for (key, val) in emits:
            # Filter on stale.
            if (current_block - val) > self._config.session_settings.metagraph.stale_emit_limit:
                continue

            # Filter on recent poll.
            last_poll = self._poll_list[key] if key in self._pubkey_index_map else -math.inf
            if (current_block - last_poll) < self._config.session_settings.metagraph.poll_every_blocks:
                continue

            # Poll.
            self._pollpubkey(key)

        self._build_torch_tensors()

    def _pollpubkey(self, pubkey):
        """ Polls info from the chain for a specific pubkey.

        Function call updates or appends new information to the stake vectors. If the neuron pubkey
        does not exist in the active set we assign an new index in the state vectors otherwise pull
        info from the local pubkey -> index mapping.
        
        """
        current_block = self.substrate.get_block_number(None)
        if pubkey in self._pubkey_index_map:
            index = self._pubkey_index_map[0]
            append = False
        else:
            index = self._n
            self._n += 1
            self._pubkey_index_map[pubkey] = index
            append = True
        stake = self.substrate.get_runtime_state(
                    module='SubtensorModule',
                    storage_function='Stake',
                    params=[pubkey]
        )['result']
        emit = self.substrate.get_runtime_state(
                    module='SubtensorModule',
                    storage_function='LastEmit',
                    params=[pubkey]
        )['result']
        info = self.substrate.get_runtime_state(
                    module='SubtensorModule',
                    storage_function='Neurons',
                    params=[pubkey]
        )['result']
        w_keys = self.substrate.get_runtime_state(
                    module='SubtensorModule',
                    storage_function='WeightKeys',
                    params=[pubkey]
        )['result']
        w_vals = self.substrate.get_runtime_state(
                    module='SubtensorModule',
                    storage_function='WeightVals',
                    params=[pubkey]
        )['result']
        ipstr = int_to_ip(info['ip'])
        port = int(info['port'])
        neuron = bittensor_pb2.Neuron (
                version=bittensor.__version__,
                public_key=pubkey,
                address=ipstr,
                port=port
        )
        if append == False:
            self._neurons_list[index] = neuron
            self._stake_list[index] = int(stake)
            self._emit_list[index] = int(emit)
            self._weight_keys[index] = list(w_keys)
            self._weight_vals[index] = list(w_vals)
            self._poll_list[index] = current_block
        else:
            self._neurons_list.append( neuron )
            self._stake_list.append( int(stake) )
            self._emit_list.append( int(emit) )
            self._weight_keys.append( list(w_keys) )
            self._weight_vals.append( list(w_vals) )
            self._poll_list.append( current_block )


    def _build_torch_tensors(self):
        """ Builds torch objects from python polled state.

        """
        # Set torch tensors from weights.
        self._stake_torch = torch.Tensor(self._stake_list)
        self._emit_torch = torch.Tensor(self._emit_list)
        self._poll_torch = torch.Tensor(self._poll_list)
        
        # Fill weights
        weights_numpy = numpy.zeros( (self._n, self._n) )
        for index_i, (keys, vals) in enumerate(list(zip(self._weight_keys, self._weight_vals))):
            val_sum = sum(vals)
            for k, val in list(zip(keys, vals)):
                if k in self._pubkey_index_map:
                    index_j = self._pubkey_index_map[k]
                    weights_numpy[index_i, index_j] = float(val) / float(val_sum)
        self._weights_torch = torch.Tensor(weights_numpy)

    def connect(self, timeout = 10) -> bool:
        """ Attempts a connection to the chain by calling substrate.get_runtime_block until success.

<<<<<<< HEAD
        Args: 
            timeout: (int): connection will ttl after length timeout seconds.
        
        """
=======
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
        self._neurons = neurons_list

    def neurons (self) -> List[bittensor_pb2.Neuron]:
        return self._neurons
    
    def poll_chain(self, poll_every_seconds: int = 15):
        self._pollchain()
        while self._running:
            if (time.time() - self._last_poll) > poll_every_seconds:
                self._last_poll = time.time()
            time.sleep(poll_every_seconds)

    def connect(self, timeout) -> bool:
>>>>>>> 6f0025b79c01083b6d75d504d7eadd3107bfd21b
        time_elapsed = 0
        while time_elapsed < timeout:
            time.sleep(1)
            time_elapsed += 1
            try:
                self.substrate.get_runtime_block()
                return True
            except Exception as e:
                logger.warning("Exception occured during connection: {}".format(e))
                continue
        return False

    def subscribe (self, timeout) -> bool:
        """ Attempts a subscription to the chain by calling subscribe until the peer exists in the active set.

        Args: 
            timeout: (int): subscribe will ttl after length timeout seconds.
        
        """
        params = {'ip': ip_to_int(self._config.session_settings.remote_ip), 'port': self._config.session_settings.axon_port, 'ip_type': 4}

        call = self.substrate.compose_call(
            call_module='SubtensorModule',
            call_function='subscribe',
            call_params=params
        )
        extrinsic = self.substrate.create_signed_extrinsic(call=call, keypair=self.__keypair)
        self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)

        # set up chain polling thread
        self.chain_polling_thread = threading.Thread(target=self.poll_chain, args=(self.poll_every_seconds,), daemon=True)

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
                    self._running = True
                    logger.info("Polling chain for the first time...")
                    self._pollchain()
                    self.chain_polling_thread.start()
                    logger.info("Starting chain polling thread...")
                    return True
        return False
            

    def unsubscribe (self, timeout):
        """ Attempts an unsubscription from chain, checks that the peer info has been removed within timeout.

        Args: 
            timeout: (int): unsubscribe will ttl after length timeout seconds.
        
        """
        logger.info('Unsubscribe from chain endpoint')
        call = self.substrate.compose_call(
            call_module='SubtensorModule',
            call_function='unsubscribe'
        )
        extrinsic = self.substrate.create_signed_extrinsic(call=call, keypair=self.__keypair)
        self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)
        time_elapsed = 0

        if self.chain_polling_thread:
            self._running = False
            self.chain_polling_thread.join()

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