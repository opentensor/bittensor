
import asyncio
import argparse
import bittensor
import math
import netaddr
import numpy
import time
import threading
import torch
import traceback

from munch import Munch
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
            url=self._config.metagraph.chain_endpoint,
            address_type=42,
            type_registry_preset='substrate-node-template',
            type_registry=custom_type_registry,
        )

        # Thread variables
        self._polling_thread = None
        self._last_poll = -math.inf
        self._running = False

        # Map from neuron pubkey -> neuron index
        self._pubkey_index_map = {}

        # Number of neurons in graph.
        self._n = 0

        # List of bittensor_pb2.Neurons ordered by index
        self._neurons_list = []

        # Unique integer key for neurons
        self._next_unique_key = 0
        self._keys_list = []
        self._keys_numpy: torch.LongTensor = None

        # List of List of weight_keys ordered by index
        self._weight_keys = []
        self._weight_vals = []
        self._weights_torch: torch.FloatTensor = None

        # List of stake values ordered by index
        self._stake_list = []
        self._stake_torch: torch.LongTensor = None

        # List of emit values ordered by index
        self._emit_list = []
        self._emit_torch: torch.LongTensor = None

        # List of last poll ordered by index
        self._poll_list = []
        self._poll_torch: torch.LongTensor = None

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        
        parser.add_argument('--metagraph.chain_endpoint', default='http://206.189.254.5:12345', type=str, 
                            help='chain endpoint.')

        parser.add_argument('--metagraph.polls_every_sec', default=25, type=int, 
                            help='Second until the next chain poll.')

        parser.add_argument('--metagraph.re_poll_neuron_every_blocks', default=20, type=int, 
                            help='Re poll info from neurons every n blocks.')

        parser.add_argument('--metagraph.stale_emit_limit', default=1000, 
                            help='Filter neurons with block time since emission greater than this value.')
        return parser

    @staticmethod   
    def check_config(config: Munch) -> Munch:
        assert config.metagraph.polls_every_sec > 5 and config.metagraph.polls_every_sec < 1000, 'metagraph.polls_every_sec must be in range [5, 1000]'
        assert config.metagraph.re_poll_neuron_every_blocks > 5 and config.metagraph.re_poll_neuron_every_blocks < 1000, 'metagraph.re_poll_neuron_every_blocks must be in range [5, 1000]'
        assert config.metagraph.stale_emit_limit > 1 and config.metagraph.re_poll_neuron_every_blocks < math.inf, 'metagraph.stale_emit_limit must be in range [1, inf]'
        return config

    def n (self) -> int:
        """ Returns the number of neurons in the network.

        Returns:
            n: (int): neuron count.
        """
        return self._n

    def neurons(self, keys: torch.LongTensor = None) -> List[bittensor_pb2.Neuron]:
        """ Returns the neurons information of each active in the network.

        Args:
            keys (bittensor.Config): 
                An bittensor config object.

        Returns:
            neurons: (List[bittensor_pb2.Neuron]): neuron info ordered by index.
        """
        return copy.deepcopy(self._neurons_list)

    def keys(self) -> torch.LongTensor:
        """ Returns a torch tensor of unique integer keys for neurons.

        Returns:
            keys (:obj:`torch.LongTensor` of shape :obj:`(self.n)`): unique keys for neurons.
        """
        return self._keys_torch

    def emit (self) -> torch.LongTensor:
        """ Returns the last block emit time of each active neuron in the network.

        Returns:
            emit: (:obj:`torch.LongTensor` of shape :obj:`(self.n)`): neuron emit block ordered by index.
        """
        return self._emit_torch

    def poll (self) -> torch.LongTensor:
        """ Returns the metagraph poll block of each active neuron in the network.

        Returns:
            poll: (:obj:`torch.LongTensor` of shape :obj:`(self.n)`): neuron poll block ordered by index.
        """
        return self._poll_torch

    def stake (self) -> torch.LongTensor:
        """ Returns the stake of each active neuron in the network.

        Returns:
            stake: (:obj:`torch.LongTensor` of shape :obj:`(self.n)`): neuron stake ordered by index.
        """
        return self._stake_torch

    def weights (self) -> torch.FloatTensor:
        """ Returns the stake of each active neuron in the network.

        Returns:
            weights: (:obj:`torch.FloatTensor` of shape :obj:`(self.n, self.n)`): neuron stake ordered by index.
        """
        return self._weights_torch

    def start(self):
        """ Start background polling thread.
        """
        logger.info('start')
        if self._running == False:
            self._running = True
            self._polling_thread = threading.Thread(target=self._continuous_poll, daemon=False)
            self._polling_thread.start()

    def stop(self):
        """ Stops background polling thread.
        """
        if self._running == True:
            self._running = False
            self._polling_thread.join()

    def _continuous_poll(self):
        """ continuously polls chain updating metagraph state until self._running is False
        """
        logger.info('_continuous_poll...')
        while self._running:
            if (time.time() - self._last_poll) > self._config.metagraph.polls_every_sec:
                logger.info('Pollchain...')
                self._last_poll = time.time()
                self.pollchain()
                logger.info('Done. ')
            time.sleep(self._config.metagraph.polls_every_sec/2)

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
            if (current_block - val) > self._config.metagraph.stale_emit_limit:
                continue

            # Filter on recent poll.
            last_poll = self._poll_list[self._pubkey_index_map[key]] if key in self._pubkey_index_map else -math.inf
            if (current_block - last_poll) < self._config.metagraph.re_poll_neuron_every_blocks:
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
        try:
            current_block = self.substrate.get_block_number(None)
            if pubkey in self._pubkey_index_map:
                index = self._pubkey_index_map[pubkey]
                append = False
            else:
                index = self._n
                key = self._next_unique_key + 1
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

            if not append:
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
                self._keys_list.append( key )

        except Exception as e:
            logger.error("Exception occurred: {}".format(e))
            traceback.print_exc()


    def _build_torch_tensors(self):
        """ Builds torch objects from python polled state.

        """
        # Set torch tensors from weights.
        self._stake_torch = torch.Tensor(self._stake_list, dtype=torch.long)
        self._emit_torch = torch.Tensor(self._emit_list, dtype=torch.long)
        self._poll_torch = torch.Tensor(self._poll_list, dtype=torch.long)
        self._keys_torch = torch.Tensor(self._keys_list, dtype=torch.long)
        
        # Fill weights
        weights_numpy = numpy.zeros( (self._n, self._n), dtype=torch.float)
        for index_i, (keys, vals) in enumerate(list(zip(self._weight_keys, self._weight_vals))):
            val_sum = sum(vals)
            for k, val in list(zip(keys, vals)):
                if k in self._pubkey_index_map:
                    index_j = self._pubkey_index_map[k]
                    weights_numpy[index_i, index_j] = float(val) / float(val_sum)
        self._weights_torch = torch.Tensor(weights_numpy)

    def connect(self, timeout = 10) -> bool:
        """ Attempts a connection to the chain by calling substrate.get_runtime_block until success.

        Args: 
            timeout: (int): connection will ttl after length timeout seconds.
        
        """
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
        """ Attempts a subscription to the chain by calling subscribe until the peer exists in the active set.

        Args: 
            timeout: (int): subscribe will ttl after length timeout seconds.
        
        """
        params = {'ip': ip_to_int(self._config.axon.remote_ip), 'port': self._config.axon.port, 'ip_type': 4}

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
        """ Attempts an unsubscription from chain, does not check that the peer info has been removed.       
        """
        logger.info('Unsubscribe from chain endpoint')
        call = self.substrate.compose_call(
            call_module='SubtensorModule',
            call_function='unsubscribe'
        )
        extrinsic = self.substrate.create_signed_extrinsic(call=call, keypair=self.__keypair)
        self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=False)
        return True