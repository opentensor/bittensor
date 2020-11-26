
import asyncio
import copy
import argparse
import bittensor
import math
import netaddr
import numpy
import time
import threading
import torch
import traceback

from threading import Thread, Lock

from munch import Munch
from loguru import logger
from bittensor import bittensor_pb2
# from substrateinterface import SubstrateInterface, Keypair
from bittensor.subtensor import WSClient, Keypair
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
        # Protected vars 
        self._config = config
        self.__keypair = keypair

        # Client for talking to chain.
        self.subtensor_client = WSClient(self._config.metagraph.chain_endpoint, self.__keypair)

        # current chain block.
        # Initialized to -1 for NA.
        self._current_block= -1

        # block that we last called emit.
        # Initialized to -1 for never.
        self._last_emit_block = -1

        # Self neuron.
        ipstr = int_to_ip(self._config.axon.remote_ip)
        port = int(self._config.axon.port)
        self._neuron = bittensor_pb2.Neuron(
                version=bittensor.__version__,
                public_key=self.__keypair.public_key,
                address=ipstr,
                port=port
            )

        # Number of neurons in graph.
        # Self neuron is the first in the graph.
        self._n = 1

        # Unique integer uid for neurons
        # Self neuron attains first uid.
        self._next_uid = 1

        # Map from neuron pubkey -> neuron index
        # Add self neuron into initial position.
        self._pubkey_index_map = {self.__keypair.public_key: 0}

        # Map from neuron uid to index.
        # Add self neuron uid=0 index=0
        self._index_for_uid = {0: 0}

        # List of bittensor_pb2.Neuron protos ordered by index
        # Self neuron has index 0.
        self._neurons = [self._neuron]

        # List of unique ids at each index
        # Self neuron has uid 0.
        self._uids = [0]

        # List of stake values ordered by index
        # Initialize self neuron stake to zero. (to be filled)
        self._stake = [0]

        # List of emit values ordered by index
        # Initialize self last emit block time to never -1.
        self._emit = [-1]

        # List of last poll ordered by index
        # Initialize self last poll to never to never -1
        self._poll = [-1]

        # Lisf of local weights for this neuron.
        # Initialize initial self weight to 1 in position 0.
        # Weights will be extended as new peers are added to the graph.
        self._local_weight_pubkeys = [self.__keypair.public_key]
        self._local_weight_vals = [1]
        self._local_weights_have_changed = True

        # List of weight pubkeys and values on chain
        # Initialize with self neuron self weight at position 0.
        # TODO(const): initialize weights on chain with self loop.
        self._chain_weight_pubkeys  = [ self._local_weight_pubkeys ]
        self._chain_weight_vals = [ self._local_weight_vals ]

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        
        parser.add_argument('--metagraph.chain_endpoint', default='206.189.254.5:12345', type=str, 
                            help='chain endpoint.')

        parser.add_argument('--metagraph.polls_every_sec', default=25, type=int, 
                            help='Second until the next chain poll.')

        parser.add_argument('--metagraph.emit_every_n_blocks', default=100, type=int, 
                            help='Blocks until weight update is sent, if weights have changed..')

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
        r""" Returns the number of neurons in the network.

        Returns:
            n: (int): neuron count.
        """
        return self._n

    def block (self) -> int:
        r""" Returns the current chain block number.

        Returns:
            block: current chain block number.
        """
        return self._current_block

    def neurons(self) -> List[bittensor_pb2.Neuron]:
        r""" Returns deepcopied information for each active neuron in the network.

        Returns:
            neurons: (List[bittensor_pb2.Neuron]): neuron info ordered by index.
        """
        return copy.deepcopy(self._neurons)

    def neurons_for_uids(self, uids: torch.Tensor) -> List[bittensor_pb2.Neuron]:
        r""" Returns a list with deepcopied information for neurons for each uid.

        Args:
            uids: (torch.LongTensor)
                uids into neurons protos

        Returns:
            neurons: (List[bittensor_pb2.Neuron]): 
                neuron info ordered by passed uids.
        """
        neurons = []
        for uid in uids.tolist():
            if uid not in self._index_for_uid:
                raise ValueError('uid does not correspond to neuron in graph, with uid {}'.format(uid))
            idx = self._index_for_uid[uid]
            neurons.append(copy.deepcopy(self._neurons[idx]))
        return neurons

    def neurons_for_indices(self, indices: torch.Tensor) -> List[bittensor_pb2.Neuron]:
        r""" Returns a list deepcopied information for neurons at passed indices.

        Args:
            indices: (torch.LongTensor)
                indexs to pull from neuron list.

        Returns:
            neurons: (List[bittensor_pb2.Neuron]): 
                neuron deepcopied info of neurons ordered by passed indices.
        """
        neurons = []
        for idx in indices.tolist():
            if idx >= self._n:
                raise ValueError('index is greater than the number of neurons, with idx {} and n {}'.format(idx, self._n))
            neurons.append(copy.deepcopy(self._neurons[int(idx)]))
        return neurons

    def indices(self) -> torch.LongTensor:
        r""" Returns a new torch tensor with indices into neuron set.

        Returns:
            indices: (:obj:`torch.LongTensor` of shape :obj:`(self.n)`):
                indices of neurons.
        """
        indices = torch.tensor(range(self._n))
        return indices.type(torch.LongTensor)

    def uids(self) -> torch.LongTensor:
        r""" Returns a new torch tensor of unique integer ids for neurons.

        Returns:
            uids (:obj:`torch.LongTensor` of shape :obj:`(self.n)`): 
                unique ids for neurons ordered by index.
        """
        uids = torch.tensor(copy.deepcopy(self._uids))
        return uids.type(torch.LongTensor)

    def emit (self) -> torch.LongTensor:
        r""" Returns a new torch tensor with the last emit block of each active neuron.

        Returns:
            emit: (:obj:`torch.LongTensor` of shape :obj:`(self.n)`): 
                neuron emit block ordered by index.
        """
        emit = torch.tensor(copy.deepcopy(self._emit))
        return emit.type(torch.LongTensor)

    def poll (self) -> torch.LongTensor:
        r""" Returns a new torch tensor with the last metagraph poll block for each active neuron.

        Returns:
            poll: (:obj:`torch.LongTensor` of shape :obj:`(self.n)`): 
                neuron poll block ordered by index.
        """
        poll = torch.tensor(copy.deepcopy(self._poll))
        return poll.type(torch.LongTensor)

    def stake (self) -> torch.LongTensor:
        r""" Returns a new torch tensot with the stake of each active neuron.

        Returns:
            stake: (:obj:`torch.LongTensor` of shape :obj:`(self.n)`): 
                neuron stake ordered by index.
        """
        stake = torch.tensor(copy.deepcopy(self._stake))
        return stake.type(torch.LongTensor)

    def weights (self) -> torch.FloatTensor:
        r""" Returns a new torch tensor with weights from each neuron.

        Returns:
            weights: (:obj:`torch.FloatTensor` of shape :obj:`(self.n, self.n)`): 
                weights ordered by index.
        """
        # TODO(const): perhaps cache this value.
        weights_numpy = numpy.zeros( (self._n, self._n))
        for index_i, (pubkeys, vals) in enumerate(list(zip(self._chain_weight_pubkeys, self._chain_weight_vals))):
            val_sum = sum(vals)
            for k, val in list(zip(pubkeys, vals)):
                if k in self._pubkey_index_map:
                    index_j = self._pubkey_index_map[k]
                    weights_numpy[index_i, index_j] = float(val) / float(val_sum)
        weights = torch.Tensor(weights_numpy)
        return weights.type(torch.FloatTensor)

    def local_weights_for_uids (self, uids: torch.Tensor):
        r""" Returns the local neuron weights. Not equivalent to those on chain.
        Args:
            uids: (:obj:`torch.LongTensor` of shape :obj:`(-1)`):
                UIDs for neurons to return weights.
        Returns:
            weights: (:obj:`torch.FloatTensor` of shape :obj:`(-1)`):
                weight values for passed uids.
        """
        if len(uids.shape) != 1 and len(uids.shape) != 2:
            raise ValueError ('uids must have rank 1 or 2, got {}'.format(len(uids.shape)))
        uids = torch.flatten(uids)
        indices = self.uids_to_indices(uids)
        indices = indices.type(torch.LongTensor)
        weights = self.local_weights()
        weights = torch.gather(weights, 0, indices)
        return weights

    def uids_to_indices(self, uids: torch.Tensor):
        r"""Return the indices of passed uids

        Args:
            uids: (:obj:`torch.LongTensor` of shape :obj:`(-1)`):
                UIDs for indices

        Returns 
            indices: (:obj:`torch.LongTensor` of shape :obj:`(-1)`):
                returned indices for passed uids.
        """
        # Return full keys
        full_uids = self.uids()
        indices = torch.nonzero(uids[..., None] == full_uids)[:,1]
        if torch.numel(full_uids) != torch.numel(indices):
            raise ValueError('Passed uids are not subset of full indices, with passed {} and full {}'.format(uids, full_uids))
        return indices

    def local_weights(self) -> torch.FloatTensor:
        r""" Returns a new torch tensor with this neurons local weights.

        Returns:
            weights: (:obj:`torch.FloatTensor` of shape :obj:`(-1)`):
                locally stored neuron weights, not nessecarily equivalent to those on chain.
        """
        local_weights_numpy = numpy.zeros((self._n))
        for pubkey, val in list(zip(self._local_weight_pubkeys, self._local_weight_vals)):
            idx = self._pubkey_index_map[pubkey]
            local_weights_numpy[idx] = val
        local_weights = torch.Tensor(local_weights_numpy)
        return local_weights.type(torch.FloatTensor)

    def set_local_weights (self, uids: torch.Tensor, weights: torch.Tensor):
        r"""Sets local neurons weights.

        Args:
            uids: (:obj:`torch.LongTensor` of shape :obj:`(-1)`):
                UIDs which identify neurons.

            weights: (:obj:`torch.FloatTensor` of shape :obj:`(-1)`):
                normalized weight values. l1 norm must be unity.
        """
        if len(uids.shape) != 1 and len(uids.shape) != 2:
            raise ValueError ('uids must have rank 1 or 2, got {}'.format(len(uids.shape)))
        if len(weights.shape) != 1 and len(weights.shape) != 2:
            raise ValueError ('weights must have rank 1 or 2, got {}'.format(len(weights.shape)))
        uids = torch.flatten(uids)
        weights = torch.flatten(weights)
        if torch.numel(uids) != torch.numel(weights):
            raise ValueError ('uids and weights must have number of elements, got {} weights and {} uids'.format(torch.numel(weights), torch.numel(uids)))

        # Check l1 norm.
        l1_norm = torch.norm(weights, p='l1')
        if l1_norm == 1:
            raise ValueError ('Weight must sum to 1, got norm of {}'.format(l1_norm))

        # Set weights.
        for _, (uid, weight) in enumerate(list(zip(uids.tolist(), weights.tolist()))):
            if uid not in self._index_for_uid[uid]:
                raise ValueError('unique id not in uid set: {}', uid)
            idx = self._index_for_uid[uid]
            neuron = self._neurons[idx]
            self._local_weight_pubkeys.append(neuron.public_key)
            self._local_weight_vals.append(weight)

        # Set weights have changed. Only emit when weights have changed.
        self._weights_have_changed = True

    async def do_emit(self):

        # Sleep until: seconds = n_blocks * __blocktime__
        sec_to_sleep = (self._config.metagraph.emit_every_n_blocks * bittensor.__blocktime__)
        await asyncio.sleep(sec_to_sleep)
    
        # Check if weight update is nessecary.
        if not self._weights_have_changed:
            # Set weights on chain.
            logger.info("***** Emit and set weights *****")
            self._weights_have_changed = False
            current_block = await self.subtensor_client.get_current_block()
            self._current_block = current_block
            self._last_emit_block = current_block

            # Convert floats to ints with precision.
            u32_int_max = 4294967295 # max int values
            weight_vals_as_ints = []
            for val in self._local_weight_vals:
                int_val = int(float(val) * int(u32_int_max)) # convert to int representation.
                weight_vals_as_ints.append(int_val) # int weights sum to u32_int_max.

            # Try set weights.
            logger.info('-> {}', list(zip(self._local_weight_pubkeys, weight_vals_as_ints)))
            await self.subtensor_client.set_weights(self._local_weight_pubkeys, weight_vals_as_ints, self.__keypair)

        # Finally rerun emit loop.
        await self.do_emit()

    async def pollchain(self):
        logger.info("***** Doing a chain poll *****")
        current_block = await self.subtensor_client.get_current_block()
        self._current_block = current_block

        # Pull the last emit data from all nodes.
        emits = await self.subtensor_client.get_last_emit_data()

        for (pubkey, val) in emits:
            # Filter on stale.
            if (self._current_block - val) > self._config.metagraph.stale_emit_limit:
                continue

            # Filter on recent poll.
            last_poll = self._poll[self._pubkey_index_map[pubkey]] if pubkey in self._pubkey_index_map else -math.inf
            if (self._current_block - last_poll) < self._config.metagraph.re_poll_neuron_every_blocks:
                continue

            # Poll.
            await self._pollpubkey(pubkey)

        await asyncio.sleep(self._config.metagraph.polls_every_sec)
        await self.pollchain()

    async def _pollpubkey(self, pubkey):
        """ Polls info from the chain for a specific pubkey.

        Function call updates or appends new information to the stake vectors. If the neuron pubkey
        does not exist in the active set we assign an new index in the state vectors otherwise pull
        info from the local pubkey -> index mapping.

        """
        current_block = await self.subtensor_client.get_current_block()
        self._current_block = current_block
        if pubkey in self._pubkey_index_map:
            index = self._pubkey_index_map[pubkey]
            append = False
        else:
            index = self._n
            self._n += 1
            uid = self._next_uid
            self._next_uid += 1
            self._pubkey_index_map[pubkey] = index
            append = True

        try:
            stake = await self.subtensor_client.get_stake(pubkey)
            emit = await self.subtensor_client.get_last_emit_data(pubkey)
            info = await self.subtensor_client.neurons(pubkey)
            w_keys = await self.subtensor_client.weight_keys(pubkey)
            w_vals = await self.subtensor_client.weight_vals(pubkey)

            #logger.info("Stake: {}", stake)
            #logger.info("Emit: {}", emit)
            #logger.info("Neurons: {}", info)
            #logger.info("Weight keys: {}", w_keys)
            #logger.info("Weight vals: {}", w_vals)

            ipstr = int_to_ip(info['ip'])
            port = int(info['port'])
            neuron = bittensor_pb2.Neuron(
                version=bittensor.__version__,
                public_key=pubkey,
                address=ipstr,
                port=port
            )

            if not append:
                self._neurons[index] = neuron
                self._stake[index] = int(stake)
                self._emit[index] = int(emit)
                self._chain_weight_pubkeys[index] = list(w_keys)
                self._chain_weight_vals[index] = list(w_vals)
                self._poll[index] = self._current_block
            else:
                self._neurons.append(neuron)
                self._stake.append(int(stake))
                self._emit.append(int(emit))
                self._chain_weight_pubkeys.append(list(w_keys))
                self._chain_weight_vals.append(list(w_vals))
                self._poll.append(self._current_block)
                self._uids.append( uid )
                self._index_for_uid[uid] = index

        except Exception as e:
            logger.error("Exception occurred: {}".format(e))
            traceback.print_exc()

    async def connect(self) -> bool:
        self.subtensor_client.connect()
        connected = await self.subtensor_client.is_connected()
        return connected

    async def subscribe (self, timeout) -> bool:
        await self.subtensor_client.subscribe(self._config.axon.remote_ip, self._config.axon.port)
        current_block = await self.subtensor_client.get_current_block()
        self._current_block = current_block
        self._last_emit_block = current_block

        time_elapsed = 0
        while time_elapsed < timeout:
            time.sleep(1)
            time_elapsed += 1
            neurons = await self.subtensor_client.neurons()
            for n in neurons:
                if n[0] == self.__keypair.public_key:
                    return True
        return False
            
    async def unsubscribe (self, timeout):
        logger.info('Unsubscribe from chain endpoint')
        await self.subtensor_client.unsubscribe()

