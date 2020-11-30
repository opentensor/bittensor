
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
from bittensor.subtensor import WSClient, Keypair
from typing import List

def int_to_ip(int_val):
    return str(netaddr.IPAddress(int_val))
 
def ip_to_int(str_val):
    return int(netaddr.IPAddress(str_val))

# Static network state object.
class TorchChainState():
    r""" Maintains the chain state as a torch object.
        Params:
            block: (int) state block number.

            uids: (:obj:`torch.LongTensor` of shape :obj:`(n)`):
                UIDs for each neuron ordered by index.
            
            indices: (:obj:`torch.LongTensor` of shape :obj:`(n)`):
                Index of neurons, range(n)

            stake: (:obj:`torch.LongTensor` of shape :obj:`(n)`):
                Stake balance for each neuron ordered by index.
                
            emit: (:obj:`torch.LongTensor` of shape :obj:`(n)`):
                Last emission call for each neuron ordered by index.

            weights: (:obj:`torch.FloatTensor` of shape :obj:`(n)`):
                This neuron's weights W[,:]

            W: (:obj:`torch.FloatTensor` of shape :obj:`(n, n)`):
                Full weight matrix on chain.

            neurons: (List[bittensor_pb2.Neuron]) 
                List of endpoints on the network.

    """
    def __init__(self):
        self.block = 0
        self.n = 0
        self.uids = torch.tensor([])
        self.indices = torch.tensor([])
        self.stake = torch.tensor([])
        self.emit = torch.tensor([])
        self.W = torch.tensor([[]])
        self.neurons = []

    @property
    def weights(self):
        r"""Return this neurons weights. W[0,:]
        Returns 
            weights: (:obj:`torch.FloatTensor` of shape :obj:`(n)`):
                returned indices for passed uids.
        """
        return self.W[0,:]

    def set_weights(self, weights):
        r"""Sets this neurons weights. W[0,:]
        Args: 
            weights: (:obj:`torch.FloatTensor` of shape :obj:`(n)`):
                weights to set in positions W[0,:]
        """
        if len(weights.tolist()) != self.n:
            raise ValueError('Trying to set weights with vector of incorrect length, got {}, require {}'.format(len(weights.tolist()),self.n))
        self.W[0,:] = weights

    def uids_to_indices(self, uids: torch.Tensor):
        r"""Return the indices of passed uids
        Args:
            uids: (:obj:`torch.LongTensor` of shape :obj:`(-1)`):
                UIDs for indices
        Returns 
            indices: (:obj:`torch.LongTensor` of shape :obj:`(-1)`):
                returned indices for passed uids.
        """
        indices = torch.nonzero(uids[..., None] == self.uids)[:,1]
        if torch.numel(uids) != torch.numel(indices):
            raise ValueError('Passed uids are not a subset of class.uids, with passed: {} and class.uids: {}'.format(uids, self.uids))
        return indices

    def uids_to_neurons(self, uids: torch.Tensor) -> List[bittensor_pb2.Neuron]:
        r""" Returns a list with neurons for each uid.
        Args:
            uids: (torch.LongTensor)
                uids into neurons protos
        Returns:
            neurons: (List[bittensor_pb2.Neuron]): 
                neuron info ordered by passed uids.
        """
        response = []
        indices = self.uids_to_indices(uids)
        for idx in indices.tolist():
            response.append(self.neurons[idx])
        return response

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

        # Self neuron.
        ipstr = int_to_ip(self._config.axon.remote_ip)
        port = int(self._config.axon.port)
        self._neuron = bittensor_pb2.Neuron(
                version=bittensor.__version__,
                public_key=self.__keypair.public_key,
                address=ipstr,
                port=port
        )

        # Keeps track of the last block we preformed a sync
        self.last_sync = 0

        # Local chain state buffer.
        self._n = 1
        self._next_uid = 1
        self._uids = [0]
        self._stake = [0]
        self._emit = [0]
        self._neuron_weights = [1]
        self._weight_pubkeys = [[self._neuron.public_key]]
        self._weight_vals = [[1]]
        self._neurons = [self._neuron]
        self._index_for_uid = {0: 0}
        self._index_for_pubkey = {self._neuron.public_key: 0}
        self._pubkey_for_index = {0: self._neuron.public_key}

        # Torch chain state.
        self.state = TorchChainState()
        self._update_state()

    def sync(self):
        r""" Synchronizes the local self.state with the chain state, sinking the trained weights and pulling 
        info from other peers. Ensures the self.state is in accordance with the state on chain at this block.
        """
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._do_sync())

    def chain_block(self):
        r""" Returns the current block on the chain.
        Returns:
            block: (int) block number on chain.
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._chain_block())

    async def _chain_block(self):
        r""" Async returns the current block on the chain.
        Returns:
            block: (int) block number on chain.
        """
        return await self.subtensor_client.get_current_block()

    async def _do_sync(self):
        r""" Async: Synchronizes the local self.state with the chain state
        """
        logger.info('do sync')
        await self._emit_state()
        await self._poll_state()
        self.last_sync = await self._chain_block()
        self._update_state()

    def _update_state(self):
        r""" Deep copies the local state cache i.e. self._stake into the state object.
        """
        state = TorchChainState()
        state.n = self._n
        state.block = self.last_sync
        state.neurons = copy.deepcopy(self._neurons)
        state.indices = torch.tensor(range(state.n), dtype=torch.int64)
        state.uids = torch.tensor(copy.deepcopy(self._uids), dtype=torch.int64)
        state.emit = torch.tensor(copy.deepcopy(self._emit), dtype=torch.int64)
        state.state = torch.tensor(copy.deepcopy(self._stake), dtype=torch.int64)
        weights_numpy = numpy.zeros( (state.n, state.n) )
        for i in range(state.n):
            keys = self._weight_pubkeys[i]
            vals = self._weight_vals[i]
            val_sum = sum(vals)
            for k, val in list(zip(keys, vals)):
                if k in self._index_for_pubkey:
                    j = self._index_for_pubkey[k]
                    weights_numpy[i, j] = float(val) / float(val_sum)
        state.W = torch.tensor(weights_numpy, dtype=torch.float32)
        self.state = state

    async def _emit_state(self):
        r""" Emits the local self.state weights to the chain after normalization. Waits for the inclusion.
        """
        # Convert floats to ints with precision.
        u32_int_max = 4294967295 # max int value.
        weight_pubkeys = []
        weight_vals_as_ints = []
        for i, val in enumerate(self.state.weights.tolist()):
            if val > 0.0001:
                weight_pubkeys.append( self._pubkey_for_index[i] )
                int_val = int(float(val) * int(u32_int_max)) # convert to int representation.
                weight_vals_as_ints.append(int_val) # int weights sum to u32_int_max.

        # Try set weights.
        logger.info('emit -> {}', list(zip(weight_pubkeys, weight_vals_as_ints)))
        try:
            await self.subtensor_client.set_weights(weight_pubkeys, weight_vals_as_ints, self.__keypair, wait_for_inclusion = False)
            await self._wait_for_inclusion(weight_pubkeys, weight_vals_as_ints)
        except Exception as e:
            logger.info('Emission failed with exception: {}', e)
        logger.info('Emission done.')
        
    async def _wait_for_inclusion(self, local_keys, local_vals):
        r""" Waits until timeout for the local keys and vals to be set on chain.
        """
        def equal(chain_keys, chain_vals):
            if len(local_keys) != len(chain_keys):
                return False
            lkey_map = {}
            ckey_map = {}
            for i in range(len(local_keys)):
                lkey_map[local_keys[i]] = local_vals[i]
                ckey_map[chain_keys[i]] = chain_vals[i]
            for key in lkey_map.keys():
                if lkey_map[key] != ckey_map[key]:
                    return False
            return True
        start_time = time.time()
        chain_keys = await self.subtensor_client.weight_keys(self.__keypair.public_key)
        chain_vals = await self.subtensor_client.weight_vals(self.__keypair.public_key)
        are_equal = equal(chain_keys, chain_vals)
        while not are_equal:
            await asyncio.sleep(3)
            chain_keys = await self.subtensor_client.weight_keys(self.__keypair.public_key)
            chain_vals = await self.subtensor_client.weight_vals(self.__keypair.public_key)
            are_equal = equal(chain_keys, chain_vals)
            if (time.time() - start_time) > 12:
                logger.info('Failed to sync weights on chain.')
        logger.info('Chain weights {}', list(zip(chain_keys,chain_vals)))
        return True

    async def _poll_state(self):
        r""" Polls neurons on chain for latest info, sinks results to local cache.
        """
        calls = []
        current_block = await self._chain_block()
        emits = await self.subtensor_client.get_last_emit_data()
        for (pubkey, last_emit) in emits:
                # Filter stale emission.
                if (current_block - last_emit) < 100:
                    calls.append(self._poll_pubkey(pubkey))
        await asyncio.gather(*calls)

    async def _poll_pubkey(self, pubkey):
        """ Polls info info for a specfic public key.
        """
        logger.info('poll: {} ', pubkey)
        if pubkey in self._index_for_pubkey:
            index = self._index_for_pubkey[pubkey]
            append = False
        else:
            index = self._n
            uid = self._next_uid
            append = True
            self._n += 1
            self._next_uid += 1
            self._index_for_pubkey[pubkey] = index
            self._pubkey_for_index[index] = pubkey
        try:
            stake = await self.subtensor_client.get_stake(pubkey)
            emit = await self.subtensor_client.get_last_emit_data(pubkey)
            info = await self.subtensor_client.neurons(pubkey)
            w_keys = await self.subtensor_client.weight_keys(pubkey)
            w_vals = await self.subtensor_client.weight_vals(pubkey)

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
                self._weight_pubkeys[index] = list(w_keys)
                self._weight_vals[index] = list(w_vals)
            else:
                self._neurons.append(neuron)
                self._stake.append(int(stake))
                self._emit.append(int(emit))
                self._weight_pubkeys.append(list(w_keys))
                self._weight_vals.append(list(w_vals))
                self._uids.append( uid )
                self._index_for_uid[uid] = index

        except Exception as e:
            logger.error("Exception occurred: {}".format(e))
            traceback.print_exc()


    async def connect(self) -> bool:
        r""" Async: Makes and awaits for a connection to the chain.
        Returns:
            connected: (bool): true if the connection is a success.
        """
        self.subtensor_client.connect()
        connected = await self.subtensor_client.is_connected()
        return connected

    async def subscribe (self, timeout) -> bool:
        r""" Async: Makes a subscribe request to the chain. Waits for subscription includion or returns False
        Returns:
            subscribed: (bool): true if the subscription is a success.
        """
        await self.subtensor_client.subscribe(self._config.axon.remote_ip, self._config.axon.port)
        current_block = await self.subtensor_client.get_current_block()
        self._block = current_block
        self._last_emit = current_block

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
        r""" Async: Makes a unsubscribe request ot the chain waits for succes.
        """
        logger.info('Unsubscribe from chain endpoint')
        await self.subtensor_client.unsubscribe()

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--metagraph.chain_endpoint', default='206.189.254.5:12345', type=str, 
                            help='chain endpoint.')

        return parser

    @staticmethod   
    def check_config(config: Munch) -> Munch:
        return config

