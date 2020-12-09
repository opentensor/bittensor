
import asyncio
import copy
import argparse
import bittensor
import math
import netaddr
import numpy
import time
import torch
import traceback

from munch import Munch
from loguru import logger
from bittensor import bittensor_pb2
from bittensor.subtensor import WSClient
from typing import List, Tuple, List

from bittensor.exceptions.handlers import rollbar

def int_to_ip(int_val):
    return str(netaddr.IPAddress(int_val))
 
def ip_to_int(str_val):
    return int(netaddr.IPAddress(str_val))

class ChainState():
    def __init__(self):
        # Cached values.
        self.n = 0
        self.next_uid = 0
        self.uids = []
        self.stake = []
        self.lastemit = []
        self.neuron_weights = []
        self.weight_pubkeys = []
        self.weight_vals = []
        self.neurons = []
        self.index_for_uid = {}
        self.index_for_pubkey = {}
        self.pubkey_for_index = {}

    def add_or_update(self, pubkey:str, ip: int, port: int, lastemit: int, stake: int, w_keys: List[str], w_vals: List[int]):
        neuron = bittensor_pb2.Neuron(
            version=bittensor.__version__,
            public_key=pubkey,
            address=int_to_ip(ip),
            port=int(port)
        )
        if pubkey in self.index_for_pubkey:
            index = self.index_for_pubkey[pubkey]
            self.neurons[index] = neuron
            self.stake[index] = int(stake)
            self.lastemit[index] = int(lastemit)
            self.weight_pubkeys[index] = list(w_keys)
            self.weight_vals[index] = list(w_vals)
        else:
            index = self.n
            uid = self.next_uid
            self.n += 1
            self.next_uid += 1
            self.index_for_pubkey[pubkey] = index
            self.pubkey_for_index[index] = pubkey
            self.neurons.append(neuron)
            self.stake.append(int(stake))
            self.lastemit.append(int(lastemit))
            self.weight_pubkeys.append(list(w_keys))
            self.weight_vals.append(list(w_vals))
            self.uids.append( uid )
            self.index_for_uid[uid] = index

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
                
            lastemit: (:obj:`torch.LongTensor` of shape :obj:`(n)`):
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
        self.lastemit = torch.tensor([])
        self.W = torch.tensor([[]])
        self.neurons = []

    @staticmethod
    def from_cache(cache: ChainState):
        r""" Deep copies from the chain state.
        """
        # Deep copies chain state into metagraph state.
        state = TorchChainState()
        state.n = cache.n
        state.neurons = copy.deepcopy(cache.neurons)
        state.indices = torch.tensor(range(state.n), dtype=torch.int64)
        state.uids = torch.tensor(copy.deepcopy(cache.uids), dtype=torch.int64)
        state.lastemit = torch.tensor(copy.deepcopy(cache.lastemit), dtype=torch.int64)
        state.state = torch.tensor(copy.deepcopy(cache.stake), dtype=torch.int64)
        weights_numpy = numpy.zeros( (state.n, state.n) )
        for i in range(state.n):
            keys = cache.weight_pubkeys[i]
            vals = cache.weight_vals[i]
            val_sum = sum(vals)
            for k, val in list(zip(keys, vals)):
                if k in cache.index_for_pubkey:
                    j = cache.index_for_pubkey[k]
                    weights_numpy[i, j] = float(val) / float(val_sum)
        state.W = torch.tensor(weights_numpy, dtype=torch.float32)
        return state

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

        # Chain state cached before converted into the torch state.
        self.cache = ChainState()

        # Chain state as torch values.
        self.state = TorchChainState.from_cache(self.cache)


    @property
    def n(self) -> int:
        r""" Return the number of known neurons on chain.
        Returns
            n: (int):
                number of known neurons.
        """
        return self.state.n

    @property
    def block(self) -> int:
        r""" Return the block number when the chain state was updated.
        Returns
            block: (int):
                local chain state block number.
        """
        return self.state.block

    @property
    def lastemit(self) -> torch.LongTensor:
        r""" Returns the last emit time for each known neuron.
        Returns
            lastemit: (int):
                last emit time.
        """
        return self.state.lastemit

    @property
    def indices(self) -> torch.LongTensor:
        r""" Return the indices of each neuron in the chain state range(n).
        Returns
            indices: (:obj:`torch.LongTensor` of shape :obj:`(n)`):
                returned indices for each neuron.
        """
        return self.state.indices

    @property
    def uids(self) -> torch.LongTensor:
        r""" Returns unique ids for each neuron in the chain state.
        Returns
            uids: (:obj:`torch.LongTensor` of shape :obj:`(n)`):
                unique id for each neuron.
        """
        return self.state.uids

    @property
    def stake(self) -> torch.LongTensor:
        r""" Returns the stake held by each known neuron.
        Returns
            stake: (:obj:`torch.LongTensor` of shape :obj:`(n)`):
                stake of each known neuron.
        """
        return self.state.stake

    @property
    def W(self) -> torch.FloatTensor:
        r""" Full chain weight matrix for each neuron.
        Returns
            W: (:obj:`torch.LongFloat` of shape :obj:`(n, n)`):
                w_ij of each neuron.
        """
        return self.state.W

    @property
    def neurons(self) -> List[bittensor_pb2.Neuron]:
        r""" Return neuron endpoint information for each neuron.
        Returns
            neurons: (:obj:`List[bittensor_pb2.Neuron]` of shape :obj:`(n, n)`):
                endpoint information for each neuron.
        """
        return self.state.neurons

    @property
    def weights(self):
        r"""Return this neuron's weights. W[0,:]
        Returns 
            weights: (:obj:`torch.FloatTensor` of shape :obj:`(n)`):
                returned indices for passed uids.
        """
        if self.state.n == 0:
            return torch.Tensor([])
        else:
            w_0 = self.state.W[0,:]
            return w_0

    def uids_to_indices(self, uids: torch.Tensor):
        r"""Return the indices of passed uids
        Args:
            uids: (:obj:`torch.LongTensor` of shape :obj:`(-1)`):
                UIDs for indices
        Returns 
            indices: (:obj:`torch.LongTensor` of shape :obj:`(-1)`):
                returned indices for passed uids.
        """
        indices = torch.nonzero(uids[..., None] == self.state.uids)[:,1]
        if torch.numel(uids) != torch.numel(indices):
            raise ValueError('Passed uids are not a subset of class.uids, with passed: {} and class.uids: {}'.format(uids, self.state.uids))
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
            response.append(self.state.neurons[idx])
        return response

    def chain_weights(self) -> torch.FloatTensor:
        r""" Returns your current weights from the chain.
        Returns:
            weights: (torch.Tensor) weights on chain as torch tensor.
        """
        def handle_async_exception(loop, ctx):
            logger.error("Exception in async task: {0}".format(ctx['exception']))
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(handle_async_exception)
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_chain_weights())

    async def async_chain_weights(self) -> torch.FloatTensor:
        r""" Async: returns your current weights from the chain.
        Returns:
            weights: (torch.FloatTensor) weights on chain for each neuron.
        """
        chain_keys = await self.subtensor_client.weight_keys(self.__keypair.public_key)
        chain_vals = await self.subtensor_client.weight_vals(self.__keypair.public_key)
        val_sum = sum(chain_vals)
        retval = torch.zeros(self._n)
        for key, val in list(zip(chain_keys, chain_vals)):
            idx = self._index_for_pubkey[key]
            if idx >= self._n:
                continue
            else:
                retval[idx] = float(val) / float(val_sum)
        return retval

    def block(self):
        r""" Returns the current block on the chain.
        Returns:
            block: (int) block number on chain.
        """
        # def handle_async_exception(loop, ctx):
        #     logger.error("Exception in async task: {0}".format(ctx['exception']))
        loop = asyncio.get_event_loop()
        # loop.set_exception_handler(handle_async_exception)
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_block())

    async def async_block(self) -> int:
        r""" Async returns the current block on the chain.
        Returns:
            block: (int) block number on chain.
        """
        return await self.subtensor_client.get_current_block()

    def subscribe(self, timeout) -> bool:
        r""" Syncronous: Makes a subscribe request to the chain. Waits for subscription inclusion or returns False
        Returns:
            subscribed: (bool): true if the subscription is a success.
        """
        # def handle_async_exception(loop, ctx):
        #     logger.error("Exception in async task: {0}".format(ctx['exception']))
        loop = asyncio.get_event_loop()
        # loop.set_exception_handler(handle_async_exception)
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_subscribe(timeout))

    async def async_subscribe (self, timeout) -> bool:
        r""" Async: Makes a subscribe request to the chain. Waits for subscription inclusion or returns False
        Returns:
            subscribed: (bool): true if the subscription is a success.
        """
        await self.subtensor_client.subscribe(self._config.axon.remote_ip, self._config.axon.port)
        return await self._wait_for_subscription(timeout=12)

    def unsubscribe(self) -> bool:
        r""" Syncronous: Unsubscribes the local neuron from the chain.
         """
        # def handle_async_exception(loop, ctx):
        #     logger.error("Exception in async task: {0}".format(ctx['exception']))
        loop = asyncio.get_event_loop()
        # loop.set_exception_handler(handle_async_exception)
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_unsubscribe())  

    async def async_unsubscribe (self):
        r""" Async: Unsubscribes the local neuron from the chain.
        """
        logger.info('Unsubscribe from chain endpoint')
        await self.subtensor_client.unsubscribe()

    def connect(self) -> bool:
        r""" Synchronous: Connects to the chain.
        Returns:
            connected: (bool): true if the connection is a success.
        """
        # def handle_async_exception(loop, ctx):
        #     logger.error("Exception in async task: {0}".format(ctx['exception']))
        loop = asyncio.get_event_loop()
        # loop.set_exception_handler(handle_async_exception)
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_connect())

    async def async_connect(self) -> bool:
        r""" Async: Makes and awaits for a connection to the chain.
        Returns:
            connected: (bool): true if the connection is a success.
        """
        self.subtensor_client.connect()
        connected = await self.subtensor_client.is_connected()
        return connected        

    def emit(self, weights: torch.FloatTensor):
        r""" Emits the passed weights to the chain. Waits for inclusion.
        Args:
            weights: (torch.FloatTensor): 
                weights to set on chain of length self.state.n
        """
        # TODO(const): this repeat code can be abstracted.
        # def handle_async_exception(loop, ctx):
        #     logger.error("Exception in async task: {0}".format(ctx['exception']))
        loop = asyncio.get_event_loop()
        # loop.set_exception_handler(handle_async_exception)
        loop.set_debug(enabled=True)
        loop.run_until_complete(self.async_emit(weights))

    async def async_emit(self, weights: torch.FloatTensor) -> bool:
        r""" Emits the passed weights to the chain. Waits for inclusion.
        Args:
            weights: (torch.FloatTensor): 
                weights to set on chain.
        Return:
            included: (bool) true is the weights were set on chain.
        """
        logger.info('Emit -> {}', weights)
        # Check that weights meet chain requirements.
        # #TODO(const) check with current weights.
        if not self._check_weights(weights):
            logger.error('Weight emit failed with weight check.')
            return False

        # Convert weights to integer represenation and get corresponding keys.
        keys, vals = self._convert_weights(weights)

        # Remove unchanged vals.
        keys, vals = await self._remove_noop(keys, vals)
        if len(keys) == 0:
            logger.error('Weight emit is a no-op.')
            return False

        # Makes weight emission call.
        # TODO(const): make wait for inclusion work.
        try:
            await self.subtensor_client.set_weights(keys, vals, self.__keypair, wait_for_inclusion = False)
        except Exception as e:
            logger.info('Failed to emit weights with error {}, and weights {}', e, list(zip(keys, vals)))
            rollbar.send_exception()

            return False

        # Checks that weight emission was included in a block after 12 seconds.
        if not await self._wait_for_emit_inclusion(keys, vals, timeout = 12):
            logger.error('Weight failed with non-inclusion after 12 seconds.')
            return False
        return True

    def sync(self, weights: torch.FloatTensor) -> torch.FloatTensor:
        r""" Synchronizes the local self.state with the chain state, sinking the trained weights and pulling 
        info from other peers. Ensures the self.state is in accordance with the state on chain at this block.
            Args:
                weights: (torch.FloatTensor):
                    weights to set on chain.
            Returns:
                weights: (torch.FloatTensor):
                    weights on chain.
        """
        # def handle_async_exception(loop, ctx):
        #     logger.error("Exception in async task: {0}".format(ctx['exception']))
        loop = asyncio.get_event_loop()
        # loop.set_exception_handler(handle_async_exception)
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_sync(weights))

    async def async_sync(self, weights: torch.FloatTensor) -> torch.FloatTensor:
        r""" Async: Synchronizes the local self.state with the chain state by polling the chain.
            Args:
                weights: (torch.FloatTensor):
                    weights to set on chain.
            Returns:
                weights: (torch.FloatTensor):
                    weights on chain.
        """
        if weights != None:
            await self.async_emit(weights)
        await self._sync_cache()
        last_sync = await self.async_block()
        self.state = TorchChainState.from_cache(self.cache)
        self.state.block = last_sync
        return self.weights

    async def _sync_cache(self):
        r""" Async: Makes calls to chain updating local chain cache with newest info.
        """
        # Make asyncronous calls to chain filling local state cache.
        calls = []
        current_block = await self.async_block()
        emits = await self.subtensor_client.get_last_emit_data()
        for (pubkey, last_emit) in emits:
                # Filter based on stale emissions.
                if (current_block - last_emit) < self._config.metagraph.stale_emit_filter:
                    calls.append(self._poll_pubkey(pubkey))
        await asyncio.gather(*calls)

    async def _poll_pubkey(self, pubkey):
        r""" Polls info info for a specfic public key.
        """
        logger.info('poll: {} ', pubkey)
        try:
            stake = await self.subtensor_client.get_stake(pubkey)
            lastemit = await self.subtensor_client.get_last_emit_data(pubkey)
            info = await self.subtensor_client.neurons(pubkey)
            w_keys = await self.subtensor_client.weight_keys(pubkey)
            w_vals = await self.subtensor_client.weight_vals(pubkey)
            self.cache.add_or_update(pubkey = pubkey, ip = info['ip'], port = info['port'], lastemit = lastemit, stake = stake, w_keys = w_keys, w_vals = w_vals)
        except Exception as e:
            logger.error("Exception occurred: {}".format(e))
            traceback.print_exc()


    async def _wait_for_subscription(self, timeout=12) -> bool:
        r""" Async: Waits for subscription info to appear on chain.
        Returns:
            subscribed: (bool): true if info is set on chain after timeout.
        """
        start_time = time.time()
        info = await self.subtensor_client.neurons(self.__keypair.public_key)
        while info == None:
            await asyncio.sleep(1)
            info = await self.subtensor_client.neurons(self.__keypair.public_key)
            if time.time() - start_time > timeout:
                return False
        return True

    async def _are_set_on_chain(self, keys, vals) -> bool:
        r""" Returns true if the passed key and vals are set on chain.
        """
        cmap = {}
        chain_keys = await self.subtensor_client.weight_keys(self.__keypair.public_key)
        chain_vals = await self.subtensor_client.weight_vals(self.__keypair.public_key)
        for key, val in list(zip(chain_keys, chain_vals)):
            cmap[key] = val
        for key, val in list(zip(keys, vals)):
            if key not in cmap:
                return False
            if cmap[key] != val:
                return False 
        return True

    async def _wait_for_emit_inclusion(self, weight_keys, weight_vals, timeout=12):
        r""" Waits until timeout for the local keys and vals to be set on chain.
        """
        start_time = time.time()
        while not await self._are_set_on_chain(weight_keys, weight_vals):
            await asyncio.sleep(3)
            if (time.time() - start_time) > timeout:
                logger.info('Timeout while waiting for emit inclusion.')
                return False
        chain_keys = await self.subtensor_client.weight_keys(self.__keypair.public_key)
        chain_vals = await self.subtensor_client.weight_vals(self.__keypair.public_key)
        logger.info('Chain weights {}', list(zip(chain_keys,chain_vals)))
        return True

    async def _remove_noop(self, weight_keys, weight_vals):
        r""" Removes weights and vals from the chain update which have not changed on chain.
        Returns:
            keys, vals:
                keys, vals with removed noops.
        """
        cmap = {}
        chain_keys = await self.subtensor_client.weight_keys(self.__keypair.public_key)
        chain_vals = await self.subtensor_client.weight_vals(self.__keypair.public_key)
        for key, val in list(zip(chain_keys, chain_vals)):
            cmap[key] = val

        ret_keys = []
        ret_vals = []
        for key, val in list(zip(weight_keys, weight_vals)):
            if key in cmap:
                if cmap[key] == val:
                    continue 
            ret_keys.append(key)
            ret_vals.append(val)

        return ret_keys, ret_vals          
      
    def _check_weights(self, weights: torch.Tensor):
        r""" Checks that weights vector being set on chain meet requirements.
        Returns:
            valid: (bool)
                True if the weight being set meet requirements to be set on chain.
        """
        as_list = weights.tolist()
        if len(as_list) != self.state.n:
            logger.error("Error trying to set weights on chain. Got length {}, but the length must match the number of neurons in self.state.n {}", len(as_lsit), self.state.n)
            return False
        sum_list = sum(as_list)
        epsilon = 0.001
        if abs(1.0 - sum_list) > epsilon:
            logger.error("Error trying to set weights on chain. Got {} but sum of weights must equal 1", sum_list)
            return False
        min_list = min(as_list)
        if min_list < 0.0:
            logger.error("Error trying to set weights on chain. Got min value {} but values must be in range [0,1]", min_list)
            return False
        max_list = max(as_list)
        if max_list > 1.0:
            logger.error("Error trying to set weights on chain. Got max value {} but values must be in range [0,1]", max_list)
            return False
        return True

    def _convert_weights(self, weights: torch.FloatTensor) -> Tuple[List[str], List[int]]:
        r""" Converts weights into integer u32 representation.
        Returns:
            keys: (List[str]):
                List of pubkeys associated with each weight from vals.
            vals: (List[int]):
                List of u32 integer representations of floating point weights.
        """
        # Convert floats to ints with precision.
        u32_int_max = 4294967295 # max u32 int value.
        weight_pubkeys = []
        weight_vals_as_ints = []
        for i, val in enumerate(weights.tolist()):
            weight_pubkeys.append( self.cache.pubkey_for_index[i] ) # Gets the pubkey at this index.
            int_val = int(float(val) * int(u32_int_max)) # convert to int representation.
            weight_vals_as_ints.append(int_val) # int weights sum to u32_int_max.
        return weight_pubkeys, weight_vals_as_ints


    @staticmethod   
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--metagraph.chain_endpoint', default='206.189.254.5:12345', type=str, 
                            help='chain endpoint.')
        parser.add_argument('--metagraph.stale_emit_filter', default=10000, type=int,
                            help='filter neurons with last emit beyond this many blocks.')

        return parser

    @staticmethod   
    def check_config(config: Munch) -> Munch:
        return config

