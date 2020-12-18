
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
from bittensor.subtensor.client import WSClient
from typing import List, Tuple, List

from bittensor.exceptions.handlers import rollbar

MAX_INT_WEIGHT = 4294967295 # Max weight value on chain.

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
            self.stake[index] = float(stake)
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
            self.stake.append(float(stake))
            self.lastemit.append(int(lastemit))
            self.weight_pubkeys.append(list(w_keys))
            self.weight_vals.append(list(w_vals))
            self.uids.append( uid )
            self.index_for_uid[uid] = index

# Static network state object.
class TorchChainState():
    r""" Maintains the chain state as a torch object.
        Params:
            tau: (int): 
                current, per block, token inflation rate.

            block: (int):
                state block number.

            uids: (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n)`):
                UIDs for each neuron ordered by index.
            
            indices: (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n)`):
                Index of neurons, range(metagraph.n)

            stake: (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n)`):
                Stake balance for each neuron ordered by index.
                
            lastemit: (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n)`):
                Last emission call for each neuron ordered by index.

            weights: (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                This neuron's weights W[,:]

            W: (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n, metagraph.n)`):
                Full weight matrix on chain.

            neurons: (List[bittensor_pb2.Neuron]) 
                List of endpoints on the network.

    """
    def __init__(self):
        self.tau = torch.tensor([50.0], dtype = torch.float32)
        self.block = 0
        self.n = 0
        self.uids = torch.tensor([])
        self.indices = torch.tensor([])
        self.stake = torch.tensor([])
        self.lastemit = torch.tensor([])
        self.W = torch.tensor([[]])
        self.neurons = []
        self.uid_for_pubkey = {}
        self.index_for_pubkey = {}

    @staticmethod
    def from_cache(cache: ChainState):
        r""" Deep copies from the chain state.
        """
        # Deep copies chain state into metagraph state.
        state = TorchChainState()
        state.n = cache.n
        state.tau = torch.tensor([50.0], dtype = torch.float32)
        state.neurons = copy.deepcopy(cache.neurons)
        state.indices = torch.tensor(range(state.n), dtype=torch.int64)
        state.uids = torch.tensor(copy.deepcopy(cache.uids), dtype=torch.int64)
        state.lastemit = torch.tensor(copy.deepcopy(cache.lastemit), dtype=torch.int64)
        state.stake = torch.tensor(copy.deepcopy(cache.stake), dtype=torch.float32)
        for idx, (uid, n) in enumerate(list(zip(cache.uids, cache.neurons))):
            state.uid_for_pubkey[n.public_key] = uid
            state.index_for_pubkey[n.public_key] = idx
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

    def __init__(self, config):
        r"""Initializes a new Metagraph subtensor interface.
        Args:
            config (bittensor.Config):
                An bittensor config object.
        """
        # Protected vars
        self._config = config
        self.__keypair = config.session.keypair

        # Client for talking to chain.
        self.subtensor_client = WSClient(self._config.metagraph.chain_endpoint, self.__keypair)

        # Chain state cached before converted into the torch state.
        self.cache = ChainState()

        # Chain state as torch values.
        self.state = TorchChainState.from_cache(self.cache)

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        # TODO(const): check this endpoint in check_config.
        parser.add_argument('--metagraph.chain_endpoint', default='206.189.254.5:12345', type=str, 
                            help='chain endpoint.')
        parser.add_argument('--metagraph.stale_emit_filter', default=10000, type=int, 
                            help='filter neurons with last emit beyond this many blocks.')

    @staticmethod   
    def check_config(config: Munch):
        pass

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
        r""" Return the indices of each neuron in the chain state range(metagraph.n).
        Returns
            indices: (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n)`):
                returned indices for each neuron.
        """
        return self.state.indices

    @property
    def uids(self) -> torch.LongTensor:
        r""" Returns unique ids for each neuron in the chain state.
        Returns
            uids: (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n)`):
                unique id for each neuron.
        """
        return self.state.uids

    @property
    def stake(self) -> torch.FloatTensor:
        r""" Returns the stake held by each known neuron.
        Returns
            stake: (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                stake of each known neuron.
        """
        return self.state.stake

    @property
    def S(self) -> torch.FloatTensor:
        r""" Returns the stake held by each known neuron.
        Returns
            S: (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                stake of each known neuron.
        """
        return self.state.stake

    @property
    def tau(self) -> torch.FloatTensor:
        r""" tau: the chain per block inflation rate. i.e. 50
        Returns
            tau: (:obj:`torchFloatTensor` of shape :obj:`(1)`):
                current chain inflation rate.
        """
        return self.state.tau

    @property
    def incentive(self) -> torch.FloatTensor:
        r""" Returns the ranks 
        Returns
            incentive: (:obj:`torch.FLoatTensor` of shape :obj:`(metagraph.n)`):
                inflation incentive of each each known neuron.
        """
        I =  (self.tau * self.ranks) / torch.sum(self.ranks)
        I = torch.where(torch.isnan(I), torch.zeros_like(I), I)
        return I

    @property
    def I(self) -> torch.FloatTensor:
        r""" Returns the inflation incentive for each peer per block.
        Returns
            I: (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                stake of each known neuron.
        """
        return self.incentive

    @property
    def ranks(self) -> torch.FloatTensor:
        r""" Returns the ranks W^t * S
        Returns
            ranks: (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                rank of each known neuron.
        """
        if self.W.shape[0] == 0:
            return torch.tensor([])
        else:
            # S.shape = [self.state.n]
            # W.shape = [self.state.n, self.state.n]
            # R.shape = [self.state.n]
            S = self.S.view(self.state.n, 1)
            W = torch.transpose(self.W.view(self.state.n, self.state.n), 0, 1)
            R = torch.matmul(W, S).view(self.state.n)
        return R

    @property
    def R(self) -> torch.FloatTensor:
        r""" Returns ranks for each known neuron in the graph.
        Returns
            rank: (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                rank of each known neuron.
        """
        return self.ranks()

    @property
    def W(self) -> torch.FloatTensor:
        r""" Full chain weight matrix for each neuron.
        Returns
            W: (:obj:`torch.LongFloat` of shape :obj:`(metagraph.n, metagraph.n)`):
                w_ij of each neuron.
        """
        return self.state.W

    @property
    def neurons(self) -> List[bittensor_pb2.Neuron]:
        r""" Return neuron endpoint information for each neuron.
        Returns
            neurons: (:obj:`List[bittensor_pb2.Neuron]` of shape :obj:`(metagraph.n, metagraph.n)`):
                endpoint information for each neuron.
        """
        return self.state.neurons

    @property
    def public_keys(self) -> List[str]:
        r""" Return the ordered public keys for state neurons.
        Returns
            public_keys: (:obj:`List[str]` of shape :obj:`(metagraph.n)`):
                public keys of all graph neurons.
        """
        return [n.public_key for n in self.state.neurons]

    @property
    def weights(self) -> torch.FloatTensor:
        r"""Return this neuron's weights. W[0,:]
        Returns 
            weights: (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
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
                uids into neuron protos
        Returns:
            neurons: (List[bittensor_pb2.Neuron]): 
                neuron info ordered by passed uids.
        """
        response = []
        indices = self.uids_to_indices(uids)
        for idx in indices.tolist():
            response.append(self.state.neurons[idx])
        return response

    def neurons_to_uids(self, neurons: List[bittensor_pb2.Neuron]) -> torch.LongTensor:
        r""" Returns uids associated with the passed neurons.
        Args:
            neurons: (List[bittensor_pb2.Neuron]): 
                neuron info ordered by passed uids.
        Returns:
            uids: (torch.LongTensor)
                uids associated with neurons.
        """
        uids = []
        for n in neurons:
            uids.append(self.state.uid_for_pubkey[n.public_key])
        return torch.tensor(uids)

    def chain_weights(self) -> torch.FloatTensor:
        r""" Returns your current weights from the chain.
        Returns:
            weights: (:obj:`torch.FloatTensor` of shape :obj:`(-1)`):
                weights on chain as torch tensor.
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.async_chain_weights())

    async def async_chain_weights(self) -> torch.FloatTensor:
        r""" Async: returns your current weights from the chain.
        Returns:
            weights: (:obj:`torch.FloatTensor` of shape :obj:`(-1)`):
                weights on chain as torch tensor.
        """
        # --- Get chain weights ----
        chain_keys = await self.subtensor_client.weight_keys(self.__keypair.public_key)
        chain_weights = await self.subtensor_client.weight_vals(self.__keypair.public_key)
        if chain_weights == None or len(chain_weights) == 0:
            return torch.tensor([])

        else:
            # ---- To be filled ----
            return_val = torch.zeros(self.state.n)

            weight_sum = sum(chain_weights)
            if weight_sum != MAX_INT_WEIGHT:
                logger.error('Chain weights do not sum to {} with vals {}', MAX_INT_WEIGHT, chain_weights)

            # ---- Fill torch tensor ----
            for key, weight in list(zip(chain_keys, chain_weights)):
                if key not in self.state.index_for_pubkey:
                    logger.critical('key {} on chain not in state.index', key)
                    continue
                else:
                    idx = self.state.index_for_pubkey[key]
                    if idx >= self.state.n:
                        logger.critical('idx {} in state.index > state.n', idx)
                        continue
                    else:
                        return_val[idx] = float(weight) / float(weight_sum)
            return return_val

    def chain_block(self):
        r""" Returns the current block on the chain.
        Returns:
            block: (int) block number on chain.
        """
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_chain_block())

    async def async_chain_block(self) -> int:
        r""" Async returns the current block on the chain.
        Returns:
            block: (int) block number on chain.
        """
        return await self.subtensor_client.get_current_block()

    def unsubscribe(self) -> bool:
        r""" Syncronous: Unsubscribes the local neuron from the chain.
         """
        loop = asyncio.get_event_loop()
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
        loop = asyncio.get_event_loop()
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

    def sync(self) -> torch.FloatTensor:
        r""" Synchronizes the local self.state with the chain state.
            Returns:
                weights: (torch.FloatTensor):
                    weights on chain.
        """
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_sync())

    async def async_sync(self) -> torch.FloatTensor:
        r""" Async: Synchronizes the local self.state with the chain state by polling the chain.
        """
        await self._sync_cache()
        last_sync = await self.async_chain_block()
        self.state = TorchChainState.from_cache(self.cache)
        self.state.block = last_sync

    async def _sync_cache(self):
        r""" Async: Makes calls to chain updating local chain cache with newest info.
        """
        # Make asyncronous calls to chain filling local state cache.
        calls = []
        current_block = await self.async_chain_block()
        emits = await self.subtensor_client.get_last_emit_data()
        for (pubkey, last_emit) in emits:
                # Filter based on stale emissions.
                if (current_block - last_emit) < self._config.metagraph.stale_emit_filter:
                    calls.append(self._poll_pubkey(pubkey))
        await asyncio.gather(*calls)

    async def _poll_pubkey(self, pubkey):
        r""" Polls info info for a specfic public key.
        """
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


    def subscribe(self, timeout) -> bool:
        r""" Syncronous: Makes a subscribe request to the chain. Waits for subscription inclusion or returns False
        Returns:
            subscribed: (bool): true if the subscription is a success.
        """
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_subscribe(timeout))


    SubscribeSuccess = 1
    SubscribeUnknownError = 2
    SubscribeTimeout = 3
    async def async_subscribe (self, timeout) -> bool:
        r""" Async: Makes a subscribe request to the chain. Waits for subscription inclusion or returns False
        Returns:
            subscribed: (bool): true if the subscription is a success.
        """

        # ---- Try Subscription ----
        try:
            code, message = await self._try_async_subscribe(timeout)

            if code == Metagraph.SubscribeSuccess:
                info = await self.subtensor_client.neurons(self.__keypair.public_key)
                logger.info('Successfully subcribed session with: {}', info)
                return True

            elif code == Metagraph.SubscribeUnknownError:
                logger.error('Subscription threw an unknown error: {}', message)
                return False

            elif code == Metagraph.SubscribeTimeout:
                logger.error('Subscription timeout {}', message)
                return False

        except Exception as e:
            logger.error('Subscription threw an uncaught error {}', e)
            return False
        
    async def _try_async_subscribe(self, timeout: int):
        r""" Makes subscription attempts to the chain, continuing to attempt until timeout and finally waiting for inclusion.

        Args:
            timeout (int):
                Time to wait before subscription times out.

        Raises:
            SubscribeSuccess:
                Raised when the subscription is a success before the timeout

            SubscribeUnknownError:
                UnknownError during subscription.

            SubscribeTimeout:
                Raised when the attempted subscription fails after timeout.
        """
        start_time = time.time()
        # ---- Make Subscription transaction ----
        while True:
            try:
                await self.subtensor_client.subscribe(self._config.axon.remote_ip, self._config.axon.port)
                break

            except Exception as e:
                if (time.time() - start_time) > timeout:
                    # --- Timeout during emit call ----
                    message = "Timed-out with Unknown Error while trying to make the subscription call. With last exception {}".format(e)
                    return Metagraph.SubscribeUnknownError, message

                else:
                    # --- Wait for inclusion, no error.
                    logger.trace('Error while attempting subscription {}', e)
                    continue

        # ---- Wait for inclusion ----
        while True:
            try:
                # ---- Request info from chain ----
                info = await self.subtensor_client.neurons(self.__keypair.public_key)

            except Exception as e:
                # ---- Catch errors in request ----
                message = "Subscription threw an unknown exception {}".format(e)
                return Metagraph.SubscribeUnknownError, message

            if info != None:
                # ---- Subscription was a success ----
                return Metagraph.SubscribeSuccess, "Subscription success"

            elif time.time() - start_time > timeout:
                # ---- wait -----
                return Metagraph.SubscribeTimeout, "Subscription timeout"

            else:
                # ---- wait -----
                await asyncio.sleep(1)

        # ---- ?! WaT ?! ----
        logger.critical('Should not get here')
        return SubscribeUnknownError, 'Should not get here'

      
    def emit(self, weights: torch.FloatTensor, wait_for_inclusion = False, timeout = 12):
        r""" Emits the passed weights to the chain. Optionally Waits for inclusion. 
        Failures are logged but do not break the process. 

        Args:
            Weights: (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                weights to set on chain of length self.state.n

            Wait_for_inclusion: (bool, default: False):
                if true, the call waits for inclusion in the block before continuing.

            Timeout: (int, default = 12 sec):
                time to wait for inclusion before raising a caught error.
        """
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        loop.run_until_complete(self.async_emit(weights, wait_for_inclusion, timeout))


    EmitSuccess = 1
    EmitValueError = 2
    EmitUnknownError = 3
    EmitTimeoutError = 4
    EmitTimeoutError = 5
    EmitNoOp = 6
    async def async_emit(self, weights: torch.FloatTensor, wait_for_inclusion = False, timeout = 12) -> bool:
        r""" Calls _try_async_emit, logs results based on raised exception. Only fails on an uncaught Exception.
        
        Args:
            weights: (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                Weights to set on chain.
            wait_for_inclusion: (bool):
                If true, the call waits for block-inclusion before continuing or throws error after timeout.
            timeout: (int, default = 12 sec):
                Time to wait for inclusion before raising a caught error.
 
        """
        # --- Try emit, optionally wait ----
        try:
            code, message= await self._try_async_emit(weights, wait_for_inclusion, timeout)
            if code == Metagraph.EmitSuccess:
                # ---- Emit was a success. ----
                logger.info("Successful emission.")

            elif code == Metagraph.EmitValueError:
                # ---- Passed weights were incorrect ----
                logger.warning("Value error during emission: {}", message)

            elif code == Metagraph.EmitUnknownError:
                # ---- Unknown error ----
                logger.error("Unknown error during emission: {}", message)

            elif code == Metagraph.EmitTimeoutError:
                # ---- Timeout while waiting for inclusion ----
                logger.warning("Emission timeout after {} seconds with error {}", timeout, message)
            
            elif code == Metagraph.EmitResultUnknown:
                # ---- Did not wait, result unknown ----
                logger.trace("Emit results unknown.")

            elif code == Metagraph.EmitNoOp:
                # ---- Emit is a NoOp ----
                logger.info("When trying to set weights on chain. Weights are unchanged, nothing to emit.")

        except Exception as e:
            # ---- Unknown error, raises error again. Should never get here ----
            logger.error("Unknown Error during emission {}", e)
            raise e

        chain_weights = await self.async_chain_weights()
        logger.info('After emit, weights on chain: {}', chain_weights)

    async def _try_async_emit(self, weights: torch.FloatTensor, wait_for_inclusion = False, timeout = 12) -> bool:
        r""" Makes emit checks, emits to chain, and raises one of the following errors.
        Args:
            weights: (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                Weights to set on chain.

            wait_for_inclusion: (bool):
                If true, the call waits for block-inclusion before continuing or throws error after timeout.

            timeout: (int, default = 12 sec):
                Time to wait for inclusion before raising a caught error.

        Raises:
            EmitSuccess:
                Raised when try_async_emit emits weights successfully with known result.

            EmitNoOp:
                Raised when calling emit does not change weights on chain.

            EmitUnknownError:
                UnknownError during emit.

            EmitValueError:
                Raised during emission when passed weights are not properly set.

            EmitTimeoutError:
                Raised during emission during a timeout.

            EmitResultUnknown:
                Called when an emit step end without a known result, for instance, 
                if the user has wait_for_inclusion = False.
        """
        # --- Check type ----
        if not isinstance(weights, torch.Tensor):
            message = "Error trying to set weights on chain. Got weights type {}, but weights must be of type {}".format(type(weights), torch.Tensor)
            return Metagraph.EmitValueError, message

        # ---- Convert weights to list ----
        weights = [float(w) for w in weights.tolist()]

        # ---- Check length > 0 ----
        if len(weights) == 0:
            message = "Error tyring to set weight on china. Got a length 0 set of values, must be at least length 1."
            return Metagraph.EmitValueError, message

        # ---- Check length ----
        if len(weights) != self.state.n:
            message = "Error trying to set weights on chain. Got length {}, but the length must match the number of neurons in metagraph.neurons {}".format(len(weights), self.state.n)
            return Metagraph.EmitValueError, message

        # ---- Check approximate sum ----
        sum_weights = sum(weights)
        epsilon = 0.001
        if abs(1.0 - sum_weights) > epsilon:
            message = "Error trying to set weights on chain. Got {} for sum, but passed weights must sum to 1 ".format(len(sum_weights), self.state.n)
            return Metagraph.EmitValueError, message

        # ---- Check min ----
        min_weights = min(weights)
        if min_weights < 0.0:
            message = "Error trying to set weights on chain. Got min value {} but values must be in range [0,1]".format(min_weights)
            return Metagraph.EmitValueError, message

        # ---- Check max ----
        max_weights = max(weights)
        if max_weights > 1.0:
            message = "Error trying to set weights on chain. Got max value {} but values must be in range [0,1]".format(max_weights)
            return Metagraph.EmitValueError, message

        # ---- Convert Weights to int-vals and pubkeys ----
        try:
            weight_keys, weight_vals = self._convert_weights_to_ints(weights)
        except Exception as e:
            message = "Unknown error when converting weights to ints with weights {} and error {}".format(weight_vals, e)
            return Metagraph.EmitUnknownError, message

        # ---- Check sum ----
        weight_sum = sum(weight_vals)
        if weight_sum != MAX_INT_WEIGHT:
            message = "Error trying to set weights on chain. Converted weights do not sum to {} with weights_vals {}".format(MAX_INT_WEIGHT, weight_vals)
            return Metagraph.EmitValueError, message

        # ---- Check NO-OP ----
        if await self._are_set_on_chain(weight_vals, weight_keys):
            message = "When trying to set weights on chain. Weights are unchanged, nothing to emit."
            return Metagraph.EmitNoOp, message

        # ---- Emit ----
        start_time = time.time()
        while True:
            try:
                # --- Make emission call ----
                logger.debug('Emit -> {} {}', weight_keys, weight_vals)
                await self.subtensor_client.emit(weight_keys, weight_vals, self.__keypair, wait_for_inclusion = False)
                break

            except Exception as e:
                logger.trace('Emit error {}', e)

                if not wait_for_inclusion:
                    # --- No wait, and error during emit call ----
                    message = "Error raised during call to emit {}".format( e )
                    return Metagraph.EmitUnknownError, message
                
                elif (time.time() - start_time) > timeout:
                    # --- Timeout during emit call ----
                    message = "Timed-out with unknown Error while trying to make the emit call. With last exception {}".format(e)
                    return Metagraph.EmitUnknownError, message

                else:
                    # --- Wait for inclusion, no error.
                    await asyncio.sleep(3) # To avoid ddos-ing the chain.
                    continue

        # --- Wait for inclusion ----
        if not wait_for_inclusion:
            message = "Emit ended but we don't know if weights were set on chain"
            return Metagraph.EmitResultUnknown, message

        else:
            while True:
                did_emit = await self._are_set_on_chain(weight_keys, weight_vals)

                if not did_emit and (time.time() - start_time) > timeout:
                    # ---- Emit caused timeout  -----
                    message = "Timed-out while waiting for inclusion."
                    return Metagraph.EmitTimeoutError, message

                elif not did_emit:
                    # ---- Did not emit, but waiting for inclusion -----
                    await asyncio.sleep(3)
                    continue

                else:
                    # --- Did emit, return latest chain weights ----
                    messages = "Successful emission"
                    return Metagraph.EmitSuccess, messages

        message = 'Should never get here'
        logger.critical(message)
        return Metagraph.EmitUnknownError, message

    async def _are_set_on_chain(self, weight_keys, weight_vals) -> bool:
        r""" Returns true if the passed key and vals are set on chain.
        """
        cmap = {}
        chain_keys = await self.subtensor_client.weight_keys(self.__keypair.public_key)
        chain_vals = await self.subtensor_client.weight_vals(self.__keypair.public_key)
        if chain_keys != None and chain_vals != None:
            n_same = 0
            for key, val in list(zip(chain_keys, chain_vals)):
                cmap[key] = val
            for key, val in list(zip(weight_keys, weight_vals)):
                if key in cmap:
                    if cmap[key] == val:
                        n_same += 1
            if n_same == len(weight_vals):
                return True
            else:
                return False
        else:
            return False 


    def _convert_weights_to_ints(self, weights: List[float]) -> Tuple[List[str], List[int]]:
        r""" Converts weights into integer u32 representation that sum to MAX_INT_WEIGHT.
        Returns:
            keys: (List[str]):
                List of pubkeys associated with each weight from vals.
            vals: (List[int]):
                List of u32 integer representations of floating point weights.
        """
        remainder = MAX_INT_WEIGHT
        weight_vals = []
        for i, val in enumerate(weights):
            int_val = int(float(val) * int(MAX_INT_WEIGHT)) # convert to int representation.
            remainder -= int_val

            # ---- Fix remainders and overflows ----
            if remainder < 0:
                int_val = int_val + remainder
                remainder = 0

            if i == (len(weights) -1) and remainder > 0: # last item.
                int_val += remainder
                remainder = 0
                
            weight_vals.append(int_val) # int weights sum to MAX_INT_WEIGHT.

        # ---- Get Pub keys ----
        weight_keys = []
        for index in range(self.state.n):
            neuron = self.state.neurons[index]
            weight_keys.append( neuron.public_key ) # Gets the pubkey at this index.

        return weight_keys, weight_vals


