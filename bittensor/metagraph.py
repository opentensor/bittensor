'''
The MIT License (MIT)
Copyright © 2021 Opentensor.ai

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.
'''
import asyncio
import copy
import argparse
import bittensor
import pandas as pd
import math
import numpy
import time
import torch
import traceback

from munch import Munch
from termcolor import colored
from loguru import logger
from typing import List, Tuple, List

import bittensor
import bittensor.config as config_utils
import bittensor.utils.networking as net
from bittensor.subtensor.client import WSClient
from bittensor.exceptions.handlers import rollbar

MAX_INT_WEIGHT = 4294967295 # Max weight value on chain.

class ChainState():
    def __init__(self):
        # Cached values.
        self.n = 0
        self.uids = []
        self.stake = []
        self.lastemit = []
        self.weight_uids = []
        self.weight_vals = []
        self.neurons = []
        self.index_for_uid = {}
        self.index_for_pubkey = {}
        self.pubkey_for_index = {}

    def add_or_update(self, pubkey:str, ip: int, port: int, uid: int, ip_type: int, modality: int, lastemit: int, stake: int, w_uids: List[int], w_vals: List[int]):
        address_str = net.int_to_ip(ip)
        neuron = bittensor.proto.Neuron(
            version = bittensor.__version__,
            public_key = pubkey,
            address = address_str,
            port = int(port),
            ip_type = int(ip_type),
            modality = int(modality),
            uid = int(uid),
        )
        if pubkey in self.index_for_pubkey:
            index = self.index_for_pubkey[pubkey]
            if self.uids[index] == uid:
                self.neurons[index] = neuron
                self.stake[index] = float(stake) / 1000000000 
                self.lastemit[index] = int(lastemit)
                self.weight_uids[index] = list(w_uids)
                self.weight_vals[index] = list(w_vals)
                self.uids[index] = int(uid)
            else:
                raise ValueError('recieved inconsistent uid - pubey pairing with uid{}, pubkey{} and expected uid {}'.format(uid, pubkey, self.uids[index]))
        else:
            index = self.n
            self.n += 1
            self.index_for_pubkey[pubkey] = index
            self.pubkey_for_index[index] = pubkey
            self.neurons.append(neuron)
            self.stake.append(float(stake) / 1000000000)
            self.lastemit.append(int(lastemit))
            self.weight_uids.append(list(w_uids))
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

            neurons: (List[bittensor.proto.Neuron]) 
                List of endpoints on the network.

    """
    def __init__(self):
        self.tau = torch.tensor([0.5], dtype = torch.float32)
        self.block = 0
        self.n = 0
        self.uids = torch.tensor([])
        self.indices = torch.tensor([])
        self.stake = torch.tensor([])
        self.lastemit = torch.tensor([])
        self.W = torch.tensor([[]])
        self.neurons = []
        self.uid_for_pubkey = {}
        self.index_for_uid = {}

    @staticmethod
    def from_cache(cache: ChainState):
        r""" Deep copies from the chain state.
        """
        # Deep copies chain state into metagraph state.
        state = TorchChainState()
        state.n = cache.n
        state.tau = torch.tensor([0.5], dtype = torch.float32)
        state.neurons = copy.deepcopy(cache.neurons)
        state.indices = torch.tensor(range(state.n), dtype=torch.int64)
        state.uids = torch.tensor(copy.deepcopy(cache.uids), dtype=torch.int64)
        state.lastemit = torch.tensor(copy.deepcopy(cache.lastemit), dtype=torch.int64)
        state.stake = torch.tensor(copy.deepcopy(cache.stake), dtype=torch.float32)
        for idx, (uid, n) in enumerate(list(zip(cache.uids, cache.neurons))):
            state.uid_for_pubkey[n.public_key] = uid
            state.index_for_uid[uid] = idx
        weights_numpy = numpy.zeros( (state.n, state.n) )
        for i in range(state.n):
            uids = cache.weight_uids[i]
            vals = cache.weight_vals[i]
            val_sum = sum(vals)
            for uid, val in list(zip(uids, vals)):
                if uid in cache.index_for_uid:
                    j = cache.index_for_uid[uid]
                    if val_sum != 0:
                        weights_numpy[i, j] = float(val) / float(val_sum)
                    else:
                        weights_numpy[i, j] = 0
        state.W = torch.tensor(weights_numpy, dtype=torch.float32)
        return state

class Metagraph():

    def __init__(self, config: 'Munch' = None, wallet: 'bittensor.wallet.Wallet' = None):
        r""" Initializes a new Metagraph chain interface.
            Args:
                config (:obj:`Munch`, `optional`): 
                    metagraph.Metagraph.config()
                wallet (:obj:`bittensor.nucleus.Nucleus`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
        """
        if config == None:
            config = Metagraph.build_config()
        self.config = config

        if wallet == None:
            wallet = bittensor.wallet.Wallet( self.config )
        self.wallet = wallet

        # Client for talking to chain.
        self.subtensor_client = WSClient(self.config.metagraph.chain_endpoint, keypair = self.wallet.keypair)

        # This neurons metadata on chain, initially None, filled on subscribe.
        self.uid = None
        self.metadata = None

        # Chain state cached before converted into the torch state.
        self.cache = ChainState()

        # Chain state as torch values.
        self.state = TorchChainState.from_cache(self.cache)

    @staticmethod
    def build_config() -> Munch:
        # Parses and returns a config Munch for this object.
        parser = argparse.ArgumentParser(); 
        Metagraph.add_args(parser) 
        config = config_utils.Config.to_config(parser); 
        Metagraph.check_config(config)
        return config

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        # TODO(const): check this endpoint in check_config.
        bittensor.wallet.Wallet.add_args( parser )
        try:
            parser.add_argument('--metagraph.chain_endpoint', default='localhost:9944', type=str, 
                                help='''The subtensor chain endpoint. The likely choices are:
                                        -- localhost:9944 -- (your locally running node)
                                        -- feynman.akira.bittensor.com:9944 (testnet)
                                        -- feynman.kusanagi.bittensor.com:12345 (mainnet)
                                        If this value remains a default (localhost) you will need to 
                                        run a subtensor node on your localbox in order to connect your neuron.
                                        (See: docs/running_a_validator.md)''')
            parser.add_argument('--metagraph.stale_emit_filter', default=10000, type=int, 
                                help='''The metagraph filters neurons with last emit beyond this many blocks.
                                        Note, this is used to trim the graph size,
                                        but may change your incentive mechanism view.''')
        except:
            pass
        
    @staticmethod   
    def check_config(config: Munch):
        bittensor.wallet.Wallet.check_config( config )

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
        r""" Returns the incentive value from each known neuron to you.
        Returns
            incentive: (:obj:`torch.FLoatTensor` of shape :obj:`(metagraph.n)`):
                inflation incentive from each known neuron.
        """
        incentive = self.tau * self.col * self.stake
        return incentive

    @property
    def I(self) -> torch.FloatTensor:
        r""" Returns the inflation incentive for each peer per block.
        Returns
            I: (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                stake of each known neuron.
        """
        I =  (self.tau * self.ranks) / torch.sum(self.ranks)
        I = torch.where(torch.isnan(I), torch.zeros_like(I), I)
        return I

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
        return self.ranks

    @property
    def row(self) -> torch.FloatTensor:
        r""" Returns this neuron's row weights, i.e. weights to other neurons.
        Returns
            row: (:obj:`torch.LongFloat` of shape :obj:`(metagraph.n)`):
                w_{i,*}
        """
        if self.uid == None:
            raise ValueError('Must be subscribed before you can return your row')
        try:
            self_idx = self.state.index_for_uid[ self.uid ] 
            return self.state.W[self_idx, :]
        except:
            logger.error('your uid is not in self.state with state.uids {} and uid {}'.format(self.state.uids, self.uid))
            return torch.tensor([])

    @property
    def col(self) -> torch.FloatTensor:
        r""" Returns this neuron's col weights, i.e. weights from other neurons to us.
        Returns
            col: (:obj:`torch.LongFloat` of shape :obj:`(metagraph.n)`):
                w_{*,i}
        """
        if self.uid == None:
            raise ValueError('Must be subscribed before you can return your col')
        try:
            self_idx = self.state.index_for_uid[ self.uid ] 
            return self.state.W[:, self_idx]
        except:
            logger.error('your uid is not in self.state with state.uids {} and uid {}'.format(self.state.uids, self.uid))
            return torch.tensor([])

    @property
    def W(self) -> torch.FloatTensor:
        r""" Full chain weight matrix for each neuron.
        Returns
            W: (:obj:`torch.LongFloat` of shape :obj:`(metagraph.n, metagraph.n)`):
                w_ij of each neuron.
        """
        return self.state.W

    @property
    def neurons(self) -> List[bittensor.proto.Neuron]:
        r""" Return neuron endpoint information for each neuron.
        Returns
            neurons: (:obj:`List[bittensor.proto.Neuron]` of shape :obj:`(metagraph.n, metagraph.n)`):
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

    def uids_to_indices(self, uids: torch.Tensor) -> torch.LongTensor:
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

    def uids_to_neurons(self, uids: torch.Tensor) -> List[bittensor.proto.Neuron]:
        r""" Returns a list with neurons for each uid.
        Args:
            uids: (torch.LongTensor)
                uids into neuron protos
        Returns:
            neurons: (List[bittensor.proto.Neuron]): 
                neuron info ordered by passed uids.
        """
        response = []
        indices = self.uids_to_indices(uids)
        for idx in indices.tolist():
            response.append(self.state.neurons[idx])
        return response

    def neurons_to_uids(self, neurons: List[bittensor.proto.Neuron]) -> torch.LongTensor:
        r""" Returns uids associated with the passed neurons.
        Args:
            neurons: (List[bittensor.proto.Neuron]): 
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
        chain_uids = await self.subtensor_client.weight_uids(self.metadata['uid'])
        chain_weights = await self.subtensor_client.weight_vals(self.metadata['uid'])
        if chain_weights == None or len(chain_weights) == 0:
            return torch.tensor([])

        else:
            # ---- To be filled ----
            return_val = torch.zeros(self.state.n)

            weight_sum = sum(chain_weights)
            if weight_sum != MAX_INT_WEIGHT:
                logger.error('Chain weights do not sum to {} with vals {}', MAX_INT_WEIGHT, chain_weights)

            # ---- Fill torch tensor ----
            for uid, weight in list(zip(chain_uids, chain_weights)):
                if uid not in self.state.index_for_uid:
                    logger.critical('uid {} on chain not in state.index', uid)
                    continue
                else:
                    idx = self.state.index_for_uid[ uid ]
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

    def sync(self):
        r""" Synchronizes the local self.state with the chain state.
        """
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        loop.run_until_complete(self.async_sync())

    async def async_sync(self):
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
        active = dict( await self.subtensor_client.get_active() )
        last_emit = dict( await self.subtensor_client.get_last_emit_data() )
        calls.append ( self._poll_uid ( self.wallet.keypair.public_key, self.uid ) )        
        for pubkey, uid in active.items():
            if uid in last_emit:
                emit_block = last_emit[ uid ]
                if (current_block - emit_block) < self.config.metagraph.stale_emit_filter:
                        calls.append( self._poll_uid ( pubkey, uid ) )
        await asyncio.gather(*calls)

    async def _poll_uid(self, pubkey: str, uid:int):
        r""" Polls info info for a specfic public key.
        """
        try:
            stake = await self.subtensor_client.get_stake_for_uid( uid )
            lastemit = await self.subtensor_client.get_last_emit_data_for_uid( uid )
            w_uids = await self.subtensor_client.weight_uids_for_uid( uid )
            w_vals = await self.subtensor_client.weight_vals_for_uid( uid )
            neuron = await self.subtensor_client.get_neuron_for_uid ( uid )
            self.cache.add_or_update(pubkey = pubkey, ip = neuron['ip'], port = neuron['port'], uid = neuron['uid'], ip_type = neuron['ip_type'], modality = neuron['modality'], lastemit = lastemit, stake = stake.rao, w_uids = w_uids, w_vals = w_vals)
        except Exception as e:
            pass
            #logger.error("Exception occurred: {}".format(e))
            #traceback.print_exc()

    ConnectSuccess = 1
    ConnectUnknownError = 2
    ConnectTimeout = 3
    def connect(self, timeout:int) -> Tuple[int, str]:
        r""" Synchronous: Connects to the chain.
        Args:
            timeout (int):
                Time to wait before connecting times out.
        Returns:
            see: _try_async_connect
        """
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_connect(timeout))

    async def async_connect(self, timeout: int) -> Tuple[int, str]:
        r""" Async: Makes and awaits for a connection to the chain.
        Args:
            timeout (int):
                Time to wait before connecting times out.
        Returns:
            see: _try_async_connect
        """
        # ---- Try Connection ----
        try:
            code, message = await self._try_async_connect(timeout)

            if code == Metagraph.ConnectSuccess:
                logger.info('Successfully connected to chain endpoint: {}', self.config.metagraph.chain_endpoint)
                return code, message

            elif code == Metagraph.ConnectUnknownError:
                logger.error('Connection threw an unknown error: {}', message)
                return code, message

            elif code == Metagraph.ConnectTimeout:
                logger.error('Connection timeout {}', message)
                return code, message

        except Exception as e:
            logger.error('Connection threw an uncaught error {}', e)
            return Metagraph.ConnectUnknownError, e

    async def _try_async_connect(self, timeout: int) -> Tuple[int, str]:
        r""" Makes connection attempts to the chain, continuing to attempt until timeout.

        Args:
            timeout (int):
                Time to wait before connecting times out.
        Raises:
            code (ENUM) {}
                ConnectSuccess:
                    Raised when the connection is a success before the timeout

                ConnectUnknownError:
                    UnknownError during connecting to chain.

                Connectimeout:
                    Raised when the attempted connection fails after timeout.
            }
            message:
                Message associated with code. 
        """
        # ---- Make Chain connection attempt  ----
        start_time = time.time()
        while True:
            # ---- Make connection call.
            try:
                self.subtensor_client.connect()
            except Exception as e:
                return Metagraph.ConnectUnknownError, e
            
            # ---- Wait for connection future to reture, or timeout.
            is_connected = self.subtensor_client.is_connected()
            try:
                await asyncio.wait_for(is_connected, timeout=timeout)
            except asyncio.TimeoutError:
                return Metagraph.ConnectTimeout, "Timeout"

            # ---- Return on success.
            if is_connected:
                return Metagraph.ConnectSuccess, "Success"

            # ---- Retry or timeout.
            elif (time.time() - start_time) > timeout:
                return Metagraph.ConnectTimeout, "Timeout"
            else:
                await asyncio.sleep(1)
                continue

    SubscribeSuccess = 1
    SubscribeUnknownError = 2
    SubscribeTimeout = 3
    SubscribeNotConnected = 4
    def subscribe(self, timeout) -> Tuple[int, str]:
        r""" Syncronous: Makes a subscribe request to the chain. Waits for subscription inclusion or returns False
        Returns:
            see: _try_async_subscribe
        """
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self.async_subscribe(timeout))

    async def async_subscribe (self, timeout) -> Tuple[int, str]:
        r""" Async: Makes a subscribe request to the chain. Waits for subscription inclusion or returns False
        Returns:
            see: _try_async_subscribe
        """

        # ---- Try Subscription ----
        try:
            code, message = await self._try_async_subscribe(timeout)

            if code == Metagraph.SubscribeSuccess:
                logger.info('Successfully subcribed with: {}', self.metadata)
                return code, message

            elif code == Metagraph.SubscribeNotConnected:
                logger.error('Subscription failed because you are not connected to a chain endpoint, call metagraph.connect() first')
                return code, message

            elif code == Metagraph.SubscribeUnknownError:
                logger.error('Subscription threw an unknown error: {}', message)
                return code, message

            elif code == Metagraph.SubscribeTimeout:
                logger.error('Subscription timeout {}', message)
                return code, message

        except Exception as e:
            logger.error('Subscription threw an uncaught error {}', e)
            return Metagraph.SubscribeUnknownError, e
        
    async def _try_async_subscribe(self, timeout: int):
        r""" Makes subscription attempts to the chain, continuing to attempt until timeout and finally waiting for inclusion.

        Args:
            timeout (int):
                Time to wait before subscription times out.

        Raises:
            code (ENUM) {}
                SubscribeSuccess:
                    Raised when the subscription is a success before the timeout

                SubscribeUnknownError:
                    UnknownError during subscription.

                SubscribeTimeout:
                    Raised when the attempted subscription fails after timeout.

                SubscribeNotConnected:
                    Raised if a subscription is attempted while before metagraph.connect is called.
                    Mush call metagraph.connect() before metagraph.subscribe()
            }
            message:
                Message associated with code. 
        """
        # --- Check that we are already connected to the chain.
        is_connected = self.subtensor_client.is_connected()
        try:
            await asyncio.wait_for(is_connected, timeout = 10)
        except asyncio.TimeoutError:
            return Metagraph.SubscribeNotConnected, "Not connected"
        if not is_connected:
            return Metagraph.SubscribeNotConnected, "Not connected"

        # ---- Make Subscription transaction ----
        logger.info("Subscribing to subtensor")
        main_start_time = time.time()
        while True:

            subscribe_start_time = time.time()
            try:
                await self.subtensor_client.subscribe(self.config.axon.external_ip, self.config.axon.external_port, bittensor.proto.Modality.TEXT, self.wallet.coldkey)

            except Exception as e:
                if (time.time() - subscribe_start_time) > 8:
                    # --- Timeout during emit call ----
                    message = "Timed-out with Unknown Error while trying to make the subscription call. With last exception {}".format(e)
                    return Metagraph.SubscribeUnknownError, message

                else:
                    # --- Wait for inclusion, no error.
                    logger.trace('Error while attempting subscription {}', e)
                    continue

            # ---- Wait for inclusion ----
            check_start_time = time.time()
            while True:
                try:
                    # ---- Request info from chain ----
                    self.uid = await self.subtensor_client.get_uid_for_pubkey(self.wallet.keypair.public_key)
                except Exception as e:
                    # ---- Catch errors in request ----
                    message = "Subscription threw an unknown exception {}".format(e)
                    return Metagraph.SubscribeUnknownError, message

                if self.uid != None:
                    # ---- Request info from chain ----
                    self.metadata = await self.subtensor_client.neurons(self.uid)
                    if not self.metadata:
                        return Metagraph.SubscribeUnknownError, "Critical error: There no metadata returned"

                    # ---- Subscription was a success ----
                    return Metagraph.SubscribeSuccess, "Subscription success"

                elif time.time() - check_start_time > 8:
                    break

                else:
                    # ---- wait -----
                    await asyncio.sleep(1)
            
            if time.time() - main_start_time > 90:
                return Metagraph.SubscribeTimeout, "Timeout occured while trying to subscribe to the chain, potentially the chain is recieving too many subsribe requests at this time."

                 

        # ---- ?! WaT ?! ----
        logger.critical('Should not get here')
        return Metagraph.SubscribeUnknownError, 'Should not get here'

      
    def set_weights(self, weights: torch.FloatTensor, wait_for_inclusion = False, timeout = 12):
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
    EmitResultUnknown = 6
    EmitNoOp = 7
    async def async_emit(self, weights: torch.FloatTensor, wait_for_inclusion = False, timeout = 12) -> Tuple[int, str]:
        r""" Calls _try_async_emit, logs and returns results based on code. Only fails on an uncaught Exception.
        
        Args:
            weights: (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                Weights to set on chain.
            wait_for_inclusion: (bool):
                If true, the call waits for block-inclusion before continuing or throws error after timeout.
            timeout: (int, default = 12 sec):
                Time to wait for inclusion before raising a caught error.

        Returns:
            see: _try_async_emit
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

        return code, message


    async def _try_async_emit(self, weights: torch.FloatTensor, wait_for_inclusion = False, timeout = 12) -> Tuple[int, str]:
        r""" Makes emit checks, emits to chain, and raises one of the following errors.
        Args:
            weights: (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                Weights to set on chain.

            wait_for_inclusion: (bool):
                If true, the call waits for block-inclusion before continuing or throws error after timeout.

            timeout: (int, default = 12 sec):
                Time to wait for inclusion before raising a caught error.

        Returns:
            code (ENUM) {
                EmitSuccess (ENUM):
                    Raised when try_async_emit emits weights successfully with known result.

                EmitNoOp (ENUM):
                    Raised when calling emit does not change weights on chain.

                EmitUnknownError (ENUM):
                    UnknownError during emit.

                EmitValueError (ENUM):
                    Raised during emission when passed weights are not properly set.

                EmitTimeoutError (ENUM):
                    Raised during emission during a timeout.

                EmitResultUnknown (ENUM):
                    Called when an emit step end without a known result, for instance, 
                    if the user has wait_for_inclusion = False.
            }
            message:
                Message associated with code.
        """
        # --- Check type ----
        if not isinstance(weights, torch.Tensor):
            message = "Error trying to set weights on chain. Got weights type {}, but weights must be of type {}".format(type(weights), torch.Tensor)
            return Metagraph.EmitValueError, message
        
        # --- Check nan ---
        if torch.any(weights.isnan()).item():
            message = "Error trying to set weight on chain. Got nan values {}".format(weights)
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
            weight_uids, weight_vals = self.convert_weights_to_emit(weights)
        except Exception as e:
            message = "Unknown error when converting weights to ints with weights {} and error {}".format(weights, e)
            return Metagraph.EmitUnknownError, message

        # ---- Check sum ----
        weight_sum = sum(weight_vals)
        if weight_sum != MAX_INT_WEIGHT:
            message = "Error trying to set weights on chain. Converted weights do not sum to {} with weights_vals {}".format(MAX_INT_WEIGHT, weight_vals)
            return Metagraph.EmitValueError, message

        # ---- Check NO-OP ----
        if await self._are_set_on_chain(weight_vals, weight_uids):
            message = "When trying to set weights on chain. Weights are unchanged, nothing to emit."
            return Metagraph.EmitNoOp, message

        # ---- Emit ----
        start_time = time.time()
        while True:
            try:
                # --- Make emission call ----
                logger.debug('Emit -> {} {}', weight_uids, weight_vals)
                await self.subtensor_client.set_weights(weight_uids, weight_vals)
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
                    logger.info('retry emit...')
                    await asyncio.sleep(3) # To avoid ddos-ing the chain.
                    continue

        # --- Wait for inclusion ----
        if not wait_for_inclusion:
            message = "Emit ended but we don't know if weights were set on chain"
            return Metagraph.EmitResultUnknown, message

        else:
            while True:
                did_emit = await self._are_set_on_chain(weight_uids, weight_vals)

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

    async def _are_set_on_chain(self, weight_uids, weight_vals) -> bool:
        r""" Returns true if the passed key and vals are set on chain.
        """
        cmap = {}
        chain_uids = await self.subtensor_client.weight_uids_for_uid(self.uid)
        chain_vals = await self.subtensor_client.weight_vals_for_uid(self.uid)
        if chain_uids != None and chain_vals != None:
            n_same = 0
            for uid, val in list(zip(chain_uids, chain_vals)):
                cmap[uid] = val
            for uid, val in list(zip(weight_uids, weight_vals)):
                if uid in cmap:
                    if cmap[uid] == val:
                        n_same += 1
            if n_same == len(weight_vals):
                return True
            else:
                return False
        else:
            return False 


    def convert_weights_to_emit(self, weights: List[float]) -> Tuple[List[str], List[int]]:
        r""" Converts weights into integer u32 representation that sum to MAX_INT_WEIGHT.
        Returns:
            keys: (List[str]):
                List of pubkeys associated with each weight from vals.
            vals: (List[int]):
                List of u32 integer representations of floating point weights.
        """
        remainder = MAX_INT_WEIGHT
        weight_vals = []
        weight_uids = []
        pos_self_uid = -1
        for i, val in enumerate(weights):
            int_val = int(float(val) * int(MAX_INT_WEIGHT)) # convert to int representation.
            remainder -= int_val
            uid_i = self.state.uids.tolist()[i]

            # ---- Fix remainders and overflows ----
            if remainder < 0:
                int_val = int_val + remainder
                remainder = 0

            if i == (len(weights) -1) and remainder > 0: # last item.
                int_val += remainder
                remainder = 0

            # Do not add zero values. 
            if int_val != 0:
                weight_vals.append( int_val ) # int weights sum to MAX_INT_WEIGHT.
                weight_uids.append( uid_i ) # Gets the uid at this index

            if uid_i == self.uid:
                pos_self_uid = i

        # Places the self weight in the first position if it exists
        if pos_self_uid != -1 and len(weight_uids) > 1:
            weight_uids.insert(0, weight_uids.pop(pos_self_uid))
            weight_vals.insert(0, weight_vals.pop(pos_self_uid))
        return weight_uids, weight_vals

    def __str__(self):
        uids = self.state.uids.tolist()
        rows = [self.S.tolist(), self.R.tolist(), self.I.tolist(), self.incentive.tolist(), self.row.tolist(), self.col.tolist()]
        for i in range(self.n):
            rows.append(self.W[i, :].tolist())
        df = pd.DataFrame(rows, columns=uids)
        df = df.rename(index={df.index[0]: 'S'})
        df = df.rename(index={df.index[1]: 'R'})
        df = df.rename(index={df.index[2]: 'I'})
        df = df.rename(index={df.index[3]: 'incentive'})
        df = df.rename(index={df.index[4]: 'row'})
        df = df.rename(index={df.index[5]: 'col'})
        for i in range(self.n):
            df = df.rename(index={df.index[i + 6]: uids[i]})
        df.rename_axis(colored('[uid]', 'red'), axis=1)
        return '\nMetagraph:\nuid: {}, inflation_rate: {} block: {} n_neurons: {} \n'.format(self.uid, self.tau.item(), self.block, self.n) + df.to_string(na_rep = '', max_rows=5000, max_cols=25, min_rows=25, line_width=1000, float_format = lambda x: '%.3f' % x, col_space=1, justify='left')

    def __to_tensorboard__(self, tensorboard, global_step):
        tensorboard.add_scalar('Metagraph/neurons', self.n, global_step)
        tensorboard.add_scalar('Metagraph/inflation_rate', self.tau.item(), global_step)


