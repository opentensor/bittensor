# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.

import argparse
import asyncio
import copy
import pandas as pd
import json
import math
import numpy
import random
import time
import torch
import tqdm.asyncio

from munch import Munch
from termcolor import colored
from loguru import logger
from typing import List, Tuple, List

import bittensor
import bittensor.config as config_utils
import bittensor.utils.networking as net
from bittensor.subtensor import Subtensor
from bittensor.crypto.keyfiles import KeyFileError

MAX_INT_WEIGHT = 4294967295 # Max weight value on chain.

class ChainState():
    """
    Describes and maintains the current state of the subtensor blockchain. 
    """
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
                raise ValueError('received inconsistent uid - pubey pairing with uid{}, pubkey{} and expected uid {}'.format(uid, pubkey, self.uids[index]))
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
    """ Maintains the chain state as a torch object.

        Args:
            tau (:obj:`int`): 
                current, per block, token inflation rate.

            block (:obj:`int`):
                state block number.

            uids (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n)`):
                UIDs for each neuron ordered by index.
            
            stake (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n)`):
                Stake balance for each neuron ordered by index.
                
            lastemit (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n)`):
                Last emission call for each neuron ordered by index.

            weights (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n, metagraph.n)`):
                Full weight matrix on chain.

            neurons (List[bittensor.proto.Neuron]) 
                List of endpoints on the network.

    """
    def __init__(self):
        self.tau = 0.5
        self.block = 0
        self.n = 0
        self.uids = torch.tensor([])
        self.stake = torch.tensor([])
        self.lastemit = torch.tensor([])
        self.weights = torch.tensor([[]])
        self.public_keys = []
        self.neurons = []

    def write_to_file(self, filepath: str ):
        json_data = {
            'n': self.n,
            'block': self.block,
            'tau': self.tau,
            'uids': self.uids.tolist(),
            'stake': self.stake.tolist(),
            'lastemit': self.lastemit.tolist(),
            'weights': self.weights.tolist(),
            'public_keys': self.public_keys,
            'neurons': [ {'uid': n.uid, 'ip': n.address, 'port': n.port, 'ip_type': n.ip_type, 'modality': n.modality, 'hotkey': n.public_key} for n in self.neurons]
        }
        with open( filepath, 'w') as fp:
            json.dump(json_data, fp)

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
    """
    Maintains the chain state as a torch object.
    """
    def __init__(   
            self, 
            config: 'Munch' = None, 
            subtensor: 'bittensor.subtensor.Subtensor' = None,
            **kwargs,
        ):
        r""" Initializes a new Metagraph chain interface.
            Args:
                config (:obj:`Munch`, `optional`): 
                    metagraph.Metagraph.config()
                subtensor (:obj:`bittensor.subtensor.Subtensor`, `optional`):
                    subtensor interface utility.
        """
        if config == None:
            config = Metagraph.default_config()
        # bittensor.config.Config.update_with_kwargs(config.metagraph, kwargs) 
        Metagraph.check_config(config)
        self.config = config

        if subtensor == None:
            subtensor = bittensor.subtensor.Subtensor( self.config )
        self.subtensor = subtensor

        # Chain state as cache and torch object.
        self.last_sync_block = 0
        self.cache = ChainState()
        self.state = TorchChainState.from_cache(self.cache)

    @staticmethod
    def default_config() -> Munch:
        # Parses and returns a config Munch for this object.
        parser = argparse.ArgumentParser(); 
        Metagraph.add_args(parser) 
        config = config_utils.Config.to_config(parser); 
        return config

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        bittensor.subtensor.Subtensor.add_args( parser )
        
    @staticmethod   
    def check_config(config: Munch):
        pass

    @property
    def tau(self) -> torch.FloatTensor:
        r""" tau: the chain per block inflation rate.
            
            Returns:
                tau (:obj:`torchFloatTensor` of shape :obj:`(1)`):
                    Current chain inflation rate.
        """
        return self.state.tau

    @property
    def n(self) -> int:
        r""" Return the number of known neurons on chain.
            
            Returns:
                n (int):
                    number of known neurons.

        """
        return self.state.n

    @property
    def block(self) -> int:
        r""" Return the block number.

             Returns:
                block (:obj:`int`):
                    local chain state block number.
        """
        return self.state.block

    @property
    def lastemit(self) -> torch.LongTensor:
        r""" Returns neuron last update block.
            
            Returns:
                lastemit (:obj:`int`):
                    last emit time.
        """
        return self.state.lastemit

    @property
    def uids(self) -> torch.LongTensor:
        r""" Returns uids of each neuron. Uids are synonymous with indices.
            Returns:
                uids (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n)`):
                    unique id for each neuron.
        """
        return self.state.uids

    @property
    def stake(self) -> torch.FloatTensor:
        r""" Returns neuron stake values.
            
            Returns:
                stake (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                    stake of each known neuron.

        """
        return self.state.stake

    @property
    def S(self) -> torch.FloatTensor:
        r""" Returns neurons stake values.
             
             Returns:
                S (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                    Stake of each known neuron.
        """
        return self.state.stake

    @property
    def I(self) -> torch.FloatTensor:
        r""" Returns neuron incentives: tau * R / sum(R)
        
            Returns:
                I (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                    Block incentive for each neuron. 
        """
        I =  (self.tau * self.ranks) / torch.sum(self.ranks)
        I = torch.where(torch.isnan(I), torch.zeros_like(I), I)
        return I.view(self.state.n)

    @property
    def ranks(self) -> torch.FloatTensor:
        r""" Returns neuron ranks: W^t * S
           
            Returns:
                ranks (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                    Rank of each neuron.

        """
        if self.W.shape[0] == 0:
            return torch.tensor([])
        else:
            S = self.S.view(self.state.n, 1)
            W = torch.transpose(self.W.view(self.state.n, self.state.n), 0, 1)
            R = torch.matmul(W, S).view(self.state.n)
        return R

    @property
    def R(self) -> torch.FloatTensor:
        r""" Returns neuron ranks: W^t * S
             
             Returns:
                rank (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                    Rank of each neuron.
        """
        return self.ranks

    @property
    def neurons(self) -> List[bittensor.proto.Neuron]:
        r""" Return neuron endpoint information for each neuron.
            
            Returns:
                neurons (:obj:`List[bittensor.proto.Neuron]` of shape :obj:`(metagraph.n)`):
                    Endpoint information for each neuron.

        """
        return self.state.neurons

    @property
    def public_keys(self) -> List[str]:
        r""" Return the ordered public keys for state neurons.
        
            Returns:
                public_keys (:obj:`List[str]` of shape :obj:`(metagraph.n)`):
                    Public keys of each neuron.

        """
        return self.state.public_keys

    @property
    def W(self) -> torch.FloatTensor:
        r""" Return full weight matrix from chain.
             Returns:
                W (:obj:`torch.LongFloat` of shape :obj:`(metagraph.n, metagraph.n)`):
                    Weight matrix.
        """
        return self.state.W

    @property
    def weights(self) -> torch.FloatTensor:
        r""" Return full weight matrix from chain.
             Returns:
                W (:obj:`torch.LongFloat` of shape :obj:`(metagraph.n, metagraph.n)`):
                    Weight matrix.
        """
        return self.state.W

    def sync(self):
        r""" Synchronizes the local self.state with the chain state.
        """
        # TODO (const) this should probably be a background process
        # however, it makes it difficult for the user if the state changes in
        # the background.
        current_block = self.subtensor.get_current_block()
        # ---- Update cache ----
        self._sync_cache()
        self.last_sync_block = current_block

        # --- Update torch state
        self.state = TorchChainState.from_cache(self.cache)
        self.state.block = current_block

    def _sync_cache(self):
        r""" Synchronizes the local self.state with the chain state.
        """
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        loop.run_until_complete(self._async_sync_cache())

    async def _async_sync_cache(self):
        r""" Async: Makes calls to chain updating local chain cache with newest info.
        """
        # Make asyncronous calls to chain filling local state cache.
        calls = []
        current_block = await self.subtensor.async_get_current_block()
        active = dict( await self.subtensor.async_get_active() )
        last_emit = dict( await self.subtensor.async_get_last_emit() )
        stake = dict( await self.subtensor.async_get_stake() )

        for pubkey, uid in active.items():
            if last_emit[ uid ] > self.last_sync_block:
                calls.append( self._poll_uid ( pubkey, uid, stake[uid], last_emit[uid] ) )

        for call in tqdm.asyncio.tqdm.as_completed( calls ):
            await call

    async def _poll_uid(self, pubkey: str, uid:int, stake_amount:int, last_emit: int ) -> bool:
        r""" Polls info info for a specfic public key.
        """
        try:
            w_uids = await self.subtensor.async_weight_uids_for_uid( uid )
            w_vals = await self.subtensor.async_weight_vals_for_uid( uid )
            neuron = await self.subtensor.async_get_neuron_for_uid ( uid )
            self.cache.add_or_update(
                pubkey = pubkey, 
                ip = neuron['ip'], 
                port = neuron['port'], 
                uid = neuron['uid'], 
                p_type = neuron['ip_type'], 
                modality = neuron['modality'], 
                lastemit = last_emit, 
                stake = stake_amount, 
                w_uids = w_uids, 
                w_vals = w_vals
            )

        except Exception as e:
            logger.trace('error while polling uid: {} with error: {}', uid, e )
            #traceback.print_exc()

    def __str__(self):
        if self.n != 0:
            peers_online = torch.numel(torch.where( self.block - self.lastemit < 1000 )[0])
        else:
            peers_online = 0
        peers_online = torch.numel(torch.where( self.block - self.lastemit < 1000 )[0])
        return '<green>Metagraph:</green> block:<cyan>{}</cyan>, inflation_rate:<cyan>{}</cyan>, staked:<green>\u03C4{}</green>/<cyan>\u03C4{}</cyan>, active:<green>{}</green>/<cyan>{}</cyan>\n'.format(self.block, self.tau.item(), torch.sum(self.S), self.block/2, peers_online, self.n)

    def __full_str__(self):
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
        return 'Metagraph:\nuid: {}, inflation_rate: {} block: {} n_neurons: {} \n'.format(self.uid, self.tau, self.block, self.n) + df.to_string(na_rep = '', max_rows=5000, max_cols=25, min_rows=25, line_width=1000, float_format = lambda x: '%.3f' % x, col_space=1, justify='left')

    def __to_tensorboard__(self, tensorboard, global_step):
        tensorboard.add_scalar('Metagraph/neurons', self.n, global_step)
        tensorboard.add_scalar('Metagraph/inflation_rate', self.tau, global_step)


