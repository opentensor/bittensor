
# The MIT License (MIT)
# Copyright © 2021 Opentensor.ai

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
import copy
import pandas as pd
import math
import numpy
import random
import time
import torch

from munch import Munch
from termcolor import colored
from loguru import logger
from typing import List, Tuple, List

import bittensor
import bittensor.config as config_utils
from bittensor.subtensor import Subtensor

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
            
            indices (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n)`):
                Index of neurons, range(metagraph.n)

            stake (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n)`):
                Stake balance for each neuron ordered by index.
                
            lastemit (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n)`):
                Last emission call for each neuron ordered by index.

            weights (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                This neuron's weights W[,:]

            W (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n, metagraph.n)`):
                Full weight matrix on chain.

            neurons (List[bittensor.proto.Neuron]) 
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
    """
    Maintains the chain state as a torch object.
    """
    def __init__(   
            self, 
            config: 'Munch' = None, 
            wallet: 'bittensor.wallet.Wallet' = None,
            subtensor: 'bittensor.subtensor.Subtensor' = None
        ):
        r""" Initializes a new Metagraph chain interface.
            Args:
                config (:obj:`Munch`, `optional`): 
                    metagraph.Metagraph.config()
                wallet (:obj:`bittensor.wallet.Wallet`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
                subtensor (:obj:`bittensor.subtensor.Subtensor`, `optional`):
                    subtensor interface utility.
        """
        if config == None:
            config = Metagraph.build_config()
        self.config = config

        if wallet == None:
            wallet = bittensor.wallet.Wallet( self.config )
        self.wallet = wallet

        if subtensor == None:
            subtensor = bittensor.subtensor.Subtensor( self.config, self.wallet )
        self.subtensor = subtensor

        # Chain state as cache and torch object.
        self.last_sync = 0
        self.cache = ChainState()
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
        bittensor.wallet.Wallet.add_args( parser )
        bittensor.subtensor.Subtensor.add_args( parser )
        try:
            parser.add_argument('--metagraph.stale_emit_filter', default=10000, type=int, 
                                help='''The metagraph filters neurons with last emit beyond this many blocks.
                                        Note, this is used to trim the graph size,
                                        but may change your incentive mechanism view.''')
        except:
            pass
        
    @staticmethod   
    def check_config(config: Munch):
        bittensor.wallet.Wallet.check_config( config )
        bittensor.subtensor.Subtensor.check_config( config )

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
        r""" Return the block number when the chain state was updated.

             Returns:
                block (:obj:`int`):
                    local chain state block number.
        """
        return self.state.block

    @property
    def lastemit(self) -> torch.LongTensor:
        r""" Returns the last emit time for each known neuron.
            
            Returns:
                lastemit (:obj:`int`):
                    last emit time.
        """
        return self.state.lastemit

    @property
    def indices(self) -> torch.LongTensor:
        r""" Return the indices of each neuron in the chain state range(metagraph.n).
            
            Returns:
                indices (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n)`):
                    returned indices for each neuron.

        """
        return self.state.indices

    @property
    def uids(self) -> torch.LongTensor:
        r""" Returns unique ids for each neuron in the chain state.
            Returns:
                uids (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n)`):
                    unique id for each neuron.
        """
        return self.state.uids

    @property
    def uids(self) -> torch.LongTensor:
        r""" Returns unique ids for each neuron in the chain state.
            Returns:
                uids (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n)`):
                    unique id for each neuron.
        """
        return self.state.uids

    @property
    def stake(self) -> torch.FloatTensor:
        r""" Returns the stake held by each known neuron.
            
            Returns:
                stake (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                    stake of each known neuron.

        """
        return self.state.stake

    @property
    def S(self) -> torch.FloatTensor:
        r""" Returns the stake held by each known neuron.
             
             Returns:
                S (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                    stake of each known neuron.
        """
        return self.state.stake

    @property
    def tau(self) -> torch.FloatTensor:
        r""" tau: the chain per block inflation rate. i.e. 50
            
            Returns:
                tau (:obj:`torchFloatTensor` of shape :obj:`(1)`):
                    current chain inflation rate.
        """
        return self.state.tau

    @property
    def incentive(self) -> torch.FloatTensor:
        r""" Returns the incentive value from each known neuron to you.
            
            Returns:
                incentive (:obj:`torch.FLoatTensor` of shape :obj:`(metagraph.n)`):
                    inflation incentive from each known neuron.
        """
        incentive = self.tau * self.col * self.stake
        return incentive

    @property
    def I(self) -> torch.FloatTensor:
        r""" Returns the inflation incentive for each peer per block.
        
            Returns:
                I (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                    stake of each known neuron.
        """
        I =  (self.tau * self.ranks) / torch.sum(self.ranks)
        I = torch.where(torch.isnan(I), torch.zeros_like(I), I)
        return I

    @property
    def ranks(self) -> torch.FloatTensor:
        r""" Returns the ranks W^t * S
           
            Returns:
                ranks (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                    rank of each known neuron.

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
        r""" Returns ranks for each known neuron in the graph.
             
             Returns:
                rank (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                    rank of each known neuron.
        """
        return self.ranks

    @property
    def row(self) -> torch.FloatTensor:
        r""" Returns this neuron's row weights, i.e. weights to other neurons.
            
            Returns:
                row: (:obj:`torch.LongFloat` of shape :obj:`(metagraph.n)`):
                    `w_{i,*}`
                
        """
        self_uid = self.uid_for_pubkey( self.wallet.hotkey.public_key )
        if self_uid == -1:
            return torch.tensor([])
        try:
            self_idx = self.state.index_for_uid[ self_uid ] 
            return self.state.W[self_idx, :]
        except:
            logger.error('your uid is not in self.state with state.uids {} and uid {}'.format(self.state.uids, self_uid))
            return torch.tensor([])

    @property
    def col(self) -> torch.FloatTensor:
        r""" Returns this neuron's col weights, i.e. weights from other neurons to us.
            
             Returns:
                col (:obj:`torch.LongFloat` of shape :obj:`(metagraph.n)`):
                    `w_{*,i}`
        """
        self_uid = self.uid_for_pubkey( self.wallet.hotkey.public_key )
        if self_uid == -1:
            return torch.tensor([])
        try:
            self_idx = self.state.index_for_uid[ self_uid ] 
            return self.state.W[:, self_idx]
        except:
            logger.error('your uid is not in self.state with state.uids {} and uid {}'.format( self.state.uids, self_uid ))
            return torch.tensor([])

    @property
    def W(self) -> torch.FloatTensor:
        r""" Full chain weight matrix for each neuron.
             
             Returns:
                W (:obj:`torch.LongFloat` of shape :obj:`(metagraph.n, metagraph.n)`):
                    w_ij of each neuron.
        """
        return self.state.W

    @property
    def neurons(self) -> List[bittensor.proto.Neuron]:
        r""" Return neuron endpoint information for each neuron.
            
            Returns:
                neurons (:obj:`List[bittensor.proto.Neuron]` of shape :obj:`(metagraph.n, metagraph.n)`):
                    endpoint information for each neuron.

        """
        return self.state.neurons

    @property
    def public_keys(self) -> List[str]:
        r""" Return the ordered public keys for state neurons.
        
            Returns:
                public_keys (:obj:`List[str]` of shape :obj:`(metagraph.n)`):
                    public keys of all graph neurons.

        """
        return [n.public_key for n in self.state.neurons]

    @property
    def weights(self) -> torch.FloatTensor:
        r"""Return this neuron's weights. W[0,:]
            Returns:
                weights (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                    returned indices for passed uids.
        """
        if self.state.n == 0:
            return torch.Tensor([])
        else:
            w_0 = self.state.W[0,:]
            return w_0

    def uids_to_indices(self, uids: torch.Tensor) -> torch.LongTensor:
        r"""Return the indices of passed uids.

            Args:
                uids: (:obj:`torch.LongTensor` of shape :obj:`(-1)`):
                    UIDs for indices
            Returns:
                indices (:obj:`torch.LongTensor` of shape :obj:`(-1)`):
                    returned indices for passed uids.

        """
        indices = torch.nonzero(uids[..., None] == self.state.uids)[:,1]
        if torch.numel(uids) != torch.numel(indices):
            raise ValueError('Passed uids are not a subset of class.uids, with passed: {} and class.uids: {}'.format(uids, self.state.uids))
        return indices

    def uids_to_neurons(self, uids: torch.Tensor) -> List[bittensor.proto.Neuron]:
        r""" Returns a list with neurons for each uid.
            
            Args:
                uids (:obj:`torch.LongTensor`)
                    uids into neuron protos
            Returns:
                neurons (:obj:`List[bittensor.proto.Neuron]`): 
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
                neurons (:obj:`List[bittensor.proto.Neuron]`): 
                    neuron info ordered by passed uids.
            Returns:
                uids (:obj:`torch.LongTensor`)
                    uids associated with neurons.
        """
        uids = []
        for n in neurons:
            uids.append(self.state.uid_for_pubkey[n.public_key])
        return torch.tensor(uids)

    def uid_for_pubkey( self, public_key: str ) -> int:
        r""" Returns uids associated with the passed public key.
            Args:
                public_key (:obj:`str`): 
                    public key of neuron.
            Returns:
                uids (:obj:`int`)
                    uid associated with this public key, or -1 if non existent.
        """
        if public_key in self.state.uid_for_pubkey:
            return self.state.uid_for_pubkey[ public_key ]
        else:
            return -1

    def sync(self):
        r""" Synchronizes the local self.state with the chain state.
        """
        current_block = self.subtensor.get_current_block()
        if (self.last_sync - current_block) > 10:
            self.last_sync = current_block
            self._sync_cache()
            self.state = TorchChainState.from_cache(self.cache)
            self.state.block = current_block

    def _sync_cache(self):
        r""" Async: Makes calls to chain updating local chain cache with newest info.
        """
        # Make a full chain state grab.
        # last_emit: List[int]
        # neurons: List[Tuple[int, dict]]
        # stake: List[int]
        # weight_vals: List[Tuple[int, List[int]]]
        # weight_uids: List[Tuple[int, List[int]]]
        # active: List[Tuple[int, str]]
        last_emit, neurons, stake, weight_vals, weight_uids, active = self.subtensor.get_full_state()

        # Sinks the chain state into our cache for later conversion to the torch
        # chain state.
        for index, uid in enumerate(last_emit):
            self.cache.add_or_update(
                pubkey = pubkey, 
                ip = neuron[index][1]['ip'], 
                port = neuron[index][1]['port'], 
                uid = neuron[index][1]['uid'], 
                ip_type = neuron[index][1]['ip_type'], 
                modality = neuron[index][1]['modality'], 
                lastemit = last_emit[ index ], 
                stake = stake[ index ], 
                w_uids = weight_uids[ index ][1], 
                w_vals = weight_vals[ index ][1],
            )

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
        self_uid = self.uid_for_pubkey( self.wallet.hotkey )
        return '\nMetagraph:\nuid: {}, inflation_rate: {} block: {} n_neurons: {} \n'.format(self_uid, self.tau.item(), self.block, self.n) + df.to_string(na_rep = '', max_rows=5000, max_cols=25, min_rows=25, line_width=1000, float_format = lambda x: '%.3f' % x, col_space=1, justify='left')

    def __to_tensorboard__(self, tensorboard, global_step):
        tensorboard.add_scalar('Metagraph/neurons', self.n, global_step)
        tensorboard.add_scalar('Metagraph/inflation_rate', self.tau.item(), global_step)


