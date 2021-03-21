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

from munch import Munch
from termcolor import colored
from loguru import logger
from typing import List, Tuple, List

import bittensor
import bittensor.config as config_utils
import multiprocessing

class _Metagraph():
    def __init__(   
        self,
        n: int,
        tau: int,
        block: int,
        uids: List[int],
        stake: List[int],
        lastemit: List[int],
        weight_vals: List[List[int]],
        weight_uids: List[List[int]],
        neurons: List[bittensor.proto.Neuron]
    ):
        self._manager = multiprocessing.Manager()
        self.n = self._manager.Value( 'i', n )
        self.tau = self._manager.Value( 'd', tau )
        self.block = self._manager.Value( 'i', block )
        self.uids = self._manager.list( uids )
        self.stake = self._manager.list( stake )
        self.lastemit = self._manager.list( lastemit )
        self.weight_vals = self._manager.list( [self._manager.list( vals ) for vals in weight_vals] )
        self.weight_uids = self._manager.list( [self._manager.list( uids ) for uids in weight_uids] )
        self.neurons = self._manager.list( neurons )
        uid_for_pubkey = {}
        for idx, (uid, neuron) in enumerate(list(zip(uids, neurons))):
            uid_for_pubkey[neuron.public_key] = uid
        self.uid_for_pubkey = self._manager.dict( uid_for_pubkey )

    def update(
        self,
        n: int,
        tau: int,
        block: int,
        uids: List[int],
        stake: List[int],
        lastemit: List[int],
        weight_vals: List[List[int]],
        weight_uids: List[List[int]],
        neurons: List[bittensor.proto.Neuron]
    ):
        del self.n
        del self.tau
        del self.block
        del self.uids
        del self.stake
        del self.lastemit
        del self.weight_vals
        del self.weight_uids
        del self.neurons
        del self.uid_for_pubkey
        self.n = self._manager.Value( 'i', n )
        self.tau = self._manager.Value( 'd', tau )
        self.block = self._manager.Value( 'i', block )
        self.uids = self._manager.list( uids )
        self.stake = self._manager.list( stake )
        self.lastemit = self._manager.list( lastemit )
        self.weight_vals = self._manager.list( [self._manager.list( vals ) for vals in weight_vals] )
        self.weight_uids = self._manager.list( [self._manager.list( uids ) for uids in weight_uids] )
        self.neurons = self._manager.list( neurons )
        uid_for_pubkey = {}
        for idx, (uid, neuron) in enumerate(list(zip(uids, neurons))):
            uid_for_pubkey[neuron.public_key] = uid
        self.uid_for_pubkey = self._manager.dict( uid_for_pubkey )



class Metagraph():
    """ Maintains the chain state as a process safe object. 

        Args:
            tau (:obj:`int`): 
                current, per block, token inflation rate.

            block (:obj:`int`):
                state block number.

            uids (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n())`):
                UIDs for each neuron ordered by uid.
            
            stake (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n())`):
                Stake balance for each neuron ordered by uid.
                
            lastemit (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n())`):
                Last emission call for each neuron ordered by uid.

            W (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n(), metagraph.n())`):
                Full weight matrix on chain.

            neurons (List[bittensor.proto.Neuron]) 
                List of endpoints on the network.

    """
    def __init__( self, subtensor: 'bittensor.Subtensor' ):
        r""" Initializes a new process-safe Metagraph
        """
        self.subtensor = subtensor
        self._lock = multiprocessing.RLock()
        self._metagraph = _Metagraph(
            n = 0,
            tau = 0.5,
            block = 0,
            uids = [],
            stake = [],
            lastemit = [],
            weight_vals = [[]],
            weight_uids = [[]],
            neurons = []
        )
        
    def n(self) -> int:
        r""" Return the number of known neurons on chain.
            
            Returns:
                n (int):
                    number of known neurons.

        """
        try:
            self._lock.acquire()
            return_value = copy.deepcopy(self._metagraph.n.value)
        finally:
            self._lock.release()
        return return_value

    def block(self) -> int:
        r""" Return the block number when the chain state was updated.

             Returns:
                block (:obj:`int`):
                    local chain state block number.
        """
        try:
            self._lock.acquire()
            return_value = copy.deepcopy(self._metagraph.block.value)
        finally:
            self._lock.release()
        return return_value

    def lastemit(self) -> torch.LongTensor:
        r""" Returns the last emit time for each known neuron.
            
            Returns:
                lastemit (:obj:`int`):
                    last emit time.
        """
        try:
            self._lock.acquire()
            return_value = torch.tensor(copy.deepcopy(self._metagraph.lastemit), dtype=torch.int64)
        finally:
            self._lock.release()
        return return_value

    def uids(self) -> torch.LongTensor:
        r""" Returns unique ids for each neuron in the chain state.
            Returns:
                uids (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n())`):
                    unique id for each neuron.
        """
        try:
            self._lock.acquire()
            return_value = torch.tensor(copy.deepcopy(self._metagraph.uids), dtype=torch.int64)
        finally:
            self._lock.release()
        return return_value

    def stake(self) -> torch.FloatTensor:
        r""" Returns the stake held by each known neuron.
            
            Returns:
                stake (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n())`):
                    stake of each known neuron.

        """
        try:
            self._lock.acquire()
            return_value = torch.tensor(copy.deepcopy(self._metagraph.stake), dtype=torch.float32)
        finally:
            self._lock.release()
        return return_value

    def S(self) -> torch.FloatTensor:
        r""" Returns the stake held by each known neuron.
             
             Returns:
                S (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n())`):
                    stake of each known neuron.
        """
        try:
            self._lock.acquire()
            return_value = torch.tensor(copy.deepcopy(self._metagraph.stake), dtype=torch.float32)
        finally:
            self._lock.release()
        return return_value

    def tau(self) -> torch.FloatTensor:
        r""" tau: the chain per block inflation rate. i.e. 50
            
            Returns:
                tau (:obj:`torchFloatTensor` of shape :obj:`(1)`):
                    current chain inflation rate.
        """
        try:
            self._lock.acquire()
            return_value = self._metagraph.tau.value
        finally:
            self._lock.release()
        return return_value

    def ranks(self) -> torch.FloatTensor:
        r""" Returns the ranks W^t * S
           
            Returns:
                ranks (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n())`):
                    rank of each known neuron.

        """
        try:
            self._lock.acquire()
            R = self._compute_R()
        finally:
            self._lock.release()
        return R

    def R(self) -> torch.FloatTensor:
        r""" Returns ranks for each known neuron in the graph.
             
             Returns:
                rank (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n())`):
                    rank of each known neuron.
        """
        try:
            self._lock.acquire()
            R = self._compute_R()
        finally:
            self._lock.release()
        return R

    def I(self) -> torch.FloatTensor:
        r""" Returns the inflation incentive for each peer per block.
        
            Returns:
                I (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n())`):
                    stake of each known neuron.
        """
        try:
            self._lock.acquire()
            I = self._compute_I()
        finally:
            self._lock.release()
        return I

    def W(self) -> torch.FloatTensor:
        try:
            self._lock.acquire()
            W = self._compute_W()
        finally:
            self._lock.release()
        return W

    def _compute_I(self) -> torch.FloatTensor:
        tau = copy.deepcopy(self._metagraph.tau.value)
        R = self._compute_R()
        I =  (tau * R) / torch.sum( R )
        I = torch.where(torch.isnan(I), torch.zeros_like(I), I)
        return I

    def _compute_R(self) -> torch.FloatTensor:
        W = self._compute_W()
        if W.shape[0] == 0:
            return torch.tensor([])
        else:
            S = torch.tensor(copy.deepcopy(self._metagraph.stake), dtype=torch.float32)
            W = torch.transpose( W.view(self._metagraph.n.value, self._metagraph.n.value), 0, 1 )
            R = torch.matmul(W, S).view(self._metagraph.n.value)
        return R

    def _compute_W(self) -> torch.FloatTensor:
        r""" Non locking weight calculation.
             
             Returns:
                W (:obj:`torch.LongFloat` of shape :obj:`(metagraph.n(), metagraph.n())`):
                    w_ij of each neuron.
        """
        n = self._metagraph.n.value
        weights_numpy = numpy.zeros( (n, n) )
        for uid_i in range( n ):
            dests = copy.deepcopy(self._metagraph.weight_uids[ uid_i ])
            weights = copy.deepcopy(self._metagraph.weight_vals[ uid_i ])
            weights_sum = sum( weights )
            for uid_j, weight_value in list(zip(dests, weights)):
                if weights_sum != 0:
                    weights_numpy[uid_i, uid_j] = float(weight_value) / float(weights_sum)
                else:
                    weights_numpy[uid_i, uid_j] = 0
        return torch.tensor(weights_numpy, dtype=torch.float32)

    def neurons(self) -> List[bittensor.proto.Neuron]:
        r""" Return neuron endpoint information for each neuron.
            
            Returns:
                neurons (:obj:`List[bittensor.proto.Neuron]` of shape :obj:`(metagraph.n(), metagraph.n()`):
                    endpoint information for each neuron.

        """
        return [copy.deepcopy(n) for n in self._metagraph.neurons]

    def public_keys(self) -> List[str]:
        r""" Return the ordered public keys for state neurons.
        
            Returns:
                public_keys (:obj:`List[str]` of shape :obj:`(metagraph.n)`):
                    public keys of all graph neurons.

        """
        try:
            self._lock.acquire()
            pubkeys = [ n.public_key for n in self._metagraph.neurons ]
        finally:
            self._lock.release()
        return pubkeys


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
        try:
            self._lock.acquire()
            for uid in uids.tolist():
                response.append( copy.deepcopy(self._metagraph.neurons[uid]) )
        finally:
            self._lock.release()
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
        try:
            self._lock.acquire()
            for n in neurons:
                uids.append( copy.deepcopy(self._metagraph.uid_for_pubkey[n.public_key]) )
        finally:
            self._lock.release()
        return torch.tensor(uids, dtype=torch.int64) 

    def uid_for_pubkey( self, public_key: str ) -> int:
        r""" Returns the uid associated with the passed public key.
            Args:
                public_key (:obj:`str`): 
                    public key of neuron.
            Returns:
                uids (:obj:`int`)
                    uid associated with this public key, or None if non existent.
        """
        response = None
        try:
            self._lock.acquire()
            if public_key in self._metagraph.uid_for_pubkey:
                response = copy.deepcopy(self._metagraph.uid_for_pubkey[ public_key ])
        finally:
            self._lock.release()
        return response


    def neuron_for_uid( self, uid: int ) -> bittensor.proto.Neuron:
        r""" Returns the metadata associated with the passed uid, or None if the uid does not exist.
            Args:
                uid (:obj:`int`): 
                    uid to query for neuron metadata.
            Returns:
                neuron_metadata (:obj:`int`)
                    metadata of neuron associated with this uid, or None.
        """
        response = None
        try:
            self._lock.acquire()
            if uid in self._metagraph.uids:
                response = copy.deepcopy( self._metagraph.neurons[ uid ] )
        finally:
            self._lock.release()
        return response

    def sync( self ):
        r""" Syncs the _metagraph with updated chain state.
        """
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        return loop.run_until_complete(self._async_sync_metagraph())

    async def _async_sync_metagraph( self ):
        r""" Asyncronously: Makes calls to the chain endpoint to update the metagraph object.
        """
        # ---- Pull previous state ---
        try:
            self._lock.acquire()
            prev_n = copy.deepcopy( self._metagraph.n.value )
            prev_tau = copy.deepcopy( self._metagraph.tau.value )
            prev_block = copy.deepcopy( self._metagraph.block.value )
            prev_uids = copy.deepcopy( self._metagraph.uids )
            prev_stake = copy.deepcopy( self._metagraph.stake )
            prev_lastemit = copy.deepcopy( self._metagraph.lastemit ) 
            prev_weight_vals = copy.deepcopy( self._metagraph.weight_vals )
            prev_weight_uids = copy.deepcopy( self._metagraph.weight_uids)
            prev_neurons = copy.deepcopy( self._metagraph.neurons )
        finally:
            self._lock.release()

        # ---- Get updated state ----
        next_n, next_tau, next_block, \
        next_uids, next_lastemit, next_stake, \
        next_weight_vals, next_weight_uids, next_neurons = await self._get_updated_state (
            prev_n = prev_n,
            prev_tau = prev_tau,
            prev_block = prev_block,
            prev_uids = prev_uids,
            prev_stake = prev_stake,
            prev_lastemit = prev_lastemit,
            prev_weight_vals = prev_weight_vals,
            prev_weight_uids = prev_weight_uids,
            prev_neurons = prev_neurons
        )

        # ---- Create new _metagraph object ----
        try:
            self._lock.acquire()
            self._metagraph.update(
                n = next_n,
                tau = next_tau,
                block = next_block,
                uids = next_uids,
                stake = next_stake,
                lastemit = next_lastemit,
                weight_vals = next_weight_vals,
                weight_uids = next_weight_uids,
                neurons = next_neurons
            )
        finally:
            self._lock.release()

    async def _get_updated_state( 
            self,
            prev_n: int,
            prev_tau: int,
            prev_block: int,
            prev_uids: List[int],
            prev_stake: List[int],
            prev_lastemit: List[int],
            prev_weight_vals: List[List[int]],
            prev_weight_uids: List[List[int]],
            prev_neurons: List[bittensor.proto.Neuron]
    ) -> Tuple[ int, int, int, List[int], List[int], List[int], List[List[int]], List[List[int]], List[bittensor.proto.Neuron] ]:
    
        # Query chain for last emit values.
        chain_lastemit = dict( await self.subtensor.async_get_last_emit() ) #  List[Tuple[uid, lastemit]]
        chain_stake = dict( await self.subtensor.async_get_stake() ) #  List[Tuple[uid, stake]]
        chain_block = int( await self.subtensor.async_get_current_block())

        # Create next state buffers.
        next_n = len( chain_lastemit )
        next_tau = prev_tau
        next_block = chain_block
        next_uids = range( next_n )
        next_lastemit = [ chain_lastemit[i] for i in range(next_n) ] 
        next_stake = [ chain_stake[i] for i in range(next_n) ] 
        next_weight_vals = []
        next_weight_uids = []
        next_neurons = []
        for uid in range(next_n):
            if uid < prev_n:
                next_weight_vals.append( prev_weight_vals[uid] )
                next_weight_uids.append( prev_weight_uids[uid] )
                next_neurons.append( prev_neurons[uid] )
            else:
                next_weight_vals.append( [] )
                next_weight_uids.append( [] )
                next_neurons.append( None )

        # Make calls for additional info.
        pending_queries = []
        for uid, lastemit in list(zip( next_uids, next_lastemit )):
            if lastemit > prev_block:
                pending_queries.append( 
                    self.fill_uid( 
                        uid = uid,
                        weight_vals_to_fill = next_weight_vals,
                        weight_uids_to_fill = next_weight_uids,
                        neurons_to_fill = next_neurons,
                    ) 
                )
        await asyncio.gather(*pending_queries)
        return next_n, next_tau, next_block, next_uids, next_lastemit, next_stake, next_weight_vals, next_weight_uids, next_neurons

    # Function which fills weights and neuron info for a uid.
    async def fill_uid ( 
        self,
        uid: int,
        weight_vals_to_fill: List[List[int]],
        weight_uids_to_fill: List[List[int]],
        neurons_to_fill: List[bittensor.proto.Neuron]
    ) -> bool:
        #try:
        weight_uids = await self.subtensor.async_weight_uids_for_uid( uid ) 
        weight_vals = await self.subtensor.async_weight_vals_for_uid( uid ) 
        neuron = await self.subtensor.async_get_neuron_for_uid( uid )
        neuron_proto = bittensor.proto.Neuron(
                version = bittensor.__version__,
                public_key = neuron['hotkey'],
                address = str(neuron['ip']),
                port = neuron['port'],
                uid = neuron['uid'], 
                modality = neuron['modality'],
                ip_type = neuron['ip_type']          
        )
        weight_vals_to_fill[uid] = weight_vals
        weight_uids_to_fill[uid] = weight_uids
        neurons_to_fill[uid] = neuron_proto
        print(colored('.', 'green'), end ="")
        return True
        # except Exception as e:
        #     print ()
        #     print(colored('x', 'red'), end ="")
        #     return False


    def __str__(self):
        uids = self.uids().tolist()
        rows = [self.S().tolist(), self.R().tolist(), self.I().tolist()]
        for i in range(self.n()):
            rows.append(self.W()[i, :].tolist())
        df = pd.DataFrame(rows, columns=uids)
        df = df.rename(index={df.index[0]: 'S'})
        df = df.rename(index={df.index[1]: 'R'})
        df = df.rename(index={df.index[2]: 'I'})
        for i in range(self.n()):
            df = df.rename(index={df.index[i + 6]: uids[i]})
        df.rename_axis(colored('[uid]', 'red'), axis=1)
        return '\nMetagraph: inflation_rate: {} block: {} n_neurons: {} \n'.format(self.tau(), self.block(), self.n()) + df.to_string(na_rep = '', max_rows=5000, max_cols=25, min_rows=25, line_width=1000, float_format = lambda x: '%.3f' % x, col_space=1, justify='left')

    def __to_tensorboard__(self, tensorboard, global_step):
        tensorboard.add_scalar('Metagraph/neurons', self.n(), global_step)
        tensorboard.add_scalar('Metagraph/inflation_rate', self.tau().item(), global_step)




