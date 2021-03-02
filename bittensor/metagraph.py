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

    def write_to_file(self, filepath: str ):
        json_data = {
            'block': self.block,
            'tau': self.tau.tolist(),
            'n': self.n,
            'uids': self.uids.tolist(),
            'indices': self.indices.tolist(),
            'stake': self.stake.tolist(),
            'lastemit': self.lastemit.tolist(),
            'W': self.W.tolist(),
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
            wallet: 'bittensor.wallet.Wallet' = None,
            subtensor: 'bittensor.subtensor.Subtensor' = None,
            **kwargs,
        ):
        r""" Initializes a new Metagraph chain interface.
            Args:
                config (:obj:`Munch`, `optional`): 
                    metagraph.Metagraph.config()
                wallet (:obj:`bittensor.wallet.Wallet`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
                subtensor (:obj:`bittensor.subtensor.Subtensor`, `optional`):
                    subtensor interface utility.
                stale_emit_filter', default=10000, type=int, 
                    The metagraph filters neurons with last emit beyond this many blocks.
                    Note, this is used to trim the graph size,but may change your incentive mechanism view.
        """
        if config == None:
            config = Metagraph.default_config()
        bittensor.config.Config.update_with_kwargs(config.metagraph, kwargs) 
        Metagraph.check_config(config)
        self.config = config

        if wallet == None:
            wallet = bittensor.wallet.Wallet( self.config )
        self.wallet = wallet

        if subtensor == None:
            subtensor = bittensor.subtensor.Subtensor( self.config, self.wallet )
        self.subtensor = subtensor

        # Chain state as cache and torch object.
        self.last_sync = 0
        self.uid = None
        self.metadata = None
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
        bittensor.wallet.Wallet.add_args( parser )
        bittensor.subtensor.Subtensor.add_args( parser )
        try:
            parser.add_argument('--metagraph.stale_emit_filter', default=-1, type=int, 
                                help='''Filter neurons who have not emitted in this number of blocks.
                                        -1 for no filter.''')
        except:
            pass
        
    @staticmethod   
    def check_config(config: Munch):
        pass

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
        self_col = self.col
        if len(self.col.tolist()) == 0:
            return torch.zeros(self.state.n)
        else:
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
        if self.uid == None:
            return torch.tensor([])
        try:
            self_idx = self.state.index_for_uid[ self.uid ] 
            return self.state.W[self_idx, :]
        except:
            logger.error('your uid is not in self.state with state.uids {} and uid {}'.format(self.state.uids, self.uid))
            return torch.tensor([])

    @property
    def col(self) -> torch.FloatTensor:
        r""" Returns this neuron's col weights, i.e. weights from other neurons to us.
            
             Returns:
                col (:obj:`torch.LongFloat` of shape :obj:`(metagraph.n)`):
                    `w_{*,i}`
        """
        if self.uid == None:
            return torch.tensor([])
        try:
            self_idx = self.state.index_for_uid[ self.uid ] 
            return self.state.W[:, self_idx]
        except:
            logger.error('your uid is not in self.state with state.uids {} and uid {}'.format( self.state.uids, self.uid ))
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
        r""" Returns the uid associated with the passed public key.
            Args:
                public_key (:obj:`str`): 
                    public key of neuron.
            Returns:
                uids (:obj:`int`)
                    uid associated with this public key, or None if non existent.
        """
        if public_key in self.state.uid_for_pubkey:
            return self.state.uid_for_pubkey[ public_key ]
        else:
            return None

    def neuron_for_uid( self, uid: int ) -> bittensor.proto.Neuron:
        r""" Returns the metadata associated with the passed uid, or None if the uid does not exist.
            Args:
                uid (:obj:`int`): 
                    uid to query for neuron metadata.
            Returns:
                neuron_metadata (:obj:`int`)
                    metadata of neuron associated with this uid, or None.
        """
        if uid in self.state.index_for_uid:
            return self.state.neurons[ self.state.index_for_uid[ uid ] ]
        else:
            return None

    def sync(self):
        r""" Synchronizes the local self.state with the chain state.
        """
        # TODO (const) this should probably be a background process
        # however, it makes it difficult for the user if the state changes in
        # the background.
        print(colored('\nSyncing metagraph:', 'white'))
        current_block = self.subtensor.get_current_block()
        # ---- Update cache ----
        self.last_sync = current_block
        self._sync_cache()

        # --- Update torch state
        self.state = TorchChainState.from_cache(self.cache)
        self.state.block = current_block

        hotkey = self.wallet.try_hotkey # Returns hotkey or None.
        if hotkey and hotkey.public_key in self.state.uid_for_pubkey:
            self.uid = self.uid_for_pubkey( hotkey.public_key )
            self.metadata = self.neuron_for_uid( self.uid )
        else:
            self.uid = None

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

        hotkey = self.wallet.try_hotkey # Returns hotkey or None.
        if hotkey != None:
            self_uid = await self.subtensor.async_get_uid_for_pubkey( hotkey.public_key )
            if self_uid != None:
                calls.append ( self._poll_uid (hotkey.public_key, self_uid ) )     

        for pubkey, uid in active.items():
            if uid in last_emit:
                emit_block = last_emit[ uid ]
                if (current_block - emit_block) < self.config.metagraph.stale_emit_filter or self.config.metagraph.stale_emit_filter < 0:
                        calls.append( self._poll_uid ( pubkey, uid ) )
        await asyncio.gather(*calls)
        print ('\n')

    async def _poll_uid(self, pubkey: str, uid:int):
        r""" Polls info info for a specfic public key.
        """
        try:
            stake = await self.subtensor.async_get_stake_for_uid( uid )
            lastemit = await self.subtensor.async_get_last_emit_data_for_uid( uid )
            w_uids = await self.subtensor.async_weight_uids_for_uid( uid )
            w_vals = await self.subtensor.async_weight_vals_for_uid( uid )
            neuron = await self.subtensor.async_get_neuron_for_uid ( uid )
            self.cache.add_or_update(pubkey = pubkey, ip = neuron['ip'], port = neuron['port'], uid = neuron['uid'], ip_type = neuron['ip_type'], modality = neuron['modality'], lastemit = lastemit, stake = stake.rao, w_uids = w_uids, w_vals = w_vals)
            print(colored('.', 'green'), end ="")

        except Exception as e:
            print(colored('x', 'red'), end ="")
            logger.trace('error while polling uid: {} with error: {}', uid, e )
            #traceback.print_exc()


    EmitSuccess = 1
    EmitValueError = 2
    EmitUnknownError = 3
    EmitTimeoutError = 4
    EmitTimeoutError = 5
    EmitResultUnknown = 6
    EmitNotInBlock = 7
    EmitNoOp = 8
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
        # --- Try emit, optionally wait ----
        code, message = self._try_emit(weights, wait_for_inclusion, timeout)
        if code == Metagraph.EmitSuccess:
            # ---- Emit was a success. ----
            logger.info("Emission was successful and entered the block.")

        elif code == Metagraph.EmitValueError:
            # ---- Passed weights were incorrect ----
            logger.info("Value error during emission: {}", message)

        elif code == Metagraph.EmitUnknownError:
            # ---- Unknown error ----
            logger.error("Unknown error during emission: {}", message)

        elif code == Metagraph.EmitTimeoutError:
            # ---- Timeout while waiting for inclusion ----
            logger.info("Emission timeout after {} seconds with error {}", timeout, message)

        elif code == Metagraph.EmitResultUnknown:
            # ---- Did not wait, result unknown ----
            logger.info("Emit results unknown.")

        elif code == Metagraph.EmitNotInBlock:
            # ---- Emit was success but did not enter the block ----
            logger.info('Emit did not enter block')

        elif code == Metagraph.EmitNoOp:
            # ---- Emit is a NoOp ----
            logger.info("When trying to set weights on chain. Weights are unchanged, nothing to emit.")

        return code, message

    def _try_emit(self, weights: torch.FloatTensor, wait_for_inclusion = False, timeout = 12) -> Tuple[int, str]:
        r""" Makes emit checks, emits to chain, and raises one of the following errors.
            Args:
                weights: (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                    Weights to set on chain.
                wait_for_inclusion: (:obj:`bool`):
                    If true, the call waits for block-inclusion before continuing or throws error after timeout.
                timeout: (:obj:`int`, default = 12 sec):
                    Time to wait for inclusion before raising a caught error.
            Returns:
                code (:obj:`ENUM`) {
                    EmitSuccess (:obj:`ENUM`):
                        Raised when try_async_emit emits weights successfully with known result.
                    EmitNoOp (:obj:`ENUM`):
                        Raised when calling emit does not change weights on chain.
                    EmitUnknownError (:obj:`ENUM`):
                        UnknownError during emit.
                    EmitValueError (:obj:`ENUM`):
                        Raised during emission when passed weights are not properly set.
                    EmitTimeoutError (:obj:`ENUM`):
                        Raised during emission during a timeout.
                    EmitResultUnknown (:obj:`ENUM`):
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
        if self._are_set_on_chain(weight_vals, weight_uids):
            message = "When trying to set weights on chain. Weights are unchanged, nothing to emit."
            return Metagraph.EmitNoOp, message

        # ---- Emit ----
        logger.info('Emitting weights -> {}', list(zip(weight_uids, weight_vals)))
        result = self.subtensor.set_weights(weight_uids, weight_vals, wait_for_inclusion=True, timeout = bittensor.__blocktime__ * 3)
        if result:
            message = "Successful emission"
            return Metagraph.EmitSuccess, message
        else:
            message = "Emission did not enter block."
            return Metagraph.EmitNotInBlock, message


    def _are_set_on_chain(self, weight_uids, weight_vals) -> bool:
        r""" Returns true if the passed key and vals are set on chain.
        """
        cmap = {}
        chain_uids = self.subtensor.weight_uids_for_uid(self.uid)
        chain_vals = self.subtensor.weight_vals_for_uid(self.uid)
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
                keys (:obj:`List[str]`):
                    List of pubkeys associated with each weight from vals.
                vals (:obj:`List[int]`):
                List of u32 integer representations of floating point weights.
        """
        remainder = MAX_INT_WEIGHT
        weight_vals = []
        weight_uids = []
        pos_self_uid = -1
        length = 0
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
                length += 1

            if uid_i == self.uid:
                pos_self_uid = (length - 1)

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


