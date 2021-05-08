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
import os
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
import bittensor.utils.weight_utils as weight_utils
from bittensor.subtensor import Subtensor
from bittensor.crypto.keyfiles import KeyFileError

class Metagraph( torch.nn.Module ):
    r""" Maintains chain state as a torch object.

        Interface:
            tau (:obj:`torch.FloatTensor` of shape :obj:`(1)`): 
                Current, per block, token inflation rate.

            block (:obj:`torch.LongTensor` of shape :obj:`(1)`):
                State block number.

            uids (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n)`):
                UIDs for each neuron.
            
            stake (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n)`):
                Stake balance for each neuron ordered by uid.
                
            lastemit (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n)`):
                Last emission call for each neuron ordered by uid.

            weights (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n, metagraph.n)`):
                Full weight matrix on chain ordered by uid.

            neurons (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n, -1)`) 
                Tokenized endpoints information.

    """
    def __init__( self ):
        r""" Initializes a new Metagraph torch chain interface object.
        """
        super(Metagraph, self).__init__()
        # State.
        self.n = torch.nn.Parameter( torch.tensor( [0], dtype = torch.int64), requires_grad=False )
        self.tau = torch.nn.Parameter( torch.tensor( [0.5], dtype = torch.float32), requires_grad=False )
        self.block = torch.nn.Parameter( torch.tensor( [0], dtype = torch.int64), requires_grad=False )
        self.uids = torch.nn.Parameter( torch.tensor( [], dtype = torch.int64), requires_grad=False )
        self.stake = torch.nn.Parameter( torch.tensor( [], dtype = torch.float32), requires_grad=False )
        self.lastemit = torch.nn.Parameter( torch.tensor( [], dtype = torch.int64), requires_grad=False )
        self.weights = torch.nn.ParameterList()
        self.neurons = torch.nn.ParameterList()

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
    def S(self) -> torch.FloatTensor:
        r""" Returns neurons stake values.
             
             Returns:
                S (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                    Stake of each known neuron.
        """
        return self.stake

    @property
    def I(self) -> torch.FloatTensor:
        r""" Returns neuron incentives: tau * R / sum(R)
        
            Returns:
                I (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                    Block incentive for each neuron. 
        """
        I =  (self.tau * self.ranks) / torch.sum(self.ranks)
        I = torch.where(torch.isnan(I), torch.zeros_like(I), I)
        return I.view(self.n)

    @property
    def ranks(self) -> torch.FloatTensor:
        r""" Returns neuron ranks: W^t * S
           
            Returns:
                ranks (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                    Rank of each neuron.

        """
        if self.n.item() == 0:
            return torch.tensor([])
        else:
            S = self.S.view(self.n, 1)
            Wt = torch.transpose(self.W, 0, 1)
            R = torch.matmul(Wt, S).view(self.n)
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
    def W(self) -> torch.FloatTensor:
        r""" Return full weight matrix from chain.
             Returns:
                W (:obj:`torch.LongFloat` of shape :obj:`(metagraph.n, metagraph.n)`):
                    Weight matrix.
        """
        return torch.stack( [row for row in self.weights], axis = 0 )

    @property
    def hotkeys( self ) -> List[str]:
        r""" Returns hotkeys for each neuron.
            Returns:
                hotkeys (:obj:`List[str] of shape :obj:`(metagraph.n)`):
                    Neuron hotkeys.
        """
        return [ neuron.hotkey for neuron in self.neuron_endpoints ]

    @property
    def coldkeys( self ) -> List[str]:
        r""" Returns coldkeys for each neuron.
            Returns:
                coldkeys (:obj:`List[str] of shape :obj:`(metagraph.n)`):
                    Neuron coldkeys.
        """
        return [ neuron.coldkey for neuron in self.neuron_endpoints ]

    @property
    def modalities( self ) -> List[str]:
        r""" Returns the modality for each neuron.
            Returns:
                coldkeys (:obj:`List[str] of shape :obj:`(metagraph.n)`):
                    Neuron coldkeys.
        """
        return [ neuron.modality for neuron in self.neuron_endpoints ]

    @property
    def addresses( self ) -> List[str]:
        r""" Returns ip addresses for each neuron.
            Returns:
                coldkeys (:obj:`List[str] of shape :obj:`(metagraph.n)`):
                    Neuron address.
        """
        return [ net.ip__str__( neuron.ip_type, neuron.ip, neuron.port ) for neuron in self.neuron_endpoints ]

    @property
    def neuron_endpoints(self) -> List[ bittensor.utils.neurons.NeuronEndpoint ]:
        r""" Return neuron endpoint information for each neuron.
            
            Returns:
                neurons (:obj:`List[ bittensor.utils.neurons.NeuronEndpoint ]` of shape :obj:`(metagraph.n)`):
                    Endpoint information for each neuron.

        """
        return [ bittensor.utils.neurons.NeuronEndpoint.from_tensor( neuron_tensor ) for neuron_tensor in self.neurons ]

    def load( self, network:str = 'kusanagi' ):
        self.load_from_path( path = '~/.bittensor/' + str(network) + '.pt')

    def save( self, network:str = 'kusanagi' ):
        self.save_to_path( path = '~/.bittensor/' + str(network) + '.pt')

    def load_from_path(self, path:str ):
        full_path = os.path.expanduser(path)
        metastate = torch.load( full_path )
        self.load_from_state_dict( metastate )

    def save_to_path(self, path:str ):
        full_path = os.path.expanduser(path)
        metastate = self.state_dict()
        torch.save(metastate, full_path)

    def load_from_state_dict(self, state_dict:dict ):
        self.n = torch.nn.Parameter( state_dict['n'], requires_grad=False )
        self.tau = torch.nn.Parameter( state_dict['tau'], requires_grad=False )
        self.block = torch.nn.Parameter( state_dict['block'], requires_grad=False )
        self.uids = torch.nn.Parameter( state_dict['uids'], requires_grad=False )
        self.stake = torch.nn.Parameter( state_dict['stake'], requires_grad=False )
        self.lastemit = torch.nn.Parameter( state_dict['lastemit'], requires_grad=False )
        self.weights = torch.nn.ParameterList([torch.nn.Parameter( state_dict['weights.' + str(i)], requires_grad=False )  for i in range(self.n.item()) ])
        self.neurons = torch.nn.ParameterList([torch.nn.Parameter( state_dict['neurons.' + str(i)], requires_grad=False )  for i in range(self.n.item()) ])

    def sync(self, subtensor: 'bittensor.subtensor.Subtensor' = None):
        r""" Synchronizes this metagraph with the chain state.
        """
        # Defaults to base subtensor connection.
        if subtensor == None:
            subtensor = bittensor.subtensor.Subtensor()
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        loop.run_until_complete(self._async_sync(subtensor))

    async def _async_sync( self, subtensor: 'bittensor.subtensor.Subtensor'):

        # Query chain info.
        chain_lastemit = dict( await subtensor.async_get_last_emit() ) #  Optional[ List[Tuple[uid, lastemit]] ]
        chain_stake = dict( await subtensor.async_get_stake() ) #  Optional[ List[Tuple[uid, stake]] ]
        chain_block = int( await subtensor.async_get_current_block()) #  Optional[ int ]

        # Build new state.
        new_size = len(chain_stake)
        old_size = self.n.item() 
        old_block = self.block.item()
        new_n = torch.tensor([new_size], dtype = torch.int64)
        new_block = torch.tensor([chain_block], dtype = torch.int64)
        new_uids = torch.tensor( range(new_size) ,  dtype = torch.int64)
        new_stake = torch.tensor([ (float(chain_stake[uid])/1000000000) for uid in range(new_size)],  dtype = torch.float32)
        new_lastemit = torch.tensor([ chain_lastemit[uid] for uid in range(new_size)], dtype = torch.int64)

        # Set params.3
        self.n = torch.nn.Parameter( new_n, requires_grad=False )
        self.block = torch.nn.Parameter( new_block, requires_grad=False )
        self.uids = torch.nn.Parameter( new_uids, requires_grad=False )
        self.stake = torch.nn.Parameter( new_stake, requires_grad=False )
        self.lastemit = torch.nn.Parameter( new_lastemit, requires_grad=False )

        # Extend weights matrix.
        for idx in range( old_size ):
            self.weights[idx] =  torch.nn.Parameter( torch.cat( [self.weights[idx], torch.zeros([new_size - len(self.weights[idx])], dtype = torch.float32)]))

        # Create buffers
        for _ in range( new_size - old_size ):
            self.weights.append( torch.nn.Parameter( torch.tensor([], dtype = torch.float32), requires_grad=False ) )
            self.neurons.append( torch.nn.Parameter( torch.tensor([], dtype = torch.int64), requires_grad=False ) )

        # Fill buffers.
        pending_queries = []
        for uid, lastemit in chain_lastemit.items():
            if lastemit > old_block:
                pending_queries.append( self.fill_uid( subtensor = subtensor, uid = uid ) )
        for query in tqdm.asyncio.tqdm.as_completed( pending_queries ):
            await query

    # Function which fills weights and neuron info for a uid.
    async def fill_uid ( self, subtensor: 'bittensor.subtensor.Subtensor', uid: int ) -> bool:
        # TODO(const): try catch block with retry.

        # Fill row from weights.
        weight_uids = await subtensor.async_weight_uids_for_uid( uid ) 
        weight_vals = await subtensor.async_weight_vals_for_uid( uid ) 
        row_weights = weight_utils.convert_weight_uids_and_vals_to_tensor( self.n.item(), weight_uids, weight_vals )
        self.weights[ uid ] = torch.nn.Parameter( row_weights, requires_grad=False )
        
        # Fill Neuron info.
        neuron = await subtensor.async_get_neuron_for_uid( uid )
        neuron_obj = bittensor.utils.neurons.NeuronEndpoint.from_dict(neuron)
        neuron_tensor = neuron_obj.to_tensor()
        self.neurons[ uid ] = torch.nn.Parameter( neuron_tensor, requires_grad=False )

        # Return.
        return True

    def __str__(self):
        if self.n != 0:
            peers_online = torch.numel(torch.where( self.block - self.lastemit < 1000 )[0])
        else:
            peers_online = 0
        peers_online = torch.numel(torch.where( self.block - self.lastemit < 1000 )[0])
        return '<green>Metagraph:</green> block:<cyan>{}</cyan>, inflation_rate:<cyan>{}</cyan>, staked:<green>\u03C4{}</green>/<cyan>\u03C4{}</cyan>, active:<green>{}</green>/<cyan>{}</cyan>'.format(self.block.item(), self.tau.item(), torch.sum(self.S), self.block.item()/2, peers_online, self.n.item())

    def __to_tensorboard__(self, tensorboard, global_step):
        tensorboard.add_scalar('Metagraph/neurons', self.n.item(), global_step)
        tensorboard.add_scalar('Metagraph/inflation_rate', self.tau.item(), global_step)


