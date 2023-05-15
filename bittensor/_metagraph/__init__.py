""" Maintains chain state as a torch.nn.Module.
"""
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

import os
import json
import torch
import bittensor

import torch.nn.functional as f
import bittensor.utils.networking as net

from os import listdir
from os.path import isfile, join
from bittensor import Balance
from typing import List, Optional, Dict, Union


# Return directory path from network and netuid
def get_save_dir(  network: str, netuid: int ) -> str: 
    return os.path.expanduser(f"~/.bittensor/metagraphs/network-{str(network)}/netuid-{str(netuid)}/")

def latest_block_path( dir_path: str ) -> int:
        latest_block = -1
        latest_file_full_path = None
        for filename in listdir(dir_path):
            full_path_filename = os.path.expanduser(join(dir_path, filename))
            try:
                block_number = int(filename.split('-')[1].split('.')[0])
                if block_number > latest_block:
                    latest_block = block_number
                    latest_file_full_path = full_path_filename
            except Exception as e:
                pass
        if not latest_file_full_path: 
            raise ValueError( f"Metagraph not found at: {dir_path}" )
        else:
            return latest_file_full_path
        
class metagraph( torch.nn.Module ):

    @property
    def S(self) -> torch.FloatTensor: return self.total_stake
    @property
    def R(self) -> torch.FloatTensor: return self.ranks
    @property
    def I(self) -> torch.FloatTensor: return self.incentive
    @property
    def E(self) -> torch.FloatTensor: return self.emission
    @property
    def C(self) -> torch.FloatTensor: return self.consensus
    @property
    def T(self) -> torch.FloatTensor: return self.trust
    @property
    def Tv(self) -> torch.FloatTensor: return self.validator_trust
    @property
    def D(self) -> torch.FloatTensor: return self.dividends
    @property
    def B(self) -> torch.FloatTensor: return self.bonds
    @property
    def W(self) -> torch.FloatTensor: return self.weights
    @property
    def hotkeys( self ) -> List[str]: return [ axon.hotkey for axon in self.axons ]
    @property
    def coldkeys( self ) -> List[str]: return [ axon.coldkey for axon in self.axons ]
    @property
    def addresses( self ) -> List[str]: return [ axon.ip_str() for axon in self.axons ]

    def __str__(self): return "Metagraph(netuid:{}, n:{}, block:{}, network:{})".format(self.netuid, self.n.item(), self.block.item(), self.network)
        
    def __repr__(self): return self.__str__()

    def metadata(self) -> dict: return {"netuid": self.netuid, "n": self.n.item(), "block": self.block.item(), "network": self.network, "version": bittensor.__version__ }

    def __init__(self, netuid: int, network: str = 'finney', lite:bool = True, sync: bool = True ) -> 'metagraph':    
        super(metagraph, self).__init__()
        self.netuid = netuid
        self.network = network
        self.version = torch.nn.Parameter( torch.tensor( [ bittensor.__version_as_int__ ], dtype=torch.int64), requires_grad=False )
        self.n = torch.nn.Parameter( torch.tensor( [0], dtype=torch.int64), requires_grad = False )
        self.block = torch.nn.Parameter( torch.tensor( [0], dtype=torch.int64), requires_grad = False )
        self.stake = torch.nn.Parameter( torch.tensor( [], dtype=torch.float32 ), requires_grad=False )
        self.total_stake = torch.nn.Parameter(  torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self.ranks = torch.nn.Parameter(  torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self.trust = torch.nn.Parameter(  torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self.consensus = torch.nn.Parameter(  torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self.validator_trust = torch.nn.Parameter(  torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self.incentive = torch.nn.Parameter(  torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self.emission = torch.nn.Parameter(  torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self.dividends = torch.nn.Parameter(  torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self.active = torch.nn.Parameter(  torch.tensor( [], dtype=torch.int64), requires_grad=False )
        self.last_update = torch.nn.Parameter(  torch.tensor( [], dtype=torch.int64), requires_grad=False )
        self.validator_permit = torch.nn.Parameter(  torch.tensor( [], dtype=torch.bool), requires_grad=False )
        self.weights = torch.nn.Parameter(  torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self.bonds = torch.nn.Parameter(  torch.tensor( [], dtype=torch.int64), requires_grad=False )
        self.uids = torch.nn.Parameter( torch.tensor([], dtype = torch.int64),requires_grad=False )
        self.axons = []
        if sync:
            self.sync( block = None, lite = lite )

    def sync ( self, block: Optional[int] = None, lite: bool = True ) -> 'metagraph':
        subtensor = bittensor.subtensor( network = self.network )
        if lite:
            self.neurons = subtensor.neurons_lite( block = block, netuid = self.netuid )
        else:
            self.neurons = subtensor.neurons(block = block, netuid = self.netuid )
        self.lite = lite
        self.n = torch.nn.Parameter( torch.tensor( len(self.neurons), dtype=torch.int64 ), requires_grad=False )
        self.version = torch.nn.Parameter( torch.tensor( [bittensor.__version_as_int__], dtype=torch.int64 ), requires_grad=False )
        self.block = torch.nn.Parameter( torch.tensor( subtensor.block, dtype=torch.int64 ), requires_grad=False )
        self.uids = torch.nn.Parameter( torch.tensor( [ neuron.uid for neuron in self.neurons ], dtype=torch.int64 ), requires_grad=False )
        self.trust = torch.nn.Parameter( torch.tensor( [ neuron.trust for neuron in self.neurons ], dtype=torch.float32 ), requires_grad=False )
        self.consensus = torch.nn.Parameter( torch.tensor( [ neuron.consensus for neuron in self.neurons ], dtype=torch.float32 ), requires_grad=False )
        self.incentive = torch.nn.Parameter( torch.tensor( [ neuron.incentive for neuron in self.neurons ], dtype=torch.float32 ), requires_grad=False )
        self.dividends = torch.nn.Parameter( torch.tensor( [ neuron.dividends for neuron in self.neurons ], dtype=torch.float32 ), requires_grad=False )
        self.ranks = torch.nn.Parameter( torch.tensor( [ neuron.rank for neuron in self.neurons ], dtype=torch.float32 ), requires_grad=False )
        self.emission = torch.nn.Parameter( torch.tensor( [ neuron.emission for neuron in self.neurons ], dtype=torch.float32 ), requires_grad=False )
        self.active = torch.nn.Parameter( torch.tensor( [ neuron.active for neuron in self.neurons ], dtype=torch.int64 ), requires_grad=False )
        self.last_update = torch.nn.Parameter( torch.tensor( [ neuron.last_update for neuron in self.neurons ], dtype=torch.int64 ), requires_grad=False )
        self.validator_permit = torch.nn.Parameter( torch.tensor( [ neuron.validator_permit for neuron in self.neurons ], dtype=torch.bool ), requires_grad=False )
        self.validator_trust = torch.nn.Parameter( torch.tensor( [ neuron.validator_trust for neuron in self.neurons ], dtype=torch.float32 ), requires_grad=False )
        self.total_stake = torch.nn.Parameter( torch.tensor( [ neuron.total_stake.tao for neuron in self.neurons ], dtype=torch.float32 ), requires_grad=False )
        self.stake = torch.nn.Parameter( torch.tensor( [ neuron.stake for neuron in self.neurons ], dtype=torch.float32 ), requires_grad=False )
        self.axons = [ n.axon_info for n in self.neurons ]
        if not lite:
            weights_array = []
            for n in self.neurons:
                w_uids, w_weights = zip(*n.weights)
                weights_array.append( bittensor.utils.weight_utils.convert_weight_uids_and_vals_to_tensor( len(self.neurons), w_uids, w_weights ).tolist() )
            self.weights = torch.nn.Parameter( torch.stack( weights_array ), requires_grad=False ) if len( weights_array ) else torch.Tensor()
            if len(weights_array) == 0:
                bittensor.logging.warning("Empty weights_array on metagraph.sync(). The 'weights' tensor is empty.")      
        if not lite:
            bonds_array = []
            for n in self.neurons:
                b_uids, b_bonds = zip(*n.bonds)
                bonds_array.append( bittensor.utils.weight_utils.convert_bond_uids_and_vals_to_tensor( len(self.neurons), b_uids, b_bonds ).tolist() )
            self.bonds = torch.nn.Parameter( torch.stack( bonds_array ), requires_grad=False ) if len( bonds_array ) else torch.Tensor()
            if len(bonds_array) == 0:
                bittensor.logging.warning("Empty bonds_array on metagraph.sync(). The 'bonds' tensor is empty.")    

    def save( self ) -> 'metagraph':
        r""" Saves this metagraph object's state_dict under bittensor root dir."""
        save_directory = get_save_dir( self.network, self.netuid  )
        os.makedirs( save_directory, exist_ok=True )
        graph_file = save_directory + f'/block-{self.block.item()}.pt'
        state_dict = self.state_dict()
        state_dict['axons'] = self.axons
        torch.save(state_dict, graph_file) 
        state_dict = torch.load( graph_file )
        return self

    def load( self ) -> 'metagraph':
        r""" Loads this metagraph object's state_dict from bittensor root dir. """
        self.load_from_path( get_save_dir( self.network, self.netuid ) )
    
    def load_from_path( self, dir_path:str ) -> 'metagraph':
        r""" Loads this metagraph object with state_dict under the specified path."""
        graph_file = latest_block_path( dir_path )
        state_dict = torch.load( graph_file )
        # self.info = bittensor.SubnetInfo.from_parameter_dict( state_dict['info'] ) if 'info' in state_dict else None
        # self.version = torch.nn.Parameter( state_dict['version'], requires_grad=False )
        self.n = torch.nn.Parameter( state_dict['n'], requires_grad=False )
        self.block = torch.nn.Parameter( state_dict['block'], requires_grad=False )
        self.uids = torch.nn.Parameter( state_dict['uids'], requires_grad=False )
        self.stake = torch.nn.Parameter( state_dict['stake'], requires_grad=False )
        self.total_stake = torch.nn.Parameter( state_dict['total_stake'], requires_grad=False )
        self.ranks = torch.nn.Parameter( state_dict['ranks'], requires_grad=False )
        self.trust = torch.nn.Parameter( state_dict['trust'], requires_grad=False )
        self.consensus = torch.nn.Parameter( state_dict['consensus'], requires_grad=False )
        self.validator_trust = torch.nn.Parameter( state_dict['validator_trust'], requires_grad=False )
        self.incentive = torch.nn.Parameter( state_dict['incentive'], requires_grad=False )
        self.emission = torch.nn.Parameter( state_dict['emission'], requires_grad=False )
        self.dividends = torch.nn.Parameter( state_dict['dividends'], requires_grad=False )
        self.active = torch.nn.Parameter( state_dict['active'], requires_grad=False )
        self.last_update = torch.nn.Parameter( state_dict['last_update'], requires_grad=False )
        self.validator_permit = torch.nn.Parameter( state_dict['validator_permit'], requires_grad=False )
        self.uids = torch.nn.Parameter( state_dict['uids'], requires_grad=False )
        self.axons = state_dict['axons']
        if 'weights' in state_dict:
            self.weights = torch.nn.Parameter( state_dict['weights'], requires_grad=False )
        if 'bonds' in state_dict:
            self.bonds = torch.nn.Parameter( state_dict['bonds'], requires_grad=False )
        return self


