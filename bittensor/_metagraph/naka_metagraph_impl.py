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

from typing import List
from loguru import logger

import ast
import pandas
import torch.nn.functional as f
import torch
import pickle
import json

import bittensor
import bittensor.utils.networking as net
import bittensor.utils.weight_utils as weight_utils

RAOPERTAO = 1000000000
U64MAX = 18446744073709551615

class Metagraph( torch.nn.Module ):
    r""" Maintains chain state as a torch.nn.Module.
        Interface:
            tau (:obj:`torch.FloatTensor` of shape :obj:`(1)`): 
                Current, per block, token emission rate.
            block (:obj:`torch.LongTensor` of shape :obj:`(1)`):
                State block number.
            uids (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n)`):
                UIDs for each neuron.
            
            stake (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n)`):
                Stake balance for each neuron ordered by uid.
                
            last_update (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n)`):
                Last emission call for each neuron ordered by uid.
            weights (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n, metagraph.n)`):
                Full weight matrix on chain ordered by uid.
            neurons (:obj:`torch.LongTensor` of shape :obj:`(metagraph.n, -1)`) 
                Tokenized endpoint information.
    """
    def __init__( self, subtensor, config ):
        r""" Initializes a new Metagraph torch chain interface object.
        """
        super(Metagraph, self).__init__()
        self.config = config
        if subtensor == None:
            subtensor = bittensor.subtensor( config = config)
        self.subtensor = subtensor
        self.clear()

    def clear( self ) -> 'Metagraph':
        r""" Erases Metagraph state.
        """
        self.version = torch.nn.Parameter( torch.tensor( [ bittensor.__version_as_int__ ], dtype=torch.int64), requires_grad=False )
        self.n = torch.nn.Parameter( torch.tensor( [0], dtype=torch.int64), requires_grad = False )
        self.tau = torch.nn.Parameter( torch.tensor( [1], dtype=torch.float32), requires_grad = False )
        self.block = torch.nn.Parameter( torch.tensor( [0], dtype=torch.int64), requires_grad = False )
        self.stake = torch.nn.Parameter(  torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self.ranks = torch.nn.Parameter(  torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self.trust = torch.nn.Parameter(  torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self.consensus = torch.nn.Parameter(  torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self.incentive = torch.nn.Parameter(  torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self.emission = torch.nn.Parameter(  torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self.dividends = torch.nn.Parameter(  torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self.active = torch.nn.Parameter(  torch.tensor( [], dtype=torch.int64), requires_grad=False )
        self.last_update = torch.nn.Parameter(  torch.tensor( [], dtype=torch.int64), requires_grad=False )
        self.weights = torch.nn.Parameter(  torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self.bonds = torch.nn.Parameter(  torch.tensor( [], dtype=torch.int64), requires_grad=False )
        self.endpoints = torch.nn.Parameter( torch.tensor( [], dtype=torch.int64), requires_grad=False )
        self.uids = torch.nn.Parameter( torch.tensor([], dtype = torch.int64),requires_grad=False )
        self._endpoint_objs = None
        self.neurons = None
        return self

    def forward (
        self, 
        uid: int, 
        row_weight: torch.FloatTensor 
    ) -> torch.FloatTensor:
        """
        Returns a dividend vector for a change in weights by computing the full incenvite function.
            Args:
                uid (int):
                    uid to set weights.
                row_weights: (torch.FloatTensor, shape =(n)):
                    normalized row to replace at uid.
            Returns:
                dividends (torch.FloatTensor):
                    Dividends for the entire network.
        """

        # Return if there are no neurons.
        if self.n.item() == 0:
            return torch.tensor([], dtype=torch.float32)

        # Raise if the passed weights are badly shaped.
        if torch.numel( row_weight ) != self.n.item():
            raise ValueError('Passed weight update must have the dimension of a row in W. Got {}, expected {}', row_weight.size(), self.n.item())

        # Reshape to fit weights.
        row_weight = row_weight.view( self.n )

        # Normalize row.
        if torch.abs( torch.sum( row_weight ) - 1 ) > 0.0001:
            row_weight = f.normalize(row_weight, p=1, dim=0)
        
        # Raise if the passed weights are badly shaped.
        if uid >= self.n.item():
            raise ValueError('Passed uid does not exist in the graph. Got {} > {}', uid, self.n.item())

        weight = self.W.detach().clone()
        weight[uid,:] = row_weight
        
        # Compute ranks.
        S = self.S.view(self.n, 1)
        Wt = torch.transpose(weight, 0, 1)
        R = torch.matmul(Wt, S).view(self.n)

        # Compute trust.
        T  = torch.matmul((Wt != 0).float(), S).view(self.n)

        # Compute consensus.
        rho = 10
        kappa = 0.5
        # Return if there is no stake.
        if torch.sum( self.S )  == 0:
            C = torch.sigmoid( rho * (T - kappa) ).view(self.n)
        else:
            C = torch.sigmoid( rho * (T / torch.sum(S) - kappa) ).view(self.n)

        # Compute incentive.
        Incentive = (R * C).view(self.n)
        print (Incentive)

        # Compute emission.
        if torch.sum(Incentive) == 0:
            Inflation = torch.zeros( (self.n.item()), dtype=torch.float32 ).view(self.n)
        else:
            Inflation = (self.tau * Incentive).view(self.n)
        print (Inflation)

        # Compute bonds.
        B = self.B.detach().clone().float()
        B_norm = f.normalize(B, p=1, dim=1)
        print (B_norm)

        # Dividends
        D = torch.matmul( B_norm.view(self.n, self.n), Inflation.view(self.n, 1) ).view(self.n) + 0.5 * Inflation.view(self.n)
        print (D)

        # Return dividends.
        return D.view(self.n)

    @property
    def S(self) -> torch.FloatTensor:
        """ Stake
        """
        return self.stake

    @property
    def R(self) -> torch.FloatTensor:
        """ Rank
        """
        return self.ranks

    @property
    def I(self) -> torch.FloatTensor:
        """ Incentive
        """
        return self.incentive

    @property
    def E(self) -> torch.FloatTensor:
        """ Emission
        """
        return self.emission

    @property
    def C(self) -> torch.FloatTensor:
        """ Consensus
        """
        return self.consensus

    @property
    def T(self) -> torch.FloatTensor:
        """ Trust
        """
        return self.trust

    @property
    def D(self) -> torch.FloatTensor:
        """ Dividends
        """
        return self.dividends

    @property
    def B(self) -> torch.FloatTensor:
        """ Bonds
        """
        return self.bonds
    
    @property
    def W(self) -> torch.FloatTensor:
        """ Weights
        """
        return self.weights

    @property
    def hotkeys( self ) -> List[str]:
        r""" Returns hotkeys for each neuron.
            Returns:
                hotkeys (:obj:`List[str] of shape :obj:`(metagraph.n)`):
                    Neuron hotkeys.
        """
        if self.n.item() == 0:
            return []
        return [ neuron.hotkey if neuron != bittensor.endpoint.dummy() else '' for neuron in self.endpoint_objs ]

    @property
    def coldkeys( self ) -> List[str]:
        r""" Returns coldkeys for each neuron.
            Returns:
                coldkeys (:obj:`List[str] of shape :obj:`(metagraph.n)`):
                    Neuron coldkeys.
        """
        if self.n.item() == 0:
            return []
        return [ neuron.coldkey if neuron != bittensor.endpoint.dummy() else '' for neuron in self.endpoint_objs ]

    @property
    def modalities( self ) -> List[str]:
        r""" Returns the modality for each neuron.
            Returns:
                coldkeys (:obj:`List[str] of shape :obj:`(metagraph.n)`):
                    Neuron coldkeys.
        """
        if self.n.item() == 0:
            return []
        return [ neuron.modality if neuron != bittensor.endpoint.dummy() else '' for neuron in self.endpoint_objs ]

    @property
    def addresses( self ) -> List[str]:
        r""" Returns ip addresses for each neuron.
            Returns:
                coldkeys (:obj:`List[str] of shape :obj:`(metagraph.n)`):
                    Neuron address.
        """
        if self.n.item() == 0:
            return []
        return [ net.ip__str__( neuron.ip_type, neuron.ip, neuron.port ) if neuron != bittensor.endpoint.dummy() else '' for neuron in self.endpoint_objs ]

    @property
    def endpoint_objs( self ) -> List['bittensor.Endpoint']:
        r""" Returns endpoints as objects.
            Returns:
                endpoint_obj (:obj:`List[bittensor.Endpoint] of shape :obj:`(metagraph.n)`):
                    Endpoints as objects.
        """
        if self.n.item() == 0:
            return []
        elif self._endpoint_objs != None:
            return self._endpoint_objs
        else:
            self._endpoint_objs = []
            for tensor in self.endpoints:
                obj = bittensor.endpoint.from_tensor( tensor )
                self._endpoint_objs.append( obj )
            return self._endpoint_objs

    def hotkey_to_uid( self, hotkey:str ) -> int:
        r""" Fetch uid according to hotkey. 
            Args: 
                hotkey: (`str`, required):
                    Hotkey to fetch the uid for.
            
            Return:
                uid: (`int`):
                    The uid for specified hotkey, -1 if hotkey does not exist.
        """ 
        if hotkey in self.hotkeys:
            return self.hotkeys.index(hotkey) 
        else:
            return -1

    def load( self, network:str = None  ) -> 'Metagraph':
        r""" Loads this metagraph object's state_dict from bittensor root dir.
            Args: 
                network: (:obj:`str`, required):
                    Name of state_dict to load, defaults to kusanagi
        """
        try:
            if network == None:
                network = self.subtensor.network
            metagraph_path = '~/.bittensor/' + str(network) + '.pt'
            metagraph_path = os.path.expanduser(metagraph_path)
            if os.path.isfile(metagraph_path):
                self.load_from_path( path = metagraph_path )
            else:
                logger.warning('Did not load metagraph from path: {}, file does not exist. Run metagraph.save() first.', metagraph_path)
        except Exception as e:
            logger.exception(e)
        return self

    def save( self, network:str = None ) -> 'Metagraph':
        r""" Saves this metagraph object's state_dict under bittensor root dir.
            Args: 
                network: (:obj:`str`, required):
                    Name of state_dict, defaults to kusanagi
        """
        if network == None:
            network = self.subtensor.network
        return self.save_to_path( path = '~/.bittensor/', filename = str(network) + '.pt')

    def load_from_path(self, path:str ) -> 'Metagraph':
        r""" Loads this metagraph object with state_dict under the specified path.
            Args: 
                path: (:obj:`str`, required):
                    Path to load state_dict.
        """
        full_path = os.path.expanduser(path)
        metastate = torch.load( full_path )
        return self.load_from_state_dict( metastate )

    def save_to_path(self, path:str, filename:str ) -> 'Metagraph':
        r""" Saves this metagraph object's state_dict to the specified path.
            Args: 
                path: (:obj:`str`, required):
                    Path to save state_dict.
        """
        full_path = os.path.expanduser(path)
        os.makedirs(full_path, exist_ok=True)
        metastate = self.state_dict()
        torch.save(metastate, full_path + '/' + filename)
        return self

    def load_from_state_dict(self, state_dict:dict ) -> 'Metagraph':
        r""" Loads this metagraph object from passed state_dict.
            Args: 
                state_dict: (:obj:`dict`, required):
                    Metagraph state_dict. Must be same as that created by save_to_path.
        """
        self.version = torch.nn.Parameter( state_dict['version'], requires_grad=False )
        self.n = torch.nn.Parameter( state_dict['n'], requires_grad=False )
        self.tau = torch.nn.Parameter( state_dict['tau'], requires_grad=False )
        self.block = torch.nn.Parameter( state_dict['block'], requires_grad=False )
        self.uids = torch.nn.Parameter( state_dict['uids'], requires_grad=False )
        self.stake = torch.nn.Parameter( state_dict['stake'], requires_grad=False )
        self.ranks = torch.nn.Parameter( state_dict['ranks'], requires_grad=False )
        self.trust = torch.nn.Parameter( state_dict['trust'], requires_grad=False )
        self.consensus = torch.nn.Parameter( state_dict['consensus'], requires_grad=False )
        self.incentive = torch.nn.Parameter( state_dict['incentive'], requires_grad=False )
        self.emission = torch.nn.Parameter( state_dict['emission'], requires_grad=False )
        self.dividends = torch.nn.Parameter( state_dict['dividends'], requires_grad=False )
        self.active = torch.nn.Parameter( state_dict['active'], requires_grad=False )
        self.last_update = torch.nn.Parameter( state_dict['last_update'], requires_grad=False )
        self.weights = torch.nn.Parameter( state_dict['weights'], requires_grad=False )
        self.bonds = torch.nn.Parameter( state_dict['bonds'], requires_grad=False )
        self.endpoints = torch.nn.Parameter( state_dict['endpoints'], requires_grad=False )
        self._endpoint_objs = None
        return self

    def retrieve_cached_neurons( self, block: int = None ):
        """
            Retrieves cached metagraph syncs from IPFS. 
        """
        raise Exception('Nakamoto is deprecated. Please use finney instead')

    def sync ( self, block: int = None, cached: bool = True, subtensor = None, netuid: int = 1 ) -> 'Metagraph':
        r""" Synchronizes this metagraph with the chain state.
        """
        logger.success(self.subtensor)
        if block == None:
            block = self.subtensor.get_current_block()
            if cached and self.subtensor.network in ("nakamoto", "local"):
                if bittensor.__use_console__:
                    with bittensor.__console__.status("Synchronizing Metagraph...", spinner="earth"):
                        try:
                            neurons = self.retrieve_cached_neurons( )
                        except:
                            # For some reason IPFS cache is down, fallback on regular sync
                            logger.warning("IPFS cache may be down, falling back to regular sync")
                            neurons = self.subtensor.neurons()
                        n_total = len(neurons)
                else:
                    try:
                        neurons = self.retrieve_cached_neurons( )
                    except:
                        # For some reason IPFS cache is down, fallback on regular sync
                        logger.warning("IPFS cache may be down, falling back to regular sync")
                        neurons = self.subtensor.neurons()
                    n_total = len(neurons)
            else:
                neurons = self.subtensor.neurons( block = block )
                n_total = len(neurons)
        else:
            if cached and self.subtensor.network in ("nakamoto", "local"):
                if bittensor.__use_console__:
                    with bittensor.__console__.status("Synchronizing Metagraph...", spinner="earth"):
                        try:
                            neurons = self.retrieve_cached_neurons( block = block )
                        except:
                            # For some reason IPFS cache is down, fallback on regular sync
                            logger.warning("IPFS cache may be down, falling back to regular sync to get block {}".format(block))
                            neurons = self.subtensor.neurons( block = block )
                        n_total = len(neurons)
                else:
                    try:
                        neurons = self.retrieve_cached_neurons( block = block )
                    except:
                        # For some reason IPFS cache is down, fallback on regular sync
                        logger.warning("IPFS cache may be down, falling back to regular sync to get block {}".format(block))
                        neurons = self.subtensor.neurons( block = block )
                    n_total = len(neurons)
            else:
                neurons = self.subtensor.neurons( block = block )
                n_total = len(neurons)

        # Fill arrays.
        uids = [ i for i in range(n_total) ]
        active = [ 0 for _ in range(n_total) ]
        stake = [ 0 for _ in range(n_total) ]
        ranks = [ 0 for _ in range(n_total) ]
        trust = [ 0 for _ in range(n_total) ]
        consensus = [ 0 for _ in range(n_total) ]
        incentive = [ 0 for _ in range(n_total) ]
        emission = [ 0 for _ in range(n_total) ]
        dividends = [ 0 for _ in range(n_total) ]
        last_updates = [ -1 for _ in range(n_total) ]
        endpoints = [ [-1 for _ in range(250) ]  for _ in range(n_total) ]
        weights = [ [ 0 for _ in range(n_total) ] for _ in range(n_total) ]
        bonds = [ [0 for _ in range(n_total) ] for _ in range(n_total) ]
        self._endpoint_objs = [ bittensor.endpoint.dummy() for _ in range(n_total) ]
        self.neurons = [None for _ in range(n_total)]
        for n in neurons:
            self.neurons[n.uid] = n
            uids[n.uid] = n.uid 
            active[n.uid] = n.active
            stake[n.uid] = n.stake 
            ranks[n.uid] = n.rank
            trust[n.uid] = n.trust
            consensus[n.uid] = n.consensus
            incentive[n.uid] = n.incentive
            dividends[n.uid] = n.dividends
            emission[n.uid] = n.emission
            last_updates[n.uid] = n.last_update
            endpoint =  bittensor.endpoint(
                version = int(n.version),
                uid = int(n.uid), 
                hotkey = str(n.hotkey), 
                ip_type = int(n.ip_type), 
                ip = str(n.ip), 
                port = int(n.port), 
                modality = int(n.modality), 
                coldkey = str(n.coldkey)
            )
            self._endpoint_objs[n.uid] = endpoint 
            endpoints[n.uid] = endpoint.to_tensor().tolist()
            if len(n.weights) > 0:
                w_uids, w_weights = zip(*n.weights)
                weights[n.uid] = weight_utils.convert_weight_uids_and_vals_to_tensor( n_total, w_uids, w_weights ).tolist()
            else:
                weights[n.uid] = [0] * n_total
            if len(n.bonds) > 0:
                b_uids, b_bonds = zip(*n.bonds)
                bonds[n.uid] = weight_utils.convert_bond_uids_and_vals_to_tensor( n_total, b_uids, b_bonds ).tolist()
            else:
                bonds[n.uid] = [0] * n_total

        # Set tensors.
        tn = torch.tensor( n_total, dtype=torch.int64 )
        tblock = torch.tensor( block, dtype=torch.int64 )
        tuids = torch.tensor( uids, dtype=torch.int64 )
        tactive = torch.tensor( active, dtype=torch.int64 )
        tstake = torch.tensor( stake, dtype=torch.float32 )
        tranks = torch.tensor( ranks, dtype=torch.float32 )
        ttrust = torch.tensor( trust, dtype=torch.float32 )
        tconsensus = torch.tensor( consensus, dtype=torch.float32 )
        tincentive = torch.tensor( incentive, dtype=torch.float32 )
        temission = torch.tensor( emission, dtype=torch.float32 )
        tdividends = torch.tensor( dividends, dtype=torch.float32 )
        tlast_update = torch.tensor( last_updates, dtype=torch.int64 )
        tbonds = torch.tensor( bonds, dtype=torch.int64 )
        tweights = torch.tensor( weights, dtype=torch.float32 )
        tendpoints = torch.tensor( endpoints, dtype=torch.int64 )

        # Normalize bond ownership.
        tbonds = torch.nn.functional.normalize( tbonds.float(), p=1, dim=0, eps=1e-12 ) * 0.5 + torch.eye( tn ) * 0.5

        # Set params.
        self.n = torch.nn.Parameter( tn, requires_grad=False )
        self.block = torch.nn.Parameter( tblock, requires_grad=False )
        self.uids = torch.nn.Parameter( tuids, requires_grad=False )
        self.stake = torch.nn.Parameter( tstake, requires_grad=False )
        self.ranks = torch.nn.Parameter( tranks, requires_grad=False )
        self.trust = torch.nn.Parameter( ttrust, requires_grad=False )
        self.consensus = torch.nn.Parameter( tconsensus, requires_grad=False )
        self.incentive = torch.nn.Parameter( tincentive, requires_grad=False )
        self.emission = torch.nn.Parameter( temission, requires_grad=False )
        self.dividends = torch.nn.Parameter( tdividends, requires_grad=False )
        self.active = torch.nn.Parameter( tactive, requires_grad=False )
        self.last_update = torch.nn.Parameter( tlast_update, requires_grad=False )
        self.weights = torch.nn.Parameter( tweights, requires_grad=False )
        self.bonds = torch.nn.Parameter( tbonds, requires_grad=False )
        self.endpoints = torch.nn.Parameter( tendpoints, requires_grad=False )
            
        # For contructor.
        return self

    def to_dataframe(self):
        try:
            index = self.uids.tolist()
            columns = [ 'uid', 'active', 'stake', 'rank', 'trust', 'consensus', 'incentive', 'dividends', 'emission']
            dataframe = pandas.DataFrame(columns = columns, index = index)
            for uid in self.uids.tolist():
                v = {
                    'uid': self.uids[uid].item(),
                    'active': self.active[uid].item(),             
                    'stake': self.stake[uid].item(),             
                    'rank': self.ranks[uid].item(),            
                    'trust': self.trust[uid].item(),             
                    'consensus': self.consensus[uid].item(),             
                    'incentive': self.incentive[uid].item(),             
                    'dividends': self.dividends[uid].item(),             
                    'emission': self.emission[uid].item()
                }
                dataframe.loc[uid] = pandas.Series( v )
            dataframe['uid'] = dataframe.index
            return dataframe
        except Exception as e:
            bittensor.logging.error('failed metagraph.to_dataframe()', str(e))
            return pandas.DataFrame()

    def to_wandb(self):
        wandb_info = {
            'metagraph_n': self.n.item(),
            'metagraph_tau': self.tau.item(),
            'metagraph_block': self.block.item(),
        }
        return wandb_info
            
    def __str__(self):
        return "Metagraph({}, {}, {})".format(self.n.item(), self.block.item(), self.subtensor.network)
        
    def __repr__(self):
        return self.__str__()
