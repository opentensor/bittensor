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

import asyncio
import os
import torch
import tqdm.asyncio

from loguru import logger
from typing import List, Tuple, List

import bittensor
import bittensor.utils.networking as net
import bittensor.utils.weight_utils as weight_utils

class Metagraph( torch.nn.Module ):
    r""" Maintains chain state as a torch.nn.Module.

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
                Tokenized endpoint information.

    """
    def __init__( self, subtensor ):
        r""" Initializes a new Metagraph torch chain interface object.
        """
        super(Metagraph, self).__init__()
        self.subtensor = subtensor
        self.version = torch.nn.Parameter( torch.tensor( [ bittensor.__version_as_int__ ], dtype=torch.int64), requires_grad=False )
        self.n = torch.nn.Parameter( torch.tensor( [0], dtype=torch.int64), requires_grad=False )
        self.tau = torch.nn.Parameter( torch.tensor( [0.5], dtype=torch.float32), requires_grad=False )
        self.block = torch.nn.Parameter( torch.tensor( [0], dtype=torch.int64), requires_grad=False )
        self.uids = torch.nn.Parameter( torch.tensor( [], dtype=torch.int64), requires_grad=False )
        self.stake = torch.nn.Parameter( torch.tensor( [], dtype=torch.float32), requires_grad=False )
        self.lastemit = torch.nn.Parameter( torch.tensor( [], dtype=torch.int64), requires_grad=False )
        self.weights = torch.nn.ParameterList()
        self.neurons = torch.nn.ParameterList()
        self.cached_endpoints = None

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
        if self.n.item() == 0:
            return torch.tensor([], dtype=torch.float32)
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
            return torch.tensor([], dtype=torch.float32)
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
        if self.n.item() == 0:
            return torch.tensor( [], dtype=torch.float32 )
        return torch.stack( [row for row in self.weights], axis = 0 )

    @property
    def hotkeys( self ) -> List[str]:
        r""" Returns hotkeys for each neuron.
            Returns:
                hotkeys (:obj:`List[str] of shape :obj:`(metagraph.n)`):
                    Neuron hotkeys.
        """
        if self.n.item() == 0:
            return []
        return [ neuron.hotkey for neuron in self.endpoints ]

    @property
    def coldkeys( self ) -> List[str]:
        r""" Returns coldkeys for each neuron.
            Returns:
                coldkeys (:obj:`List[str] of shape :obj:`(metagraph.n)`):
                    Neuron coldkeys.
        """
        if self.n.item() == 0:
            return []
        return [ neuron.coldkey for neuron in self.endpoints ]

    @property
    def modalities( self ) -> List[str]:
        r""" Returns the modality for each neuron.
            Returns:
                coldkeys (:obj:`List[str] of shape :obj:`(metagraph.n)`):
                    Neuron coldkeys.
        """
        if self.n.item() == 0:
            return []
        return [ neuron.modality for neuron in self.endpoints ]

    @property
    def addresses( self ) -> List[str]:
        r""" Returns ip addresses for each neuron.
            Returns:
                coldkeys (:obj:`List[str] of shape :obj:`(metagraph.n)`):
                    Neuron address.
        """
        if self.n.item() == 0:
            return []
        return [ net.ip__str__( neuron.ip_type, neuron.ip, neuron.port ) for neuron in self.endpoints ]

    @property
    def endpoints(self) -> List[ 'bittensor.Endpoint' ]:
        r""" Return neuron endpoint information for each neuron.
            
            Returns:
                neurons (:obj:`List[ bittensor.Endpoint ]` of shape :obj:`(metagraph.n)`):
                    Endpoint information for each neuron.

        """
        if self.n.item() == 0:
            return []
        if self.cached_endpoints != None:
            return self.cached_endpoints
        else:
            self.cached_endpoints = []
            for idx, neuron_tensor in enumerate(self.neurons):
                try:
                    neuron_endpoint = bittensor.endpoint.from_tensor( neuron_tensor )
                    self.cached_endpoints.append ( neuron_endpoint )
                except Exception as e:
                    self.cached_endpoints.append ( None )
                    logger.exception('Faulty endpoint tensor: {} got error while trying to serialize as endpoint: {} ', neuron_tensor, e)
            return self.cached_endpoints

    def load( self, network:str = None  ):
        r""" Loads this metagraph object's state_dict from bittensor root dir.
            Args: 
                network: (:obj:`str`, required):
                    Name of state_dict to load, defaults to kusanagi
        """
        if network == None:
            network = self.subtensor.network
        metagraph_path = '~/.bittensor/' + str(network) + '.pt'
        metagraph_path = os.path.expanduser(metagraph_path)
        if os.path.isfile(metagraph_path):
            self.load_from_path( path = metagraph_path )
        else:
            logger.warning('Did not load metagraph from path: {}, file does not exist. Run metagraph.save() first.', metagraph_path)

    def save( self, network:str = None ):
        r""" Saves this metagraph object's state_dict under bittensor root dir.
            Args: 
                network: (:obj:`str`, required):
                    Name of state_dict, defaults to kusanagi
        """
        if network == None:
            network = self.subtensor.network
        self.save_to_path( path = '~/.bittensor/' + str(network) + '.pt')

    def load_from_path(self, path:str ):
        r""" Loads this metagraph object with state_dict under the specified path.
            Args: 
                path: (:obj:`str`, required):
                    Path to load state_dict.
        """
        full_path = os.path.expanduser(path)
        metastate = torch.load( full_path )
        self.load_from_state_dict( metastate )

    def save_to_path(self, path:str ):
        r""" Saves this metagraph object's state_dict to the specified path.
            Args: 
                path: (:obj:`str`, required):
                    Path to save state_dict.
        """
        full_path = os.path.expanduser(path)
        metastate = self.state_dict()
        torch.save(metastate, full_path)

    def load_from_state_dict(self, state_dict:dict ):
        r""" Loads this metagraph object from passed state_dict.
            Args: 
                state_dict: (:obj:`dict`, required):
                    Metagraph state_dict. Must be same as that created by save_to_path.
        """
        if 'version' in state_dict:
            self.version = torch.nn.Parameter( state_dict['version'], requires_grad=False )
            self.n = torch.nn.Parameter( state_dict['n'], requires_grad=False )
            self.tau = torch.nn.Parameter( state_dict['tau'], requires_grad=False )
            self.block = torch.nn.Parameter( state_dict['block'], requires_grad=False )
            self.uids = torch.nn.Parameter( state_dict['uids'], requires_grad=False )
            self.stake = torch.nn.Parameter( state_dict['stake'], requires_grad=False )
            self.lastemit = torch.nn.Parameter( state_dict['lastemit'], requires_grad=False )
            self.weights = torch.nn.ParameterList([torch.nn.Parameter( state_dict['weights.' + str(i)], requires_grad=False )  for i in range(self.n.item()) ])
            self.neurons = torch.nn.ParameterList([torch.nn.Parameter( state_dict['neurons.' + str(i)], requires_grad=False )  for i in range(self.n.item()) ])
        else:
            # Dont load because we need to force a new reload.
            pass

        self.cached_endpoints = None

    def sync(self, force: bool = False ):
        r""" Synchronizes this metagraph with the chain state.
            Args: 
                subtensor: (:obj:`bittensor.Subtensor`, optional):
                    Subtensor chain interface obbject. If None, creates a default connection.
                force (bool):
                    force syncs all nodes on the graph.
        """
        loop = asyncio.get_event_loop()
        loop.set_debug(enabled=True)
        loop.run_until_complete(self._async_sync(force))

    async def _async_sync( self, force: bool = False ):
        r""" Uses the passed subtensor interface to update the metagraph chain state to reflect 
            the latest info on chain.

            Args: 
                subtensor: (:obj:`bittensor.Subtensor`, optional):
                    Subtensor chain interface obbject. If None, creates default connection to kusanagi.
        """
        # Query chain info.
        chain_lastemit = dict( await self.subtensor.async_get_last_emit() ) #  Optional[ List[Tuple[uid, lastemit]] ]
        chain_stake = dict( await self.subtensor.async_get_stake() ) #  Optional[ List[Tuple[uid, stake]] ]
        chain_block = int( await self.subtensor.async_get_current_block()) #  Optional[ int ]

        # Build new state.
        new_size = len(chain_stake)
        old_size = self.n.item() 
        old_block = self.block.item()
        new_n = torch.tensor([new_size], dtype=torch.int64)
        new_block = torch.tensor([chain_block], dtype=torch.int64)
        new_uids = torch.tensor( range(new_size) ,  dtype=torch.int64)
        new_stake = torch.tensor([ (float(chain_stake[uid])/1000000000) for uid in range(new_size)],  dtype=torch.float32)
        new_lastemit = torch.tensor([ chain_lastemit[uid] for uid in range(new_size)], dtype=torch.int64)

        # Set params.
        self.n = torch.nn.Parameter( new_n, requires_grad=False )
        self.block = torch.nn.Parameter( new_block, requires_grad=False )
        self.uids = torch.nn.Parameter( new_uids, requires_grad=False )
        self.stake = torch.nn.Parameter( new_stake, requires_grad=False )
        self.lastemit = torch.nn.Parameter( new_lastemit, requires_grad=False )

        # Extend weights matrix.
        for idx in range( old_size ):
            self.weights[idx] =  torch.nn.Parameter( torch.cat( [self.weights[idx], torch.zeros([new_size - len(self.weights[idx])], dtype=torch.float32)]))

        # Create buffers
        for _ in range( new_size - old_size ):
            self.weights.append( torch.nn.Parameter( torch.tensor([], dtype=torch.float32), requires_grad=False ) )
            self.neurons.append( torch.nn.Parameter( torch.tensor([], dtype=torch.int64), requires_grad=False ) )

        # Fill pending queries.
        pending_queries = []
        for uid, lastemit in chain_lastemit.items():
            if lastemit > old_block or force == True:
                pending_queries.append((False, uid))

        # Fill buffers with retry.
        # Below fills buffers for pending queries upto the rety cutoff.
        retries = 0
        max_retries = 3
        while True:
            if retries >= max_retries:
                logger.critical('Failed to sync metagraph. Check your subtensor connection.')
                raise RuntimeError('Failed to sync metagraph. Check your subtensor connection.')            
            queries = []
            for code, uid in pending_queries:
                if code == False:
                    queries.append( self.fill_uid( uid = uid ) )
            if len(queries) == 0:
                # Success
                break
            pending_queries = [await query for query in tqdm.asyncio.tqdm.as_completed( queries )]
            retries += 1 
            
        self.cached_endpoints = None

    # Function which fills weights and neuron info for a uid.
    async def fill_uid ( self, uid: int ) -> Tuple[int, bool]:
        r""" Uses the passed subtensor interface to update chain state for the passed uid.
            the latest info on chain.
            
            Args: 
                subtensor: (:obj:`bittensor.Subtensor`, optional):
                    Subtensor chain interface obbject. If None, creates default connection to kusanagi.
        """
        # TODO(const): try catch block with retry.
        try:
            
            # Fill row from weights.
            weight_uids = await self.subtensor.async_weight_uids_for_uid( uid ) 
            weight_vals = await self.subtensor.async_weight_vals_for_uid( uid ) 
            row_weights = weight_utils.convert_weight_uids_and_vals_to_tensor( self.n.item(), weight_uids, weight_vals )
            self.weights[ uid ] = torch.nn.Parameter( row_weights, requires_grad=False )
            
            # Fill Neuron info.
            neuron = await self.subtensor.async_get_neuron_for_uid( uid )
            neuron_obj = bittensor.endpoint.from_dict( neuron )
            neuron_tensor = neuron_obj.to_tensor()
            self.neurons[ uid ] = torch.nn.Parameter( neuron_tensor, requires_grad=False )
            
            # Return.
            return True, uid

        except:
            # Return False.
            return False, uid

    def __str__(self):
        if self.n != 0:
            peers_online = torch.numel(torch.where( self.block - self.lastemit < 1000 )[0])
        else:
            peers_online = 0
        return '<green>Metagraph:</green> block:<blue>{}</blue>, inflation_rate:<blue>{}</blue>, staked:<green>\u03C4{}</green>/<blue>\u03C4{}</blue>, active:<green>{}</green>/<blue>{}</blue>'.format(self.block.item(), self.tau.item(), torch.sum(self.S), self.block.item()/2, peers_online, self.n.item())

    def __to_tensorboard__(self, tensorboard, global_step):
        tensorboard.add_scalar('Metagraph/neurons', self.n.item(), global_step)
        tensorboard.add_scalar('Metagraph/inflation_rate', self.tau.item(), global_step)


