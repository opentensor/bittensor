
""" Maintains mocked chain state as a torch.nn.Module.
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

import pandas
import torch
from . import metagraph_impl
import bittensor
import bittensor.utils.networking as net
import scalecodec
import binascii
import random

RAOPERTAO = 1000000000
U64MAX = 18446744073709551615

class MockMetagraph( metagraph_impl.Metagraph ):
    r""" MOCKED VERSION OF THE METAGRAPH

    """
    def __init__( self ) -> None:
        r""" Initializes a new Metagraph torch chain interface object.
        """
        super(MockMetagraph, self).__init__(network="mock", netuid=-1)
        tn = torch.tensor( 2000, dtype=torch.int64 )
        tblock = torch.tensor( 0, dtype=torch.int64 )
        tuids = torch.tensor( list( range( 2000 )), dtype=torch.int64 )
        tactive = torch.tensor( [ 1 for _ in range (2000)], dtype=torch.int64 )

        ttotal_stake = torch.tensor( [ 1.0 for _ in range (2000) ], dtype=torch.float32 )

        tranks = torch.tensor(  [1.0/2000 for _ in range (2000) ], dtype=torch.float32 )
        ttrust = torch.tensor( [ 1.0 for _ in range (2000) ], dtype=torch.float32 )
        tconsensus = torch.tensor( [1.0 for _ in range (2000) ], dtype=torch.float32 )
        tvalidator_trust = torch.tensor( [ 1.0 for _ in range (2000) ], dtype=torch.float32 )
        tincentive = torch.tensor( [1.0/2000 for _ in range (2000) ], dtype=torch.float32 )
        temission = torch.tensor( [1.0/2000 for _ in range (2000) ], dtype=torch.float32 )
        tdividends = torch.tensor( [1.0/2000 for _ in range (2000) ], dtype=torch.float32 )
        tlast_update = torch.tensor( [0 for _ in range (2000) ], dtype=torch.int64 )
        tvalidator_permit = torch.tensor( [False for _ in range (2000) ], dtype=torch.bool )
        tbonds = torch.tensor( [ [1 for _ in range (2000) ] for _ in range (2000) ], dtype=torch.int64 )
        tweights = torch.tensor( [ [1.0/2000 for _ in range (2000) ] for _ in range (2000) ], dtype=torch.float32 )
        self._endpoint_objs = [ bittensor.endpoint.dummy() for _ in range (2000) ]
        tendpoints = torch.tensor( [ end.to_tensor().tolist() for end in self._endpoint_objs ], dtype=torch.int64 )
        self.n = torch.nn.Parameter( tn, requires_grad=False )
        self.block = torch.nn.Parameter( tblock, requires_grad=False )
        self.uids = torch.nn.Parameter( tuids, requires_grad=False )

        self.stake = [{ 
            scalecodec.ss58_encode( 
                random.randint(0, U64MAX).to_bytes(32, 'big', signed=False), bittensor.__ss58_format__
            ): bittensor.Balance.from_rao(1.0) for _ in range(random.randint(0, 10)) # random addresses to Balance(1.0)
        } for _ in range(2000) ] # 1 dict of addresses -> Balance per uid
        self.total_stake = torch.nn.Parameter( ttotal_stake, requires_grad=False )

        self.ranks = torch.nn.Parameter( tranks, requires_grad=False )
        self.trust = torch.nn.Parameter( ttrust, requires_grad=False )
        self.consensus = torch.nn.Parameter( tconsensus, requires_grad=False )
        self.validator_trust = torch.nn.Parameter( tvalidator_trust, requires_grad=False )
        self.incentive = torch.nn.Parameter( tincentive, requires_grad=False )
        self.emission = torch.nn.Parameter( temission, requires_grad=False )
        self.dividends = torch.nn.Parameter( tdividends, requires_grad=False )
        self.active = torch.nn.Parameter( tactive, requires_grad=False )
        self.last_update = torch.nn.Parameter( tlast_update, requires_grad=False )
        self.validator_permit = torch.nn.Parameter( tvalidator_permit, requires_grad=False )
        self.weights = torch.nn.Parameter( tweights, requires_grad=False )
        self.bonds = torch.nn.Parameter( tbonds, requires_grad=False )
        self.endpoints = torch.nn.Parameter( tendpoints, requires_grad=False )

        print("---- MOCKED METAGRAPH INITIALIZED ----")

    def clear( self ) -> 'bittensor.Metagraph':
        r""" Erases Metagraph state.
        """
        self.version = torch.nn.Parameter( torch.tensor( [ bittensor.__version_as_int__ ], dtype=torch.int64), requires_grad=False )
        self.n = torch.nn.Parameter( torch.tensor( [0], dtype=torch.int64), requires_grad = False )
        self.tau = torch.nn.Parameter( torch.tensor( [1], dtype=torch.float32), requires_grad = False )
        self.block = torch.nn.Parameter( torch.tensor( [0], dtype=torch.int64), requires_grad = False )

        self.stake = []
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
        self.endpoints = torch.nn.Parameter( torch.tensor( [], dtype=torch.int64), requires_grad=False )
        self.uids = torch.nn.Parameter( torch.tensor([], dtype = torch.int64),requires_grad=False )
        self._endpoint_objs = None
        return self

    def load( self, network:str = None  ) -> 'bittensor.Metagraph':
        r""" Loads this metagraph object's state_dict from bittensor root dir.
            Args: 
                network: (:obj:`str`, required):
                    Name of state_dict to load, defaults to kusanagi
        """
        try:
            if network == None:
                network = 'mock'
            metagraph_path = '~/.bittensor/' + str(network) + '.pt'
            metagraph_path = os.path.expanduser(metagraph_path)
            if os.path.isfile(metagraph_path):
                self.load_from_path( path = metagraph_path )
            else:
                logger.warning('Did not load metagraph from path: {}, file does not exist. Run metagraph.save() first.', metagraph_path)
        except Exception as e:
            logger.exception(e)
        return self

    def save( self, network:str = None ) -> 'bittensor.Metagraph':
        r""" Saves this metagraph object's state_dict under bittensor root dir.
            Args: 
                network: (:obj:`str`, required):
                    Name of state_dict, defaults to kusanagi
        """
        if network == None:
            network = 'mock'
        return self.save_to_path( path = '~/.bittensor/', filename = 'mock.pt')

    def sync ( self, block: int = None, cached: bool = True ) -> 'bittensor.Metagraph':
        return self