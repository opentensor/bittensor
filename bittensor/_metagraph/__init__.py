""" Create and init metagraph, 
which maintains chain state as a torch.nn.Module.
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

import copy
import torch
import argparse
import bittensor
from . import metagraph_impl
from . import metagraph_mock
from typing import Optional, List, Union
import bittensor.utils.weight_utils as weight_utils
from .naka_metagraph_impl import Metagraph as naka_metagraph

class metagraph:
    """ Factory class for the bittensor.Metagraph class or the MockMetagraph
    The Metagraph object serves as the main storage unit for the chain state. 
    By default, it stores all chain information as a torch.nn.Module which can be
    synced using a subtensor connection.

    Examples:: 
            >>> subtensor = bittensor.subtensor(network='nakamoto')
            >>> metagraph = bittensor.metagraph()
            >>> metagraph.sync(subtensor=subtensor, netuid=0)
    """
    def __new__(
            cls, 
            config: 'bittensor.config' = None,
            network: str = None,
            netuid: Optional[int] = None,
            subtensor: 'bittensor.Subtensor' = None,
            _mock:bool=None
        ) -> 'bittensor.Metagraph':
        r""" Creates a new bittensor.Metagraph object from passed arguments.
            Args:
                config (:obj:`bittensor.Config`, `optional`): 
                    bittensor.metagraph.config()
                network (default=None, type=str, optional)
                    The subtensor network flag. The likely choices are:
                            -- nobunaga (staging network)
                            -- nakamoto (main network)
                            -- local (local running network)
                    This option allows you to load a metagraph from a local file.
                    If set, overrides config.subtensor.network
                netuid (default=None, type=int)
                    The subnet netuid. If set, overrides config.netuid.
                    This option allows you to load a metagraph from a local file.
                _mock (:obj:`bool`, `optional`):
                    For testing, if true the metagraph returns mocked outputs.
        """      
        if config == None: 
            config = metagraph.config()
        config = copy.deepcopy(config)
        config.metagraph._mock = _mock if _mock != None else config.metagraph._mock
        if config.metagraph._mock:
            return metagraph_mock.MockMetagraph()
        if subtensor != None:
            network = subtensor.network
        if netuid == None:
            netuid = config.get('netuid', None)
        if network == None:
            network = config.subtensor.get('network', bittensor.defaults.subtensor.network)

        if network =='nakamoto':
            config.subtensor.network = 'nakamoto'
            return naka_metagraph(config = config, subtensor = subtensor)
        else:
            return metagraph_impl.Metagraph( network = network, netuid = netuid )

    @classmethod   
    def config(cls) -> 'bittensor.Config':
        """ Get config from teh argument parser
        Return: bittensor.config object
        """
        parser = argparse.ArgumentParser()
        metagraph.add_args( parser )
        return bittensor.config( parser )

    @classmethod   
    def help(cls):
        """ Print help to stdout
        """
        parser = argparse.ArgumentParser()
        cls.add_args( parser )
        print (cls.__new__.__doc__)
        parser.print_help()

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser, prefix: str = None ):
        """ Add specific arguments from parser, 
        which is the identical to subtensor  
        """
        prefix_str = '' if prefix == None else prefix + '.'
        try:
            parser.add_argument('--' + prefix_str + 'metagraph._mock', action='store_true', help='To turn on metagraph mocking for testing purposes.', default=False)
            bittensor.subtensor.add_args( parser )
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass
        bittensor.subtensor.add_args( parser, prefix = prefix )

    @classmethod   
    def check_config( cls, config: 'bittensor.Config' ):
        """ Check config,
        which is identical to subtensor
        """
        pass

    @staticmethod
    def from_neurons( network: str, netuid: int, info: 'bittensor.SubnetInfo', neurons: Union[List['bittensor.NeuronInfo'], List['bittensor.NeuronInfoLite']], block: int ) -> 'bittensor.Metagraph':
        r""" Creates a metagraph from a list of neurons.
            Args: 
                network: (:obj:`str`, required):
                    Name of the network for the metagraph.
                netuid: (:obj:`int`, required):
                    netuid of the subnet for the metagraph.
                info: (:obj:`SubnetInfo`, required):
                    SubnetInfo object for the metagraph, including the subnet's hyperparameters.
                neurons: (:obj:`Union[List[NeuronInfo], List[NeuronInfoLite]]`, required):
                    List of neurons to create metagraph from.
                block: (:obj:`int`, required):
                    Block number at time of the metagraph.
        """
        metagraph = metagraph_impl.Metagraph( network = network, netuid = netuid )
        metagraph.info = info

        n_total = len(neurons)

        # Fill arrays.
        uids = [ i for i in range(n_total) ]
        active = [ 0 for _ in range(n_total) ]
        stake = [ {} for _ in range(n_total) ]
        total_stake = [ 0 for _ in range(n_total) ]
        ranks = [ 0 for _ in range(n_total) ]
        trust = [ 0 for _ in range(n_total) ]
        consensus = [ 0 for _ in range(n_total) ]
        validator_trust = [ 0 for _ in range(n_total) ]
        incentive = [ 0 for _ in range(n_total) ]
        emission = [ 0 for _ in range(n_total) ]
        dividends = [ 0 for _ in range(n_total) ]
        last_updates = [ -1 for _ in range(n_total) ]
        validator_permit = [ False for _ in range(n_total) ]
        endpoints = [ [-1 for _ in range(250) ]  for _ in range(n_total) ]
        weights = [ [ 0 for _ in range(n_total) ] for _ in range(n_total) ]
        bonds = [ [0 for _ in range(n_total) ] for _ in range(n_total) ]
        metagraph._endpoint_objs = [ bittensor.endpoint.dummy() for _ in range(n_total) ]
        metagraph.neurons = [None for _ in range(n_total)]
        for n in neurons:
            metagraph.neurons[n.uid] = n
            uids[n.uid] = n.uid 
            active[n.uid] = n.active
            stake[n.uid] = n.stake # stake is a Dict[str, Balance]
            total_stake[n.uid] = n.total_stake.tao 
            ranks[n.uid] = n.rank
            trust[n.uid] = n.trust
            consensus[n.uid] = n.consensus
            validator_trust[n.uid] = n.validator_trust
            incentive[n.uid] = n.incentive
            dividends[n.uid] = n.dividends
            emission[n.uid] = n.emission
            last_updates[n.uid] = n.last_update
            validator_permit[n.uid] = n.validator_permit
            endpoint =  bittensor.endpoint.from_neuron(n)
            metagraph._endpoint_objs[n.uid] = endpoint 
            endpoints[n.uid] = endpoint.to_tensor().tolist()
            if isinstance(n, bittensor.NeuronInfoLite):
                continue
            # Weights and bonds only for full neurons.
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
       
        ttotal_stake = torch.tensor( total_stake, dtype=torch.float32 )

        tranks = torch.tensor( ranks, dtype=torch.float32 )
        ttrust = torch.tensor( trust, dtype=torch.float32 )
        tconsensus = torch.tensor( consensus, dtype=torch.float32 )
        tvalidator_trust = torch.tensor( validator_trust, dtype=torch.float32 )
        tincentive = torch.tensor( incentive, dtype=torch.float32 )
        temission = torch.tensor( emission, dtype=torch.float32 )
        tdividends = torch.tensor( dividends, dtype=torch.float32 )
        tlast_update = torch.tensor( last_updates, dtype=torch.int64 )
        tvalidator_permit = torch.tensor( validator_permit, dtype=torch.bool )
        tbonds = torch.tensor( bonds, dtype=torch.int64 )
        tweights = torch.tensor( weights, dtype=torch.float32 )
        tendpoints = torch.tensor( endpoints, dtype=torch.int64 )

        # Normalize bond ownership.
        tbonds = torch.nn.functional.normalize( tbonds.float(), p=1, dim=0, eps=1e-12 ) * 0.5 + torch.eye( tn ) * 0.5

        # Set params.
        metagraph.n = torch.nn.Parameter( tn, requires_grad=False )
        metagraph.block = torch.nn.Parameter( tblock, requires_grad=False )
        metagraph.uids = torch.nn.Parameter( tuids, requires_grad=False )

        metagraph.stake = stake
        metagraph.total_stake = torch.nn.Parameter( ttotal_stake, requires_grad=False )
        
        metagraph.ranks = torch.nn.Parameter( tranks, requires_grad=False )
        metagraph.trust = torch.nn.Parameter( ttrust, requires_grad=False )
        metagraph.consensus = torch.nn.Parameter( tconsensus, requires_grad=False )
        metagraph.validator_trust = torch.nn.Parameter( tvalidator_trust, requires_grad=False )
        metagraph.incentive = torch.nn.Parameter( tincentive, requires_grad=False )
        metagraph.emission = torch.nn.Parameter( temission, requires_grad=False )
        metagraph.dividends = torch.nn.Parameter( tdividends, requires_grad=False )
        metagraph.active = torch.nn.Parameter( tactive, requires_grad=False )
        metagraph.last_update = torch.nn.Parameter( tlast_update, requires_grad=False )
        metagraph.validator_permit = torch.nn.Parameter( tvalidator_permit, requires_grad=False )
        metagraph.weights = torch.nn.Parameter( tweights, requires_grad=False )
        metagraph.bonds = torch.nn.Parameter( tbonds, requires_grad=False )
        metagraph.endpoints = torch.nn.Parameter( tendpoints, requires_grad=False )

        return metagraph