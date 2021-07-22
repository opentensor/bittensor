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
import torch
import torch.nn.functional as F
from typing import List, Tuple
from types import SimpleNamespace
import bittensor
from . import router

class SGMOERouter( router.Router ):
    def __init__(self, config: 'bittensor.Config' = None, query_dim = bittensor.__network_dim__, **kwargs):
        super().__init__()
        if config == None: config = SGMOERouter.config();       
        self.config = config.router
        self.query_dim = query_dim
        
        # Gating weights. Should match the metagraph.n
        self.gates = torch.nn.ModuleList()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod   
    def config() -> 'bittensor.Config':
        parser = argparse.ArgumentParser()
        SGMOERouter.add_args( parser )
        return bittensor.config( parser )


    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        parser.add_argument('--router.topk', default=20, type=int, help='Number of uids to query each batch.')
        parser.add_argument('--router.stale_emit_filter', default=10000, type=int, help='Number of blocks without an update before a neuron is filtered')

    @staticmethod
    def check_config(config: 'bittensor.Config'):
        pass

    def sync_with_chain_state( self, metagraph: 'bittensor.Metagraph' ):
        r""" Creates new parameters based on metagraph size.

            Args:
                metagraph (:obj: `bittensor.Metagraph'`, `required`):
                    bittensor metagraph object.
        """
        # Add new gates for each uid.
        for uid in metagraph.uids.tolist():
            self.gate_for_uid = torch.nn.Linear( self.query_dim, 1, bias=True)
            self.gates.append( self.gate_for_uid )

    def _route(
            self, 
            metagraph: 'bittensor.Metagraph', 
            dendrite: 'bittensor.Dendrite', 
            inputs: torch.FloatTensor, 
            query: torch.FloatTensor, 
            modality: bittensor.proto.Modality
        ) -> SimpleNamespace:
        r""" Routes inputs using context and metagraph state.

            Args:
                metagraph (:obj: `bittensor.Metagraph`, `required`):
                    Bittensor metagraph object. Used to pull network endpoint info.

                dendrite (:obj: `bittensor.Dendrite`, `required`):
                    Bittensor dendrite object. Used to make queries into the network.

                inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, *-1*)`, `required`): 
                    Tensor inputs to distribute to neurons using query context.
                
                query (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, query_dimension)`, `required`): 
                    Context tensor used to select which neurons to query for each example.

                modality (:obj:`bittensor.proto.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality type. Enum in [TEXT, IMAGE, TENSOR]

            Returns:
                output = SimpleNamespace {
                    responses (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, bittensor.__network_dim__)`, `required`): 
                        Joined responses from each queried neuron.

                    weights (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.state.n)`, `required`): 
                        Weights for each neuron per example.

                    uids (:obj:`torch.LongTensor` of shape :obj:`(n_topk)`, `required`): 
                        Uids of neurons queried.

                    requests_sizes (:obj:`torch.LongTensor` of shape :obj:`(n_topk)`, `required`): 
                        Number of requests sent to each uid.

                    return_codes (:obj:`torch.LongTensor` of shape :obj:`(n_topk)`, `required`):
                        Return code from each query for each queried uid.
                }
        """
        # To be filled.
        output = SimpleNamespace ()

        # For ease of use.
        batch_size = inputs.shape[0]

        # Get all uids.
        # all_uids: (torch.LongTensor): unique keys for each peer neuron.
        # all_uids.shape = [metagraph.n]
        all_uids = metagraph.uids # Returns a list of neuron uids.

        # Filter uids based on last emit.
        # filtered_uids: (torch.LongTensor): keys filtered by emit.
        # all_uids.shape = [metagraph.n]
        current_block = metagraph.block
        lastemit = metagraph.lastemit
        staleness = (current_block - lastemit)
        filtered_uids = all_uids[torch.where(staleness < self.config.stale_emit_filter)]
        n_filtered = torch.numel(filtered_uids)

        # Return if there are no uids to query.
        if n_filtered == 0:
            # Return nill responses.
            n = metagraph.n
            output.response = torch.zeros(size=(inputs.shape[0], inputs.shape[1], bittensor.__network_dim__))
            output.weights = torch.zeros(size = [ n ], dtype=torch.float32)
            output.uids = torch.zeros([], dtype=torch.int64)
            output.requests_sizes = torch.zeros([], dtype=torch.int64)
            output.return_codes = torch.zeros([], dtype=torch.int64)
            return output

        # Get weights for uids.
        # weights: (torch.FloatTensor): weights for each filtered_uid
        # weights.shape = [n_filtered]
        weights = torch.cat( [ self.gates[ uid ].to(self.device)(query) for uid in filtered_uids.tolist() ], axis = 1)

        # Normalize weights across batch dimension. 
        # filtered_weights_mean: (torch.FloatTensor): normalized weights across batch dimension. 
        # filtered_weights_mean.shape = [ n_filtered ]
        filtered_mean_weights = torch.mean(weights, axis = 0)


        # Get indices and values for uids with highest scores.
        # topk_weights: (torch.float64): scores of uids with highest scores.
        # topk_weights.shape = [ real_topk ]
        # topk_indices: (torch.LongTensor): indicies of uids with highest scores.
        # topk_indices.shape = [ real_topk ]
        real_topk = min( n_filtered, self.config.topk )
        topk_weights, topk_indices = filtered_mean_weights.topk(real_topk, dim=0) 

        # Get the real uids with the top scores.
        # real_filtered_topk_uids: (torch.LongTensor): uids with highest scores.
        # real_filtered_topk_uids.shape = [ real_topk ]
        real_filtered_topk_uids = filtered_uids[ topk_indices ].to(self.device)
        
        # Get endpoint information for the highest scoring uids.
        # neurons: List[bittensor.proto.Neuron]: endpoint information for filtered uids.
        # len(neurons) == real_topk
        filtered_endpoints = []
        for uid in real_filtered_topk_uids:
            filtered_endpoints.append( metagraph.endpoints[ uid ] )

        # Get request for uids with highest scores.
        # requests: List[torch.FloatTensor]: requests for high scoring uids.
        # len(requests) == real_topk
        requests = [ inputs for _ in range( len(filtered_endpoints) )]

        # Makes queries into the network.
        # responses: List[torch.float64]: responses from each uid.
        # responses.shape = real_topk * [batch_size, sequence_dim, __network_dim__]
        if modality == bittensor.proto.Modality.TEXT:
            responses, retops = dendrite.forward_text(endpoints = filtered_endpoints, inputs = requests)

        elif modality == bittensor.proto.Modality.IMAGE:
            responses, retops = dendrite.forward_image(endpoints = filtered_endpoints, inputs = requests)

        elif modality == bittensor.proto.Modality.TENSOR:
            responses, retops = dendrite.forward_tensor(endpoints = filtered_endpoints, inputs = requests)
        else:
            raise NotImplementedError

        weighted_responses = torch.zeros( ( batch_size, inputs.shape[1], bittensor.__network_dim__ )).to(self.device)
        indices = torch.where(retops == 0)[0].to(self.device)
        if torch.numel(indices) > 0:
            soft_topk_weights = F.softmax( topk_weights[indices], dim = 0 ).to(self.device)
            if torch.numel(indices[0]) != 0:
                for soft_topk_weight, index in list(zip(soft_topk_weights, indices)): 
                    weighted_responses += responses[index].to(self.device) * soft_topk_weight
    

        # Normalize scores.
        # scores: (torch.FloatTensor): normalized scores.
        # scores.shape = [real_topk]
        scores = topk_weights
        scores = scores - torch.min(scores)
        scores = scores / torch.sum(scores)

        scores = scores.to(self.device)

        # Set weighted response.
        output.response = weighted_responses
        
        # Set uids that we have queried. 
        output.uids = real_filtered_topk_uids 

        # Scatter scores on to metagraph dimension.
        output.weights = torch.scatter( torch.zeros( (metagraph.n), dtype=torch.float32).to(self.device), 0, real_filtered_topk_uids, scores )
        
        # Set request sizes.
        output.request_sizes = torch.scatter( torch.zeros( (metagraph.n), dtype=torch.float32).to(self.device), 0, real_filtered_topk_uids, batch_size )
        
        # Set return codes.
        output.return_codes = retops

        # Return.
        return output