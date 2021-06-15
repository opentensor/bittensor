import argparse
import torch
import torch.nn as nn

from torch.nn import functional as F
from typing import Tuple
from types import SimpleNamespace

import bittensor
from . import router

class PKMKeys(nn.Module):

    def __init__(self, key_dim):
        super().__init__()
        self._key_dim = key_dim
        self._n_keys = 10 # Initial size = 10
        self._keys = torch.rand(self._n_keys, self._key_dim)

    def forward(self, uids: torch.Tensor) -> torch.Tensor:
        r""" Maps metagraph uids to torch keys
            Args:
                uids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_dim, channels, rows, cols)`, `required`): 
                    Image tensors to forward.

            Returns:
                keys (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, key_dim)`, `required`): 
                    Torch key for each uid.
        """
        # Get max value for possible resize.
        max_uid = torch.max(uids)
        if max_uid >= self._n_keys - 1:
            new_keys = torch.rand( (max_uid - self._n_keys) + 10, self._key_dim)
            self._keys = torch.cat([self._keys, new_keys], dim=0)
            self._n_keys = self._keys.shape[0]
        return self._keys[uids]

class PKMRouter( router.Router ):
    def __init__(self, config: 'bittensor.Config', query_dim = bittensor.__network_dim__, **kwargs):
        super().__init__()
        if config == None:
            config = PKMRouter.config();       
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # UIDs -> Keys.
        self.keys = PKMKeys(self.config.router.key_dim)
        # Query -> Keys
        self.projection = nn.Linear(query_dim, self.config.router.key_dim, bias=True).to(self.device)

    @staticmethod   
    def config() -> 'bittensor.Config':
        parser = argparse.ArgumentParser()
        PKMRouter.add_args(parser) 
        config = bittensor.config( parser ); 
        PKMRouter.check_config(config)
        return config

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:    
        parser.add_argument('--pkm.key_dim', default=100, type=int, help='Product keys dimension.')
        parser.add_argument('--pkm.topk', default=10, type=int, help='Number of keys to select for each example.')
        parser.add_argument('--pkm.stale_emit_filter', default=10000, type=int, help='Number of blocks before a neuron is filtered without a recent emit')
        return parser

    @staticmethod
    def check_config(config):   
        return config

    def sync_chain_state( self, metagraph: 'bittensor.Metagraph' ):
        r""" Creates new parameters based on metagraph size.

            Args:
                metagraph (:obj: `bittensor.Metagraph'`, `required`):
                    bittensor metagraph object.
        """
        pass

    def _route(self, metagraph: 'bittensor.Metagraph', dendrite: 'bittensor.Dendrite', inputs: torch.FloatTensor, query: torch.FloatTensor, modality: bittensor.proto.Modality) -> SimpleNamespace:
        r""" Routes inputs using context and metagraph state.

            Args:

                metagraph (:obj: `bittensor.Metagraph`, `required`):
                    bittensor metagraph object. Used to pull network endpoint info.

                dendrite (:obj: `bittensor.Dendrite`, `required`):
                    bittensor dendrite object. User to make queries into the network.

                inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, *-1*)`, `required`): 
                    tensors inputs to distribute to neurons using context.
                
                query (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, query_dimension)`, `required`): 
                    Context tensor used to select which neurons query for each example.

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
        output = SimpleNamespace ()

        # For ease of use.
        batch_size = inputs.shape[0]

        # all_uids: (torch.LongTensor): unique keys for each peer neuron.
        # all_uids.shape = [metagraph.n]
        all_uids = metagraph.uids # Returns a list of neuron uids.

        # filtered_uids: (torch.LongTensor): keys filtered by emit.
        # all_uids.shape = [metagraph.n]
        current_block = metagraph.block
        lastemit = metagraph.lastemit
        staleness = (current_block - lastemit)
        filtered_uids = all_uids[torch.where(staleness < self.config.pkm.stale_emit_filter)] 
        n_uids = torch.numel(filtered_uids)

        # Return if there are no uids to query.
        if n_uids == 0:
            # Return nill responses.
            n = metagraph.n
            output.response = torch.zeros(size=(inputs.shape[0], inputs.shape[1], bittensor.__network_dim__))
            output.weights = torch.zeros(size = ( n ), dtype=torch.float32)
            output.uids = torch.zeros([], dtype=torch.int64)
            output.requests_sizes = torch.zeros([], dtype=torch.int64)
            output.return_codes = torch.zeros([], dtype=torch.int64)
            return output

        # keys: (torch.FloatTensor): unique trainable torch keys for each uid
        # keys.shape = [n_uids, config.router.key_dim]
        keys = self.keys( filtered_uids ).to(self.device)

        # query: (torch.FloatTensor): projection of the query on to the key dimension.
        # query.shape = [batch_size, config.router.key_dim]
        # On Cuda if it's available.
        query = self.projection( query )

        # scores: (torch.FloatTensor): cartesian product between keys and projection.
        # scores.shape = [batch_size, n_uids]
        # These are all on Cuda if it's available.
        scores = F.linear(query, keys, bias=None)
        scores = F.softmax(scores, dim = 1) # Softmax scores

        # topk_scores: (torch.FloatTensor): topk scores per example
        # topk_indices: (torch.LongTensor): topk indices per example
        # topk_scores.shape = [batch_size, real_topk]
        # topk_indices.shape = [batch_size, real_topk]
        # These are all on Cuda if it's available.
        real_topk = min( n_uids, self.config.router.topk )
        topk_scores, topk_indices = scores.topk(real_topk, dim=1) 

        # gates: (torch.FloatTensor): gated scores for uid per example. Zeros for non queried uids.
        # gates.shape = [batch_size, n_uids]
        zeros = torch.zeros(batch_size, n_uids).to(self.device)
        gates = zeros.scatter(1, topk_indices, topk_scores)
        gates = F.normalize(gates, p=1, dim=1)

        # non_zero_gates: (torch.FloatTensor): indices of non-zero gate values.
        # non_zero_gates.shape = [numel(gates), 2]
        # sorted_indices: (torch.FloatTensor): sorted indices along the first dimension i.e. indices ordered by row.
        # sorted_uids.shape = [batch_size, n_uids]
        non_zero_gates = torch.nonzero(gates)
        sorted_uids, index_sorted_uids =  torch.sort(non_zero_gates, dim = 0)

        # uids_index: torch.FloatTensor): batch index of sorted uids.
        # uids_indes.shape = [topk * batch_size, 1] 
        _, uids_index = sorted_uids.split(1, dim=1)

        # batch_index: (torch.FloatTensor): batch index for each uid x example
        # batch_index.shape =  [topk * batch_size]
        batch_index = sorted_uids[index_sorted_uids[:, 1], 0]

        # inputs_expanded: (torch.FloatTensor): expanded inputs to topk * batch_size
        # inputs_expanded.shape = [topk * batch_size, -1]
        inputs_expanded = inputs[batch_index]

        # request_sizes: List (int): number of examples per uid.
        # len(part_sizes) = [n_uids]
        request_sizes = list((gates != 0.0).sum(0).cpu().numpy())

        # requests: List(torch.FloatTensor): examples for each uids
        # requests.shape = n_uids * [-1, inputs.shape[1:]]
        requests = torch.split(inputs_expanded, request_sizes, dim=0)
        
        # neurons: List[bittensor.proto.Neuron]: endpoint information for filtered keys.
        # neurons.shape = n_uids * [ bittensor.proto.Neuron ]
        # TODO(const): switch to tokenized representation.
        filtered_endpoints = []
        for uid in filtered_uids:
            filtered_endpoints.append( metagraph.endpoints[ uid ] )

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

        # stitched: (torch.FloatTensor): responses joined along the first dimension.
        # stitched.shape = [real_topk, sequence_dim, bittensor.__network_dim__] 
        # flat_stitched: (torch.FloatTensor): responses joined and flattened along the last dimension.
        # flat_stitched.shape = [real_topk * batch_size, *inputs.shape[1:]] 
        stitched = torch.cat(responses, 0)
        flat_stitched = torch.flatten(stitched, start_dim = 1).to(self.device)

        # gates_expanded: (torch.FloatTensor): gate values for each queried uid per example.
        # gates_expanded.shape = [real_topk * batch_size, n_uids]
        gates_expanded = gates[batch_index.flatten()]

        # nonzero_gates: (torch.FloatTensor): non-zero gating values for each example for each uid.
        # nonzero_gates.shape = [real_topk * batch_size, 1]
        nonzero_gates = torch.gather(gates_expanded, 1, uids_index)

        # flat_stitched: (torch.FloatTensor): responses multiplied by gate values.
        # flat_stitched.shape = [real_topk * batch_size, *inputs.shape[1:]] 
        flat_stitched = flat_stitched.mul(nonzero_gates)

        # zeros: (torch.FloatTensor): zero for combined responses. 
        # zeros.shape = [batch_size, *inputs.shape[1:]] 
        zeros = torch.zeros(batch_size, flat_stitched.shape[1], requires_grad=True)

        # combined: (torch.FloatTensor): combine responses by adding them to the corresponsing batch index.
        # combined = [batch_size, *inputs.shape[1:]] 
        combined = zeros.to(self.device).index_add(0, batch_index, flat_stitched.float())

        # combined: (torch.FloatTensor): combined responses reshaped to correct dimension.
        # combined = [batch_size, sequence_dim, bittensor.__network_dim__]
        combined = combined.view(batch_size, inputs.shape[1], bittensor.__network_dim__)

        # indices: (torch.LongTensor): indices of uids queried during this forward call.
        # indices = [batch_size, metagraph.n]
        indices = filtered_uids

        # weights: (torch.LongTensor): weights scattered onto uids per example.
        # weights.shape = [metagraph.n]
        weights = torch.zeros(inputs.shape[0], metagraph.n)
        weights = weights.to(self.device)
        indices = indices.to(self.device)
        weights.scatter_(1, indices.repeat(batch_size, 1), gates)
        weights = torch.mean( weights, axis = 0 )

        # filled_sizes: (torch.LongTensor): number of examples queried to each uid.
        # filled_sizes.shape = [metagraph.n]
        filled_request_sizes = torch.zeros(metagraph.n, dtype=torch.long).to(self.device)
        request_sizes = torch.tensor(request_sizes).to(self.device)
        filled_request_sizes.scatter_(0, indices, torch.tensor(request_sizes).to(self.device))

        # Return.
        output.response = combined
        output.weights = weights
        output.uids = filtered_uids
        output.request_sizes = filled_request_sizes
        output.return_codes = retops
        return output