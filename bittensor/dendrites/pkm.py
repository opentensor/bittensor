import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from loguru import logger
from typing import List, Tuple

import bittensor
from bittensor import bittensor_pb2

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

class PKMDendrite():
    def __init__(self, config, session, query_dim):
        self.config = config
        self.session = session
        # UIDs -> Keys.
        self.keys = PKMKeys(self.config.dendrite.key_dim)
        # Query -> Keys
        self.projection = nn.Linear(query_dim, self.config.dendrite.key_dim, bias=True)

    def _filter_uids(self, uids: torch.Tensor) -> torch.Tensor:
        # uids: filter uids based on emit time.
        # uids = [-1]
        block = self.session.metagraph.state.block 
        emit = self.session.metagraph.state.emit
        staleness = (block - emit)
        uids = uids[torch.where(staleness < self.config.dendrite.stale_emit_filter)] 
        return uids

    def _route(self, inputs, query, modality: bittensor_pb2.Modality) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        r""" Routes inputs using context and metagraph state.

            Args:
                inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, *-1*)`, `required`): 
                    tensors inputs to distribute to neurons using context.
                
                query (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, query_dimension)`, `required`): 
                    Context tensor used to select which neurons query for each example.

                modality (:obj:`bittensor_pb2.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality type. Enum in [TEXT, IMAGE, TENSOR]

            Returns:
                responses (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, bittensor.__network_dim__)`, `required`): 
                    Joined responses from each queried neuron.

                weights (:obj:`torch.LongTensor` of shape :obj:`(batch_size, metagraph.state.n)`, `optional`): 
                    weights for each neuron per example.

                retops (:obj:`torch.LongTensor` of shape :obj:`(batch_size, metagraph.state.n)`, `optional`): 
                    return op from each call per example.
        """
        # For ease of use.
        batch_size = inputs.shape[0]

        # all_uids: (torch.LongTensor): unique keys for each peer neuron.
        # all_uids.shape = [metagraph.n]
        all_uids = self.session.metagraph.state.uids # Returns a list of neuron uids.

        # filtered_uids: (torch.LongTensor): keys filtered by emit.
        # all_uids.shape = [metagraph.n]
        filtered_uids = self._filter_uids(all_uids)
        n_uids = torch.numel(filtered_uids)

        # Return if there are no uids to query
        if n_uids == 0:
            # Return nill responses.
            n = self.session.metagraph.state.n
            return torch.zeros(size=(inputs.shape[0], bittensor.__network_dim__)), torch.zeros(size=(inputs.shape[0], n)), torch.zeros(n)

        # keys: (torch.FloatTensor): unique trainable torch keys for each uid
        # keys.shape = [n_uids, config.dendrite.key_dim]
        keys = self.keys( filtered_uids )

        # query: (torch.FloatTensor): projection of the query on to the key dimension.
        # query.shape = [batch_size, config.dendrite.key_dim]
        query = self.projection( query )

        # scores: (torch.FloatTensor): cartesian product between keys and projection.
        # scores.shape = [batch_size, n_uids]
        scores = F.linear(query, keys, bias=None)

        # topk_scores: (torch.FloatTensor): topk scores per example
        # topk_indices: (torch.LongTensor): topk indices per example
        # topk_scores.shape = [batch_size, real_topk]
        # topk_indices.shape = [batch_size, real_topk]
        real_topk = min( n_uids, self.config.dendrite.topk )
        topk_scores, topk_indices = scores.topk(real_topk, dim=1) 

        # gates: (torch.FloatTensor): gated scores for uid per example. Zeros for non queried uids.
        # gates.shape = [batch_size, n_uids]
        zeros = torch.zeros(batch_size, n_uids)
        gates = zeros.scatter(1, topk_indices, topk_scores)

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

        # part_sizes: List (int): split sizes into inputs expanded.
        # len(part_sizes) = [n_uids]
        part_sizes = list((gates != 0.0).sum(0).cpu().numpy())

        # requests: List(torch.FloatTensor): examples for each uids
        # requests.shape = n_uids * [-1, inputs.shape[1:]]
        requests = torch.split(inputs_expanded, part_sizes, dim=0)
        
        # neurons: List[bittensor_pb2.Neuron]: endpoint information for filtered keys.
        # neurons.shape = n_uids * [ bittensor_pb2.Neuron ]
        neurons = self.session.metagraph.state.uids_to_neurons(filtered_uids)

        # responses: image responses from neurons.
        # responses.shape = neurons.size * [-1, sequence_dim, __network_dim__]
        if modality == bittensor_pb2.Modality.TEXT:
            responses, retops = self.session.dendrite.forward_text(neurons, requests)

        elif modality == bittensor_pb2.Modality.IMAGE:
            responses, retops = self.session.dendrite.forward_image(neurons, requests)

        elif modality == bittensor_pb2.Modality.TENSOR:
            responses, retops = self.session.dendrite.forward_tensor(neurons, requests)

        else:
            raise NotImplementedError

        # stitched: (torch.FloatTensor): responses joined along the first dimension.
        # stitched.shape = [real_topk, sequence_dim, bittensor.__network_dim__] 
        # flat_stitched: (torch.FloatTensor): responses joined and flattened along the last dimension.
        # flat_stitched.shape = [real_topk * batch_size, *inputs.shape[1:]] 
        stitched = torch.cat(responses, 0)
        flat_stitched = torch.flatten(stitched, start_dim = 1)

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
        combined = zeros.index_add(0, batch_index, flat_stitched.float())

        # combined: (torch.FloatTensor): combined responses reshaped to correct dimension.
        # combined = [batch_size, sequence_dim, bittensor.__network_dim__]
        combined = combined.view(batch_size, inputs.shape[1], bittensor.__network_dim__)

        # indices: (torch.LongTensor): indices of uids repeated by batch size
        # indices = [batch_size, metagraph.state.n]
        indices = self.session.metagraph.state.uids_to_indices(all_uids).repeat(batch_size, 1)

        # weights: (torch.LongTensor): weights scattered onto uids per example.
        # weights.shape = [batch_size, metagraph.state.n]
        weights = torch.zeros(inputs.shape[0], self.session.metagraph.state.n)
        weights.scatter_(1, indices, scores)

        # Return.
        return combined, weights, retops


    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:    
        parser.add_argument('--dendrite.key_dim', default=100, type=int, help='Product keys dimension.')
        parser.add_argument('--dendrite.topk', default=10, type=int, help='Number of keys to select for each example.')
        parser.add_argument('--dendrite.stale_emit_filter', default=10000, type=int, help='Number of blocks before a neuron is filtered without a recent emit')
        return parser

    @staticmethod
    def check_config(config):   
        return config

    def forward_image(self, images: torch.FloatTensor, query: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        r""" Forwards images to connected neurons using the passed context to learn connectivity.

            Args:
                images (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, channels, rows, cols)`, `required`): 
                    Image tensors to forward.
                
                query (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, context_dim)`, `required`): 
                    query tensor used to select which neurons query for each example.
            
            Returns:
                responses (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, bittensor.__network_dim__)`, `required`): 
                    Joined responses from each queried neuron.

                weights (:obj:`torch.LongTensor` of shape :obj:`(batch_size, metagraph.state.n)`, `optional`): 
                    weights for each neuron per example.

                retops (:obj:`torch.LongTensor` of shape :obj:`(batch_size, metagraph.state.n)`, `optional`): 
                    return op from each call per example.
        """
        return self._route(images, query, bittensor_pb2.Modality.IMAGE)

    def forward_text(self, text: torch.LongTensor, query: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        r""" Forwards text to connected neurons using the passed context to learn connectivity.

            Args:
                text (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_dim)`, `required`): 
                    tensor of tokenized sentences.
                
                query (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, query_dim)`, `required`): 
                    Context tensor used to select which neurons query for each example.
            
            Returns:
                responses (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, bittensor.__network_dim__)`, `required`): 
                    Joined responses from each queried neuron.

                weights (:obj:`torch.LongTensor` of shape :obj:`(batch_size, metagraph.state.n)`, `optional`): 
                    weights for each neuron per example.

                retops (:obj:`torch.LongTensor` of shape :obj:`(batch_size, metagraph.state.n)`, `optional`): 
                    return op from each call per example.
                
        """
        return self._route(text, query, bittensor_pb2.Modality.TEXT)


    def forward_tensor(self, tensors: torch.FloatTensor, query: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        r""" Forwards tensors to connected neurons using the passed context to learn connectivity.

            Args:
                tensors (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, bittensor.__network_dim__)`, `required`): 
                    tensors sent to connected neurons.
                
                query (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, query_dim)`, `required`): 
                    Query tensor used to select which neurons query for each example.
            
            Returns:
                responses (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, bittensor.__network_dim__)`, `required`): 
                    Joined responses from each queried neuron.

                weights (:obj:`torch.LongTensor` of shape :obj:`(batch_size, metagraph.state.n)`, `optional`): 
                    weights for each neuron per example.

                retops (:obj:`torch.LongTensor` of shape :obj:`(batch_size, metagraph.state.n)`, `optional`): 
                    return op from each call per example.
        """
        return self._route(tensors, query, bittensor_pb2.Modality.IMAGE)
