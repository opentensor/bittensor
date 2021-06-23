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

import torch
from types import SimpleNamespace
import bittensor

class Router( torch.nn.Module ):
    def __init__( self ):
        super().__init__()

    def sync_chain_state( self, metagraph: 'bittensor.Metagraph' ):
        r""" Creates new parameters based on metagraph size.

            Args:
                metagraph (:obj: `bittensor.Metagraph'`, `required`):
                    bittensor metagraph object.
        """
        raise NotImplementedError()

    def _route(self, metagraph: 'bittensor.Metagraph', dendrite: 'bittensor.Dendrite', inputs: torch.FloatTensor, query: torch.FloatTensor, modality: bittensor.proto.Modality) -> SimpleNamespace:
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
        raise NotImplementedError()
    

    def forward_image(self, metagraph: 'bittensor.Metagraph', dendrite: 'bittensor.Dendrite', images: torch.FloatTensor, query: torch.FloatTensor) -> SimpleNamespace:
        r""" Forwards images to connected neurons using the passed context to learn connectivity.

            Args:
                metagraph (:obj: `bittensor.Metagraph`, `required`):
                    bittensor metagraph object. Used to pull network endpoint info.

                dendrite (:obj: `bittensor.Dendrite`, `required`):
                    bittensor dendrite object. User to make queries into the network.

                images (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, channels, rows, cols)`, `required`): 
                    Image tensors to forward.
                
                query (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, context_dim)`, `required`): 
                    query tensor used to select which neurons query for each example.
            
            Returns:
                SimpleNamespace {
                    responses (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, bittensor.__network_dim__)`, `required`): 
                        Joined responses from each queried neuron.

                    weights (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, metagraph.state.n)`, `optional`): 
                        weights for each neuron per example.
                    
                    uids (:obj:`torch.LongTensor` of shape :obj:`(-1)`, `required`): 
                        Uids of endpoints queried.

                    requests_sizes (:obj:`torch.LongTensor` of shape :obj:`(metagraph.state.n)`, `optional`): 
                        number of requests sent to each uid in this batch.

                    return_codes (:obj:`List[torch.LongTensor]` of shape :obj:`[num_neurons]`, `required`):
                        dendrite call return codes.
                }
        """
        return self._route(metagraph, dendrite, images, query, bittensor.proto.Modality.IMAGE)

    def forward_text(self, metagraph: 'bittensor.Metagraph', dendrite: 'bittensor.Dendrite', text: torch.LongTensor, query: torch.FloatTensor) -> SimpleNamespace:
        r""" Forwards text to connected neurons using the passed context to learn connectivity.

            Args:
                metagraph (:obj: `bittensor.Metagraph`, `required`):
                    bittensor metagraph object. Used to pull network endpoint info.

                dendrite (:obj: `bittensor.Dendrite`, `required`):
                    bittensor dendrite object. User to make queries into the network.

                text (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_dim)`, `required`): 
                    tensor of tokenized sentences.
                
                query (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, query_dim)`, `required`): 
                    Context tensor used to select which neurons query for each example.
            
            Returns:
                SimpleNamespace {
                    responses (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, bittensor.__network_dim__)`, `required`): 
                        Joined responses from each queried neuron.

                    weights (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, metagraph.state.n)`, `optional`): 
                        weights for each neuron per example.

                    uids (:obj:`torch.LongTensor` of shape :obj:`(-1)`, `required`): 
                        Uids of endpoints queried.

                    requests_sizes (:obj:`torch.LongTensor` of shape :obj:`(metagraph.state.n)`, `optional`): 
                        number of requests sent to each uid in this batch.

                    return_codes (:obj:`List[torch.LongTensor]` of shape :obj:`[num_neurons]`, `required`):
                        dendrite call return codes.
                }
                
        """
        return self._route( metagraph, dendrite, text, query, bittensor.proto.Modality.TEXT)

    def forward_tensor(self, metagraph: 'bittensor.Metagraph', dendrite: 'bittensor.Dendrite', tensors: torch.FloatTensor, query: torch.FloatTensor) -> SimpleNamespace:
        r""" Forwards tensors to connected neurons using the passed context to learn connectivity.

            Args:
                metagraph (:obj: `bittensor.Metagraph`, `required`):
                    bittensor metagraph object. Used to pull network endpoint info.

                dendrite (:obj: `bittensor.Dendrite`, `required`):
                    bittensor dendrite object. User to make queries into the network.

                tensors (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, bittensor.__network_dim__)`, `required`): 
                    tensors sent to connected neurons.
                
                query (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, query_dim)`, `required`): 
                    Query tensor used to select which neurons query for each example.
            
            Returns:
                SimpleNamespace {
                    responses (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, bittensor.__network_dim__)`, `required`): 
                        Joined responses from each queried neuron.

                    weights (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, metagraph.state.n)`, `optional`): 
                        weights for each neuron per example.

                    uids (:obj:`torch.LongTensor` of shape :obj:`(-1)`, `required`): 
                        Uids of endpoints queried.

                    requests_sizes (:obj:`torch.LongTensor` of shape :obj:`(metagraph.state.n)`, `optional`): 
                        number of requests sent to each uid in this batch.

                    return_codes (:obj:`List[torch.LongTensor]` of shape :obj:`[num_neurons]`, `required`):
                        dendrite call return codes.
                }
        """
        return self._route( metagraph, dendrite, tensors, query, bittensor.proto.Modality.IMAGE )
