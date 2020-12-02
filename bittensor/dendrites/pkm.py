import argparse
import torch
from loguru import logger

import bittensor
from bittensor import bittensor_pb2
from bittensor.utils.router import Router

class PKMDendrite():
    def __init__(self, config, session, context_dim):
        self.config = config
        self.session = session
        self.context_dim = context_dim
        self.router = Router(x_dim = self.context_dim, key_dim = self.config.dendrite.key_dim, topk = self.config.dendrite.topk)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:    
        parser.add_argument('--dendrite.key_dim', default=100, type=int, help='Product keys dimension.')
        parser.add_argument('--dendrite.topk', default=10, type=int, help='Number of keys to select for each example.')
        parser.add_argument('--dendrite.stale_emit_filter', default=10000, type=int, help='Number of blocks before a neuron is filtered without a recent emit')
        return parser

    @staticmethod
    def check_config(config):   
        return config

    def forward_image(self, images, context):
        r""" Forwards images to connected neurons using the passed context to learn connectivity.

            Args:
                images (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, channels, rows, cols)`, `required`): 
                    Image tensors to forward.
                
                context (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, context_dim)`, `required`): 
                    Context tensor used to select which neurons query for each example.
            
            Returns:
                bittensor.DendriteOutput
                { 
                    responses (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, bittensor.__network_dim__)`, `required`): 
                        Joined responses from each queried neuron.

                    weights (:obj:`torch.LongTensor` of shape :obj:`(batch_size, metagraph.state.n)`, `optional`): 
                        weights for each neuron per example.
                }
        """
        return self._route(images, context, bittensor_pb2.Modality.IMAGE)

    def forward_text(self, text, context):
        r""" Forwards text to connected neurons using the passed context to learn connectivity.

            Args:
                text (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim)`, `required`): 
                    tensor of tokenized sentences.
                
                context (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, context_dim)`, `required`): 
                    Context tensor used to select which neurons query for each example.
            
            Returns:
                bittensor.DendriteOutput
                { 
                    responses (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, bittensor.__network_dim__)`, `required`): 
                        Joined responses from each queried neuron.

                    weights (:obj:`torch.LongTensor` of shape :obj:`(batch_size, metagraph.state.n)`, `optional`): 
                        weights for each neuron per example.
                }
        """
        return self._route(text, context, bittensor_pb2.Modality.TEXT)


    def forward_tensor(self, tensors, context):
        r""" Forwards tensors to connected neurons using the passed context to learn connectivity.

            Args:
                tensors (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, bittensor.__network_dim__)`, `required`): 
                    tensors sent to connected neurons.
                
                context (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, context_dim)`, `required`): 
                    Context tensor used to select which neurons query for each example.
            
            Returns:
            
                responses (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, bittensor.__network_dim__)`, `required`): 
                    Joined responses from each queried neuron.

                weights (:obj:`torch.LongTensor` of shape :obj:`(batch_size, metagraph.state.n)`, `optional`): 
                    weights for each neuron per example.

                 (:obj:`torch.LongTensor` of shape :obj:`(batch_size, metagraph.state.n)`, `optional`): 
                    weights for each neuron per example.
        """
        return self._route(tensors, context, bittensor_pb2.Modality.IMAGE)

    def _route(self, inputs, context, modality: bittensor_pb2.Modality):
        r""" Routes inputs using context and metagraph state.

            Args:
                inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, *-1*)`, `required`): 
                    tensors inputs to distribute to neurons using context.
                
                context (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, context_dimension)`, `required`): 
                    Context tensor used to select which neurons query for each example.

                modality (:obj:`bittensor_pb2.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality type. Enum in [TEXT, IMAGE, TENSOR]

            Returns:
                responses (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, bittensor.__network_dim__)`, `required`): 
                    Joined responses from each queried neuron.

                weights (:obj:`torch.LongTensor` of shape :obj:`(batch_size, metagraph.state.n)`, `optional`): 
                    weights for each neuron per example.

                retops (:obj:`torch.LongTensor` of shape :obj:`(batch_size, metagraph.state.n)`, `optional`): 
                    return op from each call per example
        """

        # uids: unique keys for peer neurons.
        # uids.shape = [metagraph.n]
        uids = self.session.metagraph.state.uids # Returns a list of neuron uids.
       
        # uids: uids with an emit call in the last 100 blocks.
        # uids = [-1]
        block = self.session.metagraph.state.block 
        emit = self.session.metagraph.state.emit
        staleness = (block - emit)
        uids = uids[torch.where(staleness < self.config.dendrite.stale_emit_filter)] 

        # Return zeros if there are no remaining peer neurons.
        if torch.numel(uids) == 0:
            n = self.session.metagraph.state.n
            remote_context = torch.zeros(size=(inputs.shape[0], bittensor.__network_dim__))
            weights = torch.zeros(size=(inputs.shape[0], n))
            retops = torch.zeros(n)
            return remote_context, weights, retops

        # neurons: endpoint information for filtered keys.
        # neurons.shape = [len(uids)]
        neurons = self.session.metagraph.state.uids_to_neurons(uids)
        
        # request: inputs routed to peers using context to filter topk.
        # request.shape = neurons.size * [-1, sequence_dim, channels, rows, cols]

        logger.info('inputs shape {}', inputs.shape)
        logger.info('neurons length {}', len(neurons))
        requests, weights = self.router.route( neurons, context, inputs ) 
        logger.info('requests length {}', len(requests))
        logger.info('request shapes {}', [el.shape for el in requests])


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

        # remote_context: Responses weighted and joined along the __network_dim__.
        # remote_context.shape = [batch_size, bittensor.__network_dim__]
        remote_context = self.router.join( responses )
        remote_context = remote_context.view(remote_context.shape[0] * remote_context.shape[1], remote_context.shape[2])

        # scatter weights back onto shape (bs, n)
        indices = self.session.metagraph.state.uids_to_indices(uids).repeat(inputs.shape[0], 1)
        filled_weights = torch.zeros(inputs.shape[0], self.session.metagraph.state.n)
        filled_weights.scatter_(1, indices, weights)
        return remote_context, filled_weights, retops