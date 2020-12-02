import argparse
import munch
import torch

class Dendrite(torch.nn.Module):
    def __init__(self, config, session):
        self.config = config
        self.session = session

    def forward_image(self, images, query):
        r""" Forwards images to connected neurons using the passed context to learn connectivity.

            Args:
                images (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, channels, rows, cols)`, `required`): 
                    Image tensors to forward.
                
                context (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, query_dim)`, `required`): 
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
        raise NotImplementedError

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
        raise NotImplementedError

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
        """
        raise NotImplementedError   

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:    
        return parser
    
    @staticmethod
    def check_config(config: munch.Munch) -> munch.Munch:  
        return config