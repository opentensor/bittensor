'''
The MIT License (MIT)
Copyright © 2021 Opentensor.ai

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.
'''

"""Feed Forward NN Synapse

Simple feed forward NN for images.

"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from types import SimpleNamespace
from munch import Munch
from loguru import logger

import bittensor
from bittensor.routers.pkm import PKMRouter
from bittensor.utils.batch_transforms import Normalize

class FFNNSynapse(bittensor.synapse.Synapse):
    """ Simple feed forward NN for images.
    """

    def __init__(self, config: Munch):
        r""" Init a new ffnn synapse module.

            Args:
                config (:obj:`munch.Munch`, `required`): 
                    munch namespace config item.
        """
        super(FFNNSynapse, self).__init__(config = config)
        if config == None:
            config = FFNNSynapse.build_config()
            
        # transform_layer: transforms images to common dimension.
        # [batch_size, -1, -1, -1] -> [batch_size, self.transform_dim]
        self.transform = Normalize((0.1307,), (0.3081,),  device=self.device)
        self.transform_pool = nn.AdaptiveAvgPool2d((28, 28))
        self.transform_conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.transform_conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.transform_drop = nn.Dropout2d()
        self.transform_dim = 320

        # context_layer: distills the remote_context from the transform layer.
        # [batch_size, transform_dim] -> [batch_size, bittensor.__network_dim__]
        self.context_layer1 = nn.Linear(self.transform_dim, 256)
        self.context_layer2 = nn.Linear(256, bittensor.__network_dim__)

        # hidden_layer: learns hidden units for network and target.
        # [batch_size, transform_dim + bittensor.__network_dim__] = [batch_size, bittensor.__network_dim__]
        self.hidden_layer1 = nn.Linear(self.transform_dim + bittensor.__network_dim__, bittensor.__network_dim__)
        self.hidden_layer2 = nn.Linear(bittensor.__network_dim__, bittensor.__network_dim__)

        # dendrite: (PKM layer) queries network using pooled embeddings as context.
        # [batch_size, -1] -> topk * [batch_size, bittensor.__network_dim__]
        self.router = PKMRouter(config, query_dim = bittensor.__network_dim__)

        # target_layer: Maps from hidden layer to target dimension
        # [batch_size, bittensor.__network_dim__] -> [batch_size, self.target_dim]
        self.target_layer1 = nn.Linear(bittensor.__network_dim__, 256)
        self.target_layer2 = nn.Linear(256, self.config.synapse.target_dim)

        self.to(self.device)

    @staticmethod   
    def build_config() -> Munch:
        parser = argparse.ArgumentParser(); 
        FFNNSynapse.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        FFNNSynapse.check_config(config)
        return config

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):    
        parser.add_argument('--synapse.target_dim', default=10, type=int, 
                            help='Final logit layer dimension. i.e. 10 for MNIST.')
        parser = PKMRouter.add_args(parser)

    @staticmethod   
    def check_config(config: Munch):
        assert config.synapse.target_dim > 0, "target dimension must be greater than 0."
        config = PKMRouter.check_config(config)

    def forward_image(self, images: torch.Tensor):
        r""" Forward image inputs through the FFNN synapse .

            Args:
                inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, channels, rows, cols)`, `required`): 
                    Image tensors produced by calling PIL.toTensor() and with sequence dimension.
            
            Returns:
                hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, bittensor.__network_dim__)`, `required`): 
                    Hidden layer encoding produced by using local_context.
        """
        # images: remove sequence dimension from images.
        # images.shape = [batch_size, channels, rows, cols] 
        images = images.view(images.shape[0] * images.shape[1], images.shape[2], images.shape[3], images.shape[4]).to(self.device)

        # hidden: hidden layer using local_contextcontext for local computation only.
        # hidden.shape = [batch_size, __network_dim__] 
        hidden = self.local_forward ( images = images ).local_hidden
        
        # hidden: re-add sequence dimension to outputs.
        # hidden.shape = [batch_size, sequence_dim, __network_dim__] 
        hidden = torch.unsqueeze(hidden, 1)

        return hidden

    def local_forward(self, images: torch.Tensor, targets: torch.Tensor = None) -> SimpleNamespace:
        r""" Forward pass non-sequential image inputs and targets through the FFNN Synapse. The call does not make 
        remote queries to the network and returns only local hidden, target and losses.

            Args:
                images (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, channels, rows, cols)`, `required`): 
                    PIL.toTensor() encoded images.

                targets (:obj:`torch.FloatTensor`  of shape :obj:`(batch_size, target_dim)`, `optional`, defaults to None): 
                    Image labels.

            Returns:
                SimpleNamespace ( 
                    local_context (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, bittensor.__network_dim__)`, `required`):
                        Pre-Hidden layer context, trained to match the remote context.

                    local_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, bittensor.__network_dim__)`, `required`):
                        Hidden layer produced from the context.

                    local_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_dim)`, `optional`):
                        FFNN Target predictions using local_context. 

                    local_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        FFNN Classification loss using local_context.

                    local_accuracy (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        Accuracy of target predictions.
                )
        """

        # Return vars to be filled.
        output = SimpleNamespace ()

        # transform: transform images to common shape.
        # transform.shape = [batch_size, self.transform_dim]
        transform = self.transform( images ).to(self.device)
        transform = F.relu(F.max_pool2d( self.transform_conv1(transform), 2) )
        transform = F.relu(F.max_pool2d( self.transform_drop(self.transform_conv2(transform)),2) )
        output.transform = transform.view(-1, self.transform_dim)

        # local_context: distillation model for remote_context.
        # local_context.shape = [batch_size, bittensor.__network_dim__]
        local_context = self.context_layer1( output.transform.detach() )
        output.local_context = self.context_layer2( local_context )

        # local_hidden: hidden layer encoding using local_context.
        # local_hidden.shape = [batch_size, bittensor.__network_dim__]
        local_hidden = torch.cat( ( output.transform, output.local_context.detach() ) , dim=1)
        local_hidden = F.relu( self.hidden_layer1(local_hidden) )
        output.local_hidden  = F.relu( self.hidden_layer2(local_hidden) )

        if targets is not None:
            # local_target: projection of local_hidden onto target dimension.
            # local_target.shape = [batch_size, target_dim]
            targets.to( self.device )
            local_target = self.target_layer1( output.local_hidden )
            local_target = self.target_layer2( local_target )
            output.local_target = F.log_softmax( local_target, dim=1 )

            # local_target_loss: loss between local_target and passed targets.
            # local_target_loss.shape = [1]
            output.local_target_loss = F.nll_loss( output.local_target, targets )

            # Record extra metadata accuracy.
            max_logit = local_target.data.max( 1, keepdim=True )[1]
            correct = max_logit.eq( targets.data.view_as( max_logit ) ).sum()
            output.local_accuracy = (100.0 * correct) / targets.shape[0] 

        return output

    def remote_forward(self, neuron: bittensor.neuron.Neuron, images: torch.Tensor, targets: torch.Tensor = None) -> SimpleNamespace:
        """
            Forward pass non-sequential image inputs and targets through the remote context of the synapse. The call
            makes RPC queries accross the network using the passed neuron's metagraph and dendrite.
            
            Args:
                neuron (:obj: `bittensor.neuron.Neuron`, `required`):
                    Bittensor neuron, used for making queries to the remote network.

                images (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, channels, rows, cols)`, `required`): 
                    PIL.toTensor() encoded images.
                                
                targets (:obj:`torch.FloatTensor`  of shape :obj:`(batch_size, target_dim)`, `optional`, defaults to None): 
                    Image labels.
            
            Returns:
                self.local_forward() + SimpleNamespace ( 

                    router (:obj:`SimpleNamespace`, `required`): 
                        Outputs from the pkm dendrite remote call.

                    distillation_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        Distillation loss between the local and remote context.

                    remote_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, bittensor.__network_dim__)`, `optional`): 
                        Hidden layer encoding produced using the remote context.

                    remote_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_dim)`, `optional`):
                        FFNN Target predictions using the remote_context.

                    remote_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`):
                        FFNN Classification loss using the remote_context.
                )
        """
        # Call the local forward pass.
        # output = bittensor.SynapseOutput
        output = self.local_forward( images, targets ) 

        # Make remote queries using the PKMRouter.
        # remote_context: responses from a bittensor remote network call.
        # remote_context.shape = [batch_size, bittensor.__network_dim__]
        images = torch.unsqueeze(images, 1)
        output.router = self.router.forward_image( neuron, images, output.local_hidden )
        remote_context = torch.squeeze( output.router.response, 1 ).to(self.device)

        # Distill the local context to match the remote context.
        # distillation_loss: distillation loss between local_context and remote_context
        # distillation_loss.shape = [1]
        output.distillation_loss = F.mse_loss(output.local_context, remote_context.detach() )

        # remote_hidden: hidden layer encoding using remote_context.
        # remote_hidden.shape = [batch_size, bittensor.__network_dim__]
        remote_hidden = torch.cat( [ output.transform, remote_context ], dim=1)
        remote_hidden = self.hidden_layer1(remote_hidden)
        output.remote_hidden = self.hidden_layer2(remote_hidden)

        if targets is not None:
            # Project hidden units onto the targets.
            # remote_target: projection of remote_hidden onto target dimension.
            # remote_target.shape = [batch_size, target_dim]
            remote_target = self.target_layer1(remote_hidden)
            remote_target = self.target_layer2(remote_target)
            output.remote_target = F.log_softmax(remote_target, dim=1)

            # Compute the target loss.
            # remote_target_loss: loss between remote_target and passed targets.
            # remote_target_loss.shape = [1]
            output.remote_target_loss = F.nll_loss(output.remote_target, targets)

            # Add extra metrics
            # Record extra metadata accuracy.
            max_logit = output.remote_target.data.max(1, keepdim=True)[1]
            correct = max_logit.eq( targets.data.view_as(max_logit) ).sum()
            output.remote_accuracy = (100.0 * correct) / targets.shape[0] 
        
        return output
