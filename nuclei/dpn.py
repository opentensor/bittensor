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

"""
    DPN nucleus

    Bittensor endpoint trained on PIL images to detect objects using DPN.
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from types import SimpleNamespace

import bittensor
from collections.abc import Callable
from bittensor.utils.batch_transforms import Normalize

class DPNNucleus(torch.nn.Module):
    """ Bittensor endpoint trained on PIL images to detect objects using an DPN.
    """

    def __init__( self, routing_callback, config: 'bittensor.Config' = None, **kwargs):
        r""" Init a new DPN nucleus module.

            Args:
                config (:obj: `bittensor.Config`, `required`)
                    munch namespace config item.
        """
        super(DPNNucleus, self).__init__()
        if config == None:
            config = DPNNucleus.config()
        DPNNucleus.check_config(config)
        self.config = config

        # To be set.
        self.routing_callback = routing_callback
        
        in_planes, out_planes = config.nucleus.in_planes, config.nucleus.out_planes
        num_blocks, dense_depth = config.nucleus.num_blocks, config.nucleus.dense_depth

        # Transform Network
        """ Transform network.
                Layers take in image inputs normalizes them and applies 
                4 convolutional layers. 
            Image encoder: transforms PIL-encoded tensors to a common shape.
            [batch_size, channels, rows, cols] -> [batch_size, -1, -1, -1] 

            Output: [batch_size, self.transform_dim (9728)]
        """
        self.transform = Normalize((0.1307,), (0.3081,), device=self.device)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((32, 32))
        self.transform_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.transform_bn1 = nn.BatchNorm2d(64)
        self.last_planes = 64
        self.transform_layer1 = self._make_layer(in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], stride=1)
        self.transform_layer2 = self._make_layer(in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], stride=2)
        self.transform_layer3 = self._make_layer(in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], stride=1)
        self.transform_layer4 = self._make_layer(in_planes[3], out_planes[3], num_blocks[3], dense_depth[3], stride=2)
        self.transform_dim = (out_planes[3] * 4)+(((num_blocks[3]+1) * 4)*dense_depth[3])
        
        # Context layers.
        """
            Distillation model for remote context. This layer takes input 
            coming from transform layer, and runs it through 3 linear layers,
            projecting it to bittensor.__network_dim__.  
        """
        self.context_layer1 = nn.Linear(self.transform_dim, 512)
        self.context_layer2 = nn.Linear(512, 256)
        self.context_layer3 = nn.Linear(256, bittensor.__network_dim__)

        # hidden layer.
        self.hidden_layer1 = nn.Linear(self.transform_dim + bittensor.__network_dim__, 512)
        self.hidden_layer2 = nn.Linear(512, 256)
        self.hidden_layer3 = nn.Linear(256, bittensor.__network_dim__)

        # Layers to project target down to target size passed by config
        # (number of classes)
        self.target_layer1 = nn.Linear(bittensor.__network_dim__, 128)
        self.target_layer2 = nn.Linear(128, self.config.nucleus.target_dim)

        self.to(self.device)

    @staticmethod   
    def config() -> 'bittensor.Config':
        parser = argparse.ArgumentParser(); 
        DPNNucleus.add_args(parser) 
        config = bittensor.config( parser ); 
        return config

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        r""" This function adds the configuration items for the DPN nucleus.
        These args are use to instantiate a Dual Path model. 
        Instantiating a configuration with the defaults will yield a "shallow" DPN-26 configuration. 

        For deeper network configurations, it is possible to set the num_blocks parameter to (3, 4, 20, 3) for a
        DPN-92. 
        
        For DPN-98 set the following:
            in_planes: (160, 320, 640, 1280)
            out_planes: (256, 512, 1024, 2048)
            num_blocks: (3, 6, 20, 3)
            dense_depth: (16, 32, 32, 128)
        """
        def to_list(arg):
            return [int(i) for i in arg.split(",")]
        parser.add_argument('--nucleus.in_planes', default='160, 320, 640, 1280', action="append", type=to_list)
        parser.add_argument('--nucleus.out_planes', default='256, 512, 1024, 2048', action="append", type=to_list)
        parser.add_argument('--nucleus.num_blocks', default='3, 6, 20, 3', action="append", type=to_list)
        parser.add_argument('--nucleus.dense_depth', default='16, 32, 32, 128', action="append", type=to_list)
        parser.add_argument('--nucleus.target_dim', default=10, type=int, help='Final logit layer dimension. i.e. 10 for CIFAR-10.')
    
    @staticmethod
    def check_config(config: 'bittensor.Config'):
        assert isinstance(config.nucleus.in_planes, list), 'nucleus.in_planes must be a tuple, got {}'.format(config.nucleus.in_planes)
        assert isinstance(config.nucleus.out_planes, list), 'nucleus.out_planes must be a tuple, got {}'.format(config.nucleus.out_planes)
        assert isinstance(config.nucleus.num_blocks, list), 'nucleus.num_blocks must be a tuple, got {}'.format(config.nucleus.num_blocks)
        assert isinstance(config.nucleus.dense_depth, list), 'nucleus.dense_depth must be a tuple, got {}'.format(config.nucleus.dense_depth)
        assert all(isinstance(el, int) for el in config.nucleus.in_planes), 'nucleus.in_planes must be a tuple of ints, got {}'.format(config.nucleus.in_planes)
        assert all(isinstance(el, int) for el in config.nucleus.out_planes), 'nucleus.out_planes must be a tuple of ints, got {}'.format(config.nucleus.out_planes)
        assert all(isinstance(el, int) for el in config.nucleus.num_blocks), 'nucleus.num_blocks must be a tuple of ints, got {}'.format(config.nucleus.num_blocks)
        assert all(isinstance(el, int) for el in config.nucleus.dense_depth), 'nucleus.dense_depth must be a tuple of ints, got {}'.format(config.nucleus.dense_depth)


    def attach_routing_callback(self, routing_callback: Callable[ [torch.Tensor, torch.Tensor], torch.Tensor ] ):
        """ Assigns the routing_callback call to this neuron.

            Returns:
                routing_callback (:callabl:`Callable[ [torch.Tensor, torch.Tensor], torch.Tensor `, `required`): 
                    Routing function to call on self.route()
        """
        self.routing_callback = routing_callback

    @property
    def route( self, inputs: torch.Tensor, query: torch.Tensor ) -> torch.FloatTensor:
        """ Calls this nucleus's subscribed routing function. self.routing_callback must be set before this call is made.

        Args:
            inputs (:obj:`torch.LongTensor` of shape :obj:`( batch_size, sequence_len )`, `required`): 
                    Batch_size length list of tokenized sentences.

            query (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, query_dimension)`, `required`): 
                    Context tensor used to select which neurons to query for each example.
            
        Returns:
            remote_context (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`): 
                Context from calling remote network.
        """
        if self.routing_callback == None:
            raise RuntimeError('The routing function must be set on this nucleus before a remote_forward call can execute.')
        else:
            return self.routing_callback( inputs = inputs, query = query )
    
    def forward_image ( self, images: torch.Tensor):
        r""" Forward image inputs through the DPN nucleus .

            Args:
                inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, channels, rows, cols)`, `required`): 
                    Image tensors produced by calling PIL.toTensor() and with sequence dimension.
            
            Returns:
                hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, bittensor.__network_dim__)`, `required`): 
                    Hidden layer encoding produced by using local_context.
        """
        # images: remove sequence dimension from images.
        # images.shape = [batch_size, channels, rows, cols] 
        images = images.view(images.shape[0] * images.shape[1], images.shape[2], images.shape[3], images.shape[4])

        # hidden: hidden layer using local context for local computation only.
        # hidden.shape = [batch_size, __network_dim__] 
        hidden = self.forward (images = images.to(self.device), remote = False).local_hidden
        
        # hidden: re-add sequence dimension to outputs.
        # hidden.shape = [batch_size, sequence_dim, __network_dim__] 
        hidden = torch.unsqueeze(hidden, 1)

        return hidden

    def local_forward ( self, images: torch.Tensor, targets: torch.Tensor = None ) -> SimpleNamespace:
        r""" Forward pass non-sequential image inputs and targets through the DPN Nucleus.

            Args:
                images (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, channels, rows, cols)`, `required`): 
                    PIL.toTensor() encoded images.

                targets (:obj:`torch.FloatTensor`  of shape :obj:`(batch_size, config.target_size)`, `optional`): 
                    Image labels.

                remote (:obj:`bool')`, `optional`):
                    Switch between local and remote context. If true, function makes quries to the remote network.

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

                    transform (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, transform_dim)`, `optional`): 
                        transformation of various sized images to batch-size transform dim.
                )
        """
        # Return vars to be filled.
        output = SimpleNamespace ()

        r"""
            Transform the images into a common shape (32x32)
        """
        # transform: transform images to common shape.
        # transform.shape = [batch_size, self.transform_dim]
        transform = self.transform(images)
        transform = self.adaptive_pool(transform)
        transform = F.relu(self.transform_bn1(self.transform_conv1(transform.detach())))
        transform = self.transform_layer1(transform)
        transform = self.transform_layer2(transform)
        transform = self.transform_layer3(transform)
        transform = self.transform_layer4(transform)
        transform = F.avg_pool2d(transform, 4)
        output.transform = torch.flatten(transform, start_dim=1)

        # local_context: distillation model for remote_context.
        # local_context.shape = [batch_size, bittensor.__network_dim__]
        local_context = self.context_layer1(output.transform.detach())
        local_context = self.context_layer2(local_context)
        output.local_context = self.context_layer3(local_context)
        
        # local_hidden: hidden layer encoding using local_context.
        # local_hidden.shape = [batch_size, bittensor.__network_dim__]
        local_hidden = torch.cat([output.transform, output.local_context], dim=1)
        local_hidden = self.hidden_layer1(local_hidden)
        local_hidden = self.hidden_layer2(local_hidden)
        output.local_hidden = self.hidden_layer3(local_hidden)
        
        if targets is not None:
            # local_target: projection of local_hidden onto target dimension.
            # local_target.shape = [batch_size, target_dim]
            targets.to(self.device)
            local_target = self.target_layer1(output.local_hidden)
            local_target = self.target_layer2(local_target)
            output.local_target = F.log_softmax(local_target, dim=1)

            # local_target_loss: loss between local_target and passed targets.
            # local_target_loss.shape = [1]
            output.local_target_loss  = F.nll_loss(output.local_target, targets)

            # Record extra metadata accuracy.
            max_logit = local_target.data.max(1, keepdim=True)[1]
            correct = max_logit.eq( targets.data.view_as(max_logit) ).sum()
            output.local_accuracy = (100.0 * correct) / targets.shape[0] 
        
        return output

    def remote_forward(self, images: torch.Tensor, targets: torch.Tensor = None) -> SimpleNamespace:
        """
            Forward pass non-sequential image inputs and targets through the nucleus. Makes RPC queries to downstream neurons.
            
            Args:
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
        # output = bittensor.NucleusOutput
        output = self.local_forward( images, targets ) 

        # Make remote queries using the PKMRouter.
        # remote_context: responses from a bittensor remote network call.
        # remote_context.shape = [batch_size, bittensor.__network_dim__]
        images = torch.unsqueeze(images, 1)
        output.remote_context = self.route( inputs = images, query = output.transform )
        output.remote_context = torch.squeeze( output.remote_context, 1 ).to(self.device)

        # Distill the local context to match the remote context.
        # distillation_loss: distillation loss between local_context and remote_context
        # distillation_loss.shape = [1]
        output.distillation_loss = F.mse_loss(output.local_context, output.remote_context.detach() )

        # remote_hidden: hidden layer encoding using remote_context.
        # remote_hidden.shape = [batch_size, bittensor.__network_dim__]
        remote_hidden = torch.cat([output.transform, output.remote_context], dim=1)
        remote_hidden = self.hidden_layer1(remote_hidden)
        remote_hidden = self.hidden_layer2(remote_hidden)
        output.remote_hidden = self.hidden_layer3(remote_hidden)

        if targets is not None:
            # remote_target: projection of remote_hidden onto target dimension.
            # remote_target.shape = [batch_size, config.target_size]
            remote_target = self.target_layer1(output.remote_hidden)
            remote_target = self.target_layer2(remote_target)
            output.remote_target = F.log_softmax(remote_target, dim=1)

            # remote_target_loss: loss between remote_target and passed targets.
            # remote_target_loss.shape = [1]
            output.remote_target_loss = F.nll_loss(output.remote_target, targets)
        
        return output

    def _make_layer(self, in_planes, out_planes, num_blocks, dense_depth, stride):
        """ Generates a sequential container containing Bottleneck layers.  

        Args:
            in_planes (tuple): 
                4-element tuple describing the in_planes config.

            out_planes (tuple): 
                4-element tuple describing the out_planes config.

            num_blocks (tuple): 
                4-element tuple describing the number of blocks at this layer.

            dense_depth (tuple): 
                4-element tuple describing the depth of this layer.
           
            stride (int): 
                Convolutional stride length.

        Returns:
            nn.Sequential: A torch.nn sequential container containing the layers outlined in the inputs.
        """
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i,stride in enumerate(strides):
            layers.append(self.Bottleneck(self.last_planes, in_planes, out_planes, dense_depth, stride, i==0))
            self.last_planes = out_planes + (i+2) * dense_depth
        return nn.Sequential(*layers)
    
    class Bottleneck(nn.Module):
        def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first_layer):
            super(DPNNucleus.Bottleneck, self).__init__()
            self.out_planes = out_planes
            self.dense_depth = dense_depth

            self.conv1 = nn.Conv2d(last_planes, in_planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=32, bias=False)
            self.bn2 = nn.BatchNorm2d(in_planes)
            self.conv3 = nn.Conv2d(in_planes, out_planes+dense_depth, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(out_planes + dense_depth)

            self.shortcut = nn.Sequential()
            if first_layer:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(last_planes, out_planes + dense_depth, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_planes + dense_depth)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            x = self.shortcut(x)
            d = self.out_planes
            out = torch.cat([x[:,:d,:,:]+out[:,:d,:,:], x[:,d:,:,:], out[:,d:,:,:]], 1)
            out = F.relu(out)
            return out
