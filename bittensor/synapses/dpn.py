"""
    DPN synapse

    Bittensor endpoint trained on PIL images to detect objects using DPN.
"""

import bittensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

class DPNConfig (bittensor.SynapseConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~DPNSynapse`.
    It is used to instantiate a Dual Path model according to the specified
    arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a 
    "shallow" DPN-26 configuration. 

    For deeper network configurations, it is possible to set the num_blocks parameter to (3, 4, 20, 3) for a
    DPN-92. 
    
    For DPN-98 set the following:
    in_planes: (160, 320, 640, 1280)
    out_planes: (256, 512, 1024, 2048)
    num_blocks: (3, 6, 20, 3)
    dense_depth: (16, 32, 32, 128)


    Args:
        target_size (:obj:`int`, `required`, defaults to (10)):
            The number of logit heads used by the target layer.      
        in_planes (:obj:`tuple`, `required`, defaults to (96,192,384,768)):
            The inputs of convolutional layers 2, 3, 4, and 5. 
        out_planes (:obj:`tuple`, `required`, defaults to (256,512,1024,2048)):
            Output planes of convolutional layers 2, 3, 4, and 5.
        num_blocks (:obj:`tuple`, `required`, defaults to (2,2,2,2)):
            How many blocks of layers to create for layers 2, 3, 4, and 5
        dense_depth (:obj:`tuple`, `required`, defaults to (16,32,24,128):
            Width increment of the densely connected path.
      

    Examples::

        >>> from bittensor.synapses.dpn import DPNConfig, DPNSynapse

        >>> # Initializing a DPN configuration
        >>> configuration = DPNConfig()

        >>> # Initializing a DNP Synapse
        >>> model = DPNSynapse ( configuration )
    """

    __default_target_size__ = 10
    __default_in_planes__ = (96,192,384,768)
    __default_out_planes__ = (256,512,1024,2048)
    __default_block_config__ = (2,2,2,2)
    __default_dense_depth__ = (16,32,24,128)
    
    def __init__(self, **kwargs):
        super(DPNConfig, self).__init__(**kwargs)
        self.target_size = kwargs.pop("target_size",
                                         self.__default_target_size__)
        self.in_planes = kwargs.pop("in_planes",
                                         self.__default_in_planes__)
        self.out_planes = kwargs.pop("out_planes",
                                         self.__default_out_planes__)
        self.block_config = kwargs.pop("block_config",
                                         self.__default_block_config__)
        self.dense_depth = kwargs.pop("dense_depth",
                                         self.__default_dense_depth__)

        self.run_type_checks()
    
    def run_type_checks(self):
        assert isinstance(self.target_size, int)
        assert isinstance(self.in_planes, tuple)
        assert isinstance(self.out_planes, tuple)
        assert isinstance(self.block_config, tuple)
        assert isinstance(self.dense_depth, tuple)
    

class DPNSynapse(bittensor.Synapse):
    """ Bittensor endpoint trained on PIL images to detect objects using DPN.
    """

    def __init__(   self, 
                    config: DPNConfig,
                    dendrite: bittensor.Dendrite = None,
                    metagraph: bittensor.Metagraph = None
                ):
        r""" Init a new DPN synapse module.

            Args:
                config (:obj: `bittensor.dpn.DPNConfig`, `required`)
                    Model configuration object used to set up what the model should 
                    contain in terms of convolutional and dense layers. See :class: bittensor.dpn.DPNConfig

                dendrite (:obj:`bittensor.Dendrite`, `optional`, defaults to bittensor.dendrite): 
                    bittensor dendrite object used for queries to remote synapses.
                    Defaults to bittensor.dendrite global.

                metagraph (:obj:`bittensor.Metagraph`, `optional`, defaults to bittensor.metagraph): 
                    bittensor metagraph containing network graph information. 
                    Defaults to bittensor.metagraph global.

        """
        super(DPNSynapse, self).__init__(
            config = config,
            dendrite = dendrite,
            metagraph = metagraph)

        in_planes, out_planes = config.in_planes, config.out_planes
        num_blocks, dense_depth = config.block_config, config.dense_depth

        # Transform Network
        """ Transform network.
                Layers take in PIL input (image in this case), normalize it and then apply 
                4 convolutional layers. 
            Image encoder: transforms PIL-encoded tensors to a common shape.
            [batch_size, channels, rows, cols] -> [batch_size, -1, -1, -1] 

            Output: [batch_size, self.transform_dim (9728)]
        """
        self.transform = bittensor.utils.batch_transforms.Normalize((0.1307,), (0.3081,), device=self.device)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((32, 32))
        self.transform_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.transform_bn1 = nn.BatchNorm2d(64)
        self.last_planes = 64
        self.transform_layer1 = self._make_layer(in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], stride=1)
        self.transform_layer2 = self._make_layer(in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], stride=2)
        self.transform_layer3 = self._make_layer(in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], stride=1)
        self.transform_layer4 = self._make_layer(in_planes[3], out_planes[3], num_blocks[3], dense_depth[3], stride=2)
        self.transform_dim = (out_planes[3] * 4)+(((num_blocks[3]+1) * 4)*dense_depth[3])
        
        # Router object for training network connectivity.
        # [Transform] -> [ROUTER] -> [Synapses] -> [ROUTER]
        self.router = bittensor.Router(x_dim = self.transform_dim , key_dim = 100, topk = 10)

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
        self.target_layer2 = nn.Linear(128, config.target_size)
    
    def forward_image (     self,  
                            images: torch.Tensor):
        r""" Forward image inputs through the DPN synapse .

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
    
    def forward (   self,
                    images: torch.Tensor,
                    targets: torch.Tensor = None,
                    remote: bool = False):
        r""" Forward pass non-sequential image inputs and targets through the DPN Synapse.

            Args:
                images (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, channels, rows, cols)`, `required`): 
                    PIL.toTensor() encoded images.

                targets (:obj:`torch.FloatTensor`  of shape :obj:`(batch_size, config.target_size)`, `optional`): 
                    Image labels.

                remote (:obj:`bool')`, `optional`):
                    Switch between local and remote context. If true, function makes quries to the remote network.

            Returns:
                bittensor.SynapseOutput  (
                    loss  (:obj:`List[str]` of shape :obj:`(batch_size)`, `required`):
                        Total loss acumulation to be used by loss.backward()

                    local_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, bittensor.__network_dim__)`, `required`):
                        Hidden layer encoding produced by using local_context.

                    local_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.target_size)`, `optional`):
                        DPN Target predictions using local_context. 

                    local_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        DPN Classification loss using local_context.

                    remote_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, bittensor.__network_dim__)`, `optional`): 
                        Hidden layer encoding produced using the remote_context.

                    remote_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.target_size)`, `optional`):
                        DPN Target predictions using the remote_context.

                    remote_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`):
                        DPN Classification loss using the remote_context.

                    distillation_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        Distillation loss between local_context and remote_context.
                )
        """
        # Return vars to be filled.
        output = bittensor.SynapseOutput (loss = torch.tensor(0.0))
    
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
        transform = torch.flatten(transform, start_dim=1)

        # remote_context: responses from a bittensor remote network call.
        # remote_context.shape = [batch_size, bittensor.__network_dim__]
        if remote:
            # If query == True make a remote call.
            images = torch.unsqueeze(images, 1) # Add sequence dimension.
            synapses = self.metagraph.synapses() # Returns a list of synapses on the network.
            requests, _ = self.router.route( synapses, transform, images ) # routes inputs to network.
            responses = self.dendrite.forward_image( synapses, requests ) # Makes network calls.
            remote_context = self.router.join( responses ) # Joins responses based on scores..
            remote_context = remote_context.view(remote_context.shape[0] * remote_context.shape[1], remote_context.shape[2]) # Squeeze the sequence dimension.

        # local_context: distillation model for remote_context.
        # local_context.shape = [batch_size, bittensor.__network_dim__]
        local_context = self.context_layer1(transform.detach())
        local_context = self.context_layer2(local_context)
        local_context = self.context_layer3(local_context)
        
        if remote:
            # distillation_loss: distillation loss between local_context and remote_context
            # distillation_loss.shape = [1]
            distillation_loss = F.mse_loss(local_context, remote_context.detach())
            output.distillation_loss = distillation_loss
            output.loss = output.loss + distillation_loss

        # local_hidden: hidden layer encoding using local_context.
        # local_hidden.shape = [batch_size, bittensor.__network_dim__]
        local_hidden = torch.cat([transform, local_context], dim=1)
        local_hidden = self.hidden_layer1(local_hidden)
        local_hidden = self.hidden_layer2(local_hidden)
        local_hidden = self.hidden_layer3(local_hidden)
        output.local_hidden = local_hidden
        if targets is not None:
            # local_target: projection of local_hidden onto target dimension.
            # local_target.shape = [batch_size, target_dim]
            targets.to(self.device)
            local_target = self.target_layer1(local_hidden)
            local_target = self.target_layer2(local_target)
            local_target = F.log_softmax(local_target, dim=1)
            output.local_target = local_target

            # local_target_loss: loss between local_target and passed targets.
            # local_target_loss.shape = [1]
            local_target_loss = F.nll_loss(local_target, targets)
            output.local_target_loss = local_target_loss
            output.loss = output.loss + local_target_loss

        
        # remote_hidden: hidden layer encoding using remote_context.
        # remote_hidden.shape = [batch_size, bittensor.__network_dim__]
        if remote:
            remote_hidden = torch.cat([transform, remote_context], dim=1)
            remote_hidden = self.hidden_layer1(remote_hidden)
            remote_hidden = self.hidden_layer2(remote_hidden)
            remote_hidden = self.hidden_layer3(remote_hidden)
            output.remote_hidden = remote_hidden
        
        if remote and targets is not None:
            # remote_target: projection of remote_hidden onto target dimension.
            # remote_target.shape = [batch_size, config.target_size]
            remote_target = self.target_layer1(remote_hidden)
            remote_target = self.target_layer2(remote_target)
            remote_target = F.log_softmax(remote_target, dim=1)
            output.remote_target = remote_target

            # remote_target_loss: loss between remote_target and passed targets.
            # remote_target_loss.shape = [1]
            remote_target_loss = F.nll_loss(remote_target, targets)
            output.loss = output.loss + remote_target_loss
            output.remote_target_loss = remote_target_loss


        return output

    def _make_layer(self, in_planes, out_planes, num_blocks, dense_depth, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i,stride in enumerate(strides):
            layers.append(self.Bottleneck(self.last_planes, in_planes, out_planes, dense_depth, stride, i==0))
            self.last_planes = out_planes + (i+2) * dense_depth
        return nn.Sequential(*layers)
    
    class Bottleneck(nn.Module):
        def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first_layer):
            super(DPNSynapse.Bottleneck, self).__init__()
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
