"""
    DPN synapse

    Bittensor endpoint trained on PIL images to detect objects using DPN.
"""

import bittensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from bittensor.synapses.dpn.config import DPNConfig

class DPNSynapse(bittensor.Synapse):
    """ Bittensor endpoint trained on PIL images to detect objects using DPN.
    """

    def __init__(   self, 
                    config: DPNConfig = None,
                    dendrite: bittensor.Dendrite = None,
                    metagraph: bittensor.Metagraph = None
                ):
        r""" Init a new DPN synapse module.

            Args:
                config (:obj: `bittensor.dpn.dpn_configuration.DPNConfig`, `required`)
                    Model configuration object used to set up what the model should 
                    contain in terms of convolutional and dense layers. See :class: bittensor.dpn.dpn_configuration.DPNConfig

                dendrite (:obj:`bittensor.Dendrite`, `optional`): 
                    bittensor dendrite object used for queries to remote synapses.
                    Defaults to bittensor.dendrite global.

                metagraph (:obj:`bittensor.Metagraph`, `optional`): 
                    bittensor metagraph containing network graph information. 
                    Defaults to bittensor.metagraph global.

        """
        super(DPNSynapse, self).__init__()
        
        # Bittensor dendrite object used for queries to remote synapses.
        # Defaults to bittensor.dendrite global object.
        self.dendrite = dendrite
        if self.dendrite == None:
            self.dendrite = bittensor.dendrite

        # Bttensor metagraph containing network graph information. 
        # Defaults to bittensor.metagraph global object.
        self.metagraph = metagraph
        if self.metagraph == None:
            self.metagraph = bittensor.metagraph
        
        self.config = config
        if self.config == None:
            self.config = DPNConfig()

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

        # Send model to appropriate device (CPU or CUDA)
        self.to(self.device)
    
    def forward_image (     self,  
                            images: torch.Tensor):
        r""" Forward image inputs through the DPN synapse .

            Args:
                inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, channels, rows, cols)`, `required`): 
                    Image tensors produced by calling PIL.toTensor() and with sequence dimension.
            
            Returns:
                hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, bittensor.__network_dim__)`, `required`): 
                    Hidden layer encoding produced by using student_context.
        """
        # images: remove sequence dimension from images.
        # images.shape = [batch_size, channels, rows, cols] 
        images = images.view(images.shape[0] * images.shape[1], images.shape[2], images.shape[3], images.shape[4])

        # hidden: hidden layer using student context for local computation only.
        # hidden.shape = [batch_size, __network_dim__] 
        hidden = self.forward (images = images.to(self.device), remote = False) ['local_hidden']
        
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

                targets (:obj:`torch.FloatTensor`  of shape :obj:`(batch_size, 10)`, `optional`): 
                    Image labels.

                remote (:obj:`bool')`, `optional`):
                    Switch between student and remote context. If true, function makes quries to the remote network.

            Returns:
                dictionary with { 
                    loss  (:obj:`List[str]` of shape :obj:`(batch_size)`, `required`):
                        Total loss acumulation to be used by loss.backward()

                    local_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, bittensor.__network_dim__)`, `required`):
                        Hidden layer encoding produced by using student_context.

                    local_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 10)`, `optional`):
                        DPN Target predictions using student_context. 

                    local_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        DPN Classification loss using student_context.

                    remote_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, bittensor.__network_dim__)`, `optional`): 
                        Hidden layer encoding produced using the remote_context.

                    remote_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 10)`, `optional`):
                        DPN Target predictions using the remote_context.

                    remote_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`):
                        DPN Classification loss using the remote_context.

                    distillation_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        Distillation loss between student_context and remote_context.
                }
        """
        # Return vars to be filled.
        loss = torch.tensor(0.0)
        local_hidden = None
        local_target = None
        local_target_loss = None
        remote_hidden = None
        remote_target = None
        remote_target_loss = None
        distillation_loss = None
        remote_context = None
        

        r"""
            Transform the images into a common shape (32x32) in this case
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

        # student_context: distillation model for remote_context.
        # student_context.shape = [batch_size, bittensor.__network_dim__]
        local_context = self.context_layer1(transform.detach())
        local_context = self.context_layer2(local_context)
        local_context = self.context_layer3(local_context)
        
        if remote:
            # distillation_loss: distillation loss between student_context and remote_context
            # distillation_loss.shape = [1]
            distillation_loss = F.mse_loss(local_context, remote_context.detach())
            loss = loss + distillation_loss

        # local_hidden: hidden layer encoding using student_context.
        # local_hidden.shape = [batch_size, bittensor.__network_dim__]
        local_hidden = torch.cat([transform, local_context], dim=1)
        local_hidden = self.hidden_layer1(local_hidden)
        local_hidden = self.hidden_layer2(local_hidden)
        local_hidden = self.hidden_layer3(local_hidden)

        if targets is not None:
            # local_target: projection of local_hidden onto target dimension.
            # local_target_loss: loss between local_target and passed targets.
            # local_target.shape = [batch_size, target_dim]
            # local_target_loss.shape = [1]
            targets.to(self.device)
            local_target = self.target_layer1(local_hidden)
            local_target = self.target_layer2(local_target)
            local_target = F.log_softmax(local_target, dim=1)
            local_target_loss = F.nll_loss(local_target, targets)
            loss = loss + local_target_loss
        
        # remote_hidden: hidden layer encoding using remote_context.
        # remote_hidden.shape = [batch_size, bittensor.__network_dim__]
        if remote:
            remote_hidden = torch.cat([transform, remote_context], dim=1)
            remote_hidden = self.hidden_layer1(remote_hidden)
            remote_hidden = self.hidden_layer2(remote_hidden)
            remote_hidden = self.hidden_layer3(remote_hidden)
        
        if remote and targets is not None:
            # remote_target: projection of remote_hidden onto target dimension.
            # remote_target_loss: loss between remote_target and passed targets.
            # remote_target.shape = [batch_size, 10]
            # remote_target_loss.shape = [1]
            remote_target = self.target_layer1(remote_hidden)
            remote_target = self.target_layer2(remote_target)
            remote_target = F.log_softmax(remote_target, dim=1)
            remote_target_loss = F.nll_loss(remote_target, targets)
            loss = loss + remote_target_loss

        return {
            'loss': loss,
            'local_hidden': local_hidden,
            'local_target': local_target,
            'local_target_loss': local_target_loss,
            'remote_hidden': remote_hidden,
            'remote_target': remote_target,
            'remote_target_loss': remote_target_loss,
            'distillation_loss': distillation_loss,
        }

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