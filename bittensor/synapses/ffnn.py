"""Feed Forward NN Synapse

Simple feed forward NN for images.

"""

import bittensor
from bittensor.utils.router import Router
from bittensor.synapse import Synapse
from bittensor.synapse import SynapseConfig
from bittensor.synapse import SynapseOutput
from bittensor.session import BTSession


from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from typing import List, Tuple, Dict, Optional
from types import SimpleNamespace

class FFNNConfig (SynapseConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~FFNNSynapse`.
    It is used to instantiate a Feed Forward model according to the specified arguments, 
    defining the model architecture. 

    Args:
        target_dim (:obj:`int`, `required`, defaults to (10)):
            The number of logit heads used by the target layer.      

    Examples::

        >>> from bittensor.synapses.ffnn import FNNSynapse, FFNNConfig

        >>> # Initializing a FFNN configuration
        >>> configuration = FFNNConfig()

        >>> # Initializing the synapse from configuration.
        >>> configuration = FNNSynapse ( configuration )
    """

    __default_target_dim__ = 10
    
    def __init__(self, **kwargs):
        super(FFNNConfig, self).__init__(**kwargs)
        self.target_dim = kwargs.pop("target_dim", self.__default_target_dim__)
        self.run_checks()
    
    def run_checks(self):
        assert isinstance(self.target_dim, int)

class FFNNSynapse(Synapse):
    """ Simple feed forward NN for images.
    """

    def __init__(self,
                 config: FFNNConfig,
                 session: BTSession):
        r""" Init a new ffnn synapse module.

            Args:
                config (:obj:`bittensor.ffnn.FFNNConfig`, `required`): 
                    ffnn configuration class.

                session (:obj:`bittensor.Session`, `required`): 
                    bittensor session object. 
        """
        super(FFNNSynapse, self).__init__(
            config = config,
            session = session)
            
        # transform_layer: transforms images to common dimension.
        # [batch_size, -1, -1, -1] -> [batch_size, self.transform_dim]
        self.transform = bittensor.utils.batch_transforms.Normalize((0.1307,), (0.3081,))
        self.transform_pool = nn.AdaptiveAvgPool2d((28, 28))
        self.transform_conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.transform_conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.transform_drop = nn.Dropout2d()
        self.transform_dim = 320

        # router: (PKM layer) queries network using transform as context.
        # [batch_size, transform_dim] -> topk * [batch_size, bittensor.__network_dim__]
        self.router = Router(x_dim = self.transform_dim, key_dim=100, topk=10)

        # context_layer: distills the remote_context from the transform layer.
        # [batch_size, transform_dim] -> [batch_size, bittensor.__network_dim__]
        self.context_layer1 = nn.Linear(self.transform_dim, 256)
        self.context_layer2 = nn.Linear(256, bittensor.__network_dim__)

        # hidden_layer: learns hidden units for network and target.
        # [batch_size, transform_dim + bittensor.__network_dim__] = [batch_size, bittensor.__network_dim__]
        self.hidden_layer1 = nn.Linear(self.transform_dim + bittensor.__network_dim__, bittensor.__network_dim__)
        self.hidden_layer2 = nn.Linear(bittensor.__network_dim__, bittensor.__network_dim__)

        # target_layer: Maps from hidden layer to target dimension
        # [batch_size, bittensor.__network_dim__] -> [batch_size, self.target_dim]
        self.target_layer1 = nn.Linear(bittensor.__network_dim__, 256)
        self.target_layer2 = nn.Linear(256, self.config.target_dim)
        
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
        images = images.view(images.shape[0] * images.shape[1], images.shape[2], images.shape[3], images.shape[4])

        # hidden: hidden layer using local_contextcontext for local computation only.
        # hidden.shape = [batch_size, __network_dim__] 
        hidden = self.forward (images = images.to(self.device), remote = False).local_hidden
        
        # hidden: re-add sequence dimension to outputs.
        # hidden.shape = [batch_size, sequence_dim, __network_dim__] 
        hidden = torch.unsqueeze(hidden, 1)

        return hidden

    def forward(self,
                images: torch.Tensor,
                targets: torch.Tensor = None,
                remote: bool = False):
        r""" Forward pass non-sequential image inputs and targets through the FFNN Synapse.

            Args:
                images (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, channels, rows, cols)`, `required`): 
                    PIL.toTensor() encoded images.

                targets (:obj:`torch.FloatTensor`  of shape :obj:`(batch_size, target_dim)`, `optional`, defaults to None): 
                    Image labels.

                remote (:obj:`bool')`, `optional`, default to False):
                    Switch between local_contextand remote context. If true, function makes quries to the remote network.

            Returns:
                bittensor.SynapseOutput ( 
                    loss  (:obj:`List[str]` of shape :obj:`(batch_size)`, `required`):
                        Total loss acumulation to be used by loss.backward()

                    local_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, bittensor.__network_dim__)`, `required`):
                        Hidden layer encoding produced using local_context.

                    local_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_dim)`, `optional`):
                        FFNN Target predictions using student_context. 

                    local_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        FFNN Classification loss using student_context.

                    remote_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, bittensor.__network_dim__)`, `optional`): 
                        Hidden layer encoding produced using the remote_context.

                    remote_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_dim)`, `optional`):
                        FFNN Target predictions using the remote_context.

                    remote_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`):
                        FFNN Classification loss using the remote_context.

                    distillation_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        Distillation loss between local_context and remote_context.
                )
        """

        # Return vars to be filled.
        output = SynapseOutput (loss = torch.tensor(0.0))

        # transform: transform images to common shape.
        # transform.shape = [batch_size, self.transform_dim]
        transform = self.transform(images)
        transform = F.relu(F.max_pool2d(self.transform_conv1(transform), 2))
        transform = F.relu(F.max_pool2d(self.transform_drop(self.transform_conv2(transform)),2))
        transform = transform.view(-1, self.transform_dim)
       
        # remote_context: responses from a bittensor remote network call.
        # remote_context.shape = [batch_size, bittensor.__network_dim__]
        if remote:
            # If query == True make a remote call.
            images = torch.unsqueeze(images, 1) # Add sequence dimension.
            synapses = self.session.metagraph.synapses() # Returns a list of synapses on the network.
            requests, _ = self.router.route( synapses, transform, images ) # routes inputs to network.
            responses = self.session.dendrite.forward_image( synapses, requests ) # Makes network calls.
            remote_context = self.router.join( responses ) # Joins responses based on scores..
            remote_context = remote_context.view(remote_context.shape[0] * remote_context.shape[1], remote_context.shape[2]) # Squeeze the sequence dimension.

        # local_context: distillation model for remote_context.
        # local_context.shape = [batch_size, bittensor.__network_dim__]
        local_context = self.context_layer1(transform.detach())
        local_context = self.context_layer2(local_context)
        if remote:
            # distillation_loss: distillation loss between local_context and remote_context
            # distillation_loss.shape = [1]
            distillation_loss = F.mse_loss(local_context, remote_context.detach())
            output.distillation_loss = distillation_loss
            output.loss = output.loss + distillation_loss

        # local_hidden: hidden layer encoding using local_context.
        # local_hidden.shape = [batch_size, bittensor.__network_dim__]
        local_hidden = torch.cat((transform, local_context.detach()), dim=1)
        local_hidden = F.relu(self.hidden_layer1(local_hidden))
        local_hidden = F.relu(self.hidden_layer2(local_hidden))
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
            output.remote_hidden = remote_hidden
        
        if remote and targets is not None:
            # remote_target: projection of remote_hidden onto target dimension.
            # remote_target.shape = [batch_size, target_dim]
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