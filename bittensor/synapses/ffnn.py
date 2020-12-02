"""Feed Forward NN Synapse

Simple feed forward NN for images.

"""

import bittensor
from bittensor.dendrites.pkm import PKMDendrite
from bittensor.synapse import Synapse
from bittensor.synapse import SynapseOutput
from bittensor.session import BTSession
from bittensor.utils.batch_transforms import Normalize

import argparse
from munch import Munch
from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNNSynapse(Synapse):
    """ Simple feed forward NN for images.
    """

    def __init__(self,
                 config: Munch,
                 session: BTSession):
        r""" Init a new ffnn synapse module.

            Args:
                config (:obj:`munch.Munch`, `required`): 
                    munch namespace config item.

                session (:obj:`bittensor.Session`, `required`): 
                    bittensor session object. 
        """
        super(FFNNSynapse, self).__init__(
            config = config,
            session = session)
            
        # transform_layer: transforms images to common dimension.
        # [batch_size, -1, -1, -1] -> [batch_size, self.transform_dim]
        self.transform = Normalize((0.1307,), (0.3081,),  device=self.device)
        self.transform_pool = nn.AdaptiveAvgPool2d((28, 28))
        self.transform_conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.transform_conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.transform_drop = nn.Dropout2d()
        self.transform_dim = 320

        # dendrite: (PKM layer) queries network using pooled embeddings as context.
        # [batch_size, -1] -> topk * [batch_size, bittensor.__network_dim__]
        self.dendrite = PKMDendrite(config, session, context_dim = self.transform_dim)

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
        self.target_layer2 = nn.Linear(256, self.config.synapse.target_dim)

        self.to(self.device)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:    
        parser.add_argument('--synapse.target_dim', default=10, type=int, 
                            help='Final logit layer dimension. i.e. 10 for MNIST.')
        parser.add_argument('--synapse.n_block_filter', default=100, type=int, 
                            help='Stale neurons are filtered after this many blocks.')
        return parser

    @staticmethod   
    def check_config(config: Munch) -> Munch:
        assert config.synapse.target_dim > 0, "target dimension must be greater than 0."
        return config

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

                    weights (:obj:`torch.LongTensor` of shape :obj:`(batch_size, metagraph.state.n)`, `optional`): 
                        weights for each active neuron.
                )
        """

        # Return vars to be filled.
        output = SynapseOutput (loss = torch.tensor(0.0))

        # transform: transform images to common shape.
        # transform.shape = [batch_size, self.transform_dim]
        transform = self.transform(images).to(self.device)
        transform = F.relu(F.max_pool2d(self.transform_conv1(transform), 2))
        transform = F.relu(F.max_pool2d(self.transform_drop(self.transform_conv2(transform)),2))
        transform = transform.view(-1, self.transform_dim)

        # local_context: distillation model for remote_context.
        # local_context.shape = [batch_size, bittensor.__network_dim__]
        local_context = self.context_layer1(transform.detach())
        local_context = self.context_layer2(local_context)

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

        if remote:
            output = self.forward_remote(local_context, output, images, transform, targets)

        return output

    def forward_remote(self, local_context, output, images, transform, targets):
        """
            Forward pass non-sequential image inputs and targets through the remote context of the synapse.
            
            Args:
                local_context (:obj: `torch.FloatTensor` of shape :obj: `(batch_size, bittensor.__network_dim__)`, `required`)
                    Distillation model for remote_context.

                output (:obj: `Bittensor.SynapseOutput`, `required`)
                    The object containing the output thus far of the local context run

                images (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, channels, rows, cols)`, `required`): 
                    PIL.toTensor() encoded images.
                
                transform (:obj: `torch.FloatTensor` of shape :obj: `(batch_size, self.transform_dim)`, `required`):
                    transform images to common shape.
                
                targets (:obj:`torch.FloatTensor`  of shape :obj:`(batch_size, target_dim)`, `optional`, defaults to None): 
                    Image labels.
            
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

                    keys (:obj:`torch.LongTensor` of shape :obj:`(-1)`, `optional`): 
                        Keys for queried neurons.

                    scores (:obj:`torch.LongTensor` of shape :obj:`(batch_size, len(keys))`, `optional`): 
                        scores for each active key per example.
                )

        """
        # remote_context: responses from a bittensor remote network call.
        # remote_context.shape = [batch_size, bittensor.__network_dim__]
        remote_context, weights = self.dendrite.forward_image(images, transform)
        output.weights = weights

        # distillation_loss: distillation loss between local_context and remote_context
        # distillation_loss.shape = [1]
        distillation_loss = F.mse_loss(local_context, remote_context.detach().to(self.device))
        output.distillation_loss = distillation_loss
        output.loss = output.loss + distillation_loss

        # remote_hidden: hidden layer encoding using remote_context.
        # remote_hidden.shape = [batch_size, bittensor.__network_dim__]
        remote_hidden = torch.cat([transform, remote_context.to(self.device)], dim=1)
        remote_hidden = self.hidden_layer1(remote_hidden)
        remote_hidden = self.hidden_layer2(remote_hidden)
        output.remote_hidden = remote_hidden

        if targets is not None:
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
